from __future__ import annotations

import json
import os
import time
from collections.abc import AsyncIterator

import redis.asyncio as aioredis
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from gateway.breaker import CircuitBreaker
from gateway.budget import BudgetTracker
from gateway.config import DEFAULT_ROUTE, MODEL_ROUTES, PROVIDERS
from gateway.limits import RateLimiter
from gateway.pricing import cost_usd, count_tokens
from gateway.providers.base import ProviderError
from gateway.providers.openai_compatible import OpenAICompatibleProvider
from gateway.retry import RetryConfig
from gateway.router import Router
from gateway.teams import Team, resolve_team
from gateway.types import ChatRequest, ChatResponse, Message, Usage

app = FastAPI(title="LLM Gateway")

redis_client = aioredis.from_url(
    os.getenv("REDIS_URL", "redis://localhost:6379"), decode_responses=True
)
rate_limiter = RateLimiter(redis_client)
budget_tracker = BudgetTracker(redis_client)

registry = {
    name: OpenAICompatibleProvider(name=name, base_url=cfg["base_url"])
    for name, cfg in PROVIDERS.items()
    if cfg["type"] == "openai_compatible"
}

breaker = CircuitBreaker(redis_client)
router = Router(registry, breaker, RetryConfig(max_attempts=3))


def route_chain(model: str) -> list[str]:
    return MODEL_ROUTES.get(model, DEFAULT_ROUTE)


class WireMessage(BaseModel):
    role: str
    content: str


class WireRequest(BaseModel):
    model: str
    messages: list[WireMessage]
    stream: bool = False
    temperature: float = 1.0
    max_tokens: int | None = None


def to_internal(w: WireRequest) -> ChatRequest:
    return ChatRequest(
        model=w.model,
        messages=[Message(role=m.role, content=m.content) for m in w.messages],
        stream=w.stream,
        temperature=w.temperature,
        max_tokens=w.max_tokens,
    )


def to_wire(r: ChatResponse) -> dict:
    return {
        "id": r.id,
        "object": "chat.completion",
        "created": r.created,
        "model": r.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": r.content},
            "finish_reason": r.finish_reason,
        }],
        "usage": {
            "prompt_tokens": r.usage.prompt_tokens,
            "completion_tokens": r.usage.completion_tokens,
            "total_tokens": r.usage.total_tokens,
        },
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/providers")
async def provider_status():
    out = {}
    for name in registry:
        out[name] = {
            "breaker": (await breaker.state(name)).value,
            "healthy": await registry[name].health(),
        }
    return out


@app.post("/v1/chat/completions")
async def completions(w: WireRequest, request: Request):
    req = to_internal(w)
    team = resolve_team(
        request.headers.get("authorization", "").removeprefix("Bearer ").strip()
    )
    chain = route_chain(req.model)
    started = time.perf_counter()

    # --- GATE 1: request rate limit ---
    rl = await rate_limiter.check_request(team.key, team.limits)
    if not rl.allowed:
        return JSONResponse(
            status_code=429,
            content={"error": {"message": "request rate limit exceeded",
                               "type": "rate_limit_exceeded"}},
            headers={"Retry-After": str(max(1, round(rl.retry_after)))},
        )

    # --- GATE 2: token rate limit ---
    prompt_tokens = sum(count_tokens(m.content) for m in req.messages)
    tl = await rate_limiter.check_tokens(team.key, team.limits, prompt_tokens)
    if not tl.allowed:
        return JSONResponse(
            status_code=429,
            content={"error": {"message": "token rate limit exceeded",
                               "type": "rate_limit_exceeded"}},
            headers={"Retry-After": str(max(1, round(tl.retry_after)))},
        )

    # --- GATE 3: budget ---
    budget = await budget_tracker.check(team.key, team.daily_budget_usd)
    if not budget.allowed:
        return JSONResponse(
            status_code=402,
            content={"error": {
                "message": f"daily budget of ${team.daily_budget_usd:.2f} exhausted",
                "type": "budget_exceeded",
            }},
        )

    # --- NON-STREAMING: routed through retry + failover + breaker ---
    if not req.stream:
        result = await router.complete(req, chain)

        if result.response is None:
            e = result.error
            return JSONResponse(
                status_code=e.status_code if e else 503,
                content={"error": {
                    "message": e.message if e else "all providers unavailable",
                    "type": "upstream_unavailable",
                    "attempts": [
                        {"provider": a.provider, "outcome": a.outcome, "detail": a.detail}
                        for a in result.attempts
                    ],
                }},
            )

        resp = result.response
        spend = cost_usd(resp.model, resp.usage.prompt_tokens, resp.usage.completion_tokens)
        await budget_tracker.charge(team.key, spend)

        elapsed_ms = (time.perf_counter() - started) * 1000
        path = " -> ".join(f"{a.provider}:{a.outcome}" for a in result.attempts)
        print(
            f"[gateway] team={team.key} route=[{path}] served_by={resp.provider} "
            f"tokens={resp.usage.total_tokens} cost=${spend:.6f} latency={elapsed_ms:.1f}ms"
        )
        headers = {
            "x-gateway-provider": resp.provider,
            "x-gateway-attempts": str(len(result.attempts)),
            "x-ratelimit-remaining-requests": str(rl.remaining),
        }
        if budget.warning:
            headers["x-budget-warning"] = f"{budget.fraction:.0%} of daily budget used"
        return JSONResponse(content=to_wire(resp), headers=headers)

    # --- STREAMING: first healthy provider in the chain, no mid-stream failover ---
    provider = None
    for name in chain:
        if (await breaker.admit(name)).admitted:
            provider = registry[name]
            break
    if provider is None:
        return JSONResponse(
            status_code=503,
            content={"error": {"message": "all providers unavailable",
                               "type": "upstream_unavailable"}},
        )

    return StreamingResponse(
        relay(provider, req, request, started, team),
        media_type="text/event-stream",
        headers={"x-gateway-provider": provider.name},
    )


async def relay(
    provider, req: ChatRequest, request: Request, started: float, team: Team
) -> AsyncIterator[str]:
    """Forward chunks to the client while accumulating the full response."""
    buffer: list[str] = []
    finish_reason = None
    disconnected = False
    failed = False
    resp = ChatResponse(content="", model=req.model, usage=Usage(), provider=provider.name)

    def frame(delta: dict, finish: str | None = None) -> str:
        return "data: " + json.dumps({
            "id": resp.id,
            "object": "chat.completion.chunk",
            "created": resp.created,
            "model": req.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }) + "\n\n"

    try:
        yield frame({"role": "assistant"})
        async for chunk in provider.stream(req):
            if await request.is_disconnected():
                disconnected = True
                break
            if chunk.content:
                buffer.append(chunk.content)
                yield frame({"content": chunk.content})
            if chunk.finish_reason:
                finish_reason = chunk.finish_reason

        if not disconnected:
            yield frame({}, finish=finish_reason or "stop")
            yield "data: [DONE]\n\n"

    except ProviderError as e:
        failed = True
        yield "data: " + json.dumps(
            {"error": {"message": e.message, "provider": e.provider}}
        ) + "\n\n"
        yield "data: [DONE]\n\n"

    finally:
        if failed:
            await breaker.record_failure(provider.name)
        elif not disconnected:
            await breaker.record_success(provider.name)

        full = "".join(buffer)
        usage = Usage(
            prompt_tokens=sum(count_tokens(m.content) for m in req.messages),
            completion_tokens=count_tokens(full),
        )
        spend = cost_usd(req.model, usage.prompt_tokens, usage.completion_tokens)
        await budget_tracker.charge(team.key, spend)

        elapsed_ms = (time.perf_counter() - started) * 1000
        print(
            f"[gateway] team={team.key} provider={provider.name} model={req.model} "
            f"tokens={usage.total_tokens} cost=${spend:.6f} latency={elapsed_ms:.1f}ms "
            f"stream=true complete={not disconnected and not failed}"
        )