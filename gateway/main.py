from __future__ import annotations

import os
import redis.asyncio as aioredis
import json
import time
from collections.abc import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from gateway.config import DEFAULT_ROUTE, MODEL_ROUTES, PROVIDERS
from gateway.providers.base import ProviderError
from gateway.providers.openai_compatible import OpenAICompatibleProvider
from gateway.types import ChatRequest, ChatResponse, Message, Usage
from gateway.budget import BudgetTracker
from gateway.limits import RateLimiter
from gateway.pricing import cost_usd, count_tokens
from gateway.teams import resolve_team

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


def pick_provider(model: str):
    chain = MODEL_ROUTES.get(model, DEFAULT_ROUTE)
    return registry[chain[0]]        # Phase 3 replaces this with the full chain


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ---- wire format (what clients send/receive) ----

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


@app.post("/v1/chat/completions")
async def completions(w: WireRequest, request: Request):
    req = to_internal(w)
    team = resolve_team(request.headers.get("authorization", "").removeprefix("Bearer ").strip())
    provider = pick_provider(req.model)
    started = time.perf_counter()

    # --- GATE 1: request rate limit ---
    rl = await rate_limiter.check_request(team.key, team.limits)
    if not rl.allowed:
        return JSONResponse(
            status_code=429,
            content={"error": {"message": "request rate limit exceeded", "type": "rate_limit_exceeded"}},
            headers={"Retry-After": str(max(1, round(rl.retry_after)))},
        )

    # --- GATE 2: token rate limit (estimate input now; it's the known part) ---
    prompt_tokens = sum(count_tokens(m.content) for m in req.messages)
    tl = await rate_limiter.check_tokens(team.key, team.limits, prompt_tokens)
    if not tl.allowed:
        return JSONResponse(
            status_code=429,
            content={"error": {"message": "token rate limit exceeded", "type": "rate_limit_exceeded"}},
            headers={"Retry-After": str(max(1, round(tl.retry_after)))},
        )

    # --- GATE 3: budget (optimistic check) ---
    budget = await budget_tracker.check(team.key, team.daily_budget_usd)
    if not budget.allowed:
        return JSONResponse(
            status_code=402,   # Payment Required — semantically perfect here
            content={"error": {
                "message": f"daily budget of ${team.daily_budget_usd:.2f} exhausted",
                "type": "budget_exceeded",
            }},
        )

    if not req.stream:
        try:
            resp = await provider.complete(req)
        except ProviderError as e:
            return JSONResponse(
                status_code=e.status_code,
                content={"error": {"message": e.message, "provider": e.provider}},
            )

        # --- reconcile: charge the REAL cost, now that we know it ---
        spend = cost_usd(resp.model, resp.usage.prompt_tokens, resp.usage.completion_tokens)
        await budget_tracker.charge(team.key, spend)

        elapsed_ms = (time.perf_counter() - started) * 1000
        print(
            f"[gateway] team={team.key} provider={resp.provider} model={resp.model} "
            f"tokens={resp.usage.total_tokens} cost=${spend:.6f} "
            f"budget={budget.fraction:.0%} latency={elapsed_ms:.1f}ms"
        )
        headers = {"x-gateway-provider": resp.provider, "x-ratelimit-remaining-requests": str(rl.remaining)}
        if budget.warning:
            headers["x-budget-warning"] = f"{budget.fraction:.0%} of daily budget used"
        return JSONResponse(content=to_wire(resp), headers=headers)

    # streaming path unchanged for now — charge in its finally block (see note)
    return StreamingResponse(
        relay(provider, req, request, started, team),
        media_type="text/event-stream",
        headers={"x-gateway-provider": provider.name},
    )


async def relay(provider, req: ChatRequest, request: Request, started: float) -> AsyncIterator[str]:
    """Forward chunks to the client while accumulating the full response."""
    buffer: list[str] = []
    finish_reason = None
    disconnected = False
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
        yield "data: " + json.dumps(
            {"error": {"message": e.message, "provider": e.provider}}
        ) + "\n\n"
        yield "data: [DONE]\n\n"

    finally:
        # Runs whether we completed, errored, or the client vanished.
        full = "".join(buffer)
        usage = Usage(
            prompt_tokens=sum(estimate_tokens(m.content) for m in req.messages),
            completion_tokens=estimate_tokens(full),
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        print(
            f"[gateway] provider={provider.name} model={req.model} "
            f"tokens={usage.total_tokens} latency={elapsed_ms:.1f}ms stream=true "
            f"complete={not disconnected and finish_reason is not None}"
        )
        # Phase 2 bills here. Phase 4 caches here — but only if complete.