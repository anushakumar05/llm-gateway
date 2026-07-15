from __future__ import annotations

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

app = FastAPI(title="LLM Gateway")

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
    provider = pick_provider(req.model)
    started = time.perf_counter()

    if not req.stream:
        try:
            resp = await provider.complete(req)
        except ProviderError as e:
            return JSONResponse(
                status_code=e.status_code,
                content={"error": {"message": e.message, "provider": e.provider}},
            )

        elapsed_ms = (time.perf_counter() - started) * 1000
        print(
            f"[gateway] provider={resp.provider} model={resp.model} "
            f"tokens={resp.usage.total_tokens} latency={elapsed_ms:.1f}ms stream=false"
        )
        return JSONResponse(
            content=to_wire(resp),
            headers={"x-gateway-provider": resp.provider},
        )

    return StreamingResponse(
        relay(provider, req, request, started),
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