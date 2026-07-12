import asyncio
import json
import random
import time
import uuid
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="Mock LLM Provider")


# ---------- chaos control ----------

class Chaos(BaseModel):
    mode: Literal["ok", "rate_limit", "server_error", "timeout", "down"] = "ok"
    failure_rate: float = 1.0    # probability the fault fires, 0.0-1.0
    latency_ms: int = 200        # base latency on healthy responses
    jitter_ms: int = 100
    timeout_s: float = 30.0      # how long "timeout" mode hangs


chaos = Chaos()


@app.get("/_chaos")
def get_chaos():
    return chaos


@app.post("/_chaos")
def set_chaos(new: Chaos):
    global chaos
    chaos = new
    return chaos


@app.get("/health")
def health():
    if chaos.mode == "down":
        raise HTTPException(status_code=503, detail="provider down")
    return {"status": "ok"}


async def apply_chaos():
    fault = None
    if chaos.mode != "ok" and random.random() < chaos.failure_rate:
        fault = chaos.mode

    if fault == "down":
        raise HTTPException(
            status_code=503,
            detail={"error": {"message": "service unavailable", "type": "server_error"}},
        )
    if fault == "rate_limit":
        raise HTTPException(
            status_code=429,
            detail={"error": {"message": "Rate limit reached", "type": "rate_limit_exceeded"}},
            headers={"Retry-After": "2"},
        )
    if fault == "server_error":
        raise HTTPException(
            status_code=500,
            detail={"error": {"message": "internal server error", "type": "server_error"}},
        )
    if fault == "timeout":
        await asyncio.sleep(chaos.timeout_s)

    delay = (chaos.latency_ms + random.uniform(0, chaos.jitter_ms)) / 1000
    await asyncio.sleep(delay)


# ---------- the OpenAI-shaped API ----------

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    stream: bool = False
    temperature: float = 1.0
    max_tokens: int | None = None


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def make_reply(req: ChatRequest) -> str:
    last_user = next(
        (m.content for m in reversed(req.messages) if m.role == "user"), ""
    )
    return f"Mock reply to: {last_user[:120]}"


async def stream_chunks(req: ChatRequest, reply: str):
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    def frame(delta: dict, finish: str | None = None) -> str:
        payload = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": req.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(payload)}\n\n"

    yield frame({"role": "assistant"})
    for word in reply.split(" "):
        await asyncio.sleep(0.02)
        yield frame({"content": word + " "})
    yield frame({}, finish="stop")
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    await apply_chaos()
    reply = make_reply(req)

    if req.stream:
        return StreamingResponse(
            stream_chunks(req, reply), media_type="text/event-stream"
        )

    prompt_tokens = sum(estimate_tokens(m.content) for m in req.messages)
    completion_tokens = estimate_tokens(reply)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": reply},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }