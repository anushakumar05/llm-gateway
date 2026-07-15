from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from gateway.providers.base import Provider, ProviderError
from gateway.types import ChatRequest, ChatResponse, Chunk, Usage

RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504, 529}


class AnthropicProvider(Provider):
    def __init__(self, api_key: str, base_url: str = "https://api.anthropic.com", timeout: float = 30.0):
        self.name = "anthropic"
        self.client = httpx.AsyncClient(
            base_url=base_url,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            timeout=timeout,
        )

    def _payload(self, req: ChatRequest, stream: bool) -> dict:
        # Anthropic takes the system prompt as a TOP-LEVEL field,
        # not as a message with role="system".
        system = req.system_prompt()
        turns = [
            {"role": m.role, "content": m.content}
            for m in req.messages
            if m.role != "system"
        ]
        body: dict = {
            "model": req.model,
            "messages": turns,
            "max_tokens": req.max_tokens or 1024,  # required by Anthropic, optional for OpenAI
            "temperature": req.temperature,
            "stream": stream,
        }
        if system:
            body["system"] = system
        return body

    async def complete(self, req: ChatRequest) -> ChatResponse:
        resp = await self.client.post("/v1/messages", json=self._payload(req, stream=False))
        if resp.status_code >= 400:
            raise ProviderError(
                resp.text,
                status_code=resp.status_code,
                provider=self.name,
                retryable=resp.status_code in RETRYABLE_STATUS,
            )

        data = resp.json()
        text = "".join(b.get("text", "") for b in data.get("content", []))
        u = data.get("usage", {})
        return ChatResponse(
            content=text,
            model=data.get("model", req.model),
            usage=Usage(
                prompt_tokens=u.get("input_tokens", 0),
                completion_tokens=u.get("output_tokens", 0),
            ),
            finish_reason=data.get("stop_reason", "stop"),
            provider=self.name,
        )

    async def stream(self, req: ChatRequest) -> AsyncIterator[Chunk]:
        async with self.client.stream(
            "POST", "/v1/messages", json=self._payload(req, stream=True)
        ) as resp:
            if resp.status_code >= 400:
                await resp.aread()
                raise ProviderError(
                    resp.text,
                    status_code=resp.status_code,
                    provider=self.name,
                    retryable=resp.status_code in RETRYABLE_STATUS,
                )
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = json.loads(line[6:])
                if data.get("type") == "content_block_delta":
                    yield Chunk(content=data["delta"].get("text", ""))
                elif data.get("type") == "message_stop":
                    yield Chunk(finish_reason="stop")
                    return

    async def health(self) -> bool:
        return True