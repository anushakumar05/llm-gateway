from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from gateway.providers.base import Provider, ProviderError
from gateway.types import ChatRequest, ChatResponse, Chunk, Usage

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class OpenAICompatibleProvider(Provider):
    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: str = "unused",
        timeout: float = 30.0,
    ):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    def _payload(self, req: ChatRequest, stream: bool) -> dict:
        body: dict = {
            "model": req.model,
            "messages": [{"role": m.role, "content": m.content} for m in req.messages],
            "stream": stream,
            "temperature": req.temperature,
        }
        if req.max_tokens is not None:
            body["max_tokens"] = req.max_tokens
        return body

    def _raise(self, resp: httpx.Response) -> None:
        try:
            body = resp.json()
            detail = body.get("detail", body)
            msg = detail.get("error", {}).get("message", resp.text)
        except Exception:
            msg = resp.text

        retry_after = resp.headers.get("retry-after")
        raise ProviderError(
            msg,
            status_code=resp.status_code,
            provider=self.name,
            retryable=resp.status_code in RETRYABLE_STATUS,
            retry_after=float(retry_after) if retry_after else None,
        )

    async def complete(self, req: ChatRequest) -> ChatResponse:
        try:
            resp = await self.client.post(
                "/v1/chat/completions", json=self._payload(req, stream=False)
            )
        except httpx.TimeoutException as e:
            raise ProviderError(
                f"timeout: {e}", status_code=504, provider=self.name, retryable=True
            ) from e
        except httpx.TransportError as e:
            raise ProviderError(
                f"transport: {e}", status_code=503, provider=self.name, retryable=True
            ) from e

        if resp.status_code >= 400:
            self._raise(resp)

        data = resp.json()
        choice = data["choices"][0]
        usage = data.get("usage", {})
        return ChatResponse(
            content=choice["message"]["content"],
            model=data.get("model", req.model),
            usage=Usage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
            ),
            finish_reason=choice.get("finish_reason", "stop"),
            provider=self.name,
            id=data.get("id", ""),
        )

    async def stream(self, req: ChatRequest) -> AsyncIterator[Chunk]:
        try:
            async with self.client.stream(
                "POST", "/v1/chat/completions", json=self._payload(req, stream=True)
            ) as resp:
                if resp.status_code >= 400:
                    await resp.aread()
                    self._raise(resp)

                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:].strip()
                    if payload == "[DONE]":
                        return

                    data = json.loads(payload)
                    choice = data["choices"][0]
                    yield Chunk(
                        content=choice.get("delta", {}).get("content") or "",
                        finish_reason=choice.get("finish_reason"),
                    )
        except httpx.TimeoutException as e:
            raise ProviderError(
                f"timeout: {e}", status_code=504, provider=self.name, retryable=True
            ) from e
        except httpx.TransportError as e:
            raise ProviderError(
                f"transport: {e}", status_code=503, provider=self.name, retryable=True
            ) from e

    async def health(self) -> bool:
        try:
            r = await self.client.get("/health", timeout=3.0)
            return r.status_code == 200
        except Exception:
            return False