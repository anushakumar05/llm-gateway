from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from gateway.types import ChatRequest, ChatResponse, Chunk


class ProviderError(Exception):
    """Normalized error from any provider."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        provider: str,
        retryable: bool,
        retry_after: float | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.provider = provider
        self.retryable = retryable
        self.retry_after = retry_after


class Provider(ABC):
    name: str

    @abstractmethod
    async def complete(self, req: ChatRequest) -> ChatResponse:
        ...

    @abstractmethod
    def stream(self, req: ChatRequest) -> AsyncIterator[Chunk]:
        ...

    @abstractmethod
    async def health(self) -> bool:
        ...