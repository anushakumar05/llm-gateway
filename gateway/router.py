from __future__ import annotations

from dataclasses import dataclass, field

from gateway.breaker import CircuitBreaker
from gateway.providers.base import Provider, ProviderError
from gateway.retry import RetryConfig, sleep_backoff
from gateway.types import ChatRequest, ChatResponse


@dataclass
class RouteAttempt:
    provider: str
    outcome: str            # "success" | "error" | "skipped_open"
    detail: str = ""


@dataclass
class RouteResult:
    response: ChatResponse | None
    attempts: list[RouteAttempt] = field(default_factory=list)
    error: ProviderError | None = None


class Router:
    def __init__(
        self,
        registry: dict[str, Provider],
        breaker: CircuitBreaker,
        retry_cfg: RetryConfig | None = None,
    ):
        self.registry = registry
        self.breaker = breaker
        self.retry = retry_cfg or RetryConfig()

    async def complete(self, req: ChatRequest, chain: list[str]) -> RouteResult:
        result = RouteResult(response=None)
        last_error: ProviderError | None = None

        for name in chain:
            provider = self.registry[name]

            admission = await self.breaker.admit(name)
            if not admission.admitted:
                result.attempts.append(
                    RouteAttempt(name, "skipped_open", f"breaker {admission.state.value}")
                )
                continue

            # retry loop within this provider
            for attempt in range(self.retry.max_attempts):
                try:
                    resp = await provider.complete(req)
                    await self.breaker.record_success(name)
                    result.attempts.append(RouteAttempt(name, "success"))
                    result.response = resp
                    return result

                except ProviderError as e:
                    last_error = e
                    await self.breaker.record_failure(name)

                    if not e.retryable:
                        result.attempts.append(RouteAttempt(name, "error", f"non-retryable {e.status_code}"))
                        break               # don't retry, don't fail over on a 401

                    if attempt < self.retry.max_attempts - 1:
                        await sleep_backoff(attempt, self.retry, e.retry_after)
                    else:
                        result.attempts.append(RouteAttempt(name, "error", f"exhausted retries {e.status_code}"))

            # if this was a probe that failed, the breaker reopens on next admit
            if last_error and not last_error.retryable:
                break                       # a 401 will fail identically everywhere

        result.error = last_error
        return result