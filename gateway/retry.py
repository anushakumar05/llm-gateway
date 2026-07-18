from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass


@dataclass
class RetryConfig:
    max_attempts: int = 3
    base_delay: float = 0.2
    max_delay: float = 5.0


def backoff_delay(attempt: int, cfg: RetryConfig, retry_after: float | None = None) -> float:
    """
    Exponential backoff with full jitter.
    attempt is 0-indexed: 0 -> ~0.2s, 1 -> ~0.4s, 2 -> ~0.8s (randomized).
    """
    if retry_after is not None:
        return min(retry_after, cfg.max_delay)
    ceiling = min(cfg.base_delay * (2 ** attempt), cfg.max_delay)
    return random.uniform(0, ceiling)     # full jitter


async def sleep_backoff(attempt: int, cfg: RetryConfig, retry_after: float | None = None) -> float:
    delay = backoff_delay(attempt, cfg, retry_after)
    await asyncio.sleep(delay)
    return delay