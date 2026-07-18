import asyncio

import pytest
import redis.asyncio as aioredis

from gateway.limits import LimitConfig, RateLimiter


@pytest.mark.asyncio
async def test_concurrent_requests_respect_exact_limit():
    """
    Fire 200 concurrent requests at a bucket with capacity 100.
    Atomic Lua => EXACTLY 100 allowed, 100 rejected. Not 100+.
    """
    client = aioredis.from_url("redis://localhost:6379", decode_responses=True)
    await client.delete("ratelimit:race-test:req")

    limiter = RateLimiter(client)
    cfg = LimitConfig(rpm=100, tpm=10_000_000)   # tpm irrelevant here

    async def one():
        r = await limiter.check_request("race-test", cfg)
        return r.allowed

    results = await asyncio.gather(*[one() for _ in range(200)])
    allowed = sum(results)

    assert allowed == 100, f"expected exactly 100 allowed, got {allowed}"
    await client.aclose()

@pytest.mark.asyncio
async def test_budget_blocks_when_exhausted():
    from gateway.budget import BudgetTracker
    client = aioredis.from_url("redis://localhost:6379", decode_responses=True)
    tracker = BudgetTracker(client)
    team = "budget-test"
    await client.delete(tracker._key(team))

    assert (await tracker.check(team, 1.0)).allowed
    await tracker.charge(team, 0.85)
    assert (await tracker.check(team, 1.0)).warning       # past 80%
    await tracker.charge(team, 0.30)
    assert not (await tracker.check(team, 1.0)).allowed    # past 100%
    await client.aclose()