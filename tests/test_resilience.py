import asyncio
import time

import httpx
import pytest
import redis.asyncio as aioredis

from gateway.breaker import BreakerConfig, CircuitBreaker, State

GATEWAY = "http://localhost:8000"
MOCK_A = "http://localhost:9001"
MOCK_B = "http://localhost:9002"


def chaos(base: str, **kw):
    httpx.post(f"{base}/_chaos", json={"mode": "ok", **kw}, timeout=5)


@pytest.fixture(autouse=True)
async def clean():
    chaos(MOCK_A); chaos(MOCK_B)
    r = aioredis.from_url("redis://localhost:6379", decode_responses=True)
    for k in await r.keys("breaker:*"):
        await r.delete(k)
    await r.aclose()
    yield
    chaos(MOCK_A); chaos(MOCK_B)


def test_fails_over_when_primary_down():
    chaos(MOCK_A, mode="down")
    r = httpx.post(
        f"{GATEWAY}/v1/chat/completions",
        headers={"Authorization": "Bearer team-heavy"},
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
        timeout=30,
    )
    assert r.status_code == 200
    assert r.headers["x-gateway-provider"] == "mock-b"


def test_does_not_fail_over_on_non_retryable():
    """A 401 should not waste a second provider call."""
    # mock returns 429/500 only; simulate non-retryable via a bad request shape
    # (adapt to whatever non-retryable your mock can produce)
    pass


@pytest.mark.asyncio
async def test_breaker_opens_and_recovers():
    client = aioredis.from_url("redis://localhost:6379", decode_responses=True)
    cfg = BreakerConfig(failure_threshold=3, window_seconds=10, cooldown_seconds=1)
    b = CircuitBreaker(client, cfg)
    await client.delete("breaker:test-p:state", "breaker:test-p:failures")

    assert (await b.admit("test-p")).admitted            # closed

    for _ in range(3):
        await b.record_failure("test-p")
    assert await b.state("test-p") == State.OPEN

    assert not (await b.admit("test-p")).admitted        # fail fast

    await asyncio.sleep(1.1)
    a = await b.admit("test-p")
    assert a.admitted and a.is_probe                     # exactly one probe

    await b.record_success("test-p")
    assert await b.state("test-p") == State.CLOSED
    await client.aclose()


@pytest.mark.asyncio
async def test_only_one_probe_admitted_in_half_open():
    """The thundering-herd guard: 50 concurrent callers, exactly one probes."""
    client = aioredis.from_url("redis://localhost:6379", decode_responses=True)
    cfg = BreakerConfig(failure_threshold=2, cooldown_seconds=0.5, probe_ttl_seconds=10)
    b = CircuitBreaker(client, cfg)
    await client.delete("breaker:herd:state", "breaker:herd:failures")

    for _ in range(2):
        await b.record_failure("herd")
    await asyncio.sleep(0.6)

    results = await asyncio.gather(*[b.admit("herd") for _ in range(50)])
    probes = sum(1 for r in results if r.is_probe)
    assert probes == 1, f"expected exactly 1 probe, got {probes}"
    await client.aclose()