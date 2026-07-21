import httpx
import pytest
import redis.asyncio as aioredis

from gateway.cache import CacheConfig, SemanticCache, partition_key
from gateway.types import ChatRequest, Message

import pytest
import redis.asyncio as aioredis


@pytest.fixture(autouse=True)
async def clear_cache():
    r = aioredis.from_url("redis://localhost:6379", decode_responses=True)
    for key in await r.keys("cache:entry:*"):
        await r.delete(key)
    await r.aclose()
    yield

GATEWAY = "http://localhost:8000"


def ask(content: str, temperature: float = 0.0):
    return httpx.post(
        f"{GATEWAY}/v1/chat/completions",
        headers={"Authorization": "Bearer team-heavy"},
        json={"model": "gpt-4o-mini", "temperature": temperature,
              "messages": [{"role": "user", "content": content}]},
        timeout=30,
    )


def test_identical_prompt_hits():
    ask("cache test unique phrase alpha")
    r = ask("cache test unique phrase alpha")
    assert r.headers["x-cache"] == "hit"


def test_paraphrase_hits():
    """Uses a pair measured at ~0.97 in the threshold sweep."""
    ask("How do I reverse a list in Python?")
    r = ask("What's the way to reverse a Python list?")
    assert r.headers["x-cache"] == "hit"


def test_distant_paraphrase_misses():
    """Documents a known limitation: loose paraphrases fall below threshold."""
    ask("What is Python?")
    r = ask("Can you explain Python to me?")
    assert r.headers["x-cache"] == "miss"   # measured 0.7527, below 0.88


def test_cached_content_matches_original():
    """A hit must return the SAME answer, not a similar one."""
    first = ask("determinism check for cache").json()
    second = ask("determinism check for cache").json()
    assert first["choices"][0]["message"]["content"] == second["choices"][0]["message"]["content"]


def test_temperature_bypasses_cache():
    ask("temp bypass test", temperature=0.0)
    r = ask("temp bypass test", temperature=0.9)
    assert r.headers["x-cache"] == "miss"


def test_different_system_prompt_does_not_share_cache():
    a = ChatRequest(model="gpt-4o-mini", messages=[
        Message("system", "You are helpful"), Message("user", "hi")])
    b = ChatRequest(model="gpt-4o-mini", messages=[
        Message("system", "You are a pirate"), Message("user", "hi")])
    assert partition_key(a) != partition_key(b)


def test_different_model_does_not_share_cache():
    a = ChatRequest(model="gpt-4o-mini", messages=[Message("user", "hi")])
    b = ChatRequest(model="gpt-4o", messages=[Message("user", "hi")])
    assert partition_key(a) != partition_key(b)