import httpx
import pytest
from openai import OpenAI

GATEWAY = "http://localhost:8000"
MOCK_A = "http://localhost:9001"

client = OpenAI(base_url=f"{GATEWAY}/v1", api_key="test-key")


def set_chaos(base: str, **kwargs):
    httpx.post(f"{base}/_chaos", json={"mode": "ok", **kwargs}, timeout=5)


@pytest.fixture(autouse=True)
def healthy():
    set_chaos(MOCK_A)
    yield
    set_chaos(MOCK_A)


def test_proxies_non_streaming():
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hello"}],
    )
    assert "Mock reply" in resp.choices[0].message.content
    assert resp.usage.total_tokens > 0


def test_reports_which_provider_served():
    r = httpx.post(
        f"{GATEWAY}/v1/chat/completions",
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
        timeout=10,
    )
    assert r.headers["x-gateway-provider"] == "mock-a"


def test_proxies_streaming():
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
    )
    text = "".join(c.choices[0].delta.content or "" for c in stream)
    assert "Mock reply" in text


def test_streaming_matches_non_streaming():
    """The gateway must not corrupt or reorder chunks."""
    msgs = [{"role": "user", "content": "determinism check"}]
    solid = client.chat.completions.create(model="gpt-4o-mini", messages=msgs)
    stream = client.chat.completions.create(model="gpt-4o-mini", messages=msgs, stream=True)
    streamed = "".join(c.choices[0].delta.content or "" for c in stream)
    assert streamed.strip() == solid.choices[0].message.content.strip()


def test_propagates_provider_error():
    set_chaos(MOCK_A, mode="rate_limit")
    r = httpx.post(
        f"{GATEWAY}/v1/chat/completions",
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
        timeout=10,
    )
    assert r.status_code == 429
    assert r.json()["error"]["provider"] == "mock-a"