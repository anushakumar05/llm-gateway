import pytest
from openai import OpenAI

client = OpenAI(base_url="http://localhost:9001/v1", api_key="not-a-real-key")


def test_non_streaming():
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hello"}],
    )
    assert resp.choices[0].message.content.startswith("Mock reply")
    assert resp.usage.total_tokens > 0


def test_streaming():
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
    )
    chunks = [c.choices[0].delta.content for c in stream]
    text = "".join(c for c in chunks if c)
    assert "Mock reply" in text


def test_deterministic():
    def ask():
        return client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "same question"}],
        ).choices[0].message.content

    assert ask() == ask()