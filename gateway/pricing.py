from __future__ import annotations

import tiktoken

# USD per 1M tokens. Realistic-ish public numbers; exact values don't matter
# for the project as long as they're consistent.
PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o":      {"input": 2.50, "output": 10.00},
}
_DEFAULT = {"input": 0.50, "output": 1.50}

_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    p = PRICING.get(model, _DEFAULT)
    return (prompt_tokens * p["input"] + completion_tokens * p["output"]) / 1_000_000