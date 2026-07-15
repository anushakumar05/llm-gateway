from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Message:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class ChatRequest:
    """The gateway's internal request shape. Providers translate FROM this."""
    model: str
    messages: list[Message]
    stream: bool = False
    temperature: float = 1.0
    max_tokens: int | None = None

    def system_prompt(self) -> str:
        return "\n".join(m.content for m in self.messages if m.role == "system")

    def last_user_message(self) -> str:
        return next(
            (m.content for m in reversed(self.messages) if m.role == "user"), ""
        )


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class ChatResponse:
    """The gateway's internal response shape. Providers translate INTO this."""
    content: str
    model: str
    usage: Usage
    finish_reason: str = "stop"
    provider: str = ""
    id: str = field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    created: int = field(default_factory=lambda: int(time.time()))


@dataclass
class Chunk:
    """One piece of a streamed response."""
    content: str = ""
    finish_reason: str | None = None