from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import redis.asyncio as redis


@dataclass
class BudgetStatus:
    spent: float
    limit: float
    allowed: bool
    warning: bool          # crossed 80%

    @property
    def fraction(self) -> float:
        return self.spent / self.limit if self.limit else 0.0


class BudgetTracker:
    def __init__(self, client: redis.Redis):
        self.client = client

    def _key(self, team: str) -> str:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return f"budget:{team}:{day}"

    async def check(self, team: str, daily_limit: float) -> BudgetStatus:
        """Read-only: is this team already over budget?"""
        raw = await self.client.get(self._key(team))
        spent = float(raw) if raw else 0.0
        return BudgetStatus(
            spent=spent,
            limit=daily_limit,
            allowed=spent < daily_limit,
            warning=spent >= 0.8 * daily_limit,
        )

    async def charge(self, team: str, cost: float) -> None:
        """Add to the running total. INCRBYFLOAT is atomic."""
        key = self._key(team)
        await self.client.incrbyfloat(key, cost)
        await self.client.expire(key, 60 * 60 * 48)   # keep 2 days