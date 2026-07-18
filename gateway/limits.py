from __future__ import annotations

import time
from dataclasses import dataclass

import redis.asyncio as redis

# Atomic token bucket in Lua. Runs start-to-finish inside Redis with no
# interleaving — this is what closes the check-then-act race.
#
# KEYS[1] = bucket key
# ARGV[1] = capacity        (max tokens the bucket holds)
# ARGV[2] = refill_per_sec  (tokens added per second)
# ARGV[3] = now             (current unix time, float)
# ARGV[4] = cost            (tokens this request wants)
#
# Returns: {allowed (1/0), remaining, retry_after_seconds}
TOKEN_BUCKET_LUA = """
local key      = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill   = tonumber(ARGV[2])
local now      = tonumber(ARGV[3])
local cost     = tonumber(ARGV[4])

local state = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(state[1])
local ts     = tonumber(state[2])

if tokens == nil then
  tokens = capacity
  ts = now
end

-- refill based on elapsed time
local elapsed = math.max(0, now - ts)
tokens = math.min(capacity, tokens + elapsed * refill)
ts = now

local allowed = 0
local retry_after = 0
if tokens >= cost then
  tokens = tokens - cost
  allowed = 1
else
  retry_after = (cost - tokens) / refill
end

redis.call('HSET', key, 'tokens', tokens, 'ts', ts)
redis.call('EXPIRE', key, math.ceil(capacity / refill) + 60)

return {allowed, math.floor(tokens), tostring(retry_after)}
"""


@dataclass
class LimitConfig:
    rpm: int          # requests per minute
    tpm: int          # tokens per minute


@dataclass
class LimitResult:
    allowed: bool
    remaining: int
    retry_after: float
    limit_type: str = ""


class RateLimiter:
    def __init__(self, client: redis.Redis):
        self.client = client
        self._sha: str | None = None

    async def _ensure_script(self) -> str:
        if self._sha is None:
            self._sha = await self.client.script_load(TOKEN_BUCKET_LUA)
        return self._sha

    async def _check_bucket(
        self, key: str, capacity: int, refill_per_sec: float, cost: int
    ) -> LimitResult:
        sha = await self._ensure_script()
        allowed, remaining, retry_after = await self.client.evalsha(
            sha, 1, key, capacity, refill_per_sec, time.time(), cost
        )
        return LimitResult(
            allowed=bool(allowed),
            remaining=int(remaining),
            retry_after=float(retry_after),
        )

    async def check_request(self, team: str, cfg: LimitConfig) -> LimitResult:
        """One request = one token from the request bucket."""
        res = await self._check_bucket(
            key=f"ratelimit:{team}:req",
            capacity=cfg.rpm,
            refill_per_sec=cfg.rpm / 60.0,
            cost=1,
        )
        res.limit_type = "requests"
        return res

    async def check_tokens(self, team: str, cfg: LimitConfig, token_cost: int) -> LimitResult:
        """token_cost tokens from the token bucket."""
        res = await self._check_bucket(
            key=f"ratelimit:{team}:tok",
            capacity=cfg.tpm,
            refill_per_sec=cfg.tpm / 60.0,
            cost=token_cost,
        )
        res.limit_type = "tokens"
        return res