from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

import redis.asyncio as redis


class State(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


# Atomic breaker admission check.
# KEYS[1] = state key (hash: state, opened_at, probe_owner)
# KEYS[2] = failure window key (sorted set of failure timestamps)
# ARGV[1] = now
# ARGV[2] = cooldown_seconds
# ARGV[3] = probe_ttl (how long one probe may hold the slot)
# Returns: {admitted (1/0), state, is_probe (1/0)}
ADMIT_LUA = """
local skey     = KEYS[1]
local now      = tonumber(ARGV[1])
local cooldown = tonumber(ARGV[2])
local probe_ttl= tonumber(ARGV[3])

local st = redis.call('HGET', skey, 'state')
if st == false then
  return {1, 'closed', 0}
end

if st == 'closed' then
  return {1, 'closed', 0}
end

local opened_at = tonumber(redis.call('HGET', skey, 'opened_at')) or 0

if st == 'open' then
  if now - opened_at < cooldown then
    return {0, 'open', 0}
  end
  -- cooldown elapsed: this caller claims the single probe slot
  redis.call('HSET', skey, 'state', 'half_open', 'probe_at', now)
  return {1, 'half_open', 1}
end

if st == 'half_open' then
  local probe_at = tonumber(redis.call('HGET', skey, 'probe_at')) or 0
  if now - probe_at < probe_ttl then
    return {0, 'half_open', 0}      -- someone else is probing; wait
  end
  redis.call('HSET', skey, 'probe_at', now)
  return {1, 'half_open', 1}        -- stale probe, claim it
end

return {1, 'closed', 0}
"""


@dataclass
class BreakerConfig:
    failure_threshold: int = 5      # failures within the window to trip
    window_seconds: float = 10.0
    cooldown_seconds: float = 5.0
    probe_ttl_seconds: float = 10.0


@dataclass
class Admission:
    admitted: bool
    state: State
    is_probe: bool


class CircuitBreaker:
    def __init__(self, client: redis.Redis, cfg: BreakerConfig | None = None):
        self.client = client
        self.cfg = cfg or BreakerConfig()
        self._sha: str | None = None

    def _skey(self, provider: str) -> str:
        return f"breaker:{provider}:state"

    def _fkey(self, provider: str) -> str:
        return f"breaker:{provider}:failures"

    async def _sha_admit(self) -> str:
        if self._sha is None:
            self._sha = await self.client.script_load(ADMIT_LUA)
        return self._sha

    async def admit(self, provider: str) -> Admission:
        sha = await self._sha_admit()
        admitted, state, is_probe = await self.client.evalsha(
            sha, 1, self._skey(provider),
            time.time(), self.cfg.cooldown_seconds, self.cfg.probe_ttl_seconds,
        )
        return Admission(bool(admitted), State(state), bool(is_probe))

    async def record_success(self, provider: str) -> None:
        # Any success closes the circuit and clears the failure window.
        await self.client.hset(self._skey(provider), mapping={"state": State.CLOSED.value})
        await self.client.delete(self._fkey(provider))

    async def record_failure(self, provider: str) -> None:
        now = time.time()
        fkey = self._fkey(provider)
        pipe = self.client.pipeline()
        pipe.zadd(fkey, {f"{now}:{id(self)}": now})
        pipe.zremrangebyscore(fkey, 0, now - self.cfg.window_seconds)  # drop old
        pipe.zcard(fkey)
        pipe.expire(fkey, int(self.cfg.window_seconds) + 60)
        results = await pipe.execute()
        recent_failures = results[2]

        if recent_failures >= self.cfg.failure_threshold:
            await self.client.hset(
                self._skey(provider),
                mapping={"state": State.OPEN.value, "opened_at": now},
            )
            print(f"[breaker] {provider} OPENED ({recent_failures} failures in window)")

    async def state(self, provider: str) -> State:
        st = await self.client.hget(self._skey(provider), "state")
        return State(st) if st else State.CLOSED