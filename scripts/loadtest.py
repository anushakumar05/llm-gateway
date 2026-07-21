"""
Load test the gateway and emit the numbers that go in the README.

Usage:
  PYTHONPATH=. python scripts/loadtest.py baseline    --n 5000 --concurrency 50
  PYTHONPATH=. python scripts/loadtest.py cache       --n 2000 --concurrency 50 --repeat-rate 0.6
  PYTHONPATH=. python scripts/loadtest.py failover    --n 1000 --concurrency 25
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from dataclasses import dataclass, field

import httpx

GATEWAY = "http://localhost:8000"
MOCK_A = "http://localhost:9001"
MOCK_B = "http://localhost:9002"
TEAM = "team-loadtest"       # effectively unlimited, so we measure the gateway, not the limiter

TOPICS = [
    "explain database indexing", "what is a load balancer", "how does TLS work",
    "describe the CAP theorem", "what is a bloom filter", "explain MVCC",
    "how do B-trees work", "what is consistent hashing", "explain write-ahead logging",
    "what is a vector clock", "how does Raft elect a leader", "explain backpressure",
]

PARAPHRASE_TEMPLATES = [
    "{}",
    "can you explain {}",
    "{} — how does that work",
    "I want to understand {}",
    "{}?",
]


def paraphrase(topic: str) -> str:
    import random
    return random.choice(PARAPHRASE_TEMPLATES).format(topic)


@dataclass
class Sample:
    status: int
    total_ms: float
    overhead_ms: float | None
    cache: str
    provider: str


@dataclass
class Results:
    samples: list[Sample] = field(default_factory=list)
    wall_seconds: float = 0.0

    def pct(self, values: list[float], p: float) -> float:
        if not values:
            return 0.0
        s = sorted(values)
        k = min(int(len(s) * p), len(s) - 1)
        return s[k]

    def report(self, label: str) -> dict:
        ok = [s for s in self.samples if s.status == 200]
        totals = [s.total_ms for s in ok]
        overheads = [s.overhead_ms for s in ok if s.overhead_ms is not None]
        hits = [s for s in ok if s.cache == "hit"]
        misses = [s for s in ok if s.cache == "miss"]
        oh_hit = [s.overhead_ms for s in hits if s.overhead_ms is not None]
        oh_miss = [s.overhead_ms for s in misses if s.overhead_ms is not None]

        r = {
            "label": label,
            "requests": len(self.samples),
            "succeeded": len(ok),
            "failed": len(self.samples) - len(ok),
            "wall_seconds": round(self.wall_seconds, 2),
            "throughput_rps": round(len(self.samples) / self.wall_seconds, 1) if self.wall_seconds else 0,
            "total_latency_ms": {
                "p50": round(self.pct(totals, .50), 2),
                "p95": round(self.pct(totals, .95), 2),
                "p99": round(self.pct(totals, .99), 2),
                "mean": round(statistics.mean(totals), 2) if totals else 0,
            },
            "gateway_overhead_ms": {
                "all_p50": round(self.pct(overheads, .50), 2),
                "all_p95": round(self.pct(overheads, .95), 2),
                "all_p99": round(self.pct(overheads, .99), 2),
                "hit_p95": round(self.pct(oh_hit, .95), 2),
                "miss_p95": round(self.pct(oh_miss, .95), 2),
                "n_samples": len(overheads),
            },
            "cache": {
                "hits": len(hits),
                "misses": len(misses),
                "hit_rate": round(len(hits) / len(ok), 4) if ok else 0,
                "hit_p95_ms": round(self.pct([s.total_ms for s in hits], .95), 2),
                "miss_p95_ms": round(self.pct([s.total_ms for s in misses], .95), 2),
            },
            "providers": {},
        }
        for s in ok:
            key = s.provider or "unknown"
            r["providers"][key] = r["providers"].get(key, 0) + 1
        return r


async def one_request(client: httpx.AsyncClient, prompt: str,
                       temperature: float = 0.0) -> Sample:
    t0 = time.perf_counter()
    try:
        resp = await client.post(
            f"{GATEWAY}/v1/chat/completions",
            headers={"Authorization": f"Bearer {TEAM}"},
            json={"model": "gpt-4o-mini", "temperature": temperature,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=60,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        oh = resp.headers.get("x-gateway-overhead-ms")
        return Sample(
            status=resp.status_code,
            total_ms=elapsed,
            overhead_ms=float(oh) if oh else None,
            cache=resp.headers.get("x-cache", "n/a"),
            provider=resp.headers.get("x-gateway-provider", ""),
        )
    except Exception:
        return Sample(status=0, total_ms=(time.perf_counter() - t0) * 1000,
                      overhead_ms=None, cache="error", provider="")


async def run(n: int, concurrency: int, prompt_for, temperature: float = 0.0) -> Results:
    res = Results()
    sem = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency + 20,
                          max_keepalive_connections=concurrency)

    async with httpx.AsyncClient(limits=limits) as client:
        async def worker(i: int):
            async with sem:
                res.samples.append(await one_request(client, prompt_for(i), temperature))

        t0 = time.perf_counter()
        await asyncio.gather(*[worker(i) for i in range(n)])
        res.wall_seconds = time.perf_counter() - t0
    return res


def set_chaos(base: str, **kw):
    httpx.post(f"{base}/_chaos", json={"mode": "ok", **kw}, timeout=5)


def flush_cache():
    import redis
    r = redis.from_url("redis://localhost:6379", decode_responses=True)
    keys = r.keys("cache:entry:*")
    if keys:
        r.delete(*keys)
    r.close()


async def scenario_baseline(args):
    """Cache bypassed (temperature>0). Measures gateway overhead on the provider path."""
    flush_cache()
    set_chaos(MOCK_A); set_chaos(MOCK_B)
    return await run(args.n, args.concurrency,
                     lambda i: f"unique prompt {i} {random.random()}",
                     temperature=0.7)


async def scenario_cache(args):
    """Realistic repetition. Measures hit rate and the latency delta."""
    flush_cache()
    set_chaos(MOCK_A); set_chaos(MOCK_B)

    def prompt_for(i: int) -> str:
        if random.random() < args.repeat_rate:
            return paraphrase(random.choice(TOPICS))          # repeated -> cacheable
        return f"novel question {i} {random.random()}"

    return await run(args.n, args.concurrency, prompt_for)


async def scenario_failover(args):
    """Cache bypassed so every request must reach a provider."""
    flush_cache()
    set_chaos(MOCK_A); set_chaos(MOCK_B)

    async def kill_primary_midway():
        await asyncio.sleep(3)
        print("  >> killing mock-a")
        set_chaos(MOCK_A, mode="down")
        await asyncio.sleep(6)
        print("  >> restoring mock-a")
        set_chaos(MOCK_A)

    killer = asyncio.create_task(kill_primary_midway())
    res = await run(args.n, args.concurrency,
                    lambda i: f"failover probe {i} {random.random()}",
                    temperature=0.7)
    await killer
    set_chaos(MOCK_A)
    return res


SCENARIOS = {
    "baseline": scenario_baseline,
    "cache": scenario_cache,
    "failover": scenario_failover,
}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", choices=SCENARIOS)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--concurrency", type=int, default=50)
    ap.add_argument("--repeat-rate", type=float, default=0.6)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print(f"running {args.scenario}: n={args.n} concurrency={args.concurrency}")
    res = await SCENARIOS[args.scenario](args)
    report = res.report(args.scenario)

    print(json.dumps(report, indent=2))
    out = args.out or f"results/{args.scenario}.json"
    import os
    os.makedirs("results", exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    asyncio.run(main())