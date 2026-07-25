# LLM API Gateway

![CI](https://github.com/anushakumar05/llm-gateway/actions/workflows/ci.yml/badge.svg)

A reverse proxy for LLM API traffic. It sits between applications and providers (OpenAI,
Anthropic, local models) and centrally enforces per-team rate limits and spending caps, fails
over automatically when a provider degrades, serves semantically similar repeat requests from
cache, and emits unified metrics for every call.

It is drop-in: point a client's `base_url` at the gateway and change nothing else. The wire
format is the OpenAI chat completions contract, verified against the official OpenAI Python SDK.

---

## Results

Measured with the load-test harness in [`scripts/loadtest.py`](scripts/loadtest.py) against
instrumented mock providers. Raw output is committed in [`results/`](results/).

| Scenario | Overhead p50 | Overhead p95 | Throughput | Notes |
|---|---|---|---|---|
| Provider path (cache bypassed) | 3.1 ms | 8.9 ms | 92 rps | 5,000 requests, 0 failures |
| Cache path, concurrency 10 | 38 ms | 98 ms | 115 rps | 84.6% hit rate |
| Cache path, concurrency 25 | 53 ms | 307 ms | 160 rps | 85.0% hit rate |
| Cache path, concurrency 50 | 13 ms | 609 ms | 177 rps | 85.2% hit rate |
| Failover under outage | — | — | 88 rps | 1,000 requests, 0 failures |

**Cache tier split:** 53% of hits served by exact match, 47% by semantic vector search.
**Failover split:** 426 requests served by the primary, 574 by the fallback, zero lost.

---

## Quick start

```bash
docker compose up -d --build

curl localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer team-normal' \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hello"}]}'
```

- Gateway — http://localhost:8000
- Grafana — http://localhost:3000 (anonymous viewer access enabled, dashboards auto-provisioned)
- Prometheus — http://localhost:9090

**No API keys are required.** The stack ships with mock providers that speak the OpenAI contract
and support fault injection, so the full system — including failover and circuit breaking — runs
and tests end to end offline and at zero cost.

Point an existing client at it:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="team-normal")
```

---

## Architecture

Every request follows the same path:

```
client
  │
  ├─ 1. auth          resolve team from API key → limits, budget
  ├─ 2. rate limit    token bucket in Redis (requests/min + tokens/min)   → 429
  ├─ 3. budget        daily spend cap for the team                        → 402
  ├─ 4. cache         exact-match tier, then semantic vector search
  ├─ 5. router        retry → failover chain → circuit breaker            → 503
  ├─ 6. provider      OpenAI / Anthropic / mock
  └─ 7. record        bill the team, store in cache, emit metrics
```

| Module | Responsibility |
|---|---|
| `gateway/limits.py` | Token-bucket rate limiting, atomic Lua executed inside Redis |
| `gateway/budget.py` | Per-team daily spend caps, atomic counters |
| `gateway/cache.py` | Two-tier cache: exact match + RediSearch vector index |
| `gateway/embeddings.py` | Local sentence-transformers inference, bounded concurrency |
| `gateway/router.py` | Retry, provider failover chain, breaker integration |
| `gateway/breaker.py` | Circuit breaker state machine (closed / open / half-open) |
| `gateway/providers/` | Provider abstraction — OpenAI-compatible, Anthropic |
| `gateway/metrics.py` | Prometheus metric definitions |
| `mock_provider/` | Fake provider with chaos injection |
| `scripts/` | Load-test harness, threshold sweep, embedding benchmark |

---

## Testing

```bash
docker compose up -d
python -m pytest tests/ -v
```

No API keys, no network, no cost. The mock provider supports fault injection via
`POST /_chaos`:

| Mode | Effect |
|---|---|
| `ok` | Normal responses |
| `rate_limit` | 429 with a `Retry-After` header |
| `server_error` | 500 |
| `timeout` | Hangs for a configurable duration |
| `down` | 503, health check fails |

---


## Repo layout

```
gateway/            the gateway service
  providers/        provider implementations behind one interface
mock_provider/      fake OpenAI with chaos injection
tests/              integration suite, no API keys required
scripts/            load test, threshold sweep, embedding benchmark
observability/      Prometheus config, provisioned Grafana dashboards
results/            committed load-test output
```

Built with Python 3.12, FastAPI, Redis Stack, Docker Compose, Prometheus, and Grafana.
