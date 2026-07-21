from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# --- request-level ---

REQUESTS = Counter(
    "gateway_requests_total",
    "Requests handled by the gateway",
    ["team", "model", "outcome"],          # outcome: success|rate_limited|budget_denied|upstream_error
)

REQUEST_LATENCY = Histogram(
    "gateway_request_duration_seconds",
    "End-to-end latency as seen by the client",
    ["model", "cache"],                    # cache: hit|miss
    buckets=(.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10),
)

# The money metric: time spent inside the gateway, excluding the provider call.
OVERHEAD_LATENCY = Histogram(
    "gateway_overhead_duration_seconds",
    "Gateway processing time, excluding upstream provider time",
    ["cache"],
    buckets=(.0005, .001, .0025, .005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5),
)

PROVIDER_LATENCY = Histogram(
    "gateway_provider_duration_seconds",
    "Time spent waiting on the upstream provider",
    ["provider"],
    buckets=(.01, .05, .1, .25, .5, 1, 2.5, 5, 10, 30),
)

# --- gates ---

RATE_LIMITED = Counter(
    "gateway_rate_limited_total", "Requests rejected by rate limits", ["team", "limit_type"]
)

BUDGET_DENIED = Counter(
    "gateway_budget_denied_total", "Requests rejected by budget cap", ["team"]
)

# --- cost ---

COST_USD = Counter(
    "gateway_cost_usd_total", "Cumulative estimated spend", ["team", "model"]
)

TOKENS = Counter(
    "gateway_tokens_total", "Tokens processed", ["model", "direction"]   # direction: input|output
)

COST_SAVED_USD = Counter(
    "gateway_cost_saved_usd_total", "Spend avoided by cache hits", ["model"]
)

# --- cache ---

CACHE_LOOKUPS = Counter(
    "gateway_cache_lookups_total", "Cache lookups", ["result"]           # hit|miss|skipped
)

CACHE_SIMILARITY = Histogram(
    "gateway_cache_similarity",
    "Similarity score of the nearest cache neighbour on hits",
    buckets=(.80, .85, .88, .90, .92, .94, .96, .98, .99, 1.0),
)

CACHE_ENTRIES = Gauge("gateway_cache_entries", "Entries currently in the cache")

# --- resilience ---

PROVIDER_ATTEMPTS = Counter(
    "gateway_provider_attempts_total",
    "Individual provider attempts",
    ["provider", "outcome"],               # success|error|skipped_open
)

FAILOVERS = Counter(
    "gateway_failovers_total", "Requests served by a non-primary provider", ["from_provider", "to_provider"]
)

BREAKER_STATE = Gauge(
    "gateway_breaker_state",
    "Circuit breaker state (0=closed, 1=half_open, 2=open)",
    ["provider"],
)

BREAKER_TRANSITIONS = Counter(
    "gateway_breaker_transitions_total", "Breaker state changes", ["provider", "to_state"]
)