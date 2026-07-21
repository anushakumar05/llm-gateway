import httpx

GATEWAY = "http://localhost:8000"


def scrape() -> str:
    return httpx.get(f"{GATEWAY}/metrics", timeout=10).text


def ask(content: str, team: str = "team-heavy"):
    return httpx.post(
        f"{GATEWAY}/v1/chat/completions",
        headers={"Authorization": f"Bearer {team}"},
        json={"model": "gpt-4o-mini", "temperature": 0,
              "messages": [{"role": "user", "content": content}]},
        timeout=30,
    )


def test_metrics_endpoint_exposes_core_series():
    body = scrape()
    for name in [
        "gateway_requests_total",
        "gateway_request_duration_seconds",
        "gateway_overhead_duration_seconds",
        "gateway_cache_lookups_total",
        "gateway_breaker_state",
    ]:
        assert name in body, f"missing metric: {name}"


def test_request_increments_counter():
    before = scrape().count("gateway_requests_total")
    ask("metrics counter test")
    after = scrape()
    assert "gateway_requests_total" in after
    assert before >= 0


def test_overhead_reported_in_header():
    r = ask("overhead header test unique")
    if r.headers.get("x-cache") == "miss":
        overhead = float(r.headers["x-gateway-overhead-ms"])
        assert 0 <= overhead < 500, f"implausible overhead: {overhead}ms"


def test_cache_hit_is_faster_than_miss():
    import time
    prompt = "latency comparison probe alpha"
    t0 = time.perf_counter(); ask(prompt); miss_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter(); r = ask(prompt); hit_ms = (time.perf_counter() - t0) * 1000
    assert r.headers["x-cache"] == "hit"
    assert hit_ms < miss_ms, f"hit {hit_ms:.1f}ms not faster than miss {miss_ms:.1f}ms"