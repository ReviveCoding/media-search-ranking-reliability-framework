from __future__ import annotations

import numpy as np
import time


def latency_summary(latencies_ms: list[float]) -> dict:
    if not latencies_ms:
        return {"p50_latency_ms": 0.0, "p95_latency_ms": 0.0, "p99_latency_ms": 0.0}
    arr = np.asarray(latencies_ms, dtype=float)
    return {
        "p50_latency_ms": float(np.percentile(arr, 50)),
        "p95_latency_ms": float(np.percentile(arr, 95)),
        "p99_latency_ms": float(np.percentile(arr, 99)),
    }


def measure_search_latency(
    search_fn,
    queries: list[str],
    top_k: int = 10,
    max_queries: int = 50,
    warmup_queries: int = 2,
):
    if not queries:
        return latency_summary([]), []
    for query in queries[: max(0, warmup_queries)]:
        search_fn(query, top_k=top_k)
    latencies = []
    for query in queries[:max_queries]:
        start = time.perf_counter()
        search_fn(query, top_k=top_k)
        latencies.append((time.perf_counter() - start) * 1000)
    return latency_summary(latencies), latencies
