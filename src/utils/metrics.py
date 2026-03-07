# Performance and cache metrics tracking
from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class _Metrics:
    cache_hits: int = 0
    cache_misses: int = 0
    total_requests: int = 0
    total_latency_seconds: float = 0.0
    # Latency buckets for a simple histogram (upper bounds in seconds)
    _latency_buckets: dict[float, int] = field(
        default_factory=lambda: {0.5: 0, 1.0: 0, 2.0: 0, 5.0: 0, 10.0: 0, float("inf"): 0}
    )
    _lock: Lock = field(default_factory=Lock)

    def record_hit(self, latency: float) -> None:
        with self._lock:
            self.cache_hits += 1
            self._record_latency(latency)

    def record_miss(self, latency: float) -> None:
        with self._lock:
            self.cache_misses += 1
            self._record_latency(latency)

    def _record_latency(self, latency: float) -> None:
        self.total_requests += 1
        self.total_latency_seconds += latency
        for bound in self._latency_buckets:
            if latency <= bound:
                self._latency_buckets[bound] += 1

    def hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total else 0.0

    def avg_latency(self) -> float:
        return self.total_latency_seconds / self.total_requests if self.total_requests else 0.0

    def to_prometheus(self) -> str:
        """Render metrics in Prometheus text exposition format."""
        lines = [
            "# HELP rtfm_cache_hits_total Total number of semantic cache hits",
            "# TYPE rtfm_cache_hits_total counter",
            f"rtfm_cache_hits_total {self.cache_hits}",
            "",
            "# HELP rtfm_cache_misses_total Total number of semantic cache misses",
            "# TYPE rtfm_cache_misses_total counter",
            f"rtfm_cache_misses_total {self.cache_misses}",
            "",
            "# HELP rtfm_request_latency_seconds Request processing latency",
            "# TYPE rtfm_request_latency_seconds histogram",
        ]
        for bound, count in self._latency_buckets.items():
            le = "+Inf" if bound == float("inf") else str(bound)
            lines.append(f'rtfm_request_latency_seconds_bucket{{le="{le}"}} {count}')
        lines.append(f"rtfm_request_latency_seconds_sum {self.total_latency_seconds:.4f}")
        lines.append(f"rtfm_request_latency_seconds_count {self.total_requests}")
        return "\n".join(lines) + "\n"


# Module-level singleton — import and use directly
metrics = _Metrics()


class Timer:
    """Context manager that measures elapsed wall-clock time."""

    def __init__(self) -> None:
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        self.elapsed = time.perf_counter() - self._start
