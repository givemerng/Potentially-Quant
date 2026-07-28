"""Monitoring Metrics Collector tracking request counts, latencies, and cache hit ratios."""

from __future__ import annotations
from threading import Lock
from typing import Dict, Any


class MetricsCollector:
    """In-Memory metrics collector for API observability."""

    def __init__(self) -> None:
        self._lock = Lock()
        self.request_count = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def record_request(self) -> None:
        with self._lock:
            self.request_count += 1

    def record_cache_hit(self) -> None:
        with self._lock:
            self.cache_hits += 1

    def record_cache_miss(self) -> None:
        with self._lock:
            self.cache_misses += 1

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            total_cache = self.cache_hits + self.cache_misses
            hit_ratio = (self.cache_hits / total_cache) if total_cache > 0 else 0.0
            return {
                "request_count": self.request_count,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_hit_ratio": round(hit_ratio, 4),
            }


metrics_collector = MetricsCollector()
