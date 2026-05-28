"""
Performance monitoring helpers for the Agentic Testing Framework (agent/util layers).

Provides sync/async decorators that log wall time; optional peak traced allocation
via ``tracemalloc`` when ``ATF_MONITOR_MEMORY`` is enabled (profile/debug).
Legacy: ``PIXELQA_MONITOR_MEMORY`` is accepted as an alias.

Also provides a simple wall-time context manager for inline sections.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import time
import tracemalloc
from contextlib import contextmanager
from typing import Any, Awaitable, Callable, Dict, Iterator, List, Optional, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def _memory_tracing_enabled() -> bool:
    """Enable tracemalloc peak/current MB in decorator logs (extra overhead)."""
    for key in ("ATF_MONITOR_MEMORY", "PIXELQA_MONITOR_MEMORY"):
        v = os.environ.get(key, "").strip().lower()
        if v in ("1", "true", "yes", "on"):
            return True
    return False


def monitor_performance(func: F) -> F:
    """
    Decorator for synchronous callables: records elapsed time; optional peak memory (tracemalloc).

    Memory tracing is controlled by env ``ATF_MONITOR_MEMORY`` (default: off) to avoid
    overhead on very hot call paths.

    Logs entry at DEBUG and completion at INFO so expensive paths stay observable without
    spamming default INFO-only setups.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        name = getattr(func, "__name__", repr(func))
        logger.debug(
            "monitor_performance(sync): enter func=%s args_len=%d kwargs_keys=%s",
            name,
            len(args),
            list(kwargs.keys())[:12],
        )
        trace_mem = _memory_tracing_enabled()
        if trace_mem:
            tracemalloc.start()
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            if trace_mem:
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                logger.info(
                    "monitor_performance(sync): done func=%s elapsed_sec=%.6f "
                    "peak_traced_memory_mb=%.4f current_traced_memory_mb=%.4f memory_trace=on",
                    name,
                    elapsed,
                    peak / (1024.0 * 1024.0),
                    current / (1024.0 * 1024.0),
                )
            else:
                logger.info(
                    "monitor_performance(sync): done func=%s elapsed_sec=%.6f",
                    name,
                    elapsed,
                )

    return wrapper  # type: ignore[return-value]


def async_monitor_performance(func: F) -> F:
    """
    Decorator for async callables: same metrics as sync; detailed logging on enter/exit.

    Workspace convention: async paths keep explicit structured logs for observability.
    """

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        name = getattr(func, "__name__", repr(func))
        logger.debug(
            "monitor_performance(async): enter func=%s args_len=%d kwargs_keys=%s",
            name,
            len(args),
            list(kwargs.keys())[:12],
        )
        trace_mem = _memory_tracing_enabled()
        if trace_mem:
            tracemalloc.start()
        start = time.perf_counter()
        try:
            return await func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            if trace_mem:
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                logger.info(
                    "monitor_performance(async): done func=%s elapsed_sec=%.6f "
                    "peak_traced_memory_mb=%.4f current_traced_memory_mb=%.4f memory_trace=on",
                    name,
                    elapsed,
                    peak / (1024.0 * 1024.0),
                    current / (1024.0 * 1024.0),
                )
            else:
                logger.info(
                    "monitor_performance(async): done func=%s elapsed_sec=%.6f",
                    name,
                    elapsed,
                )

    return wrapper  # type: ignore[return-value]


@contextmanager
def measure_wall_time(label: str, *, extra: Optional[Dict[str, Any]] = None) -> Iterator[None]:
    """
    Context manager for a single labeled section; logs start (DEBUG) and duration (INFO).

    Example:
        with measure_wall_time("load_image"):
            ...
    """
    logger.debug(
        "measure_wall_time: start label=%s extra=%s",
        label,
        extra or {},
    )
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info("measure_wall_time: end label=%s elapsed_sec=%.6f", label, elapsed)


async def gather_with_timing(
    coros: List[Awaitable[Any]],
    *,
    label: str = "gather",
) -> List[Any]:
    """
    Run asyncio.gather on awaitables and log total wall time for the batch.

    Useful when orchestrating multiple provider calls without wrapping each one.
    """
    logger.debug(
        "gather_with_timing: start label=%s count=%d",
        label,
        len(coros),
    )
    start = time.perf_counter()
    try:
        results = await asyncio.gather(*coros)
        return list(results)
    finally:
        elapsed = time.perf_counter() - start
        logger.info(
            "gather_with_timing: done label=%s elapsed_sec=%.6f count=%d",
            label,
            elapsed,
            len(coros),
        )


class PerformanceMonitor:
    """
    Lightweight cumulative stats for a named phase (e.g. one pipeline step).

    Not thread-safe; use one instance per task/thread or guard externally.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._total_sec = 0.0
        self._calls = 0

    def record(self, elapsed_sec: float) -> None:
        self._total_sec += elapsed_sec
        self._calls += 1
        logger.debug(
            "PerformanceMonitor(%s): record elapsed_sec=%.6f calls=%d cumulative_sec=%.6f",
            self.name,
            elapsed_sec,
            self._calls,
            self._total_sec,
        )

    def summary(self) -> Dict[str, Any]:
        out = {
            "name": self.name,
            "calls": self._calls,
            "total_sec": round(self._total_sec, 6),
            "avg_sec": round(self._total_sec / self._calls, 6) if self._calls else 0.0,
        }
        logger.info("PerformanceMonitor summary: %s", out)
        return out


__all__ = [
    "monitor_performance",
    "async_monitor_performance",
    "measure_wall_time",
    "gather_with_timing",
    "PerformanceMonitor",
]
