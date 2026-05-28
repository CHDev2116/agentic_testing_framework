"""
Process-pool execution for CPU-bound image metrics (Pillow stdev/mean on pixels).

Batch runs with many images spend significant time in calculate_metrics; using
multiple processes avoids the GIL and speeds throughput on multi-core hosts.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

from engine.vision_math import calculate_metrics

logger = logging.getLogger(__name__)


def _default_max_workers() -> int:
    cpu = os.cpu_count() or 1
    return max(1, min(cpu, 8))


class MetricsProcessPool:
    """
    Context manager wrapping ProcessPoolExecutor for calculate_metrics calls.
    """

    def __init__(self, max_workers: Optional[int] = None) -> None:
        self.max_workers = max(1, int(max_workers or _default_max_workers()))
        self._executor: Optional[ProcessPoolExecutor] = None

    def __enter__(self) -> MetricsProcessPool:
        logger.info(
            "MetricsProcessPool: starting process pool max_workers=%s",
            self.max_workers,
        )
        self._executor = ProcessPoolExecutor(max_workers=self.max_workers)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._executor is not None:
            logger.debug("MetricsProcessPool: shutting down process pool")
            self._executor.shutdown(wait=True)
            self._executor = None

    @property
    def executor(self) -> ProcessPoolExecutor:
        if self._executor is None:
            raise RuntimeError("MetricsProcessPool is not active; use as a context manager")
        return self._executor

    def calculate(self, photo_path: str):
        """
        Run calculate_metrics in a worker process (blocking in caller thread).
        """
        if self._executor is None:
            raise RuntimeError("MetricsProcessPool is not active")
        future = self._executor.submit(calculate_metrics, photo_path)
        return future.result()
