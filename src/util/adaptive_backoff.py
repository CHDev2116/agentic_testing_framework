"""
Adaptive rate limiting for async HTTP inference (roadmap implementation).

Detects pressure via HTTP status codes and applies exponential backoff with jitter
before retrying idempotent requests.
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass
from typing import Any, Dict

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


@dataclass
class AdaptiveBackoffSettings:
    enabled: bool = False
    max_retries: int = 2
    base_delay_s: float = 0.25
    max_delay_s: float = 8.0
    jitter_ratio: float = 0.2
    concurrency_floor: int = 1

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AdaptiveBackoffSettings":
        runtime = config.get("runtime", {})
        cfg = runtime.get("adaptive_backoff", {})
        if not isinstance(cfg, dict):
            cfg = {}
        return cls(
            enabled=bool(cfg.get("enabled", False)),
            max_retries=max(0, int(cfg.get("max_retries", 2))),
            base_delay_s=float(cfg.get("base_delay_s", 0.25)),
            max_delay_s=float(cfg.get("max_delay_s", 8.0)),
            jitter_ratio=float(cfg.get("jitter_ratio", 0.2)),
            concurrency_floor=max(1, int(cfg.get("concurrency_floor", 1))),
        )


class AdaptiveConcurrencyGate:
    """Optional semaphore cap that can be lowered under pressure."""

    def __init__(self, initial_permits: int, floor: int) -> None:
        self._floor = max(1, floor)
        self._permits = max(self._floor, initial_permits)
        self._semaphore = asyncio.Semaphore(self._permits)

    @property
    def permits(self) -> int:
        return self._permits

    def reduce_permits(self, step: int = 1) -> None:
        new_value = max(self._floor, self._permits - max(1, step))
        if new_value != self._permits:
            logger.warning(
                "adaptive_backoff: reducing concurrency permits %s -> %s",
                self._permits,
                new_value,
            )
            self._permits = new_value
            self._semaphore = asyncio.Semaphore(self._permits)

    async def acquire(self) -> None:
        await self._semaphore.acquire()

    def release(self) -> None:
        self._semaphore.release()


def compute_backoff_delay(
    attempt: int,
    *,
    base_delay_s: float,
    max_delay_s: float,
    jitter_ratio: float,
) -> float:
    delay = min(max_delay_s, base_delay_s * (2**attempt))
    jitter = delay * jitter_ratio * random.random()
    return delay + jitter


async def sleep_backoff(
    attempt: int,
    settings: AdaptiveBackoffSettings,
    *,
    reason: str,
) -> None:
    delay = compute_backoff_delay(
        attempt,
        base_delay_s=settings.base_delay_s,
        max_delay_s=settings.max_delay_s,
        jitter_ratio=settings.jitter_ratio,
    )
    logger.info(
        "adaptive_backoff: sleeping %.2fs (attempt=%s reason=%s)",
        delay,
        attempt + 1,
        reason,
    )
    await asyncio.sleep(delay)


def response_indicates_pressure(status_code: int) -> bool:
    return status_code in RETRYABLE_STATUS_CODES
