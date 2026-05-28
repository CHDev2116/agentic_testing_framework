from __future__ import annotations

import json
import logging
from importlib import import_module
from typing import Any, Dict, List, Protocol

from models.contracts import LoopbackPlan

logger = logging.getLogger(__name__)

_REQUESTS_MODULE = None


def _get_requests():
    global _REQUESTS_MODULE
    if _REQUESTS_MODULE is None:
        _REQUESTS_MODULE = import_module("requests")
    return _REQUESTS_MODULE


class LoopbackPlanner(Protocol):
    def plan(
        self,
        *,
        signal: str,
        engine_metrics: Dict[str, Any],
        thresholds_cfg: Dict[str, Any],
        loopback_guard_cfg: Dict[str, Any],
        attempt_history: List[Dict[str, Any]],
    ) -> LoopbackPlan:
        ...


class SimulatedLoopbackPlanner:
    """Rule-based planner used as deterministic fallback."""

    def plan(
        self,
        *,
        signal: str,
        engine_metrics: Dict[str, Any],
        thresholds_cfg: Dict[str, Any],
        loopback_guard_cfg: Dict[str, Any],
        attempt_history: List[Dict[str, Any]],
    ) -> LoopbackPlan:
        brightness = float(
            engine_metrics.get("avg_brightness", engine_metrics.get("brightness", 0.0))
        )
        sharpness = float(engine_metrics.get("sharpness", 0.0))
        min_brightness = float(thresholds_cfg.get("min_brightness", 40.0))
        max_brightness = float(thresholds_cfg.get("max_brightness", 220.0))
        min_sharpness = float(thresholds_cfg.get("min_sharpness", 20.0))
        overexposure_stop_ratio = float(
            loopback_guard_cfg.get("overexposure_stop_ratio", 0.95)
        )
        underexposure_stop_ratio = float(
            loopback_guard_cfg.get("underexposure_stop_ratio", 1.05)
        )

        if signal == "under":
            if brightness >= min_brightness:
                return LoopbackPlan(
                    None,
                    "engine_disagrees_underexposed",
                    "model says under but engine brightness is acceptable",
                    planner_backend="simulated",
                )
            if brightness >= (max_brightness * overexposure_stop_ratio):
                return LoopbackPlan(
                    None,
                    "near_overexposure_guard",
                    "brighten would likely push image into over-exposure",
                    planner_backend="simulated",
                )
            return LoopbackPlan(
                "brighten",
                "retry_scheduled",
                "under-exposed signal and safe brightness headroom",
                planner_backend="simulated",
            )

        if signal == "over":
            if brightness <= max_brightness:
                return LoopbackPlan(
                    None,
                    "engine_disagrees_overexposed",
                    "model says over but engine brightness is acceptable",
                    planner_backend="simulated",
                )
            if brightness <= (min_brightness * underexposure_stop_ratio):
                return LoopbackPlan(
                    None,
                    "near_underexposure_guard",
                    "dimming would likely push image into under-exposure",
                    planner_backend="simulated",
                )
            return LoopbackPlan(
                "dim",
                "retry_scheduled",
                "over-exposed signal and safe dimming headroom",
                planner_backend="simulated",
            )

        if signal == "blurry":
            if sharpness >= min_sharpness:
                return LoopbackPlan(
                    None,
                    "engine_disagrees_blurry",
                    "model says blurry but engine sharpness is acceptable",
                    planner_backend="simulated",
                )
            return LoopbackPlan(
                "sharpen",
                "retry_scheduled",
                "blurry signal and low sharpness metric",
                planner_backend="simulated",
            )

        return LoopbackPlan(
            None,
            f"signal_not_recoverable ({signal})",
            "signal is outside supported recovery actions",
            planner_backend="simulated",
        )


class LLMLoopbackPlanner:
    """LLM planner that emits next_action JSON with fallback to simulated."""

    VALID_ACTIONS = {"brighten", "dim", "sharpen", "stop"}

    def __init__(self, planner_cfg: Dict[str, Any], fallback_planner: LoopbackPlanner):
        self.fallback_planner = fallback_planner
        self.host = str(planner_cfg.get("host", "http://127.0.0.1:8080")).rstrip("/")
        self.endpoint = str(planner_cfg.get("endpoint", "/v1/chat/completions"))
        self.model = str(planner_cfg.get("model", "local-model"))
        self.timeout_s = float(planner_cfg.get("timeout_s", 20.0))
        self.temperature = float(planner_cfg.get("temperature", 0.0))
        self.max_tokens = int(planner_cfg.get("max_tokens", 200))
        self.health_check_timeout_s = float(
            planner_cfg.get("health_check_timeout_s", self.timeout_s)
        )

    def _build_payload(
        self,
        *,
        signal: str,
        engine_metrics: Dict[str, Any],
        thresholds_cfg: Dict[str, Any],
        loopback_guard_cfg: Dict[str, Any],
        attempt_history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        prompt = (
            "You are an image QA recovery planner.\n"
            "Return STRICT JSON with keys: action, rationale.\n"
            "Valid action: brighten, dim, sharpen, stop.\n"
            "Choose stop if recovery is not safe or not meaningful.\n"
            f"signal={signal}\n"
            f"engine_metrics={json.dumps(engine_metrics, ensure_ascii=False)}\n"
            f"thresholds={json.dumps(thresholds_cfg, ensure_ascii=False)}\n"
            f"loopback_guard={json.dumps(loopback_guard_cfg, ensure_ascii=False)}\n"
            f"attempt_history={json.dumps(attempt_history[-3:], ensure_ascii=False)}\n"
        )
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
            "response_format": {"type": "json_object"},
        }

    @staticmethod
    def _extract_json_object(raw_text: str) -> Dict[str, Any]:
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            start = raw_text.find("{")
            end = raw_text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return {}
            try:
                return json.loads(raw_text[start : end + 1])
            except json.JSONDecodeError:
                return {}

    def plan(
        self,
        *,
        signal: str,
        engine_metrics: Dict[str, Any],
        thresholds_cfg: Dict[str, Any],
        loopback_guard_cfg: Dict[str, Any],
        attempt_history: List[Dict[str, Any]],
    ) -> LoopbackPlan:
        payload = self._build_payload(
            signal=signal,
            engine_metrics=engine_metrics,
            thresholds_cfg=thresholds_cfg,
            loopback_guard_cfg=loopback_guard_cfg,
            attempt_history=attempt_history,
        )
        url = f"{self.host}{self.endpoint}"
        logger.info("Loopback planner (llm): requesting next action from %s", url)
        try:
            response = _get_requests().post(url, json=payload, timeout=self.timeout_s)
            response.raise_for_status()
            body = response.json()
            content = str(
                body.get("choices", [{}])[0].get("message", {}).get("content", "")
            )
            parsed = self._extract_json_object(content)
            action = str(parsed.get("action", "stop")).lower()
            rationale = str(parsed.get("rationale", "planner returned no rationale"))
            if action not in self.VALID_ACTIONS:
                logger.warning(
                    "Loopback planner (llm): invalid action=%s, fallback planner is used",
                    action,
                )
                fallback_plan = self.fallback_planner.plan(
                    signal=signal,
                    engine_metrics=engine_metrics,
                    thresholds_cfg=thresholds_cfg,
                    loopback_guard_cfg=loopback_guard_cfg,
                    attempt_history=attempt_history,
                )
                return LoopbackPlan(
                    action=fallback_plan.action,
                    stop_reason=fallback_plan.stop_reason,
                    rationale=fallback_plan.rationale,
                    fallback_used=True,
                    planner_backend="llm->simulated",
                )
            if action == "stop":
                return LoopbackPlan(
                    None, "planner_stop", rationale, fallback_used=False, planner_backend="llm"
                )
            return LoopbackPlan(
                action, "retry_scheduled", rationale, fallback_used=False, planner_backend="llm"
            )
        except Exception as exc:
            logger.warning(
                "Loopback planner (llm): failed with %s, fallback planner is used",
                exc,
            )
            fallback_plan = self.fallback_planner.plan(
                signal=signal,
                engine_metrics=engine_metrics,
                thresholds_cfg=thresholds_cfg,
                loopback_guard_cfg=loopback_guard_cfg,
                attempt_history=attempt_history,
            )
            return LoopbackPlan(
                action=fallback_plan.action,
                stop_reason=fallback_plan.stop_reason,
                rationale=fallback_plan.rationale,
                fallback_used=True,
                planner_backend="llm->simulated",
            )

    def ensure_healthy(self) -> None:
        """
        Fail fast when planner endpoint is unreachable.
        """
        url = f"{self.host}{self.endpoint}"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": "health_check"}],
            "temperature": 0.0,
            "max_tokens": 1,
            "stream": False,
        }
        try:
            response = _get_requests().post(
                url,
                json=payload,
                timeout=self.health_check_timeout_s,
            )
            logger.info(
                "Loopback planner health check: reachable endpoint %s (status=%s)",
                url,
                response.status_code,
            )
        except Exception as exc:
            raise RuntimeError(
                f"LLM planner server is not reachable at {url}. "
                "Start the planner backend server or switch --loopback-planner simulated."
            ) from exc


def create_loopback_planner(config: Dict[str, Any]) -> LoopbackPlanner:
    runtime_cfg = config.get("runtime", {})
    planner_cfg = runtime_cfg.get("loopback_planner", {})
    planner_mode = str(planner_cfg.get("mode", "simulated")).lower()
    simulated = SimulatedLoopbackPlanner()
    if planner_mode == "llm":
        llm_cfg = planner_cfg.get("llm", {})
        require_healthy_on_startup = bool(
            planner_cfg.get("require_healthy_on_startup", True)
        )
        logger.info("Loopback planner: LLM mode enabled")
        planner = LLMLoopbackPlanner(planner_cfg=llm_cfg, fallback_planner=simulated)
        if require_healthy_on_startup:
            planner.ensure_healthy()
        return planner
    logger.info("Loopback planner: simulated mode enabled")
    return simulated
