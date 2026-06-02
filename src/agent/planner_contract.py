"""JSON contract repair for LLM loopback planner responses (P1 planner scope)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from models.contract_repair import extract_json_object
from models.contract_validator import build_repair_user_message, contract_error_class

logger = logging.getLogger(__name__)

VALID_PLANNER_ACTIONS = frozenset({"brighten", "dim", "sharpen", "stop"})
REQUIRED_PLANNER_KEYS = ("action", "rationale")

DEFAULT_PLANNER_REPAIR_SUFFIX = (
    "Return ONLY a single JSON object with keys action, rationale. "
    f"Valid action values: {', '.join(sorted(VALID_PLANNER_ACTIONS))}."
)


@dataclass(frozen=True)
class PlannerRepairSettings:
    max_json_repair_attempts: int = 0
    repair_prompt_suffix: str = DEFAULT_PLANNER_REPAIR_SUFFIX

    @classmethod
    def from_planner_cfg(cls, planner_cfg: Dict[str, Any]) -> "PlannerRepairSettings":
        contract_cfg = planner_cfg.get("contract", {})
        if not isinstance(contract_cfg, dict):
            contract_cfg = {}
        return cls(
            max_json_repair_attempts=max(
                0, int(contract_cfg.get("max_json_repair_attempts", 0))
            ),
            repair_prompt_suffix=str(
                contract_cfg.get("repair_prompt_suffix", DEFAULT_PLANNER_REPAIR_SUFFIX)
            ),
        )

    @property
    def repair_enabled(self) -> bool:
        return self.max_json_repair_attempts > 0


def validate_planner_payload(payload: Any) -> List[str]:
    errors: List[str] = []
    if not isinstance(payload, dict) or not payload:
        errors.append("planner payload must be a non-empty JSON object")
        return errors
    for key in REQUIRED_PLANNER_KEYS:
        if key not in payload:
            errors.append(f"missing required planner key: {key}")
    action = str(payload.get("action", "")).lower().strip()
    if action and action not in VALID_PLANNER_ACTIONS:
        errors.append(f"planner action {action!r} is not allowed")
    rationale = payload.get("rationale")
    if isinstance(rationale, str) and not rationale.strip():
        errors.append("planner rationale must be non-empty")
    return errors


def run_planner_contract_loop(
    settings: PlannerRepairSettings,
    *,
    build_initial_prompt: Callable[[], str],
    fetch_model_text: Callable[[str], str],
) -> tuple[Dict[str, Any], int]:
    """
    Returns (parsed_payload, repair_attempts).
    When repair disabled, returns best-effort parse (may be invalid/empty).
    """
    if not settings.repair_enabled:
        raw = fetch_model_text(build_initial_prompt())
        return extract_json_object(raw), 0

    repair_attempts = 0
    prompt = build_initial_prompt()
    last_raw = ""
    last_errors: List[str] = []

    while True:
        last_raw = fetch_model_text(prompt)
        parsed = extract_json_object(last_raw)
        errors = validate_planner_payload(parsed)
        if not errors:
            return parsed, repair_attempts
        last_errors = errors
        if repair_attempts >= settings.max_json_repair_attempts:
            logger.warning(
                "planner_contract: repair_exhausted attempts=%s errors=%s",
                repair_attempts,
                last_errors,
            )
            return parsed if isinstance(parsed, dict) else {}, repair_attempts
        repair_attempts += 1
        error_class = contract_error_class(parsed, errors)
        logger.info(
            "planner_contract: repair attempt=%s error_class=%s",
            repair_attempts,
            error_class,
        )
        prompt = (
            f"{build_initial_prompt()}\n\n"
            f"{build_repair_user_message(raw_snippet=last_raw, errors=errors, required_keys=REQUIRED_PLANNER_KEYS, suffix=settings.repair_prompt_suffix)}"
        )

    return {}, repair_attempts
