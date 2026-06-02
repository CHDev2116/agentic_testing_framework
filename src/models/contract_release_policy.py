"""Release overrides driven by contract_meta (e.g. unstable JSON repair audit)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from eval.arbitrator import DecisionConflict


@dataclass(frozen=True)
class ContractReleaseSettings:
    """Inference contract knobs that affect release after parse/repair."""

    unstable_repair_release: str = "REVIEW"

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ContractReleaseSettings":
        model_cfg = config.get("model_settings", {})
        inference_cfg = (
            model_cfg.get("inference", {}) if isinstance(model_cfg, dict) else {}
        )
        contract_cfg = (
            inference_cfg.get("contract", {})
            if isinstance(inference_cfg, dict)
            else {}
        )
        if not isinstance(contract_cfg, dict):
            contract_cfg = {}
        return cls(
            unstable_repair_release=str(
                contract_cfg.get("unstable_repair_release", "REVIEW")
            ).upper(),
        )


def contract_meta_from_ai_result(ai_result: Dict[str, Any]) -> Dict[str, Any]:
    meta = ai_result.get("contract_meta")
    return meta if isinstance(meta, dict) else {}


def apply_unstable_repair_release_policy(
    release: str,
    conflict_enum: DecisionConflict,
    contract_meta: Dict[str, Any],
    settings: ContractReleaseSettings,
) -> Tuple[str, DecisionConflict]:
    """
    When repair_audit flagged unstable_repair, apply configured release (default REVIEW).
    OFF leaves release unchanged.
    """
    mode = settings.unstable_repair_release
    if mode == "OFF" or not contract_meta.get("unstable_repair"):
        return release, conflict_enum
    if mode == "NO_GO":
        return "NO_GO", DecisionConflict.UNSTABLE_JSON_REPAIR
    if mode == "REVIEW":
        return "REVIEW", DecisionConflict.UNSTABLE_JSON_REPAIR
    return release, conflict_enum
