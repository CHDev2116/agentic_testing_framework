"""Load and run historical oracle regression cases (frozen release decisions)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from eval.arbitrator import arbitrate_decision
from models.contract_release_policy import (
    ContractReleaseSettings,
    apply_unstable_repair_release_policy,
    contract_meta_from_ai_result,
)
from models.semantic_asserts import arbitrate_with_semantic_asserts
from models.semantic_eval_settings import SemanticEvalSettings

DEFAULT_THRESHOLDS: Dict[str, float] = {
    "min_brightness": 40.0,
    "max_brightness": 220.0,
    "min_sharpness": 20.0,
}


def load_oracle_cases(path: str | Path) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    file_path = Path(path)
    for line_number, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            case = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at line {line_number} in {file_path}: {exc}") from exc
        if "id" not in case:
            raise ValueError(f"Missing id at line {line_number} in {file_path}")
        cases.append(case)
    return cases


def _thresholds_for_case(case: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(DEFAULT_THRESHOLDS)
    merged.update(case.get("thresholds") or {})
    return merged


def _resolve_semantic_settings(
    case: Dict[str, Any],
    config: Optional[Dict[str, Any]],
    semantic_settings: Optional[SemanticEvalSettings],
) -> SemanticEvalSettings:
    if semantic_settings is not None:
        return semantic_settings
    if config is not None:
        return SemanticEvalSettings.from_config(config)
    return SemanticEvalSettings.from_oracle_case(case)


def _resolve_contract_release_settings(
    case: Dict[str, Any],
    config: Optional[Dict[str, Any]],
    contract_release_settings: Optional[ContractReleaseSettings],
) -> ContractReleaseSettings:
    if contract_release_settings is not None:
        return contract_release_settings
    if config is not None:
        return ContractReleaseSettings.from_config(config)
    policy = case.get("contract_policy")
    if isinstance(policy, dict) and policy.get("unstable_repair_release") is not None:
        return ContractReleaseSettings(
            unstable_repair_release=str(policy["unstable_repair_release"]).upper()
        )
    return ContractReleaseSettings()


def run_oracle_case(
    case: Dict[str, Any],
    *,
    config: Optional[Dict[str, Any]] = None,
    semantic_settings: Optional[SemanticEvalSettings] = None,
    contract_release_settings: Optional[ContractReleaseSettings] = None,
) -> Dict[str, Any]:
    """
    Execute one regression case. Returns actual outputs for assertion/diffing.
    """
    thresholds = _thresholds_for_case(case)
    engine_metrics = dict(case["engine_metrics"])
    mode = str(case.get("mode", "semantic")).lower()
    sem_settings = _resolve_semantic_settings(case, config, semantic_settings)
    contract_settings = _resolve_contract_release_settings(
        case, config, contract_release_settings
    )

    if mode == "arbitrator":
        model_inference = dict(case["model_inference"])
        release, conflict_enum = arbitrate_decision(
            engine_metrics, model_inference, thresholds
        )
        return {
            "release": release,
            "conflict": conflict_enum.value,
            "semantic_errors": [],
            "override_applied": False,
            "unstable_repair": False,
        }

    metrics = dict(case.get("metrics") or engine_metrics)
    ai_result = dict(case["ai_result"])
    contract_meta = contract_meta_from_ai_result(ai_result)
    semantic_errors: List[str] = []
    override_applied = False
    if sem_settings.enabled:
        release, conflict_enum, outcome = arbitrate_with_semantic_asserts(
            engine_metrics,
            ai_result,
            metrics,
            thresholds,
            settings=sem_settings,
        )
        semantic_errors = list(outcome.semantic_errors)
        override_applied = outcome.override_applied
    else:
        model_inference = {
            "decision": ai_result.get("decision"),
            "status": ai_result.get("decision"),
            "confidence": ai_result.get("confidence"),
        }
        release, conflict_enum = arbitrate_decision(
            engine_metrics, model_inference, thresholds
        )
    release, conflict_enum = apply_unstable_repair_release_policy(
        release, conflict_enum, contract_meta, contract_settings
    )
    return {
        "release": release,
        "conflict": conflict_enum.value,
        "semantic_errors": semantic_errors,
        "override_applied": override_applied,
        "unstable_repair": bool(contract_meta.get("unstable_repair")),
    }


def assert_case_matches(case: Dict[str, Any], actual: Dict[str, Any]) -> List[str]:
    """Return list of mismatch messages; empty if case passes."""
    errors: List[str] = []
    case_id = case["id"]

    expected_release = case.get("expected_release")
    if expected_release is not None and actual["release"] != expected_release:
        errors.append(
            f"{case_id}: expected release {expected_release!r}, got {actual['release']!r}"
        )

    expected_conflict = case.get("expected_conflict")
    if expected_conflict is not None and actual["conflict"] != expected_conflict:
        errors.append(
            f"{case_id}: expected conflict {expected_conflict!r}, got {actual['conflict']!r}"
        )

    if case.get("expect_semantic_errors") and not actual["semantic_errors"]:
        errors.append(f"{case_id}: expected semantic_errors non-empty")

    if "expect_override_applied" in case:
        expected_override = bool(case["expect_override_applied"])
        if actual["override_applied"] != expected_override:
            errors.append(
                f"{case_id}: expected override_applied={expected_override}, "
                f"got {actual['override_applied']}"
            )

    if "expect_unstable_repair" in case:
        expected_unstable = bool(case["expect_unstable_repair"])
        if bool(actual.get("unstable_repair")) != expected_unstable:
            errors.append(
                f"{case_id}: expected unstable_repair={expected_unstable}, "
                f"got {actual.get('unstable_repair')!r}"
            )

    return errors


def run_regression_suite(cases_path: str | Path) -> List[str]:
    """Run all cases; return combined mismatch messages."""
    all_errors: List[str] = []
    for case in load_oracle_cases(cases_path):
        actual = run_oracle_case(case)
        all_errors.extend(assert_case_matches(case, actual))
    return all_errors
