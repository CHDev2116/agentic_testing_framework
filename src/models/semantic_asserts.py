"""Domain semantic asserts for vision QA inference (P2a + policy layer)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from eval.arbitrator import DecisionConflict, _is_physically_ok, arbitrate_decision
from models.contract_validator import ALLOWED_DECISIONS
from models.semantic_eval_settings import SemanticEvalSettings

SEMANTIC_ASSERT_MISMATCH = "SEMANTIC_ASSERT_MISMATCH"

EXPECTED_CODE_BY_DECISION: Dict[str, frozenset] = {
    "Optimal": frozenset({"SUCCESS_200"}),
    "Blurry": frozenset({"ERR_OPTIC_SHRP_001"}),
    "Under-exposed": frozenset({"ERR_LIGHT_DARK_002"}),
    "Over-exposed": frozenset({"ERR_LIGHT_BRGT_003"}),
}


@dataclass(frozen=True)
class SemanticAssertOutcome:
    semantic_errors: List[str]
    arbitration_inference: Dict[str, Any]
    override_applied: bool
    invalid_label: bool = False
    confidence_violation: bool = False
    inference_error_verdict: bool = False


def _brightness(metrics: Dict[str, Any]) -> float:
    return float(metrics.get("avg_brightness", metrics.get("brightness", 0.0)))


def _sharpness(metrics: Dict[str, Any]) -> float:
    return float(metrics.get("sharpness", 0.0))


def metrics_implied_decision(metrics: Dict[str, Any], thresholds: Dict[str, Any]) -> str:
    thresholds = thresholds or {}
    min_b = float(thresholds.get("min_brightness", 45.0))
    max_b = float(thresholds.get("max_brightness", 220.0))
    min_s = float(thresholds.get("min_sharpness", 20.0))
    brightness = _brightness(metrics)
    sharpness = _sharpness(metrics)

    if brightness < min_b:
        return "Under-exposed"
    if brightness > max_b:
        return "Over-exposed"
    if sharpness < min_s:
        return "Blurry"
    return "Optimal"


def optimal_contradicts_metrics(metrics: Dict[str, Any], thresholds: Dict[str, Any]) -> bool:
    return metrics_implied_decision(metrics, thresholds) != "Optimal"


def validate_decision_code_consistency(decision: str, code: str) -> Optional[str]:
    decision = str(decision or "").strip()
    code = str(code or "").strip()
    if not decision or not code:
        return None
    expected_codes = EXPECTED_CODE_BY_DECISION.get(decision)
    if expected_codes is not None:
        if code not in expected_codes:
            allowed = ", ".join(sorted(expected_codes))
            return (
                f"semantic: decision {decision!r} expects code in {{{allowed}}}, got {code!r}"
            )
        return None
    if decision == "Error":
        if code == "SUCCESS_200":
            return "semantic: decision Error must not use SUCCESS_200"
        if not code.startswith("ERR_"):
            return f"semantic: decision Error expects ERR_* code, got {code!r}"
        return None
    return None


def evaluate_semantic_asserts(
    ai_result: Dict[str, Any],
    metrics: Dict[str, Any],
    thresholds: Optional[Dict[str, Any]] = None,
) -> SemanticAssertOutcome:
    thresholds = thresholds or {}
    errors: List[str] = []
    ai_result = ai_result if isinstance(ai_result, dict) else {}

    decision = str(ai_result.get("decision", "")).strip()
    arbitration: Dict[str, Any] = {
        "decision": ai_result.get("decision"),
        "status": ai_result.get("decision"),
        "confidence": ai_result.get("confidence"),
    }

    invalid_label = bool(decision and decision not in ALLOWED_DECISIONS)
    if invalid_label:
        allowed = ", ".join(sorted(ALLOWED_DECISIONS))
        errors.append(f"semantic: decision {decision!r} not in allowed set ({allowed})")

    if decision == "Optimal" and isinstance(metrics, dict) and optimal_contradicts_metrics(
        metrics, thresholds
    ):
        implied = metrics_implied_decision(metrics, thresholds)
        errors.append(
            f"semantic: Optimal contradicts metrics (expected {implied!r} from brightness/sharpness)"
        )

    code = str(ai_result.get("code", "")).strip()
    code_error = validate_decision_code_consistency(decision, code)
    if code_error:
        errors.append(code_error)

    confidence_violation = False
    raw_confidence = ai_result.get("confidence")
    if raw_confidence is not None:
        try:
            confidence = float(raw_confidence)
            if confidence < 0.0 or confidence > 1.0:
                errors.append("semantic: confidence must be in [0, 1]")
                arbitration["confidence"] = None
                confidence_violation = True
        except (TypeError, ValueError):
            errors.append("semantic: confidence must be a number")
            arbitration["confidence"] = None
            confidence_violation = True

    inference_error_verdict = decision == "Error" or str(
        ai_result.get("code", "")
    ).startswith("ERR_MODEL")
    if inference_error_verdict:
        errors.append(
            "semantic: inference error verdict (model Error or ERR_MODEL_* code)"
        )

    override_applied = False
    if errors:
        if decision == "Optimal" and isinstance(metrics, dict) and optimal_contradicts_metrics(
            metrics, thresholds
        ):
            implied = metrics_implied_decision(metrics, thresholds)
            arbitration["decision"] = implied
            arbitration["status"] = implied
            override_applied = True
        elif invalid_label:
            arbitration["decision"] = "Error"
            arbitration["status"] = "Error"
            override_applied = True

    return SemanticAssertOutcome(
        semantic_errors=errors,
        arbitration_inference=arbitration,
        override_applied=override_applied,
        invalid_label=invalid_label,
        confidence_violation=confidence_violation,
        inference_error_verdict=inference_error_verdict,
    )


def apply_semantic_release_policy(
    outcome: SemanticAssertOutcome,
    release: str,
    conflict_enum: DecisionConflict,
    *,
    engine_metrics: Dict[str, Any],
    thresholds: Optional[Dict[str, Any]],
    settings: SemanticEvalSettings,
) -> Tuple[str, DecisionConflict]:
    thresholds = thresholds or {}

    if outcome.invalid_label:
        if settings.invalid_label_release == "REVIEW":
            return "REVIEW", DecisionConflict.SEMANTIC_INVALID_LABEL
        return "NO_GO", DecisionConflict.SEMANTIC_INVALID_LABEL

    if outcome.inference_error_verdict and settings.inference_error_release == "NO_GO":
        return "NO_GO", DecisionConflict.SEMANTIC_INFERENCE_ERROR

    if (
        outcome.confidence_violation
        and settings.confidence_violation_policy == "review"
        and _is_physically_ok(engine_metrics, thresholds)
    ):
        return "REVIEW", DecisionConflict.SEMANTIC_CONFIDENCE_VIOLATION

    return release, conflict_enum


def contract_block_from_outcome(outcome: SemanticAssertOutcome) -> Dict[str, Any]:
    errors = list(outcome.semantic_errors)
    block: Dict[str, Any] = {"semantic_errors": errors}
    if any("expects code" in err or "must not use SUCCESS_200" in err for err in errors):
        block["code_mismatch"] = True
    if outcome.invalid_label:
        block["invalid_label"] = True
    if outcome.confidence_violation:
        block["confidence_violation"] = True
    if outcome.inference_error_verdict:
        block["inference_error_verdict"] = True
    return block


def arbitrate_with_semantic_asserts(
    engine_metrics: Dict[str, Any],
    ai_result: Dict[str, Any],
    metrics: Dict[str, Any],
    thresholds: Optional[Dict[str, Any]] = None,
    *,
    settings: Optional[SemanticEvalSettings] = None,
) -> tuple[str, Any, SemanticAssertOutcome]:
    policy = settings or SemanticEvalSettings()
    outcome = evaluate_semantic_asserts(ai_result, metrics, thresholds)
    release, conflict_enum = arbitrate_decision(
        engine_metrics, outcome.arbitration_inference, thresholds
    )
    release, conflict_enum = apply_semantic_release_policy(
        outcome,
        release,
        conflict_enum,
        engine_metrics=engine_metrics,
        thresholds=thresholds,
        settings=policy,
    )
    return release, conflict_enum, outcome
