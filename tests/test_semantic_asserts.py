from eval.arbitrator import arbitrate_decision
from eval.arbitrator import DecisionConflict
from models.semantic_asserts import (
    SEMANTIC_ASSERT_MISMATCH,
    apply_semantic_release_policy,
    evaluate_semantic_asserts,
    metrics_implied_decision,
    validate_decision_code_consistency,
)
from models.semantic_eval_settings import SemanticEvalSettings


THRESHOLDS = {
    "min_brightness": 40.0,
    "max_brightness": 220.0,
    "min_sharpness": 20.0,
}


def test_metrics_implied_underexposed():
    metrics = {"avg_brightness": 10.0, "sharpness": 50.0}
    assert metrics_implied_decision(metrics, THRESHOLDS) == "Under-exposed"


def test_semantic_optimal_contradicts_dark_metrics():
    ai_result = {"decision": "Optimal", "code": "SUCCESS_200", "msg": "ok", "confidence": 0.9}
    metrics = {"avg_brightness": 10.0, "sharpness": 50.0}
    outcome = evaluate_semantic_asserts(ai_result, metrics, THRESHOLDS)
    assert outcome.semantic_errors
    assert outcome.override_applied
    assert outcome.arbitration_inference["decision"] == "Under-exposed"
    assert ai_result["decision"] == "Optimal"


def test_invalid_label_policy_forces_no_go():
    ai_result = {"decision": "LoginSuccess", "code": "SUCCESS_200", "msg": "wrong"}
    metrics = {"avg_brightness": 80.0, "sharpness": 50.0}
    engine_metrics = {"avg_brightness": 80.0, "sharpness": 50.0}
    outcome = evaluate_semantic_asserts(ai_result, metrics, THRESHOLDS)
    release, conflict = apply_semantic_release_policy(
        outcome,
        "GO",
        DecisionConflict.PHYSICAL_OK_MODEL_FAIL_LOW_CONFIDENCE,
        engine_metrics=engine_metrics,
        thresholds=THRESHOLDS,
        settings=SemanticEvalSettings(),
    )
    assert release == "NO_GO"
    assert conflict == DecisionConflict.SEMANTIC_INVALID_LABEL


def test_confidence_policy_forces_review_when_physical_ok():
    ai_result = {
        "decision": "Blurry",
        "code": "ERR_OPTIC_SHRP_001",
        "msg": "soft",
        "confidence": 2.0,
    }
    metrics = {"avg_brightness": 90.0, "sharpness": 60.0}
    engine_metrics = {"avg_brightness": 90.0, "sharpness": 60.0}
    outcome = evaluate_semantic_asserts(ai_result, metrics, THRESHOLDS)
    release, conflict = apply_semantic_release_policy(
        outcome,
        "NO_GO",
        DecisionConflict.ALL_FAIL,
        engine_metrics=engine_metrics,
        thresholds=THRESHOLDS,
        settings=SemanticEvalSettings(confidence_violation_policy="review"),
    )
    assert release == "REVIEW"
    assert conflict == DecisionConflict.SEMANTIC_CONFIDENCE_VIOLATION


def test_semantic_invalid_decision_overrides_arbitration_input():
    ai_result = {"decision": "LoginSuccess", "code": "SUCCESS_200", "msg": "ok"}
    metrics = {"avg_brightness": 80.0, "sharpness": 50.0}
    outcome = evaluate_semantic_asserts(ai_result, metrics, THRESHOLDS)
    assert outcome.override_applied
    assert outcome.arbitration_inference["decision"] == "Error"


def test_arbitration_no_go_when_optimal_overridden_for_dark_image():
    ai_result = {"decision": "Optimal", "code": "SUCCESS_200", "msg": "ok", "confidence": 0.95}
    metrics = {"avg_brightness": 10.0, "sharpness": 50.0}
    engine_metrics = {
        "avg_brightness": metrics["avg_brightness"],
        "sharpness": metrics["sharpness"],
    }
    outcome = evaluate_semantic_asserts(ai_result, metrics, THRESHOLDS)
    release, conflict = arbitrate_decision(
        engine_metrics, outcome.arbitration_inference, THRESHOLDS
    )
    assert release == "NO_GO"
    assert outcome.override_applied
    assert SEMANTIC_ASSERT_MISMATCH  # constant exists for breakdown wiring


def test_decision_code_mismatch_optimal_with_blur_code():
    err = validate_decision_code_consistency("Optimal", "ERR_OPTIC_SHRP_001")
    assert err is not None
    ai_result = {
        "decision": "Optimal",
        "code": "ERR_OPTIC_SHRP_001",
        "msg": "inconsistent",
    }
    metrics = {"avg_brightness": 90.0, "sharpness": 60.0}
    outcome = evaluate_semantic_asserts(ai_result, metrics, THRESHOLDS)
    assert any("expects code" in e for e in outcome.semantic_errors)


def test_contract_block_flags_code_mismatch():
    from models.semantic_asserts import contract_block_from_outcome

    outcome = evaluate_semantic_asserts(
        {"decision": "Blurry", "code": "SUCCESS_200", "msg": "x"},
        {"avg_brightness": 80.0, "sharpness": 50.0},
        THRESHOLDS,
    )
    block = contract_block_from_outcome(outcome)
    assert block.get("code_mismatch") is True


def test_confidence_out_of_range_stripped_without_label_override():
    ai_result = {
        "decision": "Blurry",
        "code": "ERR_OPTIC_SHRP_001",
        "msg": "soft",
        "confidence": 2.0,
    }
    metrics = {"avg_brightness": 80.0, "sharpness": 10.0}
    outcome = evaluate_semantic_asserts(ai_result, metrics, THRESHOLDS)
    assert any("confidence" in err for err in outcome.semantic_errors)
    assert outcome.arbitration_inference.get("confidence") is None
    assert outcome.override_applied is False
