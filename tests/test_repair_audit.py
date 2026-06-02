from models.contract_validator import semantic_repair_drift_label
from models.repair_audit import (
    UNSTABLE_REPAIR_TAG,
    audit_has_unstable_drift,
    build_repair_audit_entry,
    detect_semantic_repair_drift,
)


def test_detect_semantic_repair_drift_under_to_optimal():
    assert detect_semantic_repair_drift("Under-exposed", "Optimal") == "Under-exposed -> Optimal"


def test_detect_semantic_repair_drift_allows_error_recovery():
    assert detect_semantic_repair_drift("Error", "Optimal") is None
    assert detect_semantic_repair_drift("Under-exposed", "Error") is None


def test_build_repair_audit_entry_marks_unstable():
    entry = build_repair_audit_entry(
        round_index=1,
        format_errors=["missing required key: msg"],
        prompt_input_snapshot="repair prompt",
        raw_output_snapshot='{"decision": "Optimal", "code": "SUCCESS_200", "msg": "ok"}',
        parsed={"decision": "Optimal", "code": "SUCCESS_200", "msg": "ok"},
        previous_decision="Under-exposed",
    )
    assert entry["stability"] == UNSTABLE_REPAIR_TAG
    assert entry["semantic_drift_from_previous"] == "Under-exposed -> Optimal"


def test_validator_hook_semantic_repair_drift_label():
    assert semantic_repair_drift_label("Blurry", "Optimal") == "Blurry -> Optimal"


def test_audit_has_unstable_drift():
    audit = [
        {"stability": "STABLE"},
        {"stability": UNSTABLE_REPAIR_TAG, "semantic_drift_from_previous": "Blurry -> Optimal"},
    ]
    assert audit_has_unstable_drift(audit)
