from eval.arbitrator import DecisionConflict
from eval.oracle_regression import run_oracle_case
from models.contract_release_policy import (
    ContractReleaseSettings,
    apply_unstable_repair_release_policy,
)


def test_unstable_repair_release_review():
    release, conflict = apply_unstable_repair_release_policy(
        "GO",
        DecisionConflict.ALL_PASS,
        {"unstable_repair": True},
        ContractReleaseSettings(unstable_repair_release="REVIEW"),
    )
    assert release == "REVIEW"
    assert conflict == DecisionConflict.UNSTABLE_JSON_REPAIR


def test_unstable_repair_release_off():
    release, conflict = apply_unstable_repair_release_policy(
        "GO",
        DecisionConflict.ALL_PASS,
        {"unstable_repair": True},
        ContractReleaseSettings(unstable_repair_release="OFF"),
    )
    assert release == "GO"
    assert conflict == DecisionConflict.ALL_PASS


def test_oracle_case_applies_unstable_repair_from_ai_result():
    case = {
        "id": "test-unstable-meta",
        "mode": "semantic",
        "engine_metrics": {"avg_brightness": 90.0, "sharpness": 60.0},
        "metrics": {"avg_brightness": 90.0, "sharpness": 60.0},
        "ai_result": {
            "decision": "Optimal",
            "code": "SUCCESS_200",
            "msg": "ok",
            "contract_meta": {
                "repair_attempts": 1,
                "unstable_repair": True,
                "repair_audit": [],
            },
        },
        "expected_release": "REVIEW",
        "expected_conflict": DecisionConflict.UNSTABLE_JSON_REPAIR.value,
        "expect_unstable_repair": True,
    }
    actual = run_oracle_case(case)
    assert actual["release"] == "REVIEW"
    assert actual["unstable_repair"] is True
