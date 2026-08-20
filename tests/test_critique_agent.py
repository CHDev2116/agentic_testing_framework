from eval.critique_agent import run_critique


def _base_config():
    return {"eval_settings": {"semantic_asserts_enabled": True}}


def _base_row(file_name: str):
    return {
        "file": file_name,
        "status": "SUCCESS",
        "decision": {
            "decision": "Optimal",
            "code": "SUCCESS_200",
            "msg": "ok",
            "backend": "simulated",
        },
        "contract": {"semantic_errors": []},
        "loopback": {
            "fallback_used": False,
            "fallback_used_count": 0,
            "stop_reason": "release_resolved",
        },
        "inference_output": {
            "final_decision": "GO",
            "error_code": "SUCCESS_200",
            "error_message": "ok",
            "steps": [],
        },
        "arbitration": {
            "release_decision": "GO",
            "semantic_assert_override": False,
        },
    }


def test_semantic_errors_with_go_generates_high_signal_issue():
    row = _base_row("a.jpg")
    row["contract"]["semantic_errors"] = ["decision/code mismatch"]

    out = run_critique(
        {"batch_id": "b1", "profile": "dev", "results": [row]},
        _base_config(),
    )

    issues = out["rows"][0]["issues"]
    assert any(item["code"] == "SEMANTIC_ERRORS_WITH_GO" for item in issues)
    assert out["rows"][0]["oracle_suggestion"]["should_append_case"] is True


def test_unstable_repair_generates_contract_review_signal():
    row = _base_row("b.jpg")
    row["decision"]["contract_meta"] = {"repair_attempts": 2, "unstable_repair": True}
    row["inference_output"]["final_decision"] = "REVIEW"

    out = run_critique(
        {"batch_id": "b2", "profile": "dev", "results": [row]},
        _base_config(),
    )

    issues = out["rows"][0]["issues"]
    assert any(item["code"] == "UNSTABLE_REPAIR_TRIGGERED" for item in issues)
    assert any(rec["type"] == "review_contract_policy" for rec in out["overall_recommendations"])


def test_planner_fallback_generates_planner_issue():
    row = _base_row("c.jpg")
    row["loopback"]["fallback_used"] = True
    row["loopback"]["fallback_used_count"] = 2
    row["loopback"]["stop_reason"] = "oscillation_detected"
    row["inference_output"]["final_decision"] = "REVIEW"

    out = run_critique(
        {"batch_id": "b3", "profile": "dev", "results": [row]},
        _base_config(),
    )

    codes = {item["code"] for item in out["rows"][0]["issues"]}
    assert "PLANNER_FALLBACK_USED" in codes
    assert "PLANNER_FALLBACK_FREQUENT" in codes
    assert "LOOPBACK_OSCILLATION_STOP" in codes


def test_batch_level_recommendations_aggregate_signals():
    row1 = _base_row("d1.jpg")
    row1["contract"]["semantic_errors"] = ["bad label"]
    row1["contract"]["invalid_label"] = True

    row2 = _base_row("d2.jpg")
    row2["decision"]["contract_meta"] = {"repair_attempts": 1, "unstable_repair": True}
    row2["inference_output"]["final_decision"] = "REVIEW"

    row3 = _base_row("d3.jpg")
    row3["loopback"]["fallback_used"] = True
    row3["loopback"]["fallback_used_count"] = 2
    row3["loopback"]["stop_reason"] = "oscillation_detected"
    row3["inference_output"]["final_decision"] = "NO_GO"
    row3["contract"]["semantic_errors"] = []

    out = run_critique(
        {"batch_id": "b4", "profile": "dev", "results": [row1, row2, row3]},
        _base_config(),
    )

    rec_types = {item["type"] for item in out["overall_recommendations"]}
    assert "add_oracle_cases" in rec_types
    assert "review_contract_policy" in rec_types
    assert "investigate_planner" in rec_types

