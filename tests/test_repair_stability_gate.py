"""Semantic Stability Gate — flag UNSTABLE_REPAIR when JSON repair flips decisions."""

from __future__ import annotations

import json


from models.contract_repair import ContractRepairSettings, run_contract_inference_loop
from models.repair_audit import UNSTABLE_REPAIR_TAG

VALID_OPTIMAL = json.dumps(
    {"decision": "Optimal", "code": "SUCCESS_200", "msg": "Quality ok."}
)


def _run_repair_sequence(responses: list[str]) -> dict:
    settings = ContractRepairSettings(max_json_repair_attempts=2)
    idx = {"n": 0}

    def fetch(_prompt: str) -> str:
        raw = responses[idx["n"]]
        idx["n"] += 1
        return raw

    return run_contract_inference_loop(
        settings,
        backend="test",
        default_msg="unparsable",
        build_initial_prompt=lambda: "vision prompt",
        fetch_model_text=fetch,
    )


def test_semantic_stability_gate_stable_format_only_repair():
    """Truncated JSON then valid Optimal — no decision drift between parsed rounds."""
    result = _run_repair_sequence(['```json\n{"decision":', VALID_OPTIMAL])
    audit = result["contract_meta"]["repair_audit"]
    assert result["contract_meta"]["unstable_repair"] is False
    assert not any(e.get("stability") == UNSTABLE_REPAIR_TAG for e in audit)


def test_semantic_stability_gate_unstable_under_exposed_to_optimal():
    """Valid partial Under-exposed (missing msg) then Optimal — UNSTABLE_REPAIR."""
    partial_under = json.dumps(
        {"decision": "Under-exposed", "code": "ERR_LIGHT_DARK_002"}
    )
    result = _run_repair_sequence([partial_under, VALID_OPTIMAL])
    meta = result["contract_meta"]
    assert meta["unstable_repair"] is True
    assert result["decision"] == "Optimal"
    drifts = [
        e["semantic_drift_from_previous"]
        for e in meta["repair_audit"]
        if e.get("semantic_drift_from_previous")
    ]
    assert "Under-exposed -> Optimal" in drifts


def test_semantic_stability_gate_ci_regression_expectation():
    """
    Documented gate: any production row with unstable_repair should be triaged as IN, not LD.

    This test is the CI anchor for the stability policy.
    """
    result = _run_repair_sequence(
        [
            json.dumps({"decision": "Blurry", "code": "ERR_OPTIC_SHRP_001"}),
            VALID_OPTIMAL,
        ]
    )
    assert result["contract_meta"]["unstable_repair"] is True
    unstable = [
        e
        for e in result["contract_meta"]["repair_audit"]
        if e.get("stability") == UNSTABLE_REPAIR_TAG
    ]
    assert len(unstable) >= 1
