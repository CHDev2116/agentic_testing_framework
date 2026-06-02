"""Repair audit trail: per-round prompts, raw output, format errors, semantic drift."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from models.contract_validator import contract_error_class, truncate_repair_snippet

UNSTABLE_REPAIR_TAG = "UNSTABLE_REPAIR"


def extract_parsed_semantic(parsed: Any) -> Tuple[Optional[str], Optional[str]]:
    if not isinstance(parsed, dict):
        return None, None
    decision = parsed.get("decision")
    code = parsed.get("code")
    d = str(decision).strip() if decision is not None and str(decision).strip() else None
    c = str(code).strip() if code is not None and str(code).strip() else None
    return d, c


def detect_semantic_repair_drift(
    before_decision: Optional[str],
    after_decision: Optional[str],
) -> Optional[str]:
    """
    Label decision change across repair rounds that is not format-recovery from Error.

    Returns e.g. 'Under-exposed -> Optimal' when unstable; None when stable or allowed.
    """
    if not before_decision or not after_decision:
        return None
    if before_decision == after_decision:
        return None
    if before_decision == "Error" or after_decision == "Error":
        return None
    return f"{before_decision} -> {after_decision}"


def build_repair_audit_entry(
    *,
    round_index: int,
    format_errors: List[str],
    prompt_input_snapshot: str,
    raw_output_snapshot: str,
    parsed: Dict[str, Any],
    previous_decision: Optional[str],
) -> Dict[str, Any]:
    """Build one repair_audit row (validator-layer hook surface)."""
    error_class = contract_error_class(parsed, format_errors)
    decision, code = extract_parsed_semantic(parsed)
    drift = detect_semantic_repair_drift(previous_decision, decision) if round_index > 0 else None
    entry: Dict[str, Any] = {
        "round": round_index,
        "format_errors": list(format_errors),
        "error_class": error_class,
        "prompt_input_snapshot": truncate_repair_snippet(prompt_input_snapshot, max_chars=1500),
        "raw_output_snapshot": truncate_repair_snippet(raw_output_snapshot, max_chars=1500),
        "parsed_decision": decision,
        "parsed_code": code,
    }
    if drift:
        entry["semantic_drift_from_previous"] = drift
        entry["stability"] = UNSTABLE_REPAIR_TAG
    else:
        entry["stability"] = "STABLE"
    return entry


def audit_has_unstable_drift(repair_audit: List[Dict[str, Any]]) -> bool:
    return any(
        entry.get("stability") == UNSTABLE_REPAIR_TAG
        or entry.get("semantic_drift_from_previous")
        for entry in repair_audit
    )


def summarize_repair_audit(repair_audit: List[Dict[str, Any]]) -> Dict[str, Any]:
    drifts = [
        str(entry["semantic_drift_from_previous"])
        for entry in repair_audit
        if entry.get("semantic_drift_from_previous")
    ]
    return {
        "rounds": len(repair_audit),
        "unstable_repair": audit_has_unstable_drift(repair_audit),
        "semantic_drifts": drifts,
    }
