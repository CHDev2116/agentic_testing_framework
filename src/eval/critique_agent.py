"""Rule-based critique summary for batch reports.

MVP goals:
- do not alter release semantics
- summarize high-signal rows for human review
- suggest oracle case expansion for semantic/contract drift
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Dict, List


HIGH_SIGNAL_ORACLE_CODES = {
    "SEMANTIC_ERRORS_WITH_GO",
    "INVALID_LABEL_DETECTED",
    "UNSTABLE_REPAIR_TRIGGERED",
    "SEMANTIC_OVERRIDE_REVIEW",
}


def _semantic_errors(row: Dict[str, Any]) -> List[str]:
    contract = row.get("contract")
    if isinstance(contract, dict):
        errors = contract.get("semantic_errors")
        if isinstance(errors, list):
            return [str(item) for item in errors]
    return []


def _contract_meta(row: Dict[str, Any]) -> Dict[str, Any]:
    decision = row.get("decision")
    if isinstance(decision, dict):
        meta = decision.get("contract_meta")
        if isinstance(meta, dict):
            return meta
    return {}


def _signals_for_row(row: Dict[str, Any]) -> Dict[str, Any]:
    contract = row.get("contract") if isinstance(row.get("contract"), dict) else {}
    contract = contract if isinstance(contract, dict) else {}
    meta = _contract_meta(row)
    loopback = row.get("loopback") if isinstance(row.get("loopback"), dict) else {}
    inference_output = (
        row.get("inference_output") if isinstance(row.get("inference_output"), dict) else {}
    )

    return {
        "semantic_errors": _semantic_errors(row),
        "contract_flags": {
            "code_mismatch": bool(contract.get("code_mismatch")),
            "invalid_label": bool(contract.get("invalid_label")),
            "confidence_violation": bool(contract.get("confidence_violation")),
            "inference_error_verdict": bool(contract.get("inference_error_verdict")),
        },
        "unstable_repair": bool(meta.get("unstable_repair")),
        "repair_attempts": int(meta.get("repair_attempts", 0) or 0),
        "strict_fallback_blocked": bool(meta.get("strict_fallback_blocked")),
        "fallback_used": bool(loopback.get("fallback_used")),
        "fallback_used_count": int(loopback.get("fallback_used_count", 0) or 0),
        "loopback_stop_reason": str(loopback.get("stop_reason", "")),
        "release_decision": str(inference_output.get("final_decision", "NO_GO")).upper(),
        "error_code": str(inference_output.get("error_code", row.get("decision", {}).get("code", "")))
        if isinstance(row.get("decision"), dict)
        else str(inference_output.get("error_code", "")),
    }


def _issue(
    *,
    category: str,
    code: str,
    severity: str,
    rationale: str,
    signals: Dict[str, Any],
    extra_evidence: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    evidence = {
        "release_decision": signals["release_decision"],
        "error_code": signals["error_code"],
        "semantic_errors_sample": signals["semantic_errors"][:3],
        "repair_attempts": signals["repair_attempts"],
        "unstable_repair": signals["unstable_repair"],
        "fallback_used": signals["fallback_used"],
        "fallback_used_count": signals["fallback_used_count"],
        "loopback_stop_reason": signals["loopback_stop_reason"],
    }
    if extra_evidence:
        evidence.update(extra_evidence)
    return {
        "category": category,
        "code": code,
        "severity": severity,
        "rationale": rationale,
        "evidence": evidence,
    }


def _issues_for_row(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    signals = _signals_for_row(row)
    flags = signals["contract_flags"]
    release = signals["release_decision"]
    semantic_errors = signals["semantic_errors"]
    issues: List[Dict[str, Any]] = []

    if semantic_errors and release == "GO":
        issues.append(
            _issue(
                category="LD",
                code="SEMANTIC_ERRORS_WITH_GO",
                severity="high",
                rationale="Semantic asserts flagged issues but final release remained GO.",
                signals=signals,
            )
        )
    if bool(row.get("arbitration", {}).get("semantic_assert_override")) and release == "REVIEW":
        issues.append(
            _issue(
                category="LD",
                code="SEMANTIC_OVERRIDE_REVIEW",
                severity="warn",
                rationale="Semantic assert override pushed row into REVIEW.",
                signals=signals,
            )
        )
    if flags["invalid_label"]:
        issues.append(
            _issue(
                category="LD",
                code="INVALID_LABEL_DETECTED",
                severity="high",
                rationale="Model emitted a label outside the accepted controlled vocabulary.",
                signals=signals,
            )
        )
    if flags["code_mismatch"]:
        issues.append(
            _issue(
                category="LD",
                code="CODE_MISMATCH_DETECTED",
                severity="warn",
                rationale="Decision/code pairing drifted from semantic expectations.",
                signals=signals,
            )
        )
    if signals["unstable_repair"]:
        issues.append(
            _issue(
                category="IN",
                code="UNSTABLE_REPAIR_TRIGGERED",
                severity="high",
                rationale="Repair rounds drifted semantically across attempts.",
                signals=signals,
            )
        )
    if signals["repair_attempts"] > 0:
        issues.append(
            _issue(
                category="IN",
                code="REPAIR_ATTEMPTS_EXCEEDED_ZERO",
                severity="info",
                rationale="Inference contract required at least one repair attempt.",
                signals=signals,
            )
        )
    if signals["strict_fallback_blocked"]:
        issues.append(
            _issue(
                category="LD",
                code="STRICT_FALLBACK_BLOCKED",
                severity="high",
                rationale="Strict contract mode blocked fallback after repair failure.",
                signals=signals,
            )
        )
    if signals["fallback_used"]:
        issues.append(
            _issue(
                category="IN",
                code="PLANNER_FALLBACK_USED",
                severity="warn",
                rationale="Loopback planner fell back during at least one step.",
                signals=signals,
            )
        )
    if signals["fallback_used_count"] >= 2:
        issues.append(
            _issue(
                category="IN",
                code="PLANNER_FALLBACK_FREQUENT",
                severity="high",
                rationale="Loopback planner fallback happened repeatedly for one row.",
                signals=signals,
            )
        )
    if signals["loopback_stop_reason"] == "oscillation_detected":
        issues.append(
            _issue(
                category="DQ",
                code="LOOPBACK_OSCILLATION_STOP",
                severity="warn",
                rationale="Loopback oscillated between signals and stopped.",
                signals=signals,
            )
        )
    if "insufficient_" in signals["loopback_stop_reason"]:
        issues.append(
            _issue(
                category="DQ",
                code="LOOPBACK_GAIN_STOP",
                severity="info",
                rationale="Loopback stopped because physical improvement was insufficient.",
                signals=signals,
            )
        )
    if release == "NO_GO" and not semantic_errors:
        issues.append(
            _issue(
                category="DQ",
                code="NO_GO_WITHOUT_SEMANTIC_ERRORS",
                severity="info",
                rationale="NO_GO appears driven by physical quality or model outcome rather than semantic drift.",
                signals=signals,
            )
        )
    if release == "REVIEW" and signals["repair_attempts"] > 0 and signals["fallback_used"]:
        issues.append(
            _issue(
                category="IN",
                code="REVIEW_WITH_HIGH_REPAIR_AND_FALLBACK",
                severity="high",
                rationale="Row required repair and planner fallback before ending in REVIEW.",
                signals=signals,
            )
        )
    return issues


def _oracle_suggestion(file_name: str, issues: List[Dict[str, Any]]) -> Dict[str, Any]:
    issue_codes = {str(item["code"]) for item in issues}
    should_append = any(code in HIGH_SIGNAL_ORACLE_CODES for code in issue_codes)
    if should_append:
        why = "High-signal semantic or contract drift detected; good oracle regression candidate."
    else:
        why = "No high-signal semantic/contract drift requiring oracle expansion."
    return {
        "should_append_case": should_append,
        "mode": "semantic",
        "why": why,
        "case_id_hint": f"hist-from-batch-{file_name}",
    }


def _overall_recommendations(row_summaries: List[Dict[str, Any]], counts: Dict[str, int]) -> List[Dict[str, Any]]:
    rows_total = max(1, counts["rows_total"])
    recommendations: List[Dict[str, Any]] = []

    def issue_count(code: str) -> int:
        return sum(1 for row in row_summaries for issue in row["issues"] if issue["code"] == code)

    if counts["semantic_error_rows"] >= 1 or issue_count("INVALID_LABEL_DETECTED") >= 1 or issue_count("CODE_MISMATCH_DETECTED") >= 1:
        high_signal_rows = sum(
            1
            for row in row_summaries
            if any(issue["code"] in HIGH_SIGNAL_ORACLE_CODES for issue in row["issues"])
        )
        recommendations.append(
            {
                "type": "add_oracle_cases",
                "priority": "high",
                "count_estimate": min(high_signal_rows, 10),
                "rationale": "Batch contains semantic or contract drift worth freezing into oracle regression.",
                "based_on": {
                    "semantic_error_rows": counts["semantic_error_rows"],
                    "high_signal_rows": high_signal_rows,
                },
            }
        )

    if counts["unstable_repair_rows"] >= 1 or issue_count("SEMANTIC_ERRORS_WITH_GO") >= 1 or issue_count("STRICT_FALLBACK_BLOCKED") >= 1:
        recommendations.append(
            {
                "type": "review_contract_policy",
                "priority": "high",
                "count_estimate": max(
                    counts["unstable_repair_rows"],
                    issue_count("SEMANTIC_ERRORS_WITH_GO"),
                    issue_count("STRICT_FALLBACK_BLOCKED"),
                ),
                "rationale": "Contract repair or semantic policy drift needs human review.",
                "based_on": {
                    "unstable_repair_rows": counts["unstable_repair_rows"],
                    "semantic_error_rows": counts["semantic_error_rows"],
                },
            }
        )

    if (counts["fallback_used_rows"] / rows_total) >= 0.1 or issue_count("PLANNER_FALLBACK_FREQUENT") >= 3 or issue_count("LOOPBACK_OSCILLATION_STOP") >= 3:
        recommendations.append(
            {
                "type": "investigate_planner",
                "priority": "high" if (counts["fallback_used_rows"] / rows_total) >= 0.2 else "medium",
                "count_estimate": counts["fallback_used_rows"],
                "rationale": "Planner fallback/oscillation rate suggests loopback instability worth investigation.",
                "based_on": {
                    "fallback_used_rows": counts["fallback_used_rows"],
                    "planner_fallback_frequent": issue_count("PLANNER_FALLBACK_FREQUENT"),
                    "oscillation_rows": issue_count("LOOPBACK_OSCILLATION_STOP"),
                },
            }
        )

    dq_rows = issue_count("NO_GO_WITHOUT_SEMANTIC_ERRORS") + issue_count("LOOPBACK_GAIN_STOP")
    if dq_rows >= 3:
        recommendations.append(
            {
                "type": "inspect_data_quality",
                "priority": "medium",
                "count_estimate": dq_rows,
                "rationale": "A meaningful share of failures look like asset quality or unrecoverable physical issues.",
                "based_on": {
                    "no_go_rows": counts["no_go_rows"],
                    "dq_signal_rows": dq_rows,
                },
            }
        )

    if not recommendations and any(row["issues"] for row in row_summaries):
        recommendations.append(
            {
                "type": "monitor_noise_only",
                "priority": "low",
                "count_estimate": sum(len(row["issues"]) for row in row_summaries),
                "rationale": "Only low-signal noise indicators were found; monitor before changing policy.",
                "based_on": {
                    "rows_total": counts["rows_total"],
                },
            }
        )

    return recommendations


def run_critique(batch_report: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    results = batch_report.get("results", [])
    rows: List[Dict[str, Any]] = []

    for row in results:
        if not isinstance(row, dict) or row.get("status") == "FAILED":
            continue
        file_name = str(row.get("file", "unknown"))
        issues = _issues_for_row(row)
        signals = _signals_for_row(row)
        rows.append(
            {
                "file": file_name,
                "release_decision": signals["release_decision"],
                "signals": signals,
                "issues": issues,
                "oracle_suggestion": _oracle_suggestion(file_name, issues),
            }
        )

    counts = {
        "rows_total": len(rows),
        "semantic_error_rows": sum(1 for row in rows if row["signals"]["semantic_errors"]),
        "unstable_repair_rows": sum(1 for row in rows if row["signals"]["unstable_repair"]),
        "fallback_used_rows": sum(1 for row in rows if row["signals"]["fallback_used"]),
        "no_go_rows": sum(1 for row in rows if row["release_decision"] == "NO_GO"),
    }

    return {
        "schema_version": "1.0",
        "batch_id": str(batch_report.get("batch_id", "unknown")),
        "profile": str(batch_report.get("profile", "unknown")),
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "criteria": {
            "semantic_asserts_enabled": bool(
                config.get("eval_settings", {}).get("semantic_asserts_enabled", True)
            )
        },
        "counts": counts,
        "rows": rows,
        "overall_recommendations": _overall_recommendations(rows, counts),
    }

