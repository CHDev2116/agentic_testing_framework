#!/usr/bin/env python3
"""Validate batch / repeatability KPI thresholds for CI gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _latest_glob(pattern: str) -> Optional[Path]:
    matches = sorted(Path(".").glob(pattern), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_threshold_defaults(path: Optional[Path]) -> Dict[str, Any]:
    defaults = {
        "max_fallback_ratio": 0.05,
        "max_review_rate": 1.0,
        "max_pass_rate_variance": 0.05,
        "max_image_score_variance": 0.05,
        "max_json_repair_exhausted_count": None,
        "max_json_repair_exhausted_ratio": None,
        "max_semantic_assert_fail_count": None,
        "max_semantic_code_mismatch_count": None,
        "max_strict_contract_violation_count": None,
        "max_llm_judge_calls": None,
        "max_llm_judge_overrides": None,
        "max_unstable_repair_count": None,
    }
    if path is None or not path.exists():
        return defaults
    file_values = _load_json(path)
    defaults.update({k: file_values.get(k, defaults[k]) for k in defaults})
    return defaults


def _check_batch_kpis(
    report: Dict[str, Any],
    *,
    max_fallback_ratio: float,
    max_review_rate: float,
    max_json_repair_exhausted_count: Optional[int],
    max_json_repair_exhausted_ratio: Optional[float],
    max_semantic_assert_fail_count: Optional[int],
    max_semantic_code_mismatch_count: Optional[int],
    max_strict_contract_violation_count: Optional[int],
    max_llm_judge_calls: Optional[int],
    max_llm_judge_overrides: Optional[int],
    max_unstable_repair_count: Optional[int],
) -> List[str]:
    errors: List[str] = []
    summary = report.get("summary", {})
    kpis = summary.get("quality_kpis", {})
    if not kpis:
        errors.append("summary.quality_kpis missing (run a recent batch first)")
        return errors

    total_tests = int(summary.get("total_tests", 0)) or len(report.get("results", []))
    fallback_ratio = float(kpis.get("fallback_ratio", 0.0))
    review_rate = float(kpis.get("review_rate", 0.0))
    if fallback_ratio > max_fallback_ratio:
        errors.append(
            f"fallback_ratio={fallback_ratio:.4f} exceeds max={max_fallback_ratio:.4f}"
        )
    if review_rate > max_review_rate:
        errors.append(
            f"review_rate={review_rate:.4f} exceeds max={max_review_rate:.4f}"
        )

    exhausted_count = int(kpis.get("json_repair_exhausted_count", 0))
    if max_json_repair_exhausted_count is not None and exhausted_count > max_json_repair_exhausted_count:
        errors.append(
            f"json_repair_exhausted_count={exhausted_count} exceeds "
            f"max={max_json_repair_exhausted_count}"
        )
    if max_json_repair_exhausted_ratio is not None and total_tests > 0:
        exhausted_ratio = exhausted_count / total_tests
        if exhausted_ratio > max_json_repair_exhausted_ratio:
            errors.append(
                f"json_repair_exhausted_ratio={exhausted_ratio:.4f} exceeds "
                f"max={max_json_repair_exhausted_ratio:.4f}"
            )

    semantic_fails = int(kpis.get("semantic_assert_fail_count", 0))
    if (
        max_semantic_assert_fail_count is not None
        and semantic_fails > max_semantic_assert_fail_count
    ):
        errors.append(
            f"semantic_assert_fail_count={semantic_fails} exceeds "
            f"max={max_semantic_assert_fail_count}"
        )

    code_mismatches = int(kpis.get("semantic_code_mismatch_count", 0))
    if (
        max_semantic_code_mismatch_count is not None
        and code_mismatches > max_semantic_code_mismatch_count
    ):
        errors.append(
            f"semantic_code_mismatch_count={code_mismatches} exceeds "
            f"max={max_semantic_code_mismatch_count}"
        )

    strict_violations = int(kpis.get("strict_contract_violation_count", 0))
    if (
        max_strict_contract_violation_count is not None
        and strict_violations > max_strict_contract_violation_count
    ):
        errors.append(
            f"strict_contract_violation_count={strict_violations} exceeds "
            f"max={max_strict_contract_violation_count}"
        )

    judge_calls = int(kpis.get("llm_judge_calls", 0))
    if max_llm_judge_calls is not None and judge_calls > max_llm_judge_calls:
        errors.append(
            f"llm_judge_calls={judge_calls} exceeds max={max_llm_judge_calls}"
        )

    judge_overrides = int(kpis.get("llm_judge_overrides", 0))
    if (
        max_llm_judge_overrides is not None
        and judge_overrides > max_llm_judge_overrides
    ):
        errors.append(
            f"llm_judge_overrides={judge_overrides} exceeds max={max_llm_judge_overrides}"
        )

    unstable_repairs = int(kpis.get("unstable_repair_count", 0))
    if (
        max_unstable_repair_count is not None
        and unstable_repairs > max_unstable_repair_count
    ):
        errors.append(
            f"unstable_repair_count={unstable_repairs} exceeds "
            f"max={max_unstable_repair_count}"
        )

    return errors


def _check_repeatability_kpis(
    report: Dict[str, Any],
    *,
    max_pass_rate_variance: float,
    max_image_score_variance: float,
) -> List[str]:
    errors: List[str] = []
    variance = report.get("variance", {})
    pass_rate_var = float(variance.get("pass_rate_variance", 0.0))
    max_score_var = float(variance.get("max_image_score_variance", 0.0))
    if pass_rate_var > max_pass_rate_variance:
        errors.append(
            f"pass_rate_variance={pass_rate_var:.6f} exceeds max={max_pass_rate_variance:.6f}"
        )
    if max_score_var > max_image_score_variance:
        errors.append(
            f"max_image_score_variance={max_score_var:.6f} exceeds max={max_image_score_variance:.6f}"
        )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Check automation KPI thresholds.")
    parser.add_argument(
        "--thresholds-file",
        default=".ci/quality_kpi_thresholds.json",
        help="JSON file with default max thresholds (CLI flags override).",
    )
    parser.add_argument(
        "--batch-report",
        default=None,
        help="Path to batch report JSON (default: latest results/**/batch_report_*.json).",
    )
    parser.add_argument(
        "--repeatability-report",
        default=None,
        help="Path to repeatability JSON (default: latest results/repeatability/repeatability_*.json).",
    )
    parser.add_argument("--max-fallback-ratio", type=float, default=None)
    parser.add_argument("--max-review-rate", type=float, default=None)
    parser.add_argument("--max-pass-rate-variance", type=float, default=None)
    parser.add_argument("--max-image-score-variance", type=float, default=None)
    parser.add_argument("--max-json-repair-exhausted-count", type=int, default=None)
    parser.add_argument("--max-json-repair-exhausted-ratio", type=float, default=None)
    parser.add_argument("--max-semantic-assert-fail-count", type=int, default=None)
    parser.add_argument("--max-semantic-code-mismatch-count", type=int, default=None)
    parser.add_argument("--max-strict-contract-violation-count", type=int, default=None)
    parser.add_argument("--max-llm-judge-calls", type=int, default=None)
    parser.add_argument("--max-llm-judge-overrides", type=int, default=None)
    parser.add_argument("--max-unstable-repair-count", type=int, default=None)
    parser.add_argument(
        "--enforce",
        action="store_true",
        help="Exit non-zero when thresholds are exceeded.",
    )
    parser.add_argument(
        "--warn-only",
        action="store_true",
        help="Print warnings but always exit 0.",
    )
    args = parser.parse_args()

    thresholds_path = Path(args.thresholds_file) if args.thresholds_file else None
    defaults = _load_threshold_defaults(thresholds_path)

    def _resolve(name: str, arg_value: Any) -> Any:
        return defaults[name] if arg_value is None else arg_value

    failures: List[str] = []

    batch_path = Path(args.batch_report) if args.batch_report else _latest_glob(
        "results/**/batch_report_*.json"
    )
    if batch_path and batch_path.exists():
        failures.extend(
            _check_batch_kpis(
                _load_json(batch_path),
                max_fallback_ratio=float(_resolve("max_fallback_ratio", args.max_fallback_ratio)),
                max_review_rate=float(_resolve("max_review_rate", args.max_review_rate)),
                max_json_repair_exhausted_count=_resolve(
                    "max_json_repair_exhausted_count", args.max_json_repair_exhausted_count
                ),
                max_json_repair_exhausted_ratio=_resolve(
                    "max_json_repair_exhausted_ratio", args.max_json_repair_exhausted_ratio
                ),
                max_semantic_assert_fail_count=_resolve(
                    "max_semantic_assert_fail_count", args.max_semantic_assert_fail_count
                ),
                max_semantic_code_mismatch_count=_resolve(
                    "max_semantic_code_mismatch_count", args.max_semantic_code_mismatch_count
                ),
                max_strict_contract_violation_count=_resolve(
                    "max_strict_contract_violation_count",
                    args.max_strict_contract_violation_count,
                ),
                max_llm_judge_calls=_resolve(
                    "max_llm_judge_calls", args.max_llm_judge_calls
                ),
                max_llm_judge_overrides=_resolve(
                    "max_llm_judge_overrides", args.max_llm_judge_overrides
                ),
                max_unstable_repair_count=_resolve(
                    "max_unstable_repair_count", args.max_unstable_repair_count
                ),
            )
        )
        print(f"[quality-kpis] batch report: {batch_path}")
    else:
        print("[quality-kpis] skip batch KPIs (no batch report found)")

    rep_path = (
        Path(args.repeatability_report)
        if args.repeatability_report
        else _latest_glob("results/repeatability/repeatability_*.json")
    )
    if rep_path and rep_path.exists():
        failures.extend(
            _check_repeatability_kpis(
                _load_json(rep_path),
                max_pass_rate_variance=float(
                    _resolve("max_pass_rate_variance", args.max_pass_rate_variance)
                ),
                max_image_score_variance=float(
                    _resolve("max_image_score_variance", args.max_image_score_variance)
                ),
            )
        )
        print(f"[quality-kpis] repeatability report: {rep_path}")
    else:
        print("[quality-kpis] skip repeatability KPIs (no repeatability report found)")

    if failures:
        for item in failures:
            label = "WARN" if args.warn_only or not args.enforce else "FAIL"
            print(f"[quality-kpis] {label}: {item}")
        if args.enforce and not args.warn_only:
            return 1
        return 0

    print("[quality-kpis] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
