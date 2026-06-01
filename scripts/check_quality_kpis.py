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


def _check_batch_kpis(
  report: Dict[str, Any],
  *,
  max_fallback_ratio: float,
  max_review_rate: float,
) -> List[str]:
    errors: List[str] = []
    summary = report.get("summary", {})
    kpis = summary.get("quality_kpis", {})
    if not kpis:
        errors.append("summary.quality_kpis missing (run a recent batch first)")
        return errors

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
        "--batch-report",
        default=None,
        help="Path to batch report JSON (default: latest results/**/batch_report_*.json).",
    )
    parser.add_argument(
        "--repeatability-report",
        default=None,
        help="Path to repeatability JSON (default: latest results/repeatability/repeatability_*.json).",
    )
    parser.add_argument("--max-fallback-ratio", type=float, default=0.05)
    parser.add_argument("--max-review-rate", type=float, default=1.0)
    parser.add_argument("--max-pass-rate-variance", type=float, default=0.05)
    parser.add_argument("--max-image-score-variance", type=float, default=0.05)
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

    failures: List[str] = []

    batch_path = Path(args.batch_report) if args.batch_report else _latest_glob(
        "results/**/batch_report_*.json"
    )
    if batch_path and batch_path.exists():
        failures.extend(
            _check_batch_kpis(
                _load_json(batch_path),
                max_fallback_ratio=args.max_fallback_ratio,
                max_review_rate=args.max_review_rate,
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
                max_pass_rate_variance=args.max_pass_rate_variance,
                max_image_score_variance=args.max_image_score_variance,
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
