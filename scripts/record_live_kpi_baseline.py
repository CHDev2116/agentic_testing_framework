#!/usr/bin/env python3
"""Record observed quality_kpis from a live batch into .ci/live_quality_kpi_baseline.json."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

KPI_FLOAT_KEYS = (
    "fallback_ratio",
    "review_rate",
)
KPI_INT_KEYS = (
    "review_count",
    "inference_fallback_count",
    "loopback_fallback_count",
    "json_repair_attempts_total",
    "json_repair_exhausted_count",
    "semantic_assert_fail_count",
    "semantic_code_mismatch_count",
    "strict_contract_violation_count",
    "llm_judge_calls",
    "llm_judge_overrides",
)


def _latest_glob(pattern: str) -> Optional[Path]:
    matches = sorted(Path(".").glob(pattern), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_observed(report: Dict[str, Any]) -> Dict[str, Any]:
    summary = report.get("summary", {})
    kpis = dict(summary.get("quality_kpis") or {})
    observed: Dict[str, Any] = {}
    for key in KPI_FLOAT_KEYS:
        if key in kpis:
            observed[key] = float(kpis[key])
    for key in KPI_INT_KEYS:
        if key in kpis:
            observed[key] = int(kpis[key])
    observed["pass_rate"] = float(summary.get("pass_rate", 0.0))
    observed["total_tests"] = int(summary.get("total_tests", len(report.get("results", []))))
    observed["release_decision"] = summary.get("release_decision")
    return observed


def _propose_thresholds(observed: Dict[str, Any]) -> Dict[str, Any]:
    total = max(int(observed.get("total_tests", 1)), 1)

    def _ceil_ratio(value: float, floor: float = 0.05) -> float:
        return round(min(1.0, max(floor, value * 1.5 + 0.02)), 4)

    def _ceil_count(value: int, pad: int = 2) -> int:
        return max(0, int(value) + pad)

    exhausted = int(observed.get("json_repair_exhausted_count", 0))
    exhausted_ratio = exhausted / total

    return {
        "profile": "live_baseline",
        "description": "Auto-proposed from recorded baseline; review before CI promotion.",
        "max_fallback_ratio": _ceil_ratio(float(observed.get("fallback_ratio", 0.0))),
        "max_review_rate": _ceil_ratio(float(observed.get("review_rate", 0.0)), floor=0.1),
        "max_pass_rate_variance": 0.15,
        "max_image_score_variance": 0.15,
        "max_json_repair_exhausted_count": _ceil_count(exhausted, pad=3),
        "max_json_repair_exhausted_ratio": round(
            min(0.5, max(0.05, exhausted_ratio * 1.5 + 0.02)), 4
        ),
        "max_semantic_assert_fail_count": _ceil_count(
            int(observed.get("semantic_assert_fail_count", 0)), pad=5
        ),
        "max_semantic_code_mismatch_count": _ceil_count(
            int(observed.get("semantic_code_mismatch_count", 0)), pad=3
        ),
        "max_strict_contract_violation_count": _ceil_count(
            int(observed.get("strict_contract_violation_count", 0)), pad=2
        ),
        "max_llm_judge_calls": _ceil_count(int(observed.get("llm_judge_calls", 0)), pad=3),
        "max_llm_judge_overrides": _ceil_count(
            int(observed.get("llm_judge_overrides", 0)), pad=3
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Record live batch KPI baseline.")
    parser.add_argument(
        "--batch-report",
        default=None,
        help="Batch JSON path (default: latest results/live_baseline/batch_report_*.json).",
    )
    parser.add_argument(
        "--baseline-out",
        default=".ci/live_quality_kpi_baseline.json",
        help="Where to write observed KPI snapshot.",
    )
    parser.add_argument(
        "--thresholds-out",
        default=".ci/live_quality_kpi_thresholds.json",
        help="Threshold file to update when --propose-thresholds is set.",
    )
    parser.add_argument(
        "--inference-backend",
        default=None,
        help="Optional label stored in baseline metadata.",
    )
    parser.add_argument(
        "--propose-thresholds",
        action="store_true",
        help="Rewrite thresholds file using observed values + headroom.",
    )
    args = parser.parse_args()

    batch_path = (
        Path(args.batch_report)
        if args.batch_report
        else _latest_glob("results/live_baseline/batch_report_*.json")
    )
    if batch_path is None or not batch_path.exists():
        print("[live-kpi] ERROR: no batch report found (run run_live_kpi_baseline.sh first)")
        return 1

    report = _load_json(batch_path)
    observed = _extract_observed(report)
    baseline_doc = {
        "profile": "live_baseline",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "batch_report": str(batch_path),
        "inference_backend": args.inference_backend or report.get("profile"),
        "observed": observed,
        "summary_release": report.get("summary", {}).get("release_decision"),
    }

    baseline_out = Path(args.baseline_out)
    baseline_out.write_text(json.dumps(baseline_doc, indent=2) + "\n", encoding="utf-8")
    print(f"[live-kpi] wrote baseline: {baseline_out}")
    for key, value in sorted(observed.items()):
        print(f"[live-kpi]   {key}={value}")

    if args.propose_thresholds:
        proposed = _propose_thresholds(observed)
        thresholds_out = Path(args.thresholds_out)
        thresholds_out.write_text(json.dumps(proposed, indent=2) + "\n", encoding="utf-8")
        print(f"[live-kpi] proposed thresholds: {thresholds_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
