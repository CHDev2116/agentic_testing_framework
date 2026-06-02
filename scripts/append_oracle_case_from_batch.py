#!/usr/bin/env python3
"""
Append a historical oracle regression case from a batch_report row.

Computes expected_release / expected_conflict via the same semantic+arbitrator
path used in production, then appends one JSON line to oracle_cases.jsonl.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eval.oracle_regression import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    load_oracle_cases,
    run_oracle_case,
)
from models.contract_release_policy import contract_meta_from_ai_result  # noqa: E402

DEFAULT_CORPUS = ROOT / "tests" / "regression" / "oracle_cases.jsonl"


def _load_report(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_row(report: Dict[str, Any], file_name: str) -> Dict[str, Any]:
    for row in report.get("results", []):
        if row.get("file") == file_name:
            return row
    raise ValueError(f"No result row with file={file_name!r}")


def _build_case_from_row(
    row: Dict[str, Any],
    *,
    case_id: str,
    description: str,
    thresholds: Dict[str, Any],
    mode: str,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    metrics = row.get("metrics")
    decision = row.get("decision")
    if not isinstance(metrics, dict) or not isinstance(decision, dict):
        raise ValueError("Row must have metrics dict and decision dict")

    engine_metrics = {
        "avg_brightness": metrics.get("avg_brightness", metrics.get("brightness", 0.0)),
        "sharpness": metrics.get("sharpness", 0.0),
    }
    case: Dict[str, Any] = {
        "id": case_id,
        "description": description,
        "mode": mode,
        "engine_metrics": engine_metrics,
        "thresholds": thresholds,
    }
    if mode == "arbitrator":
        case["model_inference"] = {
            "decision": decision.get("decision"),
            "status": decision.get("decision"),
            "confidence": decision.get("confidence"),
        }
    else:
        case["metrics"] = dict(metrics)
        case["ai_result"] = dict(decision)

    contract_meta = contract_meta_from_ai_result(case.get("ai_result", {}))
    if contract_meta.get("unstable_repair"):
        case["expect_unstable_repair"] = True

    actual = run_oracle_case(case, config=config)
    case["expected_release"] = actual["release"]
    case["expected_conflict"] = actual.get("conflict")
    if actual.get("semantic_errors"):
        case["expect_semantic_errors"] = True
    if actual.get("override_applied"):
        case["expect_override_applied"] = bool(actual["override_applied"])
    if actual.get("unstable_repair"):
        case["expect_unstable_repair"] = True
    return case


def _existing_ids(corpus_path: Path) -> set[str]:
    if not corpus_path.exists():
        return set()
    return {str(c["id"]) for c in load_oracle_cases(corpus_path)}


def append_case(corpus_path: Path, case: Dict[str, Any], *, dry_run: bool) -> None:
    if case["id"] in _existing_ids(corpus_path):
        raise ValueError(f"Case id {case['id']!r} already exists in {corpus_path}")

    line = json.dumps(case, ensure_ascii=False)
    if dry_run:
        print(line)
        return
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    with corpus_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(f"[append-oracle] appended {case['id']} -> {corpus_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Append oracle regression case from batch row.")
    parser.add_argument("--batch-report", required=True, help="Path to batch_report_*.json")
    parser.add_argument("--file", required=True, help="Row file name (e.g. photo.jpg)")
    parser.add_argument("--id", required=True, help="Unique case id (e.g. hist-013-...)")
    parser.add_argument("--description", default="", help="Human-readable case description")
    parser.add_argument(
        "--corpus",
        default=str(DEFAULT_CORPUS),
        help="Target oracle_cases.jsonl path",
    )
    parser.add_argument(
        "--mode",
        choices=("semantic", "arbitrator"),
        default="semantic",
        help="Regression execution mode",
    )
    parser.add_argument(
        "--thresholds-json",
        default=None,
        help="Optional JSON file with min_brightness / max_brightness / min_sharpness",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print JSON line only")
    parser.add_argument(
        "--profile",
        default="base",
        help="Config profile for semantic_policy / contract_policy (default: base)",
    )
    args = parser.parse_args()

    from ai_quality_agent import load_config  # noqa: E402

    config, _source = load_config(profile=args.profile)
    thresholds = dict(DEFAULT_THRESHOLDS)
    if args.thresholds_json:
        thresholds.update(json.loads(Path(args.thresholds_json).read_text(encoding="utf-8")))

    report = _load_report(Path(args.batch_report))
    row = _find_row(report, args.file)
    description = args.description or (
        f"Imported from {Path(args.batch_report).name} row {args.file}"
    )
    case = _build_case_from_row(
        row,
        case_id=args.id,
        description=description,
        thresholds=thresholds,
        mode=args.mode,
        config=config,
    )
    append_case(Path(args.corpus), case, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
