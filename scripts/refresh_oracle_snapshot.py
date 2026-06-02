#!/usr/bin/env python3
"""Write a versioned oracle semantics snapshot from current code + oracle_cases.jsonl."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eval.oracle_semantic_diff import (  # noqa: E402
    build_snapshot_document,
    collect_suite_outcomes,
)

DEFAULT_CASES = ROOT / "tests" / "regression" / "oracle_cases.jsonl"
DEFAULT_OUT = ROOT / "tests" / "regression" / "snapshots" / "oracle_semantics_v1.json"


def _git_head() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=ROOT,
                text=True,
            )
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh oracle semantics snapshot.")
    parser.add_argument(
        "--cases",
        default=str(DEFAULT_CASES),
        help="oracle_cases.jsonl path",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help="Snapshot JSON output path",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Snapshot label (default: oracle_semantics_<git short>)",
    )
    args = parser.parse_args()

    cases_path = Path(args.cases)
    out_path = Path(args.out)
    label = args.label or f"oracle_semantics_{_git_head()}"

    outcomes = collect_suite_outcomes(cases_path)
    try:
        cases_rel = str(cases_path.relative_to(ROOT))
    except ValueError:
        cases_rel = str(cases_path)
    doc = build_snapshot_document(
        outcomes,
        label=label,
        cases_path=cases_rel,
        oracle_rules_tag=_git_head(),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[oracle-snapshot] wrote {out_path} ({doc['case_count']} cases, label={label})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
