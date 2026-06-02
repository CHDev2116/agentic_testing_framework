#!/usr/bin/env python3
"""Compare current oracle rule outputs against a versioned semantics snapshot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eval.oracle_semantic_diff import diff_against_baseline_file  # noqa: E402

DEFAULT_CASES = ROOT / "tests" / "regression" / "oracle_cases.jsonl"
DEFAULT_BASELINE = ROOT / "tests" / "regression" / "snapshots" / "oracle_semantics_v1.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Emit semantic change summary (release/conflict drift) vs golden snapshot."
    )
    parser.add_argument(
        "--baseline",
        default=str(DEFAULT_BASELINE),
        help="Versioned snapshot JSON (previous rule version outcomes)",
    )
    parser.add_argument(
        "--cases",
        default=str(DEFAULT_CASES),
        help="oracle_cases.jsonl (inputs; same cases as baseline unless corpus grew)",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Optional path to write markdown report",
    )
    parser.add_argument(
        "--enforce",
        action="store_true",
        help="Exit 1 when any semantic drift vs baseline",
    )
    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    if not baseline_path.exists():
        print(
            f"[oracle-diff] ERROR: baseline missing: {baseline_path}\n"
            "Run: python scripts/refresh_oracle_snapshot.py",
            file=sys.stderr,
        )
        return 1

    report = diff_against_baseline_file(baseline_path, cases_path=args.cases)
    markdown = report.to_markdown()
    print(markdown)

    if args.report:
        Path(args.report).write_text(markdown, encoding="utf-8")
        print(f"[oracle-diff] wrote {args.report}")

    if report.has_changes:
        print(
            f"[oracle-diff] SUMMARY: {len(report.changed)} case(s) changed, "
            f"{len(report.added_cases)} added, {len(report.removed_cases)} removed"
        )
        if args.enforce:
            return 1
        return 0

    print("[oracle-diff] SUMMARY: no semantic drift vs baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
