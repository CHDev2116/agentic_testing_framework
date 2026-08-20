#!/usr/bin/env python3
"""Generate critique_summary_*.json from a batch_report_*.json."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_quality_agent import load_config  # noqa: E402
from eval.critique_agent import run_critique  # noqa: E402


def _latest_glob(pattern: str) -> Optional[Path]:
    matches = sorted(Path(".").glob(pattern), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate critique summary from batch report.")
    parser.add_argument(
        "--batch-report",
        default=None,
        help="Path to batch_report_*.json (default: latest results/**/batch_report_*.json).",
    )
    parser.add_argument(
        "--profile",
        default="base",
        help="Config profile used to resolve semantic settings (default: base).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path. Default: beside batch report as critique_summary_*.json.",
    )
    args = parser.parse_args()

    batch_path = Path(args.batch_report) if args.batch_report else _latest_glob(
        "results/**/batch_report_*.json"
    )
    if batch_path is None or not batch_path.exists():
        print("[critique-agent] ERROR: batch report not found", file=sys.stderr)
        return 1

    config, _source = load_config(profile=args.profile)
    report = _load_json(batch_path)
    critique = run_critique(report, config)

    if args.output:
        output_path = Path(args.output)
    else:
        stem = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_path = batch_path.parent / f"critique_summary_{stem}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(critique, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[critique-agent] wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

