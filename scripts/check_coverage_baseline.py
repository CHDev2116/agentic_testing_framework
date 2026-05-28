from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def _parse_coverage_xml(path: Path) -> float:
    xml = path.read_text(encoding="utf-8")
    match = re.search(r'line-rate="([0-9]*\.?[0-9]+)"', xml)
    if not match:
        raise ValueError(f"Cannot parse line-rate from {path}")
    return float(match.group(1)) * 100.0


def _parse_baseline(path: Path) -> float:
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"Coverage baseline file is empty: {path}")
    return float(content)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail when current coverage drops below baseline."
    )
    parser.add_argument(
        "--coverage-xml",
        default="coverage.xml",
        help="Path to pytest coverage XML report.",
    )
    parser.add_argument(
        "--baseline-file",
        default=".ci/coverage_baseline.txt",
        help="Path to baseline percentage file.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Floating-point tolerance in coverage percentage points.",
    )
    args = parser.parse_args()

    coverage_xml = Path(args.coverage_xml)
    baseline_file = Path(args.baseline_file)
    current = _parse_coverage_xml(coverage_xml)
    baseline = _parse_baseline(baseline_file)

    if current + args.tolerance < baseline:
        print(
            f"[coverage-baseline] FAIL: current={current:.2f}% "
            f"is below baseline={baseline:.2f}%"
        )
        return 1

    print(
        f"[coverage-baseline] PASS: current={current:.2f}% "
        f"baseline={baseline:.2f}%"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
