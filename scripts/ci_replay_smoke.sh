#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TRACE_FILE=".ci/replay_smoke_trace.jsonl"
FIXTURE_IMAGE="tests/fixtures/replay_smoke/underexposed.png"

if [[ ! -f "$TRACE_FILE" ]]; then
  echo "[replay-smoke] trace missing; generating fixture..."
  python3 scripts/generate_replay_smoke_fixture.py
fi

if [[ ! -f "$FIXTURE_IMAGE" ]]; then
  echo "[replay-smoke] FAIL: fixture image missing at $FIXTURE_IMAGE" >&2
  exit 1
fi

echo "[replay-smoke] running replay mode (simulated)"
PYTHONPATH=src python3 src/ai_quality_agent.py \
  --config configs/replay_smoke.json \
  --inference-backend simulated \
  --loopback-planner simulated \
  --replay-mode replay \
  --replay-file "$TRACE_FILE"

LATEST_REPORT="$(ls -t results/replay_smoke/batch_report_*.json 2>/dev/null | head -n 1 || true)"
if [[ -z "$LATEST_REPORT" ]]; then
  echo "[replay-smoke] FAIL: no batch report produced" >&2
  exit 1
fi

export LATEST_REPORT
PYTHONPATH=src python3 - <<'PY'
import json
import os
import sys
from pathlib import Path

report_path = Path(os.environ["LATEST_REPORT"])
report = json.loads(report_path.read_text(encoding="utf-8"))
failed = [row for row in report.get("results", []) if row.get("status") == "FAILED"]
if failed:
    print(f"[replay-smoke] FAIL: {len(failed)} failed row(s) in {report_path}", file=sys.stderr)
    for row in failed:
        print(row, file=sys.stderr)
    sys.exit(1)
print(f"[replay-smoke] PASS ({report_path})")
PY

echo "[replay-smoke] KPI gate (strict semantic + contract)"
python3 scripts/check_quality_kpis.py --enforce \
  --batch-report "$LATEST_REPORT" \
  --thresholds-file .ci/replay_quality_kpi_thresholds.json
