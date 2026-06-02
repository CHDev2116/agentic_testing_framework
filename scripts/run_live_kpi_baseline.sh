#!/usr/bin/env bash
# Run a live-inference batch on profile live_baseline, record KPIs, warn-check thresholds.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH=src

BACKEND="${INFERENCE_BACKEND:-llama_cpp}"
ASYNC_FLAGS=()
if [[ "${LIVE_ASYNC:-0}" == "1" ]]; then
  ASYNC_FLAGS=(--async-batch --async-concurrency "${LIVE_ASYNC_CONCURRENCY:-2}")
fi

echo "[live-baseline] profile=live_baseline backend=${BACKEND} async=${LIVE_ASYNC:-0}"
PYTHONPATH=src python src/ai_quality_agent.py \
  --profile live_baseline \
  --inference-backend "${BACKEND}" \
  "${ASYNC_FLAGS[@]}" \
  "$@"

LATEST="$(ls -t results/live_baseline/batch_report_*.json 2>/dev/null | head -1 || true)"
if [[ -z "${LATEST}" ]]; then
  echo "[live-baseline] ERROR: no batch report under results/live_baseline/"
  exit 1
fi

python scripts/record_live_kpi_baseline.py \
  --batch-report "${LATEST}" \
  --inference-backend "${BACKEND}"

if [[ "${PROPOSE_THRESHOLDS:-0}" == "1" ]]; then
  python scripts/record_live_kpi_baseline.py \
    --batch-report "${LATEST}" \
    --inference-backend "${BACKEND}" \
    --propose-thresholds
fi

python scripts/check_quality_kpis.py \
  --thresholds-file .ci/live_quality_kpi_thresholds.json \
  --batch-report "${LATEST}" \
  --warn-only

echo "[live-baseline] done report=${LATEST}"
