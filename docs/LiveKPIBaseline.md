# Live KPI baseline (`live_baseline` profile)

Records **observed** `summary.quality_kpis` from a real inference backend (llama.cpp or Ollama), separate from CI’s simulated/replay gates.

## Prerequisites

| Backend | CLI override | Server |
|---------|--------------|--------|
| **llama.cpp** (default) | `INFERENCE_BACKEND=llama_cpp` | HTTP on `127.0.0.1:8080` (see `docs/ModelInferenceSetup.md`) |
| **Ollama vision** | `INFERENCE_BACKEND=ollama_vision` | `ollama serve` + `llava:7b` (or edit `configs/live_baseline.json`) |

LLM judge uses Ollama text (`llama3.2` by default). If Ollama is down, judge falls back to **simulated** verdicts (logged).

## One-command run

```bash
bash scripts/run_live_kpi_baseline.sh
```

Optional env:

| Variable | Default | Meaning |
|----------|---------|---------|
| `INFERENCE_BACKEND` | `llama_cpp` | `ollama_vision` for vision-only Ollama path |
| `LIVE_ASYNC` | `0` | Set `1` to add `--async-batch` |
| `LIVE_ASYNC_CONCURRENCY` | `2` | Async worker count when `LIVE_ASYNC=1` |
| `PROPOSE_THRESHOLDS` | `0` | Set `1` to rewrite `.ci/live_quality_kpi_thresholds.json` from observed + headroom |

Examples:

```bash
INFERENCE_BACKEND=ollama_vision bash scripts/run_live_kpi_baseline.sh
LIVE_ASYNC=1 LIVE_ASYNC_CONCURRENCY=3 bash scripts/run_live_kpi_baseline.sh
PROPOSE_THRESHOLDS=1 bash scripts/run_live_kpi_baseline.sh
```

## Artifacts

| File | Role |
|------|------|
| `results/live_baseline/batch_report_*.json` | Full batch output |
| `.ci/live_quality_kpi_baseline.json` | Last observed KPI snapshot |
| `.ci/live_quality_kpi_thresholds.json` | Loose ceilings for local `--warn-only` check |

## Manual steps

```bash
PYTHONPATH=src python src/ai_quality_agent.py --profile live_baseline --inference-backend llama_cpp
python scripts/record_live_kpi_baseline.py --propose-thresholds
python scripts/check_quality_kpis.py \
  --thresholds-file .ci/live_quality_kpi_thresholds.json \
  --batch-report results/live_baseline/batch_report_*.json \
  --warn-only
```

## Profile highlights (`configs/live_baseline.json`)

- Inference contract repair: `max_json_repair_attempts: 2`
- `runtime.adaptive_backoff.enabled: true`
- `eval_settings.llm_judge.enabled: true` (Ollama, cost-capped)
- **Not** used in GitHub Actions CI

After a stable live run, review `.ci/live_quality_kpi_baseline.json` and optionally commit tightened thresholds.
