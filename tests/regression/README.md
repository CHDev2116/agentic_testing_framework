# Historical Oracle Regression

Frozen cases in `oracle_cases.jsonl` lock **release decisions** and **arbitration conflict labels**
after inference. Any change to:

- `src/eval/arbitrator.py` thresholds or logic
- `src/models/semantic_asserts.py`
- Default thresholds in batch config

must either keep these cases green or **update the JSONL with an explicit review** (intentional behavior change).

## Run locally

```bash
PYTHONPATH=src pytest tests/test_oracle_regression.py -q
```

Or validate only:

```bash
PYTHONPATH=src python -c "
from eval.oracle_regression import run_regression_suite
from pathlib import Path
errs = run_regression_suite(Path('tests/regression/oracle_cases.jsonl'))
raise SystemExit(1 if errs else 0) if errs else print('OK', len(errs))
"
```

## Add a case from a batch report

**Automated (recommended):**

```bash
PYTHONPATH=src python scripts/append_oracle_case_from_batch.py \
  --batch-report results/dev/batch_report_YYYYMMDD_HHMMSS.json \
  --file stress_image3_001.jpg \
  --id hist-016-my-incident \
  --description "Production REVIEW from 2026-06-01 run"
```

Use `--dry-run` to preview the JSON line without appending.

**Manual:**

1. Copy `metrics`, `decision` (full inference dict), and observed `arbitration.release_decision` from `batch_report_*.json`.
2. Append one JSON line with a unique `id`, `description`, and `expected_release` / `expected_conflict`.
3. Use `"mode": "semantic"` for production path (default) or `"mode": "arbitrator"` to test raw oracle only.

Optional fields:

- `expect_semantic_errors`: true
- `expect_override_applied`: true/false
- `expect_unstable_repair`: true when `ai_result.contract_meta.unstable_repair`
- `semantic_policy`: per-case override (else use `--profile` / `base.json`)
- `contract_policy`: e.g. `{"unstable_repair_release": "REVIEW"}`
- `thresholds`: override per case

CI runs this suite on every push/PR (see `.github/workflows/ci.yml`).

Frozen **`hist-017-unstable-repair-review`**: healthy Optimal + `contract_meta.unstable_repair` → `REVIEW` / `UNSTABLE_JSON_REPAIR` (default `unstable_repair_release`).

## Versioned semantics snapshots (rule drift radar)

Frozen **outputs** (release, conflict, semantic_errors) live under `snapshots/oracle_semantics_v*.json`.
They complement jsonl `expected_*`: use them when changing `semantic_asserts` / `arbitrator`.

```bash
# After rule changes — semantic changelog vs committed baseline
python scripts/diff_oracle_semantics.py

# Intentional policy change — refresh baseline after updating jsonl
python scripts/refresh_oracle_snapshot.py
```

See `docs/RegressionVersioning.md` and `docs/FailureTaxonomy.md` (DQ / LD / IN).
