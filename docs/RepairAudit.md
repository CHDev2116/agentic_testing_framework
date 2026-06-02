# JSON repair audit trail (`repair_audit`)

When `max_json_repair_attempts > 0`, each LLM round is recorded on `contract_meta` for batch audit and CI stability gates.

## `contract_meta` fields

| Field | Type | Meaning |
|-------|------|---------|
| `repair_attempts` | int | Extra LLM calls after validation failure |
| `repair_audit` | list | Per-round audit entries (see below) |
| `unstable_repair` | bool | True if any round has `UNSTABLE_REPAIR` |

## `repair_audit[]` entry

| Key | Meaning |
|-----|---------|
| `round` | 0-based fetch index (0 = initial prompt) |
| `format_errors` | Validator messages for this round |
| `error_class` | `parse` or `validation` |
| `prompt_input_snapshot` | Truncated prompt sent (max 1500 chars) |
| `raw_output_snapshot` | Truncated model raw text |
| `parsed_decision` / `parsed_code` | Values extracted from parsed JSON (even if invalid) |
| `semantic_drift_from_previous` | e.g. `Under-exposed -> Optimal` |
| `stability` | `STABLE` or `UNSTABLE_REPAIR` |

## Semantic stability rule

Across repair rounds, if **both** parsed decisions are non-`Error` and differ → `UNSTABLE_REPAIR` (likely “please the validator” drift).

`Error -> *` is **not** flagged (format recovery).

## CI

`tests/test_repair_stability_gate.py` — anchors the gate.

Batch KPI: `summary.quality_kpis.unstable_repair_count`.

## Release policy (`unstable_repair_release`)

Config (`model_settings.inference.contract`):

| Value | Effect when `unstable_repair: true` |
|-------|-------------------------------------|
| `REVIEW` (default) | Force `REVIEW` + `DecisionConflict.UNSTABLE_JSON_REPAIR` |
| `NO_GO` | Force `NO_GO` |
| `OFF` | Audit only; do not change release |

Applied in batch finalize and oracle regression (`run_oracle_case`).

## Triage

See `docs/FailureTaxonomy.md` — unstable repair rows are usually **IN** (inference noise), not **LD** (oracle policy).
