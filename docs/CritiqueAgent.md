# Critique agent (rule-based batch review)

**Status:** Shipped (MVP). Non-blocking; not a CI gate.

## Goal

Summarize **high-signal batch rows** for human review and oracle corpus expansion without letting critique output change `GO` / `REVIEW` / `NO_GO`.

Critique **recommends**; `arbitrator`, semantic asserts, and oracle regression **decide**.

## Where it lives

| Piece | Path |
|-------|------|
| Rule engine | `src/eval/critique_agent.py` — `run_critique(batch_report, config)` |
| CLI for existing reports | `scripts/run_critique_agent.py` |
| Automatic generation | `_finalize_batch_report()` in `src/ai_quality_agent.py` |

Each batch run writes a sibling artifact next to the batch report:

```text
results/dev/batch_report_YYYYMMDD_HHMMSS_ffffff.json
results/dev/critique_summary_YYYYMMDD_HHMMSS_ffffff.json
```

The batch runner return dict includes `critique_summary_path`.

## Usage

### Automatic (default)

Any successful `run_batch_test()` / CLI batch profile run emits `critique_summary_*.json` automatically.

### Manual (existing batch report)

```bash
# Latest batch report under results/**
python scripts/run_critique_agent.py --profile dev

# Specific report
python scripts/run_critique_agent.py \
  --profile dev \
  --batch-report results/dev/batch_report_YYYYMMDD_HHMMSS.json

# Custom output path
python scripts/run_critique_agent.py \
  --profile dev \
  --batch-report results/dev/batch_report_YYYYMMDD_HHMMSS.json \
  --output results/dev/my_critique.json
```

### Programmatic

```python
from eval.critique_agent import run_critique

critique = run_critique(batch_report, config)
```

## Input: batch report row fields

Critique reads `batch_report["results"][]` and skips rows with `status == "FAILED"`.

| Source field | Used for |
|--------------|----------|
| `file` | Row identity, oracle case id hint |
| `decision` | Inference dict; `contract_meta` for repair audit |
| `contract` | `semantic_errors`, `code_mismatch`, `invalid_label`, etc. |
| `contract.semantic_errors` | Semantic assert messages |
| `arbitration.semantic_assert_override` | `SEMANTIC_OVERRIDE_REVIEW` |
| `loopback.fallback_used` | Planner fallback signals |
| `loopback.fallback_used_count` | Frequent fallback rule |
| `loopback.stop_reason` | Oscillation / gain-stop rules |
| `inference_output.final_decision` | Row `release_decision` in critique output |
| `inference_output.error_code` | Evidence |

## Output schema (`critique_summary_*.json`)

Top-level object returned by `run_critique()`:

```json
{
  "schema_version": "1.0",
  "batch_id": "20260820_143014_984984",
  "profile": "dev",
  "generated_at": "2026-08-20T06:30:00.000000Z",
  "criteria": {
    "semantic_asserts_enabled": true
  },
  "counts": {
    "rows_total": 100,
    "semantic_error_rows": 4,
    "unstable_repair_rows": 1,
    "fallback_used_rows": 8,
    "no_go_rows": 26
  },
  "rows": [ "... CritiqueRow ..." ],
  "overall_recommendations": [ "... Recommendation ..." ]
}
```

### `counts`

| Field | Meaning |
|-------|---------|
| `rows_total` | Successful rows analyzed (excludes `FAILED`) |
| `semantic_error_rows` | Rows with non-empty `contract.semantic_errors` |
| `unstable_repair_rows` | Rows where `decision.contract_meta.unstable_repair == true` |
| `fallback_used_rows` | Rows where `loopback.fallback_used == true` |
| `no_go_rows` | Rows with `inference_output.final_decision == "NO_GO"` |

### `CritiqueRow` (`rows[]`)

```json
{
  "file": "image4.jpeg",
  "release_decision": "REVIEW",
  "signals": { "... RowSignals ..." },
  "issues": [ "... Issue ..." ],
  "oracle_suggestion": { "... OracleSuggestion ..." }
}
```

### `RowSignals` (`rows[].signals`)

| Field | Type | Source |
|-------|------|--------|
| `semantic_errors` | `string[]` | `contract.semantic_errors` |
| `contract_flags.code_mismatch` | `bool` | `contract.code_mismatch` |
| `contract_flags.invalid_label` | `bool` | `contract.invalid_label` |
| `contract_flags.confidence_violation` | `bool` | `contract.confidence_violation` |
| `contract_flags.inference_error_verdict` | `bool` | `contract.inference_error_verdict` |
| `unstable_repair` | `bool` | `decision.contract_meta.unstable_repair` |
| `repair_attempts` | `int` | `decision.contract_meta.repair_attempts` |
| `strict_fallback_blocked` | `bool` | `decision.contract_meta.strict_fallback_blocked` |
| `fallback_used` | `bool` | `loopback.fallback_used` |
| `fallback_used_count` | `int` | `loopback.fallback_used_count` |
| `loopback_stop_reason` | `string` | `loopback.stop_reason` |
| `release_decision` | `"GO"\|"REVIEW"\|"NO_GO"` | `inference_output.final_decision` |
| `error_code` | `string` | `inference_output.error_code` or `decision.code` |

### `Issue` (`rows[].issues[]`)

```json
{
  "category": "LD",
  "code": "SEMANTIC_ERRORS_WITH_GO",
  "severity": "high",
  "rationale": "Semantic asserts flagged issues but final release remained GO.",
  "evidence": {
    "release_decision": "GO",
    "error_code": "SUCCESS_200",
    "semantic_errors_sample": ["..."],
    "repair_attempts": 0,
    "unstable_repair": false,
    "fallback_used": false,
    "fallback_used_count": 0,
    "loopback_stop_reason": "release_resolved"
  }
}
```

| Field | Values |
|-------|--------|
| `category` | `DQ` (data quality), `LD` (logic drift), `IN` (inference noise) — see `docs/FailureTaxonomy.md` |
| `severity` | `info`, `warn`, `high` |

### `OracleSuggestion` (`rows[].oracle_suggestion`)

```json
{
  "should_append_case": true,
  "mode": "semantic",
  "why": "High-signal semantic or contract drift detected; good oracle regression candidate.",
  "case_id_hint": "hist-from-batch-image4.jpeg"
}
```

| Field | Meaning |
|-------|---------|
| `should_append_case` | `true` when any high-signal issue code matches (see below) |
| `mode` | Always `"semantic"` in MVP (use `append_oracle_case_from_batch.py --mode semantic`) |
| `case_id_hint` | Suggested id prefix; must be made unique before append |
| `why` | Human-readable reason |

**High-signal oracle codes** (`should_append_case == true`):

- `SEMANTIC_ERRORS_WITH_GO`
- `INVALID_LABEL_DETECTED`
- `UNSTABLE_REPAIR_TRIGGERED`
- `SEMANTIC_OVERRIDE_REVIEW`

To append after review:

```bash
PYTHONPATH=src python scripts/append_oracle_case_from_batch.py \
  --batch-report results/dev/batch_report_YYYYMMDD_HHMMSS.json \
  --file image4.jpeg \
  --id hist-018-image4-semantic-go \
  --description "Critique: semantic errors with GO release"
```

### `Recommendation` (`overall_recommendations[]`)

```json
{
  "type": "add_oracle_cases",
  "priority": "high",
  "count_estimate": 3,
  "rationale": "Batch contains semantic or contract drift worth freezing into oracle regression.",
  "based_on": {
    "semantic_error_rows": 4,
    "high_signal_rows": 3
  }
}
```

| Field | Values |
|-------|--------|
| `type` | See batch recommendation table below |
| `priority` | `low`, `medium`, `high` |

---

## Row issue rule table

Rules are evaluated independently; a row may have **multiple** issues.

### A. Semantic / policy drift (`LD`)

| Code | Severity | Condition |
|------|----------|-----------|
| `SEMANTIC_ERRORS_WITH_GO` | high | `len(contract.semantic_errors) > 0` **and** `release_decision == "GO"` |
| `SEMANTIC_OVERRIDE_REVIEW` | warn | `arbitration.semantic_assert_override == true` **and** `release_decision == "REVIEW"` |
| `INVALID_LABEL_DETECTED` | high | `contract.invalid_label == true` |
| `CODE_MISMATCH_DETECTED` | warn | `contract.code_mismatch == true` |
| `STRICT_FALLBACK_BLOCKED` | high | `decision.contract_meta.strict_fallback_blocked == true` |

### B. Repair / parser instability (`IN`)

| Code | Severity | Condition |
|------|----------|-----------|
| `UNSTABLE_REPAIR_TRIGGERED` | high | `decision.contract_meta.unstable_repair == true` |
| `REPAIR_ATTEMPTS_EXCEEDED_ZERO` | info | `decision.contract_meta.repair_attempts > 0` |
| `REVIEW_WITH_HIGH_REPAIR_AND_FALLBACK` | high | `release_decision == "REVIEW"` **and** `repair_attempts > 0` **and** `loopback.fallback_used == true` |

### C. Loopback / planner (`IN` / `DQ`)

| Code | Severity | Category | Condition |
|------|----------|----------|-----------|
| `PLANNER_FALLBACK_USED` | warn | IN | `loopback.fallback_used == true` |
| `PLANNER_FALLBACK_FREQUENT` | high | IN | `loopback.fallback_used_count >= 2` |
| `LOOPBACK_OSCILLATION_STOP` | warn | DQ | `loopback.stop_reason == "oscillation_detected"` |
| `LOOPBACK_GAIN_STOP` | info | DQ | `"insufficient_" in loopback.stop_reason` |

### D. Outcome patterns (`DQ` / `IN`)

| Code | Severity | Category | Condition |
|------|----------|----------|-----------|
| `NO_GO_WITHOUT_SEMANTIC_ERRORS` | info | DQ | `release_decision == "NO_GO"` **and** empty `semantic_errors` |

---

## Batch recommendation rule table

Batch-level recommendations aggregate row issues and `counts`.

| Type | Priority | Emit when |
|------|----------|-----------|
| `add_oracle_cases` | high | `semantic_error_rows >= 1` **or** any `INVALID_LABEL_DETECTED` **or** any `CODE_MISMATCH_DETECTED` |
| `review_contract_policy` | high | `unstable_repair_rows >= 1` **or** any `SEMANTIC_ERRORS_WITH_GO` **or** any `STRICT_FALLBACK_BLOCKED` |
| `investigate_planner` | medium / high | `fallback_used_rows / rows_total >= 0.1` **or** `PLANNER_FALLBACK_FREQUENT >= 3` **or** `LOOPBACK_OSCILLATION_STOP >= 3` (priority **high** when fallback ratio `>= 0.2`) |
| `inspect_data_quality` | medium | `NO_GO_WITHOUT_SEMANTIC_ERRORS + LOOPBACK_GAIN_STOP` issue count `>= 3` |
| `monitor_noise_only` | low | No other recommendation matched **but** at least one row has issues |

`count_estimate` semantics:

| Type | `count_estimate` source |
|------|-------------------------|
| `add_oracle_cases` | `min(high_signal_rows, 10)` |
| `review_contract_policy` | `max(unstable_repair_rows, SEMANTIC_ERRORS_WITH_GO count, STRICT_FALLBACK_BLOCKED count)` |
| `investigate_planner` | `fallback_used_rows` |
| `inspect_data_quality` | DQ signal row count |
| `monitor_noise_only` | Total issue count across rows |

---

## Design principles

1. **Critique recommends; oracles decide.** No critique field mutates batch release semantics.
2. **Rule-first MVP.** No LLM calls; output is reproducible from the batch JSON alone.
3. **Failure taxonomy aligned.** Issue `category` maps to DQ / LD / IN in `docs/FailureTaxonomy.md`.
4. **Oracle expansion path.** High-signal rows link to `scripts/append_oracle_case_from_batch.py`.

## Related tests

- `tests/test_critique_agent.py` — issue codes, batch recommendations
- `tests/test_loopback_integration.py` — automatic `critique_summary_path` on batch run

## Future (not shipped)

- Optional LLM narrative layer on top of rule output (cost-capped, off by default)
- PR comment / GitHub summary artifact from critique JSON
- Auto-draft oracle JSONL lines (still requiring human review before append)
