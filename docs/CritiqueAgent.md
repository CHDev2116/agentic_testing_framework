# Critique Agent (roadmap outline)

**Status:** Not a CI gate. This document captures the intended P3 capability.

## Goal

Score **assertion strength** and **schema coverage** for agent-generated tests—without letting the LLM become the sole pass/fail oracle.

## Inputs

- Batch `batch_report` rows (`decision`, `contract`, `arbitration`, `loopback.attempts`)
- Optional generated test files under `tests/`
- Historical oracle corpus (`tests/regression/oracle_cases.jsonl`)

## Outputs

- `critique_summary` on batch report: weak asserts, missing edge cases, suggested new oracle rows
- Non-blocking PR comment artifact (future)

## Principles

1. Critique **recommends**; `arbitrator` + oracle regression **decide**.
2. Never replace `vision_math` thresholds or semantic policy releases.
3. Cost-capped LLM calls similar to `eval_settings.llm_judge`.

## MVP steps (future)

1. Rule-based critic (no LLM): flag rows with semantic errors but `GO` release.
2. Suggest oracle JSONL lines from flagged rows (wrap `append_oracle_case_from_batch.py`).
3. Optional LLM narrative for human review only.
