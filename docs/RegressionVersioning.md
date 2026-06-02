# Versioned golden datasets (oracle semantics)

## Problem

`tests/test_oracle_regression.py` compares **current code** to **expected fields inside `oracle_cases.jsonl`**. That answers: “Does implementation match today’s spec?”

It does **not** answer: “What changed versus **last week’s rules**?” when you only touch `semantic_asserts.py`.

CI red/green cannot distinguish:

- Intentional policy change (LD)
- Accidental regression (LD bug)
- Stale jsonl expectation (fixture edit)

## Approach: two layers of goldens

| Layer | File | Locks |
|-------|------|--------|
| **Case spec** | `tests/regression/oracle_cases.jsonl` | Inputs + `expected_release` / `expected_conflict` (human-reviewed contract) |
| **Semantics snapshot** | `tests/regression/snapshots/oracle_semantics_v*.json` | **Outputs of current code** on those inputs at a tagged commit |

Snapshots version **rule behavior**, not images. They enable “parallel version validation” without running two binaries: you compare **snapshot vN** vs **working tree**.

True side-by-side **two code revisions** locally:

```bash
git stash
git checkout <old-commit>
python scripts/refresh_oracle_snapshot.py --out /tmp/oracle_old.json --label old_rules
git checkout -
python scripts/refresh_oracle_snapshot.py --out /tmp/oracle_new.json --label new_rules
# diff the two JSON files or use diff_snapshots in a one-liner
```

## Daily workflow

### Before changing semantic_asserts / arbitrator

```bash
PYTHONPATH=src pytest tests/test_oracle_regression.py -q   # current gate
python scripts/diff_oracle_semantics.py                     # should be clean
```

### After changing rules

```bash
python scripts/diff_oracle_semantics.py --report /tmp/semantic_changelog.md
```

Example output when one case flips:

```markdown
- **hist-009-...: release changed | release 'GO' → 'NO_GO'**
  - Taxonomy hint: `LD`
```

### Accept intentional drift

1. Update `oracle_cases.jsonl` `expected_*` where the new behavior is correct.
2. Refresh snapshot:

```bash
python scripts/refresh_oracle_snapshot.py
# or bump file: --out tests/regression/snapshots/oracle_semantics_v2.json
```

3. PR must include **both** jsonl and snapshot diff, plus a short LD note in the description.

## CI integration

| Step | Behavior |
|------|----------|
| `Oracle historical regression` | `pytest` on jsonl (`expected_*` gate) |
| `Oracle semantic changelog` | Always runs (`if: always()`); appends markdown to **GitHub job summary** |
| PR only | `CI_ORACLE_SNAPSHOT_ENFORCE=1` — fails if snapshot drift without refresh |

Script: `scripts/ci_oracle_semantic_summary.sh`

Push to `main` still gets the summary; snapshot `--enforce` is **PR-only** so intentional snapshot bumps on main are not double-blocked.

## Limits

- Snapshots reflect **deterministic** oracle runner (no live LLM in hist cases).
- Adding cases changes `added_cases` in the report until baseline is refreshed.
- Snapshots do **not** replace jsonl: jsonl is the reviewed spec; snapshot is the **diff radar** for rule changes.

See also `docs/FailureTaxonomy.md`.
