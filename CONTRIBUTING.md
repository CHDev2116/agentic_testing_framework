# Contributing

Thanks for helping improve this project. Small, focused changes are easier to review and merge.

## Prerequisites

- Python **3.9+** (CI runs on **3.11**; matching CI locally avoids surprises).
- A clone of the repository.

## Dependencies (single source of truth)

**All pinned runtime dependencies are defined in `pyproject.toml` under `[project.dependencies]`.** Do not maintain a second copy of version pins elsewhere.

- **Developers / CI**: `pip install -e ".[dev]"` (includes pytest, coverage, Ruff, and mypy).
- **Optional**: `pip install -r requirements.txt` — this file only contains `-e .[dev]` as a convenience shim for older habits or docs that still use `-r`.

When you add or bump a dependency, edit **`pyproject.toml` only**, then reinstall your venv.

## Local setup

```bash
cd agentic_testing_framework
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -U pip
pip install -e ".[dev]"
```

The CLI and tests expect `src` on the module path. Use `PYTHONPATH=src` as shown below (same as CI).

## Streamlit demo (`app.py`)

Run from the **repository root** so `agent`, `engine`, etc. resolve:

```bash
PYTHONPATH=src streamlit run app.py
```

Without `PYTHONPATH=src`, **Manual Baseline** still works; **AI Pipeline** needs the orchestrator import path above.

## Run tests

```bash
PYTHONPATH=src pytest
```

Coverage options are defined in `pyproject.toml` (`--cov=src`, XML report, and **`--cov-fail-under=34`** so total coverage cannot drift far below current levels without CI failing).

## Lint

CI runs Ruff on the full Python tree under `src` plus `tests`. Match it before opening a PR:

```bash
ruff check src tests app.py test_connection.py
```

## Type check (mypy)

Settings live in `pyproject.toml` under `[tool.mypy]`. Run the same checks as CI (two passes avoid duplicate module mapping for `src/` vs repo-root scripts):

```bash
mypy --explicit-package-bases src
MYPYPATH=src mypy --explicit-package-bases app.py test_connection.py
```

## Optional: agent smoke run (CI parity)

The workflow also runs a short end-to-end report generation:

```bash
PYTHONPATH=src python src/ai_quality_agent.py --profile dev --performance-analysis --overhead-analysis
```

## Pull requests

1. **Branch**: Open PRs against the repository default branch (usually `main`).
2. **Scope**: One logical change per PR when possible (feature, fix, or docs—not all mixed unless tightly related).
3. **Description**: Summarize *what* changed and *why*; link an issue if one exists.
4. **Green CI**: Ensure tests, Ruff, and **mypy** pass locally.
5. **Docs**: If you change CLI flags, config shape, or inference behavior, update `README.md` and any affected file under `docs/`.

## Fast pre-push check (recommended)

Use one command to run the same quality gates CI enforces for PRs:

```bash
bash scripts/dev_prepush_check.sh
```

This keeps PR feedback fast and avoids common red-X cycles caused by lint/type/test drift.

## Coverage baseline gate (ratchet-lite)

PR CI enforces two coverage constraints:

- `--cov-fail-under=34` from `pyproject.toml` (absolute floor)
- `.ci/coverage_baseline.txt` (no-regression gate for current baseline)

Local check:

```bash
PYTHONPATH=src pytest
python scripts/check_coverage_baseline.py --coverage-xml coverage.xml --baseline-file .ci/coverage_baseline.txt
```

When you intentionally improve and stabilize coverage, raise `.ci/coverage_baseline.txt` in the same PR.

## Code style

Follow existing patterns in nearby modules (logging, typing, error messages). Prefer clear names and small functions over clever one-liners.

Use **`logging.getLogger(__name__)`** instead of `print` for diagnostics. The batch CLI calls **`util.cli_logging.configure_cli_logging()`** in `__main__`, which sets `basicConfig` to include **timestamp**, **level**, and **logger name** when the root logger has no handlers yet. `app.py` does the same at import time when appropriate (e.g. under Streamlit).
