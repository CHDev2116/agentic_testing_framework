# Contributing

Thanks for helping improve this project. Small, focused changes are easier to review and merge.

## Prerequisites

- Python **3.9+** (CI runs on **3.11**; matching CI locally avoids surprises).
- A clone of the repository.

## Local setup

```bash
cd agentic_testing_framework
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -U pip
pip install -e ".[dev]"
```

Alternatively, install from the pinned list (same set Docker uses; keep versions aligned with `pyproject.toml`):

```bash
pip install -r requirements.txt
pip install pytest pytest-cov ruff
```

The CLI and tests expect `src` on the module path. Use `PYTHONPATH=src` as shown below (same as CI).

## Run tests

```bash
PYTHONPATH=src pytest
```

Coverage options are defined in `pyproject.toml` (`--cov=src`, XML report for tooling).

## Lint

CI runs Ruff on the full Python tree under `src` plus `tests`. Match it before opening a PR:

```bash
ruff check src tests
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
4. **Green CI**: Ensure tests and the lint step above pass locally.
5. **Docs**: If you change CLI flags, config shape, or inference behavior, update `README.md` and any affected file under `docs/`.

## Code style

Follow existing patterns in nearby modules (logging, typing, error messages). Prefer clear names and small functions over clever one-liners.
