#!/usr/bin/env bash
set -euo pipefail

echo "[prepush] Ruff"
ruff check src tests app.py test_connection.py

echo "[prepush] mypy"
mypy --explicit-package-bases src
MYPYPATH=src mypy --explicit-package-bases app.py test_connection.py

echo "[prepush] pytest (CI parity)"
PYTHONPATH=src pytest

echo "[prepush] all checks passed"
