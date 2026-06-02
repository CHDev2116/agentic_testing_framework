#!/usr/bin/env bash
# Post oracle semantics diff to GitHub Actions job summary; optional snapshot enforce on PR.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH=src

REPORT="${RUNNER_TEMP:-/tmp}/oracle_semantic_changelog.md"

set +e
python scripts/diff_oracle_semantics.py --report "${REPORT}"
diff_rc=$?
set -e

if [[ -f "${REPORT}" ]]; then
  if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    {
      echo "## Oracle semantic drift"
      echo ""
      cat "${REPORT}"
    } >> "${GITHUB_STEP_SUMMARY}"
  fi
  cat "${REPORT}"
fi

if [[ "${CI_ORACLE_SNAPSHOT_ENFORCE:-0}" == "1" ]]; then
  echo "[oracle-ci] enforcing snapshot parity (PR gate)"
  python scripts/diff_oracle_semantics.py --enforce
fi

exit 0
