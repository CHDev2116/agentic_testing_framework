"""Historical oracle regression — frozen cases must keep release/conflict semantics."""

from pathlib import Path

import pytest

from eval.arbitrator import aggregate_batch_decisions
from eval.oracle_regression import (
    assert_case_matches,
    load_oracle_cases,
    run_oracle_case,
    run_regression_suite,
)

REGRESSION_FILE = Path(__file__).resolve().parent / "regression" / "oracle_cases.jsonl"


def test_regression_suite_has_minimum_case_count():
    cases = load_oracle_cases(REGRESSION_FILE)
    assert len(cases) >= 17, "historical regression corpus should have at least 17 cases"


def test_regression_suite_all_cases_pass():
    failures = run_regression_suite(REGRESSION_FILE)
    if failures:
        pytest.fail("\n".join(failures))


@pytest.mark.parametrize("case", load_oracle_cases(REGRESSION_FILE), ids=lambda c: c["id"])
def test_oracle_case_individually(case):
    actual = run_oracle_case(case)
    mismatches = assert_case_matches(case, actual)
    assert not mismatches, "; ".join(mismatches)


def test_aggregate_batch_regression_invariants():
    """Batch roll-up rules used in production — guard against silent strategy changes."""
    assert aggregate_batch_decisions(["GO", "NO_GO"], "conservative") == "NO_GO"
    assert aggregate_batch_decisions(["GO", "REVIEW"], "conservative") == "REVIEW"
    assert aggregate_batch_decisions(["GO", "GO"], "conservative") == "GO"
