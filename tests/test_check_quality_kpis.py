"""Tests for scripts/check_quality_kpis.py contract KPI gates."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_quality_kpis.py"


def _run_check(batch_report: dict, thresholds: dict, *, extra_args=None):
    batch_path = ROOT / "results" / "_kpi_test_batch.json"
    batch_path.parent.mkdir(parents=True, exist_ok=True)
    batch_path.write_text(json.dumps(batch_report), encoding="utf-8")
    thresh_path = ROOT / "results" / "_kpi_test_thresholds.json"
    thresh_path.write_text(json.dumps(thresholds), encoding="utf-8")
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--enforce",
        "--batch-report",
        str(batch_path),
        "--thresholds-file",
        str(thresh_path),
        "--repeatability-report",
        str(ROOT / "results" / "_missing_repeatability.json"),
    ]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)


def test_passes_when_contract_kpis_within_thresholds():
    report = {
        "results": [{"status": "SUCCESS"}],
        "summary": {
            "total_tests": 2,
            "quality_kpis": {
                "fallback_ratio": 0.0,
                "review_rate": 0.0,
                "json_repair_exhausted_count": 0,
                "semantic_assert_fail_count": 0,
                "strict_contract_violation_count": 0,
            },
        },
    }
    thresholds = {
        "max_fallback_ratio": 0.01,
        "max_review_rate": 1.0,
        "max_json_repair_exhausted_count": 0,
        "max_json_repair_exhausted_ratio": 0.0,
        "max_semantic_assert_fail_count": 0,
        "max_strict_contract_violation_count": 0,
    }
    result = _run_check(report, thresholds)
    assert result.returncode == 0, result.stdout + result.stderr


def test_fails_when_semantic_assert_fail_count_exceeds_max():
    report = {
        "results": [{"status": "SUCCESS"}, {"status": "SUCCESS"}],
        "summary": {
            "total_tests": 2,
            "quality_kpis": {
                "fallback_ratio": 0.0,
                "review_rate": 0.0,
                "json_repair_exhausted_count": 0,
                "semantic_assert_fail_count": 2,
                "strict_contract_violation_count": 0,
            },
        },
    }
    thresholds = {
        "max_fallback_ratio": 1.0,
        "max_review_rate": 1.0,
        "max_semantic_assert_fail_count": 0,
    }
    result = _run_check(report, thresholds)
    assert result.returncode == 1
    assert "semantic_assert_fail_count" in result.stdout + result.stderr
