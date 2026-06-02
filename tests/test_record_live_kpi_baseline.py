import json
import subprocess
import sys
from pathlib import Path


def test_record_live_kpi_baseline_extracts_observed(tmp_path):
    report = {
        "summary": {
            "total_tests": 2,
            "pass_rate": 50.0,
            "release_decision": "NO_GO",
            "quality_kpis": {
                "fallback_ratio": 0.5,
                "review_rate": 0.5,
                "review_count": 1,
                "json_repair_exhausted_count": 1,
                "semantic_assert_fail_count": 2,
                "llm_judge_calls": 1,
                "llm_judge_overrides": 0,
            },
        },
        "results": [{}, {}],
    }
    batch_path = tmp_path / "batch_report_test.json"
    batch_path.write_text(json.dumps(report), encoding="utf-8")
    out_path = tmp_path / "baseline.json"
    thresholds_path = tmp_path / "thresholds.json"

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/record_live_kpi_baseline.py",
            "--batch-report",
            str(batch_path),
            "--baseline-out",
            str(out_path),
            "--thresholds-out",
            str(thresholds_path),
            "--propose-thresholds",
            "--inference-backend",
            "llama_cpp",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0

    baseline = json.loads(out_path.read_text(encoding="utf-8"))
    assert baseline["observed"]["fallback_ratio"] == 0.5
    assert baseline["observed"]["llm_judge_calls"] == 1

    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
    assert thresholds["max_fallback_ratio"] >= 0.5
    assert thresholds["max_llm_judge_calls"] >= 1
