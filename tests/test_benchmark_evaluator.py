"""Golden-style checks for ranking score, sort order, and release gate."""

from eval import benchmark_evaluator as be


def test_calculate_quality_score_success_vs_error_code_penalty():
    metrics = {"sharpness": 20.0, "avg_brightness": 130.0}
    thresholds = {
        "min_sharpness": 20.0,
        "min_brightness": 40.0,
        "max_brightness": 220.0,
    }
    ok = {"code": "SUCCESS_200", "decision": "GO"}
    bad = {"code": "TIMEOUT", "decision": "NO_GO"}

    high = be.calculate_quality_score(metrics, ok, thresholds)
    low = be.calculate_quality_score(metrics, bad, thresholds)

    assert high == 100.0
    assert low == round(100.0 * 0.4, 2)


def test_calculate_quality_score_non_dict_metrics_returns_zero():
    assert be.calculate_quality_score(None, {"code": "SUCCESS_200"}, {}) == 0.0


def test_build_rankings_sorts_by_score_latency_then_file():
    thresholds = {"min_sharpness": 10, "min_brightness": 40, "max_brightness": 220}
    rows = [
        {
            "file": "b.png",
            "metrics": {"sharpness": 10, "avg_brightness": 130},
            "decision": {"decision": "GO", "code": "SUCCESS_200"},
            "latency_ms": 20,
            "status": "OK",
        },
        {
            "file": "a.png",
            "metrics": {"sharpness": 10, "avg_brightness": 130},
            "decision": {"decision": "GO", "code": "SUCCESS_200"},
            "latency_ms": 10,
            "status": "OK",
        },
        {
            "file": "c.png",
            "metrics": {"sharpness": 5, "avg_brightness": 130},
            "decision": {"decision": "REVIEW", "code": "SUCCESS_200"},
            "latency_ms": 5,
            "status": "OK",
        },
    ]
    ranked = be.build_rankings(rows, thresholds)
    # Same score -> lower latency first; then lexicographic file name as tie-breaker.
    assert [r["file"] for r in ranked] == ["a.png", "b.png", "c.png"]
    assert [r["rank"] for r in ranked] == [1, 2, 3]


def test_get_release_decision_go_review_no_go_boundaries():
    base_cfg = {
        "quality_gate": {"target_pass_rate": 90.0},
        "thresholds": {"timeout_ms": 5000},
    }

    go, msg_go = be.get_release_decision(95.0, 2000.0, base_cfg)
    assert go == "GO"
    assert "90" in msg_go and "2500" in msg_go

    review, _ = be.get_release_decision(80.0, 2000.0, base_cfg)
    assert review == "REVIEW"

    no_go, _ = be.get_release_decision(50.0, 2000.0, base_cfg)
    assert no_go == "NO_GO"


def test_generate_benchmark_insights_returns_three_items():
    ordered = [
        {
            "profile": "dev",
            "summary": {"release_decision": "GO", "avg_latency_ms": 100, "target_pass_rate": 85},
        }
    ]
    profile_outputs = [
        {"profile": "dev", "summary": {"avg_latency_ms": 100, "target_pass_rate": 85}},
        {"profile": "strict", "summary": {"avg_latency_ms": 200, "target_pass_rate": 99}},
    ]
    insights = be.generate_benchmark_insights(profile_outputs, ordered)
    assert len(insights) == 3
    assert all("trade_off" in item for item in insights)
