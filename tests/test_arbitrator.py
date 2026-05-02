from eval.arbitrator import aggregate_batch_decisions, arbitrate_decision


def test_arbitrate_underexposed_returns_no_go():
    engine_metrics = {"avg_brightness": 10.0, "sharpness": 50.0}
    model_inference = {"status": "Under-exposed", "confidence": 0.9}
    thresholds = {"min_brightness": 40.0, "max_brightness": 220.0, "min_sharpness": 20.0}

    decision, _ = arbitrate_decision(engine_metrics, model_inference, thresholds)
    assert decision == "NO_GO"


def test_arbitrate_conflict_high_confidence_returns_review():
    engine_metrics = {"avg_brightness": 90.0, "sharpness": 60.0}
    model_inference = {"status": "Blurry", "confidence": 0.95}
    thresholds = {"min_brightness": 40.0, "max_brightness": 220.0, "min_sharpness": 20.0}

    decision, _ = arbitrate_decision(engine_metrics, model_inference, thresholds)
    assert decision == "REVIEW"


def test_aggregate_batch_decisions_conservative():
    assert aggregate_batch_decisions(["GO", "REVIEW"], "conservative") == "REVIEW"
    assert aggregate_batch_decisions(["GO", "NO_GO"], "conservative") == "NO_GO"
