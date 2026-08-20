from ai_quality_agent import classify_loopback_signal, plan_next_action


def test_classify_loopback_signal_blurry():
    signal = classify_loopback_signal(
        {"decision": "Blurry", "msg": "Sharpness is below threshold"}
    )
    assert signal == "blurry"


def test_plan_next_action_for_underexposed():
    plan = plan_next_action(
        signal="under",
        engine_metrics={"avg_brightness": 12.0, "sharpness": 30.0},
        thresholds_cfg={"min_brightness": 40.0, "max_brightness": 220.0, "min_sharpness": 20.0},
        loopback_guard_cfg={"overexposure_stop_ratio": 0.95},
    )
    assert plan.action == "brighten"
    assert plan.stop_reason == "retry_scheduled"


def test_plan_next_action_for_overexposed():
    plan = plan_next_action(
        signal="over",
        engine_metrics={"avg_brightness": 240.0, "sharpness": 30.0},
        thresholds_cfg={"min_brightness": 40.0, "max_brightness": 220.0, "min_sharpness": 20.0},
        loopback_guard_cfg={"underexposure_stop_ratio": 1.05},
    )
    assert plan.action == "dim"
    assert plan.stop_reason == "retry_scheduled"


def test_plan_next_action_for_blurry():
    plan = plan_next_action(
        signal="blurry",
        engine_metrics={"avg_brightness": 120.0, "sharpness": 8.0},
        thresholds_cfg={"min_brightness": 40.0, "max_brightness": 220.0, "min_sharpness": 20.0},
        loopback_guard_cfg={},
    )
    assert plan.action == "sharpen"
    assert plan.stop_reason == "retry_scheduled"


def test_plan_next_action_reports_rationale():
    plan = plan_next_action(
        signal="under",
        engine_metrics={"avg_brightness": 10.0, "sharpness": 30.0},
        thresholds_cfg={"min_brightness": 40.0, "max_brightness": 220.0, "min_sharpness": 20.0},
        loopback_guard_cfg={"overexposure_stop_ratio": 0.95},
    )
    assert plan.action == "brighten"
    assert plan.stop_reason == "retry_scheduled"
    assert "under-exposed" in plan.rationale
