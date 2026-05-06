from ai_quality_agent import classify_loopback_signal, decide_loopback_action


def test_classify_loopback_signal_blurry():
    signal = classify_loopback_signal(
        {"decision": "Blurry", "msg": "Sharpness is below threshold"}
    )
    assert signal == "blurry"


def test_decide_loopback_action_for_underexposed():
    action, reason = decide_loopback_action(
        signal="under",
        engine_metrics={"avg_brightness": 12.0, "sharpness": 30.0},
        thresholds_cfg={"min_brightness": 40.0, "max_brightness": 220.0, "min_sharpness": 20.0},
        loopback_guard_cfg={"overexposure_stop_ratio": 0.95},
    )
    assert action == "brighten"
    assert reason == "retry_scheduled"


def test_decide_loopback_action_for_overexposed():
    action, reason = decide_loopback_action(
        signal="over",
        engine_metrics={"avg_brightness": 240.0, "sharpness": 30.0},
        thresholds_cfg={"min_brightness": 40.0, "max_brightness": 220.0, "min_sharpness": 20.0},
        loopback_guard_cfg={"underexposure_stop_ratio": 1.05},
    )
    assert action == "dim"
    assert reason == "retry_scheduled"


def test_decide_loopback_action_for_blurry():
    action, reason = decide_loopback_action(
        signal="blurry",
        engine_metrics={"avg_brightness": 120.0, "sharpness": 8.0},
        thresholds_cfg={"min_brightness": 40.0, "max_brightness": 220.0, "min_sharpness": 20.0},
        loopback_guard_cfg={},
    )
    assert action == "sharpen"
    assert reason == "retry_scheduled"
