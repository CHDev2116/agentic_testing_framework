from agent.loopback_planner import LLMLoopbackPlanner, create_loopback_planner


def test_create_loopback_planner_default_simulated():
    planner = create_loopback_planner({"runtime": {}})
    plan = planner.plan(
        signal="under",
        engine_metrics={"avg_brightness": 10.0, "sharpness": 25.0},
        thresholds_cfg={"min_brightness": 40.0, "max_brightness": 220.0, "min_sharpness": 20.0},
        loopback_guard_cfg={},
        attempt_history=[],
    )
    assert plan.action == "brighten"
    assert plan.stop_reason == "retry_scheduled"


def test_llm_loopback_planner_falls_back_on_network_error():
    planner = LLMLoopbackPlanner(
        planner_cfg={"host": "http://127.0.0.1:9", "timeout_s": 0.01},
        fallback_planner=create_loopback_planner({"runtime": {}}),
    )
    plan = planner.plan(
        signal="blurry",
        engine_metrics={"avg_brightness": 120.0, "sharpness": 5.0},
        thresholds_cfg={"min_brightness": 40.0, "max_brightness": 220.0, "min_sharpness": 20.0},
        loopback_guard_cfg={},
        attempt_history=[],
    )
    assert plan.action == "sharpen"
    assert plan.stop_reason == "retry_scheduled"
    assert plan.fallback_used is True
    assert plan.planner_backend == "llm->simulated"


def test_create_loopback_planner_llm_health_check_fail_fast():
    try:
        create_loopback_planner(
            {
                "runtime": {
                    "loopback_planner": {
                        "mode": "llm",
                        "require_healthy_on_startup": True,
                        "llm": {"host": "http://127.0.0.1:9", "timeout_s": 0.01},
                    }
                }
            }
        )
    except RuntimeError as exc:
        assert "not reachable" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for unreachable llm planner endpoint")

