from ai_quality_agent import _build_agent_inference_output


def test_build_agent_inference_output_contains_steps():
    output = _build_agent_inference_output(
        image_path="images/a.jpg",
        attempt_history=[
            {
                "attempt": 1,
                "release": "NO_GO",
                "loopback_signal": "under",
                "action": "brighten",
                "rationale": "too dark",
                "planner_fallback_used": True,
                "avg_brightness": 10.0,
                "sharpness": 4.0,
                "latency_ms": 12.5,
            },
            {
                "attempt": 2,
                "release": "REVIEW",
                "loopback_signal": "other",
                "action": "stop",
                "rationale": "resolved enough",
                "planner_fallback_used": False,
                "avg_brightness": 42.0,
                "sharpness": 4.5,
                "latency_ms": 9.0,
            },
        ],
        final_ai_result={"code": "ERR_LIGHT_DARK_002", "msg": "Under-exposed"},
        total_latency_ms=21.5,
    )
    assert output["image_path"] == "images/a.jpg"
    assert output["final_decision"] == "REVIEW"
    assert output["error_code"] == "ERR_LIGHT_DARK_002"
    assert len(output["steps"]) == 2
    assert output["steps"][0]["fallback_used"] is True
    assert output["steps"][0]["metrics_before"]["avg_brightness"] == 10.0
    assert output["steps"][0]["metrics_after"]["avg_brightness"] == 42.0

