import ai_quality_agent as qa


def test_apply_runtime_overrides_updates_backend_and_planner():
    config = {
        "model_settings": {"inference": {"backend": "simulated"}},
        "runtime": {"loopback_planner": {"mode": "simulated"}},
    }
    source = qa._apply_runtime_overrides(
        config,
        "BASE",
        inference_backend_override="mock_api",
        loopback_planner_override="llm",
    )
    assert config["model_settings"]["inference"]["backend"] == "mock_api"
    assert config["runtime"]["loopback_planner"]["mode"] == "llm"
    assert "backend=mock_api" in source
    assert "loopback_planner=llm" in source


def test_apply_runtime_overrides_updates_planner_llm_fields():
    config = {"runtime": {"loopback_planner": {"mode": "llm", "llm": {}}}}
    source = qa._apply_runtime_overrides(
        config,
        "BASE",
        planner_timeout_s_override=7.5,
        planner_model_override="llama-planner-q4",
    )
    assert config["runtime"]["loopback_planner"]["llm"]["timeout_s"] == 7.5
    assert config["runtime"]["loopback_planner"]["llm"]["model"] == "llama-planner-q4"
    assert "planner_timeout_s=7.5" in source
    assert "planner_model=llama-planner-q4" in source


def test_apply_runtime_overrides_updates_planner_health_policy():
    config = {"runtime": {"loopback_planner": {"mode": "llm"}}}
    source = qa._apply_runtime_overrides(
        config,
        "BASE",
        planner_require_healthy_override=False,
    )
    assert config["runtime"]["loopback_planner"]["require_healthy_on_startup"] is False
    assert "planner_require_healthy=False" in source


def test_apply_runtime_overrides_updates_replay_settings():
    config = {"runtime": {}}
    source = qa._apply_runtime_overrides(
        config,
        "BASE",
        replay_mode_override="record",
        replay_file_override="results/replay_trace.jsonl",
    )
    assert config["runtime"]["replay_mode"] == "record"
    assert config["runtime"]["replay_file"] == "results/replay_trace.jsonl"
    assert "replay_mode=record" in source
    assert "replay_file=results/replay_trace.jsonl" in source

