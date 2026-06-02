from agent.planner_contract import (
    PlannerRepairSettings,
    run_planner_contract_loop,
    validate_planner_payload,
)


def test_validate_planner_payload_accepts_valid():
    assert not validate_planner_payload(
        {"action": "brighten", "rationale": "increase exposure"}
    )


def test_planner_repair_loop_succeeds_on_second_fetch():
    settings = PlannerRepairSettings(max_json_repair_attempts=2)
    calls = {"n": 0}

    def fetch(_prompt: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            return '{"action":'
        return '{"action": "stop", "rationale": "done"}'

    parsed, attempts = run_planner_contract_loop(
        settings,
        build_initial_prompt=lambda: "planner prompt",
        fetch_model_text=fetch,
    )
    assert attempts == 1
    assert parsed.get("action") == "stop"
