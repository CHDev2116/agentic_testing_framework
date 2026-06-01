import json

import pytest

from util.replay_trace import (
    ReplayTraceError,
    append_replay_step,
    build_replay_index,
    get_replay_step,
    load_replay_steps,
    planner_input_hash,
)


def _sample_step(*, image_path: str = "test_images/a.jpg", attempt: int = 1, hash_value: str = "h1"):
    return {
        "image_path": image_path,
        "attempt": attempt,
        "metrics_before": {"avg_brightness": 10.0, "sharpness": 12.3},
        "planner_input_hash": hash_value,
        "planner_output": {"action": "brighten", "rationale": "dark"},
        "action": "brighten",
        "signal": "under",
        "backend": "llama_cpp",
        "latency_ms": 4.2,
        "stop_reason": "",
        "timestamp": "2026-06-01T10:00:00",
    }


def test_planner_input_hash_is_stable_for_key_order():
    a = {"attempt": 1, "metrics": {"x": 1, "y": 2}}
    b = {"metrics": {"y": 2, "x": 1}, "attempt": 1}
    assert planner_input_hash(a) == planner_input_hash(b)


def test_append_and_load_replay_steps_roundtrip(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    s1 = _sample_step(hash_value="hash-1")
    s2 = _sample_step(image_path="test_images/b.jpg", attempt=2, hash_value="hash-2")

    append_replay_step(str(trace_path), s1)
    append_replay_step(str(trace_path), s2)

    loaded = load_replay_steps(str(trace_path))
    assert loaded == [s1, s2]


def test_load_replay_steps_raises_on_invalid_json(tmp_path):
    trace_path = tmp_path / "bad.jsonl"
    trace_path.write_text("{bad json}\n", encoding="utf-8")
    with pytest.raises(ReplayTraceError, match="Invalid JSON"):
        load_replay_steps(str(trace_path))


def test_build_replay_index_rejects_duplicate_key():
    steps = [_sample_step(), _sample_step()]
    with pytest.raises(ReplayTraceError, match="Duplicate replay step"):
        build_replay_index(steps)


def test_get_replay_step_returns_matching_record():
    step = _sample_step(hash_value="expected-hash")
    index = build_replay_index([step])
    result = get_replay_step(
        index,
        image_path="test_images/a.jpg",
        attempt=1,
        expected_planner_input_hash="expected-hash",
    )
    assert result["action"] == "brighten"


def test_get_replay_step_raises_on_hash_mismatch():
    step = _sample_step(hash_value="trace-hash")
    index = build_replay_index([step])
    with pytest.raises(ReplayTraceError, match="hash mismatch"):
        get_replay_step(
            index,
            image_path="test_images/a.jpg",
            attempt=1,
            expected_planner_input_hash="runtime-hash",
        )


def test_get_replay_step_raises_on_missing_step():
    step = _sample_step()
    index = build_replay_index([step])
    with pytest.raises(ReplayTraceError, match="Missing replay step"):
        get_replay_step(
            index,
            image_path="test_images/missing.jpg",
            attempt=1,
            expected_planner_input_hash="h1",
        )


def test_append_replay_step_validates_required_fields(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    invalid = {"image_path": "x.jpg", "attempt": 1}
    with pytest.raises(ReplayTraceError, match="missing required fields"):
        append_replay_step(str(trace_path), invalid)


def test_load_replay_steps_skips_blank_lines(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    payload = _sample_step()
    trace_path.write_text("\n" + json.dumps(payload) + "\n\n", encoding="utf-8")
    loaded = load_replay_steps(str(trace_path))
    assert len(loaded) == 1
    assert loaded[0]["image_path"] == "test_images/a.jpg"
