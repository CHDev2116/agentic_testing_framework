import asyncio
import json
from pathlib import Path

from PIL import Image
import pytest

import ai_quality_agent as qa
from models.contracts import LoopbackPlan


def _make_test_image(path: Path):
    image = Image.new("L", (32, 32), color=120)
    image.save(path)


def test_run_batch_test_async_single_image(monkeypatch, tmp_path):
    image_path = tmp_path / "good.png"
    _make_test_image(image_path)

    config = {
        "thresholds": {
            "min_sharpness": 20.0,
            "min_brightness": 40.0,
            "max_brightness": 220.0,
        },
        "runtime": {"oom_probability": 0.0, "max_retry": 0},
        "folders": {"input": str(tmp_path), "output": str(tmp_path / "out"), "logs": str(tmp_path / "logs")},
        "quality_gate": {"target_pass_rate": 80.0},
        "eval_settings": {"conflict_strategy": "conservative", "auto_tag_conflicts": True},
        "model_settings": {"name": "test", "bit_depth": 4, "inference": {"backend": "simulated"}},
    }

    captured_report = {}

    def fake_load_config(profile, config_path):
        return config, "TEST_CONFIG"

    def fake_get_all_photos(self):
        return [str(image_path)]

    async def fake_analyze_async(self, photo_path, http_client):
        return (
            {"sharpness": 50.0, "avg_brightness": 80.0},
            {"decision": "Optimal", "code": "SUCCESS_200", "backend": "simulated"},
            1.0,
        )

    def fake_save_batch_report(report_data, output_folder):
        captured_report["data"] = report_data
        return str(tmp_path / "batch_report.json")

    monkeypatch.setattr(qa, "load_config", fake_load_config)
    monkeypatch.setattr(qa.QuantizedVisionAgent, "get_all_photos", fake_get_all_photos)
    monkeypatch.setattr(qa.QuantizedVisionAgent, "analyze_photo_quality_async", fake_analyze_async)
    monkeypatch.setattr(qa, "save_batch_report", fake_save_batch_report)

    result = qa.run_batch_test_async(config_profile="dev", concurrency=2)

    assert result is not None
    assert result["summary"]["release_decision"] in {"GO", "REVIEW", "NO_GO"}
    assert captured_report["data"]["execution_mode"] == "async"
    assert captured_report["data"]["async_concurrency"] == 2
    assert len(captured_report["data"]["results"]) == 1


def test_run_batch_test_async_marks_timeouts(monkeypatch, tmp_path):
    image_path = tmp_path / "slow.png"
    _make_test_image(image_path)

    config = {
        "thresholds": {
            "min_sharpness": 20.0,
            "min_brightness": 40.0,
            "max_brightness": 220.0,
        },
        "runtime": {"oom_probability": 0.0, "max_retry": 0},
        "folders": {"input": str(tmp_path), "output": str(tmp_path / "out"), "logs": str(tmp_path / "logs")},
        "quality_gate": {"target_pass_rate": 80.0},
        "eval_settings": {"conflict_strategy": "conservative", "auto_tag_conflicts": True},
        "model_settings": {"name": "test", "bit_depth": 4, "inference": {"backend": "simulated"}},
    }

    captured_report = {}

    def fake_load_config(profile, config_path):
        return config, "TEST_CONFIG"

    def fake_get_all_photos(self):
        return [str(image_path)]

    async def fake_analyze_async(self, photo_path, http_client):
        await asyncio.sleep(0.05)
        return (
            {"sharpness": 50.0, "avg_brightness": 80.0},
            {"decision": "Optimal", "code": "SUCCESS_200", "backend": "simulated"},
            1.0,
        )

    def fake_save_batch_report(report_data, output_folder):
        captured_report["data"] = report_data
        return str(tmp_path / "batch_report.json")

    monkeypatch.setattr(qa, "load_config", fake_load_config)
    monkeypatch.setattr(qa.QuantizedVisionAgent, "get_all_photos", fake_get_all_photos)
    monkeypatch.setattr(qa.QuantizedVisionAgent, "analyze_photo_quality_async", fake_analyze_async)
    monkeypatch.setattr(qa, "save_batch_report", fake_save_batch_report)

    result = qa.run_batch_test_async(
        config_profile="dev",
        concurrency=2,
        async_per_image_timeout_s_override=0.01,
        async_backend_health_check_override=False,
    )

    assert result is not None
    assert captured_report["data"]["summary"]["async_timeout_count"] == 1
    assert captured_report["data"]["summary"]["async_per_image_timeout_s"] == 0.01
    assert captured_report["data"]["results"][0]["timed_out"] is True


def test_run_batch_test_async_fail_fast_when_backend_unreachable(monkeypatch, tmp_path):
    image_path = tmp_path / "bad.png"
    _make_test_image(image_path)

    config = {
        "thresholds": {
            "min_sharpness": 20.0,
            "min_brightness": 40.0,
            "max_brightness": 220.0,
        },
        "runtime": {"oom_probability": 0.0, "max_retry": 0},
        "folders": {"input": str(tmp_path), "output": str(tmp_path / "out"), "logs": str(tmp_path / "logs")},
        "quality_gate": {"target_pass_rate": 80.0},
        "eval_settings": {"conflict_strategy": "conservative", "auto_tag_conflicts": True},
        "model_settings": {"name": "test", "bit_depth": 4, "inference": {"backend": "simulated"}},
    }

    class FakeEngine:
        backend_name = "llama_cpp"
        host = "http://127.0.0.1:9"
        endpoint = "/v1/chat/completions"

    def fake_load_config(profile, config_path):
        return config, "TEST_CONFIG"

    def fake_build_engine(_config):
        return FakeEngine()

    monkeypatch.setattr(qa, "load_config", fake_load_config)
    monkeypatch.setattr(qa, "build_inference_engine", fake_build_engine)

    with pytest.raises(RuntimeError, match="Async inference backend not reachable"):
        qa.run_batch_test_async(config_profile="dev", concurrency=2)


def test_replay_record_then_replay_keeps_decision_stable(monkeypatch, tmp_path):
    image_path = tmp_path / "replay.png"
    _make_test_image(image_path)
    replay_file = tmp_path / "replay_trace.jsonl"

    config = {
        "thresholds": {
            "min_sharpness": 20.0,
            "min_brightness": 40.0,
            "max_brightness": 220.0,
        },
        "runtime": {
            "oom_probability": 0.0,
            "max_retry": 1,
            "replay_mode": "record",
            "replay_file": str(replay_file),
            "async_backend_health_check": False,
        },
        "folders": {"input": str(tmp_path), "output": str(tmp_path / "out"), "logs": str(tmp_path / "logs")},
        "quality_gate": {"target_pass_rate": 80.0},
        "eval_settings": {"conflict_strategy": "conservative", "auto_tag_conflicts": True},
        "model_settings": {"name": "test", "bit_depth": 4, "inference": {"backend": "simulated"}},
    }
    captured_reports = []
    planner_call_count = {"count": 0}
    per_image_attempts = {}

    class FakePlanner:
        def plan(self, **kwargs):
            planner_call_count["count"] += 1
            return LoopbackPlan(
                action="brighten",
                stop_reason="retry_scheduled",
                rationale="test planner",
                fallback_used=False,
                planner_backend="simulated",
            )

    def fake_load_config(profile, config_path):
        return json.loads(json.dumps(config)), "TEST_CONFIG"

    def fake_get_all_photos(self):
        return [str(image_path)]

    async def fake_analyze_async(self, photo_path, http_client):
        attempt = per_image_attempts.get(photo_path, 0) + 1
        per_image_attempts[photo_path] = attempt
        if attempt == 1:
            return (
                {"sharpness": 10.0, "avg_brightness": 10.0},
                {"decision": "Under-exposed", "code": "ERR_LIGHT_DARK_002", "backend": "simulated"},
                1.0,
            )
        return (
            {"sharpness": 50.0, "avg_brightness": 80.0},
            {"decision": "Optimal", "code": "SUCCESS_200", "backend": "simulated"},
            1.0,
        )

    def fake_adjust_brightness(self, current_path, level, file_stem, attempt_idx):
        return current_path

    def fake_save_batch_report(report_data, output_folder):
        captured_reports.append(report_data)
        return str(tmp_path / "batch_report.json")

    monkeypatch.setattr(qa, "load_config", fake_load_config)
    monkeypatch.setattr(qa, "create_loopback_planner", lambda _cfg: FakePlanner())
    monkeypatch.setattr(qa.QuantizedVisionAgent, "get_all_photos", fake_get_all_photos)
    monkeypatch.setattr(qa.QuantizedVisionAgent, "analyze_photo_quality_async", fake_analyze_async)
    monkeypatch.setattr(qa.ImageProcessor, "adjust_brightness", fake_adjust_brightness)
    monkeypatch.setattr(qa, "save_batch_report", fake_save_batch_report)

    record_result = qa.run_batch_test_async(config_profile="dev", concurrency=1)
    assert record_result is not None
    assert replay_file.exists()
    assert planner_call_count["count"] >= 1

    # Replay mode should not call live planner.
    config["runtime"]["replay_mode"] = "replay"
    per_image_attempts.clear()
    planner_call_count["count"] = 0
    replay_result = qa.run_batch_test_async(config_profile="dev", concurrency=1)

    assert replay_result is not None
    assert planner_call_count["count"] == 0
    assert record_result["summary"]["release_decision"] == replay_result["summary"]["release_decision"]
    assert record_result["summary"]["pass_rate"] == replay_result["summary"]["pass_rate"]
    assert len(captured_reports) == 2


def test_replay_mode_hard_fails_on_hash_mismatch(monkeypatch, tmp_path):
    image_path = tmp_path / "replay_mismatch.png"
    _make_test_image(image_path)
    replay_file = tmp_path / "replay_trace_bad.jsonl"
    replay_file.write_text(
        json.dumps(
            {
                "image_path": str(image_path),
                "attempt": 1,
                "metrics_before": {"avg_brightness": 10.0, "sharpness": 10.0},
                "planner_input_hash": "wrong-hash",
                "planner_output": {"action": "brighten", "stop_reason": "retry_scheduled", "rationale": "x"},
                "action": "brighten",
                "signal": "under",
                "backend": "simulated",
                "latency_ms": 1.0,
                "stop_reason": "retry_scheduled",
                "timestamp": "2026-06-01T11:00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    config = {
        "thresholds": {
            "min_sharpness": 20.0,
            "min_brightness": 40.0,
            "max_brightness": 220.0,
        },
        "runtime": {
            "oom_probability": 0.0,
            "max_retry": 1,
            "replay_mode": "replay",
            "replay_file": str(replay_file),
            "async_backend_health_check": False,
        },
        "folders": {"input": str(tmp_path), "output": str(tmp_path / "out"), "logs": str(tmp_path / "logs")},
        "quality_gate": {"target_pass_rate": 80.0},
        "eval_settings": {"conflict_strategy": "conservative", "auto_tag_conflicts": True},
        "model_settings": {"name": "test", "bit_depth": 4, "inference": {"backend": "simulated"}},
    }

    def fake_load_config(profile, config_path):
        return config, "TEST_CONFIG"

    def fake_get_all_photos(self):
        return [str(image_path)]

    async def fake_analyze_async(self, photo_path, http_client):
        return (
            {"sharpness": 10.0, "avg_brightness": 10.0},
            {"decision": "Under-exposed", "code": "ERR_LIGHT_DARK_002", "backend": "simulated"},
            1.0,
        )

    captured_report = {}

    def fake_save_batch_report(report_data, output_folder):
        captured_report["data"] = report_data
        return str(tmp_path / "batch_report.json")

    monkeypatch.setattr(qa, "load_config", fake_load_config)
    monkeypatch.setattr(qa.QuantizedVisionAgent, "get_all_photos", fake_get_all_photos)
    monkeypatch.setattr(qa.QuantizedVisionAgent, "analyze_photo_quality_async", fake_analyze_async)
    monkeypatch.setattr(qa, "save_batch_report", fake_save_batch_report)

    result = qa.run_batch_test_async(config_profile="dev", concurrency=1)
    assert result is not None
    assert captured_report["data"]["results"][0]["status"] == "FAILED"
    assert captured_report["data"]["results"][0]["error_type"] == "ReplayTraceError"
