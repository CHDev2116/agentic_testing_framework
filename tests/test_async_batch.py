import asyncio
from pathlib import Path

from PIL import Image
import pytest

import ai_quality_agent as qa


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
