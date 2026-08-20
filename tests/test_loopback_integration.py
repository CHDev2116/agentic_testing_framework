from pathlib import Path

from PIL import Image

import ai_quality_agent as qa


def _make_test_image(path: Path):
    image = Image.new("L", (32, 32), color=20)
    image.save(path)


def test_run_batch_test_closed_loop_recovers_underexposed(monkeypatch, tmp_path):
    image_path = tmp_path / "under.png"
    _make_test_image(image_path)

    config = {
        "thresholds": {
            "min_sharpness": 20.0,
            "min_brightness": 40.0,
            "max_brightness": 220.0,
        },
        "runtime": {
            "oom_probability": 0.0,
            "max_retry": 2,
            "loopback_guard": {
                "min_brightness_gain": 4.0,
                "min_sharpness_gain": 1.0,
                "brighten_factor": 1.2,
                "dim_factor": 0.85,
                "overexposure_stop_ratio": 0.95,
                "underexposure_stop_ratio": 1.05,
            },
        },
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

    scripted = iter(
        [
            ({"avg_brightness": 12.0, "sharpness": 35.0}, {"decision": "Under-exposed", "code": "ERR_LIGHT_DARK_002"}, 1.0),
            ({"avg_brightness": 80.0, "sharpness": 35.0}, {"decision": "Optimal", "code": "SUCCESS_200"}, 1.0),
        ]
    )

    def fake_analyze(self, photo_path):
        return next(scripted)

    def fake_adjust_brightness(self, image_path, level, file_stem, attempt_idx):
        # Return same image path to keep test deterministic and file-backed.
        return image_path

    def fake_save_batch_report(report_data, output_folder):
        captured_report["data"] = report_data
        return str(tmp_path / "batch_report.json")

    monkeypatch.setattr(qa, "load_config", fake_load_config)
    monkeypatch.setattr(qa.QuantizedVisionAgent, "get_all_photos", fake_get_all_photos)
    monkeypatch.setattr(qa.QuantizedVisionAgent, "analyze_photo_quality", fake_analyze)
    monkeypatch.setattr(qa.ImageProcessor, "adjust_brightness", fake_adjust_brightness)
    monkeypatch.setattr(qa, "save_batch_report", fake_save_batch_report)

    result = qa.run_batch_test(config_profile="dev")

    assert result["summary"]["release_decision"] in {"GO", "REVIEW"}
    attempts = captured_report["data"]["results"][0]["loopback"]["attempts"]
    assert len(attempts) == 2
    assert attempts[0]["loopback_signal"] == "under"
    assert attempts[1]["model_decision"] == "Optimal"


def test_run_batch_test_writes_critique_summary(monkeypatch, tmp_path):
    image_path = tmp_path / "under.png"
    _make_test_image(image_path)

    config = {
        "thresholds": {
            "min_sharpness": 20.0,
            "min_brightness": 40.0,
            "max_brightness": 220.0,
        },
        "runtime": {
            "oom_probability": 0.0,
            "max_retry": 0,
        },
        "folders": {"input": str(tmp_path), "output": str(tmp_path / "out"), "logs": str(tmp_path / "logs")},
        "quality_gate": {"target_pass_rate": 80.0},
        "eval_settings": {
            "conflict_strategy": "conservative",
            "auto_tag_conflicts": True,
            "semantic_asserts_enabled": True,
        },
        "model_settings": {"name": "test", "bit_depth": 4, "inference": {"backend": "simulated"}},
    }

    def fake_load_config(profile, config_path):
        return config, "TEST_CONFIG"

    def fake_get_all_photos(self):
        return [str(image_path)]

    def fake_analyze(self, photo_path):
        return (
            {"avg_brightness": 80.0, "sharpness": 35.0},
            {"decision": "Optimal", "code": "SUCCESS_200", "msg": "ok"},
            1.0,
        )

    monkeypatch.setattr(qa, "load_config", fake_load_config)
    monkeypatch.setattr(qa.QuantizedVisionAgent, "get_all_photos", fake_get_all_photos)
    monkeypatch.setattr(qa.QuantizedVisionAgent, "analyze_photo_quality", fake_analyze)

    result = qa.run_batch_test(config_profile="dev")

    critique_path = Path(result["critique_summary_path"])
    assert critique_path.exists()
    content = critique_path.read_text(encoding="utf-8")
    assert "rows" in content
    assert "under.png" in content
