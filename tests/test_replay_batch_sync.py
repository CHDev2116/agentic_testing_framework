"""Sync batch replay record/replay parity tests."""

import json
from pathlib import Path

from PIL import Image

import ai_quality_agent as qa
from models.contracts import LoopbackPlan


def _make_test_image(path: Path) -> None:
    Image.new("L", (32, 32), color=120).save(path)


def test_sync_replay_record_then_replay_keeps_decision_stable(monkeypatch, tmp_path):
    image_path = tmp_path / "replay_sync.png"
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
        },
        "folders": {
            "input": str(tmp_path),
            "output": str(tmp_path / "out"),
            "logs": str(tmp_path / "logs"),
        },
        "quality_gate": {"target_pass_rate": 80.0},
        "eval_settings": {"conflict_strategy": "conservative", "auto_tag_conflicts": True},
        "model_settings": {"name": "test", "bit_depth": 4, "inference": {"backend": "simulated"}},
    }
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

    def fake_analyze(self, photo_path):
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
        return str(tmp_path / "batch_report.json")

    monkeypatch.setattr(qa, "load_config", fake_load_config)
    monkeypatch.setattr(qa, "create_loopback_planner", lambda _cfg: FakePlanner())
    monkeypatch.setattr(qa.QuantizedVisionAgent, "get_all_photos", fake_get_all_photos)
    monkeypatch.setattr(qa.QuantizedVisionAgent, "analyze_photo_quality", fake_analyze)
    monkeypatch.setattr(qa.ImageProcessor, "adjust_brightness", fake_adjust_brightness)
    monkeypatch.setattr(qa, "save_batch_report", fake_save_batch_report)

    record_result = qa.run_batch_test(config_profile="dev")
    assert record_result is not None
    assert replay_file.exists()
    assert planner_call_count["count"] >= 1

    config["runtime"]["replay_mode"] = "replay"
    per_image_attempts.clear()
    planner_call_count["count"] = 0
    replay_result = qa.run_batch_test(config_profile="dev")

    assert replay_result is not None
    assert planner_call_count["count"] == 0
    assert (
        record_result["summary"]["release_decision"]
        == replay_result["summary"]["release_decision"]
    )
    assert record_result["summary"]["pass_rate"] == replay_result["summary"]["pass_rate"]


def test_finalize_batch_report_includes_quality_kpis():
    batch_report = {
        "batch_id": "test",
        "results": [
            {
                "file": "a.jpg",
                "status": "SUCCESS",
                "metrics": {"avg_brightness": 80.0, "sharpness": 50.0},
                "decision": {
                    "decision": "Optimal",
                    "code": "SUCCESS_200",
                    "backend": "simulated",
                    "confidence": 0.9,
                },
                "loopback": {"fallback_used": False},
            },
            {
                "file": "b.jpg",
                "status": "SUCCESS",
                "metrics": {"avg_brightness": 80.0, "sharpness": 50.0},
                "decision": {
                    "decision": "Blurry",
                    "code": "ERR_IMG_BLUR_101",
                    "backend": "llama_cpp->simulated",
                    "confidence": 0.95,
                },
                "arbitration": {
                    "release_decision": "REVIEW",
                    "conflict": "Engine OK; model negative verdict with high confidence (review)",
                },
                "loopback": {"fallback_used": True},
            },
        ],
    }
    kpis = qa._compute_batch_quality_kpis(
        batch_report,
        per_image_releases=["GO", "REVIEW"],
        review_breakdown={
            "Engine OK; model negative verdict with high confidence (review)": 1,
        },
    )
    assert kpis["review_count"] == 1
    assert kpis["review_rate"] == 0.5
    assert kpis["inference_fallback_count"] == 1
    assert kpis["loopback_fallback_count"] == 1
    assert kpis["fallback_ratio"] == 1.0
