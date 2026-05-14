"""Real-image metric extraction (Pillow path); missing file and bad path edges."""

from PIL import Image

from engine.vision_math import calculate_metrics


def test_calculate_metrics_missing_file(tmp_path):
    missing = tmp_path / "does_not_exist.png"
    assert calculate_metrics(str(missing)) is None


def test_calculate_metrics_uniform_image(tmp_path):
    path = tmp_path / "gray.png"
    Image.new("L", (64, 64), color=100).save(path)
    out = calculate_metrics(str(path))
    assert out is not None
    assert out["avg_brightness"] == 100.0
    assert out["max_brightness"] == 100
    assert out["sharpness"] == 0.0
