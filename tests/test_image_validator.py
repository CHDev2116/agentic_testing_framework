"""Exposure histogram rules on synthetic grayscale images (OpenCV read path)."""

from pathlib import Path

import numpy as np
from PIL import Image

from engine.image_validator import ImageQualityValidator


def _save_gray(path: Path, value: int) -> None:
    Image.new("L", (64, 64), color=value).save(path)


def test_analyze_exposure_missing_file(tmp_path):
    v = ImageQualityValidator()
    out = v.analyze_exposure(str(tmp_path / "missing.png"))
    assert out == "Error: Image not found"


def test_analyze_exposure_pass_mid_gray(tmp_path):
    path = tmp_path / "mid.png"
    _save_gray(path, 128)
    v = ImageQualityValidator(brightness_threshold=0.7, dark_threshold=0.7)
    result = v.analyze_exposure(str(path))
    assert result["verdict"] == "Pass"
    assert result["dark_ratio"] < 0.7
    assert result["bright_ratio"] < 0.7


def test_analyze_exposure_fail_too_dark(tmp_path):
    path = tmp_path / "dark.png"
    _save_gray(path, 0)
    v = ImageQualityValidator(dark_threshold=0.5)
    result = v.analyze_exposure(str(path))
    assert result["verdict"] == "Fail: Too Dark"


def test_analyze_exposure_fail_overexposed(tmp_path):
    path = tmp_path / "bright.png"
    _save_gray(path, 255)
    v = ImageQualityValidator(brightness_threshold=0.5)
    result = v.analyze_exposure(str(path))
    assert result["verdict"] == "Fail: Overexposed"


def test_histogram_ratios_are_normalized(tmp_path):
    """Mass in high bins should dominate bright_ratio (sanity on OpenCV histogram)."""
    path = tmp_path / "bright_strip.png"
    arr = np.zeros((32, 32), dtype=np.uint8)
    arr[:, 16:] = 250
    Image.fromarray(arr, mode="L").save(path)
    v = ImageQualityValidator(brightness_threshold=0.2)
    result = v.analyze_exposure(str(path))
    assert result["bright_ratio"] >= 0.45
