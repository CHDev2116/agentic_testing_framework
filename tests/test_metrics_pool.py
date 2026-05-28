from PIL import Image
import pytest

from engine.vision_math import calculate_metrics
from util.metrics_pool import MetricsProcessPool


def test_metrics_process_pool_matches_inline(tmp_path):
    image_path = tmp_path / "sample.png"
    Image.new("L", (64, 64), color=100).save(image_path)

    expected = calculate_metrics(str(image_path))

    try:
        with MetricsProcessPool(max_workers=2) as pool:
            actual = pool.calculate(str(image_path))
    except (PermissionError, NotImplementedError) as exc:
        pytest.skip(f"Process pool unavailable on this environment: {exc}")

    assert actual == expected
