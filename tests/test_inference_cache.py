import logging
from typing import Any, Dict

from models.inference_cache import CachingInferenceEngine, InferenceCache, InferenceCacheSettings


class CountingEngine:
    def __init__(self):
        self.count = 0

    def predict_quality(self, photo_path: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        self.count += 1
        x = metrics.get("x")
        return {
            "decision": "Optimal",
            "code": "SUCCESS_200",
            "msg": f"ok-{x}",
            "backend": "dummy",
        }


def _make_engine(cache_dir: str, *, replay_mode: str = "off", rules_tag: str = "rules_v1"):
    base = CountingEngine()
    cache_settings = InferenceCacheSettings(enabled=True, cache_dir=cache_dir)
    test_logger = logging.getLogger("test_inference_cache")
    cache = InferenceCache(cache_settings, logger_=test_logger)
    caching_engine = CachingInferenceEngine(
        base_engine=base,
        cache=cache,
        cache_context={
            "backend_id": "dummy_backend",
            "rules_tag": rules_tag,
            "replay_mode": replay_mode,
            "version_tag": cache_settings.version_tag,
        },
        logger_=test_logger,
    )
    return base, caching_engine


def test_inference_cache_hit(tmp_path):
    photo = tmp_path / "img.png"
    photo.write_bytes(b"img-bytes-1")

    base, caching = _make_engine(str(tmp_path / "cache"))

    out1 = caching.predict_quality(str(photo), {"x": 1})
    out2 = caching.predict_quality(str(photo), {"x": 1})

    assert base.count == 1
    assert out1 == out2


def test_inference_cache_miss_on_metrics_change(tmp_path):
    photo = tmp_path / "img.png"
    photo.write_bytes(b"img-bytes-1")

    base, caching = _make_engine(str(tmp_path / "cache"))

    out1 = caching.predict_quality(str(photo), {"x": 1})
    out2 = caching.predict_quality(str(photo), {"x": 2})

    assert base.count == 2
    assert out1 != out2


def test_inference_cache_bypass_on_replay_mode(tmp_path):
    photo = tmp_path / "img.png"
    photo.write_bytes(b"img-bytes-1")

    base, caching = _make_engine(str(tmp_path / "cache"), replay_mode="replay")

    caching.predict_quality(str(photo), {"x": 1})
    caching.predict_quality(str(photo), {"x": 1})

    assert base.count == 2

