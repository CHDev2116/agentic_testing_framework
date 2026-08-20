import hashlib
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


def _stable_json_dumps(obj: Any) -> str:
    """
    Deterministic JSON encoding for hashing.

    Note: we use sort_keys=True and compact separators to avoid whitespace noise.
    """

    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def stable_json_sha256(obj: Any) -> str:
    payload = _stable_json_dumps(obj).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str, *, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class InferenceCacheSettings:
    enabled: bool = False
    cache_dir: str = ".cache/inference"
    version_tag: str = "v1"


class InferenceCache:
    """
    Simple file-based cache:
      key -> <cache_dir>/<key>.json
    """

    def __init__(self, settings: InferenceCacheSettings, *, logger_: logging.Logger):
        self.settings = settings
        self.logger = logger_

    @property
    def cache_dir_path(self) -> Path:
        return Path(self.settings.cache_dir)

    def _key_to_path(self, key: str) -> Path:
        return self.cache_dir_path / f"{key}.json"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        path = self._key_to_path(key)
        if not path.exists():
            return None

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            value = payload.get("value")
            if isinstance(value, dict):
                return value
        except Exception:
            self.logger.debug("InferenceCache get failed (key=%s)", key, exc_info=True)
        return None

    def put(self, key: str, value: Dict[str, Any], *, meta: Dict[str, Any]) -> None:
        if not self.settings.enabled:
            return

        self.cache_dir_path.mkdir(parents=True, exist_ok=True)

        final_path = self._key_to_path(key)
        payload = {
            "version": self.settings.version_tag,
            "value": value,
            "meta": meta,
        }

        # Atomic write: temp file + os.replace
        tmp_dir = final_path.parent
        fd, tmp_path_str = tempfile.mkstemp(prefix=f".{key}.", suffix=".tmp", dir=str(tmp_dir))
        tmp_path = Path(tmp_path_str)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
            os.replace(str(tmp_path), str(final_path))
        finally:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                # Best-effort cleanup; don't break inference.
                pass


class CachingInferenceEngine:
    """
    Wraps an inference engine and caches its *normalized* output dict.

    Important invariants:
    - cache is bypassed when replay_mode != "off" (to keep deterministic truth)
    - cache keys include backend_id + rules_tag + metrics_hash + photo_hash
    """

    def __init__(
        self,
        *,
        base_engine: Any,
        cache: InferenceCache,
        cache_context: Dict[str, Any],
        logger_: logging.Logger,
    ):
        self.base_engine = base_engine
        self.cache = cache
        self.cache_context = cache_context
        self.logger = logger_

    def _build_key(self, *, photo_hash: str, metrics_hash: str) -> str:
        backend_id = str(self.cache_context.get("backend_id", "unknown_backend"))
        rules_tag = str(self.cache_context.get("rules_tag", "rules_v0"))
        version_tag = str(self.cache_context.get("version_tag", self.cache.settings.version_tag))
        raw = f"{version_tag}|{backend_id}|{rules_tag}|{photo_hash}|{metrics_hash}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _should_bypass(self) -> bool:
        replay_mode = str(self.cache_context.get("replay_mode", "off")).lower()
        if replay_mode != "off":
            return True
        return not bool(self.cache.settings.enabled)

    def predict_quality(self, photo_path: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        if self._should_bypass():
            return self.base_engine.predict_quality(photo_path, metrics)

        try:
            photo_hash = sha256_file(photo_path)
            metrics_hash = stable_json_sha256(metrics)
            key = self._build_key(photo_hash=photo_hash, metrics_hash=metrics_hash)
        except Exception:
            # Cache is a dev optimization; never break inference.
            self.logger.debug("InferenceCache key build failed", exc_info=True)
            return self.base_engine.predict_quality(photo_path, metrics)

        cached = self.cache.get(key)
        if cached is not None:
            self.logger.debug("InferenceCache hit (key=%s)", key)
            return cached

        self.logger.debug("InferenceCache miss (key=%s)", key)
        result = self.base_engine.predict_quality(photo_path, metrics)

        if isinstance(result, dict):
            meta = {
                "created_at_unix": time.time(),
                "photo_hash": photo_hash,
                "metrics_hash": metrics_hash,
                "backend_id": self.cache_context.get("backend_id"),
                "rules_tag": self.cache_context.get("rules_tag"),
            }
            try:
                self.cache.put(key, result, meta=meta)
            except Exception:
                # Best-effort cache put; never break inference.
                self.logger.debug("InferenceCache put failed (key=%s)", key, exc_info=True)

        return result

