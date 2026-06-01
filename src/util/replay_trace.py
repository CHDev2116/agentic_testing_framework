import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


class ReplayTraceError(ValueError):
    """Raised when replay trace content is invalid or mismatched."""


def planner_input_hash(payload: Dict[str, Any]) -> str:
    """Return a stable hash for planner input payload."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def append_replay_step(trace_path: str, step: Dict[str, Any]) -> None:
    """Append one replay step as JSONL after schema validation."""
    _validate_step(step)
    path = Path(trace_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(step, ensure_ascii=True))
        f.write("\n")


def load_replay_steps(trace_path: str) -> List[Dict[str, Any]]:
    """Load replay steps from JSONL with strict per-line validation."""
    path = Path(trace_path)
    if not path.exists():
        raise ReplayTraceError(f"Replay trace file not found: {trace_path}")

    steps: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReplayTraceError(
                    f"Invalid JSON at line {line_number} in replay trace: {exc}"
                ) from exc
            if not isinstance(parsed, dict):
                raise ReplayTraceError(
                    f"Replay trace line {line_number} must be a JSON object."
                )
            _validate_step(parsed)
            steps.append(parsed)
    return steps


def build_replay_index(steps: List[Dict[str, Any]]) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """
    Build lookup index by (image_path, attempt) and enforce uniqueness.
    """
    index: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for step in steps:
        key = (str(step["image_path"]), int(step["attempt"]))
        if key in index:
            raise ReplayTraceError(
                f"Duplicate replay step for image_path={key[0]!r}, attempt={key[1]}."
            )
        index[key] = step
    return index


def get_replay_step(
    replay_index: Dict[Tuple[str, int], Dict[str, Any]],
    *,
    image_path: str,
    attempt: int,
    expected_planner_input_hash: str,
) -> Dict[str, Any]:
    """
    Fetch replay step and ensure planner input hash matches expected runtime hash.
    """
    key = (str(image_path), int(attempt))
    step = replay_index.get(key)
    if step is None:
        raise ReplayTraceError(
            f"Missing replay step for image_path={image_path!r}, attempt={attempt}."
        )
    actual_hash = str(step.get("planner_input_hash", ""))
    if actual_hash != expected_planner_input_hash:
        raise ReplayTraceError(
            "Replay planner input hash mismatch for "
            f"image_path={image_path!r}, attempt={attempt}: "
            f"expected={expected_planner_input_hash}, trace={actual_hash}"
        )
    return step


def _validate_step(step: Dict[str, Any]) -> None:
    required_fields = [
        "image_path",
        "attempt",
        "metrics_before",
        "planner_input_hash",
        "planner_output",
        "action",
        "signal",
        "backend",
        "latency_ms",
        "stop_reason",
        "timestamp",
    ]
    missing = [field for field in required_fields if field not in step]
    if missing:
        raise ReplayTraceError(f"Replay step missing required fields: {missing}")

    if not isinstance(step["image_path"], str) or not step["image_path"]:
        raise ReplayTraceError("Replay step field 'image_path' must be a non-empty string.")
    if not isinstance(step["attempt"], int) or step["attempt"] < 1:
        raise ReplayTraceError("Replay step field 'attempt' must be integer >= 1.")
    if not isinstance(step["metrics_before"], dict):
        raise ReplayTraceError("Replay step field 'metrics_before' must be a dict.")
    if not isinstance(step["planner_output"], dict):
        raise ReplayTraceError("Replay step field 'planner_output' must be a dict.")
    if not isinstance(step["planner_input_hash"], str) or not step["planner_input_hash"]:
        raise ReplayTraceError(
            "Replay step field 'planner_input_hash' must be a non-empty string."
        )
    if not isinstance(step["action"], str) or not step["action"]:
        raise ReplayTraceError("Replay step field 'action' must be a non-empty string.")
    if not isinstance(step["signal"], str):
        raise ReplayTraceError("Replay step field 'signal' must be a string.")
    if not isinstance(step["backend"], str) or not step["backend"]:
        raise ReplayTraceError("Replay step field 'backend' must be a non-empty string.")
    if not isinstance(step["latency_ms"], (int, float)):
        raise ReplayTraceError("Replay step field 'latency_ms' must be numeric.")
    if not isinstance(step["stop_reason"], str):
        raise ReplayTraceError("Replay step field 'stop_reason' must be a string.")
    if not isinstance(step["timestamp"], str) or not step["timestamp"]:
        raise ReplayTraceError("Replay step field 'timestamp' must be a non-empty string.")
