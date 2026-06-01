#!/usr/bin/env python3
"""Record .ci/replay_smoke_trace.jsonl from tests/fixtures/replay_smoke."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "replay_smoke"
TRACE_PATH = ROOT / ".ci" / "replay_smoke_trace.jsonl"
IMAGE_PATH = FIXTURE_DIR / "underexposed.png"


def main() -> int:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if TRACE_PATH.exists():
        TRACE_PATH.unlink()

    Image.new("L", (48, 48), color=18).save(IMAGE_PATH)

    cmd = [
        sys.executable,
        str(ROOT / "src" / "ai_quality_agent.py"),
        "--config",
        str(ROOT / "configs" / "replay_smoke.json"),
        "--inference-backend",
        "simulated",
        "--loopback-planner",
        "simulated",
        "--replay-mode",
        "record",
        "--replay-file",
        str(TRACE_PATH),
    ]
    env = {**dict(**__import__("os").environ), "PYTHONPATH": str(ROOT / "src")}
    print("[replay-fixture] recording:", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)

    if not TRACE_PATH.exists() or TRACE_PATH.stat().st_size == 0:
        print("[replay-fixture] FAIL: trace file missing or empty", file=sys.stderr)
        return 1

    print(f"[replay-fixture] OK: wrote {TRACE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
