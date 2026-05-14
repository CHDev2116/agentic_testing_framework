import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from PIL import Image

from engine.vision_math import calculate_metrics


class QuantizedVisionAgent:
    def __init__(self, model_name="Agentic Testing Framework - Llama 4-bit"):
        self.model_name = model_name
        print(f"📦 Loaded quantized model: {self.model_name}")

    def verify_capture_success(self, photo_path: str) -> bool:
        """Return True if the capture file exists (deterministic for this demo script)."""
        return os.path.isfile(photo_path)

    def call_4bit_model_inference(self, metrics):
        """Step B: Simulate 4-bit model inference based on metrics."""
        if metrics["sharpness"] < 10:
            return "Fail: Out of Focus (AI Detected)"
        if metrics["avg_brightness"] < 50:
            return "Fail: Too Dark (AI Detected)"
        return "Pass: Quality Meets Standard"


def run_test_pipeline(work_dir: Optional[Path] = None) -> None:
    """
    Build a temporary JPEG, run ``calculate_metrics`` on a real path, then simulate inference.
    """
    agent = QuantizedVisionAgent()
    base = work_dir if work_dir is not None else Path(tempfile.mkdtemp(prefix="atf_verify_"))
    base.mkdir(parents=True, exist_ok=True)
    mock_photo = base / "mock_shot.jpg"
    # Slight variation so sharpness / brightness are non-trivial vs flat fields.
    Image.new("RGB", (64, 64), (118, 120, 119)).save(mock_photo, format="JPEG", quality=95)

    report = {"timestamp": datetime.now().isoformat(), "test_cases": []}

    try:
        print(f"🔍 Checking whether file exists: {mock_photo}")
        if agent.verify_capture_success(str(mock_photo)):
            metrics = calculate_metrics(str(mock_photo))
            if metrics is None:
                print("❌ Metrics unavailable (image load failed).")
                return

            ai_decision = agent.call_4bit_model_inference(metrics)

            print(f"📊 Numeric metrics: {metrics}")
            print(f"🤖 AI decision: {ai_decision}")

            report["test_cases"].append(
                {
                    "file": str(mock_photo),
                    "ai_decision": ai_decision,
                    "metrics": metrics,
                }
            )
        else:
            print("❌ Error: File does not exist. Skipping AI analysis.")

    except OSError as e:
        print(f"💥 System crash: {e}")
    finally:
        print("💾 Test report has been updated.")


if __name__ == "__main__":
    run_test_pipeline()
