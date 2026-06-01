import argparse
import asyncio
import json
import logging
import os
import random
import socket
import threading
import time
import traceback
from urllib.parse import urlparse
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from statistics import pvariance
from typing import Any, Dict, List, Optional

import httpx
from PIL import Image, ImageDraw, ImageEnhance
import psutil

from util.cli_logging import configure_cli_logging
from util.failure_memory import FailureMemoryStore
from util.metrics_pool import MetricsProcessPool
from util.monitor_performance import async_monitor_performance, gather_with_timing
from agent.loopback_planner import (
    SimulatedLoopbackPlanner,
    create_loopback_planner,
)
from engine.image_processor import ImageProcessor
from engine.vision_math import calculate_metrics
from eval.arbitrator import (
    aggregate_batch_decisions,
    arbitrate_decision,
    merge_gate_and_arbitration,
)
from eval.benchmark_evaluator import (
    build_rankings,
    generate_benchmark_insights,
    get_release_decision,
)
from models.async_inference import predict_quality_async
from models.contracts import AgentInferenceOutput, AgentStep
from models.contracts import LoopbackPlan
from models.inference_adapter import build_inference_engine
from util.replay_trace import (
    append_replay_step,
    build_replay_index,
    get_replay_step,
    load_replay_steps,
    planner_input_hash,
)

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "model_settings": {"name": "Default-Model", "bit_depth": 4},
    "thresholds": {"min_sharpness": 20, "min_brightness": 45, "max_brightness": 220},
    "folders": {"input": "test_images", "output": "results"},
    "runtime": {"oom_probability": 0.0, "max_retry": 3},
    "quality_gate": {"target_pass_rate": 90.0},
    "eval_settings": {
        "conflict_strategy": "conservative",
        "weights": {"engine_weight": 0.6, "model_weight": 0.4},
        "auto_tag_conflicts": True,
    },
}
REPORT_RETENTION_DAYS = 14


def deep_merge(base_config, override_config):
    result = dict(base_config)
    for key, value in override_config.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_json_file(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_config_path(config_path, project_root):
    if os.path.isabs(config_path):
        return config_path

    project_relative = os.path.join(project_root, config_path)
    if os.path.exists(project_relative):
        return project_relative

    return os.path.abspath(config_path)


def cleanup_old_reports(output_folder_path, max_age_days=REPORT_RETENTION_DAYS):
    cutoff_timestamp = time.time() - (max_age_days * 24 * 60 * 60)
    deleted_count = 0

    for file_name in os.listdir(output_folder_path):
        if not (file_name.startswith("batch_report_") and file_name.endswith(".json")):
            continue

        file_path = os.path.join(output_folder_path, file_name)
        if not os.path.isfile(file_path):
            continue

        try:
            if os.path.getmtime(file_path) < cutoff_timestamp:
                os.remove(file_path)
                deleted_count += 1
        except OSError as e:
            logger.warning("Could not remove old report %s: %s", file_path, e)

    return deleted_count


def count_reports(output_folder_path):
    report_count = 0
    for file_name in os.listdir(output_folder_path):
        if file_name.startswith("batch_report_") and file_name.endswith(".json"):
            file_path = os.path.join(output_folder_path, file_name)
            if os.path.isfile(file_path):
                report_count += 1
    return report_count


def ensure_sample_images(image_folder_path):
    valid_extensions = (".jpg", ".jpeg", ".png")
    existing_images = [
        file_name for file_name in os.listdir(image_folder_path)
        if file_name.lower().endswith(valid_extensions)
        and os.path.isfile(os.path.join(image_folder_path, file_name))
    ]
    if existing_images:
        return

    sample_specs = [
        ("sample_good.png", 128, 18),
        ("sample_dark.png", 25, 8),
        ("sample_bright.png", 240, 8)
    ]
    for file_name, base_brightness, step in sample_specs:
        image = Image.new("L", (128, 128), color=base_brightness)
        draw = ImageDraw.Draw(image)
        for idx in range(0, 128, step):
            intensity = min(255, max(0, base_brightness + ((idx % 32) - 16) * 2))
            draw.line((0, idx, 127, idx), fill=intensity)
            draw.line((idx, 0, idx, 127), fill=255 - intensity)
        image.save(os.path.join(image_folder_path, file_name))

    logger.info("No input images found. Generated sample dataset in input folder.")


def ensure_stress_test_images(image_folder_path, target_count=100):
    valid_extensions = (".jpg", ".jpeg", ".png")
    existing_images = [
        os.path.join(image_folder_path, file_name)
        for file_name in os.listdir(image_folder_path)
        if file_name.lower().endswith(valid_extensions)
        and os.path.isfile(os.path.join(image_folder_path, file_name))
    ]
    if len(existing_images) >= target_count:
        return

    if not existing_images:
        ensure_sample_images(image_folder_path)
        existing_images = [
            os.path.join(image_folder_path, file_name)
            for file_name in os.listdir(image_folder_path)
            if file_name.lower().endswith(valid_extensions)
            and os.path.isfile(os.path.join(image_folder_path, file_name))
        ]
    if not existing_images:
        return

    needed = target_count - len(existing_images)
    for idx in range(needed):
        source_path = existing_images[idx % len(existing_images)]
        source_name = Path(source_path).stem
        with Image.open(source_path) as src:
            rgb_img = src.convert("RGB")
            scale = 1.0 + ((idx % 5) * 0.1)
            new_w = max(64, int(rgb_img.width * scale))
            new_h = max(64, int(rgb_img.height * scale))
            resized = rgb_img.resize((new_w, new_h))
            brightness_factor = 0.75 + ((idx % 7) * 0.08)
            enhancer = ImageEnhance.Brightness(resized).enhance(brightness_factor)
            out_name = f"stress_{source_name}_{idx + 1:03d}.jpg"
            enhancer.save(os.path.join(image_folder_path, out_name), quality=90)

    logger.info("Stress-test mode: ensured at least %s images in input folder.", target_count)


def load_config(profile="dev", config_path=None):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    configs_dir = os.path.join(project_root, "configs")
    base_config_path = os.path.join(configs_dir, "base.json")
    legacy_config_path = os.path.join(project_root, "config.json")

    merged_config = dict(DEFAULT_CONFIG)
    config_sources = []

    if os.path.exists(base_config_path):
        merged_config = deep_merge(merged_config, load_json_file(base_config_path))
        config_sources.append(base_config_path)

    if config_path:
        selected_config_path = resolve_config_path(config_path, project_root)
        if not os.path.exists(selected_config_path):
            raise FileNotFoundError(f"Specified config file not found: {selected_config_path}")
        merged_config = deep_merge(merged_config, load_json_file(selected_config_path))
        config_sources.append(selected_config_path)
        return merged_config, " + ".join(config_sources)

    profile_config_path = os.path.join(configs_dir, f"{profile}.json")
    if profile != "base" and os.path.exists(profile_config_path):
        merged_config = deep_merge(merged_config, load_json_file(profile_config_path))
        config_sources.append(profile_config_path)
        return merged_config, " + ".join(config_sources)

    if profile == "base" and os.path.exists(base_config_path):
        return merged_config, " + ".join(config_sources)

    if os.path.exists(legacy_config_path):
        merged_config = deep_merge(merged_config, load_json_file(legacy_config_path))
        config_sources.append(legacy_config_path)
        return merged_config, " + ".join(config_sources)

    return merged_config, "DEFAULT_CONFIG"


def _apply_runtime_overrides(
    config: Dict[str, Any],
    config_source: str,
    *,
    inference_backend_override: Optional[str] = None,
    loopback_planner_override: Optional[str] = None,
    planner_timeout_s_override: Optional[float] = None,
    planner_model_override: Optional[str] = None,
    planner_require_healthy_override: Optional[bool] = None,
    async_per_image_timeout_s_override: Optional[float] = None,
    async_backend_health_check_override: Optional[bool] = None,
    async_backend_health_timeout_s_override: Optional[float] = None,
    replay_mode_override: Optional[str] = None,
    replay_file_override: Optional[str] = None,
):
    source = config_source
    if inference_backend_override:
        config.setdefault("model_settings", {}).setdefault("inference", {})
        config["model_settings"]["inference"]["backend"] = inference_backend_override
        source = f"{source} + CLI(backend={inference_backend_override})"
    if loopback_planner_override:
        config.setdefault("runtime", {}).setdefault("loopback_planner", {})
        config["runtime"]["loopback_planner"]["mode"] = loopback_planner_override
        source = f"{source} + CLI(loopback_planner={loopback_planner_override})"
    if planner_timeout_s_override is not None:
        config.setdefault("runtime", {}).setdefault("loopback_planner", {}).setdefault("llm", {})
        config["runtime"]["loopback_planner"]["llm"]["timeout_s"] = float(planner_timeout_s_override)
        source = f"{source} + CLI(planner_timeout_s={planner_timeout_s_override})"
    if planner_model_override:
        config.setdefault("runtime", {}).setdefault("loopback_planner", {}).setdefault("llm", {})
        config["runtime"]["loopback_planner"]["llm"]["model"] = planner_model_override
        source = f"{source} + CLI(planner_model={planner_model_override})"
    if planner_require_healthy_override is not None:
        config.setdefault("runtime", {}).setdefault("loopback_planner", {})
        config["runtime"]["loopback_planner"]["require_healthy_on_startup"] = bool(
            planner_require_healthy_override
        )
        source = (
            f"{source} + CLI(planner_require_healthy={planner_require_healthy_override})"
        )
    if async_per_image_timeout_s_override is not None:
        config.setdefault("runtime", {})
        config["runtime"]["async_per_image_timeout_s"] = float(
            async_per_image_timeout_s_override
        )
        source = (
            f"{source} + CLI(async_per_image_timeout_s={async_per_image_timeout_s_override})"
        )
    if async_backend_health_check_override is not None:
        config.setdefault("runtime", {})
        config["runtime"]["async_backend_health_check"] = bool(
            async_backend_health_check_override
        )
        source = (
            f"{source} + CLI(async_backend_health_check={async_backend_health_check_override})"
        )
    if async_backend_health_timeout_s_override is not None:
        config.setdefault("runtime", {})
        config["runtime"]["async_backend_health_timeout_s"] = float(
            async_backend_health_timeout_s_override
        )
        source = (
            f"{source} + CLI(async_backend_health_timeout_s={async_backend_health_timeout_s_override})"
        )
    if replay_mode_override is not None:
        config.setdefault("runtime", {})
        config["runtime"]["replay_mode"] = str(replay_mode_override)
        source = f"{source} + CLI(replay_mode={replay_mode_override})"
    if replay_file_override is not None:
        config.setdefault("runtime", {})
        config["runtime"]["replay_file"] = str(replay_file_override)
        source = f"{source} + CLI(replay_file={replay_file_override})"
    return source


def _extract_host_port_for_healthcheck(engine: Any) -> Optional[tuple[str, int]]:
    backend = str(getattr(engine, "backend_name", ""))
    if backend == "simulated":
        return None
    if backend in {"llama_cpp", "ollama_vision"}:
        base_url = str(getattr(engine, "host", ""))
    elif backend == "mock_api":
        base_url = str(getattr(engine, "url", ""))
    else:
        return None

    parsed = urlparse(base_url if "://" in base_url else f"http://{base_url}")
    if not parsed.hostname:
        return None
    if parsed.port is not None:
        return parsed.hostname, int(parsed.port)
    return parsed.hostname, 443 if parsed.scheme == "https" else 80


def _run_async_backend_health_check(
    engine: Any,
    *,
    enabled: bool,
    timeout_s: float,
) -> None:
    if not enabled:
        logger.info("Async backend health check: skipped by configuration")
        return

    host_port = _extract_host_port_for_healthcheck(engine)
    if host_port is None:
        logger.info(
            "Async backend health check: skipped for backend=%s",
            getattr(engine, "backend_name", type(engine).__name__),
        )
        return

    host, port = host_port
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            logger.info(
                "Async backend health check: reachable backend=%s host=%s port=%s timeout_s=%.2f",
                getattr(engine, "backend_name", type(engine).__name__),
                host,
                port,
                timeout_s,
            )
    except OSError as exc:
        raise RuntimeError(
            f"Async inference backend not reachable before batch run: "
            f"backend={getattr(engine, 'backend_name', type(engine).__name__)} "
            f"host={host} port={port} timeout_s={timeout_s}. "
            "Start the backend service or use --async-skip-backend-health-check."
        ) from exc


class QuantizedVisionAgent:
    def __init__(self, config, metrics_pool: Optional[MetricsProcessPool] = None):
        self.config = config
        self.model_info = config["model_settings"]
        self.inference_engine = build_inference_engine(config)
        self.metrics_pool = metrics_pool
        self.oom_probability = float(config.get("runtime", {}).get("oom_probability", 0.0))
        logger.info(
            "Startup mode: %s (%s-bit)",
            self.model_info["name"],
            self.model_info["bit_depth"],
        )
        logger.info("Inference backend: %s", self.inference_engine.backend_name)
        if metrics_pool is not None:
            logger.info(
                "Metrics compute: process pool (max_workers=%s)",
                metrics_pool.max_workers,
            )

    def _compute_metrics(self, photo_path: str):
        if self.metrics_pool is not None:
            return self.metrics_pool.calculate(photo_path)
        return calculate_metrics(photo_path)

    def get_all_photos(self):
        folder_name = self.config["folders"]["input"]
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(current_dir)
        full_path = os.path.join(base_dir, folder_name)

        if not os.path.exists(full_path):
            os.makedirs(full_path)
        ensure_sample_images(full_path)

        valid_extensions = (".jpg", ".jpeg", ".png")
        return [
            os.path.join(full_path, f)
            for f in sorted(os.listdir(full_path))
            if f.lower().endswith(valid_extensions)
        ]

    def analyze_photo_quality(self, photo_path):
        start_time = time.time()
        metrics = self._compute_metrics(photo_path)
        if metrics is None:
            return None, {"decision": "Error", "code": "ERR_SYS_IO_404", "msg": "Unable to read file"}, 0

        ai_result = self.inference_engine.predict_quality(photo_path, metrics)
        latency = round((time.time() - start_time) * 1000, 2)
        return metrics, ai_result, latency

    @async_monitor_performance
    async def analyze_photo_quality_async(
        self, photo_path: str, http_client: httpx.AsyncClient
    ):
        logger.debug(
            "analyze_photo_quality_async: path=%s backend=%s",
            photo_path,
            self.inference_engine.backend_name,
        )
        start_time = time.time()
        if self.metrics_pool is not None:
            loop = asyncio.get_running_loop()
            metrics = await loop.run_in_executor(
                self.metrics_pool.executor,
                calculate_metrics,
                photo_path,
            )
        else:
            metrics = await asyncio.to_thread(calculate_metrics, photo_path)
        if metrics is None:
            logger.warning("analyze_photo_quality_async: unable to read %s", photo_path)
            return None, {"decision": "Error", "code": "ERR_SYS_IO_404", "msg": "Unable to read file"}, 0

        ai_result = await predict_quality_async(
            self.inference_engine, http_client, photo_path, metrics
        )
        latency = round((time.time() - start_time) * 1000, 2)
        logger.debug(
            "analyze_photo_quality_async: done path=%s decision=%s latency_ms=%.2f",
            photo_path,
            ai_result.get("decision"),
            latency,
        )
        return metrics, ai_result, latency


def save_batch_report(report_data, output_folder):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)
    full_output_path = os.path.join(base_dir, output_folder)

    if not os.path.exists(full_output_path):
        os.makedirs(full_output_path)

    file_name = f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
    file_path = os.path.join(full_output_path, file_name)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=4, ensure_ascii=False)
    logger.info("Batch test completed. Full report saved to: %s", file_path)

    deleted_count = cleanup_old_reports(full_output_path)
    if deleted_count > 0:
        logger.info(
            "Cleaned up %s report(s) older than %s days.",
            deleted_count,
            REPORT_RETENTION_DAYS,
        )
    current_report_count = count_reports(full_output_path)
    logger.info("Current report count: %s", current_report_count)
    return file_path


def save_comparison_report(comparison_data, output_folder):
    comparison_dir = Path(output_folder) / "comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    file_path = comparison_dir / f"profile_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, indent=4, ensure_ascii=False)
    logger.info("Comparison report saved to: %s", file_path)
    return str(file_path)


def save_repeatability_report(repeatability_data, output_folder):
    repeatability_dir = Path(output_folder) / "repeatability"
    repeatability_dir.mkdir(parents=True, exist_ok=True)
    file_path = repeatability_dir / f"repeatability_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(repeatability_data, f, indent=4, ensure_ascii=False)
    logger.info("Repeatability report saved to: %s", file_path)
    return str(file_path)


def save_performance_report(performance_data, output_folder):
    performance_dir = Path(output_folder) / "performance"
    performance_dir.mkdir(parents=True, exist_ok=True)
    file_path = performance_dir / f"performance_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(performance_data, f, indent=4, ensure_ascii=False)
    logger.info("Performance report saved to: %s", file_path)
    return str(file_path)


def save_overhead_report(overhead_data, output_folder):
    overhead_dir = Path(output_folder) / "overhead"
    overhead_dir.mkdir(parents=True, exist_ok=True)
    file_path = overhead_dir / f"overhead_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(overhead_data, f, indent=4, ensure_ascii=False)
    logger.info("Overhead report saved to: %s", file_path)
    return str(file_path)


def save_error_report(error_data, output_folder):
    error_dir = Path(output_folder)
    error_dir.mkdir(parents=True, exist_ok=True)
    file_path = error_dir / f"error_report_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(error_data, f, indent=4, ensure_ascii=False)
    logger.info("Error report saved to: %s", file_path)
    deleted_count = cleanup_old_error_reports(str(error_dir))
    if deleted_count > 0:
        logger.info(
            "Cleaned up %s error report(s) older than %s days.",
            deleted_count,
            REPORT_RETENTION_DAYS,
        )
    return str(file_path)


def cleanup_old_error_reports(error_folder_path, max_age_days=REPORT_RETENTION_DAYS):
    cutoff_timestamp = time.time() - (max_age_days * 24 * 60 * 60)
    deleted_count = 0

    for file_name in os.listdir(error_folder_path):
        if not (file_name.startswith("error_report_") and file_name.endswith(".json")):
            continue

        file_path = os.path.join(error_folder_path, file_name)
        if not os.path.isfile(file_path):
            continue

        try:
            if os.path.getmtime(file_path) < cutoff_timestamp:
                os.remove(file_path)
                deleted_count += 1
        except OSError as e:
            logger.warning("Could not remove old error report %s: %s", file_path, e)

    return deleted_count


def get_image_metadata(photo_path):
    file_size_bytes = os.path.getsize(photo_path)
    with Image.open(photo_path) as image:
        width, height = image.size
    return {
        "width": int(width),
        "height": int(height),
        "pixel_count": int(width * height),
        "file_size_kb": round(file_size_bytes / 1024.0, 4),
    }


def pearson_correlation(xs, ys):
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    centered_x = [x - mean_x for x in xs]
    centered_y = [y - mean_y for y in ys]
    numerator = sum(x * y for x, y in zip(centered_x, centered_y))
    denom_x = sum(x * x for x in centered_x) ** 0.5
    denom_y = sum(y * y for y in centered_y) ** 0.5
    if denom_x == 0 or denom_y == 0:
        return 0.0
    return round(numerator / (denom_x * denom_y), 6)


def classify_exposure_signal(ai_result):
    decision_text = str((ai_result or {}).get("decision", "")).lower()
    msg_text = str((ai_result or {}).get("msg", "")).lower()
    merged = f"{decision_text} {msg_text}"
    if "under-exposed" in merged or "too dark" in merged or "dark" in merged:
        return "under"
    if "over-exposed" in merged or "too bright" in merged or "bright" in merged:
        return "over"
    return "other"


def classify_loopback_signal(ai_result):
    exposure_signal = classify_exposure_signal(ai_result)
    if exposure_signal in {"under", "over"}:
        return exposure_signal
    decision_text = str((ai_result or {}).get("decision", "")).lower()
    msg_text = str((ai_result or {}).get("msg", "")).lower()
    merged = f"{decision_text} {msg_text}"
    if "blurry" in merged or "out-of-focus" in merged or "sharpness" in merged:
        return "blurry"
    return "other"


def decide_loopback_action(signal, engine_metrics, thresholds_cfg, loopback_guard_cfg):
    planner = SimulatedLoopbackPlanner()
    plan = planner.plan(
        signal=signal,
        engine_metrics=engine_metrics,
        thresholds_cfg=thresholds_cfg,
        loopback_guard_cfg=loopback_guard_cfg,
        attempt_history=[],
    )
    return plan.action, plan.stop_reason


def plan_next_action(
    *,
    signal: str,
    engine_metrics: Dict[str, Any],
    thresholds_cfg: Dict[str, Any],
    loopback_guard_cfg: Dict[str, Any],
):
    planner = SimulatedLoopbackPlanner()
    return planner.plan(
        signal=signal,
        engine_metrics=engine_metrics,
        thresholds_cfg=thresholds_cfg,
        loopback_guard_cfg=loopback_guard_cfg,
        attempt_history=[],
    )


def summarize_performance(perf_samples):
    if not perf_samples:
        return {}

    sizes_kb = [item["file_size_kb"] for item in perf_samples]
    latencies_ms = [item["latency_ms"] for item in perf_samples]
    cpu_usages = [item["process_cpu_usage_pct"] for item in perf_samples]
    pixels = [item["pixel_count"] for item in perf_samples]
    peak_cpu_usages = [item.get("peak_cpu_usage_pct", 0.0) for item in perf_samples]
    peak_memory_mb = [item.get("peak_memory_mb", 0.0) for item in perf_samples]

    sorted_latencies = sorted(latencies_ms)
    p95_index = min(len(sorted_latencies) - 1, int(0.95 * (len(sorted_latencies) - 1)))
    p99_index = min(len(sorted_latencies) - 1, int(0.99 * (len(sorted_latencies) - 1)))

    latency_per_mb_slope = 0.0
    peak_cpu_per_mb_slope = 0.0
    if len(sizes_kb) >= 2:
        delta_size_mb = (max(sizes_kb) - min(sizes_kb)) / 1024.0
        if delta_size_mb > 0:
            latency_per_mb_slope = round((max(latencies_ms) - min(latencies_ms)) / delta_size_mb, 4)
            peak_cpu_per_mb_slope = round((max(peak_cpu_usages) - min(peak_cpu_usages)) / delta_size_mb, 4)

    return {
        "sample_count": len(perf_samples),
        "avg_latency_ms": round(sum(latencies_ms) / len(latencies_ms), 4),
        "avg_process_cpu_usage_pct": round(sum(cpu_usages) / len(cpu_usages), 4),
        "max_peak_cpu_usage_pct": round(max(peak_cpu_usages), 4),
        "max_peak_memory_mb": round(max(peak_memory_mb), 4),
        "tail_latency_p95_ms": round(sorted_latencies[p95_index], 4),
        "tail_latency_p99_ms": round(sorted_latencies[p99_index], 4),
        "latency_vs_file_size_corr": pearson_correlation(sizes_kb, latencies_ms),
        "latency_vs_pixel_count_corr": pearson_correlation(pixels, latencies_ms),
        "peak_cpu_vs_file_size_corr": pearson_correlation(sizes_kb, peak_cpu_usages),
        "peak_cpu_vs_latency_corr": pearson_correlation(latencies_ms, peak_cpu_usages),
        "peak_memory_vs_file_size_corr": pearson_correlation(sizes_kb, peak_memory_mb),
        "latency_increase_ms_per_1mb": latency_per_mb_slope,
        "peak_cpu_increase_pct_per_1mb": peak_cpu_per_mb_slope,
        "correlation_matrix": {
            "file_size_kb_vs_latency_ms": pearson_correlation(sizes_kb, latencies_ms),
            "file_size_kb_vs_peak_cpu_pct": pearson_correlation(sizes_kb, peak_cpu_usages),
            "pixel_count_vs_latency_ms": pearson_correlation(pixels, latencies_ms),
            "pixel_count_vs_peak_cpu_pct": pearson_correlation(pixels, peak_cpu_usages),
            "latency_ms_vs_peak_cpu_pct": pearson_correlation(latencies_ms, peak_cpu_usages),
        },
        "auto_insight": (
            f"圖片檔案大小每增加 1MB，推論延遲平均增加 {latency_per_mb_slope}ms，"
            f"CPU 峰值提升 {peak_cpu_per_mb_slope}% 。"
        ),
        "insight_confidence": (
            "high" if len(perf_samples) >= 100 else "low (recommend >=100 images for stable trend)"
        ),
    }


def _build_agent_inference_output(
    *,
    image_path: str,
    attempt_history: List[Dict[str, Any]],
    final_ai_result: Dict[str, Any],
    total_latency_ms: float,
) -> Dict[str, Any]:
    steps: List[AgentStep] = []
    for idx, attempt in enumerate(attempt_history):
        signal = str(attempt.get("loopback_signal", "other")).lower()
        if signal not in {"under", "over", "blurry", "other"}:
            signal = "other"
        action = str(attempt.get("action", "stop")).lower()
        if action not in {"brighten", "dim", "sharpen", "stop"}:
            action = "stop"
        metrics_before = {
            "avg_brightness": float(attempt.get("avg_brightness", 0.0)),
            "sharpness": float(attempt.get("sharpness", 0.0)),
        }
        metrics_after = None
        if idx + 1 < len(attempt_history):
            next_attempt = attempt_history[idx + 1]
            metrics_after = {
                "avg_brightness": float(next_attempt.get("avg_brightness", 0.0)),
                "sharpness": float(next_attempt.get("sharpness", 0.0)),
            }
        steps.append(
            AgentStep(
                attempt=int(attempt.get("attempt", idx + 1)),
                signal=signal,
                action=action,
                rationale=str(attempt.get("rationale", "")),
                fallback_used=bool(attempt.get("planner_fallback_used", False)),
                metrics_before=metrics_before,
                metrics_after=metrics_after,
                latency_ms=float(attempt.get("latency_ms", 0.0)),
            )
        )

    final_release = "NO_GO"
    if attempt_history:
        release = str(attempt_history[-1].get("release", "NO_GO")).upper()
        if release in {"GO", "REVIEW", "NO_GO"}:
            final_release = release

    output = AgentInferenceOutput(
        image_path=image_path,
        final_decision=final_release,
        error_code=str(final_ai_result.get("code", "SUCCESS_200")),
        error_message=str(final_ai_result.get("msg", final_ai_result.get("decision", "Optimal"))),
        steps=steps,
        total_latency_ms=float(total_latency_ms),
    )
    return output.model_dump()


def monitor_resources(stop_event, interval=0.1):
    cpu_usage = []
    memory_usage_mb = []
    process = psutil.Process(os.getpid())
    cpu_available = True
    try:
        process.cpu_percent(interval=None)
    except Exception:
        cpu_available = False

    while not stop_event.is_set():
        if cpu_available:
            try:
                cpu_usage.append(process.cpu_percent(interval=None))
            except Exception:
                cpu_available = False
                cpu_usage.append(0.0)
        else:
            cpu_usage.append(0.0)
        memory_usage_mb.append(process.memory_info().rss / (1024.0 * 1024.0))
        if stop_event.wait(interval):
            break

    return {
        "peak_cpu_usage_pct": round(max(cpu_usage), 4) if cpu_usage else 0.0,
        "peak_memory_mb": round(max(memory_usage_mb), 4) if memory_usage_mb else 0.0,
    }


def collect_peak_resources_during(fn, *args, **kwargs):
    stop_event = threading.Event()
    monitor_result = {"peak_cpu_usage_pct": 0.0, "peak_memory_mb": 0.0}

    def _runner():
        nonlocal monitor_result
        try:
            monitor_result = monitor_resources(stop_event)
        except Exception:
            monitor_result = {"peak_cpu_usage_pct": 0.0, "peak_memory_mb": 0.0}

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    try:
        result = fn(*args, **kwargs)
    finally:
        stop_event.set()
        thread.join(timeout=1.0)
    return result, monitor_result


async def collect_peak_resources_during_async(awaitable_fn, *args):
    stop_event = threading.Event()
    monitor_result = {"peak_cpu_usage_pct": 0.0, "peak_memory_mb": 0.0}

    def _runner():
        nonlocal monitor_result
        try:
            monitor_result = monitor_resources(stop_event)
        except Exception:
            monitor_result = {"peak_cpu_usage_pct": 0.0, "peak_memory_mb": 0.0}

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    try:
        result = await awaitable_fn(*args)
    finally:
        stop_event.set()
        thread.join(timeout=1.0)
    return result, monitor_result


def _build_photo_process_context(config: Dict[str, Any], agent: QuantizedVisionAgent) -> Dict[str, Any]:
    loopback_guard_cfg = config.get("runtime", {}).get("loopback_guard", {})
    loopback_planner = create_loopback_planner(config)
    runtime_cfg = config.get("runtime", {})
    replay_mode = str(runtime_cfg.get("replay_mode", "off")).lower()
    replay_file = runtime_cfg.get("replay_file")
    replay_index = None
    if replay_mode in {"record", "replay"} and not replay_file:
        raise ValueError("runtime.replay_file is required when runtime.replay_mode is record/replay")
    if replay_mode == "replay":
        replay_index = build_replay_index(load_replay_steps(str(replay_file)))
    return {
        "agent": agent,
        "image_processor": ImageProcessor(),
        "loopback_planner": loopback_planner,
        "replay_mode": replay_mode,
        "replay_file": str(replay_file) if replay_file is not None else None,
        "replay_index": replay_index,
        "max_retry": int(config.get("runtime", {}).get("max_retry", 3)),
        "thresholds_cfg": config.get("thresholds", {}),
        "loopback_guard_cfg": loopback_guard_cfg,
        "min_brightness_gain": float(loopback_guard_cfg.get("min_brightness_gain", 4.0)),
        "min_sharpness_gain": float(loopback_guard_cfg.get("min_sharpness_gain", 1.0)),
        "brighten_factor": float(loopback_guard_cfg.get("brighten_factor", 1.2)),
        "dim_factor": float(loopback_guard_cfg.get("dim_factor", 0.85)),
        "overexposure_stop_ratio": float(loopback_guard_cfg.get("overexposure_stop_ratio", 0.95)),
    }


def _build_planner_trace_payload(
    *,
    image_path: str,
    attempt: int,
    signal: str,
    engine_metrics: Dict[str, Any],
    thresholds_cfg: Dict[str, Any],
    loopback_guard_cfg: Dict[str, Any],
    attempt_history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "image_path": str(image_path),
        "attempt": int(attempt),
        "signal": str(signal),
        "engine_metrics": engine_metrics,
        "thresholds_cfg": thresholds_cfg,
        "loopback_guard_cfg": loopback_guard_cfg,
        "attempt_history": attempt_history,
    }


def _resolve_loopback_plan(
    *,
    ctx: Dict[str, Any],
    image_path: str,
    attempt: int,
    signal: str,
    engine_metrics: Dict[str, Any],
    attempt_history: List[Dict[str, Any]],
) -> LoopbackPlan:
    trace_payload = _build_planner_trace_payload(
        image_path=image_path,
        attempt=attempt,
        signal=signal,
        engine_metrics=engine_metrics,
        thresholds_cfg=ctx["thresholds_cfg"],
        loopback_guard_cfg=ctx.get("loopback_guard_cfg", {}),
        attempt_history=attempt_history,
    )
    trace_hash = planner_input_hash(trace_payload)
    replay_mode = str(ctx.get("replay_mode", "off")).lower()
    if replay_mode == "replay":
        step = get_replay_step(
            ctx["replay_index"],
            image_path=image_path,
            attempt=attempt,
            expected_planner_input_hash=trace_hash,
        )
        plan = LoopbackPlan(
            action=(step.get("action") or None),
            stop_reason=str(step.get("stop_reason") or "replay"),
            rationale=str(step.get("planner_output", {}).get("rationale", "replay step")),
            fallback_used=False,
            planner_backend=f"replay:{step.get('backend', 'unknown')}",
        )
        return plan

    planner_start = time.perf_counter()
    plan = ctx["loopback_planner"].plan(
        signal=signal,
        engine_metrics=engine_metrics,
        thresholds_cfg=ctx["thresholds_cfg"],
        loopback_guard_cfg=ctx.get("loopback_guard_cfg", {}),
        attempt_history=attempt_history,
    )
    planner_latency_ms = (time.perf_counter() - planner_start) * 1000.0
    if replay_mode == "record" and ctx.get("replay_file"):
        append_replay_step(
            ctx["replay_file"],
            {
                "image_path": str(image_path),
                "attempt": int(attempt),
                "metrics_before": {
                    "avg_brightness": float(engine_metrics.get("avg_brightness", 0.0)),
                    "sharpness": float(engine_metrics.get("sharpness", 0.0)),
                },
                "planner_input_hash": trace_hash,
                "planner_output": {
                    "action": str(plan.action or "stop"),
                    "stop_reason": str(plan.stop_reason),
                    "rationale": str(plan.rationale),
                },
                "action": str(plan.action or "stop"),
                "signal": str(signal),
                "backend": str(plan.planner_backend),
                "latency_ms": round(planner_latency_ms, 4),
                "stop_reason": str(plan.stop_reason),
                "timestamp": datetime.now().isoformat(),
            },
        )
    return plan


def _process_single_photo(path: str, ctx: Dict[str, Any]) -> Dict[str, Any]:
    agent = ctx["agent"]
    image_processor = ctx["image_processor"]
    file_name = os.path.basename(path)
    file_stem = Path(file_name).stem
    image_wall_start = time.perf_counter()

    if random.random() < agent.oom_probability:
        raise MemoryError("OOM Exception")

    image_meta = get_image_metadata(path)
    cpu_start = time.process_time()
    attempt_history = []
    current_path = path
    final_metrics = None
    final_ai_result = None
    final_latency = 0.0
    loopback_stop_reason = "not_triggered"
    peak_cpu_usage_pct = 0.0
    peak_memory_mb = 0.0

    for attempt_idx in range(ctx["max_retry"] + 1):
        (metrics, ai_result, latency), resource_peaks = collect_peak_resources_during(
            agent.analyze_photo_quality, current_path
        )
        peak_cpu_usage_pct = max(peak_cpu_usage_pct, resource_peaks.get("peak_cpu_usage_pct", 0.0))
        peak_memory_mb = max(peak_memory_mb, resource_peaks.get("peak_memory_mb", 0.0))
        final_metrics = metrics
        final_ai_result = ai_result
        final_latency += latency

        if not isinstance(metrics, dict):
            attempt_history.append({
                "attempt": attempt_idx + 1,
                "image_path": current_path,
                "model_decision": ai_result.get("decision"),
                "error_code": ai_result.get("code"),
                "release": "NO_GO",
                "loopback_signal": "other",
                "action": "stop",
                "rationale": "metrics unavailable; stop loopback",
                "latency_ms": latency,
            })
            loopback_stop_reason = "metrics_unavailable"
            break

        engine_metrics = {
            "avg_brightness": metrics.get("avg_brightness", metrics.get("brightness", 0.0)),
            "sharpness": metrics.get("sharpness", 0.0),
        }
        model_inference = {
            "decision": ai_result.get("decision"),
            "status": ai_result.get("decision"),
            "confidence": ai_result.get("confidence"),
        }
        release_decision, _ = arbitrate_decision(
            engine_metrics, model_inference, ctx["thresholds_cfg"]
        )
        loopback_signal = classify_loopback_signal(ai_result)
        attempt_history.append({
            "attempt": attempt_idx + 1,
            "image_path": current_path,
            "model_decision": ai_result.get("decision"),
            "error_code": ai_result.get("code"),
            "release": release_decision,
            "loopback_signal": loopback_signal,
            "action": "stop",
            "rationale": "release resolved or awaiting planner decision",
            "avg_brightness": round(float(engine_metrics.get("avg_brightness", 0.0)), 4),
            "sharpness": round(float(engine_metrics.get("sharpness", 0.0)), 4),
            "latency_ms": latency,
        })

        current_brightness = float(engine_metrics.get("avg_brightness", 0.0))
        current_sharpness = float(engine_metrics.get("sharpness", 0.0))
        if release_decision != "NO_GO":
            loopback_stop_reason = "release_resolved"
            break
        if attempt_idx >= ctx["max_retry"]:
            loopback_stop_reason = "max_retry_reached"
            break

        plan = _resolve_loopback_plan(
            ctx=ctx,
            image_path=current_path,
            attempt=attempt_idx + 1,
            signal=loopback_signal,
            engine_metrics=engine_metrics,
            attempt_history=attempt_history,
        )
        logger.info(
            "Loopback planner for %s attempt=%s signal=%s action=%s stop_reason=%s rationale=%s",
            file_name,
            attempt_idx + 1,
            loopback_signal,
            plan.action,
            plan.stop_reason,
            plan.rationale,
        )
        attempt_history[-1]["planner_fallback_used"] = bool(plan.fallback_used)
        attempt_history[-1]["planner_backend"] = str(plan.planner_backend)
        attempt_history[-1]["action"] = str(plan.action or "stop")
        attempt_history[-1]["rationale"] = str(plan.rationale)
        if not plan.action:
            loopback_stop_reason = plan.stop_reason
            break

        if len(attempt_history) >= 2:
            prev_attempt = attempt_history[-2]
            prev_signal = prev_attempt.get("loopback_signal")
            prev_brightness = float(prev_attempt.get("avg_brightness", 0.0))
            prev_sharpness = float(prev_attempt.get("sharpness", 0.0))
            brightness_gain = current_brightness - prev_brightness
            if prev_signal in {"under", "over"} and prev_signal != loopback_signal:
                loopback_stop_reason = "oscillation_detected"
                break
            if plan.action == "brighten" and brightness_gain < ctx["min_brightness_gain"]:
                loopback_stop_reason = f"insufficient_brightness_gain (<{ctx['min_brightness_gain']})"
                break
            if plan.action == "dim" and (prev_brightness - current_brightness) < ctx["min_brightness_gain"]:
                loopback_stop_reason = f"insufficient_dimming_gain (<{ctx['min_brightness_gain']})"
                break
            if plan.action == "sharpen" and (current_sharpness - prev_sharpness) < ctx["min_sharpness_gain"]:
                loopback_stop_reason = f"insufficient_sharpness_gain (<{ctx['min_sharpness_gain']})"
                break

        if plan.action == "brighten":
            current_path = image_processor.adjust_brightness(
                current_path,
                level=ctx["brighten_factor"],
                file_stem=file_stem,
                attempt_idx=attempt_idx + 1,
            )
            logger.info(
                "Loopback retry %s/%s for %s: detected under-exposed; brightness x%s and re-evaluate.",
                attempt_idx + 1,
                ctx["max_retry"],
                file_name,
                ctx["brighten_factor"],
            )
        elif plan.action == "dim":
            current_path = image_processor.adjust_brightness(
                current_path,
                level=ctx["dim_factor"],
                file_stem=file_stem,
                attempt_idx=attempt_idx + 1,
            )
            logger.info(
                "Loopback retry %s/%s for %s: detected over-exposed; brightness x%s and re-evaluate.",
                attempt_idx + 1,
                ctx["max_retry"],
                file_name,
                ctx["dim_factor"],
            )
        elif plan.action == "sharpen":
            current_path = image_processor.apply_sharpen(
                current_path,
                file_stem=file_stem,
                attempt_idx=attempt_idx + 1,
            )
            logger.info(
                "Loopback retry %s/%s for %s: detected blurry signal; apply sharpen and re-evaluate.",
                attempt_idx + 1,
                ctx["max_retry"],
                file_name,
            )
        loopback_stop_reason = f"retry_scheduled ({plan.action})"

    cpu_delta = max(0.0, time.process_time() - cpu_start)
    wall_delta = max(final_latency / 1000.0, 1e-6)
    process_cpu_usage_pct = round((cpu_delta / wall_delta) * 100, 4)
    logger.info(
        "Processed %s: [%s] %s (%sms total)",
        file_name,
        (final_ai_result or {}).get("code", "?"),
        (final_ai_result or {}).get("decision", "?"),
        round(final_latency, 2),
    )

    image_wall_ms = (time.perf_counter() - image_wall_start) * 1000.0
    model_latency_ms = float(round(final_latency, 2))
    framework_wall_ms = max(0.0, image_wall_ms - model_latency_ms)

    return {
        "row": {
            "file": file_name,
            "metrics": final_metrics,
            "decision": final_ai_result,
            "inference_output": _build_agent_inference_output(
                image_path=path,
                attempt_history=attempt_history,
                final_ai_result=final_ai_result or {},
                total_latency_ms=round(final_latency, 2),
            ),
            "latency_ms": round(final_latency, 2),
            "image_meta": image_meta,
            "process_cpu_usage_pct": process_cpu_usage_pct,
            "loopback": {
                "max_retry": ctx["max_retry"],
                "min_brightness_gain": ctx["min_brightness_gain"],
                "min_sharpness_gain": ctx["min_sharpness_gain"],
                "brighten_factor": ctx["brighten_factor"],
                "dim_factor": ctx["dim_factor"],
                "overexposure_stop_ratio": ctx["overexposure_stop_ratio"],
                "retry_count": max(0, len(attempt_history) - 1),
                "fallback_used_count": sum(
                    1 for item in attempt_history if bool(item.get("planner_fallback_used"))
                ),
                "fallback_used": any(
                    bool(item.get("planner_fallback_used")) for item in attempt_history
                ),
                "stop_reason": loopback_stop_reason,
                "attempts": attempt_history,
            },
            "status": "SUCCESS",
        },
        "perf_sample": {
            "file": file_name,
            "latency_ms": round(final_latency, 2),
            "process_cpu_usage_pct": process_cpu_usage_pct,
            "peak_cpu_usage_pct": round(peak_cpu_usage_pct, 4),
            "peak_memory_mb": round(peak_memory_mb, 4),
            "image_resolution": f"{image_meta.get('width', 0)}x{image_meta.get('height', 0)}",
            **image_meta,
        },
        "overhead": {
            "image_wall_ms": image_wall_ms,
            "model_latency_ms": model_latency_ms,
            "framework_wall_ms": framework_wall_ms,
            "loopback_retry_count": max(0, len(attempt_history) - 1),
        },
    }


@async_monitor_performance
async def _process_single_photo_async(
    path: str,
    ctx: Dict[str, Any],
    http_client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    async with semaphore:
        logger.debug("_process_single_photo_async: start path=%s", path)
        agent = ctx["agent"]
        image_processor = ctx["image_processor"]
        file_name = os.path.basename(path)
        file_stem = Path(file_name).stem
        image_wall_start = time.perf_counter()

        if random.random() < agent.oom_probability:
            raise MemoryError("OOM Exception")

        image_meta = await asyncio.to_thread(get_image_metadata, path)
        cpu_start = time.process_time()
        attempt_history = []
        current_path = path
        final_metrics = None
        final_ai_result = None
        final_latency = 0.0
        loopback_stop_reason = "not_triggered"
        peak_cpu_usage_pct = 0.0
        peak_memory_mb = 0.0

        for attempt_idx in range(ctx["max_retry"] + 1):

            async def _analyze(photo_path=current_path):
                return await agent.analyze_photo_quality_async(photo_path, http_client)

            (metrics, ai_result, latency), resource_peaks = await collect_peak_resources_during_async(
                _analyze
            )
            peak_cpu_usage_pct = max(peak_cpu_usage_pct, resource_peaks.get("peak_cpu_usage_pct", 0.0))
            peak_memory_mb = max(peak_memory_mb, resource_peaks.get("peak_memory_mb", 0.0))
            final_metrics = metrics
            final_ai_result = ai_result
            final_latency += latency

            if not isinstance(metrics, dict):
                attempt_history.append({
                    "attempt": attempt_idx + 1,
                    "image_path": current_path,
                    "model_decision": ai_result.get("decision"),
                    "error_code": ai_result.get("code"),
                    "release": "NO_GO",
                "loopback_signal": "other",
                "action": "stop",
                "rationale": "metrics unavailable; stop loopback",
                    "latency_ms": latency,
                })
                loopback_stop_reason = "metrics_unavailable"
                break

            engine_metrics = {
                "avg_brightness": metrics.get("avg_brightness", metrics.get("brightness", 0.0)),
                "sharpness": metrics.get("sharpness", 0.0),
            }
            model_inference = {
                "decision": ai_result.get("decision"),
                "status": ai_result.get("decision"),
                "confidence": ai_result.get("confidence"),
            }
            release_decision, _ = arbitrate_decision(
                engine_metrics, model_inference, ctx["thresholds_cfg"]
            )
            loopback_signal = classify_loopback_signal(ai_result)
            attempt_history.append({
                "attempt": attempt_idx + 1,
                "image_path": current_path,
                "model_decision": ai_result.get("decision"),
                "error_code": ai_result.get("code"),
                "release": release_decision,
                "loopback_signal": loopback_signal,
                "action": "stop",
                "rationale": "release resolved or awaiting planner decision",
                "avg_brightness": round(float(engine_metrics.get("avg_brightness", 0.0)), 4),
                "sharpness": round(float(engine_metrics.get("sharpness", 0.0)), 4),
                "latency_ms": latency,
            })

            current_brightness = float(engine_metrics.get("avg_brightness", 0.0))
            current_sharpness = float(engine_metrics.get("sharpness", 0.0))
            if release_decision != "NO_GO":
                loopback_stop_reason = "release_resolved"
                break
            if attempt_idx >= ctx["max_retry"]:
                loopback_stop_reason = "max_retry_reached"
                break

            plan = _resolve_loopback_plan(
                ctx=ctx,
                image_path=current_path,
                attempt=attempt_idx + 1,
                signal=loopback_signal,
                engine_metrics=engine_metrics,
                attempt_history=attempt_history,
            )
            logger.info(
                "Loopback planner (async) for %s attempt=%s signal=%s action=%s stop_reason=%s rationale=%s",
                file_name,
                attempt_idx + 1,
                loopback_signal,
                plan.action,
                plan.stop_reason,
                plan.rationale,
            )
            attempt_history[-1]["planner_fallback_used"] = bool(plan.fallback_used)
            attempt_history[-1]["planner_backend"] = str(plan.planner_backend)
            attempt_history[-1]["action"] = str(plan.action or "stop")
            attempt_history[-1]["rationale"] = str(plan.rationale)
            if not plan.action:
                loopback_stop_reason = plan.stop_reason
                break

            if len(attempt_history) >= 2:
                prev_attempt = attempt_history[-2]
                prev_signal = prev_attempt.get("loopback_signal")
                prev_brightness = float(prev_attempt.get("avg_brightness", 0.0))
                prev_sharpness = float(prev_attempt.get("sharpness", 0.0))
                brightness_gain = current_brightness - prev_brightness
                if prev_signal in {"under", "over"} and prev_signal != loopback_signal:
                    loopback_stop_reason = "oscillation_detected"
                    break
                if plan.action == "brighten" and brightness_gain < ctx["min_brightness_gain"]:
                    loopback_stop_reason = f"insufficient_brightness_gain (<{ctx['min_brightness_gain']})"
                    break
                if plan.action == "dim" and (prev_brightness - current_brightness) < ctx["min_brightness_gain"]:
                    loopback_stop_reason = f"insufficient_dimming_gain (<{ctx['min_brightness_gain']})"
                    break
                if plan.action == "sharpen" and (current_sharpness - prev_sharpness) < ctx["min_sharpness_gain"]:
                    loopback_stop_reason = f"insufficient_sharpness_gain (<{ctx['min_sharpness_gain']})"
                    break

            if plan.action == "brighten":
                current_path = await asyncio.to_thread(
                    image_processor.adjust_brightness,
                    current_path,
                    ctx["brighten_factor"],
                    file_stem,
                    attempt_idx + 1,
                )
            elif plan.action == "dim":
                current_path = await asyncio.to_thread(
                    image_processor.adjust_brightness,
                    current_path,
                    ctx["dim_factor"],
                    file_stem,
                    attempt_idx + 1,
                )
            elif plan.action == "sharpen":
                current_path = await asyncio.to_thread(
                    image_processor.apply_sharpen,
                    current_path,
                    file_stem,
                    attempt_idx + 1,
                )
            loopback_stop_reason = f"retry_scheduled ({plan.action})"

        cpu_delta = max(0.0, time.process_time() - cpu_start)
        wall_delta = max(final_latency / 1000.0, 1e-6)
        process_cpu_usage_pct = round((cpu_delta / wall_delta) * 100, 4)
        logger.info(
            "Processed (async) %s: [%s] %s (%sms total)",
            file_name,
            (final_ai_result or {}).get("code", "?"),
            (final_ai_result or {}).get("decision", "?"),
            round(final_latency, 2),
        )

        image_wall_ms = (time.perf_counter() - image_wall_start) * 1000.0
        model_latency_ms = float(round(final_latency, 2))
        framework_wall_ms = max(0.0, image_wall_ms - model_latency_ms)

        return {
            "row": {
                "file": file_name,
                "metrics": final_metrics,
                "decision": final_ai_result,
                "inference_output": _build_agent_inference_output(
                    image_path=path,
                    attempt_history=attempt_history,
                    final_ai_result=final_ai_result or {},
                    total_latency_ms=round(final_latency, 2),
                ),
                "latency_ms": round(final_latency, 2),
                "image_meta": image_meta,
                "process_cpu_usage_pct": process_cpu_usage_pct,
                "loopback": {
                    "max_retry": ctx["max_retry"],
                    "min_brightness_gain": ctx["min_brightness_gain"],
                    "min_sharpness_gain": ctx["min_sharpness_gain"],
                    "brighten_factor": ctx["brighten_factor"],
                    "dim_factor": ctx["dim_factor"],
                    "overexposure_stop_ratio": ctx["overexposure_stop_ratio"],
                    "retry_count": max(0, len(attempt_history) - 1),
                    "fallback_used_count": sum(
                        1 for item in attempt_history if bool(item.get("planner_fallback_used"))
                    ),
                    "fallback_used": any(
                        bool(item.get("planner_fallback_used")) for item in attempt_history
                    ),
                    "stop_reason": loopback_stop_reason,
                    "attempts": attempt_history,
                },
                "status": "SUCCESS",
            },
            "perf_sample": {
                "file": file_name,
                "latency_ms": round(final_latency, 2),
                "process_cpu_usage_pct": process_cpu_usage_pct,
                "peak_cpu_usage_pct": round(peak_cpu_usage_pct, 4),
                "peak_memory_mb": round(peak_memory_mb, 4),
                "image_resolution": f"{image_meta.get('width', 0)}x{image_meta.get('height', 0)}",
                **image_meta,
            },
            "overhead": {
                "image_wall_ms": image_wall_ms,
                "model_latency_ms": model_latency_ms,
                "framework_wall_ms": framework_wall_ms,
                "loopback_retry_count": max(0, len(attempt_history) - 1),
            },
        }


def benchmark_monitor_overhead(samples=5, sleep_s=0.2):
    per_run_ms = []
    process = psutil.Process(os.getpid())
    for _ in range(samples):
        cpu_before = time.process_time()
        rss_before = process.memory_info().rss / (1024.0 * 1024.0)
        start = time.perf_counter()
        collect_peak_resources_during(time.sleep, sleep_s)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        cpu_after = time.process_time()
        rss_after = process.memory_info().rss / (1024.0 * 1024.0)
        per_run_ms.append({
            "wall_ms": round(elapsed_ms, 4),
            "extra_wall_ms_vs_sleep": round(max(0.0, elapsed_ms - (sleep_s * 1000.0)), 4),
            "cpu_time_ms": round((cpu_after - cpu_before) * 1000.0, 4),
            "rss_delta_mb": round(rss_after - rss_before, 4),
        })

    avg_extra = sum(item["extra_wall_ms_vs_sleep"] for item in per_run_ms) / len(per_run_ms)
    avg_cpu = sum(item["cpu_time_ms"] for item in per_run_ms) / len(per_run_ms)
    max_rss_delta = max(item["rss_delta_mb"] for item in per_run_ms) if per_run_ms else 0.0
    return {
        "samples": samples,
        "sleep_s": sleep_s,
        "avg_extra_wall_ms": round(avg_extra, 4),
        "avg_cpu_time_ms": round(avg_cpu, 4),
        "max_rss_delta_mb": round(max_rss_delta, 4),
        "runs": per_run_ms,
    }


def _finalize_batch_report(
    *,
    batch_report: Dict[str, Any],
    config: Dict[str, Any],
    config_profile: str,
    config_source: str,
    perf_samples: List[Dict[str, Any]],
    failure_memory_store: FailureMemoryStore,
    overhead_counters: Dict[str, Any],
    performance_analysis: bool,
    overhead_analysis: bool,
    batch_wall_start: float,
    batch_cpu_start: float,
    batch_rss_start_mb: float,
    monitor_overhead_baseline: Optional[Dict[str, Any]],
    process: psutil.Process,
) -> Dict[str, Any]:
    total = len(batch_report["results"])
    success_count = sum(
        1 for row in batch_report["results"]
        if isinstance(row.get("decision"), dict) and row["decision"].get("code") == "SUCCESS_200"
    )
    pass_rate = (success_count / total) * 100 if total > 0 else 0
    successful_latencies = [
        row.get("latency_ms", 0) for row in batch_report["results"] if row.get("status") == "SUCCESS"
    ]
    avg_lat = sum(successful_latencies) / len(successful_latencies) if successful_latencies else 0

    rankings = build_rankings(batch_report["results"], config["thresholds"])
    gate_decision, gate_reason = get_release_decision(pass_rate, avg_lat, config)

    eval_settings = config.get("eval_settings", {})
    conflict_strategy = eval_settings.get("conflict_strategy", "conservative")
    auto_tag_conflicts = bool(eval_settings.get("auto_tag_conflicts", True))
    thresholds_cfg = config.get("thresholds", {})

    per_image_releases = []
    for row in batch_report["results"]:
        if row.get("status") != "SUCCESS":
            per_image_releases.append("NO_GO")
            continue
        metrics = row.get("metrics")
        ai_result = row.get("decision", {})
        if not isinstance(metrics, dict) or not isinstance(ai_result, dict):
            per_image_releases.append("NO_GO")
            continue
        engine_metrics = {
            "avg_brightness": metrics.get("avg_brightness", metrics.get("brightness", 0.0)),
            "sharpness": metrics.get("sharpness", 0.0),
        }
        model_inference = {
            "decision": ai_result.get("decision"),
            "status": ai_result.get("decision"),
            "confidence": ai_result.get("confidence"),
        }
        release, conflict_enum = arbitrate_decision(
            engine_metrics, model_inference, thresholds_cfg
        )
        per_image_releases.append(release)
        if auto_tag_conflicts:
            raw_c = ai_result.get("confidence")
            conf_val = float(raw_c) if raw_c is not None else None
            row["arbitration"] = {
                "release_decision": release,
                "conflict": conflict_enum.value,
                **({"model_confidence": conf_val} if conf_val is not None else {}),
            }
        if release in {"REVIEW", "NO_GO"}:
            failure_id = f"{batch_report['batch_id']}::{row.get('file', 'unknown')}::{release}"
            document = failure_memory_store.build_document(row.get("file", "unknown"), ai_result)
            metadata = failure_memory_store.build_metadata(
                config_profile,
                batch_report["batch_id"],
                row.get("file", "unknown"),
                row,
                release,
            )
            write_start = time.perf_counter()
            failure_memory_store.store_failure_case(failure_id, document, metadata)
            overhead_counters["failure_memory_write_ms"] += (time.perf_counter() - write_start) * 1000.0
            overhead_counters["failure_memory_write_count"] += 1

    arbitration_batch = aggregate_batch_decisions(per_image_releases, conflict_strategy)
    decision = merge_gate_and_arbitration(
        gate_decision, arbitration_batch, conflict_strategy
    )
    decision_reason = (
        f"Quality gate: {gate_reason} "
        f"(gate={gate_decision}) | Arbitration aggregate: {arbitration_batch} "
        f"(merged={decision}, strategy={conflict_strategy})"
    )

    top_ranking = rankings[:3]

    logger.info("%s", "\n" + "=" * 55)
    logger.info("Test Dashboard")
    logger.info("  - Total tests: %s", total)
    logger.info("  - Pass rate (Optimal): %.1f%%", pass_rate)
    logger.info("  - Average latency: %.2f ms", avg_lat)
    logger.info("  - Release decision (arbitrated): %s", decision)
    logger.info(
        "    (gate=%s, arbitration_batch=%s)",
        gate_decision,
        arbitration_batch,
    )
    logger.info("%s", "-" * 55)
    logger.info("Top ranking:")
    for item in top_ranking:
        logger.info(
            "  #%s %s | score=%s | %s",
            item["rank"],
            item["file"],
            item["score"],
            item["decision"],
        )
    logger.info("%s", "=" * 55)

    batch_report["summary"] = {
        "total_tests": total,
        "success_count": success_count,
        "pass_rate": round(pass_rate, 2),
        "target_pass_rate": float(config.get("quality_gate", {}).get("target_pass_rate", 90.0)),
        "avg_latency_ms": round(avg_lat, 2),
        "release_decision": decision,
        "release_decision_gate": gate_decision,
        "release_decision_arbitration": arbitration_batch,
        "decision_reason": decision_reason,
    }
    if batch_report.get("execution_mode") == "async":
        batch_report["summary"]["async_timeout_count"] = int(
            batch_report.get("async_timeout_count", 0)
        )
        if batch_report.get("async_per_image_timeout_s") is not None:
            batch_report["summary"]["async_per_image_timeout_s"] = float(
                batch_report["async_per_image_timeout_s"]
            )
    batch_report["ranking"] = rankings

    report_path = save_batch_report(batch_report, config["folders"]["output"])
    performance_report_path = None
    overhead_report_path = None
    if performance_analysis:
        performance_report = {
            "generated_at": datetime.now().isoformat(),
            "profile": config_profile,
            "summary": summarize_performance(perf_samples),
            "samples": perf_samples,
        }
        performance_report_path = save_performance_report(performance_report, "results")
    if overhead_analysis:
        batch_wall_ms = (time.perf_counter() - batch_wall_start) * 1000.0
        batch_cpu_ms = (time.process_time() - batch_cpu_start) * 1000.0
        batch_rss_end_mb = process.memory_info().rss / (1024.0 * 1024.0)
        framework_ratio = (
            (overhead_counters["total_framework_wall_ms"] / overhead_counters["total_image_wall_ms"]) * 100.0
            if overhead_counters["total_image_wall_ms"] > 0 else 0.0
        )
        rag_ratio = (
            (overhead_counters["failure_memory_write_ms"] / overhead_counters["total_image_wall_ms"]) * 100.0
            if overhead_counters["total_image_wall_ms"] > 0 else 0.0
        )
        overhead_report = {
            "generated_at": datetime.now().isoformat(),
            "profile": config_profile,
            "batch_id": batch_report["batch_id"],
            "summary": {
                "total_images": total,
                "batch_wall_ms": round(batch_wall_ms, 4),
                "batch_cpu_time_ms": round(batch_cpu_ms, 4),
                "batch_rss_start_mb": round(batch_rss_start_mb, 4),
                "batch_rss_end_mb": round(batch_rss_end_mb, 4),
                "batch_rss_delta_mb": round(batch_rss_end_mb - batch_rss_start_mb, 4),
                "total_image_wall_ms": round(overhead_counters["total_image_wall_ms"], 4),
                "total_model_latency_ms": round(overhead_counters["total_model_latency_ms"], 4),
                "total_framework_wall_ms": round(overhead_counters["total_framework_wall_ms"], 4),
                "framework_overhead_ratio_pct": round(framework_ratio, 4),
                "loopback_retry_count": int(overhead_counters["total_loopback_retry_count"]),
                "failure_memory_write_count": int(overhead_counters["failure_memory_write_count"]),
                "failure_memory_write_ms": round(overhead_counters["failure_memory_write_ms"], 4),
                "failure_memory_overhead_ratio_pct": round(rag_ratio, 4),
            },
            "monitor_baseline": monitor_overhead_baseline,
            "auto_insight": (
                f"Framework overhead is {round(framework_ratio, 2)}% of image-processing wall time; "
                f"failure-memory writes contribute {round(rag_ratio, 2)}%."
            ),
        }
        overhead_report_path = save_overhead_report(overhead_report, "results")

    return {
        "profile": config_profile,
        "report_path": report_path,
        "performance_report_path": performance_report_path,
        "overhead_report_path": overhead_report_path,
        "summary": batch_report["summary"],
        "top_ranked": top_ranking,
        "ranking": rankings,
        "image_files": sorted([row["file"] for row in batch_report["results"] if "file" in row]),
    }


def run_batch_test(
    config_profile="dev",
    config_path=None,
    deterministic=False,
    inference_backend_override=None,
    loopback_planner_override=None,
    planner_timeout_s_override=None,
    planner_model_override=None,
    planner_require_healthy_override=None,
    async_per_image_timeout_s_override=None,
    async_backend_health_check_override=None,
    async_backend_health_timeout_s_override=None,
    replay_mode_override=None,
    replay_file_override=None,
    performance_analysis=False,
    overhead_analysis=False,
    stress_test_count=None,
    parallel_metrics=False,
    metrics_workers=None,
):
    config, config_source = load_config(profile=config_profile, config_path=config_path)
    error_report_dir = config.get("folders", {}).get("logs", "logs/errors")
    config_source = _apply_runtime_overrides(
        config,
        config_source,
        inference_backend_override=inference_backend_override,
        loopback_planner_override=loopback_planner_override,
        planner_timeout_s_override=planner_timeout_s_override,
        planner_model_override=planner_model_override,
        planner_require_healthy_override=planner_require_healthy_override,
        async_per_image_timeout_s_override=async_per_image_timeout_s_override,
        async_backend_health_check_override=async_backend_health_check_override,
        async_backend_health_timeout_s_override=async_backend_health_timeout_s_override,
        replay_mode_override=replay_mode_override,
        replay_file_override=replay_file_override,
    )
    logger.info("Loaded config source: %s", config_source)

    pool_cm = (
        MetricsProcessPool(max_workers=metrics_workers)
        if parallel_metrics
        else nullcontext()
    )
    with pool_cm as metrics_pool:
        agent = QuantizedVisionAgent(
            config,
            metrics_pool=metrics_pool if parallel_metrics else None,
        )
        return _run_batch_test_body(
            agent=agent,
            config=config,
            config_profile=config_profile,
            config_source=config_source,
            error_report_dir=error_report_dir,
            deterministic=deterministic,
            performance_analysis=performance_analysis,
            overhead_analysis=overhead_analysis,
            stress_test_count=stress_test_count,
            parallel_metrics=parallel_metrics,
            metrics_workers=metrics_pool.max_workers if parallel_metrics else None,
        )


def _run_batch_test_body(
    *,
    agent: QuantizedVisionAgent,
    config: Dict[str, Any],
    config_profile: str,
    config_source: str,
    error_report_dir: str,
    deterministic: bool,
    performance_analysis: bool,
    overhead_analysis: bool,
    stress_test_count: Optional[int],
    parallel_metrics: bool,
    metrics_workers: Optional[int],
):
    photos = agent.get_all_photos()
    if stress_test_count:
        input_folder = config["folders"]["input"]
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(current_dir)
        full_input_path = os.path.join(base_dir, input_folder)
        ensure_stress_test_images(full_input_path, target_count=int(stress_test_count))
        photos = agent.get_all_photos()
    if deterministic:
        random.seed(42)
        agent.oom_probability = 0.0

    if not photos:
        logger.warning("No testable images were found.")
        return None

    batch_report = {
        "schema_version": "2.0",
        "profile": config_profile,
        "batch_id": datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
        "config_used": config["thresholds"],
        "config_source": config_source,
        "results": [],
    }
    if parallel_metrics:
        batch_report["parallel_metrics"] = True
        batch_report["metrics_workers"] = metrics_workers

    perf_samples = []
    failure_memory_store = FailureMemoryStore()
    photo_ctx = _build_photo_process_context(config, agent)
    process = psutil.Process(os.getpid())
    batch_wall_start = time.perf_counter()
    batch_cpu_start = time.process_time()
    batch_rss_start_mb = process.memory_info().rss / (1024.0 * 1024.0)
    monitor_overhead_baseline = benchmark_monitor_overhead() if overhead_analysis else None
    overhead_counters = {
        "total_image_wall_ms": 0.0,
        "total_model_latency_ms": 0.0,
        "total_framework_wall_ms": 0.0,
        "total_loopback_retry_count": 0,
        "failure_memory_write_ms": 0.0,
        "failure_memory_write_count": 0,
    }

    logger.info("Starting to process %s image(s)...", len(photos))

    for path in photos:
        file_name = os.path.basename(path)
        try:
            outcome = _process_single_photo(path, photo_ctx)
            batch_report["results"].append(outcome["row"])
            perf_samples.append(outcome["perf_sample"])
            oh = outcome["overhead"]
            overhead_counters["total_image_wall_ms"] += oh["image_wall_ms"]
            overhead_counters["total_model_latency_ms"] += oh["model_latency_ms"]
            overhead_counters["total_framework_wall_ms"] += oh["framework_wall_ms"]
            overhead_counters["total_loopback_retry_count"] += oh["loopback_retry_count"]

        except Exception as e:
            logger.exception("Failed to process file %s", file_name)
            error_payload = {
                "generated_at": datetime.now().isoformat(),
                "scope": "single_file",
                "profile": config_profile,
                "config_source": config_source,
                "file": file_name,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc(),
            }
            save_error_report(error_payload, error_report_dir)
            batch_report["results"].append({
                "file": file_name,
                "status": "FAILED",
                "error": str(e)
            })

    return _finalize_batch_report(
        batch_report=batch_report,
        config=config,
        config_profile=config_profile,
        config_source=config_source,
        perf_samples=perf_samples,
        failure_memory_store=failure_memory_store,
        overhead_counters=overhead_counters,
        performance_analysis=performance_analysis,
        overhead_analysis=overhead_analysis,
        batch_wall_start=batch_wall_start,
        batch_cpu_start=batch_cpu_start,
        batch_rss_start_mb=batch_rss_start_mb,
        monitor_overhead_baseline=monitor_overhead_baseline,
        process=process,
    )


@async_monitor_performance
async def _run_async_batch_processing(
    photos: List[str],
    photo_ctx: Dict[str, Any],
    batch_report: Dict[str, Any],
    perf_samples: List[Dict[str, Any]],
    overhead_counters: Dict[str, Any],
    error_report_dir: str,
    config_profile: str,
    config_source: str,
    concurrency: int,
    per_image_timeout_s: Optional[float],
) -> None:
    semaphore = asyncio.Semaphore(max(1, concurrency))
    logger.info(
        "Async batch: processing %s image(s) with concurrency=%s",
        len(photos),
        max(1, concurrency),
    )

    async with httpx.AsyncClient() as http_client:

        async def _handle_photo(path: str) -> Dict[str, Any]:
            file_name = os.path.basename(path)
            try:
                photo_task = _process_single_photo_async(
                    path, photo_ctx, http_client, semaphore
                )
                outcome = (
                    await asyncio.wait_for(photo_task, timeout=per_image_timeout_s)
                    if per_image_timeout_s is not None and per_image_timeout_s > 0
                    else await photo_task
                )
                return {
                    "status": "SUCCESS",
                    "file_name": file_name,
                    "outcome": outcome,
                }
            except asyncio.TimeoutError:
                timeout_msg = (
                    f"Per-image async timeout after {per_image_timeout_s}s while "
                    f"processing {file_name}"
                )
                logger.warning(timeout_msg)
                return {
                    "status": "FAILED",
                    "file_name": file_name,
                    "error": timeout_msg,
                    "error_type": "TimeoutError",
                    "timed_out": True,
                }
            except Exception as exc:
                logger.exception("Failed to process file %s (async)", file_name)
                error_payload = {
                    "generated_at": datetime.now().isoformat(),
                    "scope": "single_file",
                    "profile": config_profile,
                    "config_source": config_source,
                    "file": file_name,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                }
                save_error_report(error_payload, error_report_dir)
                return {
                    "status": "FAILED",
                    "file_name": file_name,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "timed_out": False,
                }

        outcomes = await gather_with_timing(
            [_handle_photo(path) for path in photos],
            label="async_batch_photos",
        )

    for item in outcomes:
        if item["status"] == "SUCCESS":
            outcome = item["outcome"]
            batch_report["results"].append(outcome["row"])
            perf_samples.append(outcome["perf_sample"])
            oh = outcome["overhead"]
            overhead_counters["total_image_wall_ms"] += oh["image_wall_ms"]
            overhead_counters["total_model_latency_ms"] += oh["model_latency_ms"]
            overhead_counters["total_framework_wall_ms"] += oh["framework_wall_ms"]
            overhead_counters["total_loopback_retry_count"] += oh["loopback_retry_count"]
        else:
            batch_report["results"].append({
                "file": item["file_name"],
                "status": "FAILED",
                "error": item["error"],
                "error_type": item.get("error_type", "Exception"),
                "timed_out": bool(item.get("timed_out", False)),
            })


def run_batch_test_async(
    config_profile="dev",
    config_path=None,
    deterministic=False,
    inference_backend_override=None,
    loopback_planner_override=None,
    planner_timeout_s_override=None,
    planner_model_override=None,
    planner_require_healthy_override=None,
    async_per_image_timeout_s_override=None,
    async_backend_health_check_override=None,
    async_backend_health_timeout_s_override=None,
    replay_mode_override=None,
    replay_file_override=None,
    performance_analysis=False,
    overhead_analysis=False,
    stress_test_count=None,
    concurrency=4,
    parallel_metrics=False,
    metrics_workers=None,
):
    """
    Parallel batch run: concurrent per-image processing with async HTTP inference.

    Use --parallel-metrics (ProcessPoolExecutor) when metrics CPU is the bottleneck;
    async helps most when waiting on llama.cpp / Ollama HTTP responses.
    """
    config, config_source = load_config(profile=config_profile, config_path=config_path)
    error_report_dir = config.get("folders", {}).get("logs", "logs/errors")
    config_source = _apply_runtime_overrides(
        config,
        config_source,
        inference_backend_override=inference_backend_override,
        loopback_planner_override=loopback_planner_override,
        planner_timeout_s_override=planner_timeout_s_override,
        planner_model_override=planner_model_override,
        planner_require_healthy_override=planner_require_healthy_override,
        async_per_image_timeout_s_override=async_per_image_timeout_s_override,
        async_backend_health_check_override=async_backend_health_check_override,
        async_backend_health_timeout_s_override=async_backend_health_timeout_s_override,
        replay_mode_override=replay_mode_override,
        replay_file_override=replay_file_override,
    )
    logger.info(
        "Loaded config source: %s (async batch, concurrency=%s, parallel_metrics=%s)",
        config_source,
        concurrency,
        parallel_metrics,
    )

    pool_cm = (
        MetricsProcessPool(max_workers=metrics_workers)
        if parallel_metrics
        else nullcontext()
    )
    with pool_cm as metrics_pool:
        agent = QuantizedVisionAgent(
            config,
            metrics_pool=metrics_pool if parallel_metrics else None,
        )
        _run_async_backend_health_check(
            agent.inference_engine,
            enabled=bool(config.get("runtime", {}).get("async_backend_health_check", True)),
            timeout_s=float(config.get("runtime", {}).get("async_backend_health_timeout_s", 2.0)),
        )
        return _run_batch_test_async_body(
            agent=agent,
            config=config,
            config_profile=config_profile,
            config_source=config_source,
            error_report_dir=error_report_dir,
            deterministic=deterministic,
            performance_analysis=performance_analysis,
            overhead_analysis=overhead_analysis,
            stress_test_count=stress_test_count,
            concurrency=concurrency,
            parallel_metrics=parallel_metrics,
            metrics_workers=metrics_pool.max_workers if parallel_metrics else None,
        )


def _run_batch_test_async_body(
    *,
    agent: QuantizedVisionAgent,
    config: Dict[str, Any],
    config_profile: str,
    config_source: str,
    error_report_dir: str,
    deterministic: bool,
    performance_analysis: bool,
    overhead_analysis: bool,
    stress_test_count: Optional[int],
    concurrency: int,
    parallel_metrics: bool,
    metrics_workers: Optional[int],
):
    photos = agent.get_all_photos()
    if stress_test_count:
        input_folder = config["folders"]["input"]
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(current_dir)
        full_input_path = os.path.join(base_dir, input_folder)
        ensure_stress_test_images(full_input_path, target_count=int(stress_test_count))
        photos = agent.get_all_photos()
    if deterministic:
        random.seed(42)
        agent.oom_probability = 0.0

    if not photos:
        logger.warning("No testable images were found.")
        return None

    batch_report = {
        "schema_version": "2.0",
        "profile": config_profile,
        "batch_id": datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
        "config_used": config["thresholds"],
        "config_source": config_source,
        "execution_mode": "async",
        "async_concurrency": max(1, int(concurrency)),
        "results": [],
    }
    per_image_timeout_s = config.get("runtime", {}).get("async_per_image_timeout_s")
    if per_image_timeout_s is not None:
        try:
            timeout_val = float(per_image_timeout_s)
            if timeout_val > 0:
                batch_report["async_per_image_timeout_s"] = timeout_val
                per_image_timeout_s = timeout_val
            else:
                per_image_timeout_s = None
        except (TypeError, ValueError):
            per_image_timeout_s = None
    if parallel_metrics:
        batch_report["parallel_metrics"] = True
        batch_report["metrics_workers"] = metrics_workers
    perf_samples: List[Dict[str, Any]] = []
    failure_memory_store = FailureMemoryStore()
    photo_ctx = _build_photo_process_context(config, agent)
    process = psutil.Process(os.getpid())
    batch_wall_start = time.perf_counter()
    batch_cpu_start = time.process_time()
    batch_rss_start_mb = process.memory_info().rss / (1024.0 * 1024.0)
    monitor_overhead_baseline = benchmark_monitor_overhead() if overhead_analysis else None
    overhead_counters = {
        "total_image_wall_ms": 0.0,
        "total_model_latency_ms": 0.0,
        "total_framework_wall_ms": 0.0,
        "total_loopback_retry_count": 0,
        "failure_memory_write_ms": 0.0,
        "failure_memory_write_count": 0,
    }

    asyncio.run(
        _run_async_batch_processing(
            photos,
            photo_ctx,
            batch_report,
            perf_samples,
            overhead_counters,
            error_report_dir,
            config_profile,
            config_source,
            concurrency=max(1, int(concurrency)),
            per_image_timeout_s=per_image_timeout_s,
        )
    )

    batch_report["async_timeout_count"] = sum(
        1 for row in batch_report["results"] if bool(row.get("timed_out"))
    )

    return _finalize_batch_report(
        batch_report=batch_report,
        config=config,
        config_profile=config_profile,
        config_source=config_source,
        perf_samples=perf_samples,
        failure_memory_store=failure_memory_store,
        overhead_counters=overhead_counters,
        performance_analysis=performance_analysis,
        overhead_analysis=overhead_analysis,
        batch_wall_start=batch_wall_start,
        batch_cpu_start=batch_cpu_start,
        batch_rss_start_mb=batch_rss_start_mb,
        monitor_overhead_baseline=monitor_overhead_baseline,
        process=process,
    )


def run_profile_comparison(
    profiles,
    inference_backend_override=None,
    loopback_planner_override=None,
    planner_timeout_s_override=None,
    planner_model_override=None,
    planner_require_healthy_override=None,
    replay_mode_override=None,
    replay_file_override=None,
):
    profile_outputs = []
    for profile in profiles:
        logger.info("Running profile: %s", profile)
        result = run_batch_test(
            config_profile=profile,
            inference_backend_override=inference_backend_override,
            loopback_planner_override=loopback_planner_override,
            planner_timeout_s_override=planner_timeout_s_override,
            planner_model_override=planner_model_override,
            planner_require_healthy_override=planner_require_healthy_override,
            replay_mode_override=replay_mode_override,
            replay_file_override=replay_file_override,
            overhead_analysis=False,
        )
        if result:
            profile_outputs.append(result)

    ordered = sorted(
        profile_outputs,
        key=lambda item: (-item["summary"]["pass_rate"], item["summary"]["avg_latency_ms"])
    )

    logger.info("%s", "\nProfile ranking (best to worst):")
    for idx, item in enumerate(ordered, start=1):
        summary = item["summary"]
        logger.info(
            "  #%s %s | pass=%s%% | latency=%sms | decision=%s",
            idx,
            item["profile"],
            summary["pass_rate"],
            summary["avg_latency_ms"],
            summary["release_decision"],
        )

    benchmark_insights = generate_benchmark_insights(profile_outputs, ordered)
    logger.info("%s", "\nBenchmark Insights:")
    for idx, insight in enumerate(benchmark_insights, start=1):
        logger.info("  [%s] Trade-off: %s", idx, insight["trade_off"])
        logger.info("      Observation: %s", insight["observation"])
        logger.info("      Decision implication: %s", insight["decision_implication"])

    comparison_report = {
        "generated_at": datetime.now().isoformat(),
        "profiles": profile_outputs,
        "ranking": [item["profile"] for item in ordered],
        "benchmark_insights": benchmark_insights
    }
    save_comparison_report(comparison_report, "results")
    return comparison_report


def run_repeatability_test(
    profile,
    runs=5,
    inference_backend_override=None,
    loopback_planner_override=None,
    planner_timeout_s_override=None,
    planner_model_override=None,
    planner_require_healthy_override=None,
    replay_mode_override=None,
    replay_file_override=None,
):
    logger.info(
        "Running repeatability test: profile=%s, runs=%s",
        profile,
        runs,
    )
    run_outputs = []
    for run_idx in range(1, runs + 1):
        logger.info("Repeatability run %s/%s", run_idx, runs)
        run_result = run_batch_test(
            config_profile=profile,
            deterministic=True,
            inference_backend_override=inference_backend_override,
            loopback_planner_override=loopback_planner_override,
            planner_timeout_s_override=planner_timeout_s_override,
            planner_model_override=planner_model_override,
            planner_require_healthy_override=planner_require_healthy_override,
            replay_mode_override=replay_mode_override,
            replay_file_override=replay_file_override,
            overhead_analysis=False,
        )
        if run_result:
            run_result["run_index"] = run_idx
            run_outputs.append(run_result)

    if not run_outputs:
        logger.error("Repeatability test failed: no run output produced.")
        return None

    pass_rates = [r["summary"]["pass_rate"] for r in run_outputs]
    latencies = [r["summary"]["avg_latency_ms"] for r in run_outputs]
    image_sets = [tuple(r["image_files"]) for r in run_outputs]
    image_set_consistent = len(set(image_sets)) == 1

    score_history = {}
    for output in run_outputs:
        for rank_item in output["ranking"]:
            score_history.setdefault(rank_item["file"], []).append(rank_item["score"])

    per_image_variance = {
        file_name: round(pvariance(scores), 6) if len(scores) > 1 else 0.0
        for file_name, scores in score_history.items()
    }

    variance_report = {
        "pass_rate_variance": round(pvariance(pass_rates), 6) if len(pass_rates) > 1 else 0.0,
        "avg_latency_variance": round(pvariance(latencies), 6) if len(latencies) > 1 else 0.0,
        "per_image_score_variance": per_image_variance
    }
    variance_report["max_image_score_variance"] = max(per_image_variance.values()) if per_image_variance else 0.0

    decision_distribution = {}
    for output in run_outputs:
        decision = output["summary"]["release_decision"]
        decision_distribution[decision] = decision_distribution.get(decision, 0) + 1

    repeatability_report = {
        "generated_at": datetime.now().isoformat(),
        "profile": profile,
        "runs": runs,
        "same_image_set": image_set_consistent,
        "image_files": list(image_sets[0]) if image_sets else [],
        "variance": variance_report,
        "decision_distribution": decision_distribution
    }
    save_repeatability_report(repeatability_report, "results")

    logger.info("%s", "\nRepeatability summary:")
    logger.info("  - Same image set across runs: %s", image_set_consistent)
    logger.info("  - Pass-rate variance: %s", variance_report["pass_rate_variance"])
    logger.info("  - Avg-latency variance: %s", variance_report["avg_latency_variance"])
    logger.info(
        "  - Max per-image score variance: %s",
        variance_report["max_image_score_variance"],
    )
    logger.info("  - Decision distribution: %s", decision_distribution)
    return repeatability_report


if __name__ == "__main__":
    configure_cli_logging()
    parser = argparse.ArgumentParser(description="Quantized Vision QA batch tester")
    parser.add_argument(
        "--profile",
        default="dev",
        choices=["dev", "benchmark", "base"],
        help="Choose a config profile under the configs directory"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Provide a config path directly (absolute or project-root relative)"
    )
    parser.add_argument(
        "--compare-profiles",
        nargs="+",
        default=None,
        help="Run multiple profiles and print cross-profile ranking"
    )
    parser.add_argument(
        "--repeatability-test",
        default=None,
        choices=["dev", "benchmark", "base"],
        help="Run the same profile multiple times and report variance"
    )
    parser.add_argument(
        "--repeatability-runs",
        type=int,
        default=5,
        help="Number of runs for repeatability test"
    )
    parser.add_argument(
        "--inference-backend",
        default=None,
        choices=["simulated", "ollama_vision", "mock_api", "llama_cpp"],
        help="Temporarily override inference backend without editing config"
    )
    parser.add_argument(
        "--loopback-planner",
        default=None,
        choices=["simulated", "llm"],
        help="Temporarily override loopback planner mode without editing config",
    )
    parser.add_argument(
        "--planner-timeout-s",
        type=float,
        default=None,
        help="Override runtime.loopback_planner.llm.timeout_s from CLI",
    )
    parser.add_argument(
        "--planner-model",
        default=None,
        help="Override runtime.loopback_planner.llm.model from CLI",
    )
    parser.add_argument(
        "--planner-skip-health-check",
        action="store_true",
        help="Skip startup health check for loopback planner in llm mode",
    )
    parser.add_argument(
        "--performance-analysis",
        action="store_true",
        help="Generate optional performance deep-dive report (latency vs image size and CPU)"
    )
    parser.add_argument(
        "--stress-test-100",
        action="store_true",
        help="Ensure at least 100 input images (synthetic variants) before batch run"
    )
    parser.add_argument(
        "--overhead-analysis",
        action="store_true",
        help="Generate overhead report for framework cost (monitoring, loopback, memory writes)"
    )
    parser.add_argument(
        "--async-batch",
        action="store_true",
        help="Run batch with asyncio parallel per-image processing (httpx for HTTP backends)",
    )
    parser.add_argument(
        "--async-concurrency",
        type=int,
        default=4,
        help="Max concurrent images when --async-batch is set (default: 4)",
    )
    parser.add_argument(
        "--async-per-image-timeout-s",
        type=float,
        default=None,
        help="Timeout per image in async batch mode (seconds); timed-out images are marked FAILED",
    )
    parser.add_argument(
        "--async-skip-backend-health-check",
        action="store_true",
        help="Skip fail-fast connectivity check for async inference backend",
    )
    parser.add_argument(
        "--async-backend-health-timeout-s",
        type=float,
        default=None,
        help="Socket timeout (seconds) for async backend startup health check",
    )
    parser.add_argument(
        "--parallel-metrics",
        action="store_true",
        help="Compute sharpness/brightness metrics in a ProcessPoolExecutor (CPU-bound speedup)",
    )
    parser.add_argument(
        "--metrics-workers",
        type=int,
        default=None,
        help="Process pool size for --parallel-metrics (default: min(cpu_count, 8))",
    )
    parser.add_argument(
        "--replay-mode",
        default="off",
        choices=["off", "record", "replay"],
        help="Deterministic replay mode for planner traces (default: off)",
    )
    parser.add_argument(
        "--replay-file",
        default=None,
        help="Replay trace JSONL file path (required for --replay-mode record|replay)",
    )
    args = parser.parse_args()
    if args.replay_mode in {"record", "replay"} and not args.replay_file:
        parser.error("--replay-file is required when --replay-mode is record or replay")
    try:
        if args.repeatability_test:
            run_repeatability_test(
                args.repeatability_test,
                runs=max(1, args.repeatability_runs),
                inference_backend_override=args.inference_backend,
                loopback_planner_override=args.loopback_planner,
                planner_timeout_s_override=args.planner_timeout_s,
                planner_model_override=args.planner_model,
                planner_require_healthy_override=(
                    False if args.planner_skip_health_check else None
                ),
                replay_mode_override=args.replay_mode,
                replay_file_override=args.replay_file,
            )
        elif args.compare_profiles:
            run_profile_comparison(
                args.compare_profiles,
                inference_backend_override=args.inference_backend,
                loopback_planner_override=args.loopback_planner,
                planner_timeout_s_override=args.planner_timeout_s,
                planner_model_override=args.planner_model,
                planner_require_healthy_override=(
                    False if args.planner_skip_health_check else None
                ),
                replay_mode_override=args.replay_mode,
                replay_file_override=args.replay_file,
            )
        else:
            batch_kwargs = dict(
                config_profile=args.profile,
                config_path=args.config,
                inference_backend_override=args.inference_backend,
                loopback_planner_override=args.loopback_planner,
                planner_timeout_s_override=args.planner_timeout_s,
                planner_model_override=args.planner_model,
                planner_require_healthy_override=(
                    False if args.planner_skip_health_check else None
                ),
                async_per_image_timeout_s_override=args.async_per_image_timeout_s,
                async_backend_health_check_override=(
                    False if args.async_skip_backend_health_check else None
                ),
                async_backend_health_timeout_s_override=args.async_backend_health_timeout_s,
                replay_mode_override=args.replay_mode,
                replay_file_override=args.replay_file,
                performance_analysis=args.performance_analysis,
                overhead_analysis=args.overhead_analysis,
                stress_test_count=100 if args.stress_test_100 else None,
                parallel_metrics=args.parallel_metrics,
                metrics_workers=args.metrics_workers,
            )
            if args.async_batch:
                run_batch_test_async(
                    **batch_kwargs,
                    concurrency=max(1, args.async_concurrency),
                )
            else:
                run_batch_test(**batch_kwargs)
    except Exception as e:
        try:
            fallback_profile = args.profile if hasattr(args, "profile") else "dev"
            config, config_source = load_config(profile=fallback_profile, config_path=args.config)
            error_report_dir = config.get("folders", {}).get("logs", "logs/errors")
        except Exception:
            config_source = "UNKNOWN"
            error_report_dir = "logs/errors"

        fatal_error_payload = {
            "generated_at": datetime.now().isoformat(),
            "scope": "pipeline_fatal",
            "profile": getattr(args, "profile", "UNKNOWN"),
            "config_source": config_source,
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
        }
        save_error_report(fatal_error_payload, error_report_dir)
        raise
