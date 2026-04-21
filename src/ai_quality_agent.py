import argparse
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from statistics import pvariance

from PIL import Image, ImageDraw

from engine.vision_math import calculate_metrics
from eval.benchmark_evaluator import (
    build_rankings,
    generate_benchmark_insights,
    get_release_decision,
)
from models.llama_quantizer import LlamaQuantizer

DEFAULT_CONFIG = {
    "model_settings": {"name": "Default-Model", "bit_depth": 4},
    "thresholds": {"min_sharpness": 20, "min_brightness": 45, "max_brightness": 220},
    "folders": {"input": "test_images", "output": "results"},
    "runtime": {"oom_probability": 0.0},
    "quality_gate": {"target_pass_rate": 90.0}
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
            print(f"WARNING: Could not remove old report {file_path}: {e}")

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

    print("No input images found. Generated sample dataset in input folder.")


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


class QuantizedVisionAgent:
    def __init__(self, config):
        self.config = config
        self.model_info = config["model_settings"]
        self.brain = LlamaQuantizer(thresholds=config["thresholds"])
        self.oom_probability = float(config.get("runtime", {}).get("oom_probability", 0.0))
        print(f"Startup mode: {self.model_info['name']} ({self.model_info['bit_depth']}-bit)")

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
        metrics = calculate_metrics(photo_path)
        if metrics is None:
            return None, {"decision": "Error", "code": "ERR_SYS_IO_404", "msg": "Unable to read file"}, 0

        ai_result = self.brain.predict_quality(metrics)
        latency = round((time.time() - start_time) * 1000, 2)
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
    print(f"\nBatch test completed. Full report saved to: {file_path}")

    deleted_count = cleanup_old_reports(full_output_path)
    if deleted_count > 0:
        print(f"Cleaned up {deleted_count} report(s) older than {REPORT_RETENTION_DAYS} days.")
    current_report_count = count_reports(full_output_path)
    print(f"Current report count: {current_report_count}")
    return file_path


def save_comparison_report(comparison_data, output_folder):
    comparison_dir = Path(output_folder) / "comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    file_path = comparison_dir / f"profile_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, indent=4, ensure_ascii=False)
    print(f"Comparison report saved to: {file_path}")
    return str(file_path)


def save_repeatability_report(repeatability_data, output_folder):
    repeatability_dir = Path(output_folder) / "repeatability"
    repeatability_dir.mkdir(parents=True, exist_ok=True)
    file_path = repeatability_dir / f"repeatability_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(repeatability_data, f, indent=4, ensure_ascii=False)
    print(f"Repeatability report saved to: {file_path}")
    return str(file_path)


def run_batch_test(config_profile="dev", config_path=None, deterministic=False):
    config, config_source = load_config(profile=config_profile, config_path=config_path)
    print(f"Loaded config source: {config_source}")
    agent = QuantizedVisionAgent(config)
    photos = agent.get_all_photos()
    if deterministic:
        random.seed(42)
        agent.oom_probability = 0.0

    if not photos:
        print("No testable images were found.")
        return None

    batch_report = {
        "schema_version": "2.0",
        "profile": config_profile,
        "batch_id": datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
        "config_used": config["thresholds"],
        "config_source": config_source,
        "results": []
    }

    print(f"Starting to process {len(photos)} image(s)...\n")

    for path in photos:
        file_name = os.path.basename(path)
        try:
            if random.random() < agent.oom_probability:
                raise MemoryError("OOM Exception")

            metrics, ai_result, latency = agent.analyze_photo_quality(path)
            print(f"Processed {file_name}: [{ai_result['code']}] {ai_result['decision']} ({latency}ms)")

            batch_report["results"].append({
                "file": file_name,
                "metrics": metrics,
                "decision": ai_result,
                "latency_ms": latency,
                "status": "SUCCESS"
            })

        except Exception as e:
            print(f"Failed to process file {file_name}: {e}")
            batch_report["results"].append({
                "file": file_name,
                "status": "FAILED",
                "error": str(e)
            })

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
    decision, decision_reason = get_release_decision(pass_rate, avg_lat, config)
    top_ranking = rankings[:3]

    print("\n" + "=" * 55)
    print("Test Dashboard")
    print(f"  - Total tests: {total}")
    print(f"  - Pass rate (Optimal): {pass_rate:.1f}%")
    print(f"  - Average latency: {avg_lat:.2f} ms")
    print(f"  - Release decision: {decision}")
    print("-" * 55)
    print("Top ranking:")
    for item in top_ranking:
        print(f"  #{item['rank']} {item['file']} | score={item['score']} | {item['decision']}")
    print("=" * 55)

    batch_report["summary"] = {
        "total_tests": total,
        "success_count": success_count,
        "pass_rate": round(pass_rate, 2),
        "target_pass_rate": float(config.get("quality_gate", {}).get("target_pass_rate", 90.0)),
        "avg_latency_ms": round(avg_lat, 2),
        "release_decision": decision,
        "decision_reason": decision_reason
    }
    batch_report["ranking"] = rankings

    report_path = save_batch_report(batch_report, config["folders"]["output"])
    return {
        "profile": config_profile,
        "report_path": report_path,
        "summary": batch_report["summary"],
        "top_ranked": top_ranking,
        "ranking": rankings,
        "image_files": sorted([row["file"] for row in batch_report["results"] if "file" in row])
    }


def run_profile_comparison(profiles):
    profile_outputs = []
    for profile in profiles:
        print(f"\nRunning profile: {profile}")
        result = run_batch_test(config_profile=profile)
        if result:
            profile_outputs.append(result)

    ordered = sorted(
        profile_outputs,
        key=lambda item: (-item["summary"]["pass_rate"], item["summary"]["avg_latency_ms"])
    )

    print("\nProfile ranking (best to worst):")
    for idx, item in enumerate(ordered, start=1):
        summary = item["summary"]
        print(
            f"  #{idx} {item['profile']} | pass={summary['pass_rate']}% | "
            f"latency={summary['avg_latency_ms']}ms | decision={summary['release_decision']}"
        )

    benchmark_insights = generate_benchmark_insights(profile_outputs, ordered)
    print("\nBenchmark Insights:")
    for idx, insight in enumerate(benchmark_insights, start=1):
        print(f"  [{idx}] Trade-off: {insight['trade_off']}")
        print(f"      Observation: {insight['observation']}")
        print(f"      Decision implication: {insight['decision_implication']}")

    comparison_report = {
        "generated_at": datetime.now().isoformat(),
        "profiles": profile_outputs,
        "ranking": [item["profile"] for item in ordered],
        "benchmark_insights": benchmark_insights
    }
    save_comparison_report(comparison_report, "results")
    return comparison_report


def run_repeatability_test(profile, runs=5):
    print(f"\nRunning repeatability test: profile={profile}, runs={runs}")
    run_outputs = []
    for run_idx in range(1, runs + 1):
        print(f"\nRepeatability run {run_idx}/{runs}")
        run_result = run_batch_test(config_profile=profile, deterministic=True)
        if run_result:
            run_result["run_index"] = run_idx
            run_outputs.append(run_result)

    if not run_outputs:
        print("Repeatability test failed: no run output produced.")
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

    print("\nRepeatability summary:")
    print(f"  - Same image set across runs: {image_set_consistent}")
    print(f"  - Pass-rate variance: {variance_report['pass_rate_variance']}")
    print(f"  - Avg-latency variance: {variance_report['avg_latency_variance']}")
    print(f"  - Max per-image score variance: {variance_report['max_image_score_variance']}")
    print(f"  - Decision distribution: {decision_distribution}")
    return repeatability_report


if __name__ == "__main__":
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
    args = parser.parse_args()
    if args.repeatability_test:
        run_repeatability_test(args.repeatability_test, runs=max(1, args.repeatability_runs))
    elif args.compare_profiles:
        run_profile_comparison(args.compare_profiles)
    else:
        run_batch_test(config_profile=args.profile, config_path=args.config)
