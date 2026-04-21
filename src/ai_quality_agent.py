import argparse
import json
import os
import time
import random
from datetime import datetime

# Import custom modules
from engine.vision_math import calculate_metrics
from models.llama_quantizer import LlamaQuantizer

DEFAULT_CONFIG = {
    "model_settings": {"name": "Default-Model", "bit_depth": 4},
    "thresholds": {"min_sharpness": 20, "min_brightness": 45, "max_brightness": 220},
    "folders": {"input": "test_images", "output": "results"}
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
            print(f"⚠️ Could not remove old report {file_path}: {e}")

    return deleted_count


def count_reports(output_folder_path):
    report_count = 0
    for file_name in os.listdir(output_folder_path):
        if file_name.startswith("batch_report_") and file_name.endswith(".json"):
            file_path = os.path.join(output_folder_path, file_name)
            if os.path.isfile(file_path):
                report_count += 1
    return report_count


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
        print(f"🚀 Startup mode: {self.model_info['name']} ({self.model_info['bit_depth']}-bit)")

    def get_all_photos(self):
        folder_name = self.config["folders"]["input"]
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(current_dir)
        full_path = os.path.join(base_dir, folder_name)
        
        if not os.path.exists(full_path):
            os.makedirs(full_path)
            return []
            
        valid_extensions = ('.jpg', '.jpeg', '.png')
        return [os.path.join(full_path, f) for f in os.listdir(full_path) if f.lower().endswith(valid_extensions)]

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
    
    file_name = f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    file_path = os.path.join(full_output_path, file_name)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=4, ensure_ascii=False)
    print(f"\n✅ Batch test completed. Full report saved to: {file_path}")
    deleted_count = cleanup_old_reports(full_output_path)
    if deleted_count > 0:
        print(f"🧹 Cleaned up {deleted_count} report(s) older than {REPORT_RETENTION_DAYS} days.")
    current_report_count = count_reports(full_output_path)
    print(f"📁 Current report count: {current_report_count}")

def run_batch_test(config_profile="dev", config_path=None):
    config, config_source = load_config(profile=config_profile, config_path=config_path)
    print(f"🧩 Loaded config source: {config_source}")
    agent = QuantizedVisionAgent(config)
    photos = agent.get_all_photos()
    
    if not photos:
        print("❌ No testable images were found.")
        return

    batch_report = {
        "batch_id": datetime.now().strftime('%Y%m%d_%H%M%S'),
        "config_used": config["thresholds"],
        "results": []
    }

    print(f"📋 Starting to process {len(photos)} image(s)...\n")

    for path in photos:
        file_name = os.path.basename(path)
        try:
            if random.random() < 0.05:
                raise MemoryError("OOM Exception")

            metrics, ai_result, latency = agent.analyze_photo_quality(path)
            
            # Optimized output: show result code and message directly.
            print(f"🔹 Processed {file_name}: [{ai_result['code']}] {ai_result['decision']} ({latency}ms)")
            
            batch_report["results"].append({
                "file": file_name,
                "metrics": metrics,
                "decision": ai_result,
                "latency_ms": latency,
                "status": "SUCCESS"
            })

        except Exception as e:
            print(f"💥 Failed to process file {file_name}: {e}")
            batch_report["results"].append({
                "file": file_name,
                "status": "FAILED",
                "error": str(e)
            })

    # --- 4. Statistics and self-diagnosis logic ---
    total = len(batch_report["results"])
    success_count = sum(
        1 for r in batch_report["results"] 
        if isinstance(r.get("decision"), dict) and r["decision"].get("code") == "SUCCESS_200"
    )
    
    pass_rate = (success_count / total) * 100 if total > 0 else 0
    successful_latencies = [r.get("latency_ms", 0) for r in batch_report["results"] if r.get("status") == "SUCCESS"]
    avg_lat = sum(successful_latencies) / len(successful_latencies) if successful_latencies else 0

    print("\n" + "="*55)
    print(f"📊 Test Dashboard (System Diagnosis Mode)")
    print(f"  - Total tests: {total}")
    print(f"  - Pass rate (Optimal): {pass_rate:.1f}%")
    print(f"  - Average latency: {avg_lat:.2f} ms")
    print("-" * 55)

    if pass_rate < 100:
        print(f"🤖 [Agent Diagnosis]: Some samples did not meet the target (pass rate: {pass_rate:.1f}%).")
        
        all_results = [r.get("decision", {}) for r in batch_report["results"] if isinstance(r.get("decision"), dict)]
        error_codes = [res.get("code", "") for res in all_results if res.get("code") != "SUCCESS_200"]
        
        if error_codes:
            unique_errors = set(error_codes)
            print(f"🔎 Detected error codes: {unique_errors}")
            
            print("\n💡 Recommended actions:")
            if any("ERR_OPTIC_SHRP" in c for c in unique_errors):
                print("  ⚠️ [SHRP] Insufficient sharpness. Clean the lens or adjust min_sharpness.")
            if any("ERR_LIGHT" in c for c in unique_errors):
                print("  ⚠️ [LIGHT] Exposure issue detected. Improve ambient lighting.")
    else:
        print("🤖 [Agent Diagnosis]: Perfect run. All samples passed the 4-bit model thresholds.")

    print("="*55)

    # 5. Export report
    save_batch_report(batch_report, config["folders"]["output"])

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
    args = parser.parse_args()
    run_batch_test(config_profile=args.profile, config_path=args.config)