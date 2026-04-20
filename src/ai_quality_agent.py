import json
import os
import time
import random
from datetime import datetime

# 引入自定義模組
from engine.vision_math import calculate_metrics
from models.llama_quantizer import LlamaQuantizer

def load_config():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)
    config_path = os.path.join(base_dir, "config.json")
    
    if not os.path.exists(config_path):
        return {
            "model_settings": {"name": "Default-Model", "bit_depth": 4},
            "thresholds": {"min_sharpness": 20, "min_brightness": 45, "max_brightness": 220},
            "folders": {"input": "test_images", "output": "results"}
        }
        
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

class QuantizedVisionAgent:
    def __init__(self, config):
        self.config = config
        self.model_info = config["model_settings"]
        self.brain = LlamaQuantizer(thresholds=config["thresholds"])
        print(f"🚀 啟動模式: {self.model_info['name']} ({self.model_info['bit_depth']}-bit)")

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
            return None, {"decision": "Error", "code": "ERR_SYS_IO_404", "msg": "無法讀取檔案"}, 0
        
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
    print(f"\n✅ 批次測試結束！完整報告已存至：{file_path}")

def run_batch_test():
    config = load_config()
    agent = QuantizedVisionAgent(config)
    photos = agent.get_all_photos()
    
    if not photos:
        print("❌ 沒有發現可測試的照片。")
        return

    batch_report = {
        "batch_id": datetime.now().strftime('%Y%m%d_%H%M%S'),
        "config_used": config["thresholds"],
        "results": []
    }

    print(f"📋 開始處理 {len(photos)} 張照片...\n")

    for path in photos:
        file_name = os.path.basename(path)
        try:
            if random.random() < 0.05:
                raise MemoryError("OOM Exception")

            metrics, ai_result, latency = agent.analyze_photo_quality(path)
            
            # 優化輸出：直接顯示代碼與訊息
            print(f"🔹 處理 {file_name}: [{ai_result['code']}] {ai_result['decision']} ({latency}ms)")
            
            batch_report["results"].append({
                "file": file_name,
                "metrics": metrics,
                "decision": ai_result,
                "latency_ms": latency,
                "status": "SUCCESS"
            })

        except Exception as e:
            print(f"💥 檔案 {file_name} 處理失敗: {e}")
            batch_report["results"].append({
                "file": file_name,
                "status": "FAILED",
                "error": str(e)
            })

    # --- 4. 統計與自我診斷邏輯 (確保在 run_batch_test 函數內) ---
    total = len(batch_report["results"])
    success_count = sum(
        1 for r in batch_report["results"] 
        if isinstance(r.get("decision"), dict) and r["decision"].get("code") == "SUCCESS_200"
    )
    
    pass_rate = (success_count / total) * 100 if total > 0 else 0
    successful_latencies = [r.get("latency_ms", 0) for r in batch_report["results"] if r.get("status") == "SUCCESS"]
    avg_lat = sum(successful_latencies) / len(successful_latencies) if successful_latencies else 0

    print("\n" + "="*55)
    print(f"📊 測試統計看板 (System Diagnosis Mode)")
    print(f"  - 總測試量: {total}")
    print(f"  - 合格率 (Optimal): {pass_rate:.1f}%")
    print(f"  - 平均處理延遲: {avg_lat:.2f} ms")
    print("-" * 55)

    if pass_rate < 100:
        print(f"🤖 [Agent 診斷]: 偵測到部分樣本未達標 (合格率: {pass_rate:.1f}%)。")
        
        all_results = [r.get("decision", {}) for r in batch_report["results"] if isinstance(r.get("decision"), dict)]
        error_codes = [res.get("code", "") for res in all_results if res.get("code") != "SUCCESS_200"]
        
        if error_codes:
            unique_errors = set(error_codes)
            print(f"🔎 偵測到異常代碼: {unique_errors}")
            
            print("\n💡 建議行動方案：")
            if any("ERR_OPTIC_SHRP" in c for c in unique_errors):
                print(f"  ⚠️ [SHRP] 銳利度不足。建議清潔鏡頭或調整 min_sharpness 設定。")
            if any("ERR_LIGHT" in c for c in unique_errors):
                print(f"  ⚠️ [LIGHT] 曝光異常。建議優化環境光源。")
    else:
        print("🤖 [Agent 診斷]: 測試表現完美！所有樣本均符合 4-bit 模型推論門檻。")

    print("="*55)

    # 5. 產出報告
    save_batch_report(batch_report, config["folders"]["output"])

if __name__ == "__main__":
    run_batch_test()