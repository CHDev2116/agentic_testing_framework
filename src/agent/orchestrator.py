import json
import time

from models.gemma_filter import GemmaFilter
from models.llama_analyst import LlamaAnalyst

class QualityOrchestrator:
    def __init__(self):
        self.gemma_filter = GemmaFilter()
        self.llama_analyst = LlamaAnalyst()

    def run_pipeline(self, image_metrics):
        image_id = image_metrics.get("id", "Unknown_IMG")
        print(f"\n--- [Pipeline Start] Analyzing {image_id} ---")

        # --- Stage 1: 快速過濾 ---
        print(f"Step 1: Running Gemma-2b for basic check...")
        gemma_raw_response = self.gemma_filter.check_basic_quality(image_metrics)
        
        gemma_res = self._parse_json(gemma_raw_response)
        if not gemma_res or not gemma_res.get("pass"):
            reason = gemma_res.get('reason') if gemma_res else "Gemma analysis failed"
            print(f"❌ Rejected by Gemma: {reason}")
            return {
                "id": image_id,
                "final_verdict": "FAIL",
                "stage": "Filter",
                "details": gemma_res
            }

        print(f"✅ Passed Gemma Filter. Reason: {gemma_res.get('reason')}")
        
        # 在兩個大模型切換間隙，讓 CPU 稍微冷卻 0.5 秒
        time.sleep(0.5)

        # --- Stage 2: 深度分析 ---
        print(f"Step 2: Dispatching to Llama-3.1 for deep analysis...")
        llama_raw_response = self.llama_analyst.analyze_quality(image_metrics)
        
        llama_res = self._parse_json(llama_raw_response)
        if not llama_res or llama_res.get("verdict") == "Error":
            print(f"⚠️ Llama Analysis stopped by Safety Guard.")
            return {"id": image_id, "error": "Llama analysis timeout or error"}

        # --- Stage 3: 彙整最終報告 ---
        print(f"✅ Final Verdict: {llama_res.get('verdict')}")
        
        return {
            "id": image_id,
            "final_verdict": llama_res.get("verdict"),
            "stage": "Full Pipeline",
            "filter_check": "PASS",
            "detailed_analysis": llama_res.get("analysis"),
            "engine": "Llama-3.1-8b-Q4_K_M"
        }

    def _parse_json(self, text):
        """
        終極 JSON 解析器：處理大小寫、多餘文字及編碼問題
        """
        if not text: return None
        try:
            # 修正 Python vs JSON 布林值與空值
            processed_text = text.replace(": True", ": true").replace(": False", ": false").replace(": None", ": null")
            
            start_idx = processed_text.find('{')
            end_idx = processed_text.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                json_str = processed_text[start_idx:end_idx + 1]
                return json.loads(json_str)
            return None
        except Exception as e:
            print(f"Parsing error logic triggered. Raw snippet: {text[:50]}...")
            return None

if __name__ == "__main__":
    # 測試用例
    test_metrics = {
        "id": "Test_Photo_PASS_CASE",
        "brightness": 120,
        "sharpness": 85,
        "noise_level": 12
    }
    
    orchestrator = QualityOrchestrator()
    report = orchestrator.run_pipeline(test_metrics)
    
    print("\n--- [Final Report Summary] ---")
    print(json.dumps(report, indent=4))