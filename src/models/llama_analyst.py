import json
import logging
import time

import requests

logger = logging.getLogger(__name__)


class LlamaAnalyst:
    def __init__(self):
        # 預設使用 completion 接口以獲得最高穩定性
        self.base_url = "http://127.0.0.1:8080"
        self.completion_url = f"{self.base_url}/completion"

    def analyze_quality(self, metrics):
        start_time = time.time()

        # 任務一：優化 Prompt 結構 (Prefix Prompting)
        prompt = f"""Analyze these camera metrics and return JSON.
Metrics: {metrics}
Return format: {{"verdict": "Good", "analysis": "Reason"}}

JSON Result:
{{"""

        payload = {
            "prompt": prompt,
            "temperature": 0.0,
            "max_tokens": 150,
            "stop": ["}", "\n\n"],
        }

        try:
            response = requests.post(self.completion_url, json=payload, timeout=30)
            response.raise_for_status()

            res_data = response.json()
            content = res_data.get("content", "").strip()

            # 手動補回左大括號並確保閉合
            full_json = "{" + content
            if not full_json.endswith("}"):
                full_json += "}"

            # 任務二：量化指標計算
            end_time = time.time()
            duration = end_time - start_time

            # 估算 Token 數量 (英文約 4 字母一個 token，這在無 usage 回傳時是專業的替代方案)
            estimated_tokens = len(content) // 4
            tps = estimated_tokens / duration if duration > 0 else 0

            logger.info("\n--- [Llama Performance Report] ---")
            logger.info("Total Latency  : %.2fs", duration)
            logger.info("Est. Tokens    : %s", estimated_tokens)
            logger.info("Throughput     : %.2f TPS", tps)
            logger.info("----------------------------------\n")

            return full_json

        except Exception:
            logger.exception("Llama inference failed")
            return json.dumps({"verdict": "Error", "analysis": "Pipeline failed."})
