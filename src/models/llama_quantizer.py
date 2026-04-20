class LlamaQuantizer:
    def __init__(self, thresholds):
        self.thresholds = thresholds

    def predict_quality(self, metrics):
        # 1. 提取物理指標
        sharpness = metrics.get("sharpness", 0)
        brightness = metrics.get("avg_brightness", 0)
        
        # 2. 從設定檔讀取門檻 (給予預設值以防萬一)
        min_s = self.thresholds.get("min_sharpness", 20.0)
        min_b = self.thresholds.get("min_brightness", 45.0)
        max_b = self.thresholds.get("max_brightness", 220.0)

        # 3. 【第一層邏輯：物理環境判定】
        # 如果亮度極端，則銳利度數據不可信，優先報亮度錯誤
        if brightness < min_b:
            return {
                "decision": "Under-exposed",
                "code": "ERR_LIGHT_DARK_002",
                "msg": f"環境過暗 ({brightness:.2f})，銳利度檢測已失效。"
            }
        elif brightness > max_b:
            return {
                "decision": "Over-exposed",
                "code": "ERR_LIGHT_BRGT_003",
                "msg": f"環境過亮/爆光 ({brightness:.2f})，無法測量邊緣細節。"
            }

        # 4. 【第二層邏輯：光學品質判定】
        # 只有在光線正常的情況下，才檢查銳利度
        if sharpness < min_s:
            return {
                "decision": "Blurry",
                "code": "ERR_OPTIC_SHRP_001",
                "msg": f"銳利度 {sharpness:.2f} 低於門檻 {min_s}。"
            }
            
        # 5. 通過所有檢查
        return {
            "decision": "Optimal",
            "code": "SUCCESS_200",
            "msg": "品質符合 4-bit 模型推論標準。"
        }