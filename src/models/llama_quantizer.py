class LlamaQuantizer:
    def __init__(self, thresholds):
        self.thresholds = thresholds

    def predict_quality(self, metrics):
        # 1. Extract physical metrics.
        sharpness = metrics.get("sharpness", 0)
        brightness = metrics.get("avg_brightness", 0)
        
        # 2. Read thresholds from config with safe defaults.
        min_s = self.thresholds.get("min_sharpness", 20.0)
        min_b = self.thresholds.get("min_brightness", 45.0)
        max_b = self.thresholds.get("max_brightness", 220.0)

        # 3. First-layer logic: physical environment validation.
        # If brightness is extreme, sharpness is unreliable.
        if brightness < min_b:
            return {
                "decision": "Under-exposed",
                "code": "ERR_LIGHT_DARK_002",
                "msg": f"Environment is too dark ({brightness:.2f}); sharpness check is invalid."
            }
        elif brightness > max_b:
            return {
                "decision": "Over-exposed",
                "code": "ERR_LIGHT_BRGT_003",
                "msg": f"Environment is too bright/overexposed ({brightness:.2f}); edge detail cannot be measured."
            }

        # 4. Second-layer logic: optical quality validation.
        # Check sharpness only under valid lighting conditions.
        if sharpness < min_s:
            return {
                "decision": "Blurry",
                "code": "ERR_OPTIC_SHRP_001",
                "msg": f"Sharpness {sharpness:.2f} is below threshold {min_s}."
            }
            
        # 5. Passed all checks.
        return {
            "decision": "Optimal",
            "code": "SUCCESS_200",
            "msg": "Quality meets 4-bit model inference standards."
        }