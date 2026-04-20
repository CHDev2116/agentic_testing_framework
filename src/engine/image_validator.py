import cv2
import numpy as np

class ImageQualityValidator:
    def __init__(self, brightness_threshold=0.7, dark_threshold=0.7):
        # 設定閾值：例如若 70% 的像素都太亮，則判定為過曝
        self.brightness_threshold = brightness_threshold
        self.dark_threshold = dark_threshold

    def analyze_exposure(self, image_path):
        # 1. 讀取影像並轉為灰階（因為我們只關心亮度）
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return "Error: Image not found"

        # 2. 計算直方圖 (0-255)
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        
        # 3. 正規化直方圖（轉為百分比）
        hist_norm = hist / hist.sum()

        # 4. 統計分析
        # 判定過暗：像素值在 0-50 之間的比例
        dark_pixels = np.sum(hist_norm[:50])
        # 判定過曝：像素值在 205-255 之間的比例
        bright_pixels = np.sum(hist_norm[205:])

        result = {
            "dark_ratio": round(float(dark_pixels), 2),
            "bright_ratio": round(float(bright_pixels), 2),
            "verdict": "Pass"
        }

        # 5. 規則驗證
        if dark_pixels > self.dark_threshold:
            result["verdict"] = "Fail: Too Dark"
        elif bright_pixels > self.brightness_threshold:
            result["verdict"] = "Fail: Overexposed"
            
        return result

# 測試使用
# validator = ImageQualityValidator()
# print(validator.analyze_exposure("test_photo.jpg"))