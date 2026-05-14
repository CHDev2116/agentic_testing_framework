import cv2
import numpy as np

class ImageQualityValidator:
    def __init__(self, brightness_threshold=0.7, dark_threshold=0.7):
        # Threshold rule: if 70% of pixels are too bright, mark as overexposed.
        self.brightness_threshold = brightness_threshold
        self.dark_threshold = dark_threshold

    def analyze_exposure(self, image_path):
        # 1. Load image and convert to grayscale (only brightness matters here).
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return "Error: Image not found"

        # 2. Compute histogram (0-255).
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        
        # 3. Normalize histogram to percentages.
        hist_norm = hist / hist.sum()

        # 4. Statistical checks.
        # Dark check: ratio of pixels in range 0-50.
        dark_pixels = np.sum(hist_norm[:50])
        # Bright check: ratio of pixels in range 205-255.
        bright_pixels = np.sum(hist_norm[205:])

        result = {
            "dark_ratio": round(float(dark_pixels), 2),
            "bright_ratio": round(float(bright_pixels), 2),
            "verdict": "Pass"
        }

        # 5. Rule validation.
        if dark_pixels > self.dark_threshold:
            result["verdict"] = "Fail: Too Dark"
        elif bright_pixels > self.brightness_threshold:
            result["verdict"] = "Fail: Overexposed"
            
        return result
