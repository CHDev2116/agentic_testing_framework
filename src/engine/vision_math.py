import os
import statistics
from PIL import Image

def calculate_metrics(photo_path):
    """
    Load a real image and calculate brightness and sharpness metrics.
    """
    if not os.path.exists(photo_path):
        print(f"⚠️ Path not found: {photo_path}")
        return None

    try:
        with Image.open(photo_path) as img:
            # 1. Convert to grayscale (L mode).
            img_gray = img.convert("L")
            # 2. Downscale image for faster computation (128x128).
            img_small = img_gray.resize((128, 128))
            # 3. Ensure all pixel values are integers.
            pixels = [int(p) for p in list(img_small.getdata())]

        if not pixels:
            return None

        # Return computed metric dictionary.
        return {
            "sharpness": round(statistics.stdev(pixels), 2),
            "avg_brightness": round(statistics.mean(pixels), 2),
            "max_brightness": int(max(pixels))
        }
    except Exception as e:
        print(f"❌ Image engine computation failed: {e}")
        return None