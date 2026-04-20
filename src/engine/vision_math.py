import os
import statistics
from PIL import Image

def calculate_metrics(photo_path):
    """
    讀取真實圖片並計算其亮度與銳利度指標
    """
    if not os.path.exists(photo_path):
        print(f"⚠️ 找不到路徑: {photo_path}")
        return None

    try:
        with Image.open(photo_path) as img:
            # 1. 轉為灰階 (L 模式)
            img_gray = img.convert("L")
            # 2. 縮放圖片提升計算效率 (128x128)
            img_small = img_gray.resize((128, 128))
            # 3. 強制確保清單內全是整數，避免型別報錯
            pixels = [int(p) for p in list(img_small.getdata())]

        if not pixels:
            return None

        # 計算並回傳數據字典
        return {
            "sharpness": round(statistics.stdev(pixels), 2),
            "avg_brightness": round(statistics.mean(pixels), 2),
            "max_brightness": int(max(pixels))
        }
    except Exception as e:
        print(f"❌ 影像引擎運算失敗: {e}")
        return None