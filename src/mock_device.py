import time
import random # 用來模擬 AI 回傳不同的分數

# 1. 模擬 AI 模型類別 (對應妳學的 4-bit 量化模型概念)
class MiniVisionModel:
    def analyze(self, file_path):
        # 在現實中，這裡會執行量化模型的推論 (Inference)
        # 目前我們用 random 模擬 AI 給出的品質分數
        return round(random.uniform(0.3, 1.0), 2)

# 2. 修改後的驗證函數
def verify_capture_success(device_controller, vision_model, previous_latest_file):
    timeout = 5
    start_time = time.time()
    
    print("🔍 開始監測新照片...")
    
    try:
        while time.time() - start_time < timeout:
            current_file = device_controller.get_latest_photo_path()
            
            if current_file != previous_latest_file:
                print(f"✅ 偵測到新檔案: {current_file}")
                
                # --- 加入 AI 品質分析邏輯 ---
                score = vision_model.analyze(current_file)
                
                if score > 0.8:
                    print(f"✨ 品質優良 (Score: {score})")
                    return True
                elif score < 0.5:
                    print(f"⚠️ 發現品質缺陷：照片過於模糊 (Score: {score})")
                    return False
                else:
                    print(f"🤔 品質待確認，建議人工複審 (Score: {score})")
                    return True 
                # --------------------------
            
            time.sleep(0.5)
            
        print("⏳ 監測超時：未發現新照片。")
        return False

    except Exception as e:
        print(f"❌ 測試過程發生異常：{e}")
        return False

# 3. 模擬手機環境
class MockDevice:
    def __init__(self, mode="normal"):
        self.has_new_file = False
        self.mode = mode

    def get_latest_photo_path(self):
        if self.mode == "crash":
            raise Exception("OOM: Out of Memory (手機記憶體滿了)")
        if self.has_new_file:
            return "/sdcard/DCIM/IMG_NEW.jpg"
        return "/sdcard/DCIM/IMG_OLD.jpg"

# --- 執行測試 ---

# 初始化 AI 模型
my_ai_model = MiniVisionModel()

# 情境 1：成功拍到照片，由 AI 檢查品質
success_phone = MockDevice(mode="normal")
success_phone.has_new_file = True
print("\n--- 測試：AI 品質分析路徑 ---")
verify_capture_success(success_phone, my_ai_model, "/sdcard/DCIM/IMG_OLD.jpg")

# 情境 2：手機崩潰測試
crash_phone = MockDevice(mode="crash")
print("\n--- 測試：手機崩潰路徑 ---")
verify_capture_success(crash_phone, my_ai_model, "/sdcard/DCIM/IMG_OLD.jpg")