import time
import random # Used to simulate AI quality scores.

# 1. Simulated AI model class.
class MiniVisionModel:
    def analyze(self, file_path):
        # In production this would run quantized-model inference.
        # For now, simulate an AI quality score with random values.
        return round(random.uniform(0.3, 1.0), 2)

# 2. Updated verification function.
def verify_capture_success(device_controller, vision_model, previous_latest_file):
    timeout = 5
    start_time = time.time()
    
    print("🔍 Start monitoring for a new photo...")
    
    try:
        while time.time() - start_time < timeout:
            current_file = device_controller.get_latest_photo_path()
            
            if current_file != previous_latest_file:
                print(f"✅ New file detected: {current_file}")
                
                # --- Add AI quality analysis logic ---
                score = vision_model.analyze(current_file)
                
                if score > 0.8:
                    print(f"✨ Excellent quality (Score: {score})")
                    return True
                elif score < 0.5:
                    print(f"⚠️ Quality issue detected: image is too blurry (Score: {score})")
                    return False
                else:
                    print(f"🤔 Quality is borderline; manual review recommended (Score: {score})")
                    return True 
                # --------------------------
            
            time.sleep(0.5)
            
        print("⏳ Monitoring timed out: no new photo found.")
        return False

    except Exception as e:
        print(f"❌ Exception occurred during test: {e}")
        return False

# 3. Simulated mobile environment.
class MockDevice:
    def __init__(self, mode="normal"):
        self.has_new_file = False
        self.mode = mode

    def get_latest_photo_path(self):
        if self.mode == "crash":
            raise Exception("OOM: Out of Memory (mobile memory full)")
        if self.has_new_file:
            return "/sdcard/DCIM/IMG_NEW.jpg"
        return "/sdcard/DCIM/IMG_OLD.jpg"

# --- Run tests ---

# Initialize AI model.
my_ai_model = MiniVisionModel()

# Scenario 1: capture succeeds and AI checks quality.
success_phone = MockDevice(mode="normal")
success_phone.has_new_file = True
print("\n--- Test: AI quality analysis path ---")
verify_capture_success(success_phone, my_ai_model, "/sdcard/DCIM/IMG_OLD.jpg")

# Scenario 2: mobile crash path.
crash_phone = MockDevice(mode="crash")
print("\n--- Test: mobile crash path ---")
verify_capture_success(crash_phone, my_ai_model, "/sdcard/DCIM/IMG_OLD.jpg")