import logging
import random
import time

logger = logging.getLogger(__name__)


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

    logger.info("Start monitoring for a new photo...")

    try:
        while time.time() - start_time < timeout:
            current_file = device_controller.get_latest_photo_path()

            if current_file != previous_latest_file:
                logger.info("New file detected: %s", current_file)

                # --- Add AI quality analysis logic ---
                score = vision_model.analyze(current_file)

                if score > 0.8:
                    logger.info("Excellent quality (Score: %s)", score)
                    return True
                if score < 0.5:
                    logger.warning("Quality issue detected: image is too blurry (Score: %s)", score)
                    return False
                logger.info("Quality is borderline; manual review recommended (Score: %s)", score)
                return True

            time.sleep(0.5)

        logger.warning("Monitoring timed out: no new photo found.")
        return False

    except Exception:
        logger.exception("Exception occurred during capture verification")
        return False


# 3. Simulated mobile environment.
class MockDevice:
    def __init__(self, mode="normal"):
        self.has_new_file = False
        self.mode = mode

    def get_latest_photo_path(self):
        if self.mode == "crash":
            raise RuntimeError("OOM: Out of Memory (mobile memory full)")
        if self.has_new_file:
            return "/sdcard/DCIM/IMG_NEW.jpg"
        return "/sdcard/DCIM/IMG_OLD.jpg"


if __name__ == "__main__":
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    my_ai_model = MiniVisionModel()

    success_phone = MockDevice(mode="normal")
    success_phone.has_new_file = True
    logger.info("\n--- Test: AI quality analysis path ---")
    verify_capture_success(success_phone, my_ai_model, "/sdcard/DCIM/IMG_OLD.jpg")

    crash_phone = MockDevice(mode="crash")
    logger.info("\n--- Test: mobile crash path ---")
    verify_capture_success(crash_phone, my_ai_model, "/sdcard/DCIM/IMG_OLD.jpg")
