import os
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter


class ImageProcessor:
    def __init__(self, output_dir="results/loopback_cache"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def adjust_brightness(self, image_path, level, file_stem, attempt_idx):
        with Image.open(image_path) as img:
            enhancer = ImageEnhance.Brightness(img)
            adjusted = enhancer.enhance(level)
            output_path = self.output_dir / f"{file_stem}_retry{attempt_idx}_bright.png"
            adjusted.save(output_path)
        return os.fspath(output_path)

    def apply_sharpen(self, image_path, file_stem, attempt_idx):
        with Image.open(image_path) as img:
            sharpened = img.filter(ImageFilter.SHARPEN)
            output_path = self.output_dir / f"{file_stem}_retry{attempt_idx}_sharp.png"
            sharpened.save(output_path)
        return os.fspath(output_path)
