"""
Preprocess raw images for FLUX.2 [klein] LoRA training.

- Resizes images to target resolution (default 1024px on the shortest side)
- Center-crops to square
- Saves as PNG to dataset/processed/
- Skips non-image files and already-processed images

Usage:
    python training/scripts/preprocess.py --input dataset/raw --output dataset/processed --size 1024
"""

import argparse
import os
from pathlib import Path

from PIL import Image


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def resize_and_crop(img: Image.Image, size: int) -> Image.Image:
    """Resize shortest side to `size`, then center-crop to square."""
    w, h = img.size
    scale = size / min(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)

    # Center crop
    left = (new_w - size) // 2
    top = (new_h - size) // 2
    img = img.crop((left, top, left + size, top + size))
    return img


def preprocess(input_dir: str, output_dir: str, size: int) -> None:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_files = [
        f for f in input_path.iterdir()
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_files)} images. Processing to {size}x{size}...")

    skipped, processed = 0, 0
    for img_path in sorted(image_files):
        out_file = output_path / (img_path.stem + ".png")
        if out_file.exists():
            skipped += 1
            continue
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                img = resize_and_crop(img, size)
                img.save(out_file, format="PNG")
                processed += 1
        except Exception as e:
            print(f"  [WARN] Skipping {img_path.name}: {e}")

    print(f"Done. {processed} processed, {skipped} already existed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess images for FLUX.2 LoRA training.")
    parser.add_argument("--input", default="dataset/raw", help="Input directory of raw images")
    parser.add_argument("--output", default="dataset/processed", help="Output directory for processed images")
    parser.add_argument("--size", type=int, default=1024, help="Target square resolution (default: 1024)")
    args = parser.parse_args()

    preprocess(args.input, args.output, args.size)
