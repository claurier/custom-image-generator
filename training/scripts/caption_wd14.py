"""
Auto-caption images using WD14 tagger, prepend a trigger token,
and save one .txt caption file per image in dataset/captions/.

WD14 generates tag-style annotations (e.g. "1girl, sword, city, night")
which are well-suited for style LoRA training. Style descriptors are
filtered out so the LoRA learns style from images, not captions.

Requirements:
    pip install onnxruntime huggingface_hub Pillow numpy

Usage:
    python training/scripts/caption_wd14.py \
        --input dataset/processed \
        --output dataset/captions \
        --trigger cmcstyle \
        --threshold 0.35
"""

import argparse
import os
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image

# Tags to exclude — we don't want style/medium descriptors in captions
STYLE_TAGS_BLOCKLIST = {
    "comic", "manga", "illustration", "sketch", "drawing", "painting",
    "watercolor", "ink", "digital art", "pixel art", "oil painting",
    "colored pencil", "monochrome", "greyscale", "lineart", "line art",
    "traditional media", "comic book", "comic strip",
}

WD14_MODEL_REPO = "SmilingWolf/wd-v1-4-convnextv2-tagger-v2"
WD14_MODEL_FILE = "model.onnx"
WD14_LABELS_FILE = "selected_tags.csv"


def load_model():
    import onnxruntime as rt

    model_path = hf_hub_download(WD14_MODEL_REPO, WD14_MODEL_FILE)
    labels_path = hf_hub_download(WD14_MODEL_REPO, WD14_LABELS_FILE)

    session = rt.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    import csv
    labels = []
    with open(labels_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(row["name"])

    return session, labels


def preprocess_image(img_path: Path, size: int = 448) -> np.ndarray:
    with Image.open(img_path) as img:
        img = img.convert("RGB")
        img = img.resize((size, size), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32)
        arr = arr[:, :, ::-1]  # RGB -> BGR
        arr = np.expand_dims(arr, axis=0)
    return arr


def predict_tags(session, labels, img_array: np.ndarray, threshold: float) -> list[str]:
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_array})
    scores = outputs[0][0]

    tags = [
        label.replace("_", " ")
        for label, score in zip(labels, scores)
        if score >= threshold and label.replace("_", " ").lower() not in STYLE_TAGS_BLOCKLIST
    ]
    return tags


def caption_images(input_dir: str, output_dir: str, trigger: str, threshold: float) -> None:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_files = sorted([
        f for f in input_path.iterdir()
        if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    ])

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"Loading WD14 model...")
    session, labels = load_model()
    print(f"Captioning {len(image_files)} images with trigger '{trigger}'...")

    for img_path in image_files:
        out_file = output_path / (img_path.stem + ".txt")
        if out_file.exists():
            print(f"  [SKIP] {img_path.name} — caption already exists")
            continue

        try:
            img_array = preprocess_image(img_path)
            tags = predict_tags(session, labels, img_array, threshold)
            caption = f"{trigger}, {', '.join(tags)}" if tags else trigger
            out_file.write_text(caption, encoding="utf-8")
            print(f"  [OK] {img_path.name} → {caption[:80]}...")
        except Exception as e:
            print(f"  [WARN] {img_path.name}: {e}")

    print("Captioning complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WD14 auto-captioning for FLUX.2 LoRA.")
    parser.add_argument("--input", default="dataset/processed", help="Processed images directory")
    parser.add_argument("--output", default="dataset/captions", help="Output captions directory")
    parser.add_argument("--trigger", default="cmcstyle", help="Trigger token to prepend")
    parser.add_argument("--threshold", type=float, default=0.35, help="WD14 confidence threshold (default: 0.35)")
    args = parser.parse_args()

    caption_images(args.input, args.output, args.trigger, args.threshold)
