# Custom Image Generator — FLUX.2 [klein] Comic Style LoRA

A project about using a base model for images and fine tuning it with an image dataset.

The first use-case is to generate comic strips for a comic author.
He provides all his images so the model can learn his style.

Fine-tune [FLUX.2-klein-base-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) on a comic strip image dataset using LoRA, runnable on a single RTX 4090 (24GB).

---

## Project Structure

```
custom-image-generator/
├── dataset/
│   ├── raw/              # Drop your original images here
│   ├── processed/        # Resized 1024x1024 PNGs (auto-generated)
│   ├── captions/         # One .txt caption per image (auto-generated)
│   └── metadata.json     # Trigger token, style notes
│
├── training/
│   ├── configs/
│   │   ├── ai-toolkit/   # ostris/ai-toolkit YAML config
│   │   └── simpleTuner/  # SimpleTuner JSON config
│   ├── scripts/
│   │   ├── preprocess.py       # Resize & crop images to 1024px
│   │   ├── caption_wd14.py     # WD14 auto-captioning
│   │   └── train_dreambooth.sh # diffusers DreamBooth LoRA training
│   └── outputs/          # Checkpoints saved here (git-ignored)
│
├── inference/
│   ├── generate.py       # RTX 4090 inference (CUDA / diffusers)
│   ├── generate_mac.py   # Apple Silicon inference (mflux, 8 GB safe)
│   └── test_prompts.txt  # Sample prompts for evaluation
│
├── loras/                # Final .safetensors LoRA weights (git-ignored)
├── requirements.txt      # RTX 4090 / CUDA dependencies
├── requirements_mac.txt  # Apple Silicon (mflux) dependencies
└── .gitignore
```

---

## Quickstart

### 1. Install dependencies
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
huggingface-cli login
```

### 2. Add your images
Copy your ~1000 comic strip images into `dataset/raw/`.

### 3. Preprocess images
```bash
python training/scripts/preprocess.py
```

### 4. Auto-caption with WD14
```bash
python training/scripts/caption_wd14.py --trigger cmcstyle
```
Review a few captions in `dataset/captions/` and adjust as needed. Ensure no style descriptors leak in (the script filters common ones automatically).

### 5. Train the LoRA

**Option A — diffusers DreamBooth (simplest):**
```bash
bash training/scripts/train_dreambooth.sh
```

**Option B — ostris/ai-toolkit (fastest iteration):**
```bash
# From ai-toolkit repo root:
python run.py /path/to/this/project/training/configs/ai-toolkit/config.yaml
```

**Option C — SimpleTuner (most stable):**
```bash
# From SimpleTuner repo root — update paths in config.json first
python train.py --config_path /path/to/training/configs/simpleTuner/config.json
```

### 6. Run inference

**On RTX 4090 (Linux / Windows):**
```bash
# Single prompt
python inference/generate.py \
  --lora loras/cmcstyle-v1.safetensors \
  --prompt "cmcstyle, a detective in a rainy alley" \
  --steps 20

# Batch from test_prompts.txt
python inference/generate.py \
  --lora loras/cmcstyle-v1.safetensors \
  --prompts inference/test_prompts.txt \
  --output_dir inference/samples/
```

**On Apple Silicon Mac (M1/M2/M3/M4):**
```bash
# Install Mac-specific deps (do NOT use requirements.txt on Mac)
pip install -r requirements_mac.txt

# Single prompt — downloads ~16 GB on first run
# Model alias "flux2-klein-4b" = FLUX.2-klein-4B (distilled, 4-step, default)
python inference/generate_mac.py \
  --prompt "cmcstyle, a detective in a rainy alley at night"

# With a trained LoRA
python inference/generate_mac.py \
  --lora loras/cmcstyle-v1.safetensors \
  --prompt "cmcstyle, a superhero leaping across rooftops"

# Batch
python inference/generate_mac.py \
  --prompts inference/test_prompts.txt \
  --output_dir inference/samples/
```

> **8 GB Mac:** Default settings (Q4, 512×512, 4 steps) are tuned for 8 GB unified memory (~4 GB peak).
> Try `--width 768 --height 768` only if no other apps are running.
> If you still OOM, use [iris.c](https://github.com/antirez/iris.c) as a fallback (~4–5 GB peak via memory-mapped weights).
>
> **16 GB Mac:** Use `--quantize 8 --width 1024 --height 1024` for full quality.

---

## Key Parameters

| Parameter | Value | Notes |
|---|---|---|
| Base model | FLUX.2-klein-base-9B | Use base, not distilled, for training |
| Trigger token | `cmcstyle` | Unique non-English word |
| LoRA rank | 32 | Increase to 64 if style underfits |
| Training steps | 3000–5000 | Sample every 250 steps to monitor |
| Quantization | int8 | Required to fit 9B on 24GB |
| Inference (quality) | 20–50 steps | ~8–15s per image on RTX 4090 |
| Inference (speed) | 4 steps (distilled) | <1s per image on RTX 4090 |
| Mac inference model | FLUX.2-klein-distilled-4B | Only 4B distilled is safe on 8 GB |
| Mac quantization | Q4 | ~4 GB peak; use Q8 on 16 GB |
| Mac resolution | 512×512 default | Try 768px if RAM allows |
| Mac inference speed | 4 steps | ~30–90s per image on M3 Air 8 GB |

---

## Captioning Rules

- **Use** your trigger token at the start of every caption
- **Describe the content** of the image (characters, setting, action)
- **Do NOT describe the style** — no "comic", "illustration", "line art", etc.
- The LoRA learns the style from the images, not the captions

---

## Resources

- [FLUX.2 klein Training Docs — BFL](https://docs.bfl.ai/flux_2/flux2_klein_training)
- [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit)
- [SimpleTuner FLUX2 Quickstart](http://docs.simpletuner.io/quickstart/FLUX2/)
- [diffusers DreamBooth FLUX.2 README](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_flux2.md)
