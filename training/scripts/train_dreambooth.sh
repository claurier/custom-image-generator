#!/usr/bin/env bash
# ============================================================
# FLUX.2 [klein] LoRA Training — DreamBooth (diffusers)
# Target: FLUX.2-klein-base-9B on RTX 4090 (24GB)
#
# Prerequisites:
#   pip install -r requirements.txt
#   huggingface-cli login
#
# Run from project root:
#   bash training/scripts/train_dreambooth.sh
# ============================================================

set -euo pipefail

# ---- Paths -----------------------------------------------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASET_DIR="$PROJECT_ROOT/dataset/processed"
CAPTIONS_DIR="$PROJECT_ROOT/dataset/captions"
OUTPUT_DIR="$PROJECT_ROOT/training/outputs/lora-$(date +%Y%m%d-%H%M)"

# ---- Model -----------------------------------------------
MODEL_ID="black-forest-labs/FLUX.2-klein-base-9B"

# ---- Hyperparameters -------------------------------------
RANK=32
ALPHA=16
LEARNING_RATE=1e-4
MAX_STEPS=3000
BATCH_SIZE=1
GRAD_ACCUM=4
RESOLUTION=1024
SAVE_EVERY=500     # Save checkpoint every N steps
SAMPLE_EVERY=250   # Generate sample images every N steps

# ---- Training --------------------------------------------
echo "Starting FLUX.2 klein LoRA training..."
echo "  Output: $OUTPUT_DIR"
echo "  Steps:  $MAX_STEPS | Rank: $RANK | LR: $LEARNING_RATE"
echo ""

mkdir -p "$OUTPUT_DIR"

accelerate launch \
  --mixed_precision="bf16" \
  --num_processes=1 \
  "$(python -c "import diffusers; import os; print(os.path.join(os.path.dirname(diffusers.__file__), 'examples/dreambooth/train_dreambooth_lora_flux2.py'))")" \
  --pretrained_model_name_or_path="$MODEL_ID" \
  --instance_data_dir="$DATASET_DIR" \
  --caption_column="$CAPTIONS_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --do_fp8_training \
  --gradient_checkpointing \
  --remote_text_encoder \
  --cache_latents \
  --resolution="$RESOLUTION" \
  --train_batch_size="$BATCH_SIZE" \
  --gradient_accumulation_steps="$GRAD_ACCUM" \
  --learning_rate="$LEARNING_RATE" \
  --lr_scheduler="constant" \
  --max_train_steps="$MAX_STEPS" \
  --rank="$RANK" \
  --lora_alpha="$ALPHA" \
  --checkpointing_steps="$SAVE_EVERY" \
  --validation_prompt="cmcstyle, a hero standing on a rooftop at night" \
  --num_validation_images=2 \
  --validation_steps="$SAMPLE_EVERY" \
  --report_to="tensorboard" \
  --seed=42

echo ""
echo "Training complete. LoRA saved to: $OUTPUT_DIR"
echo "Copy best checkpoint to loras/ when satisfied:"
echo "  cp $OUTPUT_DIR/pytorch_lora_weights.safetensors $PROJECT_ROOT/loras/cmcstyle-v1.safetensors"
