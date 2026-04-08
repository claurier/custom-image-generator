"""
Inference script for FLUX.2 [klein] with a trained comic style LoRA.

Loads the base model + LoRA weights and generates images from text prompts.
Supports both the base model (quality, ~20-50 steps) and distilled variants
(speed, 4 steps, <1 second per image).

Usage:
    python inference/generate.py \
        --lora loras/cmcstyle-v1.safetensors \
        --prompt "cmcstyle, a hero standing on a rooftop at night" \
        --steps 20 \
        --output output.png

    # Batch from test_prompts.txt
    python inference/generate.py \
        --lora loras/cmcstyle-v1.safetensors \
        --prompts inference/test_prompts.txt \
        --steps 20 \
        --output_dir inference/samples/
"""

import argparse
import time
from pathlib import Path

import torch
from diffusers import FluxPipeline


def load_pipeline(model_id: str, lora_path: str | None, device: str = "cuda") -> FluxPipeline:
    print(f"Loading base model: {model_id}")
    pipe = FluxPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    )

    if lora_path:
        print(f"Loading LoRA: {lora_path}")
        pipe.load_lora_weights(lora_path)

    pipe.enable_model_cpu_offload()  # Handles 9B on 24GB VRAM
    pipe.enable_vae_slicing()
    return pipe


def generate(
    pipe: FluxPipeline,
    prompt: str,
    steps: int = 20,
    guidance_scale: float = 4.0,
    width: int = 1024,
    height: int = 1024,
    seed: int | None = None,
) -> object:
    generator = torch.Generator("cuda").manual_seed(seed) if seed is not None else None
    image = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        generator=generator,
    ).images[0]
    return image


def main():
    parser = argparse.ArgumentParser(description="FLUX.2 klein inference with LoRA.")
    parser.add_argument("--model", default="black-forest-labs/FLUX.2-klein-base-9B")
    parser.add_argument("--lora", default=None, help="Path to .safetensors LoRA file")
    parser.add_argument("--prompt", default=None, help="Single prompt")
    parser.add_argument("--prompts", default=None, help="Path to text file with one prompt per line")
    parser.add_argument("--steps", type=int, default=20, help="Number of inference steps (default: 20)")
    parser.add_argument("--guidance", type=float, default=4.0, help="CFG guidance scale (default: 4.0)")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="output.png", help="Output path for single prompt")
    parser.add_argument("--output_dir", default="inference/samples/", help="Output dir for batch")
    args = parser.parse_args()

    if args.prompt is None and args.prompts is None:
        parser.error("Provide --prompt or --prompts")

    prompts = []
    if args.prompt:
        prompts = [args.prompt]
    elif args.prompts:
        prompts = Path(args.prompts).read_text(encoding="utf-8").strip().splitlines()
        prompts = [p.strip() for p in prompts if p.strip() and not p.startswith("#")]

    pipe = load_pipeline(args.model, args.lora)

    if len(prompts) == 1:
        t0 = time.perf_counter()
        img = generate(pipe, prompts[0], args.steps, args.guidance, args.width, args.height, args.seed)
        elapsed = time.perf_counter() - t0
        img.save(args.output)
        print(f"Saved: {args.output}  ({elapsed:.2f}s)")
    else:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, prompt in enumerate(prompts):
            t0 = time.perf_counter()
            img = generate(pipe, prompt, args.steps, args.guidance, args.width, args.height, args.seed + i)
            elapsed = time.perf_counter() - t0
            out_file = out_dir / f"sample_{i:03d}.png"
            img.save(out_file)
            print(f"[{i+1}/{len(prompts)}] {out_file}  ({elapsed:.2f}s)")


if __name__ == "__main__":
    main()
