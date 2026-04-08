"""
Mac inference script for FLUX.2 [klein] — optimised for Apple Silicon with 8 GB unified memory.

Uses mflux (MLX-native, bypasses PyTorch MPS issues) with Q4 quantization so the
full model fits in ~4 GB, leaving headroom for macOS and other processes.

Model: FLUX.2-klein-distilled-4B  (4-step, sub-minute per image on M3 Air)
       The 9B model is NOT usable on 8 GB — do not attempt it.

First run downloads ~16 GB into ~/.cache/huggingface/hub/ automatically.

Install:
    pip install -r requirements_mac.txt

Usage — single prompt:
    python inference/generate_mac.py \\
        --prompt "cmcstyle, a detective walking through a rainy alley at night" \\
        --output inference/samples/test.png

Usage — batch from file:
    python inference/generate_mac.py \\
        --prompts inference/test_prompts.txt \\
        --output_dir inference/samples/

Usage — with a trained LoRA:
    python inference/generate_mac.py \\
        --lora loras/cmcstyle-v1.safetensors \\
        --prompt "cmcstyle, a superhero leaping across rooftops"

Resolution note:
    Default is 512x512 for 8 GB safety.
    You can try --width 768 --height 768 if memory allows, but 1024 will likely OOM.

Fallback — if mflux still OOMs (e.g. background apps consuming RAM):
    iris.c uses memory-mapped weights and peaks at ~4-5 GB.
    See: https://github.com/antirez/iris.c
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


# Default model — 4B distilled is the only safe choice on 8 GB
DEFAULT_MODEL = "black-forest-labs/FLUX.2-klein-distilled-4B"


def build_mflux_command(
    model: str,
    prompt: str,
    output: str,
    steps: int,
    guidance: float,
    width: int,
    height: int,
    quantize: int,
    seed: int,
    lora_path: str | None,
    lora_scale: float,
) -> list[str]:
    cmd = [
        "mflux-generate",
        "--model", model,
        "--prompt", prompt,
        "--steps", str(steps),
        "--width", str(width),
        "--height", str(height),
        "--quantize", str(quantize),
        "--seed", str(seed),
        "--output", output,
    ]

    # Distilled models have guidance baked in — only add flag if user overrides
    if guidance != 1.0:
        cmd += ["--guidance", str(guidance)]

    if lora_path:
        cmd += ["--lora-paths", lora_path, "--lora-scales", str(lora_scale)]

    return cmd


def run_single(args, prompt: str, output: str, seed: int) -> float:
    cmd = build_mflux_command(
        model=args.model,
        prompt=prompt,
        output=output,
        steps=args.steps,
        guidance=args.guidance,
        width=args.width,
        height=args.height,
        quantize=args.quantize,
        seed=seed,
        lora_path=args.lora,
        lora_scale=args.lora_scale,
    )

    print(f"\nPrompt : {prompt}")
    print(f"Output : {output}")
    print(f"Running: {' '.join(cmd)}\n")

    t0 = time.perf_counter()
    result = subprocess.run(cmd, check=False)
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"[ERROR] mflux-generate exited with code {result.returncode}")
        print("If you see an out-of-memory error, try:")
        print("  - Closing other applications to free RAM")
        print("  - Lowering resolution: --width 384 --height 384")
        print("  - Switching to iris.c: https://github.com/antirez/iris.c")
        sys.exit(result.returncode)

    return elapsed


def main():
    parser = argparse.ArgumentParser(
        description="FLUX.2-klein inference for Apple Silicon (8 GB, mflux Q4).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="HuggingFace model ID (default: FLUX.2-klein-distilled-4B)",
    )
    parser.add_argument("--lora", default=None, help="Path to .safetensors LoRA file")
    parser.add_argument("--lora_scale", type=float, default=1.0, help="LoRA weight scale (default: 1.0)")
    parser.add_argument("--prompt", default=None, help="Single text prompt")
    parser.add_argument("--prompts", default=None, help="Path to text file, one prompt per line")
    parser.add_argument(
        "--steps",
        type=int,
        default=4,
        help="Inference steps (default: 4 — matches the distilled model)",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=1.0,
        help="CFG guidance scale (default: 1.0 for distilled — do not raise above 2.0)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Output width in px (default: 512 — safe for 8 GB; try 768 if RAM permits)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Output height in px (default: 512)",
    )
    parser.add_argument(
        "--quantize",
        type=int,
        default=4,
        choices=[4, 8],
        help="Quantization bits (default: 4 — required for 8 GB; use 8 for higher quality if RAM allows)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output", default="inference/samples/output.png", help="Output path for single prompt")
    parser.add_argument("--output_dir", default="inference/samples/", help="Output directory for batch mode")
    args = parser.parse_args()

    if args.prompt is None and args.prompts is None:
        parser.error("Provide --prompt or --prompts")

    # Collect prompts
    prompts = []
    if args.prompt:
        prompts = [args.prompt]
    else:
        lines = Path(args.prompts).read_text(encoding="utf-8").strip().splitlines()
        prompts = [l.strip() for l in lines if l.strip() and not l.startswith("#")]

    print(f"Model     : {args.model}")
    print(f"Quantize  : Q{args.quantize}  ({'~4 GB peak' if args.quantize == 4 else '~8 GB peak'})")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Steps     : {args.steps}")
    print(f"LoRA      : {args.lora or 'none'}")
    print(f"Prompts   : {len(prompts)}")

    if args.width > 768 or args.height > 768:
        print("\n[WARNING] Resolution above 768px may OOM on 8 GB. Lower with --width 512 --height 512 if needed.")

    if len(prompts) == 1:
        out = args.output
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        elapsed = run_single(args, prompts[0], out, args.seed)
        print(f"\nDone in {elapsed:.1f}s — saved to {out}")
    else:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        total_start = time.perf_counter()
        for i, prompt in enumerate(prompts):
            out = str(out_dir / f"sample_{i:03d}.png")
            elapsed = run_single(args, prompt, out, args.seed + i)
            print(f"[{i + 1}/{len(prompts)}] Done in {elapsed:.1f}s — {out}")
        total = time.perf_counter() - total_start
        print(f"\nAll {len(prompts)} images done in {total:.1f}s ({total / len(prompts):.1f}s avg)")


if __name__ == "__main__":
    main()
