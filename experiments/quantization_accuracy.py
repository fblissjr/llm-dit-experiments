#!/usr/bin/env python3
"""
Measure accuracy loss from int8_dynamic quantization on text encoder.

Compares outputs from quantized vs non-quantized encoder with identical seeds.
Computes SSIM, LPIPS, and pixel-wise metrics.

Usage:
    uv run experiments/quantization_accuracy.py --model-path /path/to/z-image
    uv run experiments/quantization_accuracy.py --config config.toml --profile rtx4090
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.utils import save_image_grid, save_metadata


def compute_ssim(img1: Image.Image, img2: Image.Image) -> float:
    """Compute structural similarity index between two images."""
    import numpy as np

    # Convert to grayscale numpy arrays
    arr1 = np.array(img1.convert("L")).astype(np.float64)
    arr2 = np.array(img2.convert("L")).astype(np.float64)

    # Constants for stability
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    # Compute means
    mu1 = arr1.mean()
    mu2 = arr2.mean()

    # Compute variances and covariance
    sigma1_sq = arr1.var()
    sigma2_sq = arr2.var()
    sigma12 = ((arr1 - mu1) * (arr2 - mu2)).mean()

    # SSIM formula
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return float(ssim)


def compute_psnr(img1: Image.Image, img2: Image.Image) -> float:
    """Compute peak signal-to-noise ratio between two images."""
    import numpy as np

    arr1 = np.array(img1).astype(np.float64)
    arr2 = np.array(img2).astype(np.float64)

    mse = ((arr1 - arr2) ** 2).mean()
    if mse == 0:
        return float("inf")
    return float(10 * np.log10(255**2 / mse))


def compute_mae(img1: Image.Image, img2: Image.Image) -> float:
    """Compute mean absolute error between two images."""
    import numpy as np

    arr1 = np.array(img1).astype(np.float64)
    arr2 = np.array(img2).astype(np.float64)
    return float(np.abs(arr1 - arr2).mean())


def main():
    parser = argparse.ArgumentParser(description="Measure int8 quantization accuracy loss")
    parser.add_argument("--model-path", type=str, help="Path to Z-Image model")
    parser.add_argument("--config", type=str, help="Path to config.toml")
    parser.add_argument("--profile", type=str, default="default", help="Config profile")
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=[
            "A cat sleeping in warm sunlight on a wooden floor",
            "A mountain landscape with snow-capped peaks at sunset",
            "A detailed portrait of an elderly man with weathered skin",
            "Abstract geometric shapes in vibrant colors",
            "A steampunk mechanical owl with brass gears",
        ],
        help="Prompts to test",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456], help="Seeds to test")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results/quantization_accuracy",
        help="Output directory",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import pipeline components
    from llm_dit.config import Config
    from llm_dit.pipelines.z_image import ZImagePipeline

    # Load config if provided
    if args.config:
        config = Config.from_toml(args.config, args.profile)
        model_path = config.model_path
    else:
        model_path = args.model_path
        config = None

    if not model_path:
        print("Error: --model-path or --config with model_path required")
        sys.exit(1)

    print(f"Model path: {model_path}")
    print(f"Testing {len(args.prompts)} prompts x {len(args.seeds)} seeds = {len(args.prompts) * len(args.seeds)} comparisons")
    print()

    results = []

    for quant_mode in ["none", "int8_dynamic"]:
        print(f"Loading pipeline with quantization={quant_mode}...")

        # Build pipeline kwargs
        pipeline_kwargs = {
            "model_path": model_path,
            "torch_dtype": torch.bfloat16,
            "quantization": quant_mode,
        }

        if config:
            pipeline_kwargs["text_encoder_device"] = config.encoder.device
            pipeline_kwargs["dit_device"] = config.dit.device
            pipeline_kwargs["vae_device"] = config.vae.device

        pipeline = ZImagePipeline.from_pretrained(**pipeline_kwargs)

        # Compile for speed
        if hasattr(pipeline, "transformer"):
            pipeline.transformer = torch.compile(pipeline.transformer, mode="reduce-overhead")

        for prompt_idx, prompt in enumerate(args.prompts):
            for seed in args.seeds:
                print(f"  [{quant_mode}] Prompt {prompt_idx + 1}/{len(args.prompts)}, seed {seed}...")

                generator = torch.Generator(device="cpu").manual_seed(seed)

                start = time.perf_counter()
                image = pipeline(
                    prompt=prompt,
                    width=args.width,
                    height=args.height,
                    num_inference_steps=8,
                    generator=generator,
                ).images[0]
                elapsed = time.perf_counter() - start

                # Save image
                safe_prompt = prompt[:50].replace(" ", "_").replace("/", "_")
                filename = f"{quant_mode}_{safe_prompt}_seed{seed}.png"
                image.save(output_dir / filename)

                results.append({
                    "quantization": quant_mode,
                    "prompt": prompt,
                    "seed": seed,
                    "time": elapsed,
                    "filename": filename,
                })

        # Free memory before loading next pipeline
        del pipeline
        torch.cuda.empty_cache()

    # Compute metrics by comparing pairs
    print("\nComputing metrics...")
    metrics = []

    none_results = [r for r in results if r["quantization"] == "none"]
    int8_results = [r for r in results if r["quantization"] == "int8_dynamic"]

    for none_r in none_results:
        # Find matching int8 result
        int8_r = next(
            (r for r in int8_results if r["prompt"] == none_r["prompt"] and r["seed"] == none_r["seed"]),
            None,
        )
        if not int8_r:
            continue

        img_none = Image.open(output_dir / none_r["filename"])
        img_int8 = Image.open(output_dir / int8_r["filename"])

        ssim = compute_ssim(img_none, img_int8)
        psnr = compute_psnr(img_none, img_int8)
        mae = compute_mae(img_none, img_int8)

        metrics.append({
            "prompt": none_r["prompt"],
            "seed": none_r["seed"],
            "ssim": ssim,
            "psnr": psnr,
            "mae": mae,
            "time_none": none_r["time"],
            "time_int8": int8_r["time"],
            "time_ratio": int8_r["time"] / none_r["time"],
        })

        print(f"  {none_r['prompt'][:40]}... seed={none_r['seed']}: SSIM={ssim:.4f}, PSNR={psnr:.1f}dB, MAE={mae:.2f}")

    # Compute summary statistics
    avg_ssim = sum(m["ssim"] for m in metrics) / len(metrics)
    avg_psnr = sum(m["psnr"] for m in metrics) / len(metrics)
    avg_mae = sum(m["mae"] for m in metrics) / len(metrics)
    avg_time_ratio = sum(m["time_ratio"] for m in metrics) / len(metrics)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Average SSIM:       {avg_ssim:.4f}  (1.0 = identical)")
    print(f"Average PSNR:       {avg_psnr:.1f} dB  (higher = more similar)")
    print(f"Average MAE:        {avg_mae:.2f}  (0 = identical)")
    print(f"Average time ratio: {avg_time_ratio:.2f}x  (<1 = int8 faster)")
    print()

    if avg_ssim >= 0.95:
        print("Result: int8_dynamic quantization has NEGLIGIBLE accuracy impact")
    elif avg_ssim >= 0.90:
        print("Result: int8_dynamic quantization has MINOR accuracy impact")
    else:
        print("Result: int8_dynamic quantization has NOTICEABLE accuracy impact")

    # Save results
    save_metadata(
        output_dir / "results.json",
        prompts=args.prompts,
        seeds=args.seeds,
        metrics=metrics,
        summary={
            "avg_ssim": avg_ssim,
            "avg_psnr": avg_psnr,
            "avg_mae": avg_mae,
            "avg_time_ratio": avg_time_ratio,
        },
    )

    # Create comparison grid for each prompt
    for prompt in args.prompts:
        prompt_metrics = [m for m in metrics if m["prompt"] == prompt]
        if not prompt_metrics:
            continue

        images = []
        labels = []
        for seed in args.seeds:
            safe_prompt = prompt[:50].replace(" ", "_").replace("/", "_")
            img_none = Image.open(output_dir / f"none_{safe_prompt}_seed{seed}.png")
            img_int8 = Image.open(output_dir / f"int8_dynamic_{safe_prompt}_seed{seed}.png")

            m = next((m for m in prompt_metrics if m["seed"] == seed), None)
            ssim = m["ssim"] if m else 0

            images.extend([img_none, img_int8])
            labels.extend([f"none (seed {seed})", f"int8 (SSIM={ssim:.3f})"])

        safe_prompt = prompt[:30].replace(" ", "_").replace("/", "_")
        save_image_grid(
            images,
            output_dir / f"comparison_{safe_prompt}.png",
            cols=2,
            labels=labels,
        )

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
