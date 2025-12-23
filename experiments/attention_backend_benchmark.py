#!/usr/bin/env python3
"""
Benchmark attention backends: FA2 vs SageAttention vs SDPA.

Compares speed AND quality to determine optimal backend for your GPU.
Quality is measured by comparing outputs against SDPA (exact reference).

Usage:
    uv run experiments/attention_backend_benchmark.py --model-path /path/to/z-image
    uv run experiments/attention_backend_benchmark.py --config config.toml --profile rtx4090
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.utils import save_image_grid, save_metadata


def compute_metrics(img1: Image.Image, img2: Image.Image) -> dict:
    """Compute similarity metrics between two images."""
    import numpy as np

    arr1 = np.array(img1).astype(np.float64)
    arr2 = np.array(img2).astype(np.float64)

    # MSE
    mse = ((arr1 - arr2) ** 2).mean()

    # PSNR
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float("inf")

    # SSIM (grayscale)
    gray1 = np.array(img1.convert("L")).astype(np.float64)
    gray2 = np.array(img2.convert("L")).astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu1, mu2 = gray1.mean(), gray2.mean()
    sigma1_sq, sigma2_sq = gray1.var(), gray2.var()
    sigma12 = ((gray1 - mu1) * (gray2 - mu2)).mean()
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    # MAE
    mae = np.abs(arr1 - arr2).mean()

    return {"mse": float(mse), "psnr": float(psnr), "ssim": float(ssim), "mae": float(mae)}


# Backends that are known to be incompatible with torch.compile
# SageAttention uses custom CUDA kernels that don't work with inductor
COMPILE_INCOMPATIBLE_BACKENDS = {"sage", "xformers"}

# Map our backend names to diffusers DIFFUSERS_ATTN_BACKEND values
DIFFUSERS_BACKEND_MAP = {
    "sage": "sage",
    "sdpa": "native",
    "flash_attn_2": "flash",
    "flash_attn_3": "flash",  # FA3 also uses "flash" in diffusers
    "xformers": "xformers",
}


def benchmark_backend(
    backend: str,
    model_path: str,
    prompts: list[str],
    seeds: list[int],
    width: int,
    height: int,
    steps: int,
    warmup_runs: int = 2,
    compile_mode: str = "default",
) -> tuple[list[Image.Image], list[float]]:
    """Run benchmark for a specific attention backend."""
    from diffusers.models.attention_dispatch import attention_backend as diffusers_attention_backend
    from llm_dit.pipelines.z_image import ZImagePipeline
    from llm_dit.utils.attention import set_attention_backend, reset_attention_backend

    # Set our backend for components that use attention_forward()
    reset_attention_backend()
    set_attention_backend(backend)

    # Map to diffusers backend name
    diffusers_backend = DIFFUSERS_BACKEND_MAP.get(backend, "native")

    print(f"\n{'=' * 60}")
    print(f"Benchmarking: {backend.upper()}")
    print(f"{'=' * 60}")
    print(f"Using diffusers backend: {diffusers_backend}")

    # Use diffusers context manager to set attention backend for transformer
    with diffusers_attention_backend(diffusers_backend):
        # Load pipeline
        print(f"Loading pipeline with {backend}...")
        pipeline = ZImagePipeline.from_pretrained(
            model_path=model_path,
            torch_dtype=torch.bfloat16,
        )

        # Compile for fair comparison (reduce-overhead uses CUDA graphs, incompatible with some backends)
        # Skip compilation for backends that use custom CUDA kernels incompatible with inductor
        actual_compile_mode = compile_mode
        if compile_mode != "none" and backend in COMPILE_INCOMPATIBLE_BACKENDS:
            print(f"Skipping torch.compile for {backend} (incompatible with inductor)")
            actual_compile_mode = "none"

        compiled = False
        if actual_compile_mode != "none":
            print(f"Compiling transformer with mode={actual_compile_mode}...")
            pipeline.transformer = torch.compile(pipeline.transformer, mode=actual_compile_mode)
            compiled = True

        # Warmup
        print(f"Warmup ({warmup_runs} runs)...")
        for i in range(warmup_runs):
            generator = torch.Generator(device="cpu").manual_seed(999)
            _ = pipeline(
                prompt="warmup test",
                width=width,
                height=height,
                num_inference_steps=steps,
                generator=generator,
            )
            torch.cuda.synchronize()

        # Benchmark runs
        images = []
        times = []

        for prompt in prompts:
            for seed in seeds:
                generator = torch.Generator(device="cpu").manual_seed(seed)

                torch.cuda.synchronize()
                start = time.perf_counter()

                image = pipeline(
                    prompt=prompt,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    generator=generator,
                )

                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start

                # Pipeline returns PIL Image directly (not a result object)
                images.append(image)
                times.append(elapsed)

                print(f"  {prompt[:40]}... seed={seed}: {elapsed:.3f}s")

        # Cleanup
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

    return images, times


def main():
    parser = argparse.ArgumentParser(description="Benchmark attention backends")
    parser.add_argument("--model-path", type=str, help="Path to Z-Image model")
    parser.add_argument("--config", type=str, help="Path to config.toml")
    parser.add_argument("--profile", type=str, default="default", help="Config profile")
    parser.add_argument(
        "--backends",
        type=str,
        nargs="+",
        default=None,
        help="Backends to test (default: all available)",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=[
            "A cat sleeping in warm sunlight",
            "A mountain landscape at sunset",
            "Portrait of an elderly man",
        ],
        help="Prompts to test",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123], help="Seeds")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--steps", type=int, default=9, help="Inference steps")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup runs per backend")
    parser.add_argument(
        "--compile",
        choices=["none", "default", "reduce-overhead", "max-autotune"],
        default=None,
        help="torch.compile mode (default: from config, or 'none' if no config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results/attention_benchmark",
        help="Output directory",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get model path and compile setting from config
    compile_mode = args.compile  # CLI override if provided
    if args.config:
        from llm_dit.config import Config
        config = Config.from_toml(args.config, args.profile)
        model_path = config.model_path
        # Use config's compile setting if CLI wasn't explicitly set
        if args.compile is None:
            compile_mode = "default" if config.optimization.compile else "none"
    else:
        model_path = args.model_path
        # No config, default to none if not specified
        if compile_mode is None:
            compile_mode = "none"

    if not model_path:
        print("Error: --model-path or --config required")
        sys.exit(1)

    print(f"Compile mode: {compile_mode}")

    # Detect available backends
    from llm_dit.utils.attention import get_available_backends
    available = get_available_backends()
    print(f"Available backends: {available}")

    # Filter to requested backends
    if args.backends:
        backends = [b for b in args.backends if b in available]
    else:
        # Test all available, but always include sdpa as reference
        backends = [b for b in available if b in ["flash_attn_2", "sage", "sdpa"]]

    if "sdpa" not in backends:
        backends.append("sdpa")  # Always need reference

    print(f"Testing backends: {backends}")
    print(f"Prompts: {len(args.prompts)}, Seeds: {len(args.seeds)}")
    print(f"Total images per backend: {len(args.prompts) * len(args.seeds)}")

    # Run benchmarks
    results = {}
    for backend in backends:
        images, times = benchmark_backend(
            backend=backend,
            model_path=model_path,
            prompts=args.prompts,
            seeds=args.seeds,
            width=args.width,
            height=args.height,
            steps=args.steps,
            warmup_runs=args.warmup,
            compile_mode=compile_mode,
        )
        results[backend] = {"images": images, "times": times}

        # Save images
        for i, img in enumerate(images):
            prompt_idx = i // len(args.seeds)
            seed_idx = i % len(args.seeds)
            seed = args.seeds[seed_idx]
            safe_prompt = args.prompts[prompt_idx][:30].replace(" ", "_")
            img.save(output_dir / f"{backend}_{safe_prompt}_seed{seed}.png")

    # Compare against SDPA reference
    print("\n" + "=" * 60)
    print("QUALITY COMPARISON (vs SDPA reference)")
    print("=" * 60)

    reference_images = results["sdpa"]["images"]
    comparison_results = {}

    for backend in backends:
        if backend == "sdpa":
            continue

        backend_images = results[backend]["images"]
        metrics_list = []

        for i, (ref_img, test_img) in enumerate(zip(reference_images, backend_images)):
            metrics = compute_metrics(ref_img, test_img)
            metrics_list.append(metrics)

        # Average metrics
        avg_metrics = {
            "ssim": sum(m["ssim"] for m in metrics_list) / len(metrics_list),
            "psnr": sum(m["psnr"] for m in metrics_list) / len(metrics_list),
            "mae": sum(m["mae"] for m in metrics_list) / len(metrics_list),
        }
        comparison_results[backend] = avg_metrics

        print(f"\n{backend.upper()} vs SDPA:")
        print(f"  SSIM: {avg_metrics['ssim']:.6f}  (1.0 = identical)")
        print(f"  PSNR: {avg_metrics['psnr']:.2f} dB  (>40 = excellent)")
        print(f"  MAE:  {avg_metrics['mae']:.4f}  (0 = identical)")

    # Speed comparison
    print("\n" + "=" * 60)
    print("SPEED COMPARISON")
    print("=" * 60)

    sdpa_avg = sum(results["sdpa"]["times"]) / len(results["sdpa"]["times"])
    speed_results = {}

    for backend in backends:
        avg_time = sum(results[backend]["times"]) / len(results[backend]["times"])
        speedup = sdpa_avg / avg_time
        speed_results[backend] = {"avg_time": avg_time, "speedup_vs_sdpa": speedup}

        print(f"\n{backend.upper()}:")
        print(f"  Avg time: {avg_time:.3f}s")
        print(f"  Speedup vs SDPA: {speedup:.2f}x")

    # Final recommendation
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)

    best_backend = None
    best_score = -1

    for backend in backends:
        if backend == "sdpa":
            continue

        quality = comparison_results[backend]["ssim"]
        speed = speed_results[backend]["speedup_vs_sdpa"]

        # Score: quality must be > 0.99 SSIM, then rank by speed
        if quality >= 0.99:
            score = speed
            if score > best_score:
                best_score = score
                best_backend = backend

    if best_backend:
        print(f"\nBest backend for RTX 4090: {best_backend.upper()}")
        print(f"  - SSIM vs SDPA: {comparison_results[best_backend]['ssim']:.6f}")
        print(f"  - Speedup: {speed_results[best_backend]['speedup_vs_sdpa']:.2f}x")
    else:
        print("\nNo backend meets quality threshold (SSIM >= 0.99)")
        print("Recommend: sdpa (reference quality)")

    # Save results
    save_metadata(
        output_dir / "benchmark_results.json",
        backends=backends,
        prompts=args.prompts,
        seeds=args.seeds,
        speed_results=speed_results,
        quality_results=comparison_results,
        recommendation=best_backend,
    )

    # Create comparison grid
    all_images = []
    labels = []
    for backend in backends:
        all_images.append(results[backend]["images"][0])  # First image from each
        speedup = speed_results[backend]["speedup_vs_sdpa"]
        if backend == "sdpa":
            labels.append(f"{backend} (ref)")
        else:
            ssim = comparison_results[backend]["ssim"]
            labels.append(f"{backend} ({speedup:.2f}x, SSIM={ssim:.4f})")

    save_image_grid(all_images, output_dir / "comparison_grid.png", cols=len(backends), labels=labels)

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
