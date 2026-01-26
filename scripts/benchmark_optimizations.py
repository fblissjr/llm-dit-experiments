#!/usr/bin/env python3
"""
Benchmark script for testing Z-Image and FLUX.2 Klein optimizations.

Tests different configurations for:
- Memory: CPU offload, encoder device placement, quantization
- Compute: attention backends (flash_attn_2, sage, sdpa), torch.compile
- Performance metrics: VRAM usage, generation time, image quality hash

Usage:
    # Run all Z-Image benchmarks
    uv run python scripts/benchmark_optimizations.py --pipeline zimage

    # Run specific test
    uv run python scripts/benchmark_optimizations.py --pipeline zimage --test attention

    # Run FLUX.2 benchmarks
    uv run python scripts/benchmark_optimizations.py --pipeline flux2

    # List available tests
    uv run python scripts/benchmark_optimizations.py --list

last updated: 2026-01-26
"""

import argparse
import gc
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any  # noqa: F401

import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    config_name: str
    pipeline: str
    vram_before_mb: float
    vram_after_load_mb: float
    vram_peak_mb: float
    load_time_s: float
    generation_time_s: float
    image_hash: str  # For consistency checking
    success: bool
    error: str | None = None
    config: dict = field(default_factory=dict)


def get_vram_mb() -> float:
    """Get current VRAM usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def get_vram_peak_mb() -> float:
    """Get peak VRAM usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def reset_vram_stats():
    """Reset VRAM peak tracking."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def clear_vram():
    """Force clear VRAM."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def image_hash(img) -> str:
    """Generate a hash of PIL Image for consistency checking."""
    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return hashlib.md5(buf.getvalue()).hexdigest()[:12]


# =============================================================================
# Z-Image Benchmark Configurations
# =============================================================================

ZIMAGE_CONFIGS = {
    # Baseline: current production config
    "baseline": {
        "description": "Production config (encoder on CPU, model CPU offload)",
        "encoder_device": "cpu",
        "dit_device": "cuda",
        "vae_device": "cuda",
        "quantization": "none",
        "attention_backend": "auto",
        "tiled_vae": False,
        "compile": False,
    },
    # Attention backend variations
    "attn_flash2": {
        "description": "Flash Attention 2 (explicit)",
        "encoder_device": "cpu",
        "dit_device": "cuda",
        "vae_device": "cuda",
        "quantization": "none",
        "attention_backend": "flash_attn_2",
        "tiled_vae": False,
        "compile": False,
    },
    "attn_sage": {
        "description": "SageAttention INT8/FP16",
        "encoder_device": "cpu",
        "dit_device": "cuda",
        "vae_device": "cuda",
        "quantization": "none",
        "attention_backend": "sage",
        "tiled_vae": False,
        "compile": False,
    },
    "attn_sdpa": {
        "description": "PyTorch SDPA (reference)",
        "encoder_device": "cpu",
        "dit_device": "cuda",
        "vae_device": "cuda",
        "quantization": "none",
        "attention_backend": "sdpa",
        "tiled_vae": False,
        "compile": False,
    },
    # Encoder device placement
    "encoder_cuda": {
        "description": "Encoder on CUDA (no CPU offload)",
        "encoder_device": "cuda",
        "dit_device": "cuda",
        "vae_device": "cuda",
        "quantization": "none",
        "attention_backend": "auto",
        "tiled_vae": False,
        "compile": False,
    },
    # Quantization variations
    "quant_fp8": {
        "description": "Encoder FP8 quantization",
        "encoder_device": "cpu",
        "dit_device": "cuda",
        "vae_device": "cuda",
        "quantization": "fp8",
        "attention_backend": "auto",
        "tiled_vae": False,
        "compile": False,
    },
    "quant_int8": {
        "description": "Encoder INT8 quantization",
        "encoder_device": "cpu",
        "dit_device": "cuda",
        "vae_device": "cuda",
        "quantization": "int8",
        "attention_backend": "auto",
        "tiled_vae": False,
        "compile": False,
    },
    "quant_4bit": {
        "description": "Encoder 4-bit quantization",
        "encoder_device": "cpu",
        "dit_device": "cuda",
        "vae_device": "cuda",
        "quantization": "4bit",
        "attention_backend": "auto",
        "tiled_vae": False,
        "compile": False,
    },
    # torch.compile
    "compile_default": {
        "description": "torch.compile (default mode)",
        "encoder_device": "cpu",
        "dit_device": "cuda",
        "vae_device": "cuda",
        "quantization": "none",
        "attention_backend": "auto",
        "tiled_vae": False,
        "compile": True,
        "compile_mode": "default",
    },
    # Best combo attempt
    "optimized": {
        "description": "Flash Attn 2 + encoder on CPU",
        "encoder_device": "cpu",
        "dit_device": "cuda",
        "vae_device": "cuda",
        "quantization": "none",
        "attention_backend": "flash_attn_2",
        "tiled_vae": False,
        "compile": False,
    },
    # Encoder on CUDA with quantization - to measure VRAM savings
    "encoder_cuda_fp8": {
        "description": "Encoder on CUDA with FP8",
        "encoder_device": "cuda",
        "dit_device": "cuda",
        "vae_device": "cuda",
        "quantization": "fp8",
        "attention_backend": "auto",
        "tiled_vae": False,
        "compile": False,
    },
    "encoder_cuda_4bit": {
        "description": "Encoder on CUDA with 4-bit",
        "encoder_device": "cuda",
        "dit_device": "cuda",
        "vae_device": "cuda",
        "quantization": "4bit",
        "attention_backend": "auto",
        "tiled_vae": False,
        "compile": False,
    },
}

# Test groups for selective running
ZIMAGE_TEST_GROUPS = {
    "attention": ["baseline", "attn_flash2", "attn_sage", "attn_sdpa"],
    "quantization": ["baseline", "quant_fp8", "quant_int8", "quant_4bit"],
    "device": ["baseline", "encoder_cuda"],
    "device_quant": ["encoder_cuda", "encoder_cuda_fp8", "encoder_cuda_4bit"],
    "compile": ["baseline", "compile_default"],
    "all": list(ZIMAGE_CONFIGS.keys()),
}


# =============================================================================
# FLUX.2 Benchmark Configurations
# =============================================================================

FLUX2_CONFIGS = {
    "baseline": {
        "description": "Production config (block offload, FP8)",
        "model_variant": "klein-9b-fp8",
        "block_offload": True,
        "attention_backend": "auto",
    },
    "no_offload": {
        "description": "No block offload (may OOM)",
        "model_variant": "klein-9b-fp8",
        "block_offload": False,
        "attention_backend": "auto",
    },
    "attn_flash2": {
        "description": "Flash Attention 2",
        "model_variant": "klein-9b-fp8",
        "block_offload": True,
        "attention_backend": "flash_attn_2",
    },
    "attn_sage": {
        "description": "SageAttention",
        "model_variant": "klein-9b-fp8",
        "block_offload": True,
        "attention_backend": "sage",
    },
}

FLUX2_TEST_GROUPS = {
    "attention": ["baseline", "attn_flash2", "attn_sage"],
    "offload": ["baseline", "no_offload"],
    "all": list(FLUX2_CONFIGS.keys()),
}


# =============================================================================
# Benchmark Runner
# =============================================================================


def run_zimage_benchmark(
    config_name: str,
    config: dict,
    prompt: str = "A cat sleeping in warm sunlight, detailed fur, soft shadows",
    steps: int = 9,
    width: int = 1024,
    height: int = 1024,
    seed: int = 42,
    warmup_runs: int = 1,
) -> BenchmarkResult:
    """Run a single Z-Image benchmark with the given configuration."""
    from llm_dit.pipelines import ZImagePipeline
    from llm_dit.pipelines.z_image import setup_attention_backend

    # Clear any existing state
    clear_vram()
    reset_vram_stats()
    vram_before = get_vram_mb()

    result = BenchmarkResult(
        config_name=config_name,
        pipeline="zimage",
        vram_before_mb=vram_before,
        vram_after_load_mb=0,
        vram_peak_mb=0,
        load_time_s=0,
        generation_time_s=0,
        image_hash="",
        success=False,
        config=config,
    )

    try:
        # Setup attention backend
        if config.get("attention_backend"):
            setup_attention_backend(config["attention_backend"])

        # Load pipeline
        print(f"  Loading pipeline...")
        load_start = time.time()

        model_path = "/home/fbliss/Storage/ZImage-Turbo"
        pipeline = ZImagePipeline.from_pretrained(
            model_path,
            encoder_device=config.get("encoder_device", "cpu"),
            dit_device=config.get("dit_device", "cuda"),
            vae_device=config.get("vae_device", "cuda"),
            quantization=config.get("quantization", "none"),
            attention_backend=config.get("attention_backend"),
            tiled_vae=config.get("tiled_vae", False),
        )

        load_time = time.time() - load_start
        result.load_time_s = load_time
        result.vram_after_load_mb = get_vram_mb()
        print(f"  Loaded in {load_time:.1f}s, VRAM: {result.vram_after_load_mb:.0f}MB")

        # Apply torch.compile if requested
        if config.get("compile"):
            print(f"  Applying torch.compile...")
            compile_mode = config.get("compile_mode", "default")
            pipeline.transformer = torch.compile(
                pipeline.transformer, mode=compile_mode
            )

        # Create generator for reproducibility
        # Note: diffusers with CPU offload requires generator on CPU
        def make_generator():
            return torch.Generator(device="cpu").manual_seed(seed)

        # Warmup runs (for compile, caching, etc.)
        if warmup_runs > 0:
            print(f"  Warmup ({warmup_runs} runs)...")
            for _ in range(warmup_runs):
                _ = pipeline(
                    prompt,
                    height=height,
                    width=width,
                    num_inference_steps=steps,
                    generator=make_generator(),
                )
                clear_vram()

        # Timed generation
        reset_vram_stats()
        print(f"  Generating...")
        gen_start = time.time()

        image = pipeline(
            prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            generator=make_generator(),
        )

        gen_time = time.time() - gen_start
        result.generation_time_s = gen_time
        result.vram_peak_mb = get_vram_peak_mb()
        result.image_hash = image_hash(image)
        result.success = True

        print(
            f"  Generated in {gen_time:.2f}s, peak VRAM: {result.vram_peak_mb:.0f}MB"
        )

        # Cleanup
        del pipeline
        clear_vram()

    except Exception as e:
        result.error = str(e)
        print(f"  ERROR: {e}")

    return result


def run_flux2_benchmark(
    config_name: str,
    config: dict,
    prompt: str = "A cat sleeping in warm sunlight, detailed fur, soft shadows",
    steps: int = 4,
    width: int = 1024,
    height: int = 1024,
    seed: int = 42,
    warmup_runs: int = 0,  # FLUX.2 is slow, skip warmup by default
) -> BenchmarkResult:
    """Run a single FLUX.2 benchmark with the given configuration."""
    # FLUX.2 benchmark implementation would go here
    # For now, return a placeholder
    return BenchmarkResult(
        config_name=config_name,
        pipeline="flux2",
        vram_before_mb=0,
        vram_after_load_mb=0,
        vram_peak_mb=0,
        load_time_s=0,
        generation_time_s=0,
        image_hash="",
        success=False,
        error="FLUX.2 benchmark not yet implemented",
        config=config,
    )


def print_results_table(results: list[BenchmarkResult]):
    """Print results in a formatted table."""
    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS")
    print("=" * 100)
    print(
        f"{'Config':<20} {'Status':<8} {'Load(s)':<8} {'Gen(s)':<8} "
        f"{'VRAM Load':<12} {'VRAM Peak':<12} {'Hash':<14}"
    )
    print("-" * 100)

    for r in results:
        status = "✓" if r.success else "✗"
        load_t = f"{r.load_time_s:.1f}" if r.success else "-"
        gen_t = f"{r.generation_time_s:.2f}" if r.success else "-"
        vram_load = f"{r.vram_after_load_mb:.0f}MB" if r.success else "-"
        vram_peak = f"{r.vram_peak_mb:.0f}MB" if r.success else "-"
        hash_val = r.image_hash if r.success else r.error[:12] if r.error else "-"

        print(
            f"{r.config_name:<20} {status:<8} {load_t:<8} {gen_t:<8} "
            f"{vram_load:<12} {vram_peak:<12} {hash_val:<14}"
        )

    print("=" * 100)

    # Check consistency (same seed should produce same hash)
    hashes = [r.image_hash for r in results if r.success and r.image_hash]
    if hashes:
        unique_hashes = set(hashes)
        if len(unique_hashes) == 1:
            print(f"✓ All successful runs produced identical outputs (hash: {hashes[0]})")
        else:
            print(f"⚠ WARNING: Different outputs detected! Hashes: {unique_hashes}")


def save_results(results: list[BenchmarkResult], output_path: Path):
    """Save results to JSON file."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "results": [
            {
                "config_name": r.config_name,
                "pipeline": r.pipeline,
                "success": r.success,
                "load_time_s": r.load_time_s,
                "generation_time_s": r.generation_time_s,
                "vram_before_mb": r.vram_before_mb,
                "vram_after_load_mb": r.vram_after_load_mb,
                "vram_peak_mb": r.vram_peak_mb,
                "image_hash": r.image_hash,
                "error": r.error,
                "config": r.config,
            }
            for r in results
        ],
    }
    output_path.write_text(json.dumps(data, indent=2))
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark optimization configurations")
    parser.add_argument(
        "--pipeline",
        choices=["zimage", "flux2"],
        default="zimage",
        help="Pipeline to benchmark",
    )
    parser.add_argument(
        "--test",
        default="all",
        help="Test group to run (attention, quantization, device, compile, all)",
    )
    parser.add_argument(
        "--config",
        help="Run a specific config by name",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available tests and configs",
    )
    parser.add_argument(
        "--prompt",
        default="A cat sleeping in warm sunlight, detailed fur, soft shadows",
        help="Prompt to use for generation",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=9,
        help="Number of inference steps",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Image width",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Image height",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup runs before timing",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file for results",
    )
    args = parser.parse_args()

    # List mode
    if args.list:
        print("\nZ-Image Configurations:")
        print("-" * 60)
        for name, cfg in ZIMAGE_CONFIGS.items():
            print(f"  {name:<20} - {cfg.get('description', '')}")
        print("\nZ-Image Test Groups:")
        for group, configs in ZIMAGE_TEST_GROUPS.items():
            print(f"  {group}: {', '.join(configs)}")

        print("\n\nFLUX.2 Configurations:")
        print("-" * 60)
        for name, cfg in FLUX2_CONFIGS.items():
            print(f"  {name:<20} - {cfg.get('description', '')}")
        print("\nFLUX.2 Test Groups:")
        for group, configs in FLUX2_TEST_GROUPS.items():
            print(f"  {group}: {', '.join(configs)}")
        return

    # Determine which configs to run
    if args.pipeline == "zimage":
        all_configs = ZIMAGE_CONFIGS
        test_groups = ZIMAGE_TEST_GROUPS
        run_fn = run_zimage_benchmark
    else:
        all_configs = FLUX2_CONFIGS
        test_groups = FLUX2_TEST_GROUPS
        run_fn = run_flux2_benchmark

    if args.config:
        # Single config
        if args.config not in all_configs:
            print(f"Unknown config: {args.config}")
            print(f"Available: {', '.join(all_configs.keys())}")
            return
        configs_to_run = {args.config: all_configs[args.config]}
    elif args.test in test_groups:
        # Test group
        configs_to_run = {
            name: all_configs[name] for name in test_groups[args.test]
        }
    else:
        print(f"Unknown test group: {args.test}")
        print(f"Available: {', '.join(test_groups.keys())}")
        return

    # Run benchmarks
    print(f"\n{'=' * 60}")
    print(f"Running {len(configs_to_run)} {args.pipeline.upper()} benchmarks")
    print(f"Prompt: {args.prompt[:50]}...")
    print(f"Resolution: {args.width}x{args.height}, Steps: {args.steps}, Seed: {args.seed}")
    print(f"{'=' * 60}\n")

    results = []
    for name, config in configs_to_run.items():
        print(f"\n[{name}] {config.get('description', '')}")
        result = run_fn(
            config_name=name,
            config=config,
            prompt=args.prompt,
            steps=args.steps,
            width=args.width,
            height=args.height,
            seed=args.seed,
            warmup_runs=args.warmup,
        )
        results.append(result)

    # Print results
    print_results_table(results)

    # Save results
    if args.output:
        save_results(results, args.output)
    else:
        # Default output path: outputs/tests/benchmarks/
        output_dir = PROJECT_ROOT / "outputs" / "tests" / "benchmarks"
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"benchmark_{args.pipeline}_{timestamp}.json"
        save_results(results, output_path)


if __name__ == "__main__":
    main()
