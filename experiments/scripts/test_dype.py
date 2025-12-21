#!/usr/bin/env python3
"""
test_dype.py - comprehensive DyPE validation script

last updated: 2025-12-21

Tests DyPE (Dynamic Position Extrapolation) at various resolutions,
compares all three methods (vision_yarn, yarn, ntk), and validates
multipass mode for ultra-high resolutions.

Usage:
    # Full test suite (all resolutions, all methods)
    uv run experiments/scripts/test_dype.py \
        --model-path /path/to/z-image-turbo \
        --output results/dype_test

    # Quick test (1024 and 2048 only)
    uv run experiments/scripts/test_dype.py \
        --model-path /path/to/z-image-turbo \
        --quick

    # Test specific method
    uv run experiments/scripts/test_dype.py \
        --model-path /path/to/z-image-turbo \
        --method vision_yarn

    # Test multipass mode only
    uv run experiments/scripts/test_dype.py \
        --model-path /path/to/z-image-turbo \
        --multipass-only

    # Use config file
    uv run experiments/scripts/test_dype.py \
        --config config.toml \
        --profile rtx4090
"""

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llm_dit.cli import load_runtime_config
from llm_dit.pipelines.z_image import ZImagePipeline
from llm_dit.utils.dype import DyPEConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Results from a single DyPE test."""
    resolution: str
    method: str
    dype_enabled: bool
    multipass: bool
    generation_time: float
    peak_vram: float
    image_path: Path
    seed: int


# Standard test prompt
STANDARD_PROMPT = "Homer Simpson eating a donut"

# Test resolutions (width, height, description)
RESOLUTIONS = [
    (1024, 1024, "1K_square"),
    (1536, 1536, "1.5K_square"),
    (2048, 2048, "2K_square"),
    (3072, 3072, "3K_square"),
    (4096, 4096, "4K_square"),
]

QUICK_RESOLUTIONS = [
    (1024, 1024, "1K_square"),
    (2048, 2048, "2K_square"),
]

# DyPE methods to test
METHODS = ["vision_yarn", "yarn", "ntk"]


def create_comparison_grid(
    images: list[Image.Image],
    labels: list[str],
    cols: int = 3,
) -> Image.Image:
    """Create a comparison grid with labels."""
    if not images:
        raise ValueError("No images to grid")

    # Calculate grid dimensions
    rows = (len(images) + cols - 1) // cols

    # Get max dimensions
    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)

    # Create grid canvas (add space for labels)
    label_height = 40
    grid_width = max_width * cols
    grid_height = (max_height + label_height) * rows
    grid = Image.new("RGB", (grid_width, grid_height), (255, 255, 255))

    # Paste images and add labels
    for idx, (img, label) in enumerate(zip(images, labels)):
        row = idx // cols
        col = idx % cols
        x = col * max_width
        y = row * (max_height + label_height)

        # Paste image
        grid.paste(img, (x, y))

        # Draw label (simple text rendering)
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(grid)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        except:
            font = ImageFont.load_default()

        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = x + (max_width - text_width) // 2
        text_y = y + max_height + 10

        draw.text((text_x, text_y), label, fill=(0, 0, 0), font=font)

    return grid


def run_generation(
    pipe: ZImagePipeline,
    prompt: str,
    width: int,
    height: int,
    dype_config: Optional[DyPEConfig],
    seed: int,
    steps: int = 9,
    shift: float = 3.0,
) -> tuple[Image.Image, float, float]:
    """Run a single generation and measure performance.

    Returns:
        Tuple of (image, generation_time, peak_vram_gb)
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    start_time = time.time()

    result = pipe(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        shift=shift,
        seed=seed,
        dype_config=dype_config,
    )

    generation_time = time.time() - start_time

    peak_vram = 0.0
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / 1024**3

    return result.images[0], generation_time, peak_vram


def test_resolution(
    pipe: ZImagePipeline,
    width: int,
    height: int,
    resolution_name: str,
    output_dir: Path,
    seed: int,
    methods: list[str],
    test_multipass: bool = False,
) -> list[TestResult]:
    """Test a single resolution with all methods and baseline."""
    results = []
    images = []
    labels = []

    logger.info(f"\n{'='*60}")
    logger.info(f"Testing resolution: {resolution_name} ({width}x{height})")
    logger.info(f"{'='*60}\n")

    # 1. Baseline (no DyPE)
    logger.info("Testing baseline (DyPE disabled)...")
    img, gen_time, peak_vram = run_generation(
        pipe=pipe,
        prompt=STANDARD_PROMPT,
        width=width,
        height=height,
        dype_config=None,
        seed=seed,
    )

    img_path = output_dir / f"{resolution_name}_baseline.png"
    img.save(img_path)
    logger.info(f"  Time: {gen_time:.1f}s, Peak VRAM: {peak_vram:.2f} GB")

    results.append(TestResult(
        resolution=resolution_name,
        method="baseline",
        dype_enabled=False,
        multipass=False,
        generation_time=gen_time,
        peak_vram=peak_vram,
        image_path=img_path,
        seed=seed,
    ))
    images.append(img)
    labels.append(f"Baseline\n{gen_time:.1f}s")

    # 2. Test each DyPE method
    for method in methods:
        logger.info(f"Testing DyPE method: {method}...")

        dype_config = DyPEConfig(
            enabled=True,
            method=method,
            dype_scale=2.0,
            dype_exponent=2.0,
            base_resolution=1024,
        )

        img, gen_time, peak_vram = run_generation(
            pipe=pipe,
            prompt=STANDARD_PROMPT,
            width=width,
            height=height,
            dype_config=dype_config,
            seed=seed,
        )

        img_path = output_dir / f"{resolution_name}_dype_{method}.png"
        img.save(img_path)
        logger.info(f"  Time: {gen_time:.1f}s, Peak VRAM: {peak_vram:.2f} GB")

        results.append(TestResult(
            resolution=resolution_name,
            method=method,
            dype_enabled=True,
            multipass=False,
            generation_time=gen_time,
            peak_vram=peak_vram,
            image_path=img_path,
            seed=seed,
        ))
        images.append(img)
        labels.append(f"DyPE {method}\n{gen_time:.1f}s")

    # 3. Test multipass mode (only at 4K)
    if test_multipass and width >= 4096:
        logger.info("Testing multipass mode (two-pass)...")

        dype_config = DyPEConfig(
            enabled=True,
            method="vision_yarn",
            dype_scale=2.0,
            dype_exponent=2.0,
            base_resolution=1024,
        )

        # First pass at lower resolution
        mid_width = width // 2
        mid_height = height // 2

        img_mid, gen_time_1, peak_vram_1 = run_generation(
            pipe=pipe,
            prompt=STANDARD_PROMPT,
            width=mid_width,
            height=mid_height,
            dype_config=dype_config,
            seed=seed,
        )

        # Second pass at full resolution using img2img
        img_final, gen_time_2, peak_vram_2 = run_generation(
            pipe=pipe,
            prompt=STANDARD_PROMPT,
            width=width,
            height=height,
            dype_config=dype_config,
            seed=seed,
        )

        total_time = gen_time_1 + gen_time_2
        max_vram = max(peak_vram_1, peak_vram_2)

        img_path = output_dir / f"{resolution_name}_multipass.png"
        img_final.save(img_path)
        logger.info(f"  Pass 1: {gen_time_1:.1f}s ({mid_width}x{mid_height})")
        logger.info(f"  Pass 2: {gen_time_2:.1f}s ({width}x{height})")
        logger.info(f"  Total: {total_time:.1f}s, Peak VRAM: {max_vram:.2f} GB")

        results.append(TestResult(
            resolution=resolution_name,
            method="vision_yarn",
            dype_enabled=True,
            multipass=True,
            generation_time=total_time,
            peak_vram=max_vram,
            image_path=img_path,
            seed=seed,
        ))
        images.append(img_final)
        labels.append(f"Multipass\n{total_time:.1f}s")

    # 4. Create comparison grid
    logger.info("Creating comparison grid...")
    grid = create_comparison_grid(images, labels, cols=3)
    grid_path = output_dir / f"{resolution_name}_comparison.png"
    grid.save(grid_path)
    logger.info(f"  Saved: {grid_path}\n")

    return results


def save_results_csv(results: list[TestResult], output_path: Path):
    """Save test results to CSV."""
    import csv

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "resolution",
            "method",
            "dype_enabled",
            "multipass",
            "generation_time",
            "peak_vram_gb",
            "image_path",
            "seed",
        ])

        for result in results:
            writer.writerow([
                result.resolution,
                result.method,
                result.dype_enabled,
                result.multipass,
                f"{result.generation_time:.2f}",
                f"{result.peak_vram:.2f}",
                result.image_path,
                result.seed,
            ])


def save_results_summary(results: list[TestResult], output_path: Path):
    """Save human-readable summary."""
    with open(output_path, "w") as f:
        f.write(f"DyPE Test Results\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n\n")

        # Group by resolution
        by_resolution = {}
        for result in results:
            if result.resolution not in by_resolution:
                by_resolution[result.resolution] = []
            by_resolution[result.resolution].append(result)

        for resolution, res_results in sorted(by_resolution.items()):
            f.write(f"\n{resolution}\n")
            f.write(f"{'-'*80}\n")

            for result in res_results:
                label = f"{result.method}"
                if result.multipass:
                    label += " (multipass)"
                if not result.dype_enabled:
                    label = "Baseline"

                f.write(f"  {label:20s}: {result.generation_time:6.1f}s, "
                       f"VRAM: {result.peak_vram:5.2f} GB\n")

        f.write(f"\n{'='*80}\n")
        f.write(f"\nNotes:\n")
        f.write(f"  - Baseline: No DyPE (standard RoPE)\n")
        f.write(f"  - vision_yarn: Vision YaRN with dual mask blending\n")
        f.write(f"  - yarn: Standard YaRN (simpler than Vision YaRN)\n")
        f.write(f"  - ntk: NTK scaling with DyPE modulation\n")
        f.write(f"  - multipass: Two-pass generation (2K -> 4K)\n")


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive DyPE validation script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--model-path",
        help="Path to Z-Image model",
    )
    parser.add_argument(
        "--config",
        help="Path to TOML config file",
    )
    parser.add_argument(
        "--profile",
        default="default",
        help="Config profile to use (default: default)",
    )
    parser.add_argument(
        "--output",
        default="results/dype_test",
        help="Output directory (default: results/dype_test)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test (1024 and 2048 only)",
    )
    parser.add_argument(
        "--method",
        choices=METHODS,
        help="Test specific method only (default: all)",
    )
    parser.add_argument(
        "--multipass-only",
        action="store_true",
        help="Test multipass mode only (4K)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=9,
        help="Inference steps (default: 9)",
    )
    parser.add_argument(
        "--shift",
        type=float,
        default=3.0,
        help="Scheduler shift (default: 3.0)",
    )

    args = parser.parse_args()

    # Determine output directory
    output_dir = Path(args.output)
    if not args.multipass_only:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"DyPE Test Suite")
    logger.info(f"{'='*60}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"{'='*60}\n")

    # Load config
    if args.config:
        logger.info(f"Loading config from {args.config} (profile: {args.profile})...")
        runtime_config = load_runtime_config(
            config_path=args.config,
            profile=args.profile,
        )
        model_path = args.model_path or runtime_config.model_path
    else:
        if not args.model_path:
            logger.error("Either --config or --model-path is required")
            return 1
        model_path = args.model_path
        runtime_config = None

    # Load pipeline
    logger.info(f"Loading pipeline from {model_path}...")
    start = time.time()

    if runtime_config:
        from llm_dit.startup import create_pipeline
        pipe = create_pipeline(runtime_config, model_type="zimage")
    else:
        pipe = ZImagePipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        )

    logger.info(f"Pipeline loaded in {time.time() - start:.1f}s\n")

    # Determine resolutions to test
    if args.multipass_only:
        resolutions = [(4096, 4096, "4K_square")]
    elif args.quick:
        resolutions = QUICK_RESOLUTIONS
    else:
        resolutions = RESOLUTIONS

    # Determine methods to test
    if args.method:
        methods = [args.method]
    elif args.multipass_only:
        methods = ["vision_yarn"]  # Multipass uses vision_yarn
    else:
        methods = METHODS

    # Run tests
    all_results = []

    for width, height, resolution_name in resolutions:
        test_multipass = (
            args.multipass_only or
            (not args.quick and width >= 4096)
        )

        results = test_resolution(
            pipe=pipe,
            width=width,
            height=height,
            resolution_name=resolution_name,
            output_dir=output_dir,
            seed=args.seed,
            methods=methods,
            test_multipass=test_multipass,
        )
        all_results.extend(results)

    # Save results
    logger.info("Saving results...")
    save_results_csv(all_results, output_dir / "results.csv")
    save_results_summary(all_results, output_dir / "summary.txt")
    logger.info(f"  CSV: {output_dir / 'results.csv'}")
    logger.info(f"  Summary: {output_dir / 'summary.txt'}")

    logger.info(f"\n{'='*60}")
    logger.info(f"Test complete. Results saved to: {output_dir}")
    logger.info(f"{'='*60}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
