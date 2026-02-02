"""
LTX-2 baseline video generation and comparison runner.

Last Updated: 2026-02-02

Generates videos with fixed seeds and compares against known-good outputs.
This module provides the infrastructure for establishing quality baselines
and detecting regressions in video generation.

Usage:
    # Generate a baseline from tier config
    result = generate_baseline(config_tier="smoke", seed=42)

    # Generate a baseline from preset
    result = generate_baseline_from_preset("ltx2_smoke_test")

    # Compare baselines
    comparison = compare_baselines(new_path, reference_path)

Run with:
    uv run pytest tests/e2e/test_ltx2_baselines.py -v -s
"""

import gc
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from tests.backends import (
    REFERENCE_CONFIG,
    SHORT_CONFIG,
    SMOKE_CONFIG,
    GenerationConfig,
    get_backend,
)
from tests.fixtures.prompts.ltx2 import (
    OFFICIAL_PROMPTS,
    REFERENCE_PROMPTS,
    SMOKE_TEST_PROMPT,
)

logger = logging.getLogger(__name__)

# Default output directory for baselines
BASELINE_OUTPUT_DIR = Path("outputs/baselines")


@dataclass
class BaselineResult:
    """Result from baseline video generation.

    Captures all metadata needed for reproducibility and comparison.
    """

    # Generation parameters
    prompt: str
    seed: int
    config_tier: str  # smoke | short | reference

    # Output info
    output_path: Path
    frames_generated: int = 0
    height: int = 0
    width: int = 0

    # Performance metrics
    generation_time_seconds: float = 0.0
    peak_vram_gb: float = 0.0

    # Video statistics (for comparison)
    mean_pixel_value: float = 0.0
    std_pixel_value: float = 0.0
    min_pixel_value: float = 0.0
    max_pixel_value: float = 0.0

    # Reproducibility info
    backend_name: str = ""
    timestamp: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "prompt": self.prompt,
            "seed": self.seed,
            "config_tier": self.config_tier,
            "output_path": str(self.output_path),
            "frames_generated": self.frames_generated,
            "height": self.height,
            "width": self.width,
            "generation_time_seconds": self.generation_time_seconds,
            "peak_vram_gb": self.peak_vram_gb,
            "video_stats": {
                "mean": self.mean_pixel_value,
                "std": self.std_pixel_value,
                "min": self.min_pixel_value,
                "max": self.max_pixel_value,
            },
            "backend_name": self.backend_name,
            "timestamp": self.timestamp,
        }


@dataclass
class ComparisonResult:
    """Result from comparing two baseline videos."""

    # Similarity metrics
    ssim: float = 0.0  # Structural Similarity Index (0-1, higher is more similar)
    psnr: float = 0.0  # Peak Signal-to-Noise Ratio (higher is more similar)
    mae: float = 0.0  # Mean Absolute Error (lower is more similar)
    mse: float = 0.0  # Mean Squared Error (lower is more similar)

    # Per-frame metrics
    per_frame_ssim: List[float] = field(default_factory=list)
    per_frame_psnr: List[float] = field(default_factory=list)

    # Overall assessment
    is_identical: bool = False  # True if bit-identical
    is_similar: bool = False  # True if SSIM > 0.95

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "ssim": self.ssim,
            "psnr": self.psnr,
            "mae": self.mae,
            "mse": self.mse,
            "per_frame_ssim": self.per_frame_ssim,
            "per_frame_psnr": self.per_frame_psnr,
            "is_identical": self.is_identical,
            "is_similar": self.is_similar,
        }


def get_baseline_config(tier: str) -> GenerationConfig:
    """Get generation config for a given tier.

    Args:
        tier: Config tier - "smoke", "short", or "reference"

    Returns:
        GenerationConfig for the requested tier

    Raises:
        ValueError: If tier is not recognized
    """
    configs = {
        "smoke": SMOKE_CONFIG,
        "short": SHORT_CONFIG,
        "reference": REFERENCE_CONFIG,
    }
    if tier not in configs:
        raise ValueError(f"Unknown config tier: {tier}. Use: {list(configs.keys())}")
    return configs[tier]


def get_baseline_prompt(tier: str) -> str:
    """Get appropriate prompt for a given tier.

    Args:
        tier: Config tier - "smoke", "short", or "reference"

    Returns:
        Prompt string appropriate for the tier
    """
    if tier == "smoke":
        return SMOKE_TEST_PROMPT
    elif tier == "short":
        return REFERENCE_PROMPTS.get("cat_walking", SMOKE_TEST_PROMPT)
    else:  # reference
        # Use a more complex prompt for reference baseline
        return OFFICIAL_PROMPTS.get("action_cinematic", REFERENCE_PROMPTS.get("cat_walking", SMOKE_TEST_PROMPT))


def _cleanup_gpu():
    """Free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def generate_baseline(
    config_tier: str = "smoke",
    seed: int = 42,
    prompt: Optional[str] = None,
    output_dir: Optional[Path] = None,
    save_video: bool = True,
) -> BaselineResult:
    """Generate a baseline video with deterministic settings.

    Args:
        config_tier: Configuration tier ("smoke", "short", "reference")
        seed: Random seed for reproducibility
        prompt: Custom prompt (uses tier-appropriate default if None)
        output_dir: Output directory (uses default baselines dir if None)
        save_video: Whether to save the video to disk

    Returns:
        BaselineResult with generation metadata and output path
    """
    import json
    from datetime import datetime

    # Get config and prompt
    config = get_baseline_config(config_tier)
    config.seed = seed

    if prompt is None:
        prompt = get_baseline_prompt(config_tier)

    # Setup output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = BASELINE_OUTPUT_DIR / f"{config_tier}_seed{seed}_{timestamp}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating {config_tier} baseline (seed={seed})")
    logger.info(f"Prompt: {prompt[:80]}...")
    logger.info(f"Config: {config.num_frames} frames, {config.height}x{config.width}")
    logger.info(f"Output: {output_dir}")

    # Clean GPU before generation
    _cleanup_gpu()

    # Track VRAM usage
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Generate video
    backend = get_backend()
    start_time = time.time()

    result = backend.generate_video(
        prompt=prompt,
        config=config,
        output_dir=output_dir,
        save_video=save_video,
    )

    generation_time = time.time() - start_time
    peak_vram = 0.0
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

    # Compute video statistics
    video = result.video
    if video.dtype == torch.uint8:
        video_float = video.float() / 255.0
    else:
        video_float = video.float()

    baseline_result = BaselineResult(
        prompt=prompt,
        seed=seed,
        config_tier=config_tier,
        output_path=output_dir / "video.mp4",
        frames_generated=video.shape[0],
        height=video.shape[1],
        width=video.shape[2],
        generation_time_seconds=generation_time,
        peak_vram_gb=peak_vram,
        mean_pixel_value=float(video_float.mean()),
        std_pixel_value=float(video_float.std()),
        min_pixel_value=float(video_float.min()),
        max_pixel_value=float(video_float.max()),
        backend_name=backend.name,
        timestamp=datetime.now().isoformat(),
    )

    # Save baseline metadata
    with open(output_dir / "baseline_result.json", "w") as f:
        json.dump(baseline_result.to_dict(), f, indent=2)

    logger.info(f"Generation complete in {generation_time:.1f}s")
    logger.info(f"Peak VRAM: {peak_vram:.1f} GB")
    logger.info(f"Video stats: mean={baseline_result.mean_pixel_value:.4f}, std={baseline_result.std_pixel_value:.4f}")

    # Cleanup
    _cleanup_gpu()

    return baseline_result


def generate_baseline_from_preset(
    preset_name: str,
    output_dir: Optional[Path] = None,
    save_video: bool = True,
    **overrides: Any,
) -> BaselineResult:
    """Generate a baseline video using a preset configuration.

    This function loads generation parameters from a preset file and uses
    them to generate a baseline video. Presets provide a structured way
    to define test configurations with all necessary parameters.

    Args:
        preset_name: Name of the preset (e.g., "ltx2_smoke_test")
        output_dir: Output directory (uses default baselines dir if None)
        save_video: Whether to save the video to disk
        **overrides: Optional parameter overrides (seed, prompt, etc.)

    Returns:
        BaselineResult with generation metadata and output path

    Example:
        result = generate_baseline_from_preset("ltx2_smoke_test")
        result = generate_baseline_from_preset("ltx2_smoke_test", seed=123)
    """
    import json
    from datetime import datetime

    from tests.fixtures.configs.presets import get_test_preset

    # Load preset
    preset = get_test_preset(preset_name, **overrides)

    # Extract parameters from preset
    seed = preset.metadata.get("seed", 42)
    prompt = preset.metadata.get("prompt", SMOKE_TEST_PROMPT)
    num_frames = preset.metadata.get("num_frames", 33)
    height = preset.metadata.get("height", 512)
    width = preset.metadata.get("width", 768)

    # Build config from preset
    config = GenerationConfig(
        num_frames=num_frames,
        height=height,
        width=width,
        num_inference_steps=getattr(preset, "steps", 30),
        guidance_scale=getattr(preset, "guidance_scale", 3.0),
        seed=seed,
    )

    # Setup output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = BASELINE_OUTPUT_DIR / f"{preset_name}_seed{seed}_{timestamp}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating baseline from preset: {preset_name}")
    logger.info(f"Prompt: {prompt[:80]}...")
    logger.info(f"Config: {config.num_frames} frames, {config.height}x{config.width}")
    logger.info(f"Output: {output_dir}")

    # Clean GPU before generation
    _cleanup_gpu()

    # Track VRAM usage
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Generate video
    backend = get_backend()
    start_time = time.time()

    result = backend.generate_video(
        prompt=prompt,
        config=config,
        output_dir=output_dir,
        save_video=save_video,
    )

    generation_time = time.time() - start_time
    peak_vram = 0.0
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

    # Compute video statistics
    video = result.video
    if video.dtype == torch.uint8:
        video_float = video.float() / 255.0
    else:
        video_float = video.float()

    baseline_result = BaselineResult(
        prompt=prompt,
        seed=seed,
        config_tier=preset_name,
        output_path=output_dir / "video.mp4",
        frames_generated=video.shape[0],
        height=video.shape[1],
        width=video.shape[2],
        generation_time_seconds=generation_time,
        peak_vram_gb=peak_vram,
        mean_pixel_value=float(video_float.mean()),
        std_pixel_value=float(video_float.std()),
        min_pixel_value=float(video_float.min()),
        max_pixel_value=float(video_float.max()),
        backend_name=backend.name,
        timestamp=datetime.now().isoformat(),
    )

    # Save baseline metadata with preset info
    metadata = {
        **baseline_result.to_dict(),
        "preset_name": preset_name,
        "preset_description": preset.description if hasattr(preset, "description") else None,
        "generation_parameters": {
            "num_frames": config.num_frames,
            "height": config.height,
            "width": config.width,
            "num_inference_steps": config.num_inference_steps,
            "guidance_scale": config.guidance_scale,
        },
    }

    with open(output_dir / "baseline_result.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Also save video_metadata.json for Z-Image test pattern compatibility
    with open(output_dir / "video_metadata.json", "w") as f:
        json.dump({
            "timestamp": baseline_result.timestamp,
            "video_file": "video.mp4",
            "prompt": prompt,
            "seed": seed,
            "num_frames": config.num_frames,
            "height": config.height,
            "width": config.width,
            "num_inference_steps": config.num_inference_steps,
            "guidance_scale": config.guidance_scale,
            "preset_name": preset_name,
            "video_stats": {
                "mean": baseline_result.mean_pixel_value,
                "std": baseline_result.std_pixel_value,
                "min": baseline_result.min_pixel_value,
                "max": baseline_result.max_pixel_value,
            },
            "performance": {
                "generation_time_seconds": generation_time,
                "peak_vram_gb": peak_vram,
            },
        }, f, indent=2)

    logger.info(f"Generation complete in {generation_time:.1f}s")
    logger.info(f"Peak VRAM: {peak_vram:.1f} GB")
    logger.info(f"Video stats: mean={baseline_result.mean_pixel_value:.4f}, std={baseline_result.std_pixel_value:.4f}")

    # Cleanup
    _cleanup_gpu()

    return baseline_result


def _load_video_frames(video_path: Path) -> np.ndarray:
    """Load video frames as numpy array.

    Args:
        video_path: Path to video file

    Returns:
        Numpy array of shape [F, H, W, C] with uint8 values
    """
    import subprocess
    import tempfile

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Get video info using ffprobe
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-count_packets",
        "-show_entries", "stream=width,height,nb_read_packets",
        "-of", "csv=p=0",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    width, height, num_frames = map(int, result.stdout.strip().split(","))

    # Extract frames using ffmpeg
    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as f:
        temp_path = f.name

    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(video_path),
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            temp_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)

        # Read raw frames
        frames = np.fromfile(temp_path, dtype=np.uint8)
        frames = frames.reshape(num_frames, height, width, 3)
        return frames
    finally:
        Path(temp_path).unlink(missing_ok=True)


def _compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Structural Similarity Index between two images.

    Uses a simplified SSIM implementation without external dependencies.

    Args:
        img1: First image [H, W, C] or [H, W]
        img2: Second image [H, W, C] or [H, W]

    Returns:
        SSIM value between 0 and 1
    """
    # Convert to grayscale if color
    if img1.ndim == 3:
        img1 = np.mean(img1, axis=2)
    if img2.ndim == 3:
        img2 = np.mean(img2, axis=2)

    # Constants for stability
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # Mean
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)

    # Variance and covariance
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

    # SSIM formula
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)

    return float(numerator / denominator)


def _compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio between two images.

    Args:
        img1: First image
        img2: Second image

    Returns:
        PSNR value in dB (higher is better, inf for identical images)
    """
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    max_pixel = 255.0
    return float(20 * np.log10(max_pixel / np.sqrt(mse)))


def compare_baselines(
    video1_path: Path,
    video2_path: Path,
) -> ComparisonResult:
    """Compare two baseline videos frame-by-frame.

    Computes similarity metrics including SSIM, PSNR, MAE, and MSE.

    Args:
        video1_path: Path to first video
        video2_path: Path to second video

    Returns:
        ComparisonResult with similarity metrics
    """
    video1_path = Path(video1_path)
    video2_path = Path(video2_path)

    logger.info(f"Comparing: {video1_path.name} vs {video2_path.name}")

    # Load videos
    frames1 = _load_video_frames(video1_path)
    frames2 = _load_video_frames(video2_path)

    # Check dimensions match
    if frames1.shape != frames2.shape:
        logger.warning(f"Shape mismatch: {frames1.shape} vs {frames2.shape}")
        # Truncate to minimum frames
        min_frames = min(frames1.shape[0], frames2.shape[0])
        frames1 = frames1[:min_frames]
        frames2 = frames2[:min_frames]

    # Check if bit-identical
    is_identical = np.array_equal(frames1, frames2)

    # Compute per-frame metrics
    per_frame_ssim = []
    per_frame_psnr = []

    for i in range(frames1.shape[0]):
        ssim = _compute_ssim(frames1[i], frames2[i])
        psnr = _compute_psnr(frames1[i], frames2[i])
        per_frame_ssim.append(ssim)
        per_frame_psnr.append(psnr)

    # Compute overall metrics
    mean_ssim = float(np.mean(per_frame_ssim))
    mean_psnr = float(np.mean([p for p in per_frame_psnr if p != float("inf")]) if per_frame_psnr else 0)

    # MAE and MSE
    mae = float(np.mean(np.abs(frames1.astype(np.float64) - frames2.astype(np.float64))))
    mse = float(np.mean((frames1.astype(np.float64) - frames2.astype(np.float64)) ** 2))

    result = ComparisonResult(
        ssim=mean_ssim,
        psnr=mean_psnr,
        mae=mae,
        mse=mse,
        per_frame_ssim=per_frame_ssim,
        per_frame_psnr=per_frame_psnr,
        is_identical=is_identical,
        is_similar=mean_ssim > 0.95,
    )

    logger.info(f"SSIM: {mean_ssim:.4f}, PSNR: {mean_psnr:.2f} dB")
    logger.info(f"MAE: {mae:.2f}, MSE: {mse:.2f}")
    logger.info(f"Identical: {is_identical}, Similar: {result.is_similar}")

    return result


def verify_reproducibility(
    config_tier: str = "smoke",
    seed: int = 42,
    num_runs: int = 2,
) -> Tuple[bool, List[BaselineResult], Optional[ComparisonResult]]:
    """Verify that generation is reproducible with the same seed.

    Generates the same prompt/seed multiple times and compares outputs.

    Args:
        config_tier: Configuration tier to test
        seed: Seed to use for all runs
        num_runs: Number of times to run generation

    Returns:
        Tuple of (is_reproducible, results, comparison)
    """
    logger.info(f"Verifying reproducibility: {config_tier} x{num_runs} (seed={seed})")

    results = []
    for i in range(num_runs):
        logger.info(f"Run {i+1}/{num_runs}")
        result = generate_baseline(
            config_tier=config_tier,
            seed=seed,
            output_dir=BASELINE_OUTPUT_DIR / f"repro_test_{config_tier}_run{i}",
        )
        results.append(result)

    # Compare first and last run
    if len(results) >= 2:
        comparison = compare_baselines(
            results[0].output_path,
            results[-1].output_path,
        )
        is_reproducible = comparison.is_identical or comparison.ssim > 0.99
        return is_reproducible, results, comparison

    return True, results, None


if __name__ == "__main__":
    """Run baseline generation from command line."""
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Generate LTX-2 baseline videos")
    parser.add_argument(
        "--tier",
        choices=["smoke", "short", "reference"],
        default="smoke",
        help="Config tier to use",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt (uses tier default if not specified)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory",
    )
    parser.add_argument(
        "--verify-reproducibility",
        action="store_true",
        help="Run reproducibility verification",
    )

    args = parser.parse_args()

    if args.verify_reproducibility:
        is_repro, results, comparison = verify_reproducibility(
            config_tier=args.tier,
            seed=args.seed,
        )
        print(f"\nReproducibility: {'PASS' if is_repro else 'FAIL'}")
        if comparison:
            print(f"SSIM: {comparison.ssim:.6f}")
    else:
        result = generate_baseline(
            config_tier=args.tier,
            seed=args.seed,
            prompt=args.prompt,
            output_dir=args.output_dir,
        )
        print(f"\nBaseline generated: {result.output_path}")
        print(f"Time: {result.generation_time_seconds:.1f}s")
        print(f"VRAM: {result.peak_vram_gb:.1f} GB")
