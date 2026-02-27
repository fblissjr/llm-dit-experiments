"""
Reference repo comparison for LTX-2 pipeline validation.

Last Updated: 2026-02-02

This module runs the official LTX-2 reference implementation and compares
its output against our implementation to verify correctness.

Usage:
    # Run full comparison
    uv run python -m tests.baselines.reference_comparison

    # Generate reference only
    uv run python -m tests.baselines.reference_comparison --reference-only

    # Compare existing videos
    uv run python -m tests.baselines.reference_comparison --compare-only \
        --ours outputs/baselines/smoke_seed42_xxx/video.mp4 \
        --reference outputs/baselines/reference_official.mp4

Prerequisites:
    - Reference repo at: coderef/LTX-2/
    - Models at: /home/fbliss/Storage/LTX-2/
    - Reference repo dependencies installed: cd coderef/LTX-2 && uv sync
"""

import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from tests.baselines.ltx2_baseline_runner import (
    BASELINE_OUTPUT_DIR,
    ComparisonResult,
    compare_baselines,
    generate_baseline,
)

logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
REFERENCE_REPO = PROJECT_ROOT / "coderef" / "LTX-2"
MODEL_PATH = Path("/home/fbliss/Storage/LTX-2")
ANALYSIS_DIR = PROJECT_ROOT / "internal" / "analysis"


@dataclass
class ReferenceComparisonConfig:
    """Configuration for reference comparison."""

    prompt: str = "A cat walking"
    seed: int = 42
    height: int = 512
    width: int = 768
    num_frames: int = 33
    num_inference_steps: int = 30
    guidance_scale: float = 3.0
    negative_prompt: str = ""
    enable_fp8: bool = True


@dataclass
class ReferenceComparisonResult:
    """Results from comparing our implementation against reference."""

    config: ReferenceComparisonConfig
    our_video_path: Path
    reference_video_path: Path
    comparison: ComparisonResult
    our_generation_time: float
    reference_generation_time: float
    timestamp: str

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "config": {
                "prompt": self.config.prompt,
                "seed": self.config.seed,
                "height": self.config.height,
                "width": self.config.width,
                "num_frames": self.config.num_frames,
                "num_inference_steps": self.config.num_inference_steps,
                "guidance_scale": self.config.guidance_scale,
                "enable_fp8": self.config.enable_fp8,
            },
            "our_video_path": str(self.our_video_path),
            "reference_video_path": str(self.reference_video_path),
            "comparison": self.comparison.to_dict(),
            "our_generation_time": self.our_generation_time,
            "reference_generation_time": self.reference_generation_time,
            "timestamp": self.timestamp,
        }


def check_reference_repo() -> bool:
    """Check if reference repo is available and set up."""
    if not REFERENCE_REPO.exists():
        logger.error(f"Reference repo not found at {REFERENCE_REPO}")
        return False

    # Check for uv.lock (indicates dependencies are installed)
    if not (REFERENCE_REPO / "uv.lock").exists():
        logger.warning("Reference repo uv.lock not found, may need to run 'uv sync'")

    return True


def check_models() -> bool:
    """Check if LTX-2 models are available."""
    required_files = [
        MODEL_PATH / "ltx-2-19b-distilled-fp8.safetensors",
        MODEL_PATH / "text_encoder",
        MODEL_PATH / "vae",
    ]

    for path in required_files:
        if not path.exists():
            logger.error(f"Required model file not found: {path}")
            return False

    return True


def generate_reference_video(
    config: ReferenceComparisonConfig,
    output_path: Path,
) -> float:
    """Generate video using official LTX-2 reference implementation.

    Args:
        config: Generation configuration
        output_path: Path to save the output video

    Returns:
        Generation time in seconds

    Raises:
        RuntimeError: If reference generation fails
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build command for reference pipeline
    cmd = [
        "uv", "run", "python", "-m", "ltx_pipelines.ti2vid_one_stage",
        "--checkpoint-path", str(MODEL_PATH / "ltx-2-19b-distilled-fp8.safetensors"),
        "--gemma-root", str(MODEL_PATH / "text_encoder"),
        "--prompt", config.prompt,
        "--output-path", str(output_path),
        "--seed", str(config.seed),
        "--height", str(config.height),
        "--width", str(config.width),
        "--num-frames", str(config.num_frames),
        "--num-inference-steps", str(config.num_inference_steps),
        "--video-cfg-guidance-scale", str(config.guidance_scale),
    ]

    if config.enable_fp8:
        cmd.append("--enable-fp8")

    if config.negative_prompt:
        cmd.extend(["--negative-prompt", config.negative_prompt])

    logger.info(f"Running reference pipeline: {' '.join(cmd[:10])}...")

    import time
    start_time = time.time()

    # Run from reference repo directory
    result = subprocess.run(
        cmd,
        cwd=REFERENCE_REPO,
        capture_output=True,
        text=True,
    )

    generation_time = time.time() - start_time

    if result.returncode != 0:
        logger.error(f"Reference generation failed:\n{result.stderr}")
        raise RuntimeError(f"Reference generation failed: {result.stderr}")

    if not output_path.exists():
        raise RuntimeError(f"Reference video not created at {output_path}")

    logger.info(f"Reference video generated in {generation_time:.1f}s: {output_path}")
    return generation_time


def run_comparison(
    config: Optional[ReferenceComparisonConfig] = None,
    output_dir: Optional[Path] = None,
) -> ReferenceComparisonResult:
    """Run full comparison between our implementation and reference.

    Args:
        config: Comparison configuration (uses defaults if None)
        output_dir: Output directory for videos and results

    Returns:
        ReferenceComparisonResult with metrics and paths
    """
    if config is None:
        config = ReferenceComparisonConfig()

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = BASELINE_OUTPUT_DIR / f"reference_comparison_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate our video
    logger.info("Generating video with our implementation...")
    our_result = generate_baseline(
        config_tier="smoke",
        seed=config.seed,
        prompt=config.prompt,
        output_dir=output_dir / "ours",
    )

    # Generate reference video
    logger.info("Generating video with reference implementation...")
    reference_output = output_dir / "reference" / "video.mp4"
    reference_time = generate_reference_video(config, reference_output)

    # Compare videos
    logger.info("Comparing videos...")
    comparison = compare_baselines(our_result.output_path, reference_output)

    result = ReferenceComparisonResult(
        config=config,
        our_video_path=our_result.output_path,
        reference_video_path=reference_output,
        comparison=comparison,
        our_generation_time=our_result.generation_time_seconds,
        reference_generation_time=reference_time,
        timestamp=datetime.now().isoformat(),
    )

    # Save results
    results_file = output_dir / "comparison_results.json"
    with open(results_file, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    logger.info(f"Results saved to {results_file}")

    return result


def generate_report(result: ReferenceComparisonResult) -> str:
    """Generate markdown report from comparison results."""
    report = f"""# LTX-2 Reference Comparison Report

*Generated: {result.timestamp}*

## Configuration

| Parameter | Value |
|-----------|-------|
| Prompt | "{result.config.prompt}" |
| Seed | {result.config.seed} |
| Resolution | {result.config.width}x{result.config.height} |
| Frames | {result.config.num_frames} |
| Steps | {result.config.num_inference_steps} |
| CFG Scale | {result.config.guidance_scale} |
| FP8 | {result.config.enable_fp8} |

## Generation Times

| Implementation | Time |
|----------------|------|
| Ours | {result.our_generation_time:.1f}s |
| Reference | {result.reference_generation_time:.1f}s |

## Similarity Metrics

| Metric | Value | Assessment |
|--------|-------|------------|
| SSIM | {result.comparison.ssim:.4f} | {"EXCELLENT" if result.comparison.ssim > 0.95 else "GOOD" if result.comparison.ssim > 0.85 else "INVESTIGATE"} |
| PSNR | {result.comparison.psnr:.2f} dB | {"EXCELLENT" if result.comparison.psnr > 30 else "GOOD" if result.comparison.psnr > 25 else "INVESTIGATE"} |
| MAE | {result.comparison.mae:.2f} | - |
| MSE | {result.comparison.mse:.2f} | - |
| Bit-identical | {result.comparison.is_identical} | - |

## Assessment

"""
    if result.comparison.is_identical:
        report += "**PERFECT MATCH**: Videos are bit-identical.\n"
    elif result.comparison.ssim > 0.95:
        report += "**EXCELLENT**: Videos are visually indistinguishable (SSIM > 0.95).\n"
    elif result.comparison.ssim > 0.85:
        report += "**GOOD**: Videos are similar with minor differences (SSIM 0.85-0.95).\n"
    else:
        report += """**INVESTIGATE**: Significant differences detected (SSIM < 0.85).

Possible causes:
1. **Scheduler differences** - Check sigma schedule step-by-step
2. **CFG formula** - Log noise_pred values at each step
3. **Text encoder** - Compare embedding outputs
4. **VAE** - Compare decoded latent statistics
5. **RNG differences** - Different random number generation

Recommended next steps:
1. Log intermediate values at each denoising step
2. Compare latent statistics (mean, std) at each step
3. Visual frame-by-frame comparison
"""

    report += f"""
## Output Paths

- Our video: `{result.our_video_path}`
- Reference video: `{result.reference_video_path}`

## Per-Frame SSIM

"""
    for i, ssim in enumerate(result.comparison.per_frame_ssim[:10]):
        report += f"- Frame {i}: {ssim:.4f}\n"

    if len(result.comparison.per_frame_ssim) > 10:
        report += f"- ... ({len(result.comparison.per_frame_ssim) - 10} more frames)\n"

    return report


def save_report(result: ReferenceComparisonResult, output_path: Optional[Path] = None):
    """Save comparison report to markdown file."""
    if output_path is None:
        ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = ANALYSIS_DIR / "reference_comparison_report.md"

    report = generate_report(result)

    with open(output_path, "w") as f:
        f.write(report)

    logger.info(f"Report saved to {output_path}")


def main():
    """Run reference comparison from command line."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Compare LTX-2 implementations")
    parser.add_argument(
        "--reference-only",
        action="store_true",
        help="Only generate reference video (skip our implementation)",
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Only compare existing videos",
    )
    parser.add_argument(
        "--ours",
        type=Path,
        help="Path to our video (for --compare-only)",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        help="Path to reference video (for --compare-only)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A cat walking",
        help="Prompt for generation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory",
    )

    args = parser.parse_args()

    # Validate prerequisites
    if not args.compare_only:
        if not check_reference_repo():
            sys.exit(1)
        if not check_models():
            sys.exit(1)

    if args.compare_only:
        # Compare existing videos
        if not args.ours or not args.reference:
            parser.error("--compare-only requires --ours and --reference")

        comparison = compare_baselines(args.ours, args.reference)
        print(f"\nComparison Results:")
        print(f"  SSIM: {comparison.ssim:.4f}")
        print(f"  PSNR: {comparison.psnr:.2f} dB")
        print(f"  Identical: {comparison.is_identical}")
        print(f"  Similar: {comparison.is_similar}")

    elif args.reference_only:
        # Generate reference only
        config = ReferenceComparisonConfig(
            prompt=args.prompt,
            seed=args.seed,
        )
        output_dir = args.output_dir or BASELINE_OUTPUT_DIR / "reference_official"
        output_path = output_dir / "video.mp4"
        generate_reference_video(config, output_path)

    else:
        # Full comparison
        config = ReferenceComparisonConfig(
            prompt=args.prompt,
            seed=args.seed,
        )
        result = run_comparison(config, args.output_dir)
        save_report(result)

        print(f"\n{'='*60}")
        print("COMPARISON COMPLETE")
        print(f"{'='*60}")
        print(f"SSIM: {result.comparison.ssim:.4f}")
        print(f"PSNR: {result.comparison.psnr:.2f} dB")
        print(f"Identical: {result.comparison.is_identical}")
        print(f"Similar: {result.comparison.is_similar}")
        print(f"\nReport: internal/analysis/reference_comparison_report.md")


if __name__ == "__main__":
    main()
