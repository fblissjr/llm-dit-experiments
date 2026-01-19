#!/usr/bin/env python3
"""
Generate reference video using official LTX-2 pipeline for 1:1 comparison.

Last Updated: 2026-01-19

This script uses the official LTX-2 pipeline with proper memory management
to generate videos that can be compared against our implementation.

Usage:
    uv run python scripts/generate_reference_video.py \
        --prompt "A cat walking through a garden" \
        --output outputs/reference/cat_walking.mp4
"""

import argparse
import gc
import logging
import sys
from pathlib import Path

import torch

# Add coderef to path for official LTX-2 imports
sys.path.insert(0, str(Path(__file__).parent.parent / "coderef/LTX-2/packages/ltx-core/src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "coderef/LTX-2/packages/ltx-pipelines/src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)8s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def cleanup_memory():
    """Free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def generate_reference_video(
    prompt: str,
    output_path: Path,
    num_frames: int = 121,
    height: int = 512,
    width: int = 768,
    num_inference_steps: int = 40,
    guidance_scale: float = 4.0,
    seed: int = 10,
    fp8: bool = True,
):
    """Generate video using official LTX-2 pipeline."""
    from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline
    from ltx_pipelines.utils.media_io import encode_video

    logger.info(f"Generating: {prompt}")
    logger.info(f"Config: {num_frames} frames, {height}x{width}, {num_inference_steps} steps, CFG {guidance_scale}")

    # Paths to model files
    checkpoint_path = "models/LTX-2/transformer/model.safetensors"
    gemma_root = "models/LTX-2/text_encoder/"

    # Create pipeline with FP8 for memory efficiency
    pipeline = TI2VidOneStagePipeline(
        checkpoint_path=checkpoint_path,
        gemma_root=gemma_root,
        loras=[],
        device=torch.device("cuda"),
        fp8transformer=fp8,
    )

    # Generate video
    video_iterator, audio = pipeline(
        prompt=prompt,
        negative_prompt="",
        seed=seed,
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=24.0,
        num_inference_steps=num_inference_steps,
        cfg_guidance_scale=guidance_scale,
        images=[],
        enhance_prompt=False,
    )

    # Collect video frames
    video_frames = list(video_iterator)
    if video_frames:
        video = video_frames[-1]  # Final denoised video
        logger.info(f"Video shape: {video.shape}")

        # Save video
        output_path.parent.mkdir(parents=True, exist_ok=True)
        encode_video(video, str(output_path), frame_rate=24)
        logger.info(f"Saved to: {output_path}")
    else:
        logger.error("No video frames generated")

    cleanup_memory()
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate reference video with LTX-2")
    parser.add_argument("--prompt", type=str, default="A cat walking through a garden",
                        help="Text prompt for video generation")
    parser.add_argument("--output", type=Path, default=Path("outputs/reference/reference.mp4"),
                        help="Output video path")
    parser.add_argument("--frames", type=int, default=121, help="Number of frames (must be 8k+1)")
    parser.add_argument("--height", type=int, default=512, help="Video height")
    parser.add_argument("--width", type=int, default=768, help="Video width")
    parser.add_argument("--steps", type=int, default=40, help="Inference steps")
    parser.add_argument("--cfg", type=float, default=4.0, help="CFG guidance scale")
    parser.add_argument("--seed", type=int, default=10, help="Random seed")
    parser.add_argument("--no-fp8", action="store_true", help="Disable FP8 quantization")

    args = parser.parse_args()

    generate_reference_video(
        prompt=args.prompt,
        output_path=args.output,
        num_frames=args.frames,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        seed=args.seed,
        fp8=not args.no_fp8,
    )


if __name__ == "__main__":
    main()
