#!/usr/bin/env python3
"""Generate video using Wan 2.1 T2V pipeline.

Usage:
    uv run python scripts/generate_wan_video.py --prompt "A cat sleeping"
    uv run python scripts/generate_wan_video.py --prompt "Ocean waves" --num-frames 33 --cfg-scale 7.0
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from llm_dit.pipelines.wan_video import WanVideoPipeline


def main():
    parser = argparse.ArgumentParser(description="Generate video with Wan 2.1 T2V")

    parser.add_argument(
        "--model-path", type=str, default="models/Wan2.1-T2V-1.3B", help="Path to Wan model"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype",
    )

    # Prompts
    parser.add_argument(
        "--prompt", "-p", type=str, required=True, help="Text prompt for video generation"
    )
    parser.add_argument(
        "--negative-prompt", "-n", type=str, default="blurry, low quality", help="Negative prompt"
    )

    # Video dimensions
    parser.add_argument(
        "--height", "-H", type=int, default=480, help="Video height (must be divisible by 16)"
    )
    parser.add_argument(
        "--width", "-W", type=int, default=832, help="Video width (must be divisible by 16)"
    )
    parser.add_argument(
        "--num-frames",
        "-f",
        type=int,
        default=17,
        help="Number of frames (must be 4N+1, e.g., 17, 33, 49, 81)",
    )

    # Generation params
    parser.add_argument("--num-steps", "-s", type=int, default=50, help="Number of denoising steps")
    parser.add_argument(
        "--cfg-scale", "-c", type=float, default=5.0, help="Classifier-free guidance scale"
    )
    parser.add_argument(
        "--shift", type=float, default=5.0, help="Flow matching shift (default: 5.0 for Wan 2.1)"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: random)")

    # Output
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output path (default: outputs/wan_<seed>.mp4)",
    )
    parser.add_argument("--fps", type=int, default=16, help="Output video FPS")

    # Analysis
    parser.add_argument(
        "--no-metrics", action="store_true", help="Skip frame-to-frame diff metrics"
    )

    args = parser.parse_args()

    # Parse dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    # Set seed
    if args.seed is None:
        args.seed = torch.randint(0, 2**32, (1,)).item()
    print(f"Using seed: {args.seed}")

    # Output path
    if args.output is None:
        Path("outputs").mkdir(exist_ok=True)
        args.output = f"outputs/wan_{args.seed}.mp4"

    # Load pipeline
    print(f"Loading model from {args.model_path}...")
    pipe = WanVideoPipeline.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
    )
    # Override shift if specified
    pipe.config.shift = args.shift

    # Generate
    print(f"Generating {args.num_frames} frames at {args.width}x{args.height}...")
    print(f"Prompt: {args.prompt}")
    print(f"Negative: {args.negative_prompt}")
    print(f"Steps: {args.num_steps}, CFG: {args.cfg_scale}, Shift: {args.shift}")

    video = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_steps,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
    )

    # Save
    pipe.save_video(video, args.output, fps=args.fps)
    print(f"Saved to {args.output}")

    # Metrics
    if not args.no_metrics:
        frames = video[0]  # [F, H, W, C]
        diffs = [
            np.abs(frames[i].astype(float) - frames[i + 1].astype(float)).mean()
            for i in range(len(frames) - 1)
        ]
                 for i in range(len(frames)-1)]
        print(f"\nFrame-to-frame diff metrics:")
        print(f"  Mean: {np.mean(diffs):.2f} (target: <5.0)")
        print(f"  Max:  {np.max(diffs):.2f}")
        print(f"  Min:  {np.min(diffs):.2f}")
        print(f"  Std:  {np.std(diffs):.2f}")


if __name__ == "__main__":
    main()
