#!/usr/bin/env python3
"""
LTX-2 Chunk Boundary Analysis Experiment

Last Updated: 2026-01-17

Test the hypothesis that LTX-2's 8-frame temporal compression creates
"chunk boundaries" where state transitions happen more cleanly than mid-chunk.

Hypothesis:
- VAE compresses 8 pixel frames → 1 latent frame
- Transitions at frame 8→9 (boundary) may be sharper than frame 4→5 (mid-chunk)
- Position embeddings might behave differently near boundaries

Test Strategy:
1. Generate videos with continuous motion (metronome, walking, rolling ball)
2. Compare frame-to-frame consistency at boundaries vs mid-chunk
3. Visual inspection for motion hitches at chunk boundaries
4. Quantitative: LPIPS perceptual distance, optical flow discontinuity

Frame Count Constraints:
- Must satisfy: (num_frames - 1) % 8 == 0
- Valid counts: 9, 17, 25, 33, 41, 49...
- Latent frames: 1 + (pixel_frames - 1) / 8

Memory-optimized for 24GB GPUs (RTX 4090).
Migrated to use LTX2ExperimentBase for standardized infrastructure.

Usage:
    # Quick test (2 frame counts x 2 prompts)
    uv run python experiments/ltx2/chunk_boundary_analysis.py --quick

    # Full sweep (4 frame counts x 5 prompts)
    uv run python experiments/ltx2/chunk_boundary_analysis.py
"""

import argparse
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from experiments.ltx2.base import LTX2ExperimentBase

logger = logging.getLogger(__name__)

# Valid frame counts: (num_frames - 1) % 8 == 0
FRAME_COUNTS = [17, 25, 33, 41]  # 2, 3, 4, 5 latent frames
QUICK_FRAME_COUNTS = [17, 33]

# Continuous motion prompts designed to reveal temporal discontinuities
CONTINUOUS_MOTION_PROMPTS = {
    "metronome": (
        "INT. MUSIC STUDIO – AFTERNOON. A classic wooden metronome sits on a grand piano, "
        "its golden arm swinging steadily from left to right. The camera holds a static "
        "medium shot, capturing the rhythmic pendulum motion against the dark wood of "
        "the piano's surface.\n"
        "The metronome clicks softly with each swing – tick, tick, tick – marking perfect "
        "time. A pianist's hand enters frame from the bottom, fingers hovering over the "
        "keys. The pianist, an older gentleman, speaks softly: \"Sixty beats per minute. "
        "The heartbeat of music.\"\n"
        "The arm swings left. Click. Right. Click. Left. Click. The reflection of "
        "afternoon sunlight glints off the brass weight as it passes center."
    ),
    "ball_rolling": (
        "INT. PHYSICS LAB – DAY. A white billiard ball rests at the left edge of a "
        "polished black table marked with evenly spaced vertical white lines creating "
        "a measurement grid. The camera holds a static overhead shot looking straight "
        "down at the smooth surface.\n"
        "A researcher's hand enters frame, wearing a white lab coat sleeve. She gives "
        "the ball a gentle push. \"Uniform velocity test, trial seven,\" she announces "
        "calmly. The ball begins rolling smoothly from left to right across the grid "
        "lines, each line passing beneath it at regular intervals."
    ),
    "pendulum": (
        "INT. CLOCKMAKER'S WORKSHOP – EVENING. A brass pendulum hangs from an ornate "
        "grandfather clock, swinging in a slow, graceful arc. Golden lamplight catches "
        "the polished metal as it passes through center, creating soft reflections on "
        "the wooden clock case behind it.\n"
        "The camera frames a tight shot on the pendulum itself, tracking its hypnotic "
        "left-to-right motion. An elderly clockmaker watches from the background."
    ),
    "walking_person": (
        "EXT. BEACH BOARDWALK – SUNSET. A young woman in a flowing white sundress walks "
        "along a wooden boardwalk, her bare feet padding softly against the weathered "
        "planks. The camera tracks alongside her in a smooth lateral dolly shot, keeping "
        "pace with her steady stride.\n"
        "One step, two steps, three steps – her gait is unhurried and rhythmic. Behind "
        "her, the waves roll in with distant white foam. The boardwalk planks provide "
        "natural markers of her progress."
    ),
    "rotating_fan": (
        "INT. VINTAGE DINER – AFTERNOON. An old ceiling fan spins lazily overhead, its "
        "wooden blades casting slow-moving shadows across red vinyl booths below. The "
        "camera tilts up to frame the fan against pressed tin ceiling tiles.\n"
        "The fan rotates with a soft whir, one blade after another passing through frame "
        "in steady succession. Sunlight from a window catches each blade as it turns."
    ),
}

QUICK_PROMPTS = ["metronome", "ball_rolling"]


class ChunkBoundaryAnalysisExperiment(LTX2ExperimentBase):
    """
    Analyze temporal chunk boundaries in LTX-2 generation.

    Generates videos at different frame counts using continuous motion prompts,
    then analyzes frame-to-frame consistency at chunk boundaries vs mid-chunk.

    Uses memory-optimized two-phase pattern:
    1. setup(): Load encoder, batch encode prompts
    2. run_iteration(): Generate at various frame counts, analyze boundaries
    """

    def __init__(
        self,
        output_dir: str = "experiments/results",
        frame_counts: Optional[List[int]] = None,
        quick: bool = False,
        seed: int = 42,
        num_inference_steps: int = 25,
        guidance_scale: float = 3.0,
        height: int = 512,
        width: int = 768,
        num_blocks_per_group: int = 1,
    ):
        super().__init__("chunk_boundary_analysis", output_dir)
        self.frame_counts = frame_counts or FRAME_COUNTS
        self.quick = quick
        self.seed = seed
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.height = height
        self.width = width
        self.num_blocks_per_group = num_blocks_per_group

        # Will be set in setup()
        self.prompt_names = []
        self.prompts = {}
        self.negative_prompt_embeds = None
        self.negative_attention_mask = None

        if quick:
            self.frame_counts = QUICK_FRAME_COUNTS
            self.prompts = {k: CONTINUOUS_MOTION_PROMPTS[k] for k in QUICK_PROMPTS}
        else:
            self.prompts = CONTINUOUS_MOTION_PROMPTS

    def setup(self) -> None:
        """
        Two-phase setup: encode all prompts, then load model.

        Since we test different frame counts with the SAME prompts,
        we only encode each prompt once (unlike layer experiments).
        """
        self.prompt_names = list(self.prompts.keys())

        total_gens = len(self.frame_counts) * len(self.prompt_names)
        logger.info(f"Chunk Boundary Analysis: {len(self.frame_counts)} frame counts × {len(self.prompt_names)} prompts = {total_gens}")
        logger.info(f"Frame counts: {self.frame_counts}")

        # Phase 1: Encoding
        logger.info("Phase 1: Loading encoder and encoding prompts")
        self.load_encoder()

        # Encode negative prompt for CFG
        logger.info("  Encoding negative prompt (empty string for CFG)")
        neg_result = self.encoder.encode_with_layer_masking(
            "",
            active_layers=list(range(49)),
            masking_mode="soft",
            return_packed=True,
        )
        self.negative_prompt_embeds = neg_result['prompt_embeds'].cpu()
        self.negative_attention_mask = neg_result['attention_mask'].cpu()

        # Encode each prompt once (reused across all frame counts)
        self._embeddings_cache = {}
        for pi, prompt_name in enumerate(self.prompt_names):
            prompt_text = self.prompts[prompt_name]
            logger.info(f"  [{pi + 1}/{len(self.prompt_names)}] Encoding: {prompt_name}")

            embeds = self.encode(
                prompt_text,
                return_packed=True,
            )
            self._embeddings_cache[prompt_name] = embeds.cpu()

        logger.info(f"Cached {len(self._embeddings_cache)} embeddings")

        # Phase 2: Generation
        logger.info("Phase 2: Offloading encoder and loading model")
        self.offload_encoder()
        self.load_model(
            use_pure_pytorch=False,
            use_group_offloading=True,
            num_blocks_per_group=self.num_blocks_per_group,
        )

    def run_iteration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate video at specific frame count, extract all frames."""
        frame_count = config["frame_count"]
        prompt_name = config["prompt_name"]
        prompt_text = self.prompts[prompt_name]

        logger.info(f"Generating: {prompt_name} × {frame_count} frames")

        # Get cached embeddings
        embeds = self._embeddings_cache[prompt_name].to(self.device)

        # Generate
        generator = torch.Generator(device="cpu").manual_seed(self.seed)

        output = self.pipeline(
            prompt_embeds=embeds,
            prompt_attention_mask=None,
            negative_prompt_embeds=self.negative_prompt_embeds.to(self.device),
            negative_prompt_attention_mask=self.negative_attention_mask.to(self.device),
            height=self.height,
            width=self.width,
            num_frames=frame_count,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            generator=generator,
        )
        frames = output.frames[0]

        # Calculate boundary metrics
        boundary_analysis = self.analyze_boundaries(frames, frame_count)

        # Save outputs
        sample_name = f"{prompt_name}_frames_{frame_count}"

        # Save video
        video_path = self.save_video(
            frames, sample_name, prompt_text,
            {
                "frame_count": frame_count,
                "latent_frames": (frame_count - 1) // 8 + 1,
                "seed": self.seed,
                **boundary_analysis,
            },
        )

        # Save first frame
        image_dir = self.run_dir / "images"
        image_dir.mkdir(exist_ok=True)
        frames[0].save(image_dir / f"{sample_name}.png")

        # Save all frames for analysis
        frames_dir = self.run_dir / "frames" / sample_name
        frames_dir.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frames):
            frame.save(frames_dir / f"frame_{i:03d}.png")

        return {
            "prompt": prompt_name,
            "frame_count": frame_count,
            "latent_frames": (frame_count - 1) // 8 + 1,
            "video_path": str(video_path),
            **boundary_analysis,
        }

    def analyze_boundaries(self, frames: List[Image.Image], num_frames: int) -> Dict[str, Any]:
        """
        Analyze frame-to-frame differences at chunk boundaries vs mid-chunk.

        Chunk boundaries occur every 8 frames (frames 8, 16, 24, ...)
        """
        # Convert frames to numpy arrays
        frame_arrays = [np.array(f).astype(np.float32) / 255.0 for f in frames]

        # Calculate frame-to-frame differences (L2 norm per pixel, averaged)
        diffs = []
        for i in range(len(frame_arrays) - 1):
            diff = np.sqrt(np.sum((frame_arrays[i + 1] - frame_arrays[i]) ** 2, axis=-1)).mean()
            diffs.append(diff)

        diffs = np.array(diffs)

        # Identify boundary frames (multiples of 8, except 0)
        # Frame indices: 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,...
        # Boundaries at: 8, 16, 24 (frame index where new chunk starts)
        boundary_indices = [i for i in range(7, len(diffs)) if (i + 1) % 8 == 0]  # 7,15,23 → diff to frame 8,16,24
        mid_indices = [i for i in range(len(diffs)) if (i + 1) % 8 == 4]  # Mid-chunk positions

        boundary_diffs = diffs[boundary_indices] if boundary_indices else np.array([])
        mid_diffs = diffs[mid_indices] if mid_indices else np.array([])

        return {
            "mean_diff_all": float(diffs.mean()) if len(diffs) > 0 else None,
            "mean_diff_boundary": float(boundary_diffs.mean()) if len(boundary_diffs) > 0 else None,
            "mean_diff_midchunk": float(mid_diffs.mean()) if len(mid_diffs) > 0 else None,
            "boundary_vs_mid_ratio": float(boundary_diffs.mean() / mid_diffs.mean())
                if len(boundary_diffs) > 0 and len(mid_diffs) > 0 and mid_diffs.mean() > 0 else None,
            "num_boundary_samples": len(boundary_indices),
            "num_mid_samples": len(mid_indices),
        }

    def get_run_configs(self) -> List[Dict[str, Any]]:
        """Generate all (frame_count, prompt) combinations."""
        configs = []
        for frame_count in self.frame_counts:
            for prompt_name in self.prompt_names:
                configs.append({"frame_count": frame_count, "prompt_name": prompt_name})
        return configs

    def aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate boundary analysis across all generations."""
        valid = [r for r in results if "error" not in r]

        if not valid:
            return {"error": "No valid results"}

        # Aggregate by frame count
        by_frame_count = defaultdict(list)
        for r in valid:
            by_frame_count[r["frame_count"]].append(r)

        frame_count_stats = {}
        for fc, fc_results in by_frame_count.items():
            boundary_ratios = [r["boundary_vs_mid_ratio"] for r in fc_results
                              if r.get("boundary_vs_mid_ratio") is not None]
            frame_count_stats[fc] = {
                "latent_frames": (fc - 1) // 8 + 1,
                "mean_boundary_ratio": float(np.mean(boundary_ratios)) if boundary_ratios else None,
                "std_boundary_ratio": float(np.std(boundary_ratios)) if len(boundary_ratios) > 1 else None,
                "n": len(fc_results),
            }

        return {
            "by_frame_count": frame_count_stats,
            "all_results": results,
        }

    def create_visualization(self, results: Dict[str, Any]) -> Path:
        """Create boundary analysis plot."""
        stats = results["by_frame_count"]

        fig, ax = plt.subplots(figsize=(10, 6))

        frame_counts = sorted(stats.keys())
        ratios = [stats[fc].get("mean_boundary_ratio") or 0 for fc in frame_counts]
        stds = [stats[fc].get("std_boundary_ratio") or 0 for fc in frame_counts]
        latent_frames = [stats[fc]["latent_frames"] for fc in frame_counts]

        x = range(len(frame_counts))
        ax.bar(x, ratios, yerr=stds, capsize=4, color='steelblue', edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{fc}\n({lf}L)" for fc, lf in zip(frame_counts, latent_frames)])
        ax.set_xlabel("Frame Count (Latent Frames)")
        ax.set_ylabel("Boundary/Mid-chunk Difference Ratio")
        ax.set_title("LTX-2 Chunk Boundary Analysis: Temporal Discontinuity")

        # Reference line at 1.0 (no difference)
        ax.axhline(y=1.0, color='red', linestyle='--', label='No difference (ratio=1.0)', alpha=0.7)
        ax.legend()

        plt.tight_layout()
        plot_path = self.run_dir / "chunk_boundary_analysis.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()

        return plot_path


def main():
    parser = argparse.ArgumentParser(description="LTX-2 Chunk Boundary Analysis")
    parser.add_argument("--output-dir", default="experiments/results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--quick", action="store_true", help="Quick test (2 frame counts × 2 prompts)")
    parser.add_argument("--frame-counts", type=int, nargs="+", help="Frame counts to test")
    parser.add_argument("--steps", type=int, default=25, help="Inference steps")
    parser.add_argument("--cfg", type=float, default=3.0, help="Guidance scale")
    parser.add_argument("--height", type=int, default=512, help="Video height")
    parser.add_argument("--width", type=int, default=768, help="Video width")
    parser.add_argument(
        "--blocks-per-group",
        type=int,
        default=1,
        help="Transformer blocks per offload group (1=min VRAM)",
    )
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Create experiment
    experiment = ChunkBoundaryAnalysisExperiment(
        output_dir=args.output_dir,
        frame_counts=args.frame_counts,
        quick=args.quick,
        seed=args.seed,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        height=args.height,
        width=args.width,
        num_blocks_per_group=args.blocks_per_group,
    )

    # Run experiment
    results = experiment.run(experiment.get_run_configs())

    # Create visualization
    plot_path = experiment.create_visualization(results)
    logger.info(f"Visualization saved to: {plot_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("CHUNK BOUNDARY ANALYSIS SUMMARY")
    print("=" * 60)

    print(f"\n{'Frame Count':<15} {'Latent':<8} {'Ratio':<12} {'Std':<10}")
    print("-" * 45)

    for fc, stats in sorted(results["by_frame_count"].items()):
        ratio_str = f"{stats['mean_boundary_ratio']:.4f}" if stats.get('mean_boundary_ratio') else "N/A"
        std_str = f"{stats['std_boundary_ratio']:.4f}" if stats.get('std_boundary_ratio') else "N/A"
        print(f"{fc:<15} {stats['latent_frames']:<8} {ratio_str:<12} {std_str:<10}")

    print(f"\nResults saved to: {experiment.run_dir}")
    print("\nInterpretation:")
    print("  - Ratio > 1.0: Larger diff at boundaries → discontinuity")
    print("  - Ratio ≈ 1.0: Consistent frame transitions")
    print("  - Ratio < 1.0: Smoother at boundaries (unlikely)")


if __name__ == "__main__":
    main()
