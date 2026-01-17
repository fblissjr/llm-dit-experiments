#!/usr/bin/env python3
"""
LTX-2 Layer Profile Sweep Experiment

Last Updated: 2026-01-17

Generate videos using each of the 49 Gemma layers in isolation to understand
what each layer contributes to video generation. Outputs are viewer-compatible
for visual comparison and analysis.

Discovery Question: "What does each Gemma layer contribute to video generation?"

Method:
1. Generate video for each layer in isolation (mask others)
2. Create 49xN grid (layers x prompts)
3. Extract first frame as PNG for viewer
4. Compute SigLIP per sample

Memory-optimized for 24GB GPUs (RTX 4090):
- Text encoder: 8-bit quantized (~13GB instead of ~54GB)
- Pipeline: Group offloading for transformer blocks
- Sequential loading: Encode first, then offload, then generate

Migrated to use LTX2ExperimentBase for standardized infrastructure.

Usage:
    # Full sweep (49 layers x 10 prompts = 490 generations)
    uv run python experiments/ltx2/layer_profile_sweep.py

    # Quick test (3 layers x 3 prompts)
    uv run python experiments/ltx2/layer_profile_sweep.py --quick

    # Custom layers
    uv run python experiments/ltx2/layer_profile_sweep.py --layers 0 1 23 47 48
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
from llm_dit.data import get_all_prompts

logger = logging.getLogger(__name__)

# LTX-2 Gemma configuration
NUM_GEMMA_LAYERS = 49  # Layers 0-48

# Quick test subset
QUICK_LAYERS = [0, 24, 48]  # Early, middle, late


class LayerProfileSweepExperiment(LTX2ExperimentBase):
    """
    Test each Gemma layer in isolation.

    Generates video using only a single layer at a time (with others masked),
    allowing analysis of what each layer contributes to generation.

    Uses memory-optimized two-phase pattern:
    1. setup(): Load encoder, batch encode ALL (layer × prompt) combinations
    2. run_iteration(): Generate from cache, score, save
    """

    def __init__(
        self,
        output_dir: str = "experiments/results",
        layers_to_test: Optional[List[int]] = None,
        quick: bool = False,
        seed: int = 42,
        num_inference_steps: int = 25,
        guidance_scale: float = 3.0,
        height: int = 512,
        width: int = 768,
        num_frames: int = 33,
        masking_mode: str = "soft",
        save_videos: bool = True,
        num_blocks_per_group: int = 1,
    ):
        super().__init__("layer_profile_sweep", output_dir)
        self.layers_to_test = layers_to_test or list(range(NUM_GEMMA_LAYERS))
        self.quick = quick
        self.seed = seed
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.masking_mode = masking_mode
        self.save_videos = save_videos
        self.num_blocks_per_group = num_blocks_per_group

        # Will be set in setup()
        self.prompt_names = []
        self.prompts = {}
        self.negative_prompt_embeds = None
        self.negative_attention_mask = None

        if quick:
            self.layers_to_test = QUICK_LAYERS

    def setup(self) -> None:
        """
        Two-phase setup: encode all, then load model.

        Phase 1: Encoder on GPU
        - Load encoder (8-bit)
        - Encode negative prompt (for CFG)
        - Batch encode all prompts with each layer in isolation
        - Cache embeddings to CPU

        Phase 2: Model on GPU
        - Offload encoder
        - Load transformer pipeline with group offloading
        """
        # Load prompts
        self.prompts = get_all_prompts(quick=self.quick)
        self.prompt_names = list(self.prompts.keys())

        total_gens = len(self.layers_to_test) * len(self.prompt_names)
        logger.info(f"Layer Profile Sweep: {len(self.layers_to_test)} layers × {len(self.prompt_names)} prompts = {total_gens}")
        logger.info(f"Masking mode: {self.masking_mode}")

        # Phase 1: Encoding
        logger.info("Phase 1: Loading encoder and encoding prompts")
        self.load_encoder()

        # Encode negative prompt for CFG
        logger.info("  Encoding negative prompt (empty string for CFG)")
        neg_result = self.encoder.encode_with_layer_masking(
            "",
            active_layers=list(range(NUM_GEMMA_LAYERS)),  # All layers
            masking_mode="soft",
            return_packed=True,
        )
        self.negative_prompt_embeds = neg_result['prompt_embeds'].cpu()
        self.negative_attention_mask = neg_result['attention_mask'].cpu()

        # Batch encode: layers × prompts
        self._embeddings_cache = {}
        encode_idx = 0

        for layer_idx in self.layers_to_test:
            logger.info(f"\n  Layer {layer_idx} (single layer active)")

            for pi, prompt_name in enumerate(self.prompt_names):
                encode_idx += 1
                prompt_text = self.prompts[prompt_name]
                logger.info(f"    [{encode_idx}/{total_gens}] Encoding: layer_{layer_idx:02d} × {prompt_name}")

                # Encode with only this layer active (others masked)
                embeds = self.encode(
                    prompt_text,
                    active_layers=[layer_idx],
                    masking_mode=self.masking_mode,
                    return_packed=True,
                )

                # Cache on CPU
                self._embeddings_cache[(layer_idx, pi)] = embeds.cpu()

        logger.info(f"\nCached {len(self._embeddings_cache)} embeddings")

        # Phase 2: Generation
        logger.info("Phase 2: Offloading encoder and loading model")
        self.offload_encoder()
        self.load_model(
            use_pure_pytorch=False,
            use_group_offloading=True,
            num_blocks_per_group=self.num_blocks_per_group,
        )

    def run_iteration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate video from cached embeddings, score, and save."""
        layer_idx = config["layer_idx"]
        prompt_idx = config["prompt_idx"]
        prompt_name = self.prompt_names[prompt_idx]
        prompt_text = self.prompts[prompt_name]

        logger.info(f"Generating: layer_{layer_idx:02d} × {prompt_name}")

        # Get cached embeddings
        embeds = self._embeddings_cache[(layer_idx, prompt_idx)].to(self.device)

        # Generate
        generator = torch.Generator(device="cpu").manual_seed(self.seed)

        output = self.pipeline(
            prompt_embeds=embeds,
            prompt_attention_mask=None,
            negative_prompt_embeds=self.negative_prompt_embeds.to(self.device),
            negative_prompt_attention_mask=self.negative_attention_mask.to(self.device),
            height=self.height,
            width=self.width,
            num_frames=self.num_frames,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            generator=generator,
        )
        frames = output.frames[0]

        # Extract first frame for scoring
        first_frame = frames[0]

        # Score
        score = self.score_video(
            torch.tensor(np.array(first_frame)).permute(2, 0, 1).unsqueeze(0),
            prompt_text,
        )

        # Save outputs
        sample_name = f"layer_{layer_idx:02d}_{prompt_name}"
        metadata = {
            "layer_idx": layer_idx,
            "masking_mode": self.masking_mode,
            "score": score,
            "seed": self.seed,
        }

        if self.save_videos:
            video_path = self.save_video(frames, sample_name, prompt_text, metadata)
        else:
            video_path = None

        # Always save first frame as PNG
        image_dir = self.run_dir / "images"
        image_dir.mkdir(exist_ok=True)
        image_path = image_dir / f"{sample_name}.png"
        first_frame.save(image_path)

        return {
            "layer_idx": layer_idx,
            "prompt": prompt_name,
            "score": score,
            "image_path": str(image_path),
            "video_path": str(video_path) if video_path else None,
        }

    def get_run_configs(self) -> List[Dict[str, Any]]:
        """Generate all (layer, prompt) combinations for run()."""
        configs = []
        for layer_idx in self.layers_to_test:
            for pi in range(len(self.prompt_names)):
                configs.append({"layer_idx": layer_idx, "prompt_idx": pi})
        return configs

    def aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute per-layer statistics."""
        valid = [r for r in results if "error" not in r]

        if not valid:
            return {"error": "No valid results"}

        # Per-layer statistics
        layer_stats = {}
        layers = sorted(set(r["layer_idx"] for r in valid))

        for layer_idx in layers:
            layer_results = [r for r in valid if r["layer_idx"] == layer_idx]
            scores = [r["score"] for r in layer_results if r.get("score") is not None]

            layer_stats[f"layer_{layer_idx:02d}"] = {
                "mean_score": float(np.mean(scores)) if scores else None,
                "std_score": float(np.std(scores)) if len(scores) > 1 else None,
                "n": len(scores),
            }

        # Per-prompt statistics
        prompt_stats = {}
        prompt_ids = sorted(set(r["prompt"] for r in valid))

        for prompt_id in prompt_ids:
            prompt_results = [r for r in valid if r["prompt"] == prompt_id]
            scores = [r["score"] for r in prompt_results if r.get("score") is not None]
            prompt_stats[prompt_id] = {
                "mean_score": float(np.mean(scores)) if scores else None,
                "std_score": float(np.std(scores)) if len(scores) > 1 else None,
            }

        return {
            "per_layer": layer_stats,
            "per_prompt": prompt_stats,
            "all_results": results,
            "total_valid": len(valid),
            "total_errors": len(results) - len(valid),
        }

    def create_visualization(self, results: Dict[str, Any]) -> Path:
        """Create layer contribution plot."""
        layer_stats = results["per_layer"]

        # Sort by layer index
        sorted_layers = sorted(
            layer_stats.items(),
            key=lambda x: int(x[0].split('_')[1])
        )

        fig, ax = plt.subplots(figsize=(14, 6))

        layer_nums = [int(x[0].split('_')[1]) for x in sorted_layers]
        scores = [x[1].get("mean_score") or 0 for x in sorted_layers]
        stds = [x[1].get("std_score") or 0 for x in sorted_layers]

        # Color by layer region
        colors = []
        for l in layer_nums:
            if l < 17:
                colors.append('lightblue')  # Early
            elif l < 35:
                colors.append('lightgreen')  # Middle
            else:
                colors.append('salmon')  # Late

        ax.bar(layer_nums, scores, yerr=stds, capsize=2, color=colors, edgecolor='black', linewidth=0.3)
        ax.set_xlabel("Layer Index")
        ax.set_ylabel("SigLIP Score (text-image alignment)")
        ax.set_title(f"LTX-2 Layer Profile: Per-Layer Contribution (masking={self.masking_mode})")

        # Add region labels
        ax.axvspan(-0.5, 16.5, alpha=0.1, color='blue', label='Early (0-16)')
        ax.axvspan(16.5, 34.5, alpha=0.1, color='green', label='Middle (17-34)')
        ax.axvspan(34.5, 48.5, alpha=0.1, color='red', label='Late (35-48)')
        ax.legend(loc='upper left')

        ax.set_xlim(-0.5, 48.5)
        plt.tight_layout()

        plot_path = self.run_dir / "layer_profile_chart.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()

        return plot_path

    def print_analysis(self, results: Dict[str, Any]) -> None:
        """Print layer contribution analysis."""
        layer_stats = results["per_layer"]

        print("\n" + "=" * 60)
        print("LAYER CONTRIBUTION ANALYSIS (SigLIP)")
        print("=" * 60)

        # Sort by score
        sorted_by_score = sorted(
            layer_stats.items(),
            key=lambda x: x[1].get('mean_score') or 0,
            reverse=True
        )

        print("\nTop 10 layers by SigLIP score (best text-image alignment):")
        print(f"  {'Layer':<10} {'SigLIP':<10} {'Std':<10}")
        print("  " + "-" * 30)
        for layer_name, stats in sorted_by_score[:10]:
            layer_num = int(layer_name.split('_')[1])
            score_str = f"{stats['mean_score']:.4f}" if stats.get('mean_score') else "N/A"
            std_str = f"{stats['std_score']:.4f}" if stats.get('std_score') else "N/A"
            print(f"  Layer {layer_num:2d}   {score_str:<10} {std_str:<10}")

        print("\nBottom 10 layers:")
        print(f"  {'Layer':<10} {'SigLIP':<10} {'Std':<10}")
        print("  " + "-" * 30)
        for layer_name, stats in sorted_by_score[-10:]:
            layer_num = int(layer_name.split('_')[1])
            score_str = f"{stats['mean_score']:.4f}" if stats.get('mean_score') else "N/A"
            std_str = f"{stats['std_score']:.4f}" if stats.get('std_score') else "N/A"
            print(f"  Layer {layer_num:2d}   {score_str:<10} {std_str:<10}")

        # Region averages
        early = [layer_stats[f"layer_{i:02d}"]["mean_score"]
                 for i in range(0, 17) if f"layer_{i:02d}" in layer_stats
                 and layer_stats[f"layer_{i:02d}"].get("mean_score")]
        middle = [layer_stats[f"layer_{i:02d}"]["mean_score"]
                  for i in range(17, 35) if f"layer_{i:02d}" in layer_stats
                  and layer_stats[f"layer_{i:02d}"].get("mean_score")]
        late = [layer_stats[f"layer_{i:02d}"]["mean_score"]
                for i in range(35, 49) if f"layer_{i:02d}" in layer_stats
                and layer_stats[f"layer_{i:02d}"].get("mean_score")]

        print("\nRegion averages:")
        if early:
            print(f"  Early (0-16):   {np.mean(early):.4f}")
        if middle:
            print(f"  Middle (17-34): {np.mean(middle):.4f}")
        if late:
            print(f"  Late (35-48):   {np.mean(late):.4f}")


def main():
    parser = argparse.ArgumentParser(description="LTX-2 Layer Profile Sweep")
    parser.add_argument("--output-dir", default="experiments/results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--quick", action="store_true", help="Quick test (3 layers × 3 prompts)")
    parser.add_argument("--layers", type=int, nargs="+", help="Layer indices to test")
    parser.add_argument("--steps", type=int, default=25, help="Inference steps")
    parser.add_argument("--cfg", type=float, default=3.0, help="Guidance scale")
    parser.add_argument("--height", type=int, default=512, help="Video height")
    parser.add_argument("--width", type=int, default=768, help="Video width")
    parser.add_argument("--frames", type=int, default=33, help="Number of frames")
    parser.add_argument(
        "--masking-mode",
        choices=["soft", "zero", "weighted"],
        default="soft",
        help="How to mask inactive layers (soft recommended)",
    )
    parser.add_argument("--no-save-videos", action="store_true", help="Only save first frames")
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
    experiment = LayerProfileSweepExperiment(
        output_dir=args.output_dir,
        layers_to_test=args.layers,
        quick=args.quick,
        seed=args.seed,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        masking_mode=args.masking_mode,
        save_videos=not args.no_save_videos,
        num_blocks_per_group=args.blocks_per_group,
    )

    # Run experiment
    results = experiment.run(experiment.get_run_configs())

    # Create visualization
    plot_path = experiment.create_visualization(results)
    logger.info(f"Visualization saved to: {plot_path}")

    # Print analysis
    experiment.print_analysis(results)

    print(f"\nResults saved to: {experiment.run_dir}")


if __name__ == "__main__":
    main()
