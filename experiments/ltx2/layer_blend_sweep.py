#!/usr/bin/env python3
"""
LTX-2 Layer Blend Sweep Experiment

Last Updated: 2026-01-17

Informed by projection matrix analysis:
- Late layers (43-47) contribute ~25% of signal when accounting for activations
- Early layers (0-4) contribute <1% of signal
- Layer 48 (final) is paradoxically low (0.02%)
- Projection W is nearly uniform; activation magnitudes create differentiation

This experiment tests weighted layer combinations using proper blending
(not zeroing, which creates OOD inputs).

Blends tested:
1. baseline: All 49 layers, uniform weights
2. late_heavy: Upweight layers 40-47 (where contribution is highest)
3. early_excluded: Zero weight on layers 0-10 (minimal contribution)
4. top_contributors: Only layers 43-47 (top 5 by contribution)
5. u_shaped: Early (0-16) + Late (40-48), skip middle
6. anti_u: Middle only (15-35), exclude early and late

Migrated to use LTX2ExperimentBase for standardized infrastructure.

Usage:
    # Quick test (3 blends × 2 prompts)
    uv run python experiments/ltx2/layer_blend_sweep.py --quick

    # Full sweep
    uv run python experiments/ltx2/layer_blend_sweep.py

    # With specific seed
    uv run python experiments/ltx2/layer_blend_sweep.py --seed 42
"""

import argparse
import json
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

# Layer blend configurations
# Each config specifies layer weights (non-active layers get 0 weight)
# Weights are normalized during encoding to preserve signal magnitude
BLEND_CONFIGS = {
    "baseline": {
        "description": "All 49 layers, uniform weights",
        "active_layers": list(range(49)),
        "weights": None,  # None means uniform
    },
    "late_heavy": {
        "description": "Upweight layers 40-47 (2x), others normal",
        "active_layers": list(range(49)),
        "weights": {i: (2.0 if 40 <= i <= 47 else 1.0) for i in range(49)},
    },
    "early_excluded": {
        "description": "Exclude layers 0-10 (near-zero contribution)",
        "active_layers": list(range(11, 49)),
        "weights": None,
    },
    "top_contributors": {
        "description": "Only layers 43-47 (~25% of baseline contribution)",
        "active_layers": list(range(43, 48)),
        "weights": None,
    },
    "late_only": {
        "description": "Only layers 40-48",
        "active_layers": list(range(40, 49)),
        "weights": None,
    },
    "u_shaped": {
        "description": "Early (0-16) + Late (40-48), skip middle",
        "active_layers": list(range(0, 17)) + list(range(40, 49)),
        "weights": None,
    },
    "anti_u": {
        "description": "Middle only (15-35), exclude early and late",
        "active_layers": list(range(15, 36)),
        "weights": None,
    },
    "bottom_excluded": {
        "description": "Exclude bottom 25 layers (0-24)",
        "active_layers": list(range(25, 49)),
        "weights": None,
    },
    "gradual": {
        "description": "Linear weight increase 0→1 across layers",
        "active_layers": list(range(49)),
        "weights": {i: (i + 1) / 49 for i in range(49)},
    },
    "exponential": {
        "description": "Exponential weight increase toward late layers",
        "active_layers": list(range(49)),
        "weights": {i: np.exp(i / 10) for i in range(49)},
    },
}

QUICK_CONFIGS = ["baseline", "late_heavy", "top_contributors"]


def create_layer_weights(
    active_layers: List[int],
    weights: Optional[Dict[int, float]] = None,
    num_layers: int = 49,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Create layer weight tensor from config.

    Args:
        active_layers: List of layer indices to activate
        weights: Optional dict mapping layer index to weight.
                 If None, active layers get weight 1.0
        num_layers: Total number of layers
        normalize: If True, normalize weights to preserve signal magnitude

    Returns:
        Tensor of shape [num_layers] with layer weights
    """
    layer_weights = torch.zeros(num_layers)

    if weights is None:
        for idx in active_layers:
            layer_weights[idx] = 1.0
    else:
        for idx in active_layers:
            layer_weights[idx] = weights.get(idx, 1.0)

    if normalize:
        weight_sum = layer_weights.sum()
        if weight_sum > 0:
            layer_weights = layer_weights / weight_sum * num_layers

    return layer_weights


class LayerBlendSweepExperiment(LTX2ExperimentBase):
    """
    Sweep over different layer weight configurations.

    Tests how different layer blending strategies affect generation quality.
    Uses memory-optimized two-phase pattern:
    1. setup(): Load encoder, batch encode ALL (config × prompt) combinations
    2. run_iteration(): Generate from cache, score, save
    """

    def __init__(
        self,
        output_dir: str = "experiments/results",
        quick: bool = False,
        seed: int = 42,
        num_inference_steps: int = 25,
        guidance_scale: float = 3.0,
        height: int = 512,
        width: int = 768,
        num_frames: int = 33,
        num_blocks_per_group: int = 1,
    ):
        super().__init__("layer_blend_sweep", output_dir)
        self.quick = quick
        self.seed = seed
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.num_blocks_per_group = num_blocks_per_group

        # Will be set in setup()
        self.config_names = []
        self.prompt_names = []
        self.prompts = {}
        self.negative_prompt_embeds = None
        self.negative_attention_mask = None

    def setup(self) -> None:
        """
        Two-phase setup: encode all, then load model.

        Phase 1: Encoder on GPU
        - Load encoder (8-bit)
        - Encode negative prompt (for CFG)
        - Batch encode all prompts with all layer configs
        - Cache embeddings to CPU

        Phase 2: Model on GPU
        - Offload encoder
        - Load transformer pipeline with group offloading
        """
        # Select configs and prompts
        self.config_names = QUICK_CONFIGS if self.quick else list(BLEND_CONFIGS.keys())
        self.prompts = get_all_prompts(quick=self.quick)
        self.prompt_names = list(self.prompts.keys())

        total_gens = len(self.config_names) * len(self.prompt_names)
        logger.info(f"Layer Blend Sweep: {len(self.config_names)} configs × {len(self.prompt_names)} prompts = {total_gens}")

        # Phase 1: Encoding
        logger.info("Phase 1: Loading encoder and encoding prompts")
        self.load_encoder()

        # Encode negative prompt for CFG
        logger.info("  Encoding negative prompt (empty string for CFG)")
        neg_result = self.encoder.encode_with_layer_masking(
            "",
            active_layers=list(range(49)),  # All layers
            masking_mode="soft",
            return_packed=True,
        )
        self.negative_prompt_embeds = neg_result['prompt_embeds'].cpu()
        self.negative_attention_mask = neg_result['attention_mask'].cpu()

        # Build configs for encode_batch
        layer_configs = []
        for config_name in self.config_names:
            cfg = BLEND_CONFIGS[config_name]
            layer_weights = create_layer_weights(
                active_layers=cfg["active_layers"],
                weights=cfg["weights"],
                normalize=True,
            )
            layer_configs.append({
                "name": config_name,
                "layer_weights": layer_weights,
                "active_layers": cfg["active_layers"],
            })

        # Batch encode: configs × prompts
        self._embeddings_cache = {}
        encode_idx = 0
        for ci, config in enumerate(layer_configs):
            config_name = config["name"]
            layer_weights = config["layer_weights"]

            for pi, prompt_name in enumerate(self.prompt_names):
                encode_idx += 1
                prompt_text = self.prompts[prompt_name]
                logger.info(f"  [{encode_idx}/{total_gens}] Encoding: {config_name} × {prompt_name}")

                # Encode with layer weights
                embeds = self.encode(
                    prompt_text,
                    layer_weights=layer_weights,
                    return_packed=True,
                )

                # Cache on CPU
                self._embeddings_cache[(ci, pi)] = embeds.cpu()

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
        """Generate video from cached embeddings, score, and save."""
        config_idx = config["config_idx"]
        prompt_idx = config["prompt_idx"]
        config_name = self.config_names[config_idx]
        prompt_name = self.prompt_names[prompt_idx]
        prompt_text = self.prompts[prompt_name]
        blend_config = BLEND_CONFIGS[config_name]

        logger.info(f"Generating: {config_name} × {prompt_name}")

        # Get cached embeddings
        embeds = self._embeddings_cache[(config_idx, prompt_idx)].to(self.device)

        # Generate
        generator = torch.Generator(device="cpu").manual_seed(self.seed)

        output = self.pipeline(
            prompt_embeds=embeds,
            prompt_attention_mask=None,  # Let pipeline handle
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
        sample_name = f"{config_name}_{prompt_name}"
        video_path = self.save_video(
            frames,
            sample_name,
            prompt_text,
            {
                "config": config_name,
                "description": blend_config["description"],
                "active_layers": blend_config["active_layers"],
                "num_active_layers": len(blend_config["active_layers"]),
                "score": score,
                "seed": self.seed,
            },
        )

        # Also save first frame as PNG
        image_path = video_path.with_suffix('.png')
        first_frame.save(image_path)

        return {
            "config": config_name,
            "prompt": prompt_name,
            "score": score,
            "num_active_layers": len(blend_config["active_layers"]),
            "video_path": str(video_path),
        }

    def get_run_configs(self) -> List[Dict[str, Any]]:
        """Generate all (config, prompt) combinations for run()."""
        configs = []
        for ci in range(len(self.config_names)):
            for pi in range(len(self.prompt_names)):
                configs.append({"config_idx": ci, "prompt_idx": pi})
        return configs

    def aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Group results by config and compute averages."""
        by_config = defaultdict(list)
        for r in results:
            if "error" not in r:
                by_config[r["config"]].append(r["score"])

        summary = {}
        for config_name, scores in by_config.items():
            blend_cfg = BLEND_CONFIGS[config_name]
            summary[config_name] = {
                "description": blend_cfg["description"],
                "num_active_layers": len(blend_cfg["active_layers"]),
                "mean_score": float(np.mean(scores)) if scores else None,
                "std_score": float(np.std(scores)) if len(scores) > 1 else None,
                "n": len(scores),
            }

        return {"by_config": summary, "all_results": results}

    def create_visualization(self, results: Dict[str, Any]) -> Path:
        """Create comparison plot of SigLIP scores by config."""
        summary = results["by_config"]

        # Sort by score (descending)
        sorted_configs = sorted(
            summary.items(),
            key=lambda x: x[1].get('mean_score') or 0,
            reverse=True
        )

        fig, ax = plt.subplots(figsize=(12, 6))

        sorted_names = [c[0] for c in sorted_configs]
        x = range(len(sorted_names))

        scores = [summary[c].get("mean_score") or 0 for c in sorted_names]
        stds = [summary[c].get("std_score") or 0 for c in sorted_names]
        layer_counts = [summary[c]["num_active_layers"] for c in sorted_names]

        # Color by layer count
        colors = plt.cm.viridis([count / 49 for count in layer_counts])

        ax.bar(x, scores, yerr=stds, capsize=4, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{n}\n({layer_counts[i]}L)" for i, n in enumerate(sorted_names)], rotation=45, ha="right")
        ax.set_ylabel("SigLIP Score (text-image alignment)")
        ax.set_xlabel("Layer Configuration (active layers)")
        ax.set_title("LTX-2 Layer Blend Sweep: Text-Image Alignment by Configuration")

        # Baseline reference line
        if "baseline" in summary and summary["baseline"].get("mean_score"):
            ax.axhline(y=summary["baseline"]["mean_score"], color='red', linestyle='--', label='Baseline', alpha=0.7)
            ax.legend()

        plt.tight_layout()
        plot_path = self.run_dir / "layer_blend_comparison.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()

        return plot_path


def main():
    parser = argparse.ArgumentParser(description="LTX-2 Layer Blend Sweep")
    parser.add_argument("--output-dir", default="experiments/results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--quick", action="store_true", help="Quick test (3 configs × 2 prompts)")
    parser.add_argument("--steps", type=int, default=25, help="Inference steps")
    parser.add_argument("--cfg", type=float, default=3.0, help="Guidance scale")
    parser.add_argument("--height", type=int, default=512, help="Video height")
    parser.add_argument("--width", type=int, default=768, help="Video width")
    parser.add_argument("--frames", type=int, default=33, help="Number of frames")
    parser.add_argument(
        "--blocks-per-group",
        type=int,
        default=1,
        help="Transformer blocks per offload group (1=min VRAM, higher=faster)",
    )
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Create and run experiment
    experiment = LayerBlendSweepExperiment(
        output_dir=args.output_dir,
        quick=args.quick,
        seed=args.seed,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        num_blocks_per_group=args.blocks_per_group,
    )

    # Run experiment
    results = experiment.run(experiment.get_run_configs())

    # Create visualization
    plot_path = experiment.create_visualization(results)
    logger.info(f"Visualization saved to: {plot_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY - SigLIP Scores by Layer Configuration")
    print("=" * 60)
    print(f"\n{'Config':<20} {'Layers':<8} {'SigLIP':<12} {'Std':<10} {'N':<5}")
    print("-" * 55)

    sorted_configs = sorted(
        results["by_config"].items(),
        key=lambda x: x[1].get('mean_score') or 0,
        reverse=True
    )

    for config_name, stats in sorted_configs:
        score_str = f"{stats['mean_score']:.4f}" if stats.get('mean_score') is not None else "N/A"
        std_str = f"{stats['std_score']:.4f}" if stats.get('std_score') is not None else "N/A"
        print(f"{config_name:<20} {stats['num_active_layers']:<8} {score_str:<12} {std_str:<10} {stats['n']:<5}")

    print(f"\nResults saved to: {experiment.run_dir}")


if __name__ == "__main__":
    main()
