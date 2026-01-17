#!/usr/bin/env python3
"""
LTX-2 Layer Contribution Analysis

Last Updated: 2026-01-17

Computes per-layer contribution scores by running single-layer ablations and
measuring the impact on generation quality. These scores can be used as:
1. Ground truth for router training (proxy reward)
2. Analysis of layer specialization
3. Informed layer selection for efficient inference

The analysis uses SigLIP to score text-video alignment for each layer config.

Output:
    layer_contributions.json - Per-layer contribution scores
    layer_correlation.json - Layer co-occurrence analysis

Usage:
    # Quick test (2 layers, 1 prompt)
    uv run python experiments/ltx2/layer_contribution_analysis.py --quick

    # Full analysis (all 49 layers, all prompts)
    uv run python experiments/ltx2/layer_contribution_analysis.py

    # Use results in router training
    uv run python experiments/ltx2/train_router.py --layer-contributions results/layer_contributions.json
"""

import argparse
import gc
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.ltx2.prompts import get_all_prompts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class LayerContributionAnalyzer:
    """Analyzes per-layer contribution to generation quality.

    Strategy:
        For each layer L, compute:
        1. Baseline score (all layers) vs ablated score (layer L zeroed)
        2. Delta = baseline - ablated
        3. Positive delta = layer contributes; negative = layer hurts

    The resulting contribution scores serve as training signal for the router.
    """

    def __init__(
        self,
        model_path: str = "models/LTX-2",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype

        # Components (lazy loaded)
        self.pipeline = None
        self.scorer = None
        self.original_weight = None

    def load_components(self):
        """Load pipeline and scorer."""
        from diffusers import LTX2Pipeline
        from llm_dit.utils.metrics import SigLIPScorer

        logger.info("Loading LTX-2 pipeline...")
        self.pipeline = LTX2Pipeline.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
        )

        # Store original projection weight for ablation
        self.original_weight = self.pipeline.connectors.text_proj_in.weight.data.clone()

        # Enable memory-efficient offloading
        self.pipeline.enable_sequential_cpu_offload()

        logger.info("Loading SigLIP scorer...")
        self.scorer = SigLIPScorer(device=self.device, dtype=self.dtype)

    def ablate_layer(self, layer_idx: int):
        """Zero out a single layer's contribution in the projection matrix."""
        hidden_dim = 3840
        start = layer_idx * hidden_dim
        end = (layer_idx + 1) * hidden_dim

        # Restore original weights first
        self.pipeline.connectors.text_proj_in.weight.data.copy_(self.original_weight)

        # Zero out this layer
        self.pipeline.connectors.text_proj_in.weight.data[:, start:end] = 0

    def restore_weights(self):
        """Restore original projection weights."""
        self.pipeline.connectors.text_proj_in.weight.data.copy_(self.original_weight)

    def generate_and_score(
        self,
        prompt: str,
        seed: int = 42,
        num_frames: int = 33,
        height: int = 512,
        width: int = 768,
        num_inference_steps: int = 25,
        guidance_scale: float = 3.0,
    ) -> float:
        """Generate video and return SigLIP alignment score."""
        generator = torch.Generator(device="cpu").manual_seed(seed)

        output = self.pipeline(
            prompt=prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        frames = output.frames[0]

        # Score with SigLIP
        _, mean_score = self.scorer.score_video(
            frames, prompt, sample_rate=max(1, len(frames) // 8)
        )

        return mean_score

    def compute_layer_contributions(
        self,
        prompts: dict[str, str],
        layer_indices: Optional[list[int]] = None,
        seed: int = 42,
        output_dir: str = "experiments/results/layer_contributions",
    ) -> dict:
        """Compute contribution score for each layer.

        For each layer:
            contribution = baseline_score - ablated_score

        Positive = layer helps; Negative = layer hurts

        Args:
            prompts: Dict of {name: prompt}
            layer_indices: Layers to analyze (default: all 49)
            seed: Random seed for reproducibility
            output_dir: Where to save results

        Returns:
            Dict with per-layer contribution scores
        """
        if layer_indices is None:
            layer_indices = list(range(49))

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.load_components()

        results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "num_prompts": len(prompts),
                "num_layers": len(layer_indices),
                "seed": seed,
            },
            "prompts": {},
            "layer_contributions": {},
        }

        # First, compute baseline scores (all layers active)
        logger.info("Computing baseline scores (all layers)...")
        self.restore_weights()

        baseline_scores = {}
        for name, prompt in prompts.items():
            logger.info(f"  Baseline: {name}")
            score = self.generate_and_score(prompt, seed=seed)
            baseline_scores[name] = score
            logger.info(f"    Score: {score:.4f}")
            gc.collect()
            torch.cuda.empty_cache()

        results["baseline_scores"] = baseline_scores

        # Now compute ablated scores for each layer
        layer_deltas = {i: [] for i in layer_indices}

        for layer_idx in layer_indices:
            logger.info(f"\nAnalyzing layer {layer_idx}...")
            self.ablate_layer(layer_idx)

            for name, prompt in prompts.items():
                logger.info(f"  Layer {layer_idx}, prompt: {name}")
                ablated_score = self.generate_and_score(prompt, seed=seed)

                delta = baseline_scores[name] - ablated_score
                layer_deltas[layer_idx].append(delta)

                if name not in results["prompts"]:
                    results["prompts"][name] = {
                        "baseline": baseline_scores[name],
                        "ablated": {},
                    }
                results["prompts"][name]["ablated"][str(layer_idx)] = {
                    "score": ablated_score,
                    "delta": delta,
                }

                logger.info(f"    Ablated: {ablated_score:.4f}, Delta: {delta:+.4f}")

                gc.collect()
                torch.cuda.empty_cache()

        # Compute aggregate layer contributions
        for layer_idx in layer_indices:
            deltas = layer_deltas[layer_idx]
            results["layer_contributions"][str(layer_idx)] = {
                "mean_delta": float(np.mean(deltas)),
                "std_delta": float(np.std(deltas)),
                "min_delta": float(np.min(deltas)),
                "max_delta": float(np.max(deltas)),
                "contributes": float(np.mean(deltas)) > 0,
            }

        # Rank layers by contribution
        ranked = sorted(
            results["layer_contributions"].items(),
            key=lambda x: x[1]["mean_delta"],
            reverse=True,
        )
        results["layer_ranking"] = [int(layer_idx) for layer_idx, _ in ranked]

        # Save results
        results_file = output_path / "layer_contributions.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nResults saved to {results_file}")

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("LAYER CONTRIBUTION SUMMARY")
        logger.info("=" * 60)

        logger.info("\nTop 10 contributing layers:")
        for i, (layer_idx, contrib) in enumerate(ranked[:10]):
            logger.info(
                f"  {i+1}. Layer {layer_idx}: delta={contrib['mean_delta']:+.4f}"
            )

        logger.info("\nBottom 5 layers (potential for removal):")
        for i, (layer_idx, contrib) in enumerate(ranked[-5:]):
            logger.info(
                f"  Layer {layer_idx}: delta={contrib['mean_delta']:+.4f}"
            )

        # Compute normalized contribution weights (for router)
        contributions = np.array([
            results["layer_contributions"][str(i)]["mean_delta"]
            for i in layer_indices
        ])
        # Shift to positive and normalize
        contributions_positive = contributions - contributions.min() + 1e-6
        contribution_weights = contributions_positive / contributions_positive.sum()

        results["contribution_weights"] = {
            str(i): float(w) for i, w in zip(layer_indices, contribution_weights)
        }

        # Save normalized weights separately for easy loading
        weights_file = output_path / "layer_weights.json"
        with open(weights_file, "w") as f:
            json.dump(results["contribution_weights"], f, indent=2)
        logger.info(f"Normalized weights saved to {weights_file}")

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Analyze per-layer contribution to LTX-2 generation quality"
    )
    parser.add_argument(
        "--model-path",
        default="models/LTX-2",
        help="Path to LTX-2 model",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/results/layer_contributions",
        help="Output directory",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: test 5 evenly-spaced layers with 2 prompts",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated layer indices to analyze (default: all 49)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # Get prompts
    prompts = get_all_prompts(quick=args.quick)
    if args.quick:
        # Even fewer prompts for quick mode
        prompts = dict(list(prompts.items())[:2])

    # Parse layer indices
    if args.layers:
        layer_indices = [int(x.strip()) for x in args.layers.split(",")]
    elif args.quick:
        # Test evenly spaced layers in quick mode
        layer_indices = [0, 12, 24, 36, 48]
    else:
        layer_indices = list(range(49))

    logger.info("=" * 60)
    logger.info("LTX-2 Layer Contribution Analysis")
    logger.info("=" * 60)
    logger.info(f"Prompts: {len(prompts)}")
    logger.info(f"Layers: {len(layer_indices)}")
    logger.info(f"Total generations: {len(prompts) * (len(layer_indices) + 1)}")

    analyzer = LayerContributionAnalyzer(model_path=args.model_path)
    results = analyzer.compute_layer_contributions(
        prompts=prompts,
        layer_indices=layer_indices,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    logger.info("\nAnalysis complete!")


if __name__ == "__main__":
    main()
