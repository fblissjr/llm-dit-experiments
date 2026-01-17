#!/usr/bin/env python3
"""
LTX-2 Layer Contribution Analysis

Last Updated: 2026-01-17

Computes per-layer contribution scores using two modes:

**Isolation Mode (default, recommended):**
    - Uses ONLY the specified layer, zeros all others
    - Reveals which layers can produce coherent output alone
    - Score = SigLIP alignment when using only that layer

**Ablation Mode:**
    - Removes one layer, keeps 48 others
    - Often shows zero delta due to layer redundancy
    - Delta = baseline_score - ablated_score

Outputs are used as:
1. Ground truth for router training (proxy reward)
2. Analysis of layer specialization
3. Informed layer selection for efficient inference

Output:
    layer_contributions.json - Per-layer contribution scores
    layer_weights.json - Normalized weights for router training

Usage:
    # Quick test with isolation mode (default)
    uv run python experiments/ltx2/layer_contribution_analysis.py --quick

    # Full analysis
    uv run python experiments/ltx2/layer_contribution_analysis.py

    # Use ablation mode (original, but often shows zero deltas)
    uv run python experiments/ltx2/layer_contribution_analysis.py --mode ablation --quick

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

    def isolate_layer(self, layer_idx: int):
        """Keep ONLY a single layer's contribution, zero all others.

        This is more informative than ablation because:
        - Ablation: Remove 2% of signal → minimal effect
        - Isolation: Keep 2% of signal → reveals layer's standalone contribution
        """
        hidden_dim = 3840

        # Start with zeros
        self.pipeline.connectors.text_proj_in.weight.data.zero_()

        # Copy only the specified layer's weights
        start = layer_idx * hidden_dim
        end = (layer_idx + 1) * hidden_dim
        self.pipeline.connectors.text_proj_in.weight.data[:, start:end] = \
            self.original_weight[:, start:end]

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
        mode: str = "isolation",
    ) -> dict:
        """Compute contribution score for each layer.

        Two modes available:

        **Isolation mode (recommended):**
            - Keep ONLY the specified layer, zero all others
            - Score = how well this layer performs alone
            - Better for finding which layers carry information

        **Ablation mode:**
            - Remove one layer, keep 48 others
            - Delta = baseline - ablated_score
            - Often shows zero delta due to redundancy

        Args:
            prompts: Dict of {name: prompt}
            layer_indices: Layers to analyze (default: all 49)
            seed: Random seed for reproducibility
            output_dir: Where to save results
            mode: "isolation" (recommended) or "ablation"

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
                "mode": mode,
            },
            "prompts": {},
            "layer_contributions": {},
        }

        # For ablation mode, compute baseline first
        baseline_scores = {}
        if mode == "ablation":
            logger.info("Computing baseline scores (all layers)...")
            self.restore_weights()

            for name, prompt in prompts.items():
                logger.info(f"  Baseline: {name}")
                score = self.generate_and_score(prompt, seed=seed)
                baseline_scores[name] = score
                logger.info(f"    Score: {score:.4f}")
                gc.collect()
                torch.cuda.empty_cache()

            results["baseline_scores"] = baseline_scores

        # Now compute scores for each layer
        layer_scores = {i: [] for i in layer_indices}

        for layer_idx in layer_indices:
            if mode == "isolation":
                logger.info(f"\nIsolating layer {layer_idx} (using ONLY this layer)...")
                self.isolate_layer(layer_idx)
            else:
                logger.info(f"\nAblating layer {layer_idx}...")
                self.ablate_layer(layer_idx)

            for name, prompt in prompts.items():
                logger.info(f"  Layer {layer_idx}, prompt: {name}")
                layer_score = self.generate_and_score(prompt, seed=seed)

                if mode == "isolation":
                    # In isolation mode, the score IS the contribution
                    layer_scores[layer_idx].append(layer_score)
                    result_key = "isolated"
                    logger.info(f"    Isolated score: {layer_score:.4f}")
                else:
                    # In ablation mode, delta = baseline - ablated
                    delta = baseline_scores[name] - layer_score
                    layer_scores[layer_idx].append(delta)
                    result_key = "ablated"
                    logger.info(f"    Ablated: {layer_score:.4f}, Delta: {delta:+.4f}")

                if name not in results["prompts"]:
                    results["prompts"][name] = {
                        "baseline": baseline_scores.get(name, 0),
                        result_key: {},
                    }
                results["prompts"][name][result_key][str(layer_idx)] = {
                    "score": layer_score,
                }

                gc.collect()
                torch.cuda.empty_cache()

        # Compute aggregate layer contributions
        metric_name = "mean_score" if mode == "isolation" else "mean_delta"
        for layer_idx in layer_indices:
            scores = layer_scores[layer_idx]
            results["layer_contributions"][str(layer_idx)] = {
                metric_name: float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
            }

        # Rank layers by contribution (higher score = better)
        ranked = sorted(
            results["layer_contributions"].items(),
            key=lambda x: x[1][metric_name],
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
        logger.info(f"LAYER CONTRIBUTION SUMMARY (mode={mode})")
        logger.info("=" * 60)

        if mode == "isolation":
            logger.info("\nTop 10 layers (best standalone performance):")
            for i, (layer_idx, contrib) in enumerate(ranked[:10]):
                logger.info(
                    f"  {i+1}. Layer {layer_idx}: score={contrib['mean_score']:.4f}"
                )

            logger.info("\nBottom 5 layers (worst standalone):")
            for i, (layer_idx, contrib) in enumerate(ranked[-5:]):
                logger.info(
                    f"  Layer {layer_idx}: score={contrib['mean_score']:.4f}"
                )
        else:
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
            results["layer_contributions"][str(i)][metric_name]
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
    parser.add_argument(
        "--mode",
        type=str,
        default="isolation",
        choices=["isolation", "ablation"],
        help="Analysis mode: 'isolation' (use only one layer) or 'ablation' (remove one layer). "
             "Isolation is recommended as ablation often shows zero deltas due to redundancy.",
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
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Prompts: {len(prompts)}")
    logger.info(f"Layers: {len(layer_indices)}")

    # In isolation mode, no baseline needed
    total_gens = len(prompts) * len(layer_indices)
    if args.mode == "ablation":
        total_gens += len(prompts)  # Add baseline generations
    logger.info(f"Total generations: {total_gens}")

    analyzer = LayerContributionAnalyzer(model_path=args.model_path)
    results = analyzer.compute_layer_contributions(
        prompts=prompts,
        layer_indices=layer_indices,
        seed=args.seed,
        output_dir=args.output_dir,
        mode=args.mode,
    )

    logger.info("\nAnalysis complete!")


if __name__ == "__main__":
    main()
