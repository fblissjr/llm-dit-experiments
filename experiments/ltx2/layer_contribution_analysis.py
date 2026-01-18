#!/usr/bin/env python3
"""
LTX-2 Layer Contribution Analysis

Last Updated: 2026-01-17

Computes per-layer contribution scores by isolating individual layers
BEFORE the normalization step. This is critical because the normalization
uses statistics from ALL layers, so modifying weights after encoding has
no effect.

**How it works:**
    1. For each layer L, encode the prompt with only layer L active
    2. Apply soft masking: inactive layers replaced with per-layer mean
    3. Pass through connectors (text_proj_in + video_connector)
    4. Generate video and score with SigLIP
    5. Higher score = layer contributes more semantic content

**Memory-optimized flow:**
    Phase 1: Load encoder, batch encode all (layer × prompt) combinations
    Phase 2: Offload encoder, load transformer + connectors + VAE
    Phase 3: Generate videos from cached embeddings

Output:
    layer_contributions.json - Per-layer contribution scores
    layer_weights.json - Normalized weights for router training

Usage:
    # Quick test (5 layers, 2 prompts)
    uv run python experiments/ltx2/layer_contribution_analysis.py --quick

    # Full analysis (all 49 layers, all prompts)
    uv run python experiments/ltx2/layer_contribution_analysis.py

    # Custom layers
    uv run python experiments/ltx2/layer_contribution_analysis.py --layers 0,10,20,30,40,48
"""

import argparse
import gc
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.ltx2.base import LTX2ExperimentBase
from llm_dit.data import get_all_prompts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class LayerContributionAnalyzer(LTX2ExperimentBase):
    """
    Analyzes per-layer contribution to video generation quality.

    Uses layer isolation: for each layer, keep ONLY that layer active
    and mask all others with their mean values. This reveals which layers
    carry the most semantic information for video conditioning.

    Results serve as:
    1. Ground truth for router training (proxy reward)
    2. Analysis of layer specialization in Gemma3 for LTX-2
    3. Guidance for layer selection in efficient inference
    """

    def __init__(
        self,
        output_dir: str = "experiments/results/layer_contributions",
        quick: bool = False,
    ):
        super().__init__("layer_contribution", output_dir)
        self.quick = quick

        # Experiment parameters
        self.layer_indices: List[int] = []
        self.prompts: Dict[str, str] = {}
        self.seed = 42

        # Cache for pre-encoded embeddings: (layer_idx, prompt_name) -> (embeds, mask)
        self._layer_embeddings: Dict = {}

    def configure(
        self,
        layer_indices: Optional[List[int]] = None,
        seed: int = 42,
    ) -> "LayerContributionAnalyzer":
        """
        Configure analysis parameters.

        Args:
            layer_indices: Layers to analyze (default: all 49 or 5 for quick)
            seed: Random seed for reproducibility

        Returns:
            self for chaining
        """
        if layer_indices is not None:
            self.layer_indices = layer_indices
        elif self.quick:
            # Evenly spaced layers for quick test
            self.layer_indices = [0, 12, 24, 36, 48]
        else:
            self.layer_indices = list(range(49))

        self.seed = seed
        return self

    def setup(self) -> None:
        """
        Two-phase setup for memory efficiency.

        Phase 1: Load encoder, batch encode all layer × prompt combinations
        Phase 2: Offload encoder, load transformer + connectors + VAE
        """
        # Get prompts
        self.prompts = get_all_prompts(quick=self.quick)
        if self.quick:
            # Even fewer prompts for quick mode
            self.prompts = dict(list(self.prompts.items())[:2])

        logger.info(f"Analyzing {len(self.layer_indices)} layers with {len(self.prompts)} prompts")
        logger.info(f"Total generations: {len(self.layer_indices) * len(self.prompts)}")

        # =====================================================================
        # Phase 1: Encoding
        # =====================================================================
        logger.info("Phase 1: Loading encoder and batch encoding...")
        self.load_encoder(use_8bit=True)

        # Pre-encode all layer × prompt combinations
        for layer_idx in self.layer_indices:
            logger.info(f"Encoding with layer {layer_idx} isolated...")
            for prompt_name, prompt_text in self.prompts.items():
                # Get packed embeddings with only this layer active
                packed_embeds, attn_mask = self.encode_packed(
                    prompt_text,
                    active_layers=[layer_idx],
                    masking_mode="soft",
                )
                # Move to CPU to free GPU memory
                self._layer_embeddings[(layer_idx, prompt_name)] = (
                    packed_embeds.cpu(),
                    attn_mask.cpu(),
                )

        logger.info(f"Cached {len(self._layer_embeddings)} embeddings")

        # =====================================================================
        # Phase 2: Generation setup
        # =====================================================================
        logger.info("Phase 2: Offloading encoder, loading generation components...")
        self.offload_encoder()

        # Load transformer, connectors, and VAE for pure PyTorch generation
        self.load_model(use_pure_pytorch=True)
        self.load_connectors()
        self.load_vae()

    def run_iteration(self, config: Dict) -> Dict:
        """
        Generate video for a single (layer, prompt) configuration.

        Args:
            config: Dict with 'layer_idx' and 'prompt_name'

        Returns:
            Dict with layer, prompt, and score
        """
        layer_idx = config["layer_idx"]
        prompt_name = config["prompt_name"]
        prompt_text = self.prompts[prompt_name]

        # Get cached embeddings
        packed_embeds, attn_mask = self._layer_embeddings[(layer_idx, prompt_name)]
        packed_embeds = packed_embeds.to(self.device)
        attn_mask = attn_mask.to(self.device)

        # Generate video
        video = self.generate_video(
            packed_embeds,
            attention_mask=attn_mask,
            num_frames=33,
            height=512,
            width=768,
            num_inference_steps=25,  # Reduced for speed
            guidance_scale=3.0,
            seed=self.seed,
        )

        # Score with SigLIP
        score = self.score_video(video, prompt_text)

        logger.info(f"Layer {layer_idx}, {prompt_name}: score={score:.4f}")

        # Optionally save video
        if self.run_dir is not None:
            self.save_video(
                video,
                f"layer{layer_idx:02d}_{prompt_name}",
                prompt_text,
                {"layer": layer_idx, "score": score},
            )

        return {
            "layer": layer_idx,
            "prompt": prompt_name,
            "score": float(score),
        }

    def aggregate_results(self, results: List[Dict]) -> Dict:
        """
        Compute per-layer statistics and rankings.

        Args:
            results: List of iteration results

        Returns:
            Dict with layer contributions, rankings, and normalized weights
        """
        # Group scores by layer
        layer_scores = {i: [] for i in self.layer_indices}
        for r in results:
            if "error" not in r:
                layer_scores[r["layer"]].append(r["score"])

        # Compute statistics per layer
        layer_contributions = {}
        for layer_idx in self.layer_indices:
            scores = layer_scores[layer_idx]
            if scores:
                layer_contributions[layer_idx] = {
                    "mean_score": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                    "min": float(np.min(scores)),
                    "max": float(np.max(scores)),
                    "n": len(scores),
                }
            else:
                layer_contributions[layer_idx] = {
                    "mean_score": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "n": 0,
                }

        # Rank layers by mean score (higher = better)
        ranked = sorted(
            layer_contributions.items(),
            key=lambda x: x[1]["mean_score"],
            reverse=True,
        )
        layer_ranking = [int(layer_idx) for layer_idx, _ in ranked]

        # Compute normalized weights for router training
        scores_array = np.array([
            layer_contributions[i]["mean_score"] for i in self.layer_indices
        ])
        # Shift to positive and normalize
        scores_positive = scores_array - scores_array.min() + 1e-6
        contribution_weights = scores_positive / scores_positive.sum()

        return {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "num_layers": len(self.layer_indices),
                "num_prompts": len(self.prompts),
                "seed": self.seed,
                "mode": "isolation",
                "masking": "soft",
            },
            "layer_contributions": {str(k): v for k, v in layer_contributions.items()},
            "layer_ranking": layer_ranking,
            "contribution_weights": {
                str(i): float(w) for i, w in zip(self.layer_indices, contribution_weights)
            },
            "all_results": results,
        }

    def get_run_configs(self) -> List[Dict]:
        """Generate all (layer, prompt) configurations."""
        configs = []
        for layer_idx in self.layer_indices:
            for prompt_name in self.prompts.keys():
                configs.append({
                    "layer_idx": layer_idx,
                    "prompt_name": prompt_name,
                })
        return configs


def main():
    parser = argparse.ArgumentParser(
        description="Analyze per-layer contribution to LTX-2 generation quality"
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/results/layer_contributions",
        help="Output directory for results",
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
        help="Comma-separated layer indices to analyze (default: all 49 or 5 for quick)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Parse layer indices if provided
    layer_indices = None
    if args.layers:
        layer_indices = [int(x.strip()) for x in args.layers.split(",")]

    # Create analyzer
    analyzer = LayerContributionAnalyzer(
        output_dir=args.output_dir,
        quick=args.quick,
    )

    # Configure
    analyzer.configure(
        layer_indices=layer_indices,
        seed=args.seed,
    )

    # Print summary
    logger.info("=" * 60)
    logger.info("LTX-2 Layer Contribution Analysis")
    logger.info("=" * 60)
    logger.info(f"Mode: isolation (layer masking before normalization)")
    logger.info(f"Layers: {len(analyzer.layer_indices)}")
    logger.info(f"Quick mode: {args.quick}")

    # Run analysis
    configs = analyzer.get_run_configs()
    results = analyzer.run(configs, save_results=True)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("LAYER CONTRIBUTION SUMMARY")
    logger.info("=" * 60)

    contributions = results["layer_contributions"]
    ranking = results["layer_ranking"]

    logger.info("\nTop 10 layers (best isolated performance):")
    for i, layer_idx in enumerate(ranking[:10]):
        contrib = contributions[str(layer_idx)]
        logger.info(f"  {i+1}. Layer {layer_idx}: score={contrib['mean_score']:.4f}")

    if len(ranking) > 5:
        logger.info("\nBottom 5 layers (worst isolated):")
        for layer_idx in ranking[-5:]:
            contrib = contributions[str(layer_idx)]
            logger.info(f"  Layer {layer_idx}: score={contrib['mean_score']:.4f}")

    # Save weights separately for easy loading in router training
    weights_file = Path(args.output_dir) / "layer_weights.json"
    with open(weights_file, "w") as f:
        json.dump(results["contribution_weights"], f, indent=2)
    logger.info(f"\nNormalized weights saved to {weights_file}")

    logger.info("\nAnalysis complete!")


if __name__ == "__main__":
    main()
