#!/usr/bin/env python3
"""
LTX-2 Router Evaluation Experiment

Last Updated: 2026-01-17

Evaluates trained TokenLayerRouter by comparing generation quality
against baseline uniform routing.

Workflow:
    1. Load encoder + pipeline + trained router
    2. Encode prompts with router (per-token layer weights)
    3. Encode same prompts with uniform weights (baseline)
    4. Generate videos with both
    5. Score with SigLIP and compare

Usage:
    # Quick test (2 prompts, fast settings)
    uv run python experiments/ltx2/router_evaluation.py --quick

    # Full evaluation with trained router
    uv run python experiments/ltx2/router_evaluation.py \
        --router-checkpoint experiments/results/router_training/run_xxx/router_epoch10.pt

    # Compare different router input modes
    uv run python experiments/ltx2/router_evaluation.py \
        --router-checkpoint path/to/router.pt \
        --router-input-mode layer_47
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.ltx2.base import LTX2ExperimentBase
from llm_dit.data import get_all_prompts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class RouterEvaluationExperiment(LTX2ExperimentBase):
    """
    Evaluate router against baseline uniform routing.

    Compares generation quality when using learned per-token layer weights
    vs uniform weights (standard LTX-2 behavior).
    """

    def __init__(
        self,
        router_checkpoint: Optional[str] = None,
        router_input_mode: str = "mean",
        routing_mode: str = "soft",
        temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__("router_evaluation", **kwargs)

        self.router_checkpoint = router_checkpoint
        self.router_input_mode = router_input_mode
        self.routing_mode = routing_mode
        self.temperature = temperature

        # Will be set in setup()
        self.router = None
        self.prompts = None

    def setup(self):
        """Load all components for evaluation."""
        logger.info("Loading encoder...")
        self.load_encoder(use_8bit=True)

        logger.info("Loading pipeline (for generation)...")
        self.load_model(use_pure_pytorch=False, use_group_offloading=True)

        logger.info("Loading router...")
        self.router = self.load_router(
            checkpoint_path=self.router_checkpoint,
            routing_mode=self.routing_mode,
            temperature=self.temperature,
        )

        # Get parameter count for logging
        router_params = sum(p.numel() for p in self.router.parameters())
        logger.info(f"Router loaded with {router_params:,} parameters")

    def run_iteration(self, config: Dict) -> Dict:
        """
        Run single prompt evaluation.

        Generates with both router and baseline, returns comparison metrics.
        """
        prompt = config["prompt"]
        prompt_name = config.get("name", "unnamed")
        seed = config.get("seed", 42)

        logger.info(f"Evaluating prompt: {prompt_name}")

        results = {
            "prompt": prompt,
            "name": prompt_name,
            "seed": seed,
        }

        # === Router Encoding ===
        logger.info("  Encoding with router...")
        router_embeds, router_stats = self.encode_with_router(
            prompt,
            router=self.router,
            router_input_mode=self.router_input_mode,
            return_stats=True,
        )

        results["router_stats"] = router_stats

        # === Baseline Encoding (uniform weights) ===
        logger.info("  Encoding with uniform weights (baseline)...")
        baseline_embeds = self.encode(prompt)

        # === Generate Videos ===
        gen_kwargs = {
            "num_frames": config.get("num_frames", 33),
            "height": config.get("height", 512),
            "width": config.get("width", 768),
            "num_inference_steps": config.get("num_inference_steps", 25),
            "guidance_scale": config.get("guidance_scale", 3.0),
            "seed": seed,
        }

        logger.info("  Generating with router embeddings...")
        router_video = self.generate_video(router_embeds, **gen_kwargs)

        logger.info("  Generating with baseline embeddings...")
        baseline_video = self.generate_video(baseline_embeds, **gen_kwargs)

        # === Score Videos ===
        logger.info("  Scoring videos...")
        self.load_scorer()

        router_score = self.score_video(router_video, prompt)
        baseline_score = self.score_video(baseline_video, prompt)

        results["router_score"] = router_score
        results["baseline_score"] = baseline_score
        results["delta"] = router_score - baseline_score
        results["improvement_pct"] = (
            (router_score - baseline_score) / baseline_score * 100
            if baseline_score > 0 else 0
        )

        logger.info(f"  Router: {router_score:.4f}, Baseline: {baseline_score:.4f}")
        logger.info(f"  Delta: {results['delta']:+.4f} ({results['improvement_pct']:+.1f}%)")

        # === Save Videos ===
        if config.get("save_videos", True):
            self.save_video(
                router_video,
                f"{prompt_name}_router",
                prompt,
                metadata={
                    "routing_stats": router_stats,
                    "score": router_score,
                    "encoding_type": "router",
                },
            )
            self.save_video(
                baseline_video,
                f"{prompt_name}_baseline",
                prompt,
                metadata={
                    "score": baseline_score,
                    "encoding_type": "baseline",
                },
            )

        # Cleanup between iterations
        self.cleanup()

        return results

    def aggregate_results(self, results: List[Dict]) -> Dict:
        """Aggregate results across all prompts."""
        import numpy as np

        # Filter out any error results
        valid_results = [r for r in results if "error" not in r]

        if not valid_results:
            return {"results": results, "summary": {"error": "No valid results"}}

        router_scores = [r["router_score"] for r in valid_results]
        baseline_scores = [r["baseline_score"] for r in valid_results]
        deltas = [r["delta"] for r in valid_results]

        summary = {
            "num_prompts": len(valid_results),
            "router_mean": float(np.mean(router_scores)),
            "router_std": float(np.std(router_scores)),
            "baseline_mean": float(np.mean(baseline_scores)),
            "baseline_std": float(np.std(baseline_scores)),
            "delta_mean": float(np.mean(deltas)),
            "delta_std": float(np.std(deltas)),
            "wins": sum(1 for d in deltas if d > 0),
            "losses": sum(1 for d in deltas if d < 0),
            "ties": sum(1 for d in deltas if d == 0),
            "avg_improvement_pct": float(np.mean([r["improvement_pct"] for r in valid_results])),
        }

        # Routing statistics aggregation
        if valid_results and "router_stats" in valid_results[0]:
            entropies = [r["router_stats"]["entropy"] for r in valid_results]
            sparsities = [r["router_stats"]["sparsity"] for r in valid_results]
            summary["routing"] = {
                "mean_entropy": float(np.mean(entropies)),
                "mean_sparsity": float(np.mean(sparsities)),
            }

        return {"results": results, "summary": summary}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate TokenLayerRouter against baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--router-checkpoint",
        type=str,
        default=None,
        help="Path to trained router checkpoint. If None, uses uniform initialization.",
    )
    parser.add_argument(
        "--router-input-mode",
        type=str,
        default="mean",
        choices=["layer_0", "layer_24", "layer_47", "layer_48", "mean"],
        help="How to extract router input from Gemma layers",
    )
    parser.add_argument(
        "--routing-mode",
        type=str,
        default="soft",
        choices=["soft", "topk", "gumbel"],
        help="Router output mode",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Router softmax temperature",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode (fewer prompts, faster settings)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results/router_evaluation",
        help="Output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # Get prompts
    prompts_dict = get_all_prompts(quick=args.quick)
    if args.quick:
        # Even fewer for quick mode
        prompts_dict = dict(list(prompts_dict.items())[:2])

    # Build configs
    configs = []
    for name, prompt in prompts_dict.items():
        configs.append({
            "name": name,
            "prompt": prompt,
            "seed": args.seed,
            "num_frames": 17 if args.quick else 33,
            "height": 256 if args.quick else 512,
            "width": 384 if args.quick else 768,
            "num_inference_steps": 12 if args.quick else 25,
            "guidance_scale": 3.0,
            "save_videos": True,
        })

    # Run experiment
    logger.info("=" * 60)
    logger.info("LTX-2 Router Evaluation")
    logger.info("=" * 60)
    logger.info(f"Router checkpoint: {args.router_checkpoint or 'None (uniform init)'}")
    logger.info(f"Router input mode: {args.router_input_mode}")
    logger.info(f"Prompts: {len(configs)}")

    experiment = RouterEvaluationExperiment(
        router_checkpoint=args.router_checkpoint,
        router_input_mode=args.router_input_mode,
        routing_mode=args.routing_mode,
        temperature=args.temperature,
        output_dir=args.output_dir,
    )

    results = experiment.run(configs)

    # Print summary
    summary = results["summary"]
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Router Mean Score: {summary['router_mean']:.4f} +/- {summary['router_std']:.4f}")
    logger.info(f"Baseline Mean Score: {summary['baseline_mean']:.4f} +/- {summary['baseline_std']:.4f}")
    logger.info(f"Delta: {summary['delta_mean']:+.4f} +/- {summary['delta_std']:.4f}")
    logger.info(f"Average Improvement: {summary['avg_improvement_pct']:+.2f}%")
    logger.info(f"Wins/Losses/Ties: {summary['wins']}/{summary['losses']}/{summary['ties']}")

    if "routing" in summary:
        logger.info(f"Mean Routing Entropy: {summary['routing']['mean_entropy']:.2f}")
        logger.info(f"Mean Effective Layers: {summary['routing']['mean_sparsity']:.1f}")

    logger.info(f"\nResults saved to: {experiment.run_dir}")


if __name__ == "__main__":
    main()
