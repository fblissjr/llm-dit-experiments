#!/usr/bin/env python3
"""
Experiment runner for Z-Image ablation studies.

Runs systematic experiments varying parameters and saving results with metadata.
Supports TOML config files for device placement, model paths, and other settings.

Usage:
    # Run with config file (recommended)
    uv run experiments/run_ablation.py \\
        --config config.toml \\
        --experiment shift_sweep

    # Run with config and profile
    uv run experiments/run_ablation.py \\
        --config config.toml \\
        --profile rtx4090 \\
        --experiment hidden_layer

    # Run a shift sweep experiment (no config)
    uv run experiments/run_ablation.py \\
        --experiment shift_sweep \\
        --model-path /path/to/z-image \\
        --output-dir results/shift_sweep/

    # Run hidden layer ablation
    uv run experiments/run_ablation.py \\
        --experiment hidden_layer \\
        --model-path /path/to/z-image \\
        --prompt-category animals

    # Run with specific prompts
    uv run experiments/run_ablation.py \\
        --experiment shift_sweep \\
        --model-path /path/to/z-image \\
        --prompt-ids animal_001,simple_002

    # Dry run (show what would be generated)
    uv run experiments/run_ablation.py \\
        --experiment shift_sweep \\
        --dry-run

Config file values are used as defaults; CLI args override them.
"""

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.prompts import (
    get_all_prompt_texts,
    get_prompt_by_id,
    get_prompt_ids,
    get_prompts_by_category,
    load_standard_prompts,
)
from llm_dit.cli import load_runtime_config, RuntimeConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# EXPERIMENT DEFINITIONS
# ============================================================

EXPERIMENTS = {
    "shift_sweep": {
        "description": "Sweep shift parameter to find optimal value",
        "variable": "shift",
        "values": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "defaults": {"steps": 9},
    },
    "shift_steps_grid": {
        "description": "Grid search over shift and steps",
        "variables": ["shift", "steps"],
        "grid": {
            "shift": [2.0, 3.0, 4.0],
            "steps": [6, 9, 12, 15],
        },
    },
    "hidden_layer": {
        "description": "Compare different hidden layer extraction points",
        "variable": "hidden_layer",
        "values": [-1, -2, -3, -4, -5, -6],
        "defaults": {"shift": 3.0, "steps": 9},
    },
    "think_block": {
        "description": "Test think block impact (DiffSynth default: empty think block)",
        "variable": "thinking_content",
        "values": [
            # DiffSynth always uses empty think block - test if content helps or hurts
            "",  # Empty think block (DiffSynth default, model trained with this)
            None,  # No think block (deviates from DiffSynth training)
            "High quality, detailed, photorealistic",  # Quality keywords
            "Soft lighting, warm colors, peaceful atmosphere",  # Mood keywords
            "Sharp focus, crisp details, professional composition",  # Technical keywords
        ],
        "defaults": {"shift": 3.0, "steps": 9},
    },
    "system_prompt": {
        "description": "Test impact of system prompts",
        "variable": "system_prompt",
        "values": [
            None,  # No system prompt
            "You are a professional photographer.",
            "You are an artistic painter.",
            "You are a technical illustrator.",
            "Generate high quality images.",
        ],
        "defaults": {"shift": 3.0, "steps": 9},
    },
    "steps_only": {
        "description": "Test different step counts",
        "variable": "steps",
        "values": [4, 6, 8, 9, 10, 12, 15, 20],
        "defaults": {"shift": 3.0},
    },
    "long_prompt_mode": {
        "description": "Compare long prompt compression modes (only affects prompts >1504 tokens)",
        "variable": "long_prompt_mode",
        "values": ["truncate", "interpolate", "pool", "attention_pool"],
        "defaults": {"shift": 3.0, "steps": 9},
    },
    "hidden_layer_blend": {
        "description": "Blend embeddings from multiple hidden layers",
        "variable": "layer_weights",
        "values": [
            # Baseline: single layers
            {-2: 1.0},  # Default (penultimate only)
            {-1: 1.0},  # Last layer only
            {-5: 1.0},  # Deeper layer only
            # Two-layer blends: semantic + structural
            {-2: 0.7, -5: 0.3},  # 70% semantic, 30% structural
            {-2: 0.5, -5: 0.5},  # Equal blend
            {-2: 0.3, -5: 0.7},  # 30% semantic, 70% structural
            # Three-layer blends
            {-1: 0.33, -2: 0.34, -3: 0.33},  # Equal top-3
            {-2: 0.5, -4: 0.25, -6: 0.25},  # Weighted across depth
        ],
        "defaults": {"shift": 3.0, "steps": 9},
    },
}


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""

    experiment_name: str
    prompt_id: str
    prompt_text: str
    seed: int
    variable_name: str
    variable_value: Any
    # Generation params
    shift: float = 3.0
    steps: int = 9
    hidden_layer: int = -2
    layer_weights: dict[int, float] | None = None  # For layer blending experiments
    long_prompt_mode: str = "interpolate"
    width: int = 1024
    height: int = 1024
    guidance_scale: float = 0.0
    # Optional params
    system_prompt: str | None = None
    thinking_content: str | None = None
    force_think_block: bool = True  # DiffSynth always uses empty think block
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ExperimentResult:
    """Result of a single experiment run."""

    config: ExperimentConfig
    output_path: str
    generation_time_seconds: float
    token_count: int | None = None
    image_reward: float | None = None  # Human-preference aligned
    siglip_score: float | None = None  # Image-text alignment
    error: str | None = None


class ExperimentRunner:
    """Runs ablation experiments with the Z-Image pipeline."""

    def __init__(
        self,
        model_path: str,
        output_dir: Path,
        text_encoder_device: str = "cpu",
        dit_device: str = "cuda",
        vae_device: str = "cuda",
        dry_run: bool = False,
        compute_metrics: bool = False,
    ):
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.text_encoder_device = text_encoder_device
        self.dit_device = dit_device
        self.vae_device = vae_device
        self.dry_run = dry_run
        self.compute_metrics = compute_metrics
        self.pipeline = None
        self._image_reward_scorer = None
        self._siglip_scorer = None

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)

    def load_pipeline(self):
        """Load the Z-Image pipeline."""
        if self.dry_run:
            logger.info("[DRY RUN] Would load pipeline from %s", self.model_path)
            return

        if self.pipeline is not None:
            return

        logger.info("Loading pipeline from %s", self.model_path)

        from llm_dit import ZImagePipeline

        self.pipeline = ZImagePipeline.from_pretrained(
            self.model_path,
            encoder_device=self.text_encoder_device,
            dit_device=self.dit_device,
            vae_device=self.vae_device,
            torch_dtype=torch.bfloat16,
        )
        logger.info("Pipeline loaded successfully")

    def run_single(self, config: ExperimentConfig) -> ExperimentResult:
        """Run a single experiment configuration."""
        # Build output filename
        filename = (
            f"{config.experiment_name}_"
            f"{config.prompt_id}_"
            f"{config.variable_name}_{config.variable_value}_"
            f"seed{config.seed}.png"
        )
        output_path = self.output_dir / "images" / filename

        if self.dry_run:
            logger.info(
                "[DRY RUN] Would generate: %s=%s, prompt=%s",
                config.variable_name,
                config.variable_value,
                config.prompt_text[:50] + "...",
            )
            return ExperimentResult(
                config=config,
                output_path=str(output_path),
                generation_time_seconds=0.0,
                token_count=None,
            )

        self.load_pipeline()

        logger.info(
            "Generating: %s=%s, seed=%d, prompt=%s",
            config.variable_name,
            config.variable_value,
            config.seed,
            config.prompt_text[:50] + "...",
        )

        start_time = time.time()
        try:
            # Prepare generation kwargs
            gen_kwargs = {
                "prompt": config.prompt_text,
                "width": config.width,
                "height": config.height,
                "num_inference_steps": config.steps,
                "guidance_scale": config.guidance_scale,
                "shift": config.shift,
                "generator": torch.Generator().manual_seed(config.seed),
            }

            # Add optional params
            if config.system_prompt:
                gen_kwargs["system_prompt"] = config.system_prompt

            # Handle thinking content and force_think_block
            if config.thinking_content is not None:
                if config.thinking_content == "":
                    # Empty string means force empty think block
                    gen_kwargs["force_think_block"] = True
                else:
                    # Non-empty thinking content
                    gen_kwargs["thinking_content"] = config.thinking_content
            else:
                # No thinking_content provided, use config default (True for DiffSynth compat)
                gen_kwargs["force_think_block"] = config.force_think_block

            # Add hidden_layer and long_prompt_mode
            gen_kwargs["hidden_layer"] = config.hidden_layer
            gen_kwargs["long_prompt_mode"] = config.long_prompt_mode

            # Add layer_weights for blending experiments
            if config.layer_weights is not None:
                gen_kwargs["layer_weights"] = config.layer_weights

            # Generate
            result = self.pipeline(**gen_kwargs)

            # Save image - pipeline returns PIL Image directly, not diffusers-style result
            if hasattr(result, "images"):
                # Diffusers-style result object
                image = result.images[0]
            elif hasattr(result, "save"):
                # Direct PIL Image
                image = result
            elif isinstance(result, list):
                # List of images
                image = result[0]
            else:
                raise ValueError(f"Unexpected result type: {type(result)}")

            image.save(output_path)

            generation_time = time.time() - start_time

            # Get token count if available
            token_count = None
            if hasattr(result, "token_count"):
                token_count = result.token_count

            # Compute metrics if requested
            image_reward_score = None
            siglip_score = None
            if self.compute_metrics:
                image_reward_score, siglip_score = self._compute_metrics(
                    config.prompt_text, output_path
                )

            return ExperimentResult(
                config=config,
                output_path=str(output_path),
                generation_time_seconds=generation_time,
                token_count=token_count,
                image_reward=image_reward_score,
                siglip_score=siglip_score,
            )

        except Exception as e:
            logger.error("Generation failed: %s", e)
            return ExperimentResult(
                config=config,
                output_path=str(output_path),
                generation_time_seconds=time.time() - start_time,
                error=str(e),
            )

    def _compute_metrics(
        self, prompt: str, image_path: Path
    ) -> tuple[float | None, float | None]:
        """Compute ImageReward and SigLIP2 scores for an image."""
        image_reward_score = None
        siglip_score = None

        # ImageReward (human preference)
        try:
            if self._image_reward_scorer is None:
                from experiments.metrics import ImageRewardScorer
                self._image_reward_scorer = ImageRewardScorer()
            image_reward_score = self._image_reward_scorer.score(prompt, image_path)
            logger.debug("ImageReward: %.4f", image_reward_score)
        except ImportError:
            logger.warning("ImageReward not installed, skipping. Install: uv add image-reward")
        except Exception as e:
            logger.warning("ImageReward computation failed: %s", e)

        # SigLIP2 (image-text alignment)
        try:
            if self._siglip_scorer is None:
                from experiments.metrics import SigLIPScorer
                self._siglip_scorer = SigLIPScorer()
            siglip_score = self._siglip_scorer.score(prompt, image_path)
            logger.debug("SigLIP Score: %.4f", siglip_score)
        except ImportError:
            logger.warning("transformers not installed, skipping SigLIP. Install: uv add transformers")
        except Exception as e:
            logger.warning("SigLIP computation failed: %s", e)

        return image_reward_score, siglip_score

    def run_experiment(
        self,
        experiment_name: str,
        prompt_ids: list[str] | None = None,
        prompt_category: str | None = None,
        seeds: list[int] | None = None,
        max_prompts: int | None = None,
    ) -> list[ExperimentResult]:
        """Run a full experiment across prompts and parameter values."""
        if experiment_name not in EXPERIMENTS:
            raise ValueError(f"Unknown experiment: {experiment_name}")

        exp_def = EXPERIMENTS[experiment_name]
        logger.info("Running experiment: %s", exp_def["description"])

        # Get prompts to use
        if prompt_ids:
            prompts = [get_prompt_by_id(pid) for pid in prompt_ids]
            prompts = [p for p in prompts if p is not None]
        elif prompt_category:
            prompts = get_prompts_by_category(prompt_category)
        else:
            prompts = load_standard_prompts()["prompts"]

        if max_prompts:
            prompts = prompts[:max_prompts]

        # Get seeds
        if seeds is None:
            seeds = [42]  # Default to single seed

        # Get parameter values
        if "grid" in exp_def:
            # Grid search over multiple variables
            configs = self._build_grid_configs(
                experiment_name, exp_def, prompts, seeds
            )
        else:
            # Single variable sweep
            configs = self._build_sweep_configs(
                experiment_name, exp_def, prompts, seeds
            )

        logger.info(
            "Running %d configurations (%d prompts x %d values x %d seeds)",
            len(configs),
            len(prompts),
            len(exp_def.get("values", [])) or len(list(self._grid_combinations(exp_def.get("grid", {})))),
            len(seeds),
        )

        # Run all configurations
        results = []
        for i, config in enumerate(configs):
            logger.info("Progress: %d/%d", i + 1, len(configs))
            result = self.run_single(config)
            results.append(result)

            # Save intermediate results
            self._save_result(result)

        # Save summary
        self._save_summary(experiment_name, results)

        return results

    def _build_sweep_configs(
        self,
        experiment_name: str,
        exp_def: dict,
        prompts: list[dict],
        seeds: list[int],
    ) -> list[ExperimentConfig]:
        """Build configs for single-variable sweep."""
        configs = []
        variable = exp_def["variable"]
        values = exp_def["values"]
        defaults = exp_def.get("defaults", {})

        for prompt in prompts:
            for seed in seeds:
                for value in values:
                    config = ExperimentConfig(
                        experiment_name=experiment_name,
                        prompt_id=prompt["id"],
                        prompt_text=prompt["prompt"],
                        seed=seed,
                        variable_name=variable,
                        variable_value=value,
                        **defaults,
                    )
                    # Set the variable value
                    if variable == "shift":
                        config.shift = value
                    elif variable == "steps":
                        config.steps = value
                    elif variable == "hidden_layer":
                        config.hidden_layer = value
                    elif variable == "thinking_content":
                        config.thinking_content = value
                        if value == "":
                            config.force_think_block = True
                    elif variable == "system_prompt":
                        config.system_prompt = value
                    elif variable == "long_prompt_mode":
                        config.long_prompt_mode = value
                    elif variable == "layer_weights":
                        config.layer_weights = value

                    configs.append(config)

        return configs

    def _build_grid_configs(
        self,
        experiment_name: str,
        exp_def: dict,
        prompts: list[dict],
        seeds: list[int],
    ) -> list[ExperimentConfig]:
        """Build configs for grid search."""
        configs = []
        grid = exp_def["grid"]

        for prompt in prompts:
            for seed in seeds:
                for combo in self._grid_combinations(grid):
                    var_str = "_".join(f"{k}{v}" for k, v in combo.items())
                    config = ExperimentConfig(
                        experiment_name=experiment_name,
                        prompt_id=prompt["id"],
                        prompt_text=prompt["prompt"],
                        seed=seed,
                        variable_name="grid",
                        variable_value=var_str,
                    )
                    # Apply grid values
                    for key, val in combo.items():
                        setattr(config, key, val)

                    configs.append(config)

        return configs

    def _grid_combinations(self, grid: dict) -> list[dict]:
        """Generate all combinations from a grid."""
        import itertools

        keys = list(grid.keys())
        values = [grid[k] for k in keys]
        for combo in itertools.product(*values):
            yield dict(zip(keys, combo))

    def _save_result(self, result: ExperimentResult):
        """Save individual result metadata."""
        metadata_path = (
            self.output_dir
            / "metadata"
            / f"{Path(result.output_path).stem}.json"
        )
        with open(metadata_path, "w") as f:
            json.dump(
                {
                    "config": asdict(result.config),
                    "output_path": result.output_path,
                    "generation_time_seconds": result.generation_time_seconds,
                    "token_count": result.token_count,
                    "image_reward": result.image_reward,
                    "siglip_score": result.siglip_score,
                    "error": result.error,
                },
                f,
                indent=2,
            )

    def _save_summary(self, experiment_name: str, results: list[ExperimentResult]):
        """Save experiment summary as CSV and JSON."""
        # CSV summary
        csv_path = self.output_dir / f"{experiment_name}_summary.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "prompt_id",
                "seed",
                "variable_name",
                "variable_value",
                "generation_time_seconds",
                "token_count",
                "image_reward",
                "siglip_score",
                "error",
                "output_path",
            ])
            for r in results:
                writer.writerow([
                    r.config.prompt_id,
                    r.config.seed,
                    r.config.variable_name,
                    r.config.variable_value,
                    r.generation_time_seconds,
                    r.token_count,
                    r.image_reward,
                    r.siglip_score,
                    r.error,
                    r.output_path,
                ])

        # Compute metric statistics
        ir_scores = [r.image_reward for r in results if r.image_reward is not None]
        siglip_scores = [r.siglip_score for r in results if r.siglip_score is not None]

        # JSON summary
        json_path = self.output_dir / f"{experiment_name}_summary.json"
        with open(json_path, "w") as f:
            summary = {
                "experiment": experiment_name,
                "timestamp": datetime.now().isoformat(),
                "total_runs": len(results),
                "successful_runs": len([r for r in results if r.error is None]),
                "failed_runs": len([r for r in results if r.error is not None]),
                "total_time_seconds": sum(r.generation_time_seconds for r in results),
            }
            # Add metric stats if available
            if ir_scores:
                summary["image_reward"] = {
                    "mean": sum(ir_scores) / len(ir_scores),
                    "min": min(ir_scores),
                    "max": max(ir_scores),
                    "count": len(ir_scores),
                }
            if siglip_scores:
                summary["siglip_score"] = {
                    "mean": sum(siglip_scores) / len(siglip_scores),
                    "min": min(siglip_scores),
                    "max": max(siglip_scores),
                    "count": len(siglip_scores),
                }
            json.dump(summary, f, indent=2)

        logger.info("Summary saved to %s", csv_path)


def main():
    parser = argparse.ArgumentParser(
        description="Run Z-Image ablation experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available experiments:
  shift_sweep       Sweep shift parameter (1.0-6.0)
  shift_steps_grid  Grid search over shift and steps
  hidden_layer      Compare hidden layer extraction points (-1 to -6)
  think_block       Test impact of think block content
  system_prompt     Test impact of system prompts
  steps_only        Test different step counts

Examples:
  # Run with config file
  uv run experiments/run_ablation.py --config config.toml --experiment shift_sweep

  # Run shift sweep on animal prompts
  uv run experiments/run_ablation.py --experiment shift_sweep --prompt-category animals

  # Run hidden layer ablation with specific prompts
  uv run experiments/run_ablation.py --experiment hidden_layer --prompt-ids animal_001,simple_002

  # Dry run to see what would be generated
  uv run experiments/run_ablation.py --experiment shift_sweep --dry-run
        """,
    )

    # Config file support
    config_group = parser.add_argument_group("Configuration")
    config_group.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to TOML config file (CLI args override config values)",
    )
    config_group.add_argument(
        "--profile",
        type=str,
        default="default",
        help="Config profile to use (default: default)",
    )

    parser.add_argument(
        "--experiment",
        choices=list(EXPERIMENTS.keys()),
        help="Experiment to run (required unless --list-experiments or --list-prompts)",
    )
    parser.add_argument(
        "--model-path",
        help="Path to Z-Image model (required unless --dry-run or in config)",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--prompt-ids",
        help="Comma-separated list of prompt IDs to use",
    )
    parser.add_argument(
        "--prompt-category",
        choices=[
            "simple_objects",
            "animals",
            "humans",
            "scenes",
            "landscapes",
            "artistic_styles",
            "lighting",
            "abstract",
            "technical",
            "text_rendering",
        ],
        help="Use prompts from specific category",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        help="Maximum number of prompts to use",
    )
    parser.add_argument(
        "--seeds",
        default="42",
        help="Comma-separated list of seeds (default: 42)",
    )
    parser.add_argument(
        "--text-encoder-device",
        default=None,
        help="Device for text encoder (default: from config or cpu)",
    )
    parser.add_argument(
        "--dit-device",
        default=None,
        help="Device for DiT (default: from config or cuda)",
    )
    parser.add_argument(
        "--vae-device",
        default=None,
        help="Device for VAE (default: from config or cuda)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without running",
    )
    parser.add_argument(
        "--list-experiments",
        action="store_true",
        help="List available experiments and exit",
    )
    parser.add_argument(
        "--list-prompts",
        action="store_true",
        help="List available prompts and exit",
    )
    parser.add_argument(
        "--compute-metrics",
        action="store_true",
        help="Compute ImageReward and SigLIP2 scores for generated images",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Debug logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # List experiments
    if args.list_experiments:
        print("\nAvailable experiments:\n")
        for name, exp in EXPERIMENTS.items():
            print(f"  {name}")
            print(f"    {exp['description']}")
            if "variable" in exp:
                print(f"    Variable: {exp['variable']}")
                print(f"    Values: {exp['values']}")
            elif "grid" in exp:
                print(f"    Grid: {exp['grid']}")
            print()
        return

    # List prompts
    if args.list_prompts:
        prompts = load_standard_prompts()
        print(f"\nStandard prompts ({prompts['metadata']['total_prompts']} total):\n")
        for category, count in prompts["metadata"]["categories"].items():
            print(f"  {category}: {count} prompts")
            for p in get_prompts_by_category(category)[:3]:
                print(f"    - {p['id']}: {p['prompt'][:50]}...")
            print()
        return

    # Require --experiment for actual runs
    if not args.experiment:
        parser.error("--experiment is required (use --list-experiments to see options)")

    # Load config if provided
    config: RuntimeConfig | None = None
    if args.config:
        config = load_runtime_config(args)
        logger.info(f"Loaded config from {args.config} (profile: {args.profile})")

    # Resolve model path (CLI > config > error)
    model_path = args.model_path
    if not model_path and config:
        model_path = config.model_path
    if not args.dry_run and not model_path:
        parser.error("--model-path is required unless using --dry-run or --config")

    # Resolve device settings (CLI > config > defaults)
    text_encoder_device = args.text_encoder_device
    if text_encoder_device is None:
        text_encoder_device = config.encoder_device if config else "cpu"

    dit_device = args.dit_device
    if dit_device is None:
        dit_device = config.dit_device if config else "cuda"

    vae_device = args.vae_device
    if vae_device is None:
        vae_device = config.vae_device if config else "cuda"

    # Resolve "auto" device settings
    def resolve_device(device: str) -> str:
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device

    text_encoder_device = resolve_device(text_encoder_device)
    dit_device = resolve_device(dit_device)
    vae_device = resolve_device(vae_device)

    # Parse prompt IDs
    prompt_ids = None
    if args.prompt_ids:
        prompt_ids = [p.strip() for p in args.prompt_ids.split(",")]

    # Parse seeds
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    # Create output dir with experiment name
    output_dir = Path(args.output_dir) / args.experiment

    logger.info(f"Model: {model_path}")
    logger.info(f"Devices: encoder={text_encoder_device}, dit={dit_device}, vae={vae_device}")

    # Run experiment
    runner = ExperimentRunner(
        model_path=model_path or "",
        output_dir=output_dir,
        text_encoder_device=text_encoder_device,
        dit_device=dit_device,
        vae_device=vae_device,
        dry_run=args.dry_run,
        compute_metrics=args.compute_metrics,
    )

    results = runner.run_experiment(
        experiment_name=args.experiment,
        prompt_ids=prompt_ids,
        prompt_category=args.prompt_category,
        seeds=seeds,
        max_prompts=args.max_prompts,
    )

    # Summary
    successful = len([r for r in results if r.error is None])
    failed = len([r for r in results if r.error is not None])
    print(f"\nExperiment complete: {successful} successful, {failed} failed")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
