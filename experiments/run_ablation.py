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
from llm_dit.cli import RuntimeConfig, load_runtime_config

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
        "description": "Blend embeddings from multiple hidden layers (late layer focus)",
        "variable": "layer_weights",
        "values": [
            # Baseline: single layers
            {-2: 1.0},   # Layer 35 - Default (penultimate, what DiT was trained on)
            {-1: 1.0},   # Layer 36 - Last layer (most abstracted)
            {-5: 1.0},   # Layer 32 - Slightly deeper
            # Two-layer blends within late region
            {-2: 0.7, -5: 0.3},  # Layers 35+32: mostly default
            {-2: 0.5, -5: 0.5},  # Layers 35+32: equal blend
            {-2: 0.3, -5: 0.7},  # Layers 35+32: favor deeper
            # Three-layer blends
            {-1: 0.33, -2: 0.34, -3: 0.33},  # Layers 36+35+34: top-3 equal
            {-2: 0.5, -5: 0.25, -8: 0.25},   # Layers 35+32+29: weighted spread
        ],
        "defaults": {"shift": 3.0, "steps": 9},
    },
    # =========================================================================
    # DEEP LAYER EXPERIMENTS
    # Qwen3-4B has 36 transformer layers. Layer indexing:
    #   -1 = Layer 36 (last, just before LM head - most abstracted for generation)
    #   -2 = Layer 35 (Z-Image default, penultimate)
    #   -19 = Layer 18 (exact middle of 36 layers)
    #   -36 = Layer 1 (first transformer layer - raw, heavy pre-training bias)
    #
    # Observed behavior:
    #   Early (1-10): Heavy pre-training bias (cultural associations dominate)
    #   Middle (12-24): Semantic sweet spot (past bias, before SFT abstraction)
    #   Late (25-35): SFT-modified for "helpful assistant" patterns
    #   Final (36): Ready for token prediction, loses visual specifics
    # =========================================================================
    "hidden_layer_deep": {
        "description": "Deep sweep across all 36 layers to find prompt-adherence sweet spot",
        "variable": "hidden_layer",
        "values": [
            -1,   # Layer 36 (last) - most abstracted, ready for generation
            -2,   # Layer 35 (Z-Image default)
            -5,   # Layer 32
            -9,   # Layer 28
            -12,  # Layer 25 - entering late region
            -15,  # Layer 22
            -19,  # Layer 18 (exact middle)
            -22,  # Layer 15
            -25,  # Layer 12 - entering middle region
            -28,  # Layer 9
            -31,  # Layer 6 - early region
            -34,  # Layer 3
            -36,  # Layer 1 (earliest) - raw, biased
        ],
        "defaults": {"shift": 3.0, "steps": 9},
    },
    "hidden_layer_middle_focus": {
        "description": "Fine-grained sweep of middle layers (12-24) where prompt adherence peaks",
        "variable": "hidden_layer",
        "values": [
            -13,  # Layer 24
            -15,  # Layer 22
            -17,  # Layer 20
            -19,  # Layer 18 (exact middle)
            -21,  # Layer 16
            -23,  # Layer 14
            -25,  # Layer 12
        ],
        "defaults": {"shift": 3.0, "steps": 9},
    },
    "middle_layer_blend": {
        "description": "Blend middle layers together (where prompt adherence is highest)",
        "variable": "layer_weights",
        "values": [
            # Single middle layers for baseline
            {-19: 1.0},  # Layer 18 (exact middle)
            {-16: 1.0},  # Layer 21
            {-22: 1.0},  # Layer 15
            # Middle layer blends
            {-16: 0.5, -22: 0.5},  # Blend around middle
            {-13: 0.33, -19: 0.34, -25: 0.33},  # Wide middle blend (layers 24, 18, 12)
            {-17: 0.5, -21: 0.5},  # Tight middle blend (layers 20, 16)
            # Middle + late blend (best of both worlds?)
            {-19: 0.7, -2: 0.3},  # 70% middle semantic, 30% late (DiT-trained)
            {-19: 0.5, -2: 0.5},  # Equal middle/late
            {-16: 0.4, -19: 0.3, -2: 0.3},  # Three-way blend
        ],
        "defaults": {"shift": 3.0, "steps": 9},
    },
    # =========================================================================
    # NEW EXPERIMENTS: Exploring layer characteristics
    # =========================================================================
    "early_layer_bias": {
        "description": "Test early layers with culturally-charged prompts to study pre-training bias",
        "variable": "hidden_layer",
        "values": [
            -36,  # Layer 1 (earliest)
            -34,  # Layer 3
            -31,  # Layer 6
            -28,  # Layer 9
            -25,  # Layer 12 (entering middle)
            -19,  # Layer 18 (middle baseline)
            -2,   # Layer 35 (default baseline)
        ],
        "defaults": {"shift": 3.0, "steps": 9},
    },
    "layer_progression": {
        "description": "Full layer progression to visualize embedding evolution (every 3rd layer)",
        "variable": "hidden_layer",
        "values": [
            -36,  # Layer 1
            -33,  # Layer 4
            -30,  # Layer 7
            -27,  # Layer 10
            -24,  # Layer 13
            -21,  # Layer 16
            -18,  # Layer 19
            -15,  # Layer 22
            -12,  # Layer 25
            -9,   # Layer 28
            -6,   # Layer 31
            -3,   # Layer 34
            -1,   # Layer 36
        ],
        "defaults": {"shift": 3.0, "steps": 9},
    },
    "skip_late_blend": {
        "description": "Blend early semantic + middle, skipping SFT-heavy late layers entirely",
        "variable": "layer_weights",
        "values": [
            # Baselines
            {-19: 1.0},  # Middle only (layer 18)
            {-2: 1.0},   # Late only (default, for comparison)
            # Skip late layers - blend early-middle only
            {-25: 0.5, -19: 0.5},  # Early-middle blend (layers 12, 18)
            {-28: 0.3, -22: 0.4, -16: 0.3},  # Wider early-middle (layers 9, 15, 21)
            {-31: 0.2, -25: 0.3, -19: 0.3, -13: 0.2},  # Very wide (layers 6, 12, 18, 24)
            # Compare to including late
            {-25: 0.4, -19: 0.4, -2: 0.2},  # Early-middle with small late contribution
        ],
        "defaults": {"shift": 3.0, "steps": 9},
    },
    "prompt_complexity_layers": {
        "description": "Test if optimal layer depends on prompt complexity (simple vs detailed)",
        "variable": "hidden_layer",
        "values": [
            -2,   # Default (late)
            -10,  # Late-ish
            -19,  # Middle
            -28,  # Early-ish
        ],
        "defaults": {"shift": 3.0, "steps": 9},
        # Note: Run this with both simple_objects AND artistic_styles categories
        # to compare optimal layers across prompt complexity
    },
    "dit_training_comparison": {
        "description": "Compare DiT-trained layer (-2) vs empirically-best middle layers",
        "variable": "layer_weights",
        "values": [
            # DiT was trained on -2 embeddings - does it expect that distribution?
            {-2: 1.0},   # What DiT was trained on
            {-19: 1.0},  # Empirically better for prompt adherence?
            # Blend to get both: DiT familiarity + better semantics
            {-19: 0.8, -2: 0.2},  # Mostly middle, hint of trained distribution
            {-19: 0.6, -2: 0.4},  # More balanced
            {-19: 0.5, -2: 0.5},  # Equal
            {-19: 0.4, -2: 0.6},  # Lean toward trained distribution
            {-19: 0.2, -2: 0.8},  # Mostly trained, hint of middle semantics
        ],
        "defaults": {"shift": 3.0, "steps": 9},
    },
    # =========================================================================
    # VISION CONDITIONING EXPERIMENTS (Qwen3-VL)
    # =========================================================================
    "vl_token_selection": {
        "description": "Compare VL token selection modes (which tokens to use from VL sequence)",
        "variable": "vl_token_selection",
        "values": [
            "all",  # Full sequence (image + text markers + text)
            "image_only",  # Image tokens including markers
            "image_only_no_markers",  # Image tokens without special markers
            "text_only",  # Text tokens only (no image)
        ],
        "defaults": {"shift": 3.0, "steps": 9, "vl_alpha": 1.0},  # Pure VL for comparison
    },
    "vl_pure": {
        "description": "Pure VL conditioning (alpha=1.0) with different token selections",
        "variable": "vl_token_selection",
        "values": ["all", "image_only_no_markers"],  # Most promising modes
        "defaults": {"shift": 3.0, "steps": 9, "vl_alpha": 1.0},
    },
    "vl_alpha_sweep": {
        "description": "Sweep alpha blend ratio (VL vs text influence)",
        "variable": "vl_alpha",
        "values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "defaults": {"shift": 3.0, "steps": 9, "vl_token_selection": "all"},
    },
    "vl_alpha_coarse": {
        "description": "Coarse alpha sweep for quick exploration",
        "variable": "vl_alpha",
        "values": [0.0, 0.25, 0.5, 0.75, 1.0],
        "defaults": {"shift": 3.0, "steps": 9, "vl_token_selection": "all"},
    },
    "vl_hidden_layer": {
        "description": "Test different hidden layers for VL extraction",
        "variable": "vl_hidden_layer",
        "values": [-1, -2, -5, -10, -15, -19],  # Sample across VL model depth
        "defaults": {"shift": 3.0, "steps": 9, "vl_alpha": 1.0, "vl_token_selection": "all"},
    },
    "vl_blend_with_text_layers": {
        "description": "VL conditioning (alpha=0.5) with text layer sweep",
        "variable": "hidden_layer",
        "values": [-1, -2, -5, -10, -15, -19],
        "defaults": {"shift": 3.0, "steps": 9, "vl_alpha": 0.5, "vl_token_selection": "all"},
    },
    # =========================================================================
    # HIDDEN LAYER vs CFG INTERACTION EXPERIMENTS
    # =========================================================================
    # Z-Image was distilled with Decoupled-DMD which "bakes in" CFG.
    # The distillation was done with layer -2 embeddings.
    # When using OOD layers (e.g., -10 to -18), the baked-in CFG might not
    # extrapolate correctly. Hypothesis: small CFG (1.5-2.5) might help
    # compensate for distribution mismatch with middle layers.
    # =========================================================================
    "hidden_layer_cfg_grid": {
        "description": "Grid search: hidden layer vs CFG to test if CFG helps OOD layers",
        "variables": ["hidden_layer", "guidance_scale"],
        "grid": {
            "hidden_layer": [-2, -10, -14, -18, -22],  # Default + middle range
            "guidance_scale": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        },
    },
    "hidden_layer_cfg_quick": {
        "description": "Quick grid: layer vs CFG with fewer values",
        "variables": ["hidden_layer", "guidance_scale"],
        "grid": {
            "hidden_layer": [-2, -14, -18],  # Default, middle, deep-middle
            "guidance_scale": [0.0, 1.5, 2.5],
        },
    },
    "hidden_layer_cfg_shift_grid": {
        "description": "Full grid: layer vs CFG vs shift for comprehensive analysis",
        "variables": ["hidden_layer", "guidance_scale", "shift"],
        "grid": {
            "hidden_layer": [-2, -14, -18],
            "guidance_scale": [0.0, 1.5, 2.5],
            "shift": [3.0, 5.0, 7.0],
        },
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
    # VL conditioning params
    vl_image_path: str | None = None  # Reference image for VL conditioning
    vl_alpha: float = 0.0  # Blend ratio (0.0 = pure text, 1.0 = pure VL)
    vl_token_selection: str = "all"  # "all", "image_only", "image_only_no_markers", "text_only"
    vl_hidden_layer: int = -2  # Hidden layer to extract from VL model
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
        vl_model_path: str | None = None,
        vl_image_path: str | None = None,
    ):
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.text_encoder_device = text_encoder_device
        self.dit_device = dit_device
        self.vae_device = vae_device
        self.dry_run = dry_run
        self.compute_metrics = compute_metrics
        self.pipeline = None

        # VL conditioning support
        self.vl_model_path = vl_model_path
        self.vl_image_path = vl_image_path
        self.vl_extractor = None
        self.vl_image = None

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

    def load_vl_extractor(self):
        """Load Qwen3-VL model for vision conditioning."""
        if self.dry_run:
            logger.info("[DRY RUN] Would load VL extractor from %s", self.vl_model_path)
            return

        if self.vl_extractor is not None:
            return

        if not self.vl_model_path:
            # Try to auto-detect
            from llm_dit.vl.qwen3_vl import VLEmbeddingExtractor

            vl_path = VLEmbeddingExtractor.find_model_path()
            if not vl_path:
                raise ValueError(
                    "VL model path not provided and auto-detection failed. "
                    "Use --vl-model-path to specify Qwen3-VL model location."
                )
            self.vl_model_path = vl_path

        logger.info("Loading VL extractor from %s", self.vl_model_path)

        from llm_dit.vl.qwen3_vl import VLEmbeddingExtractor

        # Use same device as text encoder for consistency
        device = self.text_encoder_device if self.text_encoder_device != "auto" else "cuda"
        self.vl_extractor = VLEmbeddingExtractor.from_pretrained(
            self.vl_model_path,
            device=device,
            torch_dtype=torch.bfloat16,
        )
        logger.info("VL extractor loaded successfully")

    def load_vl_image(self):
        """Load reference image for VL conditioning."""
        if self.dry_run:
            logger.info("[DRY RUN] Would load VL image from %s", self.vl_image_path)
            return

        if self.vl_image is not None:
            return

        if not self.vl_image_path:
            raise ValueError("VL image path not provided. Use --vl-image to specify reference image.")

        from PIL import Image

        logger.info("Loading VL reference image from %s", self.vl_image_path)
        self.vl_image = Image.open(self.vl_image_path).convert("RGB")
        logger.info("VL image loaded: %s", self.vl_image.size)

    def unload_vl_extractor(self):
        """Unload VL extractor to free VRAM before loading pipeline."""
        if self.vl_extractor is not None:
            logger.info("Unloading VL extractor to free VRAM...")
            self.vl_extractor.unload()
            self.vl_extractor = None

            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                allocated = torch.cuda.memory_allocated() / 1e9
                logger.info(f"GPU memory after VL unload: {allocated:.2f} GB")

    def _extract_vl_embeddings(self, config: ExperimentConfig) -> torch.Tensor:
        """Extract VL embeddings for conditioning."""
        # Load VL components if needed
        self.load_vl_extractor()
        self.load_vl_image()

        # Use config's vl_image_path if provided, otherwise use runner's default
        image_path = config.vl_image_path or self.vl_image_path
        if image_path and image_path != self.vl_image_path:
            # Config specifies a different image - load it
            from PIL import Image

            image = Image.open(image_path).convert("RGB")
        else:
            image = self.vl_image

        # Extract embeddings based on token selection mode
        kwargs = {
            "image": image,
            "text": config.prompt_text,  # Include text in VL input
            "hidden_layer": config.vl_hidden_layer,
            "scale_to_text": True,  # Scale to match text statistics
        }

        # Map token selection mode to extractor parameters
        if config.vl_token_selection == "image_only":
            kwargs["image_tokens_only"] = True
        elif config.vl_token_selection == "image_only_no_markers":
            kwargs["image_tokens_no_markers"] = True
        elif config.vl_token_selection == "text_only":
            kwargs["text_tokens_only"] = True
        # else: "all" - use all tokens (default)

        result = self.vl_extractor.extract(**kwargs)

        logger.debug(
            f"VL extraction: {result.num_tokens} tokens, "
            f"layer={result.hidden_layer}, "
            f"selection={result.token_selection}, "
            f"std={result.scaled_std:.2f}"
        )

        return result.embeddings

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

        # Handle VL conditioning if needed
        vl_embeddings = None
        if config.vl_alpha > 0.0 or config.vl_image_path:
            vl_embeddings = self._extract_vl_embeddings(config)

        self.load_pipeline()

        logger.info(
            "Generating: %s=%s, seed=%d, prompt=%s%s",
            config.variable_name,
            config.variable_value,
            config.seed,
            config.prompt_text[:50] + "...",
            f" (VL alpha={config.vl_alpha})" if config.vl_alpha > 0.0 else "",
        )

        start_time = time.time()
        try:
            # Prepare generation kwargs
            gen_kwargs = {
                "width": config.width,
                "height": config.height,
                "num_inference_steps": config.steps,
                "guidance_scale": config.guidance_scale,
                "shift": config.shift,
                "generator": torch.Generator().manual_seed(config.seed),
            }

            # If we have VL embeddings, blend with text or use pure VL
            if vl_embeddings is not None:
                if config.vl_alpha == 1.0:
                    # Pure VL conditioning - skip text encoding
                    gen_kwargs["prompt_embeds"] = vl_embeddings
                    logger.debug("Using pure VL embeddings (alpha=1.0)")
                else:
                    # Blend VL with text embeddings
                    # Determine force_think_block setting
                    if config.thinking_content is not None:
                        if config.thinking_content == "":
                            force_think = True
                            thinking = None
                        else:
                            force_think = False
                            thinking = config.thinking_content
                    else:
                        force_think = config.force_think_block
                        thinking = None

                    # Get text embeddings from encoder
                    if config.layer_weights is not None:
                        # Use blended encoding
                        text_result = self.pipeline.encoder.encode_blended(
                            prompt=config.prompt_text,
                            layer_weights=config.layer_weights,
                            system_prompt=config.system_prompt,
                            thinking_content=thinking,
                            force_think_block=force_think,
                        )
                    else:
                        # Use standard encoding with hidden_layer
                        text_result = self.pipeline.encoder.encode(
                            prompt=config.prompt_text,
                            system_prompt=config.system_prompt,
                            thinking_content=thinking,
                            force_think_block=force_think,
                            layer_index=config.hidden_layer,
                        )

                    text_embeddings = text_result.embeddings[0]  # Get first batch item

                    # Handle long prompt compression
                    from llm_dit import MAX_TEXT_SEQ_LEN
                    from llm_dit.utils.long_prompt import compress_embeddings

                    if text_embeddings.shape[0] > MAX_TEXT_SEQ_LEN:
                        text_embeddings = compress_embeddings(
                            text_embeddings, MAX_TEXT_SEQ_LEN, mode=config.long_prompt_mode
                        )

                    # Blend embeddings
                    from llm_dit.vl.blending import blend_embeddings

                    blended = blend_embeddings(
                        vl_emb=vl_embeddings,
                        text_emb=text_embeddings,
                        alpha=config.vl_alpha,
                        match_lengths=True,
                    )

                    # Use blended embeddings
                    gen_kwargs["prompt_embeds"] = blended

                    logger.debug(
                        f"Blended VL (alpha={config.vl_alpha}): "
                        f"VL {vl_embeddings.shape} + text {text_embeddings.shape} -> {blended.shape}"
                    )
            else:
                # Pure text conditioning (no VL)
                gen_kwargs["prompt"] = config.prompt_text

                if config.system_prompt:
                    gen_kwargs["system_prompt"] = config.system_prompt

                # Handle thinking content and force_think_block
                if config.thinking_content is not None:
                    if config.thinking_content == "":
                        gen_kwargs["force_think_block"] = True
                    else:
                        gen_kwargs["thinking_content"] = config.thinking_content
                else:
                    gen_kwargs["force_think_block"] = config.force_think_block

                gen_kwargs["hidden_layer"] = config.hidden_layer
                gen_kwargs["long_prompt_mode"] = config.long_prompt_mode

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

            # Note: metrics are computed in batch after all generation is done
            # See _compute_batch_metrics() called from run_experiment()

            return ExperimentResult(
                config=config,
                output_path=str(output_path),
                generation_time_seconds=generation_time,
                token_count=token_count,
            )

        except Exception as e:
            logger.error("Generation failed: %s", e)
            return ExperimentResult(
                config=config,
                output_path=str(output_path),
                generation_time_seconds=time.time() - start_time,
                error=str(e),
            )

    def _compute_batch_metrics(self, results: list[ExperimentResult]) -> None:
        """Compute metrics for all results after generation is complete.

        This is called after all images are generated, allowing us to:
        1. Unload the generation pipeline
        2. Free GPU memory
        3. Load metrics models
        4. Score all images efficiently
        """
        import gc

        # Unload pipeline to free GPU memory
        logger.info("Unloading pipeline to free GPU memory...")
        if hasattr(self, "pipeline") and self.pipeline is not None:
            del self.pipeline
            self.pipeline = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() / 1e9
            logger.info(f"GPU memory after unload: {allocated:.2f} GB")

        # Filter to successful results only
        valid_results = [r for r in results if r.error is None]
        if not valid_results:
            logger.warning("No successful results to compute metrics for")
            return

        prompts = [r.config.prompt_text for r in valid_results]
        image_paths = [r.output_path for r in valid_results]

        # ImageReward (human preference) - batch scoring
        try:
            from experiments.metrics import ImageRewardScorer

            logger.info("Computing ImageReward scores...")
            scorer = ImageRewardScorer(device="cuda" if torch.cuda.is_available() else "cpu")
            for i, result in enumerate(valid_results):
                try:
                    score = scorer.score(result.config.prompt_text, result.output_path)
                    result.image_reward = score
                    logger.debug("ImageReward [%d]: %.4f", i, score)
                except Exception as e:
                    logger.warning("ImageReward failed for %s: %s", result.output_path, e)
            # Cleanup
            del scorer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            logger.warning("ImageReward not installed, skipping. Install: uv add image-reward")
        except Exception as e:
            logger.warning("ImageReward computation failed: %s", e)

        # SigLIP2 (image-text alignment) - batch scoring
        try:
            from experiments.metrics import SigLIPScorer

            logger.info("Computing SigLIP scores...")
            scorer = SigLIPScorer(device="cuda" if torch.cuda.is_available() else "cpu")

            # Use batch scoring for efficiency
            scores = scorer.score_batch(prompts, image_paths)
            for result, score in zip(valid_results, scores):
                result.siglip_score = score
            logger.info("SigLIP scoring complete")

            # Cleanup
            del scorer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            logger.warning(
                "transformers not installed, skipping SigLIP. Install: uv add transformers"
            )
        except Exception as e:
            logger.warning("SigLIP computation failed: %s", e)

    def run_experiment(
        self,
        experiment_name: str,
        prompt_ids: list[str] | None = None,
        prompt_category: str | None = None,
        prompts_file: str | None = None,
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
            prompts = get_prompts_by_category(prompt_category, prompts_file)
        else:
            prompts = load_standard_prompts(prompts_file)["prompts"]

        if max_prompts:
            prompts = prompts[:max_prompts]

        # Get seeds
        if seeds is None:
            seeds = [42]  # Default to single seed

        # Get parameter values
        if "grid" in exp_def:
            # Grid search over multiple variables
            configs = self._build_grid_configs(experiment_name, exp_def, prompts, seeds)
        else:
            # Single variable sweep
            configs = self._build_sweep_configs(experiment_name, exp_def, prompts, seeds)

        logger.info(
            "Running %d configurations (%d prompts x %d values x %d seeds)",
            len(configs),
            len(prompts),
            len(exp_def.get("values", []))
            or len(list(self._grid_combinations(exp_def.get("grid", {})))),
            len(seeds),
        )

        # Run all configurations (generation only, no metrics yet)
        results = []
        for i, config in enumerate(configs):
            logger.info("Progress: %d/%d", i + 1, len(configs))
            result = self.run_single(config)
            results.append(result)

            # Save intermediate results (without metrics)
            self._save_result(result)

        # Compute metrics AFTER all generation is done
        # This allows us to unload the pipeline and load metrics models
        if self.compute_metrics:
            logger.info("Computing metrics for %d images...", len(results))
            self._compute_batch_metrics(results)

            # Re-save results with metrics
            for result in results:
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
                    elif variable == "vl_alpha":
                        config.vl_alpha = value
                    elif variable == "vl_token_selection":
                        config.vl_token_selection = value
                    elif variable == "vl_hidden_layer":
                        config.vl_hidden_layer = value

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
        metadata_path = self.output_dir / "metadata" / f"{Path(result.output_path).stem}.json"
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
            writer.writerow(
                [
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
                ]
            )
            for r in results:
                writer.writerow(
                    [
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
                    ]
                )

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
  Text-only experiments:
    shift_sweep       Sweep shift parameter (1.0-6.0)
    shift_steps_grid  Grid search over shift and steps
    hidden_layer      Compare hidden layer extraction points (-1 to -6)
    think_block       Test impact of think block content
    system_prompt     Test impact of system prompts
    steps_only        Test different step counts

  VL conditioning experiments:
    vl_token_selection      Compare VL token selection modes
    vl_pure                 Pure VL conditioning (alpha=1.0)
    vl_alpha_sweep          Sweep alpha blend ratio (0.0-1.0)
    vl_alpha_coarse         Coarse alpha sweep (0.0, 0.25, 0.5, 0.75, 1.0)
    vl_hidden_layer         Test VL extraction layers
    vl_blend_with_text_layers  VL + text layer combinations

Examples:
  # Run shift sweep on animal prompts
  uv run experiments/run_ablation.py --config config.toml --experiment shift_sweep animals

  # Run with custom prompts file
  uv run experiments/run_ablation.py --experiment hidden_layer scenes --prompts my_prompts.yaml

  # Run VL alpha sweep with reference image
  uv run experiments/run_ablation.py \\
    --experiment vl_alpha_sweep \\
    --vl-image /path/to/reference.jpg \\
    simple_objects

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
        "category",
        nargs="?",
        default=None,
        help="Prompt category to use (e.g., animals, scenes, artistic_styles)",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=None,
        help="Path to prompts YAML file (default: experiments/prompts/standard_prompts.yaml)",
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

    # VL conditioning arguments
    vl_group = parser.add_argument_group("Vision Conditioning (Qwen3-VL)")
    vl_group.add_argument(
        "--vl-model-path",
        help="Path to Qwen3-VL model (auto-detected if not provided)",
    )
    vl_group.add_argument(
        "--vl-image",
        help="Reference image for VL conditioning (required for VL experiments)",
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
        vl_model_path=args.vl_model_path,
        vl_image_path=args.vl_image,
    )

    results = runner.run_experiment(
        experiment_name=args.experiment,
        prompt_ids=prompt_ids,
        prompt_category=args.category,
        prompts_file=args.prompts,
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
