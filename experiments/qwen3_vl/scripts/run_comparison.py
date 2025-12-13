#!/usr/bin/env python3
"""
Run VL conditioning experiments with comparison grid generation.

This script runs experiments comparing different VL conditioning approaches
and generates comparison grids for visual analysis.

Sweep Types
-----------
--sweep alpha   Test VL influence strength (0.0=pure text to 1.0=pure VL)
                Configs: 0.0, 0.1, 0.3, 0.5, 0.7, 1.0 at layer -8, text_only

--sweep layer   Test hidden layer extraction depth
                Configs: -2, -4, -8, -16, -24 at alpha 1.0, text_only

--sweep token   Test which tokens to use from VL extraction
                Configs: full, text_only, image_only, image_no_markers
                at layer -8, alpha 1.0

--sweep normalization   Test different normalization modes for image tokens
                        Configs: global, per_dim, hybrid at layer -8,
                        image_only. Critical for fixing 600x+ per-dimension
                        outliers in image token embeddings.

--sweep outlier Test outlier dimension masking modes
                        Configs: none, zero, clamp, scale at layer -8,
                        full tokens. Targets dim 396 (617x ratio) and dim 4 (42x).

--sweep full    Comprehensive cross-product of all parameters
                Configs: alphas [0.0, 0.3, 0.5, 1.0] x layers [-2, -8, -16]
                x modes [text_only, image_only] = 24 total configs

Token Modes
-----------
full            All tokens (system + image + text + assistant)
text_only       Only text tokens (excludes image region) - best for prompt following
image_only      Only image tokens with markers - carries visual content
image_no_markers  Only image tokens without special markers

Grid Labels
-----------
Labels are human-readable descriptions of each experiment:

  Pure Text (no VL)              Qwen3-4B text only, no VL conditioning
  text tokens                    VL text token positions only
  image tokens                   VL image token positions only
  alpha=30% | text tokens        30% VL blended with text
  layer -8 | image tokens        Layer -8 extraction, image tokens
  alpha=50% | layer -8 | all     Combined params

Examples by sweep type:
  --sweep alpha  -> Pure Text (no VL), alpha=10% | text tokens, alpha=30% | ...
  --sweep layer  -> Pure Text (no VL), layer -2 | text tokens, layer -4 | ...
  --sweep token  -> Pure Text (no VL), all tokens, text tokens, image tokens, ...
  --sweep full   -> Pure Text (no VL), alpha=30% | layer -2 | text tokens, ...

Key Findings
------------
- text_only: Follows prompts but NO visual content from reference image
- image_only: Transfers visual content but ignores text prompt
- Layer -8 produces cleaner results than -2 for VL conditioning
- Cannot get both prompt following AND visual transfer with simple interpolation
- Image tokens have extreme per-dimension outliers (up to 617x std ratio)
- per_dim normalization is critical for image token quality (fixes outliers)
- text_only tokens have 0.999 correlation with Qwen3-4B and only need global scaling

Usage Examples
--------------
    # Basic test (defaults: layer -8, text_only, alpha 1.0 + baseline)
    uv run experiments/qwen3_vl/scripts/run_comparison.py \\
        -i experiments/inputs/test_scene.png \\
        -p "A cartoon house with a red roof"

    # Sweep alpha values
    uv run experiments/qwen3_vl/scripts/run_comparison.py \\
        -i image.png -p "Your prompt" --sweep alpha

    # Sweep hidden layers
    uv run experiments/qwen3_vl/scripts/run_comparison.py \\
        -i image.png -p "Your prompt" --sweep layer

    # Sweep token modes
    uv run experiments/qwen3_vl/scripts/run_comparison.py \\
        -i image.png -p "Your prompt" --sweep token

    # Custom values - combine any parameters freely
    uv run experiments/qwen3_vl/scripts/run_comparison.py \\
        -i image.png -p "Your prompt" \\
        --alphas 0.0 0.3 1.0 --layers -2 -8 --token-modes text_only image_only

    # Mix sweep with overrides (sweep alpha but test 2 layers)
    uv run experiments/qwen3_vl/scripts/run_comparison.py \\
        -i image.png -p "Your prompt" --sweep alpha --layers -2 -8

    # Full cross-product: 3 alphas x 2 layers x 2 modes = 12 configs + baseline
    uv run experiments/qwen3_vl/scripts/run_comparison.py \\
        -i image.png -p "Your prompt" \\
        --alphas 0.3 0.5 1.0 --layers -4 -8 --token-modes text_only image_only

    # Skip baseline, don't include prompt in VL extraction
    uv run experiments/qwen3_vl/scripts/run_comparison.py \\
        -i image.png -p "Your prompt" --no-baseline --no-vl-text

    # Run with prompts from standard prompts file
    uv run experiments/qwen3_vl/scripts/run_comparison.py \\
        -i image.png --prompt-ids animal_001,simple_002

    # Run all prompts in a category
    uv run experiments/qwen3_vl/scripts/run_comparison.py \\
        -i image.png --prompt-category animals

    # Run prompts by difficulty
    uv run experiments/qwen3_vl/scripts/run_comparison.py \\
        -i image.png --prompt-difficulty easy

    # Use custom prompts file
    uv run experiments/qwen3_vl/scripts/run_comparison.py \\
        -i image.png --prompts-file my_prompts.yaml --prompt-ids my_001,my_002
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Literal

import torch
import yaml
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def load_prompts_from_file(
    prompts_file: str | Path | None = None,
    prompt_ids: list[str] | None = None,
    prompt_category: str | None = None,
    prompt_difficulty: str | None = None,
) -> list[dict[str, Any]]:
    """Load and filter prompts from a YAML file.

    Args:
        prompts_file: Path to prompts YAML file. If None, uses standard_prompts.yaml
        prompt_ids: List of specific prompt IDs to include
        prompt_category: Filter by category (e.g., 'animals', 'scenes')
        prompt_difficulty: Filter by difficulty ('easy', 'medium', 'hard', 'extreme')

    Returns:
        List of prompt dicts with 'id', 'prompt', 'category', 'difficulty' keys
    """
    # Default to standard prompts file
    if prompts_file is None:
        from experiments.prompts import STANDARD_PROMPTS_FILE
        prompts_file = STANDARD_PROMPTS_FILE

    prompts_file = Path(prompts_file)
    if not prompts_file.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")

    with open(prompts_file) as f:
        data = yaml.safe_load(f)

    prompts = data.get("prompts", [])

    # Filter by IDs if specified
    if prompt_ids:
        prompts = [p for p in prompts if p.get("id") in prompt_ids]
        # Warn about missing IDs
        found_ids = {p.get("id") for p in prompts}
        missing = set(prompt_ids) - found_ids
        if missing:
            logger.warning(f"Prompt IDs not found: {missing}")

    # Filter by category
    if prompt_category:
        prompts = [p for p in prompts if p.get("category") == prompt_category]

    # Filter by difficulty
    if prompt_difficulty:
        prompts = [p for p in prompts if p.get("difficulty") == prompt_difficulty]

    return prompts


TokenMode = Literal["full", "text_only", "image_only", "image_no_markers"]
NormalizationMode = Literal["global", "per_dim", "hybrid"]
OutlierMaskingMode = Literal["none", "zero", "clamp", "scale"]
BlendMode = Literal["interpolate", "adain_per_dim", "adain", "linear"]


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    name: str  # Human-readable label for grid display
    filename: str = ""  # Filesystem-safe name (auto-generated if empty)
    alpha: float = 1.0
    hidden_layer: int = -8
    token_mode: TokenMode = "text_only"
    normalization_mode: NormalizationMode = "global"  # global, per_dim, or hybrid
    outlier_masking: OutlierMaskingMode = "none"  # none, zero, clamp, or scale
    outlier_threshold: float = 10.0  # Std ratio threshold for outlier detection
    vl_text: str | None = None  # None = no text, "__PROMPT__" = use CLI prompt
    scale_to_text: bool = True
    # Chat template configuration - ensures VL and text use identical formats
    # IMPORTANT: Official diffusers/DiffSynth use enable_thinking=True = NO think block
    force_think_block: bool = False  # False = match official (no think block)
    system_prompt: str | None = None  # Optional system message (official uses none)
    # Style transfer parameters (NEW)
    blend_mode: BlendMode = "interpolate"  # interpolate (recommended), adain_per_dim, adain, linear
    use_img2img: bool = False  # Use img2img instead of txt2img for style transfer
    strength: float = 0.9  # img2img strength (only used if use_img2img=True)

    def __post_init__(self):
        if not self.filename:
            # Generate filesystem-safe name from display name
            self.filename = self.name.replace(" | ", "_").replace(" ", "_").replace("=", "").replace("%", "pct").replace("(", "").replace(")", "")

    @property
    def image_tokens_only(self) -> bool:
        return self.token_mode == "image_only"

    @property
    def image_tokens_no_markers(self) -> bool:
        return self.token_mode == "image_no_markers"

    @property
    def text_tokens_only(self) -> bool:
        return self.token_mode == "text_only"


def build_configs(
    alphas: list[float] | None = None,
    layers: list[int] | None = None,
    token_modes: list[TokenMode] | None = None,
    normalization_modes: list[NormalizationMode] | None = None,
    outlier_masking_modes: list[OutlierMaskingMode] | None = None,
    outlier_threshold: float = 10.0,
    include_baseline: bool = True,
    vl_text: str | None = "__PROMPT__",
    force_think_block: bool = False,
    system_prompt: str | None = None,
) -> list[ExperimentConfig]:
    """
    Build experiment configurations from parameters.

    Args:
        alphas: Alpha values to test. Default [1.0]
        layers: Hidden layers to test. Default [-8]
        token_modes: Token selection modes. Default ["text_only"]
        normalization_modes: Normalization modes. Default ["global"]
        outlier_masking_modes: Outlier masking modes. Default ["none"]
        outlier_threshold: Std ratio threshold for outlier detection. Default 10.0
        include_baseline: Add pure text baseline (alpha=0.0)
        vl_text: Text to include in VL extraction. "__PROMPT__" = use CLI prompt
        force_think_block: If True, add empty think block (matches text encoding format)
        system_prompt: Optional system message (matches text encoding format)

    Returns:
        List of ExperimentConfig objects
    """
    configs = []

    # Defaults
    alphas = alphas or [1.0]
    layers = layers or [-8]
    token_modes = token_modes or ["text_only"]
    normalization_modes = normalization_modes or ["global"]
    outlier_masking_modes = outlier_masking_modes or ["none"]

    # Add baseline if requested
    if include_baseline and 0.0 not in alphas:
        configs.append(ExperimentConfig(
            name="Pure Text (no VL)",
            alpha=0.0,
            vl_text=None,
            force_think_block=force_think_block,
            system_prompt=system_prompt,
        ))

    # Generate configs for all combinations
    for alpha in alphas:
        if alpha == 0.0:
            continue  # Skip, handled by baseline

        for layer in layers:
            for mode in token_modes:
                for norm_mode in normalization_modes:
                    for outlier_mode in outlier_masking_modes:
                        name_parts = []

                        # Alpha in name if varying
                        if len(alphas) > 1:
                            name_parts.append(f"alpha={alpha:.0%}")

                        # Layer in name if varying
                        if len(layers) > 1:
                            name_parts.append(f"layer {layer}")

                        # Mode in name - always include for clarity
                        mode_labels = {
                            "full": "all tokens",
                            "text_only": "text tokens",
                            "image_only": "image tokens",
                            "image_no_markers": "image (no markers)",
                        }
                        name_parts.append(mode_labels.get(mode, mode))

                        # Normalization in name if varying
                        if len(normalization_modes) > 1:
                            norm_labels = {
                                "global": "global norm",
                                "per_dim": "per-dim norm",
                                "hybrid": "hybrid norm",
                            }
                            name_parts.append(norm_labels.get(norm_mode, norm_mode))

                        # Outlier masking in name if varying
                        if len(outlier_masking_modes) > 1:
                            outlier_labels = {
                                "none": "no masking",
                                "zero": "zero outliers",
                                "clamp": "clamp outliers",
                                "scale": "scale outliers",
                            }
                            name_parts.append(outlier_labels.get(outlier_mode, outlier_mode))

                        configs.append(ExperimentConfig(
                            name=" | ".join(name_parts),
                            alpha=alpha,
                            hidden_layer=layer,
                            token_mode=mode,
                            normalization_mode=norm_mode,
                            outlier_masking=outlier_mode,
                            outlier_threshold=outlier_threshold,
                            vl_text=vl_text,
                            force_think_block=force_think_block,
                            system_prompt=system_prompt,
                        ))

    return configs


def get_sweep_configs(sweep_type: str) -> list[ExperimentConfig]:
    """Get predefined sweep configurations."""

    if sweep_type == "alpha":
        return build_configs(
            alphas=[0.0, 0.1, 0.3, 0.5, 0.7, 1.0],
            layers=[-8],
            token_modes=["text_only"],
        )

    elif sweep_type == "layer":
        return build_configs(
            alphas=[1.0],
            layers=[-2, -4, -8, -16, -24],
            token_modes=["text_only"],
        )

    elif sweep_type == "token":
        return build_configs(
            alphas=[1.0],
            layers=[-8],
            token_modes=["full", "text_only", "image_only", "image_no_markers"],
        )

    elif sweep_type == "normalization":
        # Test different normalization modes on image tokens (where it matters most)
        return build_configs(
            alphas=[1.0],
            layers=[-8],
            token_modes=["image_only"],
            normalization_modes=["global", "per_dim", "hybrid"],
        )

    elif sweep_type == "full":
        # Comprehensive sweep
        return build_configs(
            alphas=[0.0, 0.3, 0.5, 1.0],
            layers=[-2, -8, -16],
            token_modes=["text_only", "image_only"],
        )

    elif sweep_type == "outlier":
        # Test outlier masking modes on image tokens (where outliers matter most)
        # Key outliers: dimension 396 (617x std ratio), dimension 4 (42x ratio)
        return build_configs(
            alphas=[1.0],
            layers=[-8],
            token_modes=["full"],  # Image tokens where outliers are extreme
            normalization_modes=["global"],
            outlier_masking_modes=["none", "zero", "clamp", "scale"],
        )

    else:
        raise ValueError(f"Unknown sweep type: {sweep_type}")


def generate_comparison_grid(
    results: list[dict],
    output_dir: Path,
    reference_image_path: str,
    prompt: str,
    thumbnail_size: int = 256,
) -> Path | None:
    """
    Generate a comparison grid from experiment results.

    Automatically detects if results form a 2D grid (e.g., layers x token_modes)
    and creates appropriate row/column headers.
    """
    from PIL import Image as PILImage, ImageDraw, ImageFont

    successful = [r for r in results if r.get("success", False)]
    if not successful:
        logger.warning("No successful results to create grid from")
        return None

    # Load fonts
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        header_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
        header_font = font
        small_font = font

    # Analyze what parameters vary
    alphas = sorted(set(r["alpha"] for r in successful))
    layers = sorted(set(r["hidden_layer"] for r in successful), reverse=True)
    modes = list(dict.fromkeys(r["token_mode"] for r in successful))  # preserve order
    norm_modes = list(dict.fromkeys(r.get("normalization_mode", "global") for r in successful))

    # Determine grid layout based on varying parameters
    # Priority: rows=layers, cols=modes, or rows=alphas, cols=modes
    varying_params = []
    if len(alphas) > 1:
        varying_params.append(("alpha", alphas))
    if len(layers) > 1:
        varying_params.append(("layer", layers))
    if len(modes) > 1:
        varying_params.append(("mode", modes))
    if len(norm_modes) > 1:
        varying_params.append(("normalization", norm_modes))

    # Build 2D grid if exactly 2 params vary, otherwise 1D
    if len(varying_params) == 2:
        row_param, row_values = varying_params[0]
        col_param, col_values = varying_params[1]
    elif len(varying_params) == 1:
        row_param, row_values = None, [None]
        col_param, col_values = varying_params[0]
    else:
        # No params vary or 3+ params - fall back to 1D list
        row_param, row_values = None, [None]
        col_param, col_values = "name", [r["name"] for r in successful]

    # Helper to get result for specific param values
    def get_result(row_val, col_val):
        for r in successful:
            match = True
            if row_param == "alpha" and r["alpha"] != row_val:
                match = False
            elif row_param == "layer" and r["hidden_layer"] != row_val:
                match = False
            elif row_param == "mode" and r["token_mode"] != row_val:
                match = False
            elif row_param == "normalization" and r.get("normalization_mode", "global") != row_val:
                match = False
            if col_param == "alpha" and r["alpha"] != col_val:
                match = False
            elif col_param == "layer" and r["hidden_layer"] != col_val:
                match = False
            elif col_param == "mode" and r["token_mode"] != col_val:
                match = False
            elif col_param == "normalization" and r.get("normalization_mode", "global") != col_val:
                match = False
            elif col_param == "name" and r["name"] != col_val:
                match = False
            if match:
                return r
        return None

    # Format labels
    def format_label(param, val):
        if param == "alpha":
            return f"alpha={val:.0%}"
        elif param == "layer":
            return f"layer {val}"
        elif param == "mode":
            mode_labels = {
                "full": "all tokens",
                "text_only": "text tokens",
                "image_only": "image tokens",
                "image_no_markers": "img (no markers)",
            }
            return mode_labels.get(val, val)
        elif param == "normalization":
            norm_labels = {
                "global": "global norm",
                "per_dim": "per-dim norm",
                "hybrid": "hybrid norm",
            }
            return norm_labels.get(val, val)
        elif param == "name":
            return val[:18] + ".." if len(val) > 20 else val
        return str(val)

    # Grid dimensions
    n_rows = len(row_values)
    n_cols = len(col_values)

    padding = 8
    header_height = 40
    row_label_width = 120 if row_param else 0
    col_label_height = 30

    # Calculate canvas size
    total_width = row_label_width + n_cols * (thumbnail_size + padding) + padding
    total_height = header_height + col_label_height + n_rows * (thumbnail_size + padding) + padding

    # Add reference image column
    total_width += thumbnail_size + padding

    canvas = PILImage.new("RGB", (total_width, total_height), color=(30, 30, 30))
    draw = ImageDraw.Draw(canvas)

    # Draw title
    prompt_display = prompt[:70] + "..." if len(prompt) > 70 else prompt
    draw.text(
        (total_width // 2, header_height // 2),
        f"Prompt: {prompt_display}",
        fill=(200, 200, 200),
        font=header_font,
        anchor="mm",
    )

    # Draw column headers
    x_start = row_label_width + padding + thumbnail_size + padding  # After ref column
    for col_idx, col_val in enumerate(col_values):
        x = x_start + col_idx * (thumbnail_size + padding) + thumbnail_size // 2
        y = header_height + col_label_height // 2
        label = format_label(col_param, col_val)
        draw.text((x, y), label, fill=(255, 200, 100), font=font, anchor="mm")

    # Draw "Reference" header
    ref_x = row_label_width + padding + thumbnail_size // 2
    draw.text((ref_x, header_height + col_label_height // 2), "Reference", fill=(100, 200, 255), font=font, anchor="mm")

    # Load reference image
    ref_img = PILImage.open(reference_image_path).convert("RGB")
    ref_thumb = ref_img.copy()
    ref_thumb.thumbnail((thumbnail_size, thumbnail_size), PILImage.Resampling.LANCZOS)

    # Draw grid
    y_start = header_height + col_label_height + padding
    for row_idx, row_val in enumerate(row_values):
        y = y_start + row_idx * (thumbnail_size + padding)

        # Draw row label
        if row_param:
            label = format_label(row_param, row_val)
            draw.text(
                (row_label_width - 10, y + thumbnail_size // 2),
                label,
                fill=(100, 255, 100),
                font=font,
                anchor="rm",
            )

        # Draw reference image in first column (only for first row, span all rows visually)
        if row_idx == 0 or n_rows == 1:
            ref_x = row_label_width + padding
            ref_y = y_start + (n_rows * (thumbnail_size + padding) - padding) // 2 - thumbnail_size // 2
            if n_rows == 1:
                ref_y = y
            paste_x = ref_x + (thumbnail_size - ref_thumb.width) // 2
            paste_y = ref_y + (thumbnail_size - ref_thumb.height) // 2
            canvas.paste(ref_thumb, (paste_x, paste_y))

        # Draw result images
        for col_idx, col_val in enumerate(col_values):
            x = x_start + col_idx * (thumbnail_size + padding)

            result = get_result(row_val, col_val)
            if result:
                try:
                    img = PILImage.open(result["output_path"]).convert("RGB")
                    thumb = img.copy()
                    thumb.thumbnail((thumbnail_size, thumbnail_size), PILImage.Resampling.LANCZOS)
                    paste_x = x + (thumbnail_size - thumb.width) // 2
                    paste_y = y + (thumbnail_size - thumb.height) // 2
                    canvas.paste(thumb, (paste_x, paste_y))
                except Exception as e:
                    logger.warning(f"Could not load {result['output_path']}: {e}")
                    draw.text((x + thumbnail_size // 2, y + thumbnail_size // 2), "Error", fill=(255, 0, 0), font=font, anchor="mm")
            else:
                draw.text((x + thumbnail_size // 2, y + thumbnail_size // 2), "N/A", fill=(128, 128, 128), font=font, anchor="mm")

    grid_path = output_dir / "comparison_grid.png"
    canvas.save(grid_path)
    logger.info(f"Saved comparison grid to {grid_path}")

    return grid_path


def run_experiments(
    image_path: str,
    prompt: str,
    output_dir: str,
    configs: list[ExperimentConfig],
    vl_model_path: str | None = None,
    z_image_config: str = "config.toml",
    z_image_profile: str = "rtx4090",
    seed: int = 42,
    steps: int = 9,
    force_think_block: bool = False,
    system_prompt: str | None = None,
):
    """Run all experiment configurations.

    Args:
        force_think_block: If True, add empty think block to text encoding.
            Default False to match official Z-Image format (enable_thinking=True = no think block).
        system_prompt: Optional system message for text encoding (official uses none).
    """
    from llm_dit.vl import VLEmbeddingExtractor
    from llm_dit.vl.blending import blend_embeddings
    from llm_dit.startup import PipelineLoader
    from blend_and_generate import encode_text_prompt, TextEncodingResult

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load image
    image = Image.open(image_path).convert("RGB")
    logger.info(f"Loaded image: {image_path} ({image.size[0]}x{image.size[1]})")

    # Load Z-Image config
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
    from llm_dit.cli import load_runtime_config

    class ConfigArgs:
        pass
    config_args = ConfigArgs()
    config_args.config = z_image_config
    config_args.profile = z_image_profile
    config_args.steps = steps
    for attr in ['model_path', 'text_encoder_device', 'dit_device', 'vae_device',
                 'cpu_offload', 'flash_attn', 'compile', 'debug', 'verbose',
                 'attention_backend', 'use_custom_scheduler', 'tiled_vae',
                 'embedding_cache', 'long_prompt_mode', 'hidden_layer', 'shift',
                 'lora', 'api_url', 'api_model', 'local_encoder', 'templates_dir',
                 'torch_dtype', 'text_encoder_path', 'tile_size', 'tile_overlap',
                 'cache_size', 'rewriter_use_api', 'rewriter_api_url', 'rewriter_api_model',
                 'rewriter_temperature', 'rewriter_top_p', 'rewriter_top_k',
                 'rewriter_min_p', 'rewriter_presence_penalty', 'rewriter_max_tokens',
                 'width', 'height', 'guidance_scale', 'negative_prompt', 'seed',
                 'embeddings_file', 'template', 'system_prompt', 'thinking_content',
                 'assistant_content', 'enable_thinking',
                 'vl_model_path', 'vl_device', 'vl_hidden_layer', 'vl_alpha',
                 'vl_blend_mode', 'vl_auto_unload']:
        if not hasattr(config_args, attr):
            setattr(config_args, attr, None)
    z_config = load_runtime_config(config_args)

    # Encode text prompt once (with format matching VL's injected think block)
    logger.info(f"Encoding text prompt: {prompt[:50]}...")
    logger.info(f"  force_think_block={force_think_block}, system_prompt={system_prompt}")
    text_result = encode_text_prompt(
        prompt,
        z_config,
        force_think_block=force_think_block,
        system_prompt=system_prompt,
    )
    text_emb = text_result.embeddings
    text_formatted_prompt = text_result.formatted_prompt
    logger.info(f"Text embeddings: shape={text_emb.shape}, std={text_emb.std():.2f}")
    logger.debug(f"Text formatted prompt:\n{text_formatted_prompt}")

    # Get VL model path
    if not vl_model_path and hasattr(z_config, 'vl_model_path'):
        vl_model_path = z_config.vl_model_path
    if not vl_model_path:
        candidates = [
            Path.home() / "Storage" / "Qwen3-VL-4B-Instruct",
            Path.home() / "models" / "Qwen3-VL-4B-Instruct",
        ]
        for candidate in candidates:
            if candidate.exists():
                vl_model_path = str(candidate)
                break

    if not vl_model_path:
        raise ValueError("Could not find Qwen3-VL model. Set vl.model_path in config.toml or use --vl-model-path")

    # Load VL extractor
    vl_device = getattr(z_config, 'vl_device', None) or "cuda"
    vl_dtype = torch.bfloat16 if vl_device == "cuda" else torch.float32
    logger.info(f"Loading VLEmbeddingExtractor from {vl_model_path}")
    vl_extractor = VLEmbeddingExtractor.from_pretrained(
        vl_model_path,
        device=vl_device,
        torch_dtype=vl_dtype,
    )

    # Load Z-Image pipeline
    logger.info("Loading Z-Image pipeline...")
    loader = PipelineLoader(z_config)
    pipeline_result = loader.load_pipeline()
    pipe = pipeline_result.pipeline
    logger.info("Pipeline loaded successfully")

    generator = torch.Generator()
    results = []

    for i, config in enumerate(configs):
        logger.info(f"\n{'='*60}")
        logger.info(f"[{i+1}/{len(configs)}] Running: {config.name}")
        logger.info(f"  alpha={config.alpha}, layer={config.hidden_layer}, mode={config.token_mode}")
        if config.outlier_masking != "none":
            logger.info(f"  outlier_masking={config.outlier_masking}, threshold={config.outlier_threshold}")
        logger.info(f"{'='*60}")

        start_time = time.time()

        try:
            # Handle __PROMPT__ placeholder
            vl_text = config.vl_text
            if vl_text == "__PROMPT__":
                vl_text = prompt

            # Extract VL embeddings with format configuration
            vl_result = vl_extractor.extract(
                image=image,
                text=vl_text,
                hidden_layer=config.hidden_layer,
                image_tokens_only=config.image_tokens_only,
                image_tokens_no_markers=config.image_tokens_no_markers,
                text_tokens_only=config.text_tokens_only,
                scale_to_text=config.scale_to_text,
                normalization_mode=config.normalization_mode,
                force_think_block=config.force_think_block,
                system_prompt=config.system_prompt,
                outlier_masking=config.outlier_masking,
                outlier_threshold=config.outlier_threshold,
            )
            vl_emb = vl_result.embeddings

            # Blend embeddings
            if config.alpha == 0.0:
                blended = text_emb
            elif config.alpha == 1.0:
                blended = vl_emb
            else:
                blended = blend_embeddings(vl_emb, text_emb, config.alpha)

            # Generate
            output_path = output_dir / f"{config.filename}.png"
            generator.manual_seed(seed)

            logger.info(f"  Generating {z_config.width}x{z_config.height} image...")
            gen_start = time.time()

            result_image = pipe(
                prompt_embeds=blended,
                height=z_config.height,
                width=z_config.width,
                num_inference_steps=z_config.steps,
                guidance_scale=z_config.guidance_scale,
                generator=generator,
            )

            gen_time = time.time() - gen_start
            logger.info(f"  Generation time: {gen_time:.1f}s")

            output_path.parent.mkdir(parents=True, exist_ok=True)
            result_image.save(output_path)
            logger.info(f"  Saved to {output_path}")

            elapsed = time.time() - start_time

            results.append({
                **asdict(config),
                "output_path": str(output_path),
                "vl_shape": list(vl_emb.shape),
                "vl_original_std": vl_result.original_std,
                "vl_scaled_std": vl_result.scaled_std,
                "vl_scale_factor": vl_result.scale_factor,
                "blended_std": blended.std().item(),
                "elapsed_seconds": elapsed,
                "generation_time": gen_time,
                "success": True,
                # Full prompt with special tokens for debugging
                "vl_full_prompt": vl_result.full_prompt_with_tokens,
                "vl_chat_template_format": vl_result.chat_template_format,
                # Outlier masking info
                "vl_masked_dimensions": vl_result.masked_dimensions,
                "vl_masked_dim_ratios": vl_result.masked_dim_ratios,
            })

        except Exception as e:
            logger.error(f"Failed: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                **asdict(config),
                "success": False,
                "error": str(e),
            })

    # Cleanup
    logger.info("\nCleaning up models...")
    del vl_extractor
    del pipe
    del pipeline_result
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save metadata with full prompt details and model info
    metadata = {
        "image_path": str(image_path),
        "prompt": prompt,
        "seed": seed,
        "steps": steps,
        # Text encoder configuration and formatted prompt
        "text_encoder": {
            "model": "Qwen3-4B",  # Z-Image text encoder
            "force_think_block": force_think_block,
            "system_prompt": system_prompt,
            "formatted_prompt": text_formatted_prompt,  # Full prompt with special tokens
        },
        # VL model info
        "vl_model": {
            "model": "Qwen3-VL-4B-Instruct",
            "path": vl_model_path,
        },
        "results": results,
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Generate comparison grid
    logger.info("\nGenerating comparison grid...")
    generate_comparison_grid(
        results=results,
        output_dir=output_dir,
        reference_image_path=image_path,
        prompt=prompt,
    )

    logger.info(f"\nResults saved to {output_dir}")
    logger.info(f"Total experiments: {len(results)}")
    logger.info(f"Successful: {sum(1 for r in results if r.get('success', False))}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run VL conditioning experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic test
  %(prog)s -i image.png -p "Your prompt"

  # Alpha sweep
  %(prog)s -i image.png -p "Your prompt" --sweep alpha

  # Custom values
  %(prog)s -i image.png -p "Your prompt" --alphas 0.0 0.5 1.0 --layers -2 -8
        """,
    )

    # Required
    parser.add_argument("--image", "-i", required=True, help="Reference image path")

    # Prompt source (one of these is required)
    prompt_group = parser.add_argument_group("Prompt source (use ONE of these)")
    prompt_group.add_argument("--prompt", "-p",
                              help="Single text prompt (for quick tests)")
    prompt_group.add_argument("--prompt-ids",
                              help="Comma-separated prompt IDs from prompts file (e.g., animal_001,scene_002)")
    prompt_group.add_argument("--prompt-category",
                              help="Run all prompts in category (e.g., animals, scenes, landscapes)")
    prompt_group.add_argument("--prompt-difficulty",
                              choices=["easy", "medium", "hard", "extreme"],
                              help="Run all prompts of a difficulty level")

    # Prompts file (optional, defaults to standard_prompts.yaml)
    parser.add_argument("--prompts-file",
                        help="Path to prompts YAML file (default: experiments/prompts/standard_prompts.yaml)")

    # Output
    parser.add_argument("--output-dir", "-o", default=None,
                        help="Output directory (default: auto-generated)")

    # Sweep presets
    parser.add_argument("--sweep", "-s", choices=["alpha", "layer", "token", "normalization", "outlier", "full"],
                        help="Use predefined sweep configuration")

    # Custom parameters
    parser.add_argument("--alphas", type=float, nargs="+",
                        help="Alpha values to test (e.g., 0.0 0.3 0.5 1.0)")
    parser.add_argument("--layers", type=int, nargs="+",
                        help="Hidden layers to test (e.g., -2 -8 -16)")
    parser.add_argument("--token-modes", nargs="+",
                        choices=["full", "text_only", "image_only", "image_no_markers"],
                        help="Token selection modes to test")
    parser.add_argument("--normalization-modes", nargs="+",
                        choices=["global", "per_dim", "hybrid"],
                        help="Normalization modes to test")
    parser.add_argument("--outlier-masking", nargs="+",
                        choices=["none", "zero", "clamp", "scale"],
                        help="Outlier masking modes to test (dim 396=617x, dim 4=42x std ratio)")
    parser.add_argument("--outlier-threshold", type=float, default=10.0,
                        help="Std ratio threshold for outlier detection (default: 10.0)")

    # Style transfer parameters (NEW)
    parser.add_argument("--blend-modes", nargs="+",
                        choices=["interpolate", "adain_per_dim", "adain", "linear"],
                        default=["interpolate"],
                        help="Blend modes to test. interpolate (recommended), adain_per_dim (best for style), "
                             "adain, linear (WARNING: truncates, loses 99%% VL info)")
    parser.add_argument("--img2img", action="store_true",
                        help="Use img2img (reference image as VAE latent init) for style transfer")
    parser.add_argument("--strengths", type=float, nargs="+",
                        default=[0.9],
                        help="img2img strengths to test (requires --img2img). 0.9 recommended.")

    # Flags
    parser.add_argument("--no-baseline", action="store_true",
                        help="Skip pure text baseline")
    parser.add_argument("--no-vl-text", action="store_true",
                        help="Don't include prompt text in VL extraction")

    # Model paths
    parser.add_argument("--vl-model-path", help="Qwen3-VL model path")
    parser.add_argument("--config", default="config.toml", help="Z-Image config file")
    parser.add_argument("--profile", default="rtx4090", help="Z-Image config profile")

    # Generation
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--steps", type=int, default=9, help="Inference steps")

    # Chat template format (default matches official diffusers/DiffSynth implementation)
    parser.add_argument("--force-think-block", action="store_true",
                        help="Add empty think block to text encoding (NOT default - official uses none)")
    parser.add_argument("--no-think-block", action="store_true",
                        help="Explicitly disable think block (this is the default, matching official)")
    parser.add_argument("--system-prompt", type=str, default=None,
                        help="System prompt for text encoding")

    args = parser.parse_args()

    # Validate prompt source - need at least one
    has_prompt_source = any([
        args.prompt,
        args.prompt_ids,
        args.prompt_category,
        args.prompt_difficulty,
    ])
    if not has_prompt_source:
        parser.error("Must provide a prompt source: --prompt, --prompt-ids, --prompt-category, or --prompt-difficulty")

    # Build list of prompts to run
    if args.prompt:
        # Single prompt mode - create a simple dict
        prompts_to_run = [{
            "id": "cli_prompt",
            "prompt": args.prompt,
            "category": "cli",
            "difficulty": "unknown",
        }]
    else:
        # Load from prompts file with filters
        prompt_ids = None
        if args.prompt_ids:
            prompt_ids = [p.strip() for p in args.prompt_ids.split(",")]

        prompts_to_run = load_prompts_from_file(
            prompts_file=args.prompts_file,
            prompt_ids=prompt_ids,
            prompt_category=args.prompt_category,
            prompt_difficulty=args.prompt_difficulty,
        )

        if not prompts_to_run:
            logger.error("No prompts found matching the specified filters")
            return 1

        logger.info(f"Loaded {len(prompts_to_run)} prompts from file")

    # Handle think block flag
    force_think_block = args.force_think_block and not args.no_think_block

    # Build configs - sweep sets defaults, CLI args override
    alphas = args.alphas
    layers = args.layers
    token_modes = args.token_modes
    normalization_modes = args.normalization_modes
    outlier_masking_modes = args.outlier_masking
    outlier_threshold = args.outlier_threshold

    # If sweep specified, use as defaults for any unset params
    if args.sweep:
        sweep_defaults = {
            "alpha": {"alphas": [0.0, 0.1, 0.3, 0.5, 0.7, 1.0], "layers": [-8], "token_modes": ["text_only"], "normalization_modes": ["global"], "outlier_masking_modes": ["none"]},
            "layer": {"alphas": [1.0], "layers": [-2, -4, -8, -16, -24], "token_modes": ["text_only"], "normalization_modes": ["global"], "outlier_masking_modes": ["none"]},
            "token": {"alphas": [1.0], "layers": [-8], "token_modes": ["full", "text_only", "image_only", "image_no_markers"], "normalization_modes": ["global"], "outlier_masking_modes": ["none"]},
            "normalization": {"alphas": [1.0], "layers": [-8], "token_modes": ["image_only"], "normalization_modes": ["global", "per_dim", "hybrid"], "outlier_masking_modes": ["none"]},
            "outlier": {"alphas": [1.0], "layers": [-8], "token_modes": ["full"], "normalization_modes": ["global"], "outlier_masking_modes": ["none", "zero", "clamp", "scale"]},
            "full": {"alphas": [0.0, 0.3, 0.5, 1.0], "layers": [-2, -8, -16], "token_modes": ["text_only", "image_only"], "normalization_modes": ["global"], "outlier_masking_modes": ["none"]},
        }
        defaults = sweep_defaults[args.sweep]
        alphas = alphas or defaults["alphas"]
        layers = layers or defaults["layers"]
        token_modes = token_modes or defaults["token_modes"]
        normalization_modes = normalization_modes or defaults["normalization_modes"]
        outlier_masking_modes = outlier_masking_modes or defaults["outlier_masking_modes"]

    configs = build_configs(
        alphas=alphas,
        layers=layers,
        token_modes=token_modes,
        normalization_modes=normalization_modes,
        outlier_masking_modes=outlier_masking_modes,
        outlier_threshold=outlier_threshold,
        include_baseline=not args.no_baseline,
        vl_text=None if args.no_vl_text else "__PROMPT__",
        force_think_block=force_think_block,
        system_prompt=args.system_prompt,
    )

    # Generate base output directory
    from datetime import datetime
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sweep_name = args.sweep or "custom"
        base_output_dir = f"experiments/results/vl_{sweep_name}_{timestamp}"
    else:
        base_output_dir = args.output_dir

    total_configs = len(configs) * len(prompts_to_run)
    logger.info(f"Running {len(configs)} configs x {len(prompts_to_run)} prompts = {total_configs} total experiments")
    logger.info(f"Base output directory: {base_output_dir}")

    # Run experiments for each prompt
    all_results = []
    for i, prompt_data in enumerate(prompts_to_run):
        prompt_id = prompt_data["id"]
        prompt_text = prompt_data["prompt"]

        # Create subdirectory for each prompt if multiple prompts
        if len(prompts_to_run) > 1:
            output_dir = f"{base_output_dir}/{prompt_id}"
        else:
            output_dir = base_output_dir

        logger.info(f"\n{'='*60}")
        logger.info(f"Prompt {i+1}/{len(prompts_to_run)}: {prompt_id}")
        logger.info(f"  Text: {prompt_text[:80]}{'...' if len(prompt_text) > 80 else ''}")
        logger.info(f"  Output: {output_dir}")
        logger.info(f"{'='*60}")

        run_experiments(
            image_path=args.image,
            prompt=prompt_text,
            output_dir=output_dir,
            configs=configs,
            vl_model_path=args.vl_model_path,
            z_image_config=args.config,
            z_image_profile=args.profile,
            seed=args.seed,
            steps=args.steps,
            force_think_block=force_think_block,
            system_prompt=args.system_prompt,
        )

        all_results.append({
            "prompt_id": prompt_id,
            "prompt_text": prompt_text,
            "output_dir": output_dir,
        })

    # If multiple prompts, save a summary
    if len(prompts_to_run) > 1:
        summary_path = Path(base_output_dir) / "prompts_summary.json"
        with open(summary_path, "w") as f:
            json.dump({
                "prompts_file": str(args.prompts_file) if args.prompts_file else "standard_prompts.yaml",
                "prompt_ids": args.prompt_ids,
                "prompt_category": args.prompt_category,
                "prompt_difficulty": args.prompt_difficulty,
                "total_prompts": len(prompts_to_run),
                "configs_per_prompt": len(configs),
                "results": all_results,
            }, f, indent=2)
        logger.info(f"\nSummary saved to {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
