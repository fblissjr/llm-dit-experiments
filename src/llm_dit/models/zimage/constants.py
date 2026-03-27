"""
Z-Image model constants and variant detection.

Last Updated: 2026-03-26

This module defines variant configurations and provides utilities for
auto-detecting which Z-Image variant is being used based on scheduler settings.

Key Differences: Base vs Turbo
------------------------------
| Setting            | Z-Image Base    | Z-Image Turbo           |
|--------------------|-----------------|-------------------------|
| Scheduler shift    | 6.0             | 3.0                     |
| Steps              | 40 (28-50 rec.) | 9 (8 actual forwards)   |
| CFG/guidance_scale | 4.0 (3.0-5.0)   | 0.0 (baked in)          |
| Negative prompts   | Supported       | Not used                |
| CFG normalization  | Optional        | Not applicable          |
| Model path         | models/Z-Image  | models/Z-Image-Turbo    |

Architecture is identical: same pipeline, transformer, VAE, text encoder, scheduler.
"""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# =============================================================================
# Variant Configurations
# =============================================================================

ZIMAGE_VARIANTS: dict[str, dict[str, Any]] = {
    "turbo": {
        "shift": 3.0,
        "distilled": True,
        "defaults": {
            "num_inference_steps": 9,
            "guidance_scale": 0.0,
            "cfg_normalization": 0.0,
            "negative_prompt": None,
        },
        "description": "Turbo distilled (fast, 8-9 steps, CFG baked in)",
    },
    "base": {
        "shift": 6.0,
        "distilled": False,
        "defaults": {
            "num_inference_steps": 40,
            "guidance_scale": 4.0,
            "cfg_normalization": 0.0,
            "negative_prompt": "",
        },
        "description": "Base model (quality, 28-50 steps, full CFG)",
    },
}


# =============================================================================
# Variant Detection
# =============================================================================


def detect_zimage_variant(model_path: str) -> str:
    """
    Auto-detect Z-Image variant from scheduler_config.json shift value.

    The shift parameter in the scheduler config distinguishes variants:
    - shift=6.0 -> base model
    - shift=3.0 -> turbo model

    Args:
        model_path: Path to Z-Image model directory

    Returns:
        Variant name: "turbo" or "base"

    Example:
        >>> variant = detect_zimage_variant("models/Z-Image-Turbo")
        >>> variant
        'turbo'

        >>> variant = detect_zimage_variant("models/Z-Image")
        >>> variant
        'base'
    """
    model_dir = Path(model_path)
    scheduler_config_path = model_dir / "scheduler" / "scheduler_config.json"

    # Try alternative paths if scheduler/ doesn't exist
    if not scheduler_config_path.exists():
        scheduler_config_path = model_dir / "scheduler_config.json"

    if not scheduler_config_path.exists():
        # Fall back to name-based detection
        model_name = model_dir.name.lower()
        if "turbo" in model_name:
            logger.info(f"No scheduler_config.json found, detected 'turbo' from name: {model_name}")
            return "turbo"
        logger.info(f"No scheduler_config.json found, defaulting to 'turbo' variant")
        return "turbo"

    try:
        with open(scheduler_config_path) as f:
            config = json.load(f)

        shift = config.get("shift", 3.0)

        # shift=6.0 indicates base model, shift=3.0 indicates turbo
        if shift >= 5.5:  # Allow some tolerance
            logger.info(f"Detected Z-Image base variant (shift={shift})")
            return "base"
        else:
            logger.info(f"Detected Z-Image turbo variant (shift={shift})")
            return "turbo"

    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to read scheduler_config.json: {e}, defaulting to turbo")
        return "turbo"


def get_variant_defaults(variant: str) -> dict[str, Any]:
    """
    Get default generation parameters for a Z-Image variant.

    Args:
        variant: Variant name ("turbo" or "base")

    Returns:
        Dict with default generation parameters:
        - num_inference_steps: Default diffusion steps
        - guidance_scale: Default CFG scale
        - cfg_normalization: Default CFG normalization
        - negative_prompt: Default negative prompt (None for turbo)
        - shift: Scheduler shift value
        - distilled: Whether model is distilled

    Example:
        >>> defaults = get_variant_defaults("base")
        >>> defaults["num_inference_steps"]
        40
        >>> defaults["guidance_scale"]
        4.0
    """
    variant = variant.lower()

    if variant not in ZIMAGE_VARIANTS:
        available = list(ZIMAGE_VARIANTS.keys())
        raise ValueError(
            f"Unknown Z-Image variant: {variant}. Available: {available}"
        )

    variant_config = ZIMAGE_VARIANTS[variant]

    # Merge defaults with top-level config
    result = variant_config["defaults"].copy()
    result["shift"] = variant_config["shift"]
    result["distilled"] = variant_config["distilled"]

    return result


def get_variant_description(variant: str) -> str:
    """Get human-readable description of a variant."""
    variant = variant.lower()
    if variant not in ZIMAGE_VARIANTS:
        return f"Unknown variant: {variant}"
    return ZIMAGE_VARIANTS[variant]["description"]
