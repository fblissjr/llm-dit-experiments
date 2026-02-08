"""
Z-Image Model Components.

Last Updated: 2026-01-27

Z-Image is the primary image generation model with two variants:
- Turbo: Fast distilled variant (9 steps, CFG baked in, shift=3.0)
- Base: Quality variant (35 steps, full CFG control, shift=6.0)

Both variants share identical architecture but differ in generation parameters.

Primary Exports:
    ZIMAGE_VARIANTS: Dict of variant configurations
    detect_zimage_variant: Auto-detect variant from scheduler_config.json
    get_variant_defaults: Get default generation parameters for variant
"""

from llm_dit.models.zimage.constants import (
    ZIMAGE_VARIANTS,
    detect_zimage_variant,
    get_variant_defaults,
)

__all__ = [
    "ZIMAGE_VARIANTS",
    "detect_zimage_variant",
    "get_variant_defaults",
]
