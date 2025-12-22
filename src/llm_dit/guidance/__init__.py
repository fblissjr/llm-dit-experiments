"""
Guidance techniques for diffusion models.

This module provides pluggable guidance methods for improving generation quality:
- SkipLayerGuidance: Improves structure/anatomy by comparing with layer-skipped predictions
- FMTTGuidance: Test-time reward optimization using flow maps (FMTT)
- ClassifierFreeGuidance: Standard CFG (not needed for Z-Image which has CFG=0 baked in)

Usage:
    from llm_dit.guidance import SkipLayerGuidance, FMTTGuidance

    # Skip Layer Guidance
    slg = SkipLayerGuidance(
        skip_layers=[10, 15, 20],
        guidance_scale=2.8,
    )

    # FMTT (Flow Map Trajectory Tilting)
    from llm_dit.rewards import DifferentiableSigLIP
    fmtt = FMTTGuidance(
        vae=pipe.vae,
        reward_fn=DifferentiableSigLIP(device="cuda"),
        guidance_scale=1.0,
    )
"""

from llm_dit.guidance.skip_layer import (
    SkipLayerGuidance,
    LayerSkipConfig,
    apply_layer_skip,
    remove_layer_skip,
)
from llm_dit.guidance.fmtt import (
    FMTTGuidance,
    flow_map_direct,
    create_fmtt_guidance,
)

__all__ = [
    # Skip Layer Guidance
    "SkipLayerGuidance",
    "LayerSkipConfig",
    "apply_layer_skip",
    "remove_layer_skip",
    # FMTT
    "FMTTGuidance",
    "flow_map_direct",
    "create_fmtt_guidance",
]
