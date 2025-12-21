"""
Guidance techniques for diffusion models.

This module provides pluggable guidance methods for improving generation quality:
- SkipLayerGuidance: Improves structure/anatomy by comparing with layer-skipped predictions
- ClassifierFreeGuidance: Standard CFG (not needed for Z-Image which has CFG=0 baked in)

Usage:
    from llm_dit.guidance import SkipLayerGuidance

    slg = SkipLayerGuidance(
        skip_layers=[10, 15, 20],
        guidance_scale=2.8,
    )

    # In denoising loop:
    pred_cond = model(latents, t, prompt_embeds)
    with slg.skip_layers_context(model):
        pred_skip = model(latents, t, prompt_embeds)
    guided_pred = slg.guide(pred_cond, pred_skip)
"""

from llm_dit.guidance.skip_layer import (
    SkipLayerGuidance,
    LayerSkipConfig,
    apply_layer_skip,
    remove_layer_skip,
)

__all__ = [
    "SkipLayerGuidance",
    "LayerSkipConfig",
    "apply_layer_skip",
    "remove_layer_skip",
]
