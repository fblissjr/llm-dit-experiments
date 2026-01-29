"""
Diffusers-based pipeline implementations.

This module preserves the original diffusers-based Z-Image pipeline for
backward compatibility. The main pipeline in the parent module has been
ported to pure PyTorch for better control and performance.

Usage:
    # Original diffusers-based pipeline
    from llm_dit.pipelines.diffusers import ZImagePipeline

    # New pure PyTorch pipeline (recommended)
    from llm_dit.pipelines import ZImagePipeline
"""

from .z_image import ZImagePipeline

__all__ = ["ZImagePipeline"]
