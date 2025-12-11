"""
Qwen3-VL Vision Conditioning for Z-Image.

This module provides tools for using Qwen3-VL vision embeddings to condition
Z-Image generation. Key discovery: Qwen3-VL's text model hidden states
(after processing an image) are compatible with Z-Image because both use
Qwen3-4B architecture (hidden_size=2560).

Classes:
    VLEmbeddingExtractor: Extract vision-conditioned embeddings from Qwen3-VL

Functions:
    blend_embeddings: Interpolate between VL and text embeddings
    scale_embeddings: Scale embeddings to match target statistics

Key parameters:
    - alpha: Interpolation ratio (0.0=text, 1.0=VL, recommended: 0.3)
    - hidden_layer: Which layer to extract (-2 recommended)
    - image_tokens_only: Use only image tokens vs full sequence
    - scale_to_text: Scale VL embeddings to match text statistics

See experiments/qwen3_vl/README.md for detailed documentation.
"""

from .blending import (
    blend_attention_weighted,
    blend_embeddings,
    blend_per_token,
    blend_style_only,
    create_graduated_alpha,
    scale_embeddings,
)
from .qwen3_vl import VLEmbeddingExtractor, estimate_token_count

__all__ = [
    "VLEmbeddingExtractor",
    "estimate_token_count",
    "blend_embeddings",
    "scale_embeddings",
    "blend_style_only",
    "blend_per_token",
    "blend_attention_weighted",
    "create_graduated_alpha",
]
