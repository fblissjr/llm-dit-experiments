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
    scale_embeddings: Scale embeddings to match target statistics (global)
    normalize_per_dimension: Per-dimension normalization using Qwen3-4B reference
    normalize_hybrid: Blend of global and per-dimension normalization
    get_reference_stats: Get precomputed Qwen3-4B statistics
    mask_outlier_dimensions: Mask dimensions with extreme std ratios
    get_outlier_dimensions: List outlier dimensions above threshold

Key parameters:
    - alpha: Interpolation ratio (0.0=text, 1.0=VL, recommended: 0.3)
    - hidden_layer: Which layer to extract (-8 recommended for VL, -2 for text)
    - text_tokens_only: Use only text token positions (recommended: True)
    - normalization_mode: "global", "per_dim", or "hybrid"

Key Finding (2025-12-12):
    VL text tokens have 0.999 correlation with Qwen3-4B per-dimension statistics.
    VL image tokens have only 0.737 correlation with extreme outliers (600x+ ratio).
    For image tokens, use normalization_mode="per_dim" for best results.

See experiments/qwen3_vl/README.md for detailed documentation.
"""

from .blending import (
    blend_adain,
    blend_adain_per_dim,
    blend_attention_weighted,
    blend_embeddings,
    blend_interpolate,
    blend_per_token,
    blend_style_only,
    blend_with_style_delta,
    compute_style_delta,
    create_graduated_alpha,
    get_outlier_dimensions,
    get_reference_stats,
    mask_outlier_dimensions,
    normalize_hybrid,
    normalize_per_dimension,
    scale_embeddings,
)
from .qwen3_vl import VLEmbeddingExtractor, estimate_token_count

__all__ = [
    "VLEmbeddingExtractor",
    "estimate_token_count",
    "blend_embeddings",
    "blend_interpolate",
    "scale_embeddings",
    "normalize_per_dimension",
    "normalize_hybrid",
    "get_reference_stats",
    "get_outlier_dimensions",
    "mask_outlier_dimensions",
    "blend_style_only",
    "blend_per_token",
    "blend_attention_weighted",
    "create_graduated_alpha",
    "compute_style_delta",
    "blend_with_style_delta",
    "blend_adain",
    "blend_adain_per_dim",
]
