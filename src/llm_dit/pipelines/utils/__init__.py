"""
Pipeline utilities for LTX-2 generation enhancements.

Last Updated: 2026-01-16

This module provides utility functions ported from ComfyUI-LTXVideo and
ComfyUI-KJNodes for enhancing video generation quality and performance:

Enhancement Techniques:
- Latent Normalization: Prevents CFG-induced drift ("overbaking")
- NAG: Normalized Attention Guidance for better CFG quality
- FETA: Feature Temporal Attention for temporal consistency
- TeaCache: Inference speedup via temporal caching (4-10x)
- FFN Chunking: Memory reduction via chunked feedforward
- Audio Normalization: Per-step audio latent normalization

All techniques are pure PyTorch tensor operations that can be enabled
independently or in combination via the EnhancementConfig.
"""

# Latent Normalization
from .latent_norm import (
    statistical_normalize,
    adain_normalize,
    PerStepNormalizer,
    NormalizationConfig,
    # Audio latent normalization
    normalize_audio_latents,
    separate_audio_video_latents,
    recombine_audio_video_latents,
    AudioLatentNormalizer,
    AudioNormalizationConfig,
)

# FETA (Feature Temporal Attention)
from .feta import (
    compute_feta_score,
    compute_feta_score_simple,
    FETAEnhancer,
    FETAConfig,
    create_feta_attention_patch,
)

# NAG (Normalized Attention Guidance)
from .nag import (
    normalized_attention_guidance,
    normalized_attention_guidance_batched,
    NAGEnhancer,
    NAGConfig,
    create_nag_cfg_function,
)

# TeaCache (Temporal Efficient Attention Caching)
from .tea_cache import (
    TeaCache,
    TeaCacheManager,
    TeaCacheConfig,
    compute_relative_l1_distance,
    rescale_distance,
    LTX2_COEFFICIENTS,
)

# FFN Chunking
from .ffn_chunking import (
    chunked_ffn_forward,
    create_chunked_ffn_wrapper,
    ChunkedFFN,
    patch_ffn_chunking,
    unpatch_ffn_chunking,
    FFNChunkingConfig,
    estimate_memory_savings,
)

# Cross-Attention Extraction
from .attention import (
    AttentionExtractor,
    AttentionExtractorHook,
    AttentionMapInfo,
    extract_cross_attention_on_step,
    visualize_attention_heatmap,
)

__all__ = [
    # Latent Normalization
    "statistical_normalize",
    "adain_normalize",
    "PerStepNormalizer",
    "NormalizationConfig",
    # Audio Normalization
    "normalize_audio_latents",
    "separate_audio_video_latents",
    "recombine_audio_video_latents",
    "AudioLatentNormalizer",
    "AudioNormalizationConfig",
    # FETA
    "compute_feta_score",
    "compute_feta_score_simple",
    "FETAEnhancer",
    "FETAConfig",
    "create_feta_attention_patch",
    # NAG
    "normalized_attention_guidance",
    "normalized_attention_guidance_batched",
    "NAGEnhancer",
    "NAGConfig",
    "create_nag_cfg_function",
    # TeaCache
    "TeaCache",
    "TeaCacheManager",
    "TeaCacheConfig",
    "compute_relative_l1_distance",
    "rescale_distance",
    "LTX2_COEFFICIENTS",
    # FFN Chunking
    "chunked_ffn_forward",
    "create_chunked_ffn_wrapper",
    "ChunkedFFN",
    "patch_ffn_chunking",
    "unpatch_ffn_chunking",
    "FFNChunkingConfig",
    "estimate_memory_savings",
    # Cross-Attention Extraction
    "AttentionExtractor",
    "AttentionExtractorHook",
    "AttentionMapInfo",
    "extract_cross_attention_on_step",
    "visualize_attention_heatmap",
]
