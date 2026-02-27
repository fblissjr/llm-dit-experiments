"""
LTX-2 Model Components.

Last Updated: 2026-01-18

Pure PyTorch implementation of the LTX-2 diffusion transformer for video
and audio generation. This module provides granular access to all components
for research experiments.

Architecture Overview:
- 48 transformer blocks with self-attention + cross-attention + FFN
- 32 attention heads with 128-dim head (4096 inner dim)
- 3840-dim text conditioning (Gemma3) projected to 4096
- RoPE position embeddings for 3D video (T, H, W)
- AdaLN-single for timestep conditioning
- Optional audio branch for AV generation

Primary Exports:
    LTX2Transformer: Main transformer model
    LTXModelType: Model variant enum (VideoOnly, AudioVideo, AudioOnly)
    load_ltx2_transformer: Load from official checkpoints
    Modality: Input container for video/audio latents
    VideoEncoder, VideoDecoder: VAE for latent ↔ pixel conversion

Components:
    transformer.py: Main transformer model
    components.py: Modality, AdaLN, FeedForward, etc.
    attention.py: Self/cross-attention with RoPE
    rope.py: Rotary position embeddings
    connectors.py: Text conditioning connectors
    loader.py: Weight loading utilities
    vae/: Video VAE (encoder/decoder) for latent ↔ pixel conversion
"""

from llm_dit.models.ltx2.transformer import (
    LTX2Transformer,
    LTXModelType,
    BasicTransformerBlock,
    TransformerArgs,
    TransformerArgsPreprocessor,
    TransformerConfig,
    PerturbationType,
    Perturbation,
    PerturbationConfig,
    BatchedPerturbationConfig,
    to_velocity,
    to_denoised,
)
from llm_dit.models.ltx2.av_block import BasicAVTransformerBlock
from llm_dit.layers import rms_norm
from llm_dit.models.ltx2.components import (
    Modality,
    AdaLayerNormSingle,
    FeedForward,
    PixArtAlphaTextProjection,
)
from llm_dit.models.ltx2.attention import (
    Attention,
    AttentionFunction,
    AttentionCallable,
    get_available_attention_backends,
    get_default_attention_function,
    is_compile_enabled,
    get_compile_mode,
)
from llm_dit.models.ltx2.rope import (
    LTXRopeType,
    precompute_freqs_cis,
    apply_rotary_emb,
)
from llm_dit.models.ltx2.connectors import (
    LTX2TextConnectors,
    LTX2ConnectorTransformer1d,
    load_ltx2_connectors,
)
from llm_dit.models.ltx2.loader import (
    load_ltx2_transformer,
    load_ltx2_transformer_quantized,
    load_ltx2_transformer_from_fp8,
)
from llm_dit.models.ltx2.constants import (
    # Generation defaults
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    DEFAULT_NUM_FRAMES,
    DEFAULT_FRAME_RATE,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_SEED,
    # Scheduler
    SCHEDULER_BASE_SHIFT,
    SCHEDULER_MAX_SHIFT,
    SCHEDULER_TERMINAL,
    SCHEDULER_STRETCH,
    # VAE
    VAE_TEMPORAL_COMPRESSION,
    VAE_SPATIAL_COMPRESSION,
    VAE_LATENT_CHANNELS,
    # Helpers
    get_reference_config,
    get_quick_test_config,
    calculate_latent_tokens,
)

# VAE components
from llm_dit.models.ltx2.vae import (
    VideoEncoder,
    VideoDecoder,
    TilingConfig,
    SpatialTilingConfig,
    TemporalTilingConfig,
    decode_video,
    patchify,
    unpatchify,
    PerChannelStatistics,
    SpatioTemporalScaleFactors,
    VideoLatentShape,
    load_ltx2_vae_decoder,
)

__all__ = [
    # Main model
    "LTX2Transformer",
    "LTXModelType",
    "load_ltx2_transformer",
    "load_ltx2_transformer_quantized",
    "load_ltx2_transformer_from_fp8",
    # Input types
    "Modality",
    "TransformerArgs",
    "TransformerConfig",
    # Transformer components
    "BasicTransformerBlock",
    "BasicAVTransformerBlock",
    "TransformerArgsPreprocessor",
    # Perturbation model (STG)
    "PerturbationType",
    "Perturbation",
    "PerturbationConfig",
    "BatchedPerturbationConfig",
    "AdaLayerNormSingle",
    "FeedForward",
    "PixArtAlphaTextProjection",
    "rms_norm",
    # Attention
    "Attention",
    "AttentionFunction",
    "AttentionCallable",
    "get_available_attention_backends",
    "get_default_attention_function",
    "is_compile_enabled",
    "get_compile_mode",
    # RoPE
    "LTXRopeType",
    "precompute_freqs_cis",
    "apply_rotary_emb",
    # Connectors
    "LTX2TextConnectors",
    "LTX2ConnectorTransformer1d",
    "load_ltx2_connectors",
    # Utilities
    "to_velocity",
    "to_denoised",
    # VAE
    "VideoEncoder",
    "VideoDecoder",
    "TilingConfig",
    "SpatialTilingConfig",
    "TemporalTilingConfig",
    "decode_video",
    "patchify",
    "unpatchify",
    "PerChannelStatistics",
    "SpatioTemporalScaleFactors",
    "VideoLatentShape",
    "load_ltx2_vae_decoder",
    # Constants
    "DEFAULT_HEIGHT",
    "DEFAULT_WIDTH",
    "DEFAULT_NUM_FRAMES",
    "DEFAULT_FRAME_RATE",
    "DEFAULT_NUM_INFERENCE_STEPS",
    "DEFAULT_GUIDANCE_SCALE",
    "DEFAULT_SEED",
    "SCHEDULER_BASE_SHIFT",
    "SCHEDULER_MAX_SHIFT",
    "SCHEDULER_TERMINAL",
    "SCHEDULER_STRETCH",
    "VAE_TEMPORAL_COMPRESSION",
    "VAE_SPATIAL_COMPRESSION",
    "VAE_LATENT_CHANNELS",
    "get_reference_config",
    "get_quick_test_config",
    "calculate_latent_tokens",
]
