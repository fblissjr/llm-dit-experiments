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

Components:
    transformer.py: Main transformer model
    components.py: Modality, AdaLN, FeedForward, etc.
    attention.py: Self/cross-attention with RoPE
    rope.py: Rotary position embeddings
    connectors.py: Text conditioning connectors
    loader.py: Weight loading utilities
"""

from llm_dit.models.ltx2.transformer import (
    LTX2Transformer,
    LTXModelType,
    BasicTransformerBlock,
    TransformerArgs,
    TransformerArgsPreprocessor,
    TransformerConfig,
    to_velocity,
    to_denoised,
)
from llm_dit.models.ltx2.components import (
    Modality,
    AdaLayerNormSingle,
    FeedForward,
    PixArtAlphaTextProjection,
    rms_norm,
)
from llm_dit.models.ltx2.attention import (
    Attention,
    AttentionFunction,
    AttentionCallable,
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
)

__all__ = [
    # Main model
    "LTX2Transformer",
    "LTXModelType",
    "load_ltx2_transformer",
    # Input types
    "Modality",
    "TransformerArgs",
    "TransformerConfig",
    # Transformer components
    "BasicTransformerBlock",
    "TransformerArgsPreprocessor",
    "AdaLayerNormSingle",
    "FeedForward",
    "PixArtAlphaTextProjection",
    "rms_norm",
    # Attention
    "Attention",
    "AttentionFunction",
    "AttentionCallable",
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
]
