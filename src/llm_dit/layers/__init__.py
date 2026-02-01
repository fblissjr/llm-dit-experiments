"""
L4 Component Library - Shared atomic components across all models.

Last Updated: 2026-02-01

This package provides canonical implementations of common neural network
components used across multiple model families (LTX-2, FLUX.2, Z-Image, etc.).

Design Goals:
- Single source of truth for each component type
- Configurable to match model-specific requirements
- Backward compatible with existing weight loading
- Well-tested with numerical equivalence guarantees

Modules:
- normalization: RMSNorm, LayerNorm variants
- feedforward: FeedForward, SwiGLU, GeGLU variants
- (future) attention: Attention primitives
"""

from llm_dit.layers.normalization import RMSNorm, rms_norm, T5LayerNorm
from llm_dit.layers.feedforward import (
    FeedForward,
    FFNType,
    LTX2_FFN_PRESET,
    ZIMAGE_FFN_PRESET,
    WAN_T5_FFN_PRESET,
    CONNECTOR_FFN_PRESET,
)

__all__ = [
    # Normalization
    "RMSNorm",
    "rms_norm",
    "T5LayerNorm",
    # FeedForward
    "FeedForward",
    "FFNType",
    "LTX2_FFN_PRESET",
    "ZIMAGE_FFN_PRESET",
    "WAN_T5_FFN_PRESET",
    "CONNECTOR_FFN_PRESET",
]
