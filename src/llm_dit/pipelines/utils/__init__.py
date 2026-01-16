"""
Pipeline utilities for LTX-2 generation enhancements.

Last Updated: 2026-01-16

This module provides utility functions ported from ComfyUI-LTXVideo:
- Latent normalization (prevents CFG overbaking)
- FETA (Feature Temporal Attention) enhancement
"""

from .latent_norm import (
    statistical_normalize,
    adain_normalize,
    PerStepNormalizer,
    NormalizationConfig,
)
from .feta import (
    compute_feta_score,
    FETAConfig,
)

__all__ = [
    "statistical_normalize",
    "adain_normalize",
    "PerStepNormalizer",
    "NormalizationConfig",
    "compute_feta_score",
    "FETAConfig",
]
