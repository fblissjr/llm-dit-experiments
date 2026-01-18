"""
LTX-2 VAE Enumerations.

Last Updated: 2026-01-18

Enums for configuring VAE components.

Ported from: ltx_core.model.video_vae.enums
Original source: https://github.com/Lightricks/LTX-2
License: LTX-2 Community License
Copyright (c) 2025 Lightricks Ltd.
"""

from enum import Enum


class NormLayerType(Enum):
    """Normalization layer types for VAE blocks."""
    GROUP_NORM = "group_norm"
    PIXEL_NORM = "pixel_norm"


class LogVarianceType(Enum):
    """Log variance modes for VAE encoder output."""
    PER_CHANNEL = "per_channel"
    UNIFORM = "uniform"
    CONSTANT = "constant"
    NONE = "none"


class PaddingModeType(Enum):
    """Padding modes for convolutions."""
    ZEROS = "zeros"
    REFLECT = "reflect"
    REPLICATE = "replicate"
    CIRCULAR = "circular"
