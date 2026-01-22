"""
LTX-2 VAE Type Definitions.

Last Updated: 2026-01-18

Named tuples and type definitions for video VAE.

Ported from: ltx_core.types
Original source: https://github.com/Lightricks/LTX-2
License: LTX-2 Community License
Copyright (c) 2025 Lightricks Ltd.
"""

from typing import NamedTuple

import torch


class SpatioTemporalScaleFactors(NamedTuple):
    """
    Describes the spatiotemporal downscaling between decoded video space and
    the corresponding VAE latent grid.

    For LTX-2:
    - time: 8 (33 frames -> 5 latent frames)
    - height: 32 (512 pixels -> 16 latent)
    - width: 32 (768 pixels -> 24 latent)
    """
    time: int
    width: int
    height: int

    @classmethod
    def default(cls) -> "SpatioTemporalScaleFactors":
        """Default LTX-2 scale factors."""
        return cls(time=8, width=32, height=32)


VIDEO_SCALE_FACTORS = SpatioTemporalScaleFactors.default()


class VideoLatentShape(NamedTuple):
    """
    Shape of the tensor representing video in VAE latent space.

    The latent representation is a 5D tensor with dimensions ordered as
    (batch, channels, frames, height, width). Spatial and temporal dimensions
    are downscaled relative to pixel space according to the VAE's scale factors.
    """
    batch: int
    channels: int
    frames: int
    height: int
    width: int

    def to_torch_shape(self) -> torch.Size:
        """Convert to PyTorch tensor shape."""
        return torch.Size([self.batch, self.channels, self.frames, self.height, self.width])

    @staticmethod
    def from_torch_shape(shape: torch.Size) -> "VideoLatentShape":
        """Create from PyTorch tensor shape."""
        return VideoLatentShape(
            batch=shape[0],
            channels=shape[1],
            frames=shape[2],
            height=shape[3],
            width=shape[4],
        )

    def mask_shape(self) -> "VideoLatentShape":
        """Return shape for a single-channel mask."""
        return self._replace(channels=1)

    def upscale(self, factors: SpatioTemporalScaleFactors) -> "VideoLatentShape":
        """
        Scale latent dimensions to video dimensions.

        For LTX-2: frames' = 1 + (frames - 1) * 8, height/width *= 32
        """
        return VideoLatentShape(
            batch=self.batch,
            channels=3,  # RGB output
            frames=1 + (self.frames - 1) * factors.time,
            height=self.height * factors.height,
            width=self.width * factors.width,
        )
