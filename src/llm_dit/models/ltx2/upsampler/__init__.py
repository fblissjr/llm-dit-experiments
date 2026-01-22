"""
Latent upsampler model components for LTX-2.

Last Updated: 2026-01-18

This module provides spatial and temporal upsampling for VAE latents,
supporting integer and fractional scale factors.

Public API:
    - LatentUpsampler: Main upsampler model
    - LatentUpsamplerConfigurator: Create from config dict
    - upsample_video: Helper for VAE-normalized upsampling
    - PixelShuffleND: N-dimensional pixel shuffle
    - ResBlock: Residual block with GroupNorm
    - BlurDownsample: Anti-aliased downsampling
    - SpatialRationalResampler: Fractional spatial scaling
"""

from llm_dit.models.ltx2.upsampler.blur_downsample import BlurDownsample
from llm_dit.models.ltx2.upsampler.model import LatentUpsampler, upsample_video
from llm_dit.models.ltx2.upsampler.model_configurator import LatentUpsamplerConfigurator
from llm_dit.models.ltx2.upsampler.pixel_shuffle import PixelShuffleND
from llm_dit.models.ltx2.upsampler.res_block import ResBlock
from llm_dit.models.ltx2.upsampler.spatial_rational_resampler import SpatialRationalResampler

__all__ = [
    "LatentUpsampler",
    "LatentUpsamplerConfigurator",
    "upsample_video",
    "PixelShuffleND",
    "ResBlock",
    "BlurDownsample",
    "SpatialRationalResampler",
]
