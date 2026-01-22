"""
Rational spatial resampler for fractional scale factors.

Last Updated: 2026-01-18

SpatialRationalResampler enables fractional spatial scaling (0.75x, 1.5x, 2.0x, 4.0x)
by combining PixelShuffle upsampling with anti-aliased blur downsampling.
"""

from typing import Tuple

import torch
from einops import rearrange

from llm_dit.models.ltx2.upsampler.blur_downsample import BlurDownsample
from llm_dit.models.ltx2.upsampler.pixel_shuffle import PixelShuffleND


def _rational_for_scale(scale: float) -> Tuple[int, int]:
    """
    Map floating-point scale to rational numerator/denominator pair.

    Args:
        scale: Desired scale factor (0.75, 1.5, 2.0, or 4.0)

    Returns:
        Tuple of (upsample_factor, downsample_factor) that achieves the scale.

    Raises:
        ValueError: If scale is not in the supported set.
    """
    mapping = {
        0.75: (3, 4),  # Upsample 3x, downsample 4x → 0.75x
        1.5: (3, 2),  # Upsample 3x, downsample 2x → 1.5x
        2.0: (2, 1),  # Upsample 2x, no downsample → 2.0x
        4.0: (4, 1),  # Upsample 4x, no downsample → 4.0x
    }
    if float(scale) not in mapping:
        raise ValueError(f"Unsupported scale {scale}. Choose from {list(mapping.keys())}")
    return mapping[float(scale)]


class SpatialRationalResampler(torch.nn.Module):
    """
    Fully-learned rational spatial scaling.

    Achieves fractional scales by:
    1. Learnable Conv2d to expand channels for PixelShuffle
    2. PixelShuffle upsampling by 'num' factor
    3. Anti-aliased BlurDownsample by 'den' factor

    Operates on H,W only. For 5D input (B,C,F,H,W), processes per-frame
    with temporal axis unchanged.

    Args:
        mid_channels: Number of intermediate channels (before PixelShuffle expansion)
        scale: Spatial scaling factor. Supported values:
            - 0.75: Reduce spatial size to 3/4
            - 1.5: Increase spatial size by 3/2
            - 2.0: Double spatial size
            - 4.0: Quadruple spatial size
    """

    def __init__(self, mid_channels: int, scale: float):
        super().__init__()
        self.scale = float(scale)
        self.num, self.den = _rational_for_scale(self.scale)

        # Conv expands channels for PixelShuffle: C → C * num^2 (for 2D)
        self.conv = torch.nn.Conv2d(mid_channels, (self.num**2) * mid_channels, kernel_size=3, padding=1)
        self.pixel_shuffle = PixelShuffleND(2, upscale_factors=(self.num, self.num, 1))
        self.blur_down = BlurDownsample(dims=2, stride=self.den)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, f, h, w = x.shape
        # Process per-frame
        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.blur_down(x)
        x = rearrange(x, "(b f) c h w -> b c f h w", b=b, f=f)
        return x
