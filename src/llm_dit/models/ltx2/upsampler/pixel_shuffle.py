"""
N-dimensional pixel shuffle operation for latent upsampling.

Last Updated: 2026-01-18

PixelShuffleND generalizes torch.nn.PixelShuffle to 1D (temporal), 2D (spatial),
and 3D (spatiotemporal) upsampling using einops rearrange patterns.
"""

import torch
from einops import rearrange


class PixelShuffleND(torch.nn.Module):
    """
    N-dimensional pixel shuffle operation for upsampling tensors.

    Args:
        dims: Number of dimensions to apply pixel shuffle to.
            - 1: Temporal (frames only)
            - 2: Spatial (height and width)
            - 3: Spatiotemporal (depth, height, width)
        upscale_factors: Upscaling factors for each dimension.
            For dims=1, only the first value is used.
            For dims=2, the first two values are used.
            For dims=3, all three values are used.

    The input tensor is rearranged so that the channel dimension is split into
    smaller channels and upscaling factors, and the upscaling factors are moved
    into the corresponding spatial/temporal dimensions.
    """

    def __init__(self, dims: int, upscale_factors: tuple[int, int, int] = (2, 2, 2)):
        super().__init__()
        assert dims in [1, 2, 3], "dims must be 1, 2, or 3"
        self.dims = dims
        self.upscale_factors = upscale_factors

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dims == 3:
            # Input: (B, C*p1*p2*p3, D, H, W) -> Output: (B, C, D*p1, H*p2, W*p3)
            return rearrange(
                x,
                "b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)",
                p1=self.upscale_factors[0],
                p2=self.upscale_factors[1],
                p3=self.upscale_factors[2],
            )
        elif self.dims == 2:
            # Input: (B, C*p1*p2, H, W) -> Output: (B, C, H*p1, W*p2)
            return rearrange(
                x,
                "b (c p1 p2) h w -> b c (h p1) (w p2)",
                p1=self.upscale_factors[0],
                p2=self.upscale_factors[1],
            )
        elif self.dims == 1:
            # Input: (B, C*p1, F, H, W) -> Output: (B, C, F*p1, H, W)
            return rearrange(
                x,
                "b (c p1) f h w -> b c (f p1) h w",
                p1=self.upscale_factors[0],
            )
        else:
            raise ValueError(f"Unsupported dims: {self.dims}")
