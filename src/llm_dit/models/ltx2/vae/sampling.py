"""
LTX-2 VAE Sampling Layers.

Last Updated: 2026-01-18

Downsampling and upsampling layers for the Video VAE using space-to-depth
and depth-to-space operations.

Ported from: ltx_core.model.video_vae.sampling
Original source: https://github.com/Lightricks/LTX-2
License: LTX-2 Community License
Copyright (c) 2025 Lightricks Ltd.
"""

import math
from typing import Tuple, Union

import torch
from einops import rearrange
from torch import nn

from .convolution import make_conv_nd
from .enums import PaddingModeType


class SpaceToDepthDownsample(nn.Module):
    """
    Downsampling using space-to-depth with residual connection.

    This layer reduces spatial/temporal dimensions by moving elements into
    channels, then applies a convolution for mixing. The residual connection
    helps preserve information.
    """

    def __init__(
        self,
        dims: Union[int, Tuple[int, int]],
        in_channels: int,
        out_channels: int,
        stride: Tuple[int, int, int],
        spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
    ):
        """
        Args:
            dims: Convolution dimensions.
            in_channels: Input channels.
            out_channels: Output channels.
            stride: Downsampling factor for (D, H, W).
            spatial_padding_mode: Padding mode for convolutions.
        """
        super().__init__()
        self.stride = stride
        self.group_size = in_channels * math.prod(stride) // out_channels
        self.conv = make_conv_nd(
            dims=dims,
            in_channels=in_channels,
            out_channels=out_channels // math.prod(stride),
            kernel_size=3,
            stride=1,
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )

    def forward(
        self,
        x: torch.Tensor,
        causal: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, C, D, H, W)
            causal: Whether to use causal convolution.

        Returns:
            Downsampled tensor.
        """
        # Pad temporal dimension if needed
        if self.stride[0] == 2:
            x = torch.cat([x[:, :, :1, :, :], x], dim=2)  # duplicate first frame

        # Skip connection: rearrange then average groups
        x_in = rearrange(
            x,
            "b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w",
            p1=self.stride[0],
            p2=self.stride[1],
            p3=self.stride[2],
        )
        x_in = rearrange(x_in, "b (c g) d h w -> b c g d h w", g=self.group_size)
        x_in = x_in.mean(dim=2)

        # Convolution path
        x = self.conv(x, causal=causal)
        x = rearrange(
            x,
            "b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w",
            p1=self.stride[0],
            p2=self.stride[1],
            p3=self.stride[2],
        )

        # Residual addition
        x = x + x_in

        return x


class DepthToSpaceUpsample(nn.Module):
    """
    Upsampling using depth-to-space with optional residual connection.

    This layer increases spatial/temporal dimensions by moving channels into
    space, after applying a convolution for mixing.
    """

    def __init__(
        self,
        dims: Union[int, Tuple[int, int]],
        in_channels: int,
        stride: Tuple[int, int, int],
        residual: bool = False,
        out_channels_reduction_factor: int = 1,
        spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
    ):
        """
        Args:
            dims: Convolution dimensions.
            in_channels: Input channels.
            stride: Upsampling factor for (D, H, W).
            residual: Whether to add residual connection.
            out_channels_reduction_factor: Factor to reduce output channels.
            spatial_padding_mode: Padding mode for convolutions.
        """
        super().__init__()
        self.stride = stride
        self.out_channels = math.prod(stride) * in_channels // out_channels_reduction_factor
        self.conv = make_conv_nd(
            dims=dims,
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.residual = residual
        self.out_channels_reduction_factor = out_channels_reduction_factor

    def forward(
        self,
        x: torch.Tensor,
        causal: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, C, D, H, W)
            causal: Whether to use causal convolution.

        Returns:
            Upsampled tensor.
        """
        x_in: torch.Tensor | None = None
        if self.residual:
            # Reshape and duplicate input for residual
            x_in = rearrange(
                x,
                "b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)",
                p1=self.stride[0],
                p2=self.stride[1],
                p3=self.stride[2],
            )
            num_repeat = math.prod(self.stride) // self.out_channels_reduction_factor
            x_in = x_in.repeat(1, num_repeat, 1, 1, 1)
            if self.stride[0] == 2:
                x_in = x_in[:, :, 1:, :, :]  # Remove first frame padding

        # Convolution then depth-to-space
        x = self.conv(x, causal=causal)
        # DEBUG: Trace DepthToSpaceUpsample conv output
        if x.shape[1] in [4096, 2048, 1024]:  # Only trace upsample conv outputs
            print(f"[TRACE D2S] After conv ({x.shape[1]} ch): mean={x.mean():.4f}, std={x.std():.4f}")
        x = rearrange(
            x,
            "b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)",
            p1=self.stride[0],
            p2=self.stride[1],
            p3=self.stride[2],
        )

        # Remove first frame if temporal upsampling
        if self.stride[0] == 2:
            x = x[:, :, 1:, :, :]

        # DEBUG: After depth-to-space rearrange
        if x.shape[1] in [512, 256, 128]:
            print(f"[TRACE D2S] After rearrange ({x.shape[1]} ch): mean={x.mean():.4f}, std={x.std():.4f}")

        # Add residual if enabled
        if self.residual:
            assert x_in is not None  # Always set when residual=True
            # DEBUG: Before and after residual
            if x.shape[1] in [512, 256, 128]:
                print(f"[TRACE D2S] Residual x_in: mean={x_in.mean():.4f}, std={x_in.std():.4f}")
            x = x + x_in
            if x.shape[1] in [512, 256, 128]:
                print(f"[TRACE D2S] After residual: mean={x.mean():.4f}, std={x.std():.4f}")

        return x
