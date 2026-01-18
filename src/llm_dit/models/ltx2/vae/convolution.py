"""
LTX-2 VAE Convolution Layers.

Last Updated: 2026-01-18

Custom convolution layers for the Video VAE, including causal 3D convolutions
and dual (factorized) 3D convolutions.

Ported from: ltx_core.model.video_vae.convolution
Original source: https://github.com/Lightricks/LTX-2
License: LTX-2 Community License
Copyright (c) 2025 Lightricks Ltd.
"""

from typing import Tuple, Union

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from .enums import PaddingModeType


def make_conv_nd(
    dims: Union[int, Tuple[int, int]],
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    bias: bool = True,
    causal: bool = False,
    spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
    temporal_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
) -> nn.Module:
    """
    Factory function for creating N-dimensional convolution layers.

    Args:
        dims: Convolution dimensions. 2 for 2D, 3 for 3D, (2,1) for factorized.
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution.
        padding: Zero-padding added to both sides of the input.
        dilation: Spacing between kernel elements.
        groups: Number of blocked connections from input to output channels.
        bias: If True, adds a learnable bias to the output.
        causal: If True, uses causal convolution (only depends on past frames).
        spatial_padding_mode: Padding mode for spatial dimensions.
        temporal_padding_mode: Padding mode for temporal dimension.

    Returns:
        Convolution module.
    """
    if not (spatial_padding_mode == temporal_padding_mode or causal):
        raise NotImplementedError("spatial and temporal padding modes must be equal")

    if dims == 2:
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=spatial_padding_mode.value,
        )
    elif dims == 3:
        if causal:
            return CausalConv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                groups=groups,
                bias=bias,
                spatial_padding_mode=spatial_padding_mode,
            )
        return nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=spatial_padding_mode.value,
        )
    elif dims == (2, 1):
        return DualConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            padding_mode=spatial_padding_mode.value,
        )
    else:
        raise ValueError(f"unsupported dimensions: {dims}")


def make_linear_nd(
    dims: int,
    in_channels: int,
    out_channels: int,
    bias: bool = True,
) -> nn.Module:
    """
    Create a 1x1 convolution (linear projection in conv form).

    Args:
        dims: Number of dimensions (2 or 3).
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        bias: If True, adds a learnable bias.

    Returns:
        1x1 convolution module.
    """
    if dims == 2:
        return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias)
    elif dims in (3, (2, 1)):
        return nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias)
    else:
        raise ValueError(f"unsupported dimensions: {dims}")


class DualConv3d(nn.Module):
    """
    Factorized 3D convolution: spatial 2D followed by temporal 1D.

    This decomposition is more efficient than a full 3D convolution and allows
    different handling of spatial vs temporal dimensions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ) -> None:
        super(DualConv3d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding_mode = padding_mode

        # Ensure parameters are tuples of length 3
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if kernel_size == (1, 1, 1):
            raise ValueError("kernel_size must be greater than 1. Use make_linear_nd instead.")
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)

        self.groups = groups
        self.bias = bias

        # Intermediate channels: larger of in/out
        intermediate_channels = out_channels if in_channels < out_channels else in_channels

        # First convolution: spatial (H, W)
        self.weight1 = nn.Parameter(
            torch.Tensor(
                intermediate_channels,
                in_channels // groups,
                1,
                kernel_size[1],
                kernel_size[2],
            )
        )
        self.stride1 = (1, stride[1], stride[2])
        self.padding1 = (0, padding[1], padding[2])
        self.dilation1 = (1, dilation[1], dilation[2])
        if bias:
            self.bias1 = nn.Parameter(torch.Tensor(intermediate_channels))
        else:
            self.register_parameter("bias1", None)

        # Second convolution: temporal (D)
        self.weight2 = nn.Parameter(
            torch.Tensor(out_channels, intermediate_channels // groups, kernel_size[0], 1, 1)
        )
        self.stride2 = (stride[0], 1, 1)
        self.padding2 = (padding[0], 0, 0)
        self.dilation2 = (dilation[0], 1, 1)
        if bias:
            self.bias2 = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias2", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights using Kaiming uniform initialization."""
        nn.init.kaiming_uniform_(self.weight1, a=torch.sqrt(torch.tensor(5.0)))
        nn.init.kaiming_uniform_(self.weight2, a=torch.sqrt(torch.tensor(5.0)))
        if self.bias:
            fan_in1, _ = nn.init._calculate_fan_in_and_fan_out(self.weight1)
            bound1 = 1 / torch.sqrt(torch.tensor(fan_in1, dtype=torch.float32))
            nn.init.uniform_(self.bias1, -bound1.item(), bound1.item())
            fan_in2, _ = nn.init._calculate_fan_in_and_fan_out(self.weight2)
            bound2 = 1 / torch.sqrt(torch.tensor(fan_in2, dtype=torch.float32))
            nn.init.uniform_(self.bias2, -bound2.item(), bound2.item())

    def forward(
        self,
        x: torch.Tensor,
        use_conv3d: bool = False,
        skip_time_conv: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, C, D, H, W)
            use_conv3d: Use 3D convolution (slower but exact). Default False uses 2D+1D.
            skip_time_conv: Skip temporal convolution (for debugging).

        Returns:
            Output tensor.
        """
        if use_conv3d:
            return self.forward_with_3d(x=x, skip_time_conv=skip_time_conv)
        else:
            return self.forward_with_2d(x=x, skip_time_conv=skip_time_conv)

    def forward_with_3d(self, x: torch.Tensor, skip_time_conv: bool = False) -> torch.Tensor:
        """Forward using 3D convolutions."""
        x = F.conv3d(x, self.weight1, self.bias1, self.stride1, self.padding1, self.dilation1, self.groups)
        if skip_time_conv:
            return x
        x = F.conv3d(x, self.weight2, self.bias2, self.stride2, self.padding2, self.dilation2, self.groups)
        return x

    def forward_with_2d(self, x: torch.Tensor, skip_time_conv: bool = False) -> torch.Tensor:
        """Forward using factorized 2D spatial + 1D temporal convolutions."""
        b, _, _, h, w = x.shape

        # First: 2D spatial convolution
        x = rearrange(x, "b c d h w -> (b d) c h w")
        weight1 = self.weight1.squeeze(2)
        stride1 = (self.stride1[1], self.stride1[2])
        padding1 = (self.padding1[1], self.padding1[2])
        dilation1 = (self.dilation1[1], self.dilation1[2])
        x = F.conv2d(x, weight1, self.bias1, stride1, padding1, dilation1, self.groups)

        _, _, h, w = x.shape

        if skip_time_conv:
            x = rearrange(x, "(b d) c h w -> b c d h w", b=b)
            return x

        # Second: 1D temporal convolution
        x = rearrange(x, "(b d) c h w -> (b h w) c d", b=b)
        weight2 = self.weight2.squeeze(-1).squeeze(-1)
        x = F.conv1d(x, weight2, self.bias2, self.stride2[0], self.padding2[0], self.dilation2[0], self.groups)
        x = rearrange(x, "(b h w) c d -> b c d h w", b=b, h=h, w=w)

        return x

    @property
    def weight(self) -> torch.Tensor:
        """Return the temporal weight (for compatibility)."""
        return self.weight2


class CausalConv3d(nn.Module):
    """
    Causal 3D convolution that only depends on past and current frames.

    Implements causality by padding the temporal dimension at the start
    (replicating the first frame) rather than symmetrically.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: Union[int, Tuple[int, ...]] = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        kernel_size = (kernel_size, kernel_size, kernel_size)
        self.time_kernel_size = kernel_size[0]

        dilation = (dilation, 1, 1)

        # Spatial padding (symmetric)
        height_pad = kernel_size[1] // 2
        width_pad = kernel_size[2] // 2
        padding = (0, height_pad, width_pad)  # No temporal padding in conv

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            padding_mode=spatial_padding_mode.value,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, C, D, H, W)
            causal: If True, use causal padding. If False, use symmetric padding.

        Returns:
            Output tensor.
        """
        if causal:
            # Causal: pad by repeating first frame
            first_frame_pad = x[:, :, :1, :, :].repeat((1, 1, self.time_kernel_size - 1, 1, 1))
            x = torch.concatenate((first_frame_pad, x), dim=2)
        else:
            # Non-causal: symmetric padding with first/last frame
            first_frame_pad = x[:, :, :1, :, :].repeat((1, 1, (self.time_kernel_size - 1) // 2, 1, 1))
            last_frame_pad = x[:, :, -1:, :, :].repeat((1, 1, (self.time_kernel_size - 1) // 2, 1, 1))
            x = torch.concatenate((first_frame_pad, x, last_frame_pad), dim=2)

        x = self.conv(x)
        return x

    @property
    def weight(self) -> torch.Tensor:
        """Return the convolution weight."""
        return self.conv.weight
