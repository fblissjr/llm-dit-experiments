"""
LTX-2 Audio VAE Building Blocks - 2D Convolution Components.

Last Updated: 2026-02-26

2D building blocks for the audio VAE decoder and encoder. These operate on
mel spectrogram tensors (B, C, time, freq) and are the audio counterparts
to the video VAE's 3D blocks.

Key components:
    - CausalConv2d: 2D convolution with causal padding along time axis
    - ResnetBlock: 2D residual block with optional timestep conditioning
    - Downsample/Upsample: Spatial up/downsampling with optional causal conv
    - AttnBlock: 2D self-attention block
    - PerChannelStatistics: Latent normalization/denormalization

Ported from: DiffSynth-Studio ltx2_audio_vae (lines 263-871)
License: LTX-2 Community License
Copyright (c) 2025 Lightricks Ltd.
"""

from enum import Enum
from typing import Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..vae.normalization import NormType, PixelNorm, build_normalization_layer


class AttentionType(Enum):
    """Attention mechanism type for AttnBlock."""
    VANILLA = "vanilla"
    LINEAR = "linear"
    NONE = "none"


class CausalityAxis(Enum):
    """Axis along which to apply causal padding in CausalConv2d.

    For audio spectrograms, HEIGHT corresponds to the time axis and
    WIDTH corresponds to the frequency axis. Causal along HEIGHT means
    the output at time t only depends on inputs at time <= t.
    """
    NONE = None
    WIDTH = "width"
    HEIGHT = "height"
    WIDTH_COMPATIBILITY = "width-compatibility"


class CausalConv2d(nn.Module):
    """2D convolution with causal (asymmetric) padding.

    Ensures output at time t depends only on inputs at time <= t by
    applying all padding before the current position on the causal axis.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int = 1,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        causality_axis: CausalityAxis = CausalityAxis.HEIGHT,
    ) -> None:
        super().__init__()
        self.causality_axis = causality_axis

        kernel_size = nn.modules.utils._pair(kernel_size)
        dilation = nn.modules.utils._pair(dilation)

        pad_h = (kernel_size[0] - 1) * dilation[0]
        pad_w = (kernel_size[1] - 1) * dilation[1]

        # Padding tuple: (pad_left, pad_right, pad_top, pad_bottom)
        match self.causality_axis:
            case CausalityAxis.NONE:
                self.padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
            case CausalityAxis.WIDTH | CausalityAxis.WIDTH_COMPATIBILITY:
                self.padding = (pad_w, 0, pad_h // 2, pad_h - pad_h // 2)
            case CausalityAxis.HEIGHT:
                self.padding = (pad_w // 2, pad_w - pad_w // 2, pad_h, 0)
            case _:
                raise ValueError(f"Invalid causality_axis: {causality_axis}")

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=0, dilation=dilation,
            groups=groups, bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, self.padding)
        return self.conv(x)


def make_conv2d(
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple[int, int],
    stride: int = 1,
    padding: tuple[int, int, int, int] | None = None,
    dilation: int = 1,
    groups: int = 1,
    bias: bool = True,
    causality_axis: CausalityAxis | None = None,
) -> nn.Module:
    """Create a 2D convolution, causal or standard depending on causality_axis."""
    if causality_axis is not None:
        return CausalConv2d(
            in_channels, out_channels, kernel_size, stride,
            dilation, groups, bias, causality_axis,
        )
    if padding is None:
        padding = kernel_size // 2 if isinstance(kernel_size, int) else tuple(k // 2 for k in kernel_size)
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, stride,
        padding, dilation, groups, bias,
    )


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class AttnBlock(nn.Module):
    """2D self-attention block for audio spectrograms."""

    def __init__(self, in_channels: int, norm_type: NormType = NormType.GROUP) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.norm = build_normalization_layer(in_channels, normtype=norm_type)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w).permute(0, 2, 1).contiguous()
        k = k.reshape(b, c, h * w).contiguous()
        w_ = torch.bmm(q, k) * (int(c) ** (-0.5))
        w_ = F.softmax(w_, dim=2)

        v = v.reshape(b, c, h * w).contiguous()
        w_ = w_.permute(0, 2, 1).contiguous()
        h_ = torch.bmm(v, w_).reshape(b, c, h, w).contiguous()
        h_ = self.proj_out(h_)

        return x + h_


def make_attn(
    in_channels: int,
    attn_type: AttentionType = AttentionType.VANILLA,
    norm_type: NormType = NormType.GROUP,
) -> nn.Module:
    """Factory for attention blocks."""
    match attn_type:
        case AttentionType.VANILLA:
            return AttnBlock(in_channels, norm_type=norm_type)
        case AttentionType.NONE:
            return nn.Identity()
        case AttentionType.LINEAR:
            raise NotImplementedError(f"Attention type {attn_type.value} not supported")
        case _:
            raise ValueError(f"Unknown attention type: {attn_type}")


# ---------------------------------------------------------------------------
# Residual block
# ---------------------------------------------------------------------------

class ResnetBlock(nn.Module):
    """2D residual block with optional timestep conditioning.

    Uses causal convolutions along the specified axis (typically HEIGHT
    for time-causal audio processing).
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int | None = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: int = 512,
        norm_type: NormType = NormType.GROUP,
        causality_axis: CausalityAxis = CausalityAxis.HEIGHT,
    ) -> None:
        super().__init__()
        self.causality_axis = causality_axis

        if self.causality_axis != CausalityAxis.NONE and norm_type == NormType.GROUP:
            raise ValueError("Causal ResnetBlock with GroupNorm is not supported.")

        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = build_normalization_layer(in_channels, normtype=norm_type)
        self.non_linearity = nn.SiLU()
        self.conv1 = make_conv2d(
            in_channels, out_channels, kernel_size=3, stride=1,
            causality_axis=causality_axis,
        )
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = build_normalization_layer(out_channels, normtype=norm_type)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = make_conv2d(
            out_channels, out_channels, kernel_size=3, stride=1,
            causality_axis=causality_axis,
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = make_conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1,
                    causality_axis=causality_axis,
                )
            else:
                self.nin_shortcut = make_conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1,
                    causality_axis=causality_axis,
                )

    def forward(self, x: torch.Tensor, temb: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm1(x)
        h = self.non_linearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(self.non_linearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = self.non_linearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x) if self.use_conv_shortcut else self.nin_shortcut(x)

        return x + h


# ---------------------------------------------------------------------------
# Down/Upsampling
# ---------------------------------------------------------------------------

class Downsample(nn.Module):
    """2D downsampling with strided convolution or average pooling."""

    def __init__(
        self,
        in_channels: int,
        with_conv: bool,
        causality_axis: CausalityAxis = CausalityAxis.WIDTH,
    ) -> None:
        super().__init__()
        self.with_conv = with_conv
        self.causality_axis = causality_axis

        if self.causality_axis != CausalityAxis.NONE and not self.with_conv:
            raise ValueError("Causality only supported with with_conv=True.")

        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.with_conv:
            match self.causality_axis:
                case CausalityAxis.NONE:
                    pad = (0, 1, 0, 1)
                case CausalityAxis.WIDTH:
                    pad = (2, 0, 0, 1)
                case CausalityAxis.HEIGHT:
                    pad = (0, 1, 2, 0)
                case CausalityAxis.WIDTH_COMPATIBILITY:
                    pad = (1, 0, 0, 1)
                case _:
                    raise ValueError(f"Invalid causality_axis: {self.causality_axis}")

            x = F.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class Upsample(nn.Module):
    """2D upsampling with nearest-neighbor interpolation and optional causal conv."""

    def __init__(
        self,
        in_channels: int,
        with_conv: bool,
        causality_axis: CausalityAxis = CausalityAxis.HEIGHT,
    ) -> None:
        super().__init__()
        self.with_conv = with_conv
        self.causality_axis = causality_axis
        if self.with_conv:
            self.conv = make_conv2d(
                in_channels, in_channels, kernel_size=3, stride=1,
                causality_axis=causality_axis,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
            # Drop FIRST element on causal axis to undo encoder's padding
            match self.causality_axis:
                case CausalityAxis.NONE | CausalityAxis.WIDTH_COMPATIBILITY:
                    pass
                case CausalityAxis.HEIGHT:
                    x = x[:, :, 1:, :]
                case CausalityAxis.WIDTH:
                    x = x[:, :, :, 1:]
                case _:
                    raise ValueError(f"Invalid causality_axis: {self.causality_axis}")
        return x


# ---------------------------------------------------------------------------
# Path builders
# ---------------------------------------------------------------------------

def build_downsampling_path(
    *,
    ch: int,
    ch_mult: Tuple[int, ...],
    num_resolutions: int,
    num_res_blocks: int,
    resolution: int,
    temb_channels: int,
    dropout: float,
    norm_type: NormType,
    causality_axis: CausalityAxis,
    attn_type: AttentionType,
    attn_resolutions: Set[int],
    resamp_with_conv: bool,
) -> tuple[nn.ModuleList, int]:
    """Build the encoder downsampling path."""
    down_modules = nn.ModuleList()
    curr_res = resolution
    in_ch_mult = (1, *tuple(ch_mult))
    block_in = ch

    for i_level in range(num_resolutions):
        block = nn.ModuleList()
        attn = nn.ModuleList()
        block_in = ch * in_ch_mult[i_level]
        block_out = ch * ch_mult[i_level]

        for _ in range(num_res_blocks):
            block.append(
                ResnetBlock(
                    in_channels=block_in, out_channels=block_out,
                    temb_channels=temb_channels, dropout=dropout,
                    norm_type=norm_type, causality_axis=causality_axis,
                )
            )
            block_in = block_out
            if curr_res in attn_resolutions:
                attn.append(make_attn(block_in, attn_type=attn_type, norm_type=norm_type))

        down = nn.Module()
        down.block = block
        down.attn = attn
        if i_level != num_resolutions - 1:
            down.downsample = Downsample(block_in, resamp_with_conv, causality_axis=causality_axis)
            curr_res = curr_res // 2
        down_modules.append(down)

    return down_modules, block_in


def build_upsampling_path(
    *,
    ch: int,
    ch_mult: Tuple[int, ...],
    num_resolutions: int,
    num_res_blocks: int,
    resolution: int,
    temb_channels: int,
    dropout: float,
    norm_type: NormType,
    causality_axis: CausalityAxis,
    attn_type: AttentionType,
    attn_resolutions: Set[int],
    resamp_with_conv: bool,
    initial_block_channels: int,
) -> tuple[nn.ModuleList, int]:
    """Build the decoder upsampling path."""
    up_modules = nn.ModuleList()
    block_in = initial_block_channels
    curr_res = resolution // (2 ** (num_resolutions - 1))

    for level in reversed(range(num_resolutions)):
        stage = nn.Module()
        stage.block = nn.ModuleList()
        stage.attn = nn.ModuleList()
        block_out = ch * ch_mult[level]

        for _ in range(num_res_blocks + 1):
            stage.block.append(
                ResnetBlock(
                    in_channels=block_in, out_channels=block_out,
                    temb_channels=temb_channels, dropout=dropout,
                    norm_type=norm_type, causality_axis=causality_axis,
                )
            )
            block_in = block_out
            if curr_res in attn_resolutions:
                stage.attn.append(make_attn(block_in, attn_type=attn_type, norm_type=norm_type))

        if level != 0:
            stage.upsample = Upsample(block_in, resamp_with_conv, causality_axis=causality_axis)
            curr_res *= 2

        up_modules.insert(0, stage)

    return up_modules, block_in


# ---------------------------------------------------------------------------
# Mid block
# ---------------------------------------------------------------------------

def build_mid_block(
    channels: int,
    temb_channels: int,
    dropout: float,
    norm_type: NormType,
    causality_axis: CausalityAxis,
    attn_type: AttentionType,
    add_attention: bool,
) -> nn.Module:
    """Build middle block with two ResNet blocks and optional attention."""
    mid = nn.Module()
    mid.block_1 = ResnetBlock(
        in_channels=channels, out_channels=channels,
        temb_channels=temb_channels, dropout=dropout,
        norm_type=norm_type, causality_axis=causality_axis,
    )
    mid.attn_1 = (
        make_attn(channels, attn_type=attn_type, norm_type=norm_type)
        if add_attention
        else nn.Identity()
    )
    mid.block_2 = ResnetBlock(
        in_channels=channels, out_channels=channels,
        temb_channels=temb_channels, dropout=dropout,
        norm_type=norm_type, causality_axis=causality_axis,
    )
    return mid


def run_mid_block(mid: nn.Module, features: torch.Tensor) -> torch.Tensor:
    """Run features through the middle block."""
    features = mid.block_1(features, temb=None)
    features = mid.attn_1(features)
    return mid.block_2(features, temb=None)


# ---------------------------------------------------------------------------
# Per-channel statistics (audio-specific)
# ---------------------------------------------------------------------------

class PerChannelStatistics(nn.Module):
    """Per-channel statistics for audio latent normalization/denormalization.

    Operates on patchified latent tensors (B, T, C*F) where the last
    dimension has size latent_channels. Loaded from the audio VAE checkpoint.

    Unlike the video VAE's 5D version, this operates on the flattened
    patch dimension directly.
    """

    def __init__(self, latent_channels: int = 128) -> None:
        super().__init__()
        self.register_buffer("std-of-means", torch.empty(latent_channels))
        self.register_buffer("mean-of-means", torch.empty(latent_channels))

    def un_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize patchified latents back to original scale."""
        return (x * self.get_buffer("std-of-means").to(x)) + self.get_buffer("mean-of-means").to(x)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize patchified latents to zero mean and unit variance."""
        return (x - self.get_buffer("mean-of-means").to(x)) / self.get_buffer("std-of-means").to(x)
