"""
Qwen-Image VAE components - 3D causal VAE layers.

This is a simplified port of the DiffSynth-Studio QwenImageVAE implementation,
containing the necessary layers for encoding and decoding single images.

Based on: coderef/DiffSynth-Studio/diffsynth/models/qwen_image_vae.py
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union


class QwenImageCausalConv3d(nn.Conv3d):
    """
    3D causal convolution with proper temporal padding.

    For single images (temporal dim = 1), this behaves like a 2D convolution
    with causal padding in the time dimension.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        # Set up causal padding: (W_left, W_right, H_left, H_right, T_left, T_right)
        # Causal = all temporal padding on the left
        self._padding = (
            self.padding[2], self.padding[2],  # Width
            self.padding[1], self.padding[1],  # Height
            2 * self.padding[0], 0,            # Time (causal)
        )
        self.padding = (0, 0, 0)

    def forward(self, x: torch.Tensor, cache_x: torch.Tensor = None) -> torch.Tensor:
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = nn.functional.pad(x, padding)
        return super().forward(x)


class QwenImageRMSNorm(nn.Module):
    """RMS normalization layer."""

    def __init__(
        self,
        dim: int,
        channel_first: bool = True,
        images: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_dim = 1 if self.channel_first else -1
        return (
            nn.functional.normalize(x, dim=norm_dim)
            * self.scale
            * self.gamma
            + self.bias
        )


class QwenImageResidualBlock(nn.Module):
    """Residual block with RMS normalization and SiLU activation."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        non_linearity: str = "silu",
    ) -> None:
        super().__init__()
        self.nonlinearity = nn.SiLU()

        self.norm1 = QwenImageRMSNorm(in_dim, images=False)
        self.conv1 = QwenImageCausalConv3d(in_dim, out_dim, 3, padding=1)
        self.norm2 = QwenImageRMSNorm(out_dim, images=False)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = QwenImageCausalConv3d(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut = (
            QwenImageCausalConv3d(in_dim, out_dim, 1)
            if in_dim != out_dim
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_shortcut(x)
        x = self.norm1(x)
        x = self.nonlinearity(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.nonlinearity(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x + h


class QwenImageAttentionBlock(nn.Module):
    """Self-attention block for spatial dimensions."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.norm = QwenImageRMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        batch_size, channels, time, height, width = x.size()

        # Reshape for 2D attention
        x = x.permute(0, 2, 1, 3, 4).reshape(batch_size * time, channels, height, width)
        x = self.norm(x)

        # Compute QKV
        qkv = self.to_qkv(x)
        qkv = qkv.reshape(batch_size * time, 1, channels * 3, -1)
        qkv = qkv.permute(0, 1, 3, 2).contiguous()
        q, k, v = qkv.chunk(3, dim=-1)

        # Attention
        x = nn.functional.scaled_dot_product_attention(q, k, v)

        # Project back
        x = x.squeeze(1).permute(0, 2, 1).reshape(batch_size * time, channels, height, width)
        x = self.proj(x)

        # Reshape back to 5D
        x = x.view(batch_size, time, channels, height, width)
        x = x.permute(0, 2, 1, 3, 4)

        return x + identity


class QwenImageUpsample(nn.Upsample):
    """Upsample with dtype preservation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type_as(x)


class QwenImageResample(nn.Module):
    """Resampling module for up/downsampling."""

    def __init__(self, dim: int, mode: str) -> None:
        super().__init__()
        self.dim = dim
        self.mode = mode

        if mode == "upsample2d":
            self.resample = nn.Sequential(
                QwenImageUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim // 2, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                QwenImageUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim // 2, 3, padding=1),
            )
            self.time_conv = QwenImageCausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == "downsample2d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)),
            )
        elif mode == "downsample3d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)),
            )
            self.time_conv = QwenImageCausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        else:
            self.resample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.size()

        # Handle 3D upsampling (temporal)
        if self.mode == "upsample3d" and t > 1:
            x = self.time_conv(x)
            x = x.reshape(b, 2, c, t, h, w)
            x = torch.stack((x[:, 0], x[:, 1]), 3)
            x = x.reshape(b, c, t * 2, h, w)

        # Spatial resampling
        t = x.shape[2]
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.resample(x)
        x = x.view(b, t, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)

        # Handle 3D downsampling (temporal)
        if self.mode == "downsample3d" and x.shape[2] > 1:
            x = self.time_conv(x)

        return x


class QwenImageMidBlock(nn.Module):
    """Middle block with attention and residual blocks."""

    def __init__(
        self,
        dim: int,
        dropout: float = 0.0,
        non_linearity: str = "silu",
        num_layers: int = 1,
    ):
        super().__init__()
        resnets = [QwenImageResidualBlock(dim, dim, dropout, non_linearity)]
        attentions = []
        for _ in range(num_layers):
            attentions.append(QwenImageAttentionBlock(dim))
            resnets.append(QwenImageResidualBlock(dim, dim, dropout, non_linearity))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnets[0](x)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                x = attn(x)
            x = resnet(x)
        return x


class QwenImageEncoder3d(nn.Module):
    """3D encoder for the causal VAE."""

    def __init__(
        self,
        dim: int = 128,
        z_dim: int = 4,
        dim_mult: List[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        attn_scales: List[float] = [],
        temperal_downsample: List[bool] = [True, True, False],
        dropout: float = 0.0,
        non_linearity: str = "silu",
        image_channels: int = 3,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.nonlinearity = nn.SiLU()

        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        self.conv_in = QwenImageCausalConv3d(image_channels, dims[0], 3, padding=1)

        # Downsample blocks
        self.down_blocks = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            for _ in range(num_res_blocks):
                self.down_blocks.append(QwenImageResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    self.down_blocks.append(QwenImageAttentionBlock(out_dim))
                in_dim = out_dim

            if i != len(dim_mult) - 1:
                mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
                self.down_blocks.append(QwenImageResample(out_dim, mode=mode))
                scale /= 2.0

        self.mid_block = QwenImageMidBlock(out_dim, dropout, non_linearity, num_layers=1)
        self.norm_out = QwenImageRMSNorm(out_dim, images=False)
        self.conv_out = QwenImageCausalConv3d(out_dim, z_dim, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.mid_block(x)
        x = self.norm_out(x)
        x = self.nonlinearity(x)
        x = self.conv_out(x)
        return x


class QwenImageUpBlock(nn.Module):
    """Upsampling block for decoder."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        dropout: float = 0.0,
        upsample_mode: Optional[str] = None,
        non_linearity: str = "silu",
    ):
        super().__init__()
        resnets = []
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(QwenImageResidualBlock(current_dim, out_dim, dropout, non_linearity))
            current_dim = out_dim
        self.resnets = nn.ModuleList(resnets)

        self.upsamplers = None
        if upsample_mode is not None:
            self.upsamplers = nn.ModuleList([QwenImageResample(out_dim, mode=upsample_mode)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            x = resnet(x)
        if self.upsamplers is not None:
            x = self.upsamplers[0](x)
        return x


class QwenImageDecoder3d(nn.Module):
    """3D decoder for the causal VAE."""

    def __init__(
        self,
        dim: int = 128,
        z_dim: int = 4,
        dim_mult: List[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        attn_scales: List[float] = [],
        temperal_upsample: List[bool] = [False, True, True],
        dropout: float = 0.0,
        non_linearity: str = "silu",
        image_channels: int = 3,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.nonlinearity = nn.SiLU()

        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2 ** (len(dim_mult) - 2)

        self.conv_in = QwenImageCausalConv3d(z_dim, dims[0], 3, padding=1)
        self.mid_block = QwenImageMidBlock(dims[0], dropout, non_linearity, num_layers=1)

        # Upsample blocks
        self.up_blocks = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i > 0:
                in_dim = in_dim // 2

            upsample_mode = None
            if i != len(dim_mult) - 1:
                upsample_mode = "upsample3d" if temperal_upsample[i] else "upsample2d"

            up_block = QwenImageUpBlock(
                in_dim=in_dim,
                out_dim=out_dim,
                num_res_blocks=num_res_blocks,
                dropout=dropout,
                upsample_mode=upsample_mode,
                non_linearity=non_linearity,
            )
            self.up_blocks.append(up_block)

            if upsample_mode is not None:
                scale *= 2.0

        self.norm_out = QwenImageRMSNorm(out_dim, images=False)
        self.conv_out = QwenImageCausalConv3d(out_dim, image_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        x = self.mid_block(x)
        for up_block in self.up_blocks:
            x = up_block(x)
        x = self.norm_out(x)
        x = self.nonlinearity(x)
        x = self.conv_out(x)
        return x
