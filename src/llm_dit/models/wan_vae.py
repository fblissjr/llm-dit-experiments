"""
Wan Video VAE for video encoding/decoding.

Last Updated: 2026-01-11

This implements the Wan VAE architecture from:
https://github.com/Phantom-video/HuMo (humo/models/wan_modules/vae.py)

The VAE uses:
- 3D causal convolutions for temporal consistency
- Downsampling: 8x spatial, 4x temporal
- Latent dimension: 16 channels
- Mean/std normalization for stable training

Weight file: Wan2.1_VAE.pth from Wan-AI/Wan2.1-T2V-1.3B
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv3d(nn.Module):
    """
    3D causal convolution with padding for temporal causality.

    Ensures outputs at time t only depend on inputs at times <= t.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size, kernel_size),
            stride=(stride, stride, stride) if isinstance(stride, int) else stride,
            padding=(kernel_size - 1, padding, padding),
        )
        self.causal_padding = kernel_size - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, H, W]
        x = self.conv(x)
        # Remove future frames from temporal dimension
        if self.causal_padding > 0:
            x = x[:, :, :-self.causal_padding, :, :]
        return x


class GroupNorm32(nn.GroupNorm):
    """Group normalization with 32 groups, float32 computation."""

    def __init__(self, num_channels: int):
        super().__init__(32, num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).to(x.dtype)


class ResBlock3D(nn.Module):
    """3D residual block with causal convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = GroupNorm32(in_channels)
        self.conv1 = CausalConv3d(in_channels, out_channels, 3, 1, 1)

        self.norm2 = GroupNorm32(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = CausalConv3d(out_channels, out_channels, 3, 1, 1)

        if in_channels != out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.skip(x)


class Downsample3D(nn.Module):
    """3D downsampling with strided convolution."""

    def __init__(
        self,
        channels: int,
        temporal_stride: int = 1,
        spatial_stride: int = 2,
    ):
        super().__init__()
        stride = (temporal_stride, spatial_stride, spatial_stride)
        self.conv = nn.Conv3d(channels, channels, 3, stride, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample3D(nn.Module):
    """3D upsampling with interpolation and convolution."""

    def __init__(
        self,
        channels: int,
        temporal_scale: int = 1,
        spatial_scale: int = 2,
    ):
        super().__init__()
        self.temporal_scale = temporal_scale
        self.spatial_scale = spatial_scale
        self.conv = CausalConv3d(channels, channels, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        x = F.interpolate(
            x,
            size=(T * self.temporal_scale, H * self.spatial_scale, W * self.spatial_scale),
            mode="nearest",
        )
        return self.conv(x)


class SelfAttention3D(nn.Module):
    """3D self-attention for video features."""

    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.norm = GroupNorm32(channels)
        self.qkv = nn.Conv3d(channels, channels * 3, 1)
        self.proj = nn.Conv3d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        h = self.norm(x)

        qkv = self.qkv(h)
        qkv = qkv.reshape(B, 3, self.num_heads, C // self.num_heads, T * H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        # Attention
        scale = (C // self.num_heads) ** -0.5
        attn = torch.einsum("bhdn,bhdm->bhnm", q, k) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum("bhnm,bhdm->bhdn", attn, v)

        out = out.reshape(B, C, T, H, W)
        out = self.proj(out)

        return x + out


class Encoder3D(nn.Module):
    """3D video encoder."""

    def __init__(
        self,
        in_channels: int = 3,
        z_dim: int = 16,
        dims: Tuple[int, ...] = (128, 256, 512, 512),
        num_res_blocks: int = 2,
        temporal_strides: Tuple[int, ...] = (1, 2, 2, 1),
        spatial_strides: Tuple[int, ...] = (2, 2, 2, 1),
        use_attention: Tuple[bool, ...] = (False, False, True, True),
    ):
        super().__init__()

        # Initial convolution
        self.conv_in = CausalConv3d(in_channels, dims[0], 3, 1, 1)

        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        in_dim = dims[0]
        for i, out_dim in enumerate(dims):
            block = nn.ModuleList()

            # Residual blocks
            for _ in range(num_res_blocks):
                block.append(ResBlock3D(in_dim, out_dim))
                in_dim = out_dim

            # Attention
            if use_attention[i]:
                block.append(SelfAttention3D(out_dim))

            # Downsample
            if temporal_strides[i] > 1 or spatial_strides[i] > 1:
                block.append(Downsample3D(out_dim, temporal_strides[i], spatial_strides[i]))

            self.down_blocks.append(block)

        # Middle blocks
        self.mid_block1 = ResBlock3D(dims[-1], dims[-1])
        self.mid_attn = SelfAttention3D(dims[-1])
        self.mid_block2 = ResBlock3D(dims[-1], dims[-1])

        # Output
        self.norm_out = GroupNorm32(dims[-1])
        self.conv_out = CausalConv3d(dims[-1], z_dim * 2, 3, 1, 1)  # mu and log_var

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, T, H, W]
        h = self.conv_in(x)

        # Downsampling
        for block in self.down_blocks:
            for layer in block:
                h = layer(h)

        # Middle
        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)

        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)

        return h


class Decoder3D(nn.Module):
    """3D video decoder."""

    def __init__(
        self,
        out_channels: int = 3,
        z_dim: int = 16,
        dims: Tuple[int, ...] = (512, 512, 256, 128),
        num_res_blocks: int = 2,
        temporal_strides: Tuple[int, ...] = (1, 2, 2, 1),
        spatial_strides: Tuple[int, ...] = (1, 2, 2, 2),
        use_attention: Tuple[bool, ...] = (True, True, False, False),
    ):
        super().__init__()

        # Input convolution
        self.conv_in = CausalConv3d(z_dim, dims[0], 3, 1, 1)

        # Middle blocks
        self.mid_block1 = ResBlock3D(dims[0], dims[0])
        self.mid_attn = SelfAttention3D(dims[0])
        self.mid_block2 = ResBlock3D(dims[0], dims[0])

        # Upsampling blocks
        self.up_blocks = nn.ModuleList()
        in_dim = dims[0]
        for i, out_dim in enumerate(dims):
            block = nn.ModuleList()

            # Residual blocks
            for _ in range(num_res_blocks + 1):
                block.append(ResBlock3D(in_dim, out_dim))
                in_dim = out_dim

            # Attention
            if use_attention[i]:
                block.append(SelfAttention3D(out_dim))

            # Upsample
            if temporal_strides[i] > 1 or spatial_strides[i] > 1:
                block.append(Upsample3D(out_dim, temporal_strides[i], spatial_strides[i]))

            self.up_blocks.append(block)

        # Output
        self.norm_out = GroupNorm32(dims[-1])
        self.conv_out = CausalConv3d(dims[-1], out_channels, 3, 1, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, z_dim, T', H', W']
        h = self.conv_in(z)

        # Middle
        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)

        # Upsampling
        for block in self.up_blocks:
            for layer in block:
                h = layer(h)

        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)

        return h


class WanVAE_(nn.Module):
    """Inner VAE model combining encoder and decoder."""

    def __init__(self, z_dim: int = 16):
        super().__init__()
        self.z_dim = z_dim
        self.encoder = Encoder3D(z_dim=z_dim)
        self.decoder = Decoder3D(z_dim=z_dim)

    def encode(self, x: torch.Tensor, scale: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Encode video to latent space."""
        h = self.encoder(x)
        mu, log_var = h.chunk(2, dim=1)

        # Apply scaling
        mean, inv_std = scale
        mu = (mu - mean.view(1, -1, 1, 1, 1)) * inv_std.view(1, -1, 1, 1, 1)

        return mu  # Return mean for deterministic encoding

    def decode(self, z: torch.Tensor, scale: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Decode latent to video."""
        # Reverse scaling
        mean, inv_std = scale
        z = z / inv_std.view(1, -1, 1, 1, 1) + mean.view(1, -1, 1, 1, 1)

        x = self.decoder(z)
        return x


class WanVAE(nn.Module):
    """
    Wan Video VAE wrapper with scaling normalization.

    Handles loading pretrained weights and provides encode/decode interface
    with proper scaling for stable latent distributions.

    Args:
        z_dim: Latent dimension (default: 16)
        dtype: Model dtype
        device: Target device
    """

    # Pre-computed scaling factors from HuMo
    # These normalize the latent distribution
    DEFAULT_MEAN = torch.tensor([
        -0.9550, -0.2344, -0.5676, 0.0602, -0.2411, 0.0965, 0.0291, -0.1245,
        1.5527, 0.8210, -0.4052, 0.3337, 0.0856, 0.5285, 0.5874, 0.3952
    ])
    DEFAULT_STD = torch.tensor([
        3.2765, 2.0246, 2.6274, 2.4534, 2.6318, 2.0892, 2.0189, 2.2684,
        2.4907, 2.3623, 2.0694, 2.0502, 2.0760, 2.0819, 1.8831, 1.1145
    ])

    def __init__(
        self,
        z_dim: int = 16,
        dtype: torch.dtype = torch.float32,
        device: str = "cuda",
    ):
        super().__init__()
        self.z_dim = z_dim
        self.dtype = dtype

        # Inner model
        self.model = WanVAE_(z_dim=z_dim)

        # Scaling parameters
        self.register_buffer("mean", self.DEFAULT_MEAN)
        self.register_buffer("inv_std", 1.0 / self.DEFAULT_STD)

    @property
    def scale(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get scaling parameters."""
        return (self.mean, self.inv_std)

    def encode(self, videos: torch.Tensor) -> torch.Tensor:
        """
        Encode videos to latent space.

        Args:
            videos: [B, 3, T, H, W] RGB videos in [-1, 1] range

        Returns:
            [B, z_dim, T', H', W'] latent codes
        """
        with torch.cuda.amp.autocast(dtype=self.dtype):
            return self.model.encode(videos, self.scale)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latents to videos.

        Args:
            latents: [B, z_dim, T', H', W'] latent codes

        Returns:
            [B, 3, T, H, W] RGB videos clamped to [-1, 1]
        """
        with torch.cuda.amp.autocast(dtype=self.dtype):
            videos = self.model.decode(latents, self.scale)
            return videos.clamp(-1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode then decode (for training)."""
        z = self.encode(x)
        return self.decode(z)
