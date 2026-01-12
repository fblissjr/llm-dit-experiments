"""
Wan Video VAE for video encoding/decoding.

Last Updated: 2026-01-12

This implements the official Wan VAE architecture to match checkpoint weights.
NOTE: ONLY TESTED AND BUILT FOR Wan-AI/Wan2.1-T2V-1.3B
- IMPORTANT: NEED TO TEST FOR OTHER VARIANTS OF WAN 2.1 and WAN 2.2

The VAE uses:
- 3D causal convolutions for temporal consistency
- Downsampling: 8x spatial, 4x temporal
- Latent dimension: 16 channels
- Mean/std normalization for stable training

Weight file: Wan2.1_VAE.safetensors
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

CACHE_T = 2


class CausalConv3d(nn.Conv3d):
    """Causal 3D convolution with temporal padding for causality."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Store original padding and set to zero (will pad manually)
        self._padding = (
            self.padding[2],
            self.padding[2],  # Width
            self.padding[1],
            self.padding[1],  # Height
            2 * self.padding[0],
            0,  # Temporal: causal
        )
        self.padding = (0, 0, 0)

    def forward(self, x: torch.Tensor, cache_x: torch.Tensor = None) -> torch.Tensor:
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        return super().forward(x)


class RMS_norm(nn.Module):
    """RMS normalization with gamma parameter (matches official weights)."""

    def __init__(
        self, dim: int, channel_first: bool = True, images: bool = False, bias: bool = False
    ):
        super().__init__()
        # images=False gives 4D shape [C,1,1,1] for video, images=True gives 3D [C,1,1] for 2D
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = (
            F.normalize(x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma
            + self.bias
        )
        return x.to(dtype)


class Upsample(nn.Upsample):
    """Upsample with bfloat16 fix."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type_as(x)


class Resample(nn.Module):
    """Spatial/temporal resampling module."""

    def __init__(self, dim: int, mode: str):
        assert mode in ("none", "upsample2d", "upsample3d", "downsample2d", "downsample3d")
        super().__init__()
        self.dim = dim
        self.mode = mode

        if mode in ("upsample2d", "upsample3d"):
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim // 2, 3, padding=1),
            )
            if mode == "upsample3d":
                self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode in ("downsample2d", "downsample3d"):
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)),
            )
            if mode == "downsample3d":
                self.time_conv = CausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        else:
            self.resample = nn.Identity()

    def forward(self, x: torch.Tensor, feat_cache: dict = None) -> torch.Tensor:
        b, c, t, h, w = x.size()

        if self.mode == "downsample3d":
            if feat_cache is not None:
                key = id(self.time_conv)
                if key not in feat_cache:
                    feat_cache[key] = x.clone()
                else:
                    cache_x = x[:, :, -1:, :, :].clone()
                    x = self.time_conv(torch.cat([feat_cache[key][:, :, -1:, :, :], x], 2))
                    feat_cache[key] = cache_x
            else:
                x = self.time_conv(x)

        t = x.shape[2]
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.resample(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=t)
        return x


class ResidualBlock(nn.Module):
    """Residual block with RMS norm and causal conv."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Main path: norm -> silu -> conv -> norm -> silu -> dropout -> conv
        self.residual = nn.Sequential(
            RMS_norm(in_dim),  # [0]
            nn.SiLU(),  # [1]
            CausalConv3d(in_dim, out_dim, 3, padding=1),  # [2]
            RMS_norm(out_dim),  # [3]
            nn.SiLU(),  # [4]
            nn.Dropout(dropout),  # [5]
            CausalConv3d(out_dim, out_dim, 3, padding=1),  # [6]
        )

        # Shortcut if dimensions change
        if in_dim != out_dim:
            self.shortcut = CausalConv3d(in_dim, out_dim, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, feat_cache: dict = None) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.residual):
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                key = id(layer)
                cache_x = h[:, :, -CACHE_T:, :, :].clone() if h.shape[2] >= CACHE_T else h.clone()
                h = layer(h, feat_cache.get(key))
                feat_cache[key] = cache_x
            else:
                h = layer(h)

        if isinstance(self.shortcut, CausalConv3d) and feat_cache is not None:
            key = id(self.shortcut)
            cache_x = x[:, :, -CACHE_T:, :, :].clone() if x.shape[2] >= CACHE_T else x.clone()
            x = self.shortcut(x, feat_cache.get(key))
            feat_cache[key] = cache_x
        else:
            x = self.shortcut(x)

        return x + h


class AttentionBlock(nn.Module):
    """Self-attention block for VAE (uses 2D conv applied per-frame)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # Attention norm uses images=True (3D gamma shape) per checkpoint
        self.norm = RMS_norm(dim, images=True)
        # Uses 2D conv (matches checkpoint shape [C*3, C, 1, 1])
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)  # Named 'proj' in checkpoint

    def forward(self, x: torch.Tensor, feat_cache: dict = None) -> torch.Tensor:
        b, c, t, h, w = x.shape
        residual = x

        # Reshape to 4D, apply norm with 3D gamma [C,1,1], process with 2D conv
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.norm(x)  # norm gamma is [C,1,1] for 4D input
        qkv = self.to_qkv(x)
        x = rearrange(qkv, "(b t) c h w -> b c t h w", t=t)

        q, k, v = x.chunk(3, dim=1)

        # Reshape for attention (across all spatial-temporal positions)
        q = rearrange(q, "b c t h w -> b (t h w) c")
        k = rearrange(k, "b c t h w -> b (t h w) c")
        v = rearrange(v, "b c t h w -> b (t h w) c")

        # Scaled dot-product attention
        scale = c**-0.5
        attn = torch.bmm(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, v)

        # Apply output projection per-frame
        out = rearrange(out, "b (t h w) c -> (b t) c h w", t=t, h=h, w=w)
        out = self.proj(out)
        out = rearrange(out, "(b t) c h w -> b c t h w", t=t)

        return residual + out


class Encoder3d(nn.Module):
    """3D video encoder."""

    def __init__(
        self,
        dim: int = 96,
        z_dim: int = 16,
        dim_mult: Optional[List[int]] = None,
        num_res_blocks: int = 2,
        attn_scales: Optional[List[float]] = None,
        temporal_downsample: Optional[List[bool]] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if dim_mult is None:
            dim_mult = [1, 2, 4, 4]
        if attn_scales is None:
            attn_scales = []
        if temporal_downsample is None:
            temporal_downsample = [False, True, True]

        self.dim = dim
        self.z_dim = z_dim

        # Dimensions at each level
        dims = [dim * u for u in [1] + dim_mult]

        # Input convolution
        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)

        # Downsample blocks
        downsamples = []
        scale = 1.0
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # Residual blocks
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # Downsample (except last level)
            if i != len(dim_mult) - 1:
                mode = "downsample3d" if temporal_downsample[i] else "downsample2d"
                downsamples.append(Resample(out_dim, mode=mode))
                scale /= 2.0

        self.downsamples = nn.ModuleList(downsamples)

        # Middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(dims[-1], dims[-1], dropout),
            AttentionBlock(dims[-1]),
            ResidualBlock(dims[-1], dims[-1], dropout),
        )

        # Output
        self.head = nn.Sequential(
            RMS_norm(dims[-1]),
            nn.SiLU(),
            CausalConv3d(dims[-1], z_dim * 2, 3, padding=1),  # mu and log_var
        )

    def forward(self, x: torch.Tensor, feat_cache: dict = None) -> torch.Tensor:
        x = self.conv1(x)

        for block in self.downsamples:
            x = block(x, feat_cache)

        for block in self.middle:
            x = block(x, feat_cache) if hasattr(block, "forward") else block(x)

        x = self.head[0](x)  # norm
        x = self.head[1](x)  # silu
        x = self.head[2](x, feat_cache.get(id(self.head[2])) if feat_cache else None)

        return x


class Decoder3d(nn.Module):
    """3D video decoder."""

    def __init__(
        self,
        dim: int = 96,
        z_dim: int = 16,
        dim_mult: Optional[List[int]] = None,
        num_res_blocks: int = 2,
        attn_scales: Optional[List[float]] = None,
        temporal_upsample: Optional[List[bool]] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if dim_mult is None:
            dim_mult = [1, 2, 4, 4]
        if attn_scales is None:
            attn_scales = []
        if temporal_upsample is None:
            temporal_upsample = [True, True, False]

        self.dim = dim
        self.z_dim = z_dim

        # Dimensions (reversed for decoder)
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]

        # Input convolution
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # Middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout),
            AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout),
        )

        # Upsample blocks
        upsamples = []
        scale = 1.0 / 2 ** (len(dim_mult) - 2)
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # Adjust input dim after upsample
            if i in [1, 2, 3]:
                in_dim = in_dim // 2

            # Residual blocks (one extra compared to encoder)
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # Upsample (except last level)
            if i != len(dim_mult) - 1:
                mode = "upsample3d" if temporal_upsample[i] else "upsample2d"
                upsamples.append(Resample(out_dim, mode=mode))
                scale *= 2.0

        self.upsamples = nn.ModuleList(upsamples)

        # Output head
        self.head = nn.Sequential(
            RMS_norm(dims[-1]),
            nn.SiLU(),
            CausalConv3d(dims[-1], 3, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, feat_cache: dict = None) -> torch.Tensor:
        x = self.conv1(x, feat_cache.get(id(self.conv1)) if feat_cache else None)

        for block in self.middle:
            x = block(x, feat_cache) if hasattr(block, "forward") else block(x)

        for block in self.upsamples:
            x = block(x, feat_cache)

        x = self.head[0](x)  # norm
        x = self.head[1](x)  # silu
        x = self.head[2](x, feat_cache.get(id(self.head[2])) if feat_cache else None)

        return x


class VideoVAE(nn.Module):
    """
    Video VAE matching official Wan2.1 weights.

    Weight keys:
    - encoder.conv1.*, encoder.downsamples.*, encoder.middle.*, encoder.head.*
    - decoder.conv1.*, decoder.middle.*, decoder.upsamples.*, decoder.head.*
    """

    def __init__(
        self,
        dim: int = 96,
        z_dim: int = 16,
        dim_mult: Optional[List[int]] = None,
        num_res_blocks: int = 2,
        attn_scales: Optional[List[float]] = None,
        temporal_downsample: Optional[List[bool]] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if dim_mult is None:
            dim_mult = [1, 2, 4, 4]
        if attn_scales is None:
            attn_scales = []
        if temporal_downsample is None:
            temporal_downsample = [False, True, True]

        self.dim = dim
        self.z_dim = z_dim

        temporal_upsample = temporal_downsample[::-1]

        self.encoder = Encoder3d(
            dim, z_dim, dim_mult, num_res_blocks, attn_scales, temporal_downsample, dropout
        )
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(
            dim, z_dim, dim_mult, num_res_blocks, attn_scales, temporal_upsample, dropout
        )

    def encode(self, x: torch.Tensor, scale: Tuple[float, float] = None) -> torch.Tensor:
        """Encode video to latents."""
        h = self.encoder(x)
        h = self.conv1(h)
        mu, log_var = h.chunk(2, dim=1)

        if scale is not None:
            mean, inv_std = scale
            if isinstance(mean, torch.Tensor):
                mu = (mu - mean.view(1, self.z_dim, 1, 1, 1)) * inv_std.view(1, self.z_dim, 1, 1, 1)
            else:
                mu = (mu - mean) * inv_std

        return mu

    def decode(self, z: torch.Tensor, scale: Tuple[float, float] = None) -> torch.Tensor:
        """Decode latents to video."""
        if scale is not None:
            mean, inv_std = scale
            if isinstance(mean, torch.Tensor):
                z = z / inv_std.view(1, self.z_dim, 1, 1, 1) + mean.view(1, self.z_dim, 1, 1, 1)
            else:
                z = z / inv_std + mean

        z = self.conv2(z)
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode then decode (for training)."""
        h = self.encoder(x)
        h = self.conv1(h)
        mu, log_var = h.chunk(2, dim=1)
        z = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)
        z = self.conv2(z)
        x_recon = self.decoder(z)
        return x_recon, mu, log_var


class WanVAE(nn.Module):
    """
    High-level wrapper for Wan Video VAE with scaling.

    Provides simple encode/decode interface with mean/std normalization.
    """

    # Normalization constants from official implementation
    MEAN = torch.tensor(
        [
            -0.7571,
            -0.7089,
            -0.9113,
            0.1075,
            -0.1745,
            0.9653,
            -0.1517,
            1.5508,
            0.4134,
            -0.0715,
            0.5517,
            -0.3632,
            -0.1922,
            -0.9497,
            0.2503,
            -0.2921,
        ]
    )
    STD = torch.tensor(
        [
            2.8184,
            1.4541,
            2.3275,
            2.6558,
            1.2196,
            2.6780,
            2.4944,
            2.3565,
            2.1346,
            2.3635,
            1.6543,
            1.7366,
            2.5187,
            2.0825,
            2.7897,
            2.1521,
        ]
    )

    def __init__(self, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.dtype = dtype
        self.model = VideoVAE()

        # Precompute inverse std for efficiency
        self.register_buffer("mean", self.MEAN)
        self.register_buffer("inv_std", 1.0 / self.STD)

    @property
    def scale(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return (self.mean, self.inv_std)

    def encode(self, videos: torch.Tensor) -> torch.Tensor:
        """
        Encode videos to latent space.

        Args:
            videos: [B, 3, T, H, W] RGB videos in [-1, 1]

        Returns:
            [B, z_dim, T', H', W'] latent codes
        """
        with torch.amp.autocast(device_type=videos.device.type, dtype=self.dtype):
            return self.model.encode(videos, self.scale)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latents to videos.

        Args:
            latents: [B, z_dim, T', H', W'] latent codes

        Returns:
            [B, 3, T, H, W] RGB videos clamped to [-1, 1]
        """
        with torch.amp.autocast(device_type=latents.device.type, dtype=self.dtype):
            videos = self.model.decode(latents, self.scale)
            return videos.clamp(-1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode then decode (for training)."""
        z = self.encode(x)
        return self.decode(z)
