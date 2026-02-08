"""
Flux VAE Encoder for Z-Image.

Last updated: 2026-01-29

Implements the 16-channel VAE encoder used by Z-Image.
Architecture based on DiffSynth-Studio flux_vae.py reference.

The encoder takes RGB images and produces 16-channel latents:
- Input: (B, 3, H, W) images in [-1, 1] range
- Output: (B, 16, H/8, W/8) latents
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..constants import FluxVAEConfig
from .decoder import Attention, ResnetBlock, VAEAttentionBlock, TileWorker


class DownSampler(nn.Module):
    """2x downsampling layer."""

    def __init__(self, channels: int, padding: int = 1, extra_padding: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=padding)
        self.extra_padding = extra_padding

    def forward(
        self,
        hidden_states: torch.Tensor,
        time_emb: Optional[torch.Tensor],
        text_emb: Optional[torch.Tensor],
        res_stack: Optional[torch.Tensor],
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.extra_padding:
            hidden_states = F.pad(hidden_states, (0, 1, 0, 1), mode="constant", value=0)
        hidden_states = self.conv(hidden_states)
        return hidden_states, time_emb, text_emb, res_stack


class FluxVAEEncoder(nn.Module):
    """
    Flux VAE Encoder for Z-Image.

    Encodes RGB images to 16-channel latents.

    Architecture:
    - conv_in: 3 → 128
    - down_blocks: 4x DownEncoderBlock2D with channel progression 128→256→512→512
    - mid_block: ResNet + Attention + ResNet
    - conv_out: 512 → 32 (only first 16 channels are mean)

    Args:
        use_conv_attention: Use Conv2d-based attention (default: False uses Linear)

    Properties:
        config: Returns VAE configuration with scaling_factor and shift_factor
    """

    def __init__(self, use_conv_attention: bool = False):
        super().__init__()
        self.scaling_factor = FluxVAEConfig.ENCODER["scaling_factor"]
        self.shift_factor = FluxVAEConfig.ENCODER["shift_factor"]

        # Input convolution (3 → 128)
        self.conv_in = nn.Conv2d(3, 128, kernel_size=3, padding=1)

        # Build blocks
        self.blocks = nn.ModuleList([
            # DownEncoderBlock2D (128 → 128)
            ResnetBlock(128, 128, eps=1e-6),
            ResnetBlock(128, 128, eps=1e-6),
            DownSampler(128, padding=0, extra_padding=True),
            # DownEncoderBlock2D (128 → 256)
            ResnetBlock(128, 256, eps=1e-6),
            ResnetBlock(256, 256, eps=1e-6),
            DownSampler(256, padding=0, extra_padding=True),
            # DownEncoderBlock2D (256 → 512)
            ResnetBlock(256, 512, eps=1e-6),
            ResnetBlock(512, 512, eps=1e-6),
            DownSampler(512, padding=0, extra_padding=True),
            # DownEncoderBlock2D (512 → 512)
            ResnetBlock(512, 512, eps=1e-6),
            ResnetBlock(512, 512, eps=1e-6),
            # UNetMidBlock2D
            ResnetBlock(512, 512, eps=1e-6),
            VAEAttentionBlock(1, 512, 512, 1, eps=1e-6),
            ResnetBlock(512, 512, eps=1e-6),
        ])

        # Output layers (512 → 32, but only first 16 used as mean)
        self.conv_norm_out = nn.GroupNorm(num_channels=512, num_groups=32, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(512, 32, kernel_size=3, padding=1)

        # Config proxy for compatibility
        self._config = _VAEConfigProxy(
            scaling_factor=self.scaling_factor,
            shift_factor=self.shift_factor,
            block_out_channels=FluxVAEConfig.ENCODER["block_out_channels"],
        )

    @property
    def config(self):
        """Return config for compatibility with diffusers interface."""
        return self._config

    @property
    def dtype(self) -> torch.dtype:
        """Return model dtype."""
        return next(self.parameters()).dtype

    def tiled_forward(
        self,
        sample: torch.Tensor,
        tile_size: int = 64,
        tile_stride: int = 32,
    ) -> torch.Tensor:
        """Run forward with tiling for large images."""
        return TileWorker().tiled_forward(
            lambda x: self._forward_core(x),
            sample,
            tile_size,
            tile_stride,
            tile_device=sample.device,
            tile_dtype=sample.dtype,
        )

    def _forward_core(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Core forward pass."""
        time_emb = None
        text_emb = None
        res_stack = None

        # Process through blocks
        for block in self.blocks:
            hidden_states, time_emb, text_emb, res_stack = block(
                hidden_states, time_emb, text_emb, res_stack
            )

        # Output projection
        hidden_states = self.conv_norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)

        return hidden_states

    def forward(
        self,
        sample: torch.Tensor,
        tiled: bool = False,
        tile_size: int = 64,
        tile_stride: int = 32,
        **kwargs,
    ) -> torch.Tensor:
        """
        Encode image to latents.

        Args:
            sample: Image tensor (B, 3, H, W) in [-1, 1] range
            tiled: Use tiled processing for large images
            tile_size: Tile size for tiled processing
            tile_stride: Tile stride for tiled processing

        Returns:
            Latent tensor (B, 16, H/8, W/8)
        """
        if tiled:
            return self.tiled_forward(sample, tile_size=tile_size, tile_stride=tile_stride)

        # 1. Input convolution
        hidden_states = self.conv_in(sample)

        # 2. Core forward
        hidden_states = self._forward_core(hidden_states)

        # 3. Take first 16 channels as mean and scale
        hidden_states = hidden_states[:, :16]
        hidden_states = (hidden_states - self.shift_factor) * self.scaling_factor

        return hidden_states

    def encode(
        self,
        x: torch.Tensor,
        return_dict: bool = True,
        **kwargs,
    ):
        """
        Encode image to latents (diffusers-compatible interface).

        Args:
            x: Image tensor (B, 3, H, W) in [-1, 1] range
            return_dict: Whether to return a dict (ignored)

        Returns:
            Object with .sample() method that returns latents
        """
        latents = self.forward(x, **kwargs)
        return _LatentDistProxy(latents)


class _VAEConfigProxy:
    """Simple config proxy to match diffusers interface."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get(self, key: str, default=None):
        return getattr(self, key, default)


class _LatentDistProxy:
    """Proxy class to mimic diffusers latent distribution interface."""

    def __init__(self, mean: torch.Tensor):
        self._mean = mean

    def sample(self) -> torch.Tensor:
        """Return the mean (no sampling for now)."""
        return self._mean

    @property
    def latent_dist(self):
        """Return self for compatibility."""
        return self
