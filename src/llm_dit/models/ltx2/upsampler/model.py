"""
Latent Upsampler model for LTX-2 video generation.

Last Updated: 2026-01-18

LatentUpsampler provides spatial and/or temporal upsampling of VAE latents
with learnable transformations via ResBlocks.
"""

from typing import TYPE_CHECKING

import torch
from einops import rearrange

from llm_dit.models.ltx2.upsampler.pixel_shuffle import PixelShuffleND
from llm_dit.models.ltx2.upsampler.res_block import ResBlock
from llm_dit.models.ltx2.upsampler.spatial_rational_resampler import SpatialRationalResampler

if TYPE_CHECKING:
    from llm_dit.models.ltx2.vae.video_vae import VideoEncoder


class LatentUpsampler(torch.nn.Module):
    """
    Model to upsample VAE latents spatially and/or temporally.

    Architecture:
        Input (B, in_channels, F, H, W)
          ↓
        initial_conv (in_channels → mid_channels)
          ↓
        GroupNorm(32) + SiLU
          ↓
        num_blocks_per_stage × ResBlock (pre-upsample)
          ↓
        Upsampler (PixelShuffle-based, mode-dependent)
          ↓
        num_blocks_per_stage × ResBlock (post-upsample)
          ↓
        final_conv (mid_channels → in_channels)
          ↓
        Output (B, in_channels, F', H', W')

    Args:
        in_channels: Number of channels in the input latent. Default: 128.
        mid_channels: Number of channels in the middle layers. Default: 512.
        num_blocks_per_stage: Number of ResBlocks per stage (pre/post). Default: 4.
        dims: Number of dimensions for convolutions (2 or 3). Default: 3.
        spatial_upsample: Whether to spatially upsample. Default: True.
        temporal_upsample: Whether to temporally upsample. Default: False.
        spatial_scale: Scale factor for spatial upsampling. Default: 2.0.
        rational_resampler: Use rational resampler for fractional scales. Default: False.
    """

    def __init__(
        self,
        in_channels: int = 128,
        mid_channels: int = 512,
        num_blocks_per_stage: int = 4,
        dims: int = 3,
        spatial_upsample: bool = True,
        temporal_upsample: bool = False,
        spatial_scale: float = 2.0,
        rational_resampler: bool = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.num_blocks_per_stage = num_blocks_per_stage
        self.dims = dims
        self.spatial_upsample = spatial_upsample
        self.temporal_upsample = temporal_upsample
        self.spatial_scale = float(spatial_scale)
        self.rational_resampler = rational_resampler

        conv = torch.nn.Conv2d if dims == 2 else torch.nn.Conv3d

        # Initial projection
        self.initial_conv = conv(in_channels, mid_channels, kernel_size=3, padding=1)
        self.initial_norm = torch.nn.GroupNorm(32, mid_channels)
        self.initial_activation = torch.nn.SiLU()

        # Pre-upsample ResBlocks
        self.res_blocks = torch.nn.ModuleList([ResBlock(mid_channels, dims=dims) for _ in range(num_blocks_per_stage)])

        # Upsampler (mode-dependent)
        if spatial_upsample and temporal_upsample:
            # 3D spatiotemporal: upsample all dimensions
            self.upsampler = torch.nn.Sequential(
                torch.nn.Conv3d(mid_channels, 8 * mid_channels, kernel_size=3, padding=1),
                PixelShuffleND(3),  # 2x2x2 upscale
            )
        elif spatial_upsample:
            if rational_resampler:
                # Fractional spatial scale via rational resampler
                self.upsampler = SpatialRationalResampler(mid_channels=mid_channels, scale=self.spatial_scale)
            else:
                # Standard 2x spatial upscale
                self.upsampler = torch.nn.Sequential(
                    torch.nn.Conv2d(mid_channels, 4 * mid_channels, kernel_size=3, padding=1),
                    PixelShuffleND(2),  # 2x2 upscale
                )
        elif temporal_upsample:
            # Temporal-only: 2x temporal upscale
            self.upsampler = torch.nn.Sequential(
                torch.nn.Conv3d(mid_channels, 2 * mid_channels, kernel_size=3, padding=1),
                PixelShuffleND(1),  # 2x temporal upscale
            )
        else:
            raise ValueError("Either spatial_upsample or temporal_upsample must be True")

        # Post-upsample ResBlocks
        self.post_upsample_res_blocks = torch.nn.ModuleList(
            [ResBlock(mid_channels, dims=dims) for _ in range(num_blocks_per_stage)]
        )

        # Final projection back to input channels
        self.final_conv = conv(mid_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        b, c, f, h, w = latent.shape

        if self.dims == 2:
            # For dims=2, process frames independently
            x = rearrange(latent, "b c f h w -> (b f) c h w")
            x = self.initial_conv(x)
            x = self.initial_norm(x)
            x = self.initial_activation(x)

            for block in self.res_blocks:
                x = block(x)

            x = self.upsampler(x)

            for block in self.post_upsample_res_blocks:
                x = block(x)

            x = self.final_conv(x)
            x = rearrange(x, "(b f) c h w -> b c f h w", b=b, f=f)
        else:
            # dims=3: full 3D processing
            x = self.initial_conv(latent)
            x = self.initial_norm(x)
            x = self.initial_activation(x)

            for block in self.res_blocks:
                x = block(x)

            if self.temporal_upsample:
                # Temporal or spatiotemporal upsampling
                x = self.upsampler(x)
                # Remove the first frame after temporal upsampling.
                # This is done because the first frame encodes one pixel frame
                # in the LTX-2 latent representation.
                x = x[:, :, 1:, :, :]
            elif isinstance(self.upsampler, SpatialRationalResampler):
                # Rational resampler handles per-frame internally
                x = self.upsampler(x)
            else:
                # Spatial-only with standard upsampler: process per-frame
                x = rearrange(x, "b c f h w -> (b f) c h w")
                x = self.upsampler(x)
                x = rearrange(x, "(b f) c h w -> b c f h w", b=b, f=f)

            for block in self.post_upsample_res_blocks:
                x = block(x)

            x = self.final_conv(x)

        return x


def upsample_video(latent: torch.Tensor, video_encoder: "VideoEncoder", upsampler: LatentUpsampler) -> torch.Tensor:
    """
    Apply upsampling to the latent representation with proper normalization handling.

    The latent must be un-normalized before upsampling and re-normalized after,
    using the video encoder's per-channel statistics.

    Args:
        latent: Input latent tensor of shape [B, C, F, H, W].
        video_encoder: VideoEncoder with per_channel_statistics for normalization.
        upsampler: LatentUpsampler module to perform upsampling.

    Returns:
        Upsampled and re-normalized latent tensor.
    """
    latent = video_encoder.per_channel_statistics.un_normalize(latent)
    latent = upsampler(latent)
    latent = video_encoder.per_channel_statistics.normalize(latent)
    return latent
