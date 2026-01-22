"""
LTX-2 Video VAE - Encoder and Decoder.

Last Updated: 2026-01-18

Pure PyTorch implementation of the LTX-2 Video VAE for encoding video frames
to latents and decoding latents back to video frames.

Key specifications:
- Spatial compression: 32x (512x768 -> 16x24 latents)
- Temporal compression: 8x (33 frames -> 5 latent frames)
- Latent channels: 128
- Frame requirement: 1 + 8*k frames (1, 9, 17, 25, 33...)

Ported from: ltx_core.model.video_vae.video_vae
Original source: https://github.com/Lightricks/LTX-2
License: LTX-2 Community License
Copyright (c) 2025 Lightricks Ltd.
"""

from dataclasses import replace
from typing import Any, Callable, Iterator, List, Optional, Tuple

import torch
from einops import rearrange
from torch import nn

from .convolution import make_conv_nd
from .enums import LogVarianceType, NormLayerType, PaddingModeType
from .normalization import PixelNorm
from .ops import PerChannelStatistics, patchify, unpatchify
from .resnet import ResnetBlock3D, UNetMidBlock3D
from .sampling import DepthToSpaceUpsample, SpaceToDepthDownsample
from .timestep_embedding import PixArtAlphaCombinedTimestepSizeEmbeddings
from .tiling import (
    DEFAULT_MAPPING_OPERATION,
    DEFAULT_SPLIT_OPERATION,
    DimensionIntervals,
    MappingOperation,
    SplitOperation,
    Tile,
    TilingConfig,
    compute_trapezoidal_mask_1d,
    create_tiles,
)
from .types import SpatioTemporalScaleFactors, VideoLatentShape


def _make_encoder_block(
    block_name: str,
    block_config: dict[str, Any],
    in_channels: int,
    convolution_dimensions: int,
    norm_layer: NormLayerType,
    norm_num_groups: int,
    spatial_padding_mode: PaddingModeType,
) -> Tuple[nn.Module, int]:
    """
    Create an encoder block based on the block name.

    Returns:
        Tuple of (block module, output channels).
    """
    out_channels = in_channels

    if block_name == "res_x":
        block = UNetMidBlock3D(
            dims=convolution_dimensions,
            in_channels=in_channels,
            num_layers=block_config["num_layers"],
            resnet_eps=1e-6,
            resnet_groups=norm_num_groups,
            norm_layer=norm_layer,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "res_x_y":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = ResnetBlock3D(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            eps=1e-6,
            groups=norm_num_groups,
            norm_layer=norm_layer,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_time":
        block = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=(2, 1, 1),
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_space":
        block = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=(1, 2, 2),
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_all":
        block = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=(2, 2, 2),
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_all_x_y":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=(2, 2, 2),
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_all_res":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = SpaceToDepthDownsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=(2, 2, 2),
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_space_res":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = SpaceToDepthDownsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=(1, 2, 2),
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_time_res":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = SpaceToDepthDownsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=(2, 1, 1),
            spatial_padding_mode=spatial_padding_mode,
        )
    else:
        raise ValueError(f"unknown block: {block_name}")

    return block, out_channels


class VideoEncoder(nn.Module):
    """
    Video VAE Encoder: Encodes video frames into a latent representation.

    The encoder compresses the input video through a series of downsampling operations.
    The output is a normalized latent tensor with shape (B, 128, F', H', W').

    Compression Behavior:
        - Initial spatial compression via patchify: H -> H/4, W -> W/4 (patch_size=4)
        - Sequential compression through encoder_blocks based on their stride patterns

    Standard LTX Video configuration:
        - patch_size=4
        - encoder_blocks: 1x compress_space_res, 1x compress_time_res, 2x compress_all_res
        - Final dimensions: F' = 1 + (F-1)/8, H' = H/32, W' = W/32
        - Example: (B, 3, 33, 512, 768) -> (B, 128, 5, 16, 24)

    Note: Input must have 1 + 8*k frames (e.g., 1, 9, 17, 25, 33...)
    """

    _DEFAULT_NORM_NUM_GROUPS = 32

    def __init__(
        self,
        convolution_dimensions: int = 3,
        in_channels: int = 3,
        out_channels: int = 128,
        encoder_blocks: List[Tuple[str, int | dict[str, Any]]] = [],
        patch_size: int = 4,
        norm_layer: NormLayerType = NormLayerType.PIXEL_NORM,
        latent_log_var: LogVarianceType = LogVarianceType.UNIFORM,
        encoder_spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
    ):
        """
        Args:
            convolution_dimensions: Number of dimensions for convolutions (2D or 3D).
            in_channels: Number of input channels. For RGB images, this is 3.
            out_channels: Number of output channels (latent channels). Default 128.
            encoder_blocks: List of (block_name, params) tuples for encoder architecture.
            patch_size: Patch size for initial spatial compression.
            norm_layer: Normalization layer type (group_norm or pixel_norm).
            latent_log_var: Log variance mode for VAE output.
            encoder_spatial_padding_mode: Padding mode for convolutions.
        """
        super().__init__()

        self.patch_size = patch_size
        self.norm_layer = norm_layer
        self.latent_channels = out_channels
        self.latent_log_var = latent_log_var
        self._norm_num_groups = self._DEFAULT_NORM_NUM_GROUPS

        # Per-channel statistics for normalizing latents
        self.per_channel_statistics = PerChannelStatistics(latent_channels=out_channels)

        in_channels = in_channels * patch_size**2
        feature_channels = out_channels

        self.conv_in = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=feature_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            causal=True,
            spatial_padding_mode=encoder_spatial_padding_mode,
        )

        self.down_blocks = nn.ModuleList([])

        for block_name, block_params in encoder_blocks:
            block_config = {"num_layers": block_params} if isinstance(block_params, int) else block_params

            block, feature_channels = _make_encoder_block(
                block_name=block_name,
                block_config=block_config,
                in_channels=feature_channels,
                convolution_dimensions=convolution_dimensions,
                norm_layer=norm_layer,
                norm_num_groups=self._norm_num_groups,
                spatial_padding_mode=encoder_spatial_padding_mode,
            )

            self.down_blocks.append(block)

        # Output normalization
        if norm_layer == NormLayerType.GROUP_NORM:
            self.conv_norm_out = nn.GroupNorm(
                num_channels=feature_channels, num_groups=self._norm_num_groups, eps=1e-6
            )
        elif norm_layer == NormLayerType.PIXEL_NORM:
            self.conv_norm_out = PixelNorm()

        self.conv_act = nn.SiLU()

        conv_out_channels = out_channels
        if latent_log_var == LogVarianceType.PER_CHANNEL:
            conv_out_channels *= 2
        elif latent_log_var in {LogVarianceType.UNIFORM, LogVarianceType.CONSTANT}:
            conv_out_channels += 1
        elif latent_log_var != LogVarianceType.NONE:
            raise ValueError(f"Invalid latent_log_var: {latent_log_var}")

        self.conv_out = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=feature_channels,
            out_channels=conv_out_channels,
            kernel_size=3,
            padding=1,
            causal=True,
            spatial_padding_mode=encoder_spatial_padding_mode,
        )

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        """
        Encode video frames into normalized latent representation.

        Args:
            sample: Input video (B, C, F, H, W). F must be 1 + 8*k (e.g., 1, 9, 17, 25, 33...).

        Returns:
            Normalized latent means (B, 128, F', H', W') where F' = 1+(F-1)/8, H' = H/32, W' = W/32.
            Example: (B, 3, 33, 512, 768) -> (B, 128, 5, 16, 24).
        """
        # Validate frame count
        frames_count = sample.shape[2]
        if ((frames_count - 1) % 8) != 0:
            raise ValueError(
                "Invalid number of frames: Encode input must have 1 + 8 * x frames "
                "(e.g., 1, 9, 17, ...). Please check your input."
            )

        # Initial spatial compression via patchify
        sample = patchify(sample, patch_size_hw=self.patch_size, patch_size_t=1)
        sample = self.conv_in(sample)

        for down_block in self.down_blocks:
            sample = down_block(sample)

        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if self.latent_log_var == LogVarianceType.UNIFORM:
            # Expand single logvar to match number of means channels
            if sample.shape[1] < 2:
                raise ValueError(
                    f"Invalid channel count for UNIFORM mode: expected at least 2 channels, got {sample.shape[1]}"
                )
            means = sample[:, :-1, ...]
            logvar = sample[:, -1:, ...]
            num_channels = means.shape[1]
            repeat_shape = [1, num_channels] + [1] * (sample.ndim - 2)
            repeated_logvar = logvar.repeat(*repeat_shape)
            sample = torch.cat([means, repeated_logvar], dim=1)
        elif self.latent_log_var == LogVarianceType.CONSTANT:
            sample = sample[:, :-1, ...]
            approx_ln_0 = -30
            sample = torch.cat(
                [sample, torch.ones_like(sample, device=sample.device) * approx_ln_0],
                dim=1,
            )

        # Split into means and logvar, then normalize means
        means, _ = torch.chunk(sample, 2, dim=1)
        return self.per_channel_statistics.normalize(means)


def _make_decoder_block(
    block_name: str,
    block_config: dict[str, Any],
    in_channels: int,
    convolution_dimensions: int,
    norm_layer: NormLayerType,
    timestep_conditioning: bool,
    norm_num_groups: int,
    spatial_padding_mode: PaddingModeType,
) -> Tuple[nn.Module, int]:
    """Create a decoder block based on the block name."""
    out_channels = in_channels

    if block_name == "res_x":
        block = UNetMidBlock3D(
            dims=convolution_dimensions,
            in_channels=in_channels,
            num_layers=block_config["num_layers"],
            resnet_eps=1e-6,
            resnet_groups=norm_num_groups,
            norm_layer=norm_layer,
            inject_noise=block_config.get("inject_noise", False),
            timestep_conditioning=timestep_conditioning,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "attn_res_x":
        block = UNetMidBlock3D(
            dims=convolution_dimensions,
            in_channels=in_channels,
            num_layers=block_config["num_layers"],
            resnet_groups=norm_num_groups,
            norm_layer=norm_layer,
            inject_noise=block_config.get("inject_noise", False),
            timestep_conditioning=timestep_conditioning,
            attention_head_dim=block_config.get("attention_head_dim"),
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "res_x_y":
        out_channels = in_channels // block_config.get("multiplier", 2)
        block = ResnetBlock3D(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            eps=1e-6,
            groups=norm_num_groups,
            norm_layer=norm_layer,
            inject_noise=block_config.get("inject_noise", False),
            timestep_conditioning=False,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_time":
        block = DepthToSpaceUpsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            stride=(2, 1, 1),
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_space":
        block = DepthToSpaceUpsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            stride=(1, 2, 2),
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_all":
        out_channels = in_channels // block_config.get("multiplier", 1)
        block = DepthToSpaceUpsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            stride=(2, 2, 2),
            residual=block_config.get("residual", False),
            out_channels_reduction_factor=block_config.get("multiplier", 1),
            spatial_padding_mode=spatial_padding_mode,
        )
    else:
        raise ValueError(f"unknown layer: {block_name}")

    return block, out_channels


class VideoDecoder(nn.Module):
    """
    Video VAE Decoder: Decodes latent representation into video frames.

    The decoder upsamples latents through a series of upsampling operations (inverse of encoder).
    Output dimensions: F = 8x(F'-1) + 1, H = 32xH', W = 32xW' for standard LTX Video configuration.

    Upsampling blocks expand dimensions by 2x in specified dimensions:
        - "compress_time": temporal only
        - "compress_space": spatial only (H and W)
        - "compress_all": all dimensions (F, H, W)
        - "res_x" / "res_x_y" / "attn_res_x": no upsampling

    Causal Mode:
        causal=False (standard): Symmetric padding, allows future frame dependencies.
        causal=True: Causal padding, each frame depends only on past/current frames.

    Example: (B, 128, 5, 16, 24) -> (B, 3, 33, 512, 768)
    """

    _DEFAULT_NORM_NUM_GROUPS = 32

    def __init__(
        self,
        convolution_dimensions: int = 3,
        in_channels: int = 128,
        out_channels: int = 3,
        decoder_blocks: List[Tuple[str, int | dict]] = [],
        patch_size: int = 4,
        norm_layer: NormLayerType = NormLayerType.PIXEL_NORM,
        causal: bool = False,
        timestep_conditioning: bool = False,
        decoder_spatial_padding_mode: PaddingModeType = PaddingModeType.REFLECT,
    ):
        """
        Args:
            convolution_dimensions: Number of dimensions for convolutions (2D or 3D).
            in_channels: Number of input channels (latent channels). Default 128.
            out_channels: Number of output channels. For RGB, this is 3.
            decoder_blocks: List of (block_name, params) tuples for decoder architecture.
            patch_size: Final spatial expansion factor.
            norm_layer: Normalization layer type.
            causal: Whether to use causal convolutions.
            timestep_conditioning: Whether to condition on timestep for denoising.
            decoder_spatial_padding_mode: Padding mode for convolutions.
        """
        super().__init__()

        # Scale factors for upsampling
        self.video_downscale_factors = SpatioTemporalScaleFactors(time=8, width=32, height=32)

        self.patch_size = patch_size
        out_channels = out_channels * patch_size**2
        self.causal = causal
        self.timestep_conditioning = timestep_conditioning
        self._norm_num_groups = self._DEFAULT_NORM_NUM_GROUPS

        # Per-channel statistics for denormalizing latents
        self.per_channel_statistics = PerChannelStatistics(latent_channels=in_channels)

        # Noise and timestep parameters for decoder conditioning
        self.decode_noise_scale = 0.025
        self.decode_timestep = 0.05

        # Compute initial feature_channels by going through blocks in reverse
        feature_channels = in_channels
        for block_name, block_params in list(reversed(decoder_blocks)):
            block_config = block_params if isinstance(block_params, dict) else {}
            if block_name == "res_x_y":
                feature_channels = feature_channels * block_config.get("multiplier", 2)
            if block_name == "compress_all":
                feature_channels = feature_channels * block_config.get("multiplier", 1)

        self.conv_in = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=feature_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            causal=True,
            spatial_padding_mode=decoder_spatial_padding_mode,
        )

        self.up_blocks = nn.ModuleList([])

        for block_name, block_params in list(reversed(decoder_blocks)):
            block_config = {"num_layers": block_params} if isinstance(block_params, int) else block_params

            block, feature_channels = _make_decoder_block(
                block_name=block_name,
                block_config=block_config,
                in_channels=feature_channels,
                convolution_dimensions=convolution_dimensions,
                norm_layer=norm_layer,
                timestep_conditioning=timestep_conditioning,
                norm_num_groups=self._norm_num_groups,
                spatial_padding_mode=decoder_spatial_padding_mode,
            )

            self.up_blocks.append(block)

        # Output normalization
        if norm_layer == NormLayerType.GROUP_NORM:
            self.conv_norm_out = nn.GroupNorm(
                num_channels=feature_channels, num_groups=self._norm_num_groups, eps=1e-6
            )
        elif norm_layer == NormLayerType.PIXEL_NORM:
            self.conv_norm_out = PixelNorm()

        self.conv_act = nn.SiLU()
        self.conv_out = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=feature_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            causal=True,
            spatial_padding_mode=decoder_spatial_padding_mode,
        )

        if timestep_conditioning:
            self.timestep_scale_multiplier = nn.Parameter(torch.tensor(1000.0))
            self.last_time_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(
                embedding_dim=feature_channels * 2, size_emb_dim=0
            )
            self.last_scale_shift_table = nn.Parameter(torch.empty(2, feature_channels))

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Decode latent representation into video frames.

        Args:
            sample: Latent tensor (B, 128, F', H', W').
            timestep: Timestep for conditioning (if timestep_conditioning=True).
            generator: Random generator for deterministic noise injection.

        Returns:
            Decoded video (B, 3, F, H, W) where F = 8x(F'-1) + 1, H = 32xH', W = 32xW'.
            Example: (B, 128, 5, 16, 24) -> (B, 3, 33, 512, 768).
        """
        batch_size = sample.shape[0]

        # Add noise if timestep conditioning is enabled
        if self.timestep_conditioning:
            noise = (
                torch.randn(
                    sample.size(),
                    generator=generator,
                    dtype=sample.dtype,
                    device=sample.device,
                )
                * self.decode_noise_scale
            )
            sample = noise + (1.0 - self.decode_noise_scale) * sample

        # Denormalize latents (internal normalization applied by encoder)
        sample = self.per_channel_statistics.un_normalize(sample)

        # Use default decode_timestep if timestep not provided
        if timestep is None and self.timestep_conditioning:
            timestep = torch.full((batch_size,), self.decode_timestep, device=sample.device, dtype=sample.dtype)

        sample = self.conv_in(sample, causal=self.causal)

        scaled_timestep = None
        if self.timestep_conditioning:
            if timestep is None:
                raise ValueError("'timestep' parameter must be provided when 'timestep_conditioning' is True")
            scaled_timestep = timestep * self.timestep_scale_multiplier.to(sample)

        for i, up_block in enumerate(self.up_blocks):
            if isinstance(up_block, UNetMidBlock3D):
                block_kwargs = {
                    "causal": self.causal,
                    "timestep": scaled_timestep if self.timestep_conditioning else None,
                    "generator": generator,
                }
                sample = up_block(sample, **block_kwargs)
            elif isinstance(up_block, ResnetBlock3D):
                sample = up_block(sample, causal=self.causal, generator=generator)
            else:
                sample = up_block(sample, causal=self.causal)

        sample = self.conv_norm_out(sample)

        if self.timestep_conditioning:
            assert scaled_timestep is not None  # Always set when timestep_conditioning=True
            embedded_timestep = self.last_time_embedder(
                timestep=scaled_timestep.flatten(),
                hidden_dtype=sample.dtype,
            )
            embedded_timestep = embedded_timestep.view(batch_size, embedded_timestep.shape[-1], 1, 1, 1)
            ada_values = self.last_scale_shift_table[None, ..., None, None, None].to(
                device=sample.device, dtype=sample.dtype
            ) + embedded_timestep.reshape(
                batch_size,
                2,
                -1,
                embedded_timestep.shape[-3],
                embedded_timestep.shape[-2],
                embedded_timestep.shape[-1],
            )
            shift, scale = ada_values.unbind(dim=1)
            sample = sample * (1 + scale) + shift

        sample = self.conv_act(sample)
        sample = self.conv_out(sample, causal=self.causal)

        # Final spatial expansion via unpatchify
        sample = unpatchify(sample, patch_size_hw=self.patch_size, patch_size_t=1)

        return sample

    def tiled_decode(
        self,
        latent: torch.Tensor,
        tiling_config: "TilingConfig",
        generator: Optional[torch.Generator] = None,
    ) -> Iterator[torch.Tensor]:
        """
        Decode video latents using tiled processing for memory efficiency.

        Args:
            latent: Latent tensor [B, C, F, H, W]
            tiling_config: Configuration for spatial/temporal tiling
            generator: Optional random generator for deterministic decoding

        Yields:
            Decoded video chunks [B, C, F_chunk, H, W]

        Note:
            This method is not yet implemented. It requires integration with
            the tiling utilities for large video decoding.
        """
        raise NotImplementedError(
            "tiled_decode is not yet implemented. "
            "For tiled video decoding, use decode_video() with tiling_config=None "
            "and process chunks manually, or implement this method."
        )


# Utility functions


def decode_video(
    latent: torch.Tensor,
    video_decoder: VideoDecoder,
    tiling_config: Optional[TilingConfig] = None,
    generator: Optional[torch.Generator] = None,
) -> Iterator[torch.Tensor]:
    """
    Decode a video latent tensor with the given decoder.

    Args:
        latent: Tensor [B, C, F, H, W]
        video_decoder: Decoder module.
        tiling_config: Optional tiling settings.
        generator: Optional random generator for deterministic decoding.

    Yields:
        Decoded chunk [F, H, W, C], uint8 in [0, 255].
    """

    def convert_to_uint8(frames: torch.Tensor) -> torch.Tensor:
        frames = (((frames + 1.0) / 2.0).clamp(0.0, 1.0) * 255.0).to(torch.uint8)
        frames = rearrange(frames[0], "c f h w -> f h w c")
        return frames

    if tiling_config is not None:
        for frames in video_decoder.tiled_decode(latent, tiling_config, generator=generator):
            yield convert_to_uint8(frames)
    else:
        decoded_video = video_decoder(latent, generator=generator)
        yield convert_to_uint8(decoded_video)


def get_video_chunks_number(num_frames: int, tiling_config: Optional[TilingConfig] = None) -> int:
    """
    Get the number of video chunks for a given number of frames and tiling configuration.

    Args:
        num_frames: Number of frames in the video.
        tiling_config: Tiling configuration.

    Returns:
        Number of video chunks.
    """
    if not tiling_config or not tiling_config.temporal_config:
        return 1
    cfg = tiling_config.temporal_config
    frame_stride = cfg.tile_size_in_frames - cfg.tile_overlap_in_frames
    return (num_frames - 1 + frame_stride - 1) // frame_stride
