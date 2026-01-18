"""
LTX-2 Video VAE Package.

Last Updated: 2026-01-18

Pure PyTorch implementation of the LTX-2 Video VAE for encoding video frames
to latents and decoding latents back to video frames.

Key Components:
    - VideoEncoder: Encodes (B, 3, F, H, W) video to (B, 128, F', H', W') latents
    - VideoDecoder: Decodes latents back to video frames
    - TilingConfig: Configuration for memory-efficient tiled processing

Specifications:
    - Spatial compression: 32x (512x768 -> 16x24 latents)
    - Temporal compression: 8x (33 frames -> 5 latent frames)
    - Latent channels: 128
    - Frame requirement: 1 + 8*k frames (1, 9, 17, 25, 33...)

Example:
    ```python
    from llm_dit.models.ltx2.vae import VideoEncoder, VideoDecoder

    # Create encoder/decoder (or load from checkpoint)
    encoder = VideoEncoder(...)
    decoder = VideoDecoder(...)

    # Encode video to latents
    video = torch.randn(1, 3, 33, 512, 768)  # [B, C, F, H, W]
    latents = encoder(video)  # [B, 128, 5, 16, 24]

    # Decode latents back to video
    reconstructed = decoder(latents)  # [B, 3, 33, 512, 768]
    ```

Ported from: ltx_core.model.video_vae
Original source: https://github.com/Lightricks/LTX-2
License: LTX-2 Community License
Copyright (c) 2025 Lightricks Ltd.
"""

# Enums
from .enums import LogVarianceType, NormLayerType, PaddingModeType

# Types
from .types import SpatioTemporalScaleFactors, VideoLatentShape, VIDEO_SCALE_FACTORS

# Normalization
from .normalization import PixelNorm, NormType, build_normalization_layer

# Operations
from .ops import patchify, unpatchify, PerChannelStatistics

# Convolution
from .convolution import make_conv_nd, make_linear_nd, CausalConv3d, DualConv3d

# Sampling (up/downsampling)
from .sampling import SpaceToDepthDownsample, DepthToSpaceUpsample

# ResNet blocks
from .resnet import ResnetBlock3D, UNetMidBlock3D

# Timestep embedding
from .timestep_embedding import (
    Timesteps,
    TimestepEmbedding,
    PixArtAlphaCombinedTimestepSizeEmbeddings,
)

# Tiling
from .tiling import (
    TilingConfig,
    SpatialTilingConfig,
    TemporalTilingConfig,
    Tile,
    DimensionIntervals,
    LatentIntervals,
    compute_trapezoidal_mask_1d,
    create_tiles,
    DEFAULT_SPLIT_OPERATION,
    DEFAULT_MAPPING_OPERATION,
)

# Main VAE classes
from .video_vae import (
    VideoEncoder,
    VideoDecoder,
    decode_video,
    get_video_chunks_number,
)

__all__ = [
    # Enums
    "LogVarianceType",
    "NormLayerType",
    "PaddingModeType",
    "NormType",
    # Types
    "SpatioTemporalScaleFactors",
    "VideoLatentShape",
    "VIDEO_SCALE_FACTORS",
    # Normalization
    "PixelNorm",
    "build_normalization_layer",
    # Operations
    "patchify",
    "unpatchify",
    "PerChannelStatistics",
    # Convolution
    "make_conv_nd",
    "make_linear_nd",
    "CausalConv3d",
    "DualConv3d",
    # Sampling
    "SpaceToDepthDownsample",
    "DepthToSpaceUpsample",
    # ResNet
    "ResnetBlock3D",
    "UNetMidBlock3D",
    # Timestep
    "Timesteps",
    "TimestepEmbedding",
    "PixArtAlphaCombinedTimestepSizeEmbeddings",
    # Tiling
    "TilingConfig",
    "SpatialTilingConfig",
    "TemporalTilingConfig",
    "Tile",
    "DimensionIntervals",
    "LatentIntervals",
    "compute_trapezoidal_mask_1d",
    "create_tiles",
    "DEFAULT_SPLIT_OPERATION",
    "DEFAULT_MAPPING_OPERATION",
    # Main VAE
    "VideoEncoder",
    "VideoDecoder",
    "decode_video",
    "get_video_chunks_number",
]
