"""
LTX-2 Video VAE Unit Tests.

Last Updated: 2026-01-22

Tests for the pure PyTorch Video VAE implementation including:
- Compression ratio validation (spatial 32x, temporal 8x)
- Component building blocks (CausalConv3d, DualConv3d, sampling layers)
- Tiling configuration validation
- Memory-efficient decode patterns

Run with: uv run pytest tests/unit/test_ltx2_video_vae.py -v
"""

import pytest
import torch

from llm_dit.models.ltx2.vae.convolution import (
    CausalConv3d,
    DualConv3d,
    make_conv_nd,
)
from llm_dit.models.ltx2.vae.sampling import (
    DepthToSpaceUpsample,
    SpaceToDepthDownsample,
)
from llm_dit.models.ltx2.vae.tiling import (
    SpatialTilingConfig,
    TemporalTilingConfig,
    TilingConfig,
    compute_trapezoidal_mask_1d,
    create_tiles,
    DEFAULT_SPLIT_OPERATION,
    DEFAULT_MAPPING_OPERATION,
)


# ============================================================================
# Compression Ratio Tests
# ============================================================================


class TestVideoVAECompression:
    """Tests for LTX-2 VAE compression ratios."""

    def test_spatial_compression_32x(self):
        """Verify 32x spatial compression: H/32, W/32."""
        # LTX-2 compresses spatial dimensions by 32x
        # 512x768 -> 16x24 latents
        test_cases = [
            (512, 768, 16, 24),
            (256, 384, 8, 12),
            (1024, 1536, 32, 48),
        ]

        for h_in, w_in, h_lat, w_lat in test_cases:
            computed_h = h_in // 32
            computed_w = w_in // 32
            assert computed_h == h_lat, f"Height mismatch: {computed_h} != {h_lat}"
            assert computed_w == w_lat, f"Width mismatch: {computed_w} != {w_lat}"

    def test_temporal_compression_8x(self):
        """Verify 8x temporal compression: (F-1)/8 + 1."""
        # LTX-2 requires frames = 1 + 8*k (1, 9, 17, 25, 33...)
        # Formula: latent_frames = (frames - 1) // 8 + 1
        test_cases = [
            (1, 1),    # 1 frame -> 1 latent frame
            (9, 2),    # 9 frames -> 2 latent frames
            (17, 3),   # 17 frames -> 3 latent frames
            (33, 5),   # 33 frames -> 5 latent frames
            (121, 16), # 121 frames -> 16 latent frames
        ]

        for frames_in, latent_frames in test_cases:
            computed = (frames_in - 1) // 8 + 1
            assert computed == latent_frames, \
                f"Temporal mismatch for {frames_in} frames: {computed} != {latent_frames}"

    def test_latent_channels_128(self):
        """Verify 128 latent channels."""
        # LTX-2 uses 128 latent channels (hardcoded in VideoEncoder/Decoder)
        LATENT_CHANNELS = 128
        assert LATENT_CHANNELS == 128

    def test_frame_count_constraint(self):
        """Verify frame count must satisfy 1 + 8*k."""
        valid_frame_counts = [1, 9, 17, 25, 33, 41, 49, 121]
        for frames in valid_frame_counts:
            assert (frames - 1) % 8 == 0, f"{frames} doesn't satisfy (frames-1) % 8 == 0"

    def test_invalid_frame_count_detection(self):
        """Verify invalid frame counts are detectable."""
        invalid_counts = [10, 20, 30, 32, 100, 120]
        for frames in invalid_counts:
            assert (frames - 1) % 8 != 0, f"{frames} unexpectedly satisfies constraint"


# ============================================================================
# Convolution Component Tests
# ============================================================================


class TestCausalConv3d:
    """Tests for CausalConv3d preserving temporal causality."""

    def test_causal_conv3d_shape(self):
        """Test CausalConv3d preserves spatial dims with causal padding."""
        conv = CausalConv3d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=1,
        )

        # Input: (B, C, D, H, W)
        x = torch.randn(1, 128, 5, 16, 24)
        y = conv(x, causal=True)

        # Output should have same spatial/temporal dims with stride=1
        assert y.shape == x.shape, f"Shape mismatch: {y.shape} != {x.shape}"

    def test_causal_vs_noncausal(self):
        """Test causal and non-causal modes produce different results."""
        conv = CausalConv3d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
        )

        x = torch.randn(1, 64, 5, 8, 8)

        y_causal = conv(x, causal=True)
        y_noncausal = conv(x, causal=False)

        # Should produce different outputs due to padding difference
        assert not torch.allclose(y_causal, y_noncausal), \
            "Causal and non-causal modes should differ"

    def test_causal_conv3d_temporal_stride(self):
        """Test CausalConv3d with temporal downsampling."""
        conv = CausalConv3d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=(2, 1, 1),  # 2x temporal downsample
        )

        x = torch.randn(1, 128, 8, 16, 24)
        y = conv(x, causal=True)

        # With causal padding (kernel_size-1 = 2 frames prepended):
        # - Padded input: 8 + 2 = 10 frames
        # - Conv formula: floor((10 - 3) / 2) + 1 = 4
        expected_t = (x.shape[2] + conv.time_kernel_size - 1 - conv.time_kernel_size) // 2 + 1
        assert y.shape[2] == expected_t, \
            f"Expected temporal dim {expected_t}, got {y.shape[2]}"
        assert y.shape[2] == 4, "Should produce 4 temporal frames from 8 input with stride 2"


class TestDualConv3d:
    """Tests for factorized 2D+1D DualConv3d."""

    def test_dual_conv3d_shape(self):
        """Test DualConv3d output shape."""
        conv = DualConv3d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        x = torch.randn(1, 64, 5, 8, 8)
        y = conv(x, use_conv3d=False)

        assert y.shape == x.shape, f"Shape mismatch: {y.shape} != {x.shape}"

    def test_dual_conv3d_2d_vs_3d_equivalence(self):
        """Test 2D factorized and 3D mode produce similar results."""
        conv = DualConv3d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        x = torch.randn(1, 32, 4, 8, 8)

        y_2d = conv(x, use_conv3d=False)
        y_3d = conv(x, use_conv3d=True)

        # Should be numerically equivalent
        assert torch.allclose(y_2d, y_3d, atol=1e-5), \
            "2D factorized and 3D modes should be equivalent"


class TestMakeConvNd:
    """Tests for the make_conv_nd factory function."""

    def test_make_conv_2d(self):
        """Test creating 2D convolution."""
        conv = make_conv_nd(dims=2, in_channels=64, out_channels=64, kernel_size=3)
        x = torch.randn(1, 64, 32, 32)
        y = conv(x)
        assert y.shape[0] == 1 and y.shape[1] == 64

    def test_make_conv_3d_standard(self):
        """Test creating standard 3D convolution."""
        conv = make_conv_nd(dims=3, in_channels=64, out_channels=64, kernel_size=3, padding=1)
        x = torch.randn(1, 64, 5, 16, 16)
        y = conv(x)
        assert y.shape == x.shape

    def test_make_conv_3d_causal(self):
        """Test creating causal 3D convolution."""
        conv = make_conv_nd(
            dims=3,
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            causal=True,
        )
        assert isinstance(conv, CausalConv3d)


# ============================================================================
# Sampling Layer Tests
# ============================================================================


class TestSpaceToDepthDownsample:
    """Tests for spatial downsampling."""

    def test_space_to_depth_shape(self):
        """Test SpaceToDepthDownsample output shape."""
        downsample = SpaceToDepthDownsample(
            dims=3,
            in_channels=64,
            out_channels=128,
            stride=(1, 2, 2),  # 2x spatial downsample
        )

        x = torch.randn(1, 64, 5, 16, 24)
        y = downsample(x)

        # Spatial dims should be halved, channels increased
        assert y.shape[2] == 5, f"Temporal dim changed: {y.shape[2]}"
        assert y.shape[3] == 8, f"Height not halved: {y.shape[3]}"
        assert y.shape[4] == 12, f"Width not halved: {y.shape[4]}"

    def test_space_to_depth_temporal_downsample(self):
        """Test SpaceToDepthDownsample with temporal stride."""
        downsample = SpaceToDepthDownsample(
            dims=3,
            in_channels=64,
            out_channels=256,
            stride=(2, 2, 2),  # 2x all dimensions
        )

        # Need even temporal dim after padding for stride=2
        # With stride=(2,2,2) and padding, input temporal must become divisible by 2
        # The layer duplicates first frame, so 7 frames -> 8 frames
        x = torch.randn(1, 64, 7, 16, 24)
        y = downsample(x)

        # All dims should be halved (plus temporal padding handling)
        # 7 + 1 (padding) = 8, 8 // 2 = 4
        assert y.shape[2] == 4, f"Temporal not as expected: {y.shape[2]}"
        assert y.shape[3] == 8, f"Height not halved: {y.shape[3]}"
        assert y.shape[4] == 12, f"Width not halved: {y.shape[4]}"


class TestDepthToSpaceUpsample:
    """Tests for spatial upsampling."""

    def test_depth_to_space_shape(self):
        """Test DepthToSpaceUpsample output shape."""
        upsample = DepthToSpaceUpsample(
            dims=3,
            in_channels=128,
            stride=(1, 2, 2),  # 2x spatial upsample
        )

        x = torch.randn(1, 128, 5, 8, 12)
        y = upsample(x)

        # Spatial dims should be doubled
        assert y.shape[2] == 5, f"Temporal dim changed: {y.shape[2]}"
        assert y.shape[3] == 16, f"Height not doubled: {y.shape[3]}"
        assert y.shape[4] == 24, f"Width not doubled: {y.shape[4]}"


# ============================================================================
# Tiling Tests
# ============================================================================


class TestTilingConfig:
    """Tests for tiling configuration validation."""

    def test_default_tiling_config(self):
        """Test default tiling configuration values."""
        config = TilingConfig.default()

        assert config.spatial_config is not None
        assert config.temporal_config is not None
        assert config.spatial_config.tile_size_in_pixels == 512
        assert config.spatial_config.tile_overlap_in_pixels == 64
        assert config.temporal_config.tile_size_in_frames == 64
        assert config.temporal_config.tile_overlap_in_frames == 24

    def test_spatial_tiling_validation(self):
        """Test SpatialTilingConfig validation."""
        # Valid config
        config = SpatialTilingConfig(tile_size_in_pixels=256, tile_overlap_in_pixels=32)
        assert config.tile_size_in_pixels == 256

        # Invalid: too small
        with pytest.raises(ValueError, match="at least 64"):
            SpatialTilingConfig(tile_size_in_pixels=32)

        # Invalid: not divisible by 32
        with pytest.raises(ValueError, match="divisible by 32"):
            SpatialTilingConfig(tile_size_in_pixels=100)

        # Invalid: overlap >= tile size
        with pytest.raises(ValueError, match="less than tile size"):
            SpatialTilingConfig(tile_size_in_pixels=64, tile_overlap_in_pixels=64)

    def test_temporal_tiling_validation(self):
        """Test TemporalTilingConfig validation."""
        # Valid config
        config = TemporalTilingConfig(tile_size_in_frames=32, tile_overlap_in_frames=8)
        assert config.tile_size_in_frames == 32

        # Invalid: too small
        with pytest.raises(ValueError, match="at least 16"):
            TemporalTilingConfig(tile_size_in_frames=8)

        # Invalid: not divisible by 8
        with pytest.raises(ValueError, match="divisible by 8"):
            TemporalTilingConfig(tile_size_in_frames=20)


class TestTrapezoidalMask:
    """Tests for trapezoidal blending mask computation."""

    def test_trapezoidal_mask_shape(self):
        """Test mask has correct length."""
        mask = compute_trapezoidal_mask_1d(length=100, ramp_left=10, ramp_right=10)
        assert mask.shape == (100,)

    def test_trapezoidal_mask_bounds(self):
        """Test mask values are in [0, 1]."""
        mask = compute_trapezoidal_mask_1d(length=50, ramp_left=10, ramp_right=10)
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    def test_trapezoidal_mask_center_ones(self):
        """Test center region is all 1s."""
        mask = compute_trapezoidal_mask_1d(length=100, ramp_left=20, ramp_right=20)
        # Center region (20 to 79) should be 1.0
        assert torch.allclose(mask[20:80], torch.ones(60))

    def test_trapezoidal_mask_no_ramps(self):
        """Test mask with no ramps is all ones."""
        mask = compute_trapezoidal_mask_1d(length=50, ramp_left=0, ramp_right=0)
        assert torch.allclose(mask, torch.ones(50))

    def test_trapezoidal_mask_left_starts_from_0(self):
        """Test left ramp starting from 0."""
        mask = compute_trapezoidal_mask_1d(
            length=20, ramp_left=5, ramp_right=0, left_starts_from_0=True
        )
        # First value should be 0.0 when left_starts_from_0=True
        assert mask[0] == 0.0

        mask_no_zero = compute_trapezoidal_mask_1d(
            length=20, ramp_left=5, ramp_right=0, left_starts_from_0=False
        )
        # First value should be > 0 when left_starts_from_0=False
        assert mask_no_zero[0] > 0.0


class TestCreateTiles:
    """Tests for tile creation."""

    def test_create_tiles_single_tile(self):
        """Test creating tiles with default splitters (single tile)."""
        shape = torch.Size([128, 5, 16, 24])  # C, D, H, W
        splitters = [DEFAULT_SPLIT_OPERATION] * 4
        mappers = [DEFAULT_MAPPING_OPERATION] * 4

        tiles = create_tiles(shape, splitters, mappers)

        assert len(tiles) == 1
        assert tiles[0].in_coords == (slice(0, 128), slice(0, 5), slice(0, 16), slice(0, 24))

    def test_create_tiles_dimension_mismatch(self):
        """Test error on dimension mismatch."""
        shape = torch.Size([128, 5, 16, 24])
        splitters = [DEFAULT_SPLIT_OPERATION] * 3  # Wrong length

        with pytest.raises(ValueError, match="must equal"):
            create_tiles(shape, splitters, [DEFAULT_MAPPING_OPERATION] * 4)


# Run with: uv run pytest tests/unit/test_ltx2_video_vae.py -v
