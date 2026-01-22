"""
Tests for LTX-2 Latent Upsampler implementation.

Last Updated: 2026-01-18

These tests verify the latent upsampler module for LTX-2 video generation.
Tests cover shape correctness, dtype preservation, gradient flow, determinism,
and numerical stability.

Run with: uv run pytest tests/unit/test_upsampler.py -v
"""

import gc

import pytest
import torch
import torch.nn as nn


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def cleanup_gpu():
    """Clean up GPU memory before and after each test."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture
def device():
    """Get available device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# PixelShuffleND Tests
# ============================================================================


class TestPixelShuffleND:
    """Tests for N-dimensional pixel shuffle operation."""

    def test_1d_output_shape(self):
        """Test 1D (temporal) pixel shuffle output shape."""
        from llm_dit.models.ltx2.upsampler import PixelShuffleND

        # dims=1: temporal upsampling
        # Input: (B, C*p1, F, H, W) -> Output: (B, C, F*p1, H, W)
        ps = PixelShuffleND(dims=1, upscale_factors=(2, 2, 2))  # Only first factor used
        x = torch.randn(2, 256, 4, 8, 8)  # 256 = 128 * 2
        out = ps(x)
        assert out.shape == (2, 128, 8, 8, 8), f"Expected (2, 128, 8, 8, 8), got {out.shape}"

    def test_2d_output_shape(self):
        """Test 2D (spatial) pixel shuffle output shape."""
        from llm_dit.models.ltx2.upsampler import PixelShuffleND

        # dims=2: spatial upsampling (height, width)
        # Input: (B, C*p1*p2, H, W) -> Output: (B, C, H*p1, W*p2)
        ps = PixelShuffleND(dims=2, upscale_factors=(2, 2, 2))  # First two factors used
        x = torch.randn(2, 512, 8, 8)  # 512 = 128 * 2 * 2
        out = ps(x)
        assert out.shape == (2, 128, 16, 16), f"Expected (2, 128, 16, 16), got {out.shape}"

    def test_3d_output_shape(self):
        """Test 3D (spatiotemporal) pixel shuffle output shape."""
        from llm_dit.models.ltx2.upsampler import PixelShuffleND

        # dims=3: spatiotemporal upsampling
        # Input: (B, C*p1*p2*p3, D, H, W) -> Output: (B, C, D*p1, H*p2, W*p3)
        ps = PixelShuffleND(dims=3, upscale_factors=(2, 2, 2))
        x = torch.randn(2, 1024, 4, 8, 8)  # 1024 = 128 * 2 * 2 * 2
        out = ps(x)
        assert out.shape == (2, 128, 8, 16, 16), f"Expected (2, 128, 8, 16, 16), got {out.shape}"

    def test_dtype_preservation_float32(self):
        """Test float32 dtype is preserved."""
        from llm_dit.models.ltx2.upsampler import PixelShuffleND

        ps = PixelShuffleND(dims=2, upscale_factors=(2, 2, 2))
        x = torch.randn(1, 512, 4, 4, dtype=torch.float32)
        out = ps(x)
        assert out.dtype == torch.float32

    def test_dtype_preservation_bfloat16(self):
        """Test bfloat16 dtype is preserved."""
        from llm_dit.models.ltx2.upsampler import PixelShuffleND

        ps = PixelShuffleND(dims=2, upscale_factors=(2, 2, 2))
        x = torch.randn(1, 512, 4, 4, dtype=torch.bfloat16)
        out = ps(x)
        assert out.dtype == torch.bfloat16

    def test_gradient_flow(self):
        """Test gradients flow correctly through pixel shuffle."""
        from llm_dit.models.ltx2.upsampler import PixelShuffleND

        ps = PixelShuffleND(dims=2, upscale_factors=(2, 2, 2))
        x = torch.randn(2, 512, 4, 4, requires_grad=True)
        out = ps(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_deterministic_output(self):
        """Test same input produces same output."""
        from llm_dit.models.ltx2.upsampler import PixelShuffleND

        ps = PixelShuffleND(dims=2, upscale_factors=(2, 2, 2))
        x = torch.randn(1, 512, 4, 4)
        out1 = ps(x.clone())
        out2 = ps(x.clone())
        assert torch.allclose(out1, out2)

    def test_no_nan_or_inf(self):
        """Test output contains no NaN or Inf values."""
        from llm_dit.models.ltx2.upsampler import PixelShuffleND

        ps = PixelShuffleND(dims=2, upscale_factors=(2, 2, 2))
        x = torch.randn(2, 512, 8, 8)
        out = ps(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_invalid_dims_raises(self):
        """Test invalid dims value raises assertion."""
        from llm_dit.models.ltx2.upsampler import PixelShuffleND

        with pytest.raises(AssertionError):
            PixelShuffleND(dims=4, upscale_factors=(2, 2, 2))

        with pytest.raises(AssertionError):
            PixelShuffleND(dims=0, upscale_factors=(2, 2, 2))

    def test_asymmetric_upscale_factors(self):
        """Test asymmetric upscale factors work correctly."""
        from llm_dit.models.ltx2.upsampler import PixelShuffleND

        # 3x upscale in height, 2x in width
        ps = PixelShuffleND(dims=2, upscale_factors=(3, 2, 1))
        x = torch.randn(1, 768, 4, 4)  # 768 = 128 * 3 * 2
        out = ps(x)
        assert out.shape == (1, 128, 12, 8), f"Expected (1, 128, 12, 8), got {out.shape}"

    def test_2d_matches_torch_pixelshuffle(self):
        """Test 2D pixel shuffle matches torch.nn.PixelShuffle for 2x2."""
        from llm_dit.models.ltx2.upsampler import PixelShuffleND

        ps_ours = PixelShuffleND(dims=2, upscale_factors=(2, 2, 2))
        ps_torch = nn.PixelShuffle(2)

        x = torch.randn(2, 512, 8, 8)
        out_ours = ps_ours(x)
        out_torch = ps_torch(x)
        assert torch.allclose(out_ours, out_torch, atol=1e-6), "Should match torch.nn.PixelShuffle"

    def test_batch_independence(self):
        """Test each batch item is processed independently."""
        from llm_dit.models.ltx2.upsampler import PixelShuffleND

        ps = PixelShuffleND(dims=2, upscale_factors=(2, 2, 2))
        x1 = torch.randn(1, 512, 4, 4)
        x2 = torch.randn(1, 512, 4, 4)
        x_batched = torch.cat([x1, x2], dim=0)

        out_batched = ps(x_batched)
        out1 = ps(x1)
        out2 = ps(x2)

        assert torch.allclose(out_batched[0], out1[0])
        assert torch.allclose(out_batched[1], out2[0])


# ============================================================================
# ResBlock Tests
# ============================================================================


class TestResBlock:
    """Tests for residual block with GroupNorm and SiLU."""

    def test_2d_shape_preserved(self):
        """Test 2D ResBlock preserves input shape."""
        from llm_dit.models.ltx2.upsampler import ResBlock

        block = ResBlock(channels=512, dims=2)
        x = torch.randn(2, 512, 16, 16)
        out = block(x)
        assert out.shape == x.shape

    def test_3d_shape_preserved(self):
        """Test 3D ResBlock preserves input shape."""
        from llm_dit.models.ltx2.upsampler import ResBlock

        block = ResBlock(channels=512, dims=3)
        x = torch.randn(2, 512, 8, 16, 16)
        out = block(x)
        assert out.shape == x.shape

    def test_residual_connection(self):
        """Test output is not identical to input (residual modifies)."""
        from llm_dit.models.ltx2.upsampler import ResBlock

        block = ResBlock(channels=512, dims=2)
        x = torch.randn(2, 512, 8, 8)
        out = block(x)
        # Should not be identical (transformations applied)
        assert not torch.allclose(out, x)

    def test_groupnorm_32_groups(self):
        """Test GroupNorm uses 32 groups."""
        from llm_dit.models.ltx2.upsampler import ResBlock

        block = ResBlock(channels=512, dims=2)
        assert block.norm1.num_groups == 32
        assert block.norm2.num_groups == 32

    def test_silu_activation(self):
        """Test SiLU activation is used."""
        from llm_dit.models.ltx2.upsampler import ResBlock

        block = ResBlock(channels=512, dims=2)
        assert isinstance(block.activation, nn.SiLU)

    def test_gradient_flow(self):
        """Test gradients flow through ResBlock."""
        from llm_dit.models.ltx2.upsampler import ResBlock

        block = ResBlock(channels=512, dims=2)
        x = torch.randn(2, 512, 8, 8, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None

    def test_dtype_preservation(self):
        """Test dtype is preserved through forward pass."""
        from llm_dit.models.ltx2.upsampler import ResBlock

        block = ResBlock(channels=512, dims=2).to(torch.bfloat16)
        x = torch.randn(1, 512, 8, 8, dtype=torch.bfloat16)
        out = block(x)
        assert out.dtype == torch.bfloat16

    def test_mid_channels_different(self):
        """Test ResBlock with different mid_channels."""
        from llm_dit.models.ltx2.upsampler import ResBlock

        block = ResBlock(channels=512, mid_channels=256, dims=2)
        x = torch.randn(1, 512, 8, 8)
        out = block(x)
        assert out.shape == x.shape
        # Verify mid_channels used in conv1
        assert block.conv1.out_channels == 256


# ============================================================================
# BlurDownsample Tests
# ============================================================================


class TestBlurDownsample:
    """Tests for anti-aliased blur downsampling."""

    def test_2d_shape_stride2(self):
        """Test 2D downsampling with stride 2."""
        from llm_dit.models.ltx2.upsampler import BlurDownsample

        blur = BlurDownsample(dims=2, stride=2)
        x = torch.randn(2, 128, 16, 16)
        out = blur(x)
        assert out.shape == (2, 128, 8, 8)

    def test_2d_shape_stride4(self):
        """Test 2D downsampling with stride 4."""
        from llm_dit.models.ltx2.upsampler import BlurDownsample

        blur = BlurDownsample(dims=2, stride=4)
        x = torch.randn(2, 128, 32, 32)
        out = blur(x)
        assert out.shape == (2, 128, 8, 8)

    def test_3d_spatial_only(self):
        """Test 3D applies blur per-frame on H,W only."""
        from llm_dit.models.ltx2.upsampler import BlurDownsample

        blur = BlurDownsample(dims=3, stride=2)
        x = torch.randn(2, 128, 4, 16, 16)  # B, C, F, H, W
        out = blur(x)
        # Temporal dimension unchanged, spatial halved
        assert out.shape == (2, 128, 4, 8, 8)

    def test_stride1_identity(self):
        """Test stride=1 returns input unchanged."""
        from llm_dit.models.ltx2.upsampler import BlurDownsample

        blur = BlurDownsample(dims=2, stride=1)
        x = torch.randn(2, 128, 16, 16)
        out = blur(x)
        assert torch.allclose(out, x)

    def test_binomial_kernel_shape(self):
        """Test binomial kernel has correct shape."""
        from llm_dit.models.ltx2.upsampler import BlurDownsample

        blur = BlurDownsample(dims=2, stride=2, kernel_size=5)
        assert blur.kernel.shape == (1, 1, 5, 5)

    def test_binomial_kernel_normalized(self):
        """Test binomial kernel sums to 1."""
        from llm_dit.models.ltx2.upsampler import BlurDownsample

        blur = BlurDownsample(dims=2, stride=2, kernel_size=5)
        assert abs(blur.kernel.sum().item() - 1.0) < 1e-5

    def test_gradient_flow(self):
        """Test gradients flow through blur downsample."""
        from llm_dit.models.ltx2.upsampler import BlurDownsample

        blur = BlurDownsample(dims=2, stride=2)
        x = torch.randn(2, 128, 16, 16, requires_grad=True)
        out = blur(x)
        out.sum().backward()
        assert x.grad is not None

    def test_antialias_effect(self):
        """Test blur creates smooth output for high-frequency patterns."""
        from llm_dit.models.ltx2.upsampler import BlurDownsample

        blur = BlurDownsample(dims=2, stride=2)

        # High-frequency alternating pattern with -1 and +1
        x = torch.ones(1, 1, 8, 8)
        x[0, 0, ::2, 1::2] = -1.0  # Alternate in checkerboard
        x[0, 0, 1::2, ::2] = -1.0

        out_blur = blur(x)

        # Blur should average out high frequencies, producing values near 0
        # rather than maintaining extreme +1/-1 values
        assert out_blur.abs().mean() < 0.5, "Blur should smooth high-frequency content"


# ============================================================================
# SpatialRationalResampler Tests
# ============================================================================


class TestSpatialRationalResampler:
    """Tests for rational spatial resampling (fractional scales)."""

    @pytest.mark.parametrize(
        "scale,expected_h,expected_w",
        [
            (0.75, 12, 12),  # 16 * 3/4 = 12
            (1.5, 24, 24),  # 16 * 3/2 = 24
            (2.0, 32, 32),  # 16 * 2 = 32
            (4.0, 64, 64),  # 16 * 4 = 64
        ],
    )
    def test_supported_scales(self, scale, expected_h, expected_w):
        """Test all supported scale factors produce correct output size."""
        from llm_dit.models.ltx2.upsampler import SpatialRationalResampler

        resampler = SpatialRationalResampler(mid_channels=512, scale=scale)
        x = torch.randn(1, 512, 4, 16, 16)  # B, C, F, H, W
        out = resampler(x)
        assert out.shape == (
            1,
            512,
            4,
            expected_h,
            expected_w,
        ), f"Scale {scale}: expected (1, 512, 4, {expected_h}, {expected_w}), got {out.shape}"

    def test_unsupported_scale_raises(self):
        """Test unsupported scale raises ValueError."""
        from llm_dit.models.ltx2.upsampler import SpatialRationalResampler

        with pytest.raises(ValueError, match="Unsupported scale"):
            SpatialRationalResampler(mid_channels=512, scale=3.0)

    def test_rational_mapping_075(self):
        """Test 0.75 scale uses (3, 4) rational."""
        from llm_dit.models.ltx2.upsampler import SpatialRationalResampler

        resampler = SpatialRationalResampler(mid_channels=512, scale=0.75)
        assert resampler.num == 3
        assert resampler.den == 4

    def test_rational_mapping_15(self):
        """Test 1.5 scale uses (3, 2) rational."""
        from llm_dit.models.ltx2.upsampler import SpatialRationalResampler

        resampler = SpatialRationalResampler(mid_channels=512, scale=1.5)
        assert resampler.num == 3
        assert resampler.den == 2

    def test_gradient_flow(self):
        """Test gradients flow through rational resampler."""
        from llm_dit.models.ltx2.upsampler import SpatialRationalResampler

        resampler = SpatialRationalResampler(mid_channels=512, scale=2.0)
        x = torch.randn(1, 512, 2, 8, 8, requires_grad=True)
        out = resampler(x)
        out.sum().backward()
        assert x.grad is not None

    def test_temporal_preserved(self):
        """Test temporal dimension is unchanged."""
        from llm_dit.models.ltx2.upsampler import SpatialRationalResampler

        resampler = SpatialRationalResampler(mid_channels=512, scale=2.0)
        x = torch.randn(2, 512, 8, 16, 16)  # 8 temporal frames
        out = resampler(x)
        assert out.shape[2] == 8, "Temporal dimension should be preserved"

    def test_dtype_preservation(self):
        """Test bfloat16 dtype is preserved."""
        from llm_dit.models.ltx2.upsampler import SpatialRationalResampler

        resampler = SpatialRationalResampler(mid_channels=512, scale=2.0).to(torch.bfloat16)
        x = torch.randn(1, 512, 2, 8, 8, dtype=torch.bfloat16)
        out = resampler(x)
        assert out.dtype == torch.bfloat16

    def test_uses_blur_downsample(self):
        """Test resampler contains BlurDownsample for anti-aliasing."""
        from llm_dit.models.ltx2.upsampler import BlurDownsample, SpatialRationalResampler

        resampler = SpatialRationalResampler(mid_channels=512, scale=0.75)
        assert isinstance(resampler.blur_down, BlurDownsample)

    def test_uses_pixel_shuffle(self):
        """Test resampler contains PixelShuffleND for upsampling."""
        from llm_dit.models.ltx2.upsampler import PixelShuffleND, SpatialRationalResampler

        resampler = SpatialRationalResampler(mid_channels=512, scale=2.0)
        assert isinstance(resampler.pixel_shuffle, PixelShuffleND)


# ============================================================================
# LatentUpsampler Tests
# ============================================================================


class TestLatentUpsampler:
    """Tests for the main LatentUpsampler model."""

    def test_2d_spatial_upsample_shape(self):
        """Test 2D spatial-only upsampling."""
        from llm_dit.models.ltx2.upsampler import LatentUpsampler

        upsampler = LatentUpsampler(
            in_channels=128,
            mid_channels=512,
            dims=2,
            spatial_upsample=True,
            temporal_upsample=False,
        )
        x = torch.randn(1, 128, 4, 16, 16)  # B, C, F, H, W
        out = upsampler(x)
        # Spatial 2x, temporal unchanged
        assert out.shape == (1, 128, 4, 32, 32)

    def test_3d_spatial_upsample_shape(self):
        """Test 3D spatial-only upsampling (per-frame rearrange)."""
        from llm_dit.models.ltx2.upsampler import LatentUpsampler

        upsampler = LatentUpsampler(
            in_channels=128,
            mid_channels=512,
            dims=3,
            spatial_upsample=True,
            temporal_upsample=False,
        )
        x = torch.randn(1, 128, 4, 16, 16)
        out = upsampler(x)
        # Spatial 2x, temporal unchanged
        assert out.shape == (1, 128, 4, 32, 32)

    def test_3d_temporal_upsample_shape(self):
        """Test 3D temporal-only upsampling (with first frame removal)."""
        from llm_dit.models.ltx2.upsampler import LatentUpsampler

        upsampler = LatentUpsampler(
            in_channels=128,
            mid_channels=512,
            dims=3,
            spatial_upsample=False,
            temporal_upsample=True,
        )
        x = torch.randn(1, 128, 4, 16, 16)
        out = upsampler(x)
        # Temporal 2x minus first frame: (4*2 - 1) = 7
        assert out.shape == (1, 128, 7, 16, 16)

    def test_3d_spatiotemporal_upsample_shape(self):
        """Test 3D spatiotemporal upsampling (with first frame removal)."""
        from llm_dit.models.ltx2.upsampler import LatentUpsampler

        upsampler = LatentUpsampler(
            in_channels=128,
            mid_channels=512,
            dims=3,
            spatial_upsample=True,
            temporal_upsample=True,
        )
        x = torch.randn(1, 128, 4, 16, 16)
        out = upsampler(x)
        # Temporal 2x minus first frame: (4*2 - 1) = 7, spatial 2x
        assert out.shape == (1, 128, 7, 32, 32)

    def test_rational_resampler_shape(self):
        """Test rational resampler for fractional scales."""
        from llm_dit.models.ltx2.upsampler import LatentUpsampler

        upsampler = LatentUpsampler(
            in_channels=128,
            mid_channels=512,
            dims=3,
            spatial_upsample=True,
            temporal_upsample=False,
            spatial_scale=1.5,
            rational_resampler=True,
        )
        x = torch.randn(1, 128, 4, 16, 16)
        out = upsampler(x)
        # 1.5x spatial scale
        assert out.shape == (1, 128, 4, 24, 24)

    def test_channel_architecture(self):
        """Test 128 -> 512 -> 128 channel pattern."""
        from llm_dit.models.ltx2.upsampler import LatentUpsampler

        upsampler = LatentUpsampler(in_channels=128, mid_channels=512, dims=2)
        assert upsampler.initial_conv.in_channels == 128
        assert upsampler.initial_conv.out_channels == 512
        assert upsampler.final_conv.in_channels == 512
        assert upsampler.final_conv.out_channels == 128

    def test_num_resblocks_per_stage(self):
        """Test correct number of ResBlocks per stage (default 4)."""
        from llm_dit.models.ltx2.upsampler import LatentUpsampler

        upsampler = LatentUpsampler(dims=2, num_blocks_per_stage=4)
        assert len(upsampler.res_blocks) == 4
        assert len(upsampler.post_upsample_res_blocks) == 4

    def test_gradient_flow(self):
        """Test gradients flow through entire model."""
        from llm_dit.models.ltx2.upsampler import LatentUpsampler

        upsampler = LatentUpsampler(dims=2, num_blocks_per_stage=2)
        x = torch.randn(1, 128, 2, 8, 8, requires_grad=True)
        out = upsampler(x)
        out.sum().backward()
        assert x.grad is not None

    def test_dtype_preservation(self):
        """Test bfloat16 dtype is preserved."""
        from llm_dit.models.ltx2.upsampler import LatentUpsampler

        upsampler = LatentUpsampler(dims=2, num_blocks_per_stage=1).to(torch.bfloat16)
        x = torch.randn(1, 128, 2, 8, 8, dtype=torch.bfloat16)
        out = upsampler(x)
        assert out.dtype == torch.bfloat16

    def test_no_upsample_raises(self):
        """Test must have at least one upsample mode."""
        from llm_dit.models.ltx2.upsampler import LatentUpsampler

        with pytest.raises(ValueError, match="spatial_upsample or temporal_upsample"):
            LatentUpsampler(dims=2, spatial_upsample=False, temporal_upsample=False)

    def test_deterministic_output(self):
        """Test same input produces same output in inference mode."""
        from llm_dit.models.ltx2.upsampler import LatentUpsampler

        upsampler = LatentUpsampler(dims=2, num_blocks_per_stage=1)
        # Switch to inference mode (no dropout, etc)
        with torch.inference_mode():
            x = torch.randn(1, 128, 2, 8, 8)
            out1 = upsampler(x.clone())
            out2 = upsampler(x.clone())
        assert torch.allclose(out1, out2)

    def test_no_nan_or_inf(self):
        """Test output contains no NaN or Inf."""
        from llm_dit.models.ltx2.upsampler import LatentUpsampler

        upsampler = LatentUpsampler(dims=2, num_blocks_per_stage=1)
        x = torch.randn(2, 128, 4, 16, 16)
        out = upsampler(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_batch_independence(self):
        """Test each batch item is processed independently."""
        from llm_dit.models.ltx2.upsampler import LatentUpsampler

        upsampler = LatentUpsampler(dims=2, num_blocks_per_stage=1)
        # Use inference mode for determinism
        with torch.inference_mode():
            x1 = torch.randn(1, 128, 2, 8, 8)
            x2 = torch.randn(1, 128, 2, 8, 8)
            x_batched = torch.cat([x1, x2], dim=0)

            out_batched = upsampler(x_batched)
            out1 = upsampler(x1)
            out2 = upsampler(x2)

        assert torch.allclose(out_batched[0], out1[0], atol=1e-5)
        assert torch.allclose(out_batched[1], out2[0], atol=1e-5)


# ============================================================================
# Model Configurator Tests
# ============================================================================


class TestLatentUpsamplerConfigurator:
    """Tests for model configuration and loading."""

    def test_from_config_defaults(self):
        """Test creating model from empty config uses defaults."""
        from llm_dit.models.ltx2.upsampler import LatentUpsamplerConfigurator

        model = LatentUpsamplerConfigurator.from_config({})
        assert model.in_channels == 128
        assert model.mid_channels == 512
        assert model.num_blocks_per_stage == 4
        assert model.dims == 3
        assert model.spatial_upsample is True
        assert model.temporal_upsample is False
        assert model.spatial_scale == 2.0
        assert model.rational_resampler is False

    def test_from_config_custom(self):
        """Test creating model from custom config."""
        from llm_dit.models.ltx2.upsampler import LatentUpsamplerConfigurator

        config = {
            "in_channels": 64,
            "mid_channels": 256,
            "num_blocks_per_stage": 2,
            "dims": 2,
            "spatial_scale": 4.0,
        }
        model = LatentUpsamplerConfigurator.from_config(config)
        assert model.in_channels == 64
        assert model.mid_channels == 256
        assert model.num_blocks_per_stage == 2
        assert model.dims == 2
        assert model.spatial_scale == 4.0


# ============================================================================
# Integration: upsample_video Helper
# ============================================================================


class TestUpsampleVideo:
    """Tests for the upsample_video helper function."""

    def test_unnormalize_upsample_normalize_roundtrip(self):
        """Test normalization is properly applied around upsampling."""
        from unittest.mock import MagicMock

        from llm_dit.models.ltx2.upsampler import LatentUpsampler, upsample_video

        # Create mock video encoder with per_channel_statistics
        mock_stats = MagicMock()
        # Use side_effect to return transformed tensor while keeping MagicMock methods
        mock_stats.un_normalize.side_effect = lambda x: x * 2.0  # Scale up
        mock_stats.normalize.side_effect = lambda x: x / 2.0  # Scale down

        mock_encoder = MagicMock()
        mock_encoder.per_channel_statistics = mock_stats

        upsampler = LatentUpsampler(dims=2, num_blocks_per_stage=1)
        latent = torch.randn(1, 128, 2, 8, 8)

        out = upsample_video(latent, mock_encoder, upsampler)

        # Verify shape is upsampled
        assert out.shape == (1, 128, 2, 16, 16)

        # Verify normalize and un_normalize were called
        mock_stats.un_normalize.assert_called_once()
        mock_stats.normalize.assert_called_once()


# ============================================================================
# GPU Tests (conditional)
# ============================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGPU:
    """GPU-specific tests."""

    def test_upsampler_cuda_forward(self):
        """Test upsampler forward pass on CUDA."""
        from llm_dit.models.ltx2.upsampler import LatentUpsampler

        upsampler = LatentUpsampler(dims=2, num_blocks_per_stage=1).cuda()
        x = torch.randn(1, 128, 2, 16, 16, device="cuda")
        out = upsampler(x)
        assert out.device.type == "cuda"
        assert out.shape == (1, 128, 2, 32, 32)

    def test_upsampler_cuda_bfloat16(self):
        """Test upsampler with bfloat16 on CUDA."""
        from llm_dit.models.ltx2.upsampler import LatentUpsampler

        upsampler = LatentUpsampler(dims=2, num_blocks_per_stage=1).cuda().to(torch.bfloat16)
        x = torch.randn(1, 128, 2, 16, 16, device="cuda", dtype=torch.bfloat16)
        out = upsampler(x)
        assert out.dtype == torch.bfloat16


# Run with: uv run pytest tests/unit/test_upsampler.py -v
# Run GPU tests: uv run pytest tests/unit/test_upsampler.py -v -k GPU
