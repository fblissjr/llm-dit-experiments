#!/usr/bin/env python3
"""
Latent Upsampler Integration Tests

Last Updated: 2026-01-18

Tests the LatentUpsampler in realistic scenarios with proper GPU integration
and normalization handling.

Usage:
    uv run pytest tests/integration/test_upsampler_integration.py -v
"""

import gc

import pytest
import torch

# Skip all tests if CUDA not available
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def cleanup_gpu():
    """Free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class TestUpsamplerGPUIntegration:
    """GPU integration tests for LatentUpsampler."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and cleanup for each test."""
        cleanup_gpu()
        yield
        cleanup_gpu()

    def test_realistic_latent_shape_spatial_2x(self):
        """Test spatial 2x upsampling with realistic LTX-2 latent dimensions."""
        from llm_dit.models.ltx2.upsampler import LatentUpsampler

        # LTX-2 latent: 128 channels, 5 frames (33 pixels), 16x24 spatial (512x768)
        upsampler = LatentUpsampler(
            in_channels=128,
            mid_channels=512,
            dims=2,
            spatial_upsample=True,
            temporal_upsample=False,
            num_blocks_per_stage=4,
        ).cuda().to(torch.bfloat16)

        x = torch.randn(1, 128, 5, 16, 24, device="cuda", dtype=torch.bfloat16)
        out = upsampler(x)

        # 2x spatial: 16→32, 24→48
        assert out.shape == (1, 128, 5, 32, 48)
        assert out.device.type == "cuda"
        assert out.dtype == torch.bfloat16

    def test_temporal_upsample_with_frame_stripping(self):
        """Test temporal upsampling properly removes first frame."""
        from llm_dit.models.ltx2.upsampler import LatentUpsampler

        upsampler = LatentUpsampler(
            in_channels=128,
            mid_channels=512,
            dims=3,
            spatial_upsample=False,
            temporal_upsample=True,
            num_blocks_per_stage=2,
        ).cuda().to(torch.bfloat16)

        # 5 frames in latent space
        x = torch.randn(1, 128, 5, 16, 16, device="cuda", dtype=torch.bfloat16)
        out = upsampler(x)

        # Temporal 2x minus first frame: 5*2 - 1 = 9
        assert out.shape == (1, 128, 9, 16, 16)

    def test_rational_resampler_15x(self):
        """Test 1.5x spatial scaling via rational resampler."""
        from llm_dit.models.ltx2.upsampler import LatentUpsampler

        upsampler = LatentUpsampler(
            in_channels=128,
            mid_channels=512,
            dims=3,
            spatial_upsample=True,
            temporal_upsample=False,
            spatial_scale=1.5,
            rational_resampler=True,
            num_blocks_per_stage=2,
        ).cuda().to(torch.bfloat16)

        x = torch.randn(1, 128, 4, 16, 16, device="cuda", dtype=torch.bfloat16)
        out = upsampler(x)

        # 1.5x spatial: 16→24
        assert out.shape == (1, 128, 4, 24, 24)

    def test_batch_processing(self):
        """Test batch processing produces correct shapes."""
        from llm_dit.models.ltx2.upsampler import LatentUpsampler

        upsampler = LatentUpsampler(
            dims=2,
            num_blocks_per_stage=1,
        ).cuda().to(torch.bfloat16)

        # Batch of 4
        x = torch.randn(4, 128, 3, 16, 16, device="cuda", dtype=torch.bfloat16)
        out = upsampler(x)

        assert out.shape == (4, 128, 3, 32, 32)

    def test_gradient_flow_on_gpu(self):
        """Test gradients flow correctly during GPU training."""
        from llm_dit.models.ltx2.upsampler import LatentUpsampler

        upsampler = LatentUpsampler(
            dims=2,
            num_blocks_per_stage=2,
        ).cuda().to(torch.float32)  # float32 for gradient stability

        x = torch.randn(1, 128, 2, 8, 8, device="cuda", dtype=torch.float32, requires_grad=True)
        out = upsampler(x)

        # Simulate loss and backprop
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()


class TestUpsamplerWithMockNormalization:
    """Test upsampler with VAE-style normalization flow."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and cleanup for each test."""
        cleanup_gpu()
        yield
        cleanup_gpu()

    def test_normalization_roundtrip_preserves_scale(self):
        """Test un_normalize → upsample → normalize preserves reasonable bounds."""
        from llm_dit.models.ltx2.upsampler import LatentUpsampler

        # Simulate VAE per_channel_statistics with simple scaling
        class MockStats:
            def un_normalize(self, x: torch.Tensor) -> torch.Tensor:
                return x * 10.0  # Scale up from normalized range

            def normalize(self, x: torch.Tensor) -> torch.Tensor:
                return x / 10.0  # Scale back down

        upsampler = LatentUpsampler(dims=2, num_blocks_per_stage=1).cuda().to(torch.bfloat16)

        # Simulate normalized latent (small values)
        latent = torch.randn(1, 128, 2, 8, 8, device="cuda", dtype=torch.bfloat16) * 0.5

        # Apply the pipeline
        stats = MockStats()
        unnorm = stats.un_normalize(latent)
        upsampled = upsampler(unnorm)
        renorm = stats.normalize(upsampled)

        # Output should be in reasonable range (not exploded)
        assert renorm.abs().max() < 100.0, "Output should remain in reasonable bounds"
        assert renorm.shape == (1, 128, 2, 16, 16)


class TestUpsamplerConfigurator:
    """Test model creation from config dictionaries."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and cleanup for each test."""
        cleanup_gpu()
        yield
        cleanup_gpu()

    def test_from_config_on_gpu(self):
        """Test creating model from config and moving to GPU."""
        from llm_dit.models.ltx2.upsampler import LatentUpsamplerConfigurator

        config = {
            "in_channels": 128,
            "mid_channels": 256,  # Smaller for test
            "num_blocks_per_stage": 2,
            "dims": 2,
            "spatial_upsample": True,
            "temporal_upsample": False,
        }

        model = LatentUpsamplerConfigurator.from_config(config)
        model = model.cuda().to(torch.bfloat16)

        x = torch.randn(1, 128, 2, 16, 16, device="cuda", dtype=torch.bfloat16)
        out = model(x)

        assert out.shape == (1, 128, 2, 32, 32)
        assert out.device.type == "cuda"


# Run with: uv run pytest tests/integration/test_upsampler_integration.py -v
