"""Tests for long prompt handling utilities."""

import pytest
import torch

from llm_dit.utils.long_prompt import (
    LongPromptMode,
    compress_embeddings,
    estimate_quality_loss,
    _interpolate_embeddings,
    _pool_embeddings,
    _attention_pool_embeddings,
)


class TestLongPromptMode:
    """Test LongPromptMode enum."""

    def test_enum_values(self):
        """Test all enum values exist."""
        assert LongPromptMode.TRUNCATE == "truncate"
        assert LongPromptMode.INTERPOLATE == "interpolate"
        assert LongPromptMode.POOL == "pool"
        assert LongPromptMode.ATTENTION_POOL == "attention_pool"

    def test_enum_is_string(self):
        """Test enum values are strings."""
        for mode in LongPromptMode:
            assert isinstance(mode.value, str)


class TestCompressEmbeddings:
    """Test compress_embeddings function."""

    def test_no_compression_when_within_limit(self):
        """Test that embeddings within limit are returned unchanged."""
        embeddings = torch.randn(512, 2560)
        result = compress_embeddings(embeddings, max_len=1024, mode="truncate")
        assert torch.equal(result, embeddings)
        assert result.shape == (512, 2560)

    def test_no_compression_at_exact_limit(self):
        """Test that embeddings at exact limit are returned unchanged."""
        embeddings = torch.randn(1024, 2560)
        result = compress_embeddings(embeddings, max_len=1024, mode="truncate")
        assert torch.equal(result, embeddings)
        assert result.shape == (1024, 2560)

    def test_truncate_mode(self):
        """Test truncation mode cuts off at max_len."""
        embeddings = torch.randn(1500, 2560)
        result = compress_embeddings(embeddings, max_len=1024, mode="truncate")
        assert result.shape == (1024, 2560)
        # Truncated result should match first max_len tokens
        assert torch.equal(result, embeddings[:1024])

    def test_interpolate_mode(self):
        """Test interpolation mode resamples embeddings."""
        embeddings = torch.randn(1500, 2560)
        result = compress_embeddings(embeddings, max_len=1024, mode="interpolate")
        assert result.shape == (1024, 2560)
        # Result should not be simple truncation
        assert not torch.equal(result, embeddings[:1024])

    def test_pool_mode(self):
        """Test pooling mode averages embeddings."""
        embeddings = torch.randn(1500, 2560)
        result = compress_embeddings(embeddings, max_len=1024, mode="pool")
        assert result.shape == (1024, 2560)
        # Result should not be simple truncation
        assert not torch.equal(result, embeddings[:1024])

    def test_attention_pool_mode(self):
        """Test attention pooling mode."""
        embeddings = torch.randn(1500, 2560)
        result = compress_embeddings(embeddings, max_len=1024, mode="attention_pool")
        assert result.shape == (1024, 2560)
        # Result should not be simple truncation
        assert not torch.equal(result, embeddings[:1024])

    def test_invalid_mode_raises_error(self):
        """Test that invalid mode raises ValueError."""
        embeddings = torch.randn(1500, 2560)
        with pytest.raises(ValueError, match="Unknown mode"):
            compress_embeddings(embeddings, max_len=1024, mode="invalid")

    def test_preserves_dtype(self):
        """Test that compression preserves tensor dtype."""
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            embeddings = torch.randn(1500, 2560, dtype=dtype)
            for mode in ["truncate", "interpolate", "pool", "attention_pool"]:
                result = compress_embeddings(embeddings, max_len=1024, mode=mode)
                assert result.dtype == dtype, f"Mode {mode} changed dtype from {dtype} to {result.dtype}"

    def test_preserves_device(self):
        """Test that compression preserves tensor device."""
        embeddings = torch.randn(1500, 2560)
        for mode in ["truncate", "interpolate", "pool", "attention_pool"]:
            result = compress_embeddings(embeddings, max_len=1024, mode=mode)
            assert result.device == embeddings.device

    def test_custom_max_len(self):
        """Test compression with custom max_len."""
        embeddings = torch.randn(1000, 2560)
        result = compress_embeddings(embeddings, max_len=500, mode="truncate")
        assert result.shape == (500, 2560)

    def test_extreme_compression_ratio(self):
        """Test compression with very high ratio."""
        embeddings = torch.randn(5000, 2560)
        for mode in ["truncate", "interpolate", "pool", "attention_pool"]:
            result = compress_embeddings(embeddings, max_len=1024, mode=mode)
            assert result.shape == (1024, 2560)


class TestInterpolateEmbeddings:
    """Test _interpolate_embeddings helper function."""

    def test_basic_interpolation(self):
        """Test basic interpolation works."""
        embeddings = torch.randn(2000, 2560)
        result = _interpolate_embeddings(embeddings, target_len=1024)
        assert result.shape == (1024, 2560)

    def test_preserves_first_and_last(self):
        """Test that first and last positions are approximately preserved."""
        embeddings = torch.randn(2000, 2560)
        result = _interpolate_embeddings(embeddings, target_len=1024)
        # First position should be very close to original first
        assert torch.allclose(result[0], embeddings[0], atol=1e-5)
        # Last position should be very close to original last
        assert torch.allclose(result[-1], embeddings[-1], atol=1e-5)

    def test_small_compression(self):
        """Test small compression ratios."""
        embeddings = torch.randn(1100, 2560)
        result = _interpolate_embeddings(embeddings, target_len=1024)
        assert result.shape == (1024, 2560)


class TestPoolEmbeddings:
    """Test _pool_embeddings helper function."""

    def test_basic_pooling(self):
        """Test basic pooling works."""
        embeddings = torch.randn(2000, 2560)
        result = _pool_embeddings(embeddings, target_len=1024)
        assert result.shape == (1024, 2560)

    def test_pooling_averages_regions(self):
        """Test that pooling creates regional averages."""
        # Create embeddings with clear patterns
        embeddings = torch.zeros(2000, 2560)
        embeddings[:1000] = 1.0
        embeddings[1000:] = -1.0

        result = _pool_embeddings(embeddings, target_len=1024)
        # First half should be close to 1.0, second half close to -1.0
        assert result[:512].mean() > 0.5
        assert result[512:].mean() < -0.5


class TestAttentionPoolEmbeddings:
    """Test _attention_pool_embeddings helper function."""

    def test_basic_attention_pooling(self):
        """Test basic attention pooling works."""
        embeddings = torch.randn(2000, 2560)
        result = _attention_pool_embeddings(embeddings, target_len=1024)
        assert result.shape == (1024, 2560)

    def test_weights_important_tokens(self):
        """Test that tokens with higher norms get more weight."""
        # Create embeddings where some tokens have much higher norms
        embeddings = torch.randn(2000, 2560) * 0.1
        # Make every 10th token have high norm
        embeddings[::10] *= 10

        result = _attention_pool_embeddings(embeddings, target_len=1024)
        # Result should exist and have correct shape
        assert result.shape == (1024, 2560)
        # Average norm of result should be influenced by high-norm tokens
        assert result.norm(dim=-1).mean() > embeddings.norm(dim=-1).mean() * 0.5


class TestEstimateQualityLoss:
    """Test estimate_quality_loss function."""

    def test_no_compression(self):
        """Test no compression case."""
        result = estimate_quality_loss(512, 1024, "truncate")
        assert "None" in result

    def test_truncate_high_loss(self):
        """Test high loss estimation for truncate."""
        result = estimate_quality_loss(3000, 1024, "truncate")
        assert "HIGH" in result

    def test_truncate_medium_loss(self):
        """Test medium loss estimation for truncate."""
        result = estimate_quality_loss(1400, 1024, "truncate")
        assert "Medium" in result

    def test_truncate_low_loss(self):
        """Test low loss estimation for truncate."""
        result = estimate_quality_loss(1200, 1024, "truncate")
        assert "Low" in result

    def test_interpolate_high_loss(self):
        """Test high loss estimation for interpolate."""
        result = estimate_quality_loss(3000, 1024, "interpolate")
        assert "HIGH" in result

    def test_pool_loss_levels(self):
        """Test pool loss estimation."""
        high = estimate_quality_loss(3000, 1024, "pool")
        assert "HIGH" in high

        medium = estimate_quality_loss(1600, 1024, "pool")
        assert "Medium" in medium

    def test_attention_pool_loss_levels(self):
        """Test attention_pool loss estimation."""
        high = estimate_quality_loss(3000, 1024, "attention_pool")
        assert "HIGH" in high or "Medium" in high  # Attention pool mitigates somewhat

    def test_unknown_mode(self):
        """Test unknown mode returns Unknown."""
        result = estimate_quality_loss(1500, 1024, "unknown_mode")
        assert "Unknown" in result


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_token_embedding(self):
        """Test with single token embedding."""
        embeddings = torch.randn(1, 2560)
        for mode in ["truncate", "interpolate", "pool", "attention_pool"]:
            result = compress_embeddings(embeddings, max_len=1024, mode=mode)
            assert torch.equal(result, embeddings)

    def test_target_len_one(self):
        """Test compressing to single token."""
        embeddings = torch.randn(100, 2560)
        result = compress_embeddings(embeddings, max_len=1, mode="pool")
        assert result.shape == (1, 2560)

    def test_large_hidden_dim(self):
        """Test with large hidden dimension."""
        embeddings = torch.randn(1500, 4096)
        result = compress_embeddings(embeddings, max_len=1024, mode="interpolate")
        assert result.shape == (1024, 4096)

    def test_small_hidden_dim(self):
        """Test with small hidden dimension."""
        embeddings = torch.randn(1500, 64)
        result = compress_embeddings(embeddings, max_len=1024, mode="interpolate")
        assert result.shape == (1024, 64)

    def test_just_over_limit(self):
        """Test compression when just over limit."""
        embeddings = torch.randn(1025, 2560)
        result = compress_embeddings(embeddings, max_len=1024, mode="interpolate")
        assert result.shape == (1024, 2560)
