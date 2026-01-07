"""
Unit tests for llm_dit.vl.blending module.

Tests cover:
- Embedding scaling and normalization
- Blend functions (standard, interpolate, per-token, style)
- Outlier dimension masking
- AdaIN blending
- Style delta arithmetic
"""

from unittest.mock import MagicMock, patch
import pytest
import torch
import numpy as np


# ============================================================================
# Scaling and Normalization Tests
# ============================================================================

class TestScaleEmbeddings:
    """Test scale_embeddings function."""

    def test_scale_embeddings_basic(self):
        """Test basic embedding scaling."""
        from llm_dit.vl.blending import scale_embeddings

        embeddings = torch.randn(100, 2560) * 10  # std ~10
        target_std = 61.0

        scaled = scale_embeddings(embeddings, target_std=target_std)

        # Check std is close to target
        assert abs(scaled.std().item() - target_std) < 5

    def test_scale_embeddings_default_target(self):
        """Test scaling with default target std."""
        from llm_dit.vl.blending import scale_embeddings, DEFAULT_TARGET_STD

        embeddings = torch.randn(100, 2560) * 20

        scaled = scale_embeddings(embeddings)

        assert abs(scaled.std().item() - DEFAULT_TARGET_STD) < 5

    def test_scale_embeddings_zero_std(self):
        """Test scaling handles zero std gracefully."""
        from llm_dit.vl.blending import scale_embeddings

        embeddings = torch.zeros(100, 2560)

        # Should not raise, returns unchanged
        scaled = scale_embeddings(embeddings)
        assert torch.allclose(scaled, embeddings)


class TestNormalizePerDimension:
    """Test normalize_per_dimension function."""

    def test_normalize_per_dimension_with_stats(self, tmp_path):
        """Test per-dimension normalization with reference stats."""
        from llm_dit.vl.blending import normalize_per_dimension

        # Create embeddings with varying per-dim stats
        embeddings = torch.randn(100, 2560)
        # Make some dimensions have larger variance
        embeddings[:, 0] *= 100

        with patch('llm_dit.vl.blending._load_reference_stats') as mock_stats:
            mock_stats.return_value = {
                "per_dim_mean": np.zeros(2560),
                "per_dim_std": np.ones(2560) * 10,
                "global_mean": 0.0,
                "global_std": 10.0,
            }

            normalized = normalize_per_dimension(embeddings)

            # Shape should be preserved
            assert normalized.shape == embeddings.shape

    def test_normalize_per_dimension_fallback(self):
        """Test per-dim normalization falls back when no stats."""
        from llm_dit.vl.blending import normalize_per_dimension

        embeddings = torch.randn(100, 2560) * 5

        with patch('llm_dit.vl.blending._load_reference_stats', return_value={}):
            # Should fall back to global scaling
            normalized = normalize_per_dimension(embeddings)
            assert normalized.shape == embeddings.shape


class TestNormalizeHybrid:
    """Test normalize_hybrid function."""

    def test_normalize_hybrid_basic(self):
        """Test hybrid normalization."""
        from llm_dit.vl.blending import normalize_hybrid

        embeddings = torch.randn(100, 2560) * 5

        with patch('llm_dit.vl.blending._load_reference_stats') as mock_stats:
            mock_stats.return_value = {
                "per_dim_mean": np.zeros(2560),
                "per_dim_std": np.ones(2560) * 10,
                "global_mean": 0.0,
                "global_std": 10.0,
            }

            hybrid = normalize_hybrid(embeddings)
            assert hybrid.shape == embeddings.shape


# ============================================================================
# Blend Functions Tests
# ============================================================================

class TestBlendEmbeddings:
    """Test blend_embeddings function."""

    def test_blend_embeddings_pure_text(self):
        """Test blend with alpha=0 returns pure text."""
        from llm_dit.vl.blending import blend_embeddings

        vl_emb = torch.randn(100, 2560)
        text_emb = torch.randn(100, 2560)

        blended = blend_embeddings(vl_emb, text_emb, alpha=0.0)

        assert torch.allclose(blended, text_emb)

    def test_blend_embeddings_pure_vl(self):
        """Test blend with alpha=1 returns pure VL."""
        from llm_dit.vl.blending import blend_embeddings

        vl_emb = torch.randn(100, 2560)
        text_emb = torch.randn(100, 2560)

        blended = blend_embeddings(vl_emb, text_emb, alpha=1.0)

        assert torch.allclose(blended, vl_emb[:100])  # Truncated to match

    def test_blend_embeddings_interpolation(self):
        """Test linear interpolation at alpha=0.5."""
        from llm_dit.vl.blending import blend_embeddings

        vl_emb = torch.randn(100, 2560)
        text_emb = torch.randn(100, 2560)

        blended = blend_embeddings(vl_emb, text_emb, alpha=0.5)

        expected = 0.5 * vl_emb + 0.5 * text_emb
        assert torch.allclose(blended, expected)

    def test_blend_embeddings_length_mismatch(self):
        """Test blending with different sequence lengths."""
        from llm_dit.vl.blending import blend_embeddings

        vl_emb = torch.randn(150, 2560)  # Longer
        text_emb = torch.randn(100, 2560)  # Shorter

        blended = blend_embeddings(vl_emb, text_emb, alpha=0.5, match_lengths=True)

        # Should be truncated to shorter length
        assert blended.shape[0] == 100

    def test_blend_embeddings_invalid_alpha(self):
        """Test blend raises on invalid alpha."""
        from llm_dit.vl.blending import blend_embeddings

        vl_emb = torch.randn(100, 2560)
        text_emb = torch.randn(100, 2560)

        with pytest.raises(ValueError):
            blend_embeddings(vl_emb, text_emb, alpha=1.5)

        with pytest.raises(ValueError):
            blend_embeddings(vl_emb, text_emb, alpha=-0.1)


class TestBlendInterpolate:
    """Test blend_interpolate function."""

    def test_blend_interpolate_pure_text(self):
        """Test interpolate blend with alpha=0."""
        from llm_dit.vl.blending import blend_interpolate

        vl_emb = torch.randn(150, 2560)
        text_emb = torch.randn(100, 2560)

        blended = blend_interpolate(vl_emb, text_emb, alpha=0.0)

        assert torch.allclose(blended, text_emb)

    def test_blend_interpolate_preserves_length(self):
        """Test interpolate blend preserves text length."""
        from llm_dit.vl.blending import blend_interpolate

        vl_emb = torch.randn(200, 2560)
        text_emb = torch.randn(100, 2560)

        blended = blend_interpolate(vl_emb, text_emb, alpha=0.5)

        # Output should match text length
        assert blended.shape == text_emb.shape


class TestBlendPerToken:
    """Test blend_per_token function."""

    def test_blend_per_token_uniform(self):
        """Test per-token blend with uniform alphas."""
        from llm_dit.vl.blending import blend_per_token

        vl_emb = torch.randn(100, 2560)
        text_emb = torch.randn(100, 2560)
        alphas = torch.full((100,), 0.5)

        blended = blend_per_token(vl_emb, text_emb, alphas)

        expected = 0.5 * vl_emb + 0.5 * text_emb
        assert torch.allclose(blended, expected)

    def test_blend_per_token_varying(self):
        """Test per-token blend with varying alphas."""
        from llm_dit.vl.blending import blend_per_token

        vl_emb = torch.randn(100, 2560)
        text_emb = torch.randn(100, 2560)
        alphas = torch.linspace(0, 1, 100)

        blended = blend_per_token(vl_emb, text_emb, alphas)

        # First token should be pure text (alpha=0)
        assert torch.allclose(blended[0], text_emb[0], atol=1e-5)
        # Last token should be pure VL (alpha=1)
        assert torch.allclose(blended[-1], vl_emb[-1], atol=1e-5)

    def test_blend_per_token_from_list(self):
        """Test per-token blend with list of alphas."""
        from llm_dit.vl.blending import blend_per_token

        vl_emb = torch.randn(5, 2560)
        text_emb = torch.randn(5, 2560)
        alphas = [0.0, 0.25, 0.5, 0.75, 1.0]

        blended = blend_per_token(vl_emb, text_emb, alphas)

        assert blended.shape == text_emb.shape


class TestCreateGraduatedAlpha:
    """Test create_graduated_alpha function."""

    def test_create_graduated_alpha_linear(self):
        """Test linear graduated alpha."""
        from llm_dit.vl.blending import create_graduated_alpha

        alphas = create_graduated_alpha(
            seq_len=10,
            start_alpha=0.0,
            end_alpha=1.0,
            curve="linear",
        )

        assert len(alphas) == 10
        assert alphas[0].item() == pytest.approx(0.0)
        assert alphas[-1].item() == pytest.approx(1.0)
        assert alphas[5].item() == pytest.approx(0.5, abs=0.1)

    def test_create_graduated_alpha_ease_in(self):
        """Test ease-in graduated alpha."""
        from llm_dit.vl.blending import create_graduated_alpha

        alphas = create_graduated_alpha(
            seq_len=10,
            start_alpha=0.0,
            end_alpha=1.0,
            curve="ease_in",
        )

        # Ease-in starts slow
        assert alphas[2].item() < 0.2  # Should be less than linear

    def test_create_graduated_alpha_ease_out(self):
        """Test ease-out graduated alpha."""
        from llm_dit.vl.blending import create_graduated_alpha

        alphas = create_graduated_alpha(
            seq_len=10,
            start_alpha=0.0,
            end_alpha=1.0,
            curve="ease_out",
        )

        # Ease-out starts fast
        assert alphas[2].item() > 0.3  # Should be more than linear

    def test_create_graduated_alpha_invalid_curve(self):
        """Test invalid curve raises error."""
        from llm_dit.vl.blending import create_graduated_alpha

        with pytest.raises(ValueError):
            create_graduated_alpha(10, 0, 1, curve="invalid")


# ============================================================================
# Outlier Masking Tests
# ============================================================================

class TestMaskOutlierDimensions:
    """Test mask_outlier_dimensions function."""

    def test_mask_outlier_dimensions_no_outliers(self):
        """Test masking when no outliers present."""
        from llm_dit.vl.blending import mask_outlier_dimensions

        embeddings = torch.randn(100, 2560)

        with patch('llm_dit.vl.blending._load_reference_stats') as mock_stats:
            mock_stats.return_value = {
                "per_dim_std": np.ones(2560) * embeddings.std().item(),
            }

            masked, info = mask_outlier_dimensions(embeddings, threshold=10.0)

            assert info["masked_dimensions"] == []

    def test_mask_outlier_dimensions_zero_mode(self):
        """Test masking with 'zero' mode."""
        from llm_dit.vl.blending import mask_outlier_dimensions

        embeddings = torch.randn(100, 2560)
        # Make dimension 0 an extreme outlier
        embeddings[:, 0] *= 1000

        with patch('llm_dit.vl.blending._load_reference_stats') as mock_stats:
            ref_std = np.ones(2560)
            mock_stats.return_value = {"per_dim_std": ref_std}

            masked, info = mask_outlier_dimensions(
                embeddings,
                threshold=10.0,
                mode="zero",
            )

            assert 0 in info["masked_dimensions"]
            # Dimension 0 should be zeroed
            assert masked[:, 0].abs().max().item() == 0.0

    def test_mask_outlier_dimensions_clamp_mode(self):
        """Test masking with 'clamp' mode."""
        from llm_dit.vl.blending import mask_outlier_dimensions

        embeddings = torch.randn(100, 2560)
        embeddings[:, 0] *= 1000

        with patch('llm_dit.vl.blending._load_reference_stats') as mock_stats:
            ref_std = np.ones(2560)
            mock_stats.return_value = {"per_dim_std": ref_std}

            masked, info = mask_outlier_dimensions(
                embeddings,
                threshold=10.0,
                mode="clamp",
            )

            assert 0 in info["masked_dimensions"]
            # Dimension 0 should be reduced but not zero
            assert masked[:, 0].std().item() < embeddings[:, 0].std().item()

    def test_mask_outlier_dimensions_scale_mode(self):
        """Test masking with 'scale' mode."""
        from llm_dit.vl.blending import mask_outlier_dimensions

        embeddings = torch.randn(100, 2560)
        embeddings[:, 0] *= 1000

        with patch('llm_dit.vl.blending._load_reference_stats') as mock_stats:
            ref_std = np.ones(2560)
            mock_stats.return_value = {"per_dim_std": ref_std}

            masked, info = mask_outlier_dimensions(
                embeddings,
                threshold=10.0,
                mode="scale",
            )

            assert 0 in info["masked_dimensions"]

    def test_mask_outlier_dimensions_invalid_mode(self):
        """Test invalid mode raises error."""
        from llm_dit.vl.blending import mask_outlier_dimensions

        embeddings = torch.randn(100, 2560)

        with patch('llm_dit.vl.blending._load_reference_stats') as mock_stats:
            mock_stats.return_value = {"per_dim_std": np.ones(2560)}

            with pytest.raises(ValueError):
                mask_outlier_dimensions(embeddings, mode="invalid")


class TestGetOutlierDimensions:
    """Test get_outlier_dimensions function."""

    def test_get_outlier_dimensions_sorted(self):
        """Test outlier dimensions returned sorted by ratio."""
        from llm_dit.vl.blending import get_outlier_dimensions

        embeddings = torch.randn(100, 2560)
        embeddings[:, 0] *= 100
        embeddings[:, 1] *= 50

        with patch('llm_dit.vl.blending._load_reference_stats') as mock_stats:
            ref_std = np.ones(2560)
            mock_stats.return_value = {"per_dim_std": ref_std}

            outliers = get_outlier_dimensions(embeddings, threshold=5.0)

            # Should be sorted by ratio descending
            if len(outliers) >= 2:
                assert outliers[0][1] >= outliers[1][1]


# ============================================================================
# Style Delta Tests
# ============================================================================

class TestComputeStyleDelta:
    """Test compute_style_delta function."""

    def test_compute_style_delta_basic(self):
        """Test basic style delta computation."""
        from llm_dit.vl.blending import compute_style_delta

        styled = torch.randn(100, 2560) + 5  # Offset
        neutral = torch.randn(100, 2560)

        delta = compute_style_delta(styled, neutral, normalize=False)

        expected = styled - neutral
        assert torch.allclose(delta, expected)

    def test_compute_style_delta_normalized(self):
        """Test normalized style delta."""
        from llm_dit.vl.blending import compute_style_delta

        styled = torch.randn(100, 2560) * 10
        neutral = torch.randn(100, 2560) * 10

        delta = compute_style_delta(styled, neutral, normalize=True)

        # Should have unit std (approximately)
        assert abs(delta.std().item() - 1.0) < 0.1

    def test_compute_style_delta_length_mismatch(self):
        """Test style delta with length mismatch."""
        from llm_dit.vl.blending import compute_style_delta

        styled = torch.randn(150, 2560)
        neutral = torch.randn(100, 2560)

        delta = compute_style_delta(styled, neutral)

        # Should use minimum length
        assert delta.shape[0] == 100


class TestBlendWithStyleDelta:
    """Test blend_with_style_delta function."""

    def test_blend_with_style_delta_basic(self):
        """Test adding style delta to text embeddings."""
        from llm_dit.vl.blending import blend_with_style_delta

        text_emb = torch.randn(100, 2560)
        style_delta = torch.randn(100, 2560)

        result = blend_with_style_delta(text_emb, style_delta, alpha=0.5, scale_to_text=False)

        expected = text_emb + 0.5 * style_delta
        assert torch.allclose(result, expected)

    def test_blend_with_style_delta_zero_alpha(self):
        """Test style delta with zero alpha."""
        from llm_dit.vl.blending import blend_with_style_delta

        text_emb = torch.randn(100, 2560)
        style_delta = torch.randn(100, 2560)

        result = blend_with_style_delta(text_emb, style_delta, alpha=0.0)

        assert torch.allclose(result, text_emb)


# ============================================================================
# AdaIN Blending Tests
# ============================================================================

class TestBlendAdaIN:
    """Test blend_adain function."""

    def test_blend_adain_preserves_structure(self):
        """Test AdaIN preserves relative structure."""
        from llm_dit.vl.blending import blend_adain

        text_emb = torch.randn(100, 2560)
        vl_emb = torch.randn(100, 2560) * 2 + 5  # Different stats

        result = blend_adain(text_emb, vl_emb, alpha=1.0)

        # Result should have VL statistics but text structure
        assert result.shape == text_emb.shape

    def test_blend_adain_alpha_zero(self):
        """Test AdaIN with alpha=0 returns original text."""
        from llm_dit.vl.blending import blend_adain

        text_emb = torch.randn(100, 2560)
        vl_emb = torch.randn(100, 2560)

        result = blend_adain(text_emb, vl_emb, alpha=0.0)

        assert torch.allclose(result, text_emb[:100])


class TestBlendAdaINPerDim:
    """Test blend_adain_per_dim function."""

    def test_blend_adain_per_dim_basic(self):
        """Test per-dimension AdaIN blending."""
        from llm_dit.vl.blending import blend_adain_per_dim

        text_emb = torch.randn(100, 2560)
        vl_emb = torch.randn(100, 2560)

        result = blend_adain_per_dim(text_emb, vl_emb, alpha=0.5)

        assert result.shape == text_emb.shape


# ============================================================================
# Helper Function Tests
# ============================================================================

class TestInterpolateSequence:
    """Test _interpolate_sequence helper."""

    def test_interpolate_same_length(self):
        """Test interpolation when lengths match."""
        from llm_dit.vl.blending import _interpolate_sequence

        emb = torch.randn(100, 2560)
        result = _interpolate_sequence(emb, 100)

        assert torch.allclose(result, emb)

    def test_interpolate_downsample(self):
        """Test interpolation downsampling."""
        from llm_dit.vl.blending import _interpolate_sequence

        emb = torch.randn(200, 2560)
        result = _interpolate_sequence(emb, 100)

        assert result.shape == (100, 2560)

    def test_interpolate_upsample(self):
        """Test interpolation upsampling."""
        from llm_dit.vl.blending import _interpolate_sequence

        emb = torch.randn(50, 2560)
        result = _interpolate_sequence(emb, 100)

        assert result.shape == (100, 2560)


class TestComputeBlendStats:
    """Test compute_blend_stats function."""

    def test_compute_blend_stats(self):
        """Test blend statistics computation."""
        from llm_dit.vl.blending import compute_blend_stats

        vl_emb = torch.randn(100, 2560) * 10
        text_emb = torch.randn(80, 2560) * 60

        stats = compute_blend_stats(vl_emb, text_emb)

        assert stats["vl_shape"] == [100, 2560]
        assert stats["text_shape"] == [80, 2560]
        assert "vl_mean" in stats
        assert "vl_std" in stats
        assert "text_mean" in stats
        assert "text_std" in stats
