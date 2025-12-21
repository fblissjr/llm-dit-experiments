"""
Unit tests for DyPE (Dynamic Position Extrapolation).

Tests DyPE configuration, core computation functions, Vision YaRN
frequency computation, and ZImageDyPERoPE wrapper without requiring
GPU or model weights.
"""

import math

import pytest
import torch

from llm_dit.utils.dype import (
    DyPEConfig,
    DyPEPosEmbed,
    ZImageDyPERoPE,
    axis_token_span,
    compute_dype_shift,
    compute_k_t,
    compute_mscale,
    patch_zimage_rope,
    set_zimage_timestep,
)
from llm_dit.utils.vision_yarn import (
    find_correction_factor,
    find_correction_range,
    find_newbase_ntk,
    get_1d_ntk_pos_embed,
    get_1d_vision_yarn_pos_embed,
    get_1d_yarn_pos_embed,
    linear_ramp_mask,
)

pytestmark = pytest.mark.unit


class TestDyPEConfig:
    """Test DyPEConfig dataclass."""

    def test_default_values(self):
        config = DyPEConfig()
        assert config.enabled is False
        assert config.method == "vision_yarn"
        assert config.dype_scale == 2.0
        assert config.dype_exponent == 2.0
        assert config.dype_start_sigma == 1.0
        assert config.base_shift == 0.5
        assert config.max_shift == 1.15
        assert config.base_resolution == 1024
        assert config.anisotropic is False

    def test_enabled_config(self):
        config = DyPEConfig(enabled=True)
        assert config.enabled is True

    def test_method_selection(self):
        for method in ["vision_yarn", "yarn", "ntk"]:
            config = DyPEConfig(method=method)
            assert config.method == method

    def test_parameter_customization(self):
        config = DyPEConfig(
            dype_scale=3.0,
            dype_exponent=1.5,
            base_resolution=2048,
        )
        assert config.dype_scale == 3.0
        assert config.dype_exponent == 1.5
        assert config.base_resolution == 2048

    def test_dype_start_sigma_clamping(self):
        # Test lower bound
        config = DyPEConfig(dype_start_sigma=-0.5)
        assert config.dype_start_sigma == 0.001

        # Test upper bound
        config = DyPEConfig(dype_start_sigma=1.5)
        assert config.dype_start_sigma == 1.0

        # Test valid value
        config = DyPEConfig(dype_start_sigma=0.5)
        assert config.dype_start_sigma == 0.5

    def test_to_dict(self):
        config = DyPEConfig(enabled=True, method="yarn")
        d = config.to_dict()

        assert d["enabled"] is True
        assert d["method"] == "yarn"
        assert d["dype_scale"] == 2.0
        assert "base_resolution" in d

    def test_from_dict(self):
        d = {
            "enabled": True,
            "method": "ntk",
            "dype_scale": 1.5,
            "base_resolution": 2048,
        }
        config = DyPEConfig.from_dict(d)

        assert config.enabled is True
        assert config.method == "ntk"
        assert config.dype_scale == 1.5
        assert config.base_resolution == 2048

    def test_from_dict_partial(self):
        # Test with missing fields (should use defaults)
        d = {"enabled": True}
        config = DyPEConfig.from_dict(d)

        assert config.enabled is True
        assert config.method == "vision_yarn"  # default


class TestComputeDypeShift:
    """Test compute_dype_shift function."""

    def test_at_base_resolution(self):
        config = DyPEConfig(base_shift=0.5, max_shift=1.15)
        base_seq_len = 64 * 64  # 1024px -> 64 patches
        max_seq_len = 256 * 256  # 4096px -> 256 patches

        shift = compute_dype_shift(
            image_seq_len=base_seq_len,
            base_seq_len=base_seq_len,
            max_seq_len=max_seq_len,
            config=config,
        )

        assert shift == config.base_shift

    def test_at_max_resolution(self):
        config = DyPEConfig(base_shift=0.5, max_shift=1.15)
        base_seq_len = 64 * 64
        max_seq_len = 256 * 256

        shift = compute_dype_shift(
            image_seq_len=max_seq_len,
            base_seq_len=base_seq_len,
            max_seq_len=max_seq_len,
            config=config,
        )

        assert abs(shift - config.max_shift) < 1e-6

    def test_interpolation(self):
        config = DyPEConfig(base_shift=0.5, max_shift=1.15)
        base_seq_len = 64 * 64
        max_seq_len = 256 * 256
        mid_seq_len = (base_seq_len + max_seq_len) // 2

        shift = compute_dype_shift(
            image_seq_len=mid_seq_len,
            base_seq_len=base_seq_len,
            max_seq_len=max_seq_len,
            config=config,
        )

        # Should be between base and max
        assert config.base_shift < shift < config.max_shift

    def test_max_seq_len_equals_base(self):
        config = DyPEConfig(base_shift=0.5, max_shift=1.15)
        base_seq_len = 64 * 64

        shift = compute_dype_shift(
            image_seq_len=base_seq_len,
            base_seq_len=base_seq_len,
            max_seq_len=base_seq_len,  # Same as base
            config=config,
        )

        assert shift == config.base_shift


class TestComputeKt:
    """Test compute_k_t function."""

    def test_at_sigma_1(self):
        config = DyPEConfig(dype_scale=2.0, dype_exponent=2.0)
        k_t = compute_k_t(sigma=1.0, config=config)

        # k_t = 2.0 * (1.0 ^ 2.0) = 2.0
        assert abs(k_t - 2.0) < 1e-6

    def test_at_sigma_0(self):
        config = DyPEConfig(dype_scale=2.0, dype_exponent=2.0)
        k_t = compute_k_t(sigma=0.0, config=config)

        # k_t = 2.0 * (0.0 ^ 2.0) = 0.0
        assert abs(k_t - 0.0) < 1e-6

    def test_decay_with_quadratic_exponent(self):
        config = DyPEConfig(dype_scale=2.0, dype_exponent=2.0)

        k_t_1 = compute_k_t(sigma=1.0, config=config)
        k_t_half = compute_k_t(sigma=0.5, config=config)
        k_t_quarter = compute_k_t(sigma=0.25, config=config)

        # Quadratic decay
        assert k_t_1 > k_t_half > k_t_quarter

        # Check quadratic relationship
        expected_half = 2.0 * (0.5**2.0)
        assert abs(k_t_half - expected_half) < 1e-6

    def test_different_scales(self):
        config_low = DyPEConfig(dype_scale=1.0, dype_exponent=2.0)
        config_high = DyPEConfig(dype_scale=3.0, dype_exponent=2.0)

        k_t_low = compute_k_t(sigma=1.0, config=config_low)
        k_t_high = compute_k_t(sigma=1.0, config=config_high)

        assert k_t_high > k_t_low

    def test_different_exponents(self):
        config_linear = DyPEConfig(dype_scale=2.0, dype_exponent=1.0)
        config_quadratic = DyPEConfig(dype_scale=2.0, dype_exponent=2.0)

        sigma = 0.5
        k_t_linear = compute_k_t(sigma=sigma, config=config_linear)
        k_t_quadratic = compute_k_t(sigma=sigma, config=config_quadratic)

        # At sigma < 1, higher exponent means more aggressive decay
        assert k_t_quadratic < k_t_linear


class TestComputeMscale:
    """Test compute_mscale function."""

    def test_no_scaling(self):
        config = DyPEConfig()
        mscale = compute_mscale(scale_global=1.0, sigma=1.0, config=config)

        # No scaling needed
        assert mscale == 1.0

    def test_scaling_at_sigma_1(self):
        config = DyPEConfig(dype_exponent=2.0)
        mscale = compute_mscale(scale_global=2.0, sigma=1.0, config=config)

        # At sigma >= dype_start_sigma, t_norm = 1.0
        # mscale = 1.0 + (mscale_start - 1.0) * 1.0 = mscale_start
        expected_start = 0.1 * math.log(2.0) + 1.0
        assert abs(mscale - expected_start) < 1e-6

    def test_scaling_at_sigma_0(self):
        config = DyPEConfig(dype_exponent=2.0)
        mscale = compute_mscale(scale_global=2.0, sigma=0.0, config=config)

        # At sigma = 0, t_norm = 0.0
        # mscale = 1.0 + (mscale_start - 1.0) * 0.0 = 1.0
        assert abs(mscale - 1.0) < 1e-6

    def test_interpolation(self):
        config = DyPEConfig(dype_exponent=2.0, dype_start_sigma=1.0)
        mscale_low = compute_mscale(scale_global=2.0, sigma=0.0, config=config)
        mscale_mid = compute_mscale(scale_global=2.0, sigma=0.5, config=config)
        mscale_high = compute_mscale(scale_global=2.0, sigma=1.0, config=config)

        # mscale should transition from 1.0 (end) to mscale_start
        assert mscale_low < mscale_mid < mscale_high

    def test_higher_scale_factor(self):
        config = DyPEConfig()
        mscale_2x = compute_mscale(scale_global=2.0, sigma=1.0, config=config)
        mscale_4x = compute_mscale(scale_global=4.0, sigma=1.0, config=config)

        # Higher scale should produce higher mscale_start
        assert mscale_4x > mscale_2x


class TestAxisTokenSpan:
    """Test axis_token_span function."""

    def test_single_token(self):
        pos = torch.tensor([0.0])
        span = axis_token_span(pos)
        assert span == 1.0

    def test_sequential_integers(self):
        pos = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
        span = axis_token_span(pos)
        # Span from 0 to 4 = 4 steps + 1 = 5 positions
        assert span == 5.0

    def test_fractional_positions(self):
        # Z-Image uses fractional positions (PI mode)
        pos = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0])
        span = axis_token_span(pos)
        # 4 steps of 0.5 = 2.0 span, (2.0 / 0.5) + 1 = 5
        assert span == 5.0

    def test_2d_positions(self):
        # Test with 2D position grid
        pos = torch.tensor([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
        span = axis_token_span(pos)
        # Should flatten and compute span
        assert span == 3.0

    def test_duplicate_positions(self):
        pos = torch.tensor([0.0, 0.0, 0.0])
        span = axis_token_span(pos)
        # All same position
        assert span == 1.0


class TestVisionYaRNHelpers:
    """Test Vision YaRN helper functions."""

    def test_find_correction_factor(self):
        factor = find_correction_factor(
            num_rotations=1.0,
            dim=128,
            base=10000,
            max_position_embeddings=2048,
        )
        assert isinstance(factor, float)
        assert factor > 0

    def test_find_correction_range(self):
        low, high = find_correction_range(
            low_ratio=1.25,
            high_ratio=0.75,
            dim=128,
            base=256,
            ori_max_pe_len=64,
        )
        assert isinstance(low, int)
        assert isinstance(high, int)
        assert 0 <= low <= high
        assert high < 128

    def test_linear_ramp_mask(self):
        mask = linear_ramp_mask(min_val=0.0, max_val=10.0, dim=11)
        assert mask.shape == (11,)
        assert mask[0] == 0.0
        assert mask[10] == 1.0
        # Check monotonic increase
        assert torch.all(mask[1:] >= mask[:-1])

    def test_find_newbase_ntk(self):
        base = 256
        scale = 2.0
        dim = 128

        new_base = find_newbase_ntk(dim, base, scale)

        # NTK formula: base * (scale ^ (dim / (dim - 2)))
        expected = base * (scale ** (dim / (dim - 2)))
        assert abs(new_base - expected) < 1e-6


class TestGet1DNTKPosEmbed:
    """Test get_1d_ntk_pos_embed function."""

    def test_basic_computation(self):
        dim = 32
        pos = torch.arange(0, 10).float()
        theta = 256
        ntk_factor = 1.0

        cos, sin = get_1d_ntk_pos_embed(
            dim=dim,
            pos=pos,
            theta=theta,
            ntk_factor=ntk_factor,
        )

        # Check shapes
        assert cos.shape == (10, dim)
        assert sin.shape == (10, dim)

        # Check value ranges
        assert torch.all(cos >= -1.0) and torch.all(cos <= 1.0)
        assert torch.all(sin >= -1.0) and torch.all(sin <= 1.0)

    def test_ntk_scaling(self):
        dim = 32
        pos = torch.arange(0, 10).float()
        theta = 256

        cos_1x, sin_1x = get_1d_ntk_pos_embed(
            dim=dim, pos=pos, theta=theta, ntk_factor=1.0
        )
        cos_2x, sin_2x = get_1d_ntk_pos_embed(
            dim=dim, pos=pos, theta=theta, ntk_factor=2.0
        )

        # Different scaling should produce different embeddings
        assert not torch.allclose(cos_1x, cos_2x)
        assert not torch.allclose(sin_1x, sin_2x)


class TestGet1DVisionYaRNPosEmbed:
    """Test get_1d_vision_yarn_pos_embed function."""

    def test_basic_computation(self):
        dim = 48
        pos = torch.arange(0, 64).float()
        theta = 256
        linear_scale = 2.0
        ntk_scale = 2.0
        ori_max_pe_len = 64

        cos, sin = get_1d_vision_yarn_pos_embed(
            dim=dim,
            pos=pos,
            theta=theta,
            linear_scale=linear_scale,
            ntk_scale=ntk_scale,
            ori_max_pe_len=ori_max_pe_len,
            dype=False,
        )

        # Check shapes
        assert cos.shape == (64, dim)
        assert sin.shape == (64, dim)

        # Check value ranges (mscale can push values slightly above 1.0)
        # This is expected behavior - mscale is amplitude compensation
        assert torch.all(torch.isfinite(cos))
        assert torch.all(torch.isfinite(sin))

    def test_dype_modulation(self):
        dim = 48
        pos = torch.arange(0, 64).float()
        theta = 256
        linear_scale = 2.0
        ntk_scale = 2.0
        ori_max_pe_len = 64

        # Without DyPE
        cos_no_dype, sin_no_dype = get_1d_vision_yarn_pos_embed(
            dim=dim,
            pos=pos,
            theta=theta,
            linear_scale=linear_scale,
            ntk_scale=ntk_scale,
            ori_max_pe_len=ori_max_pe_len,
            dype=False,
        )

        # With DyPE at sigma=1.0
        cos_dype, sin_dype = get_1d_vision_yarn_pos_embed(
            dim=dim,
            pos=pos,
            theta=theta,
            linear_scale=linear_scale,
            ntk_scale=ntk_scale,
            ori_max_pe_len=ori_max_pe_len,
            dype=True,
            current_timestep=1.0,
            dype_scale=2.0,
            dype_exponent=2.0,
        )

        # DyPE should produce different embeddings
        assert not torch.allclose(cos_no_dype, cos_dype)
        assert not torch.allclose(sin_no_dype, sin_dype)

    def test_timestep_progression(self):
        """Test that DyPE effect changes with timestep."""
        dim = 48
        pos = torch.arange(0, 64).float()
        theta = 256
        linear_scale = 2.0
        ntk_scale = 2.0
        ori_max_pe_len = 64

        cos_early, _ = get_1d_vision_yarn_pos_embed(
            dim=dim,
            pos=pos,
            theta=theta,
            linear_scale=linear_scale,
            ntk_scale=ntk_scale,
            ori_max_pe_len=ori_max_pe_len,
            dype=True,
            current_timestep=1.0,  # Early (noisy)
        )

        cos_late, _ = get_1d_vision_yarn_pos_embed(
            dim=dim,
            pos=pos,
            theta=theta,
            linear_scale=linear_scale,
            ntk_scale=ntk_scale,
            ori_max_pe_len=ori_max_pe_len,
            dype=True,
            current_timestep=0.0,  # Late (clean)
        )

        # Different timesteps should produce different embeddings
        assert not torch.allclose(cos_early, cos_late)


class TestGet1DYaRNPosEmbed:
    """Test get_1d_yarn_pos_embed function."""

    def test_basic_computation(self):
        dim = 48
        pos = torch.arange(0, 128).float()
        theta = 256
        max_pe_len = torch.tensor(128.0)
        ori_max_pe_len = 64

        cos, sin = get_1d_yarn_pos_embed(
            dim=dim,
            pos=pos,
            theta=theta,
            max_pe_len=max_pe_len,
            ori_max_pe_len=ori_max_pe_len,
            dype=False,
        )

        # Check shapes
        assert cos.shape == (128, dim)
        assert sin.shape == (128, dim)

        # Check value ranges (mscale can push values slightly above 1.0)
        # This is expected behavior - mscale is amplitude compensation
        assert torch.all(torch.isfinite(cos))
        assert torch.all(torch.isfinite(sin))


class TestZImageDyPERoPE:
    """Test ZImageDyPERoPE wrapper."""

    def test_initialization(self):
        # Create mock original embedder
        class MockRoPEEmbedder:
            theta = 256
            axes_dim = [32, 48, 48]

            def __call__(self, ids):
                # Return dummy output
                return torch.zeros(ids.shape[0], 1, ids.shape[1], 128, 2, 2)

        original = MockRoPEEmbedder()
        config = DyPEConfig(enabled=True)

        wrapper = ZImageDyPERoPE(
            original_embedder=original,
            config=config,
            scale_hint=1.0,
        )

        assert wrapper.theta == 256
        assert wrapper.axes_dim == [32, 48, 48]
        assert wrapper.config.enabled is True

    def test_set_timestep(self):
        class MockRoPEEmbedder:
            theta = 256
            axes_dim = [32, 48, 48]

            def __call__(self, ids):
                return torch.zeros(ids.shape[0], 1, ids.shape[1], 128, 2, 2)

        original = MockRoPEEmbedder()
        config = DyPEConfig(enabled=True)
        wrapper = ZImageDyPERoPE(original, config)

        wrapper.set_timestep(0.5)
        assert wrapper.current_sigma == 0.5

    def test_set_scale_hint(self):
        class MockRoPEEmbedder:
            theta = 256
            axes_dim = [32, 48, 48]

            def __call__(self, ids):
                return torch.zeros(ids.shape[0], 1, ids.shape[1], 128, 2, 2)

        original = MockRoPEEmbedder()
        config = DyPEConfig(enabled=True)
        wrapper = ZImageDyPERoPE(original, config)

        wrapper.set_scale_hint(2.0)
        assert wrapper.scale_hint == 2.0

        # Test clamping to >= 1.0
        wrapper.set_scale_hint(0.5)
        assert wrapper.scale_hint == 1.0

    def test_disabled_dype_delegates_to_original(self):
        class MockRoPEEmbedder:
            theta = 256
            axes_dim = [32, 48, 48]
            call_count = 0

            def __call__(self, ids):
                self.call_count += 1
                return torch.zeros(ids.shape[0], 1, ids.shape[1], 128, 2, 2)

        original = MockRoPEEmbedder()
        config = DyPEConfig(enabled=False)
        wrapper = ZImageDyPERoPE(original, config)

        ids = torch.zeros(1, 100, 3)
        wrapper(ids)

        # Should delegate to original
        assert original.call_count == 1


class TestDyPEPosEmbed:
    """Test DyPEPosEmbed base class."""

    def test_initialization(self):
        config = DyPEConfig(enabled=True, method="vision_yarn")
        embedder = DyPEPosEmbed(
            theta=256,
            axes_dim=[32, 48, 48],
            config=config,
        )

        assert embedder.theta == 256
        assert embedder.axes_dim == [32, 48, 48]
        assert embedder.config.enabled is True

    def test_set_timestep(self):
        embedder = DyPEPosEmbed(
            theta=256,
            axes_dim=[32, 48, 48],
            config=DyPEConfig(enabled=True),
        )

        embedder.set_timestep(0.7)
        assert embedder.current_sigma == 0.7

    def test_base_patch_grid_auto(self):
        config = DyPEConfig(base_resolution=1024)
        embedder = DyPEPosEmbed(
            theta=256,
            axes_dim=[32, 48, 48],
            config=config,
        )

        # 1024px -> 128 latent -> 64 patches (patch_size=2)
        expected_patches = (1024 // 8) // 2
        assert embedder.base_patch_grid == (expected_patches, expected_patches)

    def test_base_patch_grid_custom(self):
        embedder = DyPEPosEmbed(
            theta=256,
            axes_dim=[32, 48, 48],
            config=DyPEConfig(),
            base_patch_grid=(64, 64),
        )

        assert embedder.base_patch_grid == (64, 64)


class TestPatchZImageRoPE:
    """Test patch_zimage_rope function."""

    def test_patch_creates_wrapper(self):
        # Create mock transformer
        class MockTransformer:
            class RoPEEmbedder:
                theta = 256
                axes_dim = [32, 48, 48]

                def __call__(self, ids):
                    return torch.zeros(ids.shape[0], 1, ids.shape[1], 128, 2, 2)

            rope_embedder = RoPEEmbedder()

        transformer = MockTransformer()
        config = DyPEConfig(enabled=True)

        patched = patch_zimage_rope(transformer, config, width=2048, height=2048)

        # Check that rope_embedder was replaced
        assert isinstance(patched.rope_embedder, ZImageDyPERoPE)
        assert patched.rope_embedder.config.enabled is True

    def test_patch_computes_scale_hint(self):
        class MockTransformer:
            class RoPEEmbedder:
                theta = 256
                axes_dim = [32, 48, 48]

                def __call__(self, ids):
                    return torch.zeros(ids.shape[0], 1, ids.shape[1], 128, 2, 2)

            rope_embedder = RoPEEmbedder()

        transformer = MockTransformer()
        config = DyPEConfig(enabled=True, base_resolution=1024)

        patched = patch_zimage_rope(transformer, config, width=2048, height=2048)

        # 2048px -> 128 patches, 1024px -> 64 patches
        # scale = 128 / 64 = 2.0
        assert patched.rope_embedder.scale_hint == 2.0

    def test_patch_without_rope_embedder_raises(self):
        class MockTransformer:
            pass  # No rope_embedder attribute

        transformer = MockTransformer()
        config = DyPEConfig(enabled=True)

        with pytest.raises(ValueError, match="rope_embedder"):
            patch_zimage_rope(transformer, config, width=1024, height=1024)


class TestSetZImageTimestep:
    """Test set_zimage_timestep function."""

    def test_sets_timestep_on_dype_embedder(self):
        class MockTransformer:
            class RoPEEmbedder:
                theta = 256
                axes_dim = [32, 48, 48]

                def __call__(self, ids):
                    return torch.zeros(ids.shape[0], 1, ids.shape[1], 128, 2, 2)

            rope_embedder = RoPEEmbedder()

        transformer = MockTransformer()
        config = DyPEConfig(enabled=True)

        # Patch transformer
        patched = patch_zimage_rope(transformer, config, width=1024, height=1024)

        # Set timestep
        set_zimage_timestep(patched, sigma=0.3)

        # Check that timestep was set
        assert patched.rope_embedder.current_sigma == 0.3

    def test_does_nothing_on_non_dype_embedder(self):
        class MockTransformer:
            class RoPEEmbedder:
                pass

            rope_embedder = RoPEEmbedder()

        transformer = MockTransformer()

        # Should not raise error
        set_zimage_timestep(transformer, sigma=0.5)
