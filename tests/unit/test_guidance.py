"""
Unit tests for llm_dit.guidance module.

Tests cover:
- FMTT (Flow Map Trajectory Tilting) guidance
- Skip Layer Guidance (SLG)
"""

from unittest.mock import MagicMock, patch
import pytest
import torch
import torch.nn as nn


# ============================================================================
# FMTT Guidance Tests
# ============================================================================

class TestFlowMapDirect:
    """Test flow_map_direct function."""

    def test_flow_map_direct_basic(self):
        """Test basic flow map computation."""
        from llm_dit.guidance.fmtt import flow_map_direct

        x_t = torch.randn(1, 4, 64, 64)
        velocity = torch.randn(1, 4, 64, 64)
        sigma = 0.5

        result = flow_map_direct(x_t, velocity, sigma)

        # x_clean = x_t + velocity * sigma
        expected = x_t + velocity * sigma
        assert torch.allclose(result, expected)

    def test_flow_map_direct_zero_sigma(self):
        """Test flow map with zero sigma (no noise)."""
        from llm_dit.guidance.fmtt import flow_map_direct

        x_t = torch.randn(1, 4, 64, 64)
        velocity = torch.randn(1, 4, 64, 64)
        sigma = 0.0

        result = flow_map_direct(x_t, velocity, sigma)

        # With sigma=0, result should equal x_t
        assert torch.allclose(result, x_t)

    def test_flow_map_direct_full_sigma(self):
        """Test flow map with sigma=1 (full noise)."""
        from llm_dit.guidance.fmtt import flow_map_direct

        x_t = torch.randn(1, 4, 64, 64)
        velocity = torch.randn(1, 4, 64, 64)
        sigma = 1.0

        result = flow_map_direct(x_t, velocity, sigma)

        expected = x_t + velocity
        assert torch.allclose(result, expected)


class TestFMTTGuidance:
    """Test FMTTGuidance class."""

    @pytest.fixture
    def mock_vae(self):
        """Create mock VAE."""
        vae = MagicMock()
        vae.config = MagicMock()
        vae.config.scaling_factor = 0.18215
        vae.config.shift_factor = 0.0
        vae.dtype = torch.float32

        def decode(x):
            # Return mock image tensor
            return MagicMock(sample=torch.randn(1, 3, 512, 512))

        vae.decode = decode
        return vae

    @pytest.fixture
    def mock_reward_fn(self):
        """Create mock reward function."""
        reward_fn = MagicMock()

        def compute_reward(image, prompt):
            # Return differentiable reward
            return image.sum() * 0.001  # Simple differentiable function

        reward_fn.compute_reward = compute_reward
        return reward_fn

    def test_fmtt_init_default(self, mock_vae, mock_reward_fn):
        """Test FMTTGuidance initialization with defaults."""
        from llm_dit.guidance.fmtt import FMTTGuidance

        fmtt = FMTTGuidance(
            vae=mock_vae,
            reward_fn=mock_reward_fn,
        )

        assert fmtt.guidance_scale == 1.0
        assert fmtt.guidance_start == 0.0
        assert fmtt.guidance_stop == 0.5
        assert fmtt.normalize_mode == "unit"
        assert fmtt.decode_scale == 0.5

    def test_fmtt_init_custom(self, mock_vae, mock_reward_fn):
        """Test FMTTGuidance with custom params."""
        from llm_dit.guidance.fmtt import FMTTGuidance

        fmtt = FMTTGuidance(
            vae=mock_vae,
            reward_fn=mock_reward_fn,
            guidance_scale=2.0,
            guidance_start=0.1,
            guidance_stop=0.6,
            normalize_mode="clip",
            clip_value=2.0,
        )

        assert fmtt.guidance_scale == 2.0
        assert fmtt.guidance_start == 0.1
        assert fmtt.guidance_stop == 0.6
        assert fmtt.normalize_mode == "clip"
        assert fmtt.clip_value == 2.0

    def test_fmtt_init_invalid_normalize_mode(self, mock_vae, mock_reward_fn):
        """Test FMTTGuidance raises on invalid normalize mode."""
        from llm_dit.guidance.fmtt import FMTTGuidance

        with pytest.raises(ValueError, match="normalize_mode"):
            FMTTGuidance(
                vae=mock_vae,
                reward_fn=mock_reward_fn,
                normalize_mode="invalid",
            )

    def test_fmtt_is_active(self, mock_vae, mock_reward_fn):
        """Test is_active step checking."""
        from llm_dit.guidance.fmtt import FMTTGuidance

        fmtt = FMTTGuidance(
            vae=mock_vae,
            reward_fn=mock_reward_fn,
            guidance_start=0.0,
            guidance_stop=0.5,
        )

        # Should be active for first half of steps
        assert fmtt.is_active(0, 10) is True
        assert fmtt.is_active(2, 10) is True
        assert fmtt.is_active(4, 10) is True
        assert fmtt.is_active(5, 10) is False  # 50% = stop point
        assert fmtt.is_active(8, 10) is False

    def test_fmtt_is_active_custom_range(self, mock_vae, mock_reward_fn):
        """Test is_active with custom range."""
        from llm_dit.guidance.fmtt import FMTTGuidance

        fmtt = FMTTGuidance(
            vae=mock_vae,
            reward_fn=mock_reward_fn,
            guidance_start=0.2,
            guidance_stop=0.8,
        )

        assert fmtt.is_active(0, 10) is False  # 0% < 20%
        assert fmtt.is_active(2, 10) is True   # 20% = start
        assert fmtt.is_active(5, 10) is True   # 50% in range
        assert fmtt.is_active(8, 10) is False  # 80% = stop

    def test_fmtt_guide_velocity(self, mock_vae, mock_reward_fn):
        """Test guide_velocity applies gradient."""
        from llm_dit.guidance.fmtt import FMTTGuidance

        fmtt = FMTTGuidance(
            vae=mock_vae,
            reward_fn=mock_reward_fn,
            guidance_scale=2.0,
        )

        velocity = torch.randn(1, 4, 64, 64)
        grad = torch.randn(1, 4, 64, 64)

        guided = fmtt.guide_velocity(velocity, grad)

        expected = velocity + 2.0 * grad
        assert torch.allclose(guided, expected)

    def test_fmtt_normalize_gradient_unit(self, mock_vae, mock_reward_fn):
        """Test gradient normalization to unit norm."""
        from llm_dit.guidance.fmtt import FMTTGuidance

        fmtt = FMTTGuidance(
            vae=mock_vae,
            reward_fn=mock_reward_fn,
            normalize_mode="unit",
        )

        grad = torch.randn(1, 4, 64, 64) * 10  # Large gradient

        normalized = fmtt._normalize_gradient(grad)

        # Should have unit norm (approximately)
        assert abs(normalized.norm().item() - 1.0) < 1e-5

    def test_fmtt_normalize_gradient_clip(self, mock_vae, mock_reward_fn):
        """Test gradient clipping."""
        from llm_dit.guidance.fmtt import FMTTGuidance

        fmtt = FMTTGuidance(
            vae=mock_vae,
            reward_fn=mock_reward_fn,
            normalize_mode="clip",
            clip_value=1.0,
        )

        grad = torch.randn(1, 4, 64, 64) * 10  # Large gradient

        normalized = fmtt._normalize_gradient(grad)

        # Norm should be <= clip_value
        assert normalized.norm().item() <= 1.0 + 1e-5

    def test_fmtt_normalize_gradient_none(self, mock_vae, mock_reward_fn):
        """Test no gradient normalization."""
        from llm_dit.guidance.fmtt import FMTTGuidance

        fmtt = FMTTGuidance(
            vae=mock_vae,
            reward_fn=mock_reward_fn,
            normalize_mode="none",
        )

        grad = torch.randn(1, 4, 64, 64) * 10

        normalized = fmtt._normalize_gradient(grad)

        # Should be unchanged
        assert torch.allclose(normalized, grad)


# ============================================================================
# Skip Layer Guidance Tests
# ============================================================================

class TestLayerSkipConfig:
    """Test LayerSkipConfig dataclass."""

    def test_layer_skip_config_basic(self):
        """Test basic LayerSkipConfig creation."""
        from llm_dit.guidance.skip_layer import LayerSkipConfig

        config = LayerSkipConfig(indices=[7, 8, 9])
        assert config.indices == [7, 8, 9]
        assert config.fqn == "auto"
        assert config.skip_attention is True
        assert config.skip_ff is True
        assert config.dropout == 1.0

    def test_layer_skip_config_custom(self):
        """Test LayerSkipConfig with custom values."""
        from llm_dit.guidance.skip_layer import LayerSkipConfig

        config = LayerSkipConfig(
            indices=[10, 11, 12],
            fqn="transformer_blocks",
            skip_attention=True,
            skip_ff=False,
            dropout=0.5,
        )
        assert config.indices == [10, 11, 12]
        assert config.fqn == "transformer_blocks"
        assert config.skip_ff is False
        assert config.dropout == 0.5

    def test_layer_skip_config_single_index(self):
        """Test LayerSkipConfig with single integer index."""
        from llm_dit.guidance.skip_layer import LayerSkipConfig

        config = LayerSkipConfig(indices=5)
        assert config.indices == [5]

    def test_layer_skip_config_invalid_dropout(self):
        """Test LayerSkipConfig raises on invalid dropout."""
        from llm_dit.guidance.skip_layer import LayerSkipConfig

        with pytest.raises(ValueError):
            LayerSkipConfig(indices=[1], dropout=1.5)

        with pytest.raises(ValueError):
            LayerSkipConfig(indices=[1], dropout=-0.1)

    def test_layer_skip_config_empty_indices(self):
        """Test LayerSkipConfig raises on empty indices."""
        from llm_dit.guidance.skip_layer import LayerSkipConfig

        with pytest.raises(ValueError):
            LayerSkipConfig(indices=[])

    def test_layer_skip_config_to_dict(self):
        """Test LayerSkipConfig.to_dict()."""
        from llm_dit.guidance.skip_layer import LayerSkipConfig

        config = LayerSkipConfig(indices=[7, 8], dropout=0.8)
        data = config.to_dict()

        assert data["indices"] == [7, 8]
        assert data["dropout"] == 0.8
        assert "fqn" in data

    def test_layer_skip_config_from_dict(self):
        """Test LayerSkipConfig.from_dict()."""
        from llm_dit.guidance.skip_layer import LayerSkipConfig

        data = {"indices": [1, 2, 3], "dropout": 0.5, "fqn": "blocks"}
        config = LayerSkipConfig.from_dict(data)

        assert config.indices == [1, 2, 3]
        assert config.dropout == 0.5
        assert config.fqn == "blocks"


class TestSkipLayerGuidance:
    """Test SkipLayerGuidance class."""

    def test_slg_init_from_list(self):
        """Test SLG initialization from list of indices."""
        from llm_dit.guidance.skip_layer import SkipLayerGuidance

        slg = SkipLayerGuidance(skip_layers=[7, 8, 9, 10])

        assert slg.guidance_scale == 2.8
        assert slg.guidance_start == 0.01
        assert slg.guidance_stop == 0.2
        assert len(slg.configs) == 1
        assert slg.configs[0].indices == [7, 8, 9, 10]

    def test_slg_init_from_config(self):
        """Test SLG initialization from LayerSkipConfig."""
        from llm_dit.guidance.skip_layer import SkipLayerGuidance, LayerSkipConfig

        config = LayerSkipConfig(indices=[5, 6, 7])
        slg = SkipLayerGuidance(skip_layers=config, guidance_scale=3.0)

        assert slg.guidance_scale == 3.0
        assert len(slg.configs) == 1
        assert slg.configs[0].indices == [5, 6, 7]

    def test_slg_init_custom_params(self):
        """Test SLG with custom guidance parameters."""
        from llm_dit.guidance.skip_layer import SkipLayerGuidance

        slg = SkipLayerGuidance(
            skip_layers=[7, 8, 9],
            guidance_scale=4.0,
            guidance_start=0.05,
            guidance_stop=0.3,
        )

        assert slg.guidance_scale == 4.0
        assert slg.guidance_start == 0.05
        assert slg.guidance_stop == 0.3

    def test_slg_is_active(self):
        """Test is_active step checking."""
        from llm_dit.guidance.skip_layer import SkipLayerGuidance

        slg = SkipLayerGuidance(
            skip_layers=[7, 8],
            guidance_start=0.0,
            guidance_stop=0.2,
        )

        # Active for first 20% of steps
        assert slg.is_active(0, 10) is True
        assert slg.is_active(1, 10) is True
        assert slg.is_active(2, 10) is False  # 20% = stop point
        assert slg.is_active(5, 10) is False

    def test_slg_skip_layer_indices_property(self):
        """Test skip_layer_indices property."""
        from llm_dit.guidance.skip_layer import SkipLayerGuidance

        slg = SkipLayerGuidance(skip_layers=[7, 8, 9, 10, 11, 12])

        indices = slg.skip_layer_indices
        assert indices == {7, 8, 9, 10, 11, 12}

    def test_slg_guide_no_cfg(self):
        """Test guide without CFG (Z-Image mode)."""
        from llm_dit.guidance.skip_layer import SkipLayerGuidance

        slg = SkipLayerGuidance(skip_layers=[7, 8], guidance_scale=2.0)

        pred_cond = torch.randn(1, 4, 64, 64)
        pred_skip = torch.randn(1, 4, 64, 64)

        guided = slg.guide(pred_cond, pred_skip)

        # pred = pred_cond + scale * (pred_cond - pred_skip)
        expected = pred_cond + 2.0 * (pred_cond - pred_skip)
        assert torch.allclose(guided, expected)

    def test_slg_guide_with_cfg(self):
        """Test guide with CFG (standard mode)."""
        from llm_dit.guidance.skip_layer import SkipLayerGuidance

        slg = SkipLayerGuidance(skip_layers=[7, 8], guidance_scale=2.0)

        pred_cond = torch.randn(1, 4, 64, 64)
        pred_skip = torch.randn(1, 4, 64, 64)
        pred_uncond = torch.randn(1, 4, 64, 64)

        guided = slg.guide(pred_cond, pred_skip, pred_uncond=pred_uncond, cfg_scale=7.5)

        # pred = pred_uncond + cfg * (pred_cond - pred_uncond) + slg * (pred_cond - pred_skip)
        cfg_shift = pred_cond - pred_uncond
        slg_shift = pred_cond - pred_skip
        expected = pred_uncond + 7.5 * cfg_shift + 2.0 * slg_shift
        assert torch.allclose(guided, expected)

    def test_slg_context_manager(self):
        """Test skip_layers_context manager applies and removes hooks."""
        from llm_dit.guidance.skip_layer import SkipLayerGuidance

        # Create model with blocks
        class SimpleTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([
                    nn.Linear(32, 32) for _ in range(10)
                ])

        model = SimpleTransformer()
        slg = SkipLayerGuidance(skip_layers=[2, 3, 4], fqn="blocks")

        # Before context: no hooks
        assert len(slg._active_hooks) == 0

        with slg.skip_layers_context(model):
            # Inside context: hooks applied
            assert len(slg._active_hooks) == 3

        # After context: hooks removed
        assert len(slg._active_hooks) == 0

    def test_slg_context_manager_invalid_layer_index(self):
        """Test skip_layers_context raises on invalid layer index."""
        from llm_dit.guidance.skip_layer import SkipLayerGuidance

        class SimpleTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([nn.Linear(32, 32) for _ in range(5)])

        model = SimpleTransformer()
        slg = SkipLayerGuidance(skip_layers=[10], fqn="blocks")  # Index out of range

        with pytest.raises(ValueError, match="out of range"):
            with slg.skip_layers_context(model):
                pass


class TestSkipHook:
    """Test _SkipHook internal class."""

    def test_skip_hook_full_skip(self):
        """Test hook with full skip (dropout=1.0)."""
        from llm_dit.guidance.skip_layer import _SkipHook

        hook = _SkipHook(layer_idx=5, dropout=1.0)

        hidden_states = torch.randn(1, 10, 32)
        output = torch.randn(1, 10, 32)

        # Hook should return input, not output
        result = hook(
            module=MagicMock(),
            args=(),
            kwargs={"hidden_states": hidden_states},
            output=output,
        )

        assert torch.allclose(result, hidden_states)

    def test_skip_hook_partial_skip(self):
        """Test hook with partial skip (dropout<1.0)."""
        from llm_dit.guidance.skip_layer import _SkipHook

        hook = _SkipHook(layer_idx=5, dropout=0.5)

        hidden_states = torch.randn(1, 10, 32)
        output = torch.randn(1, 10, 32)

        result = hook(
            module=MagicMock(),
            args=(),
            kwargs={"hidden_states": hidden_states},
            output=output,
        )

        # Result should be dropout-scaled output
        # Can't check exact values due to randomness, just check shape
        assert result.shape == output.shape

    def test_skip_hook_register_and_remove(self):
        """Test hook registration and removal."""
        from llm_dit.guidance.skip_layer import _SkipHook

        hook = _SkipHook(layer_idx=0)
        module = nn.Linear(32, 32)

        # Register
        hook.register(module)
        assert hook.handle is not None

        # Remove
        hook.remove()
        assert hook.handle is None


# ============================================================================
# apply_layer_skip and remove_layer_skip Tests
# ============================================================================

class TestLayerSkipFunctions:
    """Test apply_layer_skip and remove_layer_skip functions."""

    def test_apply_and_remove_layer_skip(self):
        """Test applying and removing layer skip hooks."""
        from llm_dit.guidance.skip_layer import (
            apply_layer_skip,
            remove_layer_skip,
            LayerSkipConfig,
        )

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([nn.Linear(32, 32) for _ in range(5)])

        model = SimpleModel()
        config = LayerSkipConfig(indices=[1, 2], fqn="blocks")

        hooks = apply_layer_skip(model, config)
        assert len(hooks) == 2

        remove_layer_skip(hooks)
        for hook in hooks:
            assert hook.handle is None
