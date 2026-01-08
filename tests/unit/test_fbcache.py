"""Unit tests for FBCache (Forward Block Cache)."""

import pytest
import torch

from llm_dit.models.fbcache import (
    FBCacheConfig,
    FBCacheContext,
    FBCacheLayersWrapper,
    FBCacheState,
    relative_l1_distance,
)


class TestRelativeL1Distance:
    """Tests for the relative_l1_distance function."""

    def test_identical_tensors_returns_zero(self):
        """Identical tensors should have zero relative distance."""
        t = torch.randn(2, 128, 256)
        dist = relative_l1_distance(t, t)
        assert dist == pytest.approx(0.0, abs=1e-6)

    def test_scaled_tensor(self):
        """A tensor scaled by 1.05 should have ~5% relative distance."""
        t = torch.randn(2, 128, 256)
        scaled = t * 1.05
        dist = relative_l1_distance(scaled, t)
        # Relative L1 of (1.05t - t) / |t| = 0.05 * |t| / |t| = 0.05
        assert dist == pytest.approx(0.05, rel=0.01)

    def test_order_matters(self):
        """Distance is not symmetric due to normalization by previous."""
        t1 = torch.ones(10)
        t2 = torch.ones(10) * 2

        # (2 - 1) / |1| = 1.0
        dist_1_to_2 = relative_l1_distance(t2, t1)
        assert dist_1_to_2 == pytest.approx(1.0, abs=1e-6)

        # (1 - 2) / |2| = 0.5
        dist_2_to_1 = relative_l1_distance(t1, t2)
        assert dist_2_to_1 == pytest.approx(0.5, abs=1e-6)

    def test_handles_zero_tensor(self):
        """Should handle zero tensor gracefully (eps prevents division by zero)."""
        zero = torch.zeros(10)
        nonzero = torch.ones(10)
        # Should not raise, eps prevents inf
        dist = relative_l1_distance(nonzero, zero)
        assert dist > 0  # Large positive value
        assert not torch.isinf(torch.tensor(dist))


class TestFBCacheConfig:
    """Tests for FBCacheConfig dataclass."""

    def test_defaults(self):
        """Test default configuration values."""
        config = FBCacheConfig()
        assert config.enabled is False
        assert config.early_threshold == 0.01
        assert config.middle_threshold == 0.05
        assert config.late_threshold == 0.01
        assert config.log_residuals is False

    def test_custom_thresholds(self):
        """Test custom threshold configuration."""
        config = FBCacheConfig(
            enabled=True,
            early_threshold=0.02,
            middle_threshold=0.10,
            late_threshold=0.005,
        )
        assert config.enabled is True
        assert config.early_threshold == 0.02
        assert config.middle_threshold == 0.10
        assert config.late_threshold == 0.005


class TestFBCacheState:
    """Tests for FBCacheState class."""

    def test_initial_state(self):
        """Test initial state after creation."""
        config = FBCacheConfig(enabled=True)
        state = FBCacheState(config, num_inference_steps=8)

        assert state.step_count == 0
        assert state.prev_first_residual is None
        assert state.cached_remaining_residual is None
        assert state.skips_count == 0
        assert state.computes_count == 0

    def test_reset_clears_state(self):
        """Test that reset clears all state."""
        config = FBCacheConfig(enabled=True)
        state = FBCacheState(config, num_inference_steps=8)

        # Simulate some state
        state.step_count = 5
        state.prev_first_residual = torch.randn(2, 100, 256)
        state.cached_remaining_residual = torch.randn(2, 100, 256)
        state.skips_count = 3
        state.computes_count = 2

        state.reset()

        assert state.step_count == 0
        assert state.prev_first_residual is None
        assert state.cached_remaining_residual is None
        assert state.skips_count == 0
        assert state.computes_count == 0

    def test_reset_updates_num_steps(self):
        """Test that reset can update num_inference_steps."""
        config = FBCacheConfig(enabled=True)
        state = FBCacheState(config, num_inference_steps=8)

        state.reset(num_inference_steps=12)
        assert state.num_inference_steps == 12

    def test_get_threshold_for_sigma_early(self):
        """Test threshold selection for high sigma (early phase)."""
        config = FBCacheConfig(
            enabled=True,
            early_threshold=0.01,
            middle_threshold=0.05,
            late_threshold=0.01,
            early_sigma_min=0.7,
        )
        state = FBCacheState(config, num_inference_steps=8)

        # Sigma > 0.7 -> early threshold
        assert state.get_threshold_for_sigma(0.9) == 0.01
        assert state.get_threshold_for_sigma(0.8) == 0.01
        assert state.get_threshold_for_sigma(0.71) == 0.01

    def test_get_threshold_for_sigma_middle(self):
        """Test threshold selection for middle sigma."""
        config = FBCacheConfig(
            enabled=True,
            early_threshold=0.01,
            middle_threshold=0.05,
            late_threshold=0.01,
            early_sigma_min=0.7,
            late_sigma_max=0.3,
        )
        state = FBCacheState(config, num_inference_steps=8)

        # 0.3 < sigma < 0.7 -> middle threshold
        assert state.get_threshold_for_sigma(0.5) == 0.05
        assert state.get_threshold_for_sigma(0.4) == 0.05
        assert state.get_threshold_for_sigma(0.31) == 0.05

    def test_get_threshold_for_sigma_late(self):
        """Test threshold selection for low sigma (late phase)."""
        config = FBCacheConfig(
            enabled=True,
            early_threshold=0.01,
            middle_threshold=0.05,
            late_threshold=0.01,
            late_sigma_max=0.3,
        )
        state = FBCacheState(config, num_inference_steps=8)

        # Sigma < 0.3 -> late threshold
        assert state.get_threshold_for_sigma(0.2) == 0.01
        assert state.get_threshold_for_sigma(0.1) == 0.01
        assert state.get_threshold_for_sigma(0.0) == 0.01

    def test_should_skip_disabled(self):
        """Test that disabled config never skips."""
        config = FBCacheConfig(enabled=False)
        state = FBCacheState(config, num_inference_steps=8)

        # Even with cache populated, should not skip
        state.prev_first_residual = torch.randn(2, 100, 256)
        state.cached_remaining_residual = torch.randn(2, 100, 256)
        state.step_count = 3  # Not first or last

        assert state.should_skip(torch.randn(2, 100, 256), sigma=0.5) is False

    def test_should_skip_first_step(self):
        """Test that first step is never skipped."""
        config = FBCacheConfig(enabled=True)
        state = FBCacheState(config, num_inference_steps=8)
        state.step_count = 0

        # Should not skip even with identical residuals
        residual = torch.randn(2, 100, 256)
        state.prev_first_residual = residual.clone()
        state.cached_remaining_residual = torch.randn(2, 100, 256)

        assert state.should_skip(residual, sigma=0.5) is False

    def test_should_skip_last_step(self):
        """Test that last step is never skipped."""
        config = FBCacheConfig(enabled=True)
        state = FBCacheState(config, num_inference_steps=8)
        state.step_count = 7  # Last step (0-indexed)

        # Should not skip even with identical residuals
        residual = torch.randn(2, 100, 256)
        state.prev_first_residual = residual.clone()
        state.cached_remaining_residual = torch.randn(2, 100, 256)

        assert state.should_skip(residual, sigma=0.1) is False

    def test_should_skip_no_prior_residual(self):
        """Test that no skip without prior residual."""
        config = FBCacheConfig(enabled=True)
        state = FBCacheState(config, num_inference_steps=8)
        state.step_count = 3  # Middle step

        # No prior residual
        assert state.should_skip(torch.randn(2, 100, 256), sigma=0.5) is False

    def test_should_skip_no_cached_output(self):
        """Test that no skip without cached output."""
        config = FBCacheConfig(enabled=True)
        state = FBCacheState(config, num_inference_steps=8)
        state.step_count = 3  # Middle step
        state.prev_first_residual = torch.randn(2, 100, 256)
        # No cached_remaining_residual

        assert state.should_skip(torch.randn(2, 100, 256), sigma=0.5) is False

    def test_should_skip_similar_residual(self):
        """Test that similar residuals trigger skip."""
        config = FBCacheConfig(
            enabled=True,
            middle_threshold=0.05,  # 5%
        )
        state = FBCacheState(config, num_inference_steps=8)
        state.step_count = 3  # Middle step

        # Setup prior state
        prior_residual = torch.randn(2, 100, 256)
        state.prev_first_residual = prior_residual
        state.cached_remaining_residual = torch.randn(2, 100, 256)

        # Current residual is only 1% different (below 5% threshold)
        current_residual = prior_residual * 1.01

        assert state.should_skip(current_residual, sigma=0.5) is True

    def test_should_skip_different_residual(self):
        """Test that different residuals don't skip."""
        config = FBCacheConfig(
            enabled=True,
            middle_threshold=0.05,  # 5%
        )
        state = FBCacheState(config, num_inference_steps=8)
        state.step_count = 3  # Middle step

        # Setup prior state
        prior_residual = torch.randn(2, 100, 256)
        state.prev_first_residual = prior_residual
        state.cached_remaining_residual = torch.randn(2, 100, 256)

        # Current residual is 10% different (above 5% threshold)
        current_residual = prior_residual * 1.10

        assert state.should_skip(current_residual, sigma=0.5) is False

    def test_update_cache(self):
        """Test cache update stores clones."""
        config = FBCacheConfig(enabled=True)
        state = FBCacheState(config, num_inference_steps=8)

        first_residual = torch.randn(2, 100, 256)
        remaining_residual = torch.randn(2, 100, 256)

        state.update_cache(first_residual, remaining_residual)

        # Should be equal but not same object
        assert torch.allclose(state.prev_first_residual, first_residual)
        assert torch.allclose(state.cached_remaining_residual, remaining_residual)
        assert state.prev_first_residual is not first_residual
        assert state.cached_remaining_residual is not remaining_residual
        assert state.computes_count == 1

    def test_mark_skipped(self):
        """Test skip counting."""
        config = FBCacheConfig(enabled=True)
        state = FBCacheState(config, num_inference_steps=8)

        state.mark_skipped()
        state.mark_skipped()
        state.mark_skipped()

        assert state.skips_count == 3

    def test_advance_step(self):
        """Test step advancement."""
        config = FBCacheConfig(enabled=True)
        state = FBCacheState(config, num_inference_steps=8)

        state.advance_step()
        assert state.step_count == 1

        state.advance_step()
        assert state.step_count == 2

    def test_get_stats(self):
        """Test statistics collection."""
        config = FBCacheConfig(enabled=True)
        state = FBCacheState(config, num_inference_steps=8)

        state.computes_count = 5
        state.skips_count = 3

        stats = state.get_stats()
        assert stats["computes"] == 5
        assert stats["skips"] == 3
        assert stats["total_steps"] == 8
        assert stats["skip_ratio"] == pytest.approx(0.375, abs=0.001)
        assert stats["estimated_speedup"] > 1.0

    def test_logging_enabled(self):
        """Test that logging populates residual_log."""
        config = FBCacheConfig(enabled=True, log_residuals=True)
        state = FBCacheState(config, num_inference_steps=8)
        state.step_count = 3

        # Setup state for skip check
        prior_residual = torch.randn(2, 100, 256)
        state.prev_first_residual = prior_residual
        state.cached_remaining_residual = torch.randn(2, 100, 256)

        # Trigger skip check (which logs)
        state.should_skip(prior_residual * 1.01, sigma=0.5)

        assert len(state.residual_log) == 1
        entry = state.residual_log[0]
        assert entry["step"] == 3
        assert entry["sigma"] == 0.5
        assert "rel_diff" in entry
        assert "threshold" in entry
        assert "skipped" in entry


class TestFBCacheIntegration:
    """Integration tests simulating actual diffusion loop behavior."""

    def test_typical_8_step_generation(self):
        """Simulate a typical 8-step Z-Image turbo generation."""
        config = FBCacheConfig(
            enabled=True,
            early_threshold=0.01,
            middle_threshold=0.05,
            late_threshold=0.01,
        )
        state = FBCacheState(config, num_inference_steps=8)

        # Simulate sigmas for 8 steps (high to low)
        sigmas = [0.95, 0.80, 0.65, 0.50, 0.35, 0.20, 0.10, 0.02]

        # Simulate residuals that become more stable in middle steps
        base_residual = torch.randn(1, 256, 1024)

        for step, sigma in enumerate(sigmas):
            # Residuals vary more early, less in middle, more late
            if sigma > 0.7:
                noise_scale = 0.15  # 15% variation
            elif sigma > 0.3:
                noise_scale = 0.02  # 2% variation - should skip
            else:
                noise_scale = 0.10  # 10% variation

            current_residual = base_residual * (1.0 + noise_scale * (torch.rand_like(base_residual) - 0.5))

            should_skip = state.should_skip(current_residual, sigma)

            if should_skip:
                state.mark_skipped()
            else:
                # Full computation - update cache
                remaining_residual = torch.randn_like(base_residual)
                state.update_cache(current_residual.clone(), remaining_residual)

            state.advance_step()
            base_residual = current_residual  # Update for next iteration

        stats = state.get_stats()
        # Should have some skips in middle steps
        assert stats["total_steps"] == 8
        # At minimum, first and last are computed (no skip)
        assert stats["computes"] >= 2


class MockTransformerLayer(torch.nn.Module):
    """Mock transformer layer for testing."""

    def __init__(self, add_value: float = 0.1):
        super().__init__()
        self.add_value = add_value

    def forward(self, x, *args, **kwargs):
        """Simple forward that adds a fixed value."""
        return x + self.add_value


class MockTransformer(torch.nn.Module):
    """Mock transformer with layers attribute for FBCache testing."""

    def __init__(self, num_layers: int = 5):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            MockTransformerLayer(add_value=0.1 * (i + 1))
            for i in range(num_layers)
        ])

    def forward(self, x):
        """Forward through all layers."""
        for layer in self.layers:
            x = layer(x)
        return x


class TestFBCacheLayersWrapper:
    """Tests for FBCacheLayersWrapper."""

    def test_wrapper_iteration(self):
        """Test that wrapper yields correct number of layers."""
        config = FBCacheConfig(enabled=True)
        state = FBCacheState(config, num_inference_steps=8)

        original_layers = torch.nn.ModuleList([
            MockTransformerLayer() for _ in range(5)
        ])
        wrapper = FBCacheLayersWrapper(original_layers, state)

        assert len(wrapper) == 5
        layers_list = list(wrapper)
        assert len(layers_list) == 5

    def test_wrapper_indexing(self):
        """Test that wrapper supports indexing."""
        config = FBCacheConfig(enabled=True)
        state = FBCacheState(config, num_inference_steps=8)

        layer0 = MockTransformerLayer(add_value=0.1)
        layer1 = MockTransformerLayer(add_value=0.2)
        original_layers = torch.nn.ModuleList([layer0, layer1])
        wrapper = FBCacheLayersWrapper(original_layers, state)

        # Indexing should return original layers (for controlnet checks)
        assert wrapper[0] is layer0
        assert wrapper[1] is layer1


class TestFBCacheContext:
    """Tests for FBCacheContext."""

    def test_context_manager_basic(self):
        """Test context manager enters and exits cleanly."""
        config = FBCacheConfig(enabled=True)
        state = FBCacheState(config, num_inference_steps=8)
        transformer = MockTransformer(num_layers=3)

        original_layers = transformer.layers

        with FBCacheContext(transformer, state) as ctx:
            # Layers should be wrapped
            assert isinstance(transformer.layers, FBCacheLayersWrapper)
            assert transformer.layers is not original_layers

        # After exit, layers should be restored
        assert transformer.layers is original_layers

    def test_context_disabled(self):
        """Test that disabled config doesn't wrap layers."""
        config = FBCacheConfig(enabled=False)
        state = FBCacheState(config, num_inference_steps=8)
        transformer = MockTransformer(num_layers=3)

        original_layers = transformer.layers

        with FBCacheContext(transformer, state) as ctx:
            # Layers should NOT be wrapped when disabled
            assert transformer.layers is original_layers

    def test_context_requires_layers_attribute(self):
        """Test that context raises error without layers attribute."""
        config = FBCacheConfig(enabled=True)
        state = FBCacheState(config, num_inference_steps=8)

        # Module without layers attribute
        class NoLayersModule(torch.nn.Module):
            pass

        with pytest.raises(ValueError, match="layers"):
            FBCacheContext(NoLayersModule(), state)

    def test_full_forward_with_context(self):
        """Test full forward pass through wrapped layers."""
        config = FBCacheConfig(enabled=True, log_residuals=True)
        state = FBCacheState(config, num_inference_steps=4)
        transformer = MockTransformer(num_layers=3)

        x = torch.ones(1, 10, 64)

        with FBCacheContext(transformer, state) as ctx:
            # Step 0: Always compute (first step)
            ctx.set_sigma(0.9)
            out0 = transformer(x.clone())
            ctx.advance_step()

            # Step 1: Should compute (no cached output yet for remaining layers)
            ctx.set_sigma(0.5)
            out1 = transformer(x.clone())
            ctx.advance_step()

            # Step 2: Similar input, should potentially skip
            ctx.set_sigma(0.5)
            out2 = transformer(x.clone())
            ctx.advance_step()

            # Step 3: Last step, always compute
            ctx.set_sigma(0.1)
            out3 = transformer(x.clone())
            ctx.advance_step()

        stats = state.get_stats()
        assert stats["total_steps"] == 4
        # At minimum steps 0, 1 are computed (first has no cache, second builds cache)
        assert stats["computes"] >= 2
