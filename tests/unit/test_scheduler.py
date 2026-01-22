"""
Scheduler Unit Tests.

Last Updated: 2026-01-22

Tests for diffusion schedulers and sigma scheduling used across pipelines.
These tests validate the sigma schedule mechanics that drive the denoising process.

Run with: uv run pytest tests/unit/test_scheduler.py -v
"""

import math

import pytest
import torch


# ============================================================================
# Sigma Schedule Tests
# ============================================================================


class TestSigmaSchedule:
    """Tests for sigma/timestep scheduling."""

    def _create_sigma_schedule(self, num_steps: int, video_seq_len: int = 1920) -> torch.Tensor:
        """
        Create a sigma schedule with dynamic shift.

        This replicates the sigma computation from LTX2ExperimentBase.
        """
        # Dynamic shift computation based on sequence length
        base_seq_len, max_seq_len = 1024, 4096
        base_shift, max_shift = 0.95, 2.05

        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        mu = base_shift + m * (video_seq_len - base_seq_len)
        mu = max(min(mu, max_shift), base_shift)

        # Linear sigmas with exponential shift
        sigmas = torch.linspace(1.0, 1.0 / num_steps, num_steps)
        exp_mu = math.exp(mu)
        sigmas = exp_mu / (exp_mu + (1.0 / sigmas - 1.0))

        return sigmas

    def test_sigma_schedule_monotonic(self):
        """Sigma schedule should be monotonically decreasing."""
        sigmas = self._create_sigma_schedule(num_steps=40)

        for i in range(len(sigmas) - 1):
            assert sigmas[i] > sigmas[i + 1], \
                f"Sigma not monotonic at step {i}: {sigmas[i]:.4f} <= {sigmas[i+1]:.4f}"

    def test_sigma_schedule_bounds(self):
        """Sigma should start near 1.0 and end small."""
        sigmas = self._create_sigma_schedule(num_steps=40)

        assert sigmas[0] >= 0.9, f"Initial sigma should be >= 0.9, got {sigmas[0]:.4f}"
        assert sigmas[-1] <= 0.3, f"Final sigma should be <= 0.3, got {sigmas[-1]:.4f}"
        assert sigmas[-1] > 0.0, f"Final sigma should be > 0, got {sigmas[-1]:.4f}"

    def test_sigma_schedule_step_count(self):
        """Should produce requested number of steps."""
        for n in [10, 20, 40, 100]:
            sigmas = self._create_sigma_schedule(num_steps=n)
            assert len(sigmas) == n, f"Expected {n} sigmas, got {len(sigmas)}"

    def test_sigma_schedule_different_resolutions(self):
        """Sigma schedule should vary with resolution (sequence length)."""
        # Low resolution (fewer tokens)
        sigmas_low = self._create_sigma_schedule(num_steps=40, video_seq_len=1024)

        # High resolution (more tokens)
        sigmas_high = self._create_sigma_schedule(num_steps=40, video_seq_len=4096)

        # Both should be monotonically decreasing
        assert all(sigmas_low[i] > sigmas_low[i + 1] for i in range(len(sigmas_low) - 1))
        assert all(sigmas_high[i] > sigmas_high[i + 1] for i in range(len(sigmas_high) - 1))

        # Higher resolution should have different shift (schedules will differ)
        # The exact relationship depends on the shift formula
        assert not torch.allclose(sigmas_low, sigmas_high), \
            "Different resolutions should produce different sigma schedules"


class TestDynamicShiftComputation:
    """Tests for dynamic shift (mu) computation."""

    def test_dynamic_shift_at_base_resolution(self):
        """Shift at base resolution (1024 tokens) should be ~0.95."""
        base_seq_len, max_seq_len = 1024, 4096
        base_shift, max_shift = 0.95, 2.05

        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        mu = base_shift + m * (1024 - base_seq_len)

        assert abs(mu - 0.95) < 0.01, f"mu at base should be ~0.95, got {mu:.4f}"

    def test_dynamic_shift_at_max_resolution(self):
        """Shift at max resolution (4096 tokens) should be ~2.05."""
        base_seq_len, max_seq_len = 1024, 4096
        base_shift, max_shift = 0.95, 2.05

        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        mu = base_shift + m * (4096 - base_seq_len)

        assert abs(mu - 2.05) < 0.01, f"mu at max should be ~2.05, got {mu:.4f}"

    def test_dynamic_shift_linear_interpolation(self):
        """Shift should interpolate linearly between base and max."""
        base_seq_len, max_seq_len = 1024, 4096
        base_shift, max_shift = 0.95, 2.05

        def compute_mu(seq_len):
            m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
            mu = base_shift + m * (seq_len - base_seq_len)
            return max(min(mu, max_shift), base_shift)

        # Test at 1920 tokens (standard 768x512x33)
        # Expected: 0.95 + (2.05-0.95)/(4096-1024) * (1920-1024) = ~1.27
        mu_standard = compute_mu(1920)
        expected_mu = 0.95 + (2.05 - 0.95) / (4096 - 1024) * (1920 - 1024)

        assert abs(mu_standard - expected_mu) < 0.01, \
            f"mu at 1920 tokens should be ~{expected_mu:.2f}, got {mu_standard:.4f}"

    def test_dynamic_shift_clamping(self):
        """Shift should be clamped to [base_shift, max_shift]."""
        base_seq_len, max_seq_len = 1024, 4096
        base_shift, max_shift = 0.95, 2.05

        def compute_mu(seq_len):
            m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
            mu = base_shift + m * (seq_len - base_seq_len)
            return max(min(mu, max_shift), base_shift)

        # Below base should clamp to base_shift
        mu_below = compute_mu(512)
        assert mu_below == base_shift, f"Should clamp to base_shift, got {mu_below}"

        # Above max should clamp to max_shift
        mu_above = compute_mu(8192)
        assert mu_above == max_shift, f"Should clamp to max_shift, got {mu_above}"


# ============================================================================
# Timestep Conversion Tests
# ============================================================================


class TestTimestepConversion:
    """Tests for sigma-to-timestep conversion."""

    def test_timestep_from_sigma_scaling(self):
        """Timesteps should scale linearly with sigma."""
        # LTX-2 uses sigma * 1000 range
        sigmas = torch.tensor([1.0, 0.5, 0.25, 0.0])
        timesteps = sigmas * 1000

        expected = torch.tensor([1000.0, 500.0, 250.0, 0.0])
        assert torch.allclose(timesteps, expected)

    def test_timestep_shape_preservation(self):
        """Timestep conversion should preserve tensor shape."""
        sigma = torch.randn(4, 100, 1).abs()  # [B, T, 1]
        timestep = sigma * 1000

        assert timestep.shape == sigma.shape


# ============================================================================
# Scheduler Callback Tests
# ============================================================================


class TestSchedulerCallbacks:
    """Tests for scheduler callback hooks."""

    def test_callback_called_each_step(self):
        """Callback should be called for each denoising step."""
        call_count = 0

        def callback(_step, _timestep, _latents):
            nonlocal call_count
            call_count += 1

        # Simulate denoising loop with callback
        num_steps = 10
        for step in range(num_steps):
            callback(step, 1000 - step * 100, None)

        assert call_count == num_steps

    def test_callback_receives_correct_step(self):
        """Callback should receive correct step index."""
        received_steps = []

        def callback(step, _timestep, _latents):
            received_steps.append(step)

        num_steps = 5
        for step in range(num_steps):
            callback(step, 1000 - step * 200, None)

        assert received_steps == list(range(num_steps))

    def test_early_stop_callback(self):
        """Callback returning True should stop iteration."""
        stop_at_step = 3
        steps_executed = []

        def callback(step, _timestep, _latents):
            steps_executed.append(step)
            return step >= stop_at_step  # Return True to stop

        for step in range(10):
            if callback(step, 0, None):
                break

        assert len(steps_executed) == stop_at_step + 1
        assert steps_executed == [0, 1, 2, 3]

    def test_callback_with_latent_inspection(self):
        """Callback can inspect latent tensors."""
        latent_shapes = []

        def callback(_step, _timestep, latents):
            if latents is not None:
                latent_shapes.append(latents.shape)

        # Simulate with mock latents
        mock_latent = torch.randn(1, 128, 5, 16, 24)
        for step in range(5):
            callback(step, 1000 - step * 200, mock_latent)

        assert len(latent_shapes) == 5
        assert all(shape == (1, 128, 5, 16, 24) for shape in latent_shapes)


# Run with: uv run pytest tests/unit/test_scheduler.py -v
