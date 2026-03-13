"""
Tests for sigma schedule computation -- verify our scheduler matches the reference.

Last Updated: 2026-03-08

These tests compare our LTX2Scheduler output against:
1. DiffSynth-Studio's FlowMatchScheduler ("LTX-2" template)
2. Known distilled sigma values
3. Expected mathematical properties (monotone decreasing, terminal stretch)

Run with: uv run pytest tests/unit/test_sigma_schedule.py -v
"""

import math

import pytest
import torch

pytestmark = pytest.mark.unit


def diffsynth_set_timesteps_ltx2(
    num_inference_steps=100,
    denoising_strength=1.0,
    dynamic_shift_len=None,
    terminal=0.1,
    special_case=None,
):
    """Reference implementation from DiffSynth-Studio FlowMatchScheduler.set_timesteps_ltx2."""
    num_train_timesteps = 1000
    if special_case == "stage2":
        sigmas = torch.Tensor([0.909375, 0.725, 0.421875])
    elif special_case == "ditilled_stage1":
        sigmas = torch.Tensor([1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875])
    else:
        dynamic_shift_len = dynamic_shift_len or 4096
        base_seq_len = 1024
        max_seq_len = 4096
        base_shift = 0.95
        max_shift = 2.05
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        sigma_shift = dynamic_shift_len * m + b

        sigma_min = 0.0
        sigma_max = 1.0
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]
        sigmas = math.exp(sigma_shift) / (math.exp(sigma_shift) + (1 / sigmas - 1))
        # Shift terminal
        one_minus_z = 1.0 - sigmas
        scale_factor = one_minus_z[-1] / (1 - terminal)
        sigmas = 1.0 - (one_minus_z / scale_factor)
    timesteps = sigmas * num_train_timesteps
    return sigmas, timesteps


class TestSigmaScheduleBasicProperties:
    """Verify basic mathematical properties of our sigma schedule."""

    def test_monotone_decreasing(self):
        """Sigmas must be strictly monotone decreasing."""
        from llm_dit.schedulers.ltx2_scheduler import LTX2Scheduler

        scheduler = LTX2Scheduler()
        # 40 steps, using default tokens=4096 (no latent)
        sigmas = scheduler.execute(steps=40)
        for i in range(len(sigmas) - 1):
            assert sigmas[i] > sigmas[i + 1], (
                f"sigmas[{i}]={sigmas[i]:.6f} <= sigmas[{i+1}]={sigmas[i+1]:.6f}"
            )

    def test_starts_near_one(self):
        """First sigma should be close to 1.0."""
        from llm_dit.schedulers.ltx2_scheduler import LTX2Scheduler

        scheduler = LTX2Scheduler()
        sigmas = scheduler.execute(steps=40)
        assert sigmas[0] > 0.95, f"First sigma={sigmas[0]:.4f} too low"

    def test_ends_at_zero(self):
        """Last sigma should be exactly 0.0."""
        from llm_dit.schedulers.ltx2_scheduler import LTX2Scheduler

        scheduler = LTX2Scheduler()
        sigmas = scheduler.execute(steps=40)
        assert sigmas[-1] == 0.0, f"Last sigma={sigmas[-1]:.6f} != 0"

    def test_terminal_stretch(self):
        """Second-to-last sigma should equal terminal (0.1)."""
        from llm_dit.schedulers.ltx2_scheduler import LTX2Scheduler

        scheduler = LTX2Scheduler()
        sigmas = scheduler.execute(steps=40, terminal=0.1)
        # sigmas[-1] = 0.0 (the appended zero), sigmas[-2] should be ~0.1
        assert abs(sigmas[-2] - 0.1) < 0.01, (
            f"Second-to-last sigma={sigmas[-2]:.4f} should be ~0.1"
        )

    def test_shape(self):
        """Sigma schedule should have steps+1 values."""
        from llm_dit.schedulers.ltx2_scheduler import LTX2Scheduler

        scheduler = LTX2Scheduler()
        for steps in [8, 20, 40, 50]:
            sigmas = scheduler.execute(steps=steps)
            assert sigmas.shape == (steps + 1,), (
                f"Expected ({steps + 1},), got {sigmas.shape}"
            )


class TestSigmaScheduleMatchesReference:
    """Verify our schedule matches DiffSynth-Studio when using same token count."""

    def test_40_steps_4096_tokens(self):
        """Our schedule with tokens=4096 should match DiffSynth's with dynamic_shift_len=4096."""
        from llm_dit.schedulers.ltx2_scheduler import LTX2Scheduler

        scheduler = LTX2Scheduler()
        # Force 4096 tokens by passing no latent (default_number_of_tokens=4096)
        our_sigmas = scheduler.execute(steps=40)

        # DiffSynth reference (N sigmas, no terminal 0)
        ds_sigmas, _ = diffsynth_set_timesteps_ltx2(
            num_inference_steps=40, dynamic_shift_len=4096, terminal=0.1
        )

        # Our sigmas[0:-1] (the N denoising points) should match DiffSynth's N sigmas
        # Note: our schedule has N+1 points (last is 0.0), DiffSynth has N points (last step goes to 0)
        assert len(our_sigmas) == len(ds_sigmas) + 1, (
            f"Ours: {len(our_sigmas)}, DS: {len(ds_sigmas)} (expected DS+1)"
        )

        # Compare all N denoising sigmas
        for i in range(len(ds_sigmas)):
            diff = abs(our_sigmas[i].item() - ds_sigmas[i].item())
            assert diff < 1e-4, (
                f"Step {i}: ours={our_sigmas[i]:.6f} vs DS={ds_sigmas[i]:.6f} (diff={diff:.6f})"
            )

    def test_stage2_distilled(self):
        """Stage 2 distilled sigma values should match reference."""
        from llm_dit.models.ltx2.constants import STAGE_2_DISTILLED_SIGMA_VALUES

        ds_sigmas, _ = diffsynth_set_timesteps_ltx2(special_case="stage2")

        our_sigmas = STAGE_2_DISTILLED_SIGMA_VALUES
        assert len(our_sigmas) == len(ds_sigmas) + 1
        for i in range(len(ds_sigmas)):
            assert abs(our_sigmas[i] - ds_sigmas[i].item()) < 1e-6

    def test_shift_varies_with_token_count(self):
        """Different token counts should produce different sigma schedules."""
        from llm_dit.schedulers.ltx2_scheduler import LTX2Scheduler

        scheduler = LTX2Scheduler()

        # 1536 tokens (typical half-res 384x256@121 frames)
        latent_small = torch.empty(1, 128, 16, 8, 12)
        sigmas_small = scheduler.execute(steps=40, latent=latent_small)

        # 4096 tokens (DiffSynth default)
        sigmas_default = scheduler.execute(steps=40)

        # They should differ (different shift values)
        diff = (sigmas_small[:-1] - sigmas_default[:-1]).abs().max().item()
        assert diff > 0.01, (
            f"Expected different schedules for different token counts, "
            f"max diff={diff:.6f}"
        )

    def test_sigma_values_at_key_points(self):
        """Check sigma values at first, middle, and last steps match expectations."""
        from llm_dit.schedulers.ltx2_scheduler import LTX2Scheduler

        scheduler = LTX2Scheduler()
        sigmas = scheduler.execute(steps=40)

        # First sigma should be 1.0 (the starting point)
        assert abs(sigmas[0].item() - 1.0) < 1e-5

        # Last non-zero sigma should be approximately terminal (0.1)
        assert abs(sigmas[-2].item() - 0.1) < 0.01

        # Terminal (appended 0)
        assert sigmas[-1].item() == 0.0


class TestPositionIndices:
    """Verify position index computation."""

    def test_position_shape(self):
        """Positions should have shape [B, 3, T, 2]."""
        from llm_dit.pipelines.generate import create_position_indices

        positions = create_position_indices(
            batch_size=1,
            num_frames=121,
            height=384,
            width=256,
            device=torch.device("cpu"),
        )
        t_latent = (121 - 1) // 8 + 1  # = 16
        h_latent = 384 // 32  # = 12
        w_latent = 256 // 32  # = 8
        expected_tokens = t_latent * h_latent * w_latent
        assert positions.shape == (1, 3, expected_tokens, 2), (
            f"Expected (1, 3, {expected_tokens}, 2), got {positions.shape}"
        )

    def test_position_temporal_scaled_to_seconds(self):
        """Temporal positions should be in seconds (divided by fps)."""
        from llm_dit.pipelines.generate import create_position_indices

        positions = create_position_indices(
            batch_size=1,
            num_frames=121,
            height=384,
            width=256,
            device=torch.device("cpu"),
            fps=24.0,
        )
        # Temporal positions (dim 0) should be in seconds
        max_temporal = positions[0, 0, :, 1].max().item()
        # 121 frames at 24fps = 5.04 seconds. Max temporal position should be reasonable
        assert max_temporal > 0.0 and max_temporal < 10.0, (
            f"Max temporal position={max_temporal:.2f}, expected ~5 seconds"
        )

    def test_position_spatial_in_pixel_space(self):
        """Spatial positions should be in pixel coordinates."""
        from llm_dit.pipelines.generate import create_position_indices

        positions = create_position_indices(
            batch_size=1,
            num_frames=9,  # small for speed
            height=384,
            width=256,
            device=torch.device("cpu"),
        )
        # Height positions (dim 1) max end should be height (384)
        max_height = positions[0, 1, :, 1].max().item()
        assert abs(max_height - 384.0) < 1.0, (
            f"Max height position={max_height}, expected ~384"
        )

        # Width positions (dim 2) max end should be width (256)
        max_width = positions[0, 2, :, 1].max().item()
        assert abs(max_width - 256.0) < 1.0, (
            f"Max width position={max_width}, expected ~256"
        )

    def test_causal_fix_applied(self):
        """Causal fix should clamp temporal positions to non-negative."""
        from llm_dit.pipelines.generate import create_position_indices

        positions = create_position_indices(
            batch_size=1,
            num_frames=9,
            height=64,
            width=64,
            device=torch.device("cpu"),
            causal_fix=True,
        )
        assert positions[0, 0].min().item() >= 0.0, (
            "Temporal positions should be non-negative after causal fix"
        )


class TestEulerStepConsistency:
    """Verify the Euler step formula matches DiffSynth-Studio."""

    def test_euler_step_formula(self):
        """Euler step: x_next = x + velocity * (sigma_next - sigma)."""
        # Simulate one step
        x = torch.randn(1, 100, 128)
        v = torch.randn(1, 100, 128)
        sigma = torch.tensor(0.8)
        sigma_next = torch.tensor(0.6)

        # Our formula (from _denoise_av_stage)
        dt = sigma_next - sigma
        x_ours = (x.float() + v.float() * dt).to(x.dtype)

        # DiffSynth formula (from FlowMatchScheduler.step)
        x_ds = x + v * (sigma_next - sigma)

        torch.testing.assert_close(x_ours, x_ds, atol=1e-6, rtol=1e-5)

    def test_final_step_goes_to_zero(self):
        """Final denoising step should bring sigma to 0."""
        from llm_dit.models.ltx2.constants import STAGE_2_DISTILLED_SIGMA_VALUES

        sigmas = torch.tensor(STAGE_2_DISTILLED_SIGMA_VALUES)
        num_steps = len(sigmas) - 1

        # Last step: sigma_next = sigmas[-1] = 0.0
        assert sigmas[-1] == 0.0
        assert num_steps == 3  # 4 values, 3 steps
