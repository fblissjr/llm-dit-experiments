"""
Integration tests for LTX-2 Conditioning System.

Last Updated: 2026-01-18

These tests verify the conditioning system integration with the generation
pipeline and GPU operations. Tests require CUDA and are skipped otherwise.

Tests cover:
- LatentState with GPU tensors
- Conditioning with bfloat16 dtype
- Memory-efficient conditioning operations
- Integration with generate.py patterns

Run with: uv run pytest tests/integration/test_conditioning_integration.py -v
"""

import gc

import pytest
import torch


# Skip all tests in this module if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def cleanup_gpu():
    """Clean up GPU memory before and after each test."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    yield
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


# ============================================================================
# TestConditioningWithGenerate - 3 Tests
# ============================================================================


class TestConditioningWithGenerate:
    """Tests for conditioning integration with generation patterns."""

    def test_i2v_generation_flow(self):
        """Test Image-to-Video conditioning flow matches generate.py patterns."""
        from llm_dit.conditioning import (
            LatentState,
            VideoConditionByLatentIndex,
            timesteps_from_mask,
            post_process_latent,
        )

        # Simulate generation config
        # 33 frames -> 5 latent frames, 512x768 -> 16x24 latent spatial
        num_frames = 33
        height = 512
        width = 768
        t_latent = (num_frames - 1) // 8 + 1  # 5
        h_latent = height // 32  # 16
        w_latent = width // 32  # 24
        num_tokens = t_latent * h_latent * w_latent  # 1920

        # Create LatentState matching generate.py
        state = LatentState.create(
            shape=(1, num_tokens, 128),
            num_frames=num_frames,
            height=height,
            width=width,
            device="cuda",
            dtype=torch.bfloat16,
        )

        # Simulate VAE-encoded image for first frame
        # 1 frame, 16x24 spatial -> 384 tokens
        image_latent = torch.randn(1, 128, 1, h_latent, w_latent, device="cuda", dtype=torch.bfloat16)

        # Apply I2V conditioning
        cond = VideoConditionByLatentIndex(latent=image_latent, latent_idx=0, strength=1.0)
        state = cond.apply_to(state)

        # Add noise respecting mask
        generator = torch.Generator(device="cuda").manual_seed(42)
        state = state.add_noise(generator=generator, noise_scale=1.0)

        # Verify state is ready for denoising
        assert state.latent.shape == (1, num_tokens, 128)
        assert state.denoise_mask.shape == (1, num_tokens, 1)
        assert state.clean_latent is not None

        # First frame should have mask=0 (no denoising)
        first_frame_tokens = h_latent * w_latent
        assert (state.denoise_mask[:, :first_frame_tokens, :] == 0.0).all()
        # Other frames should have mask=1 (full denoising)
        assert (state.denoise_mask[:, first_frame_tokens:, :] == 1.0).all()

        # Test timestep scaling
        sigma = torch.tensor(0.5, device="cuda", dtype=torch.bfloat16)
        timesteps = timesteps_from_mask(state.denoise_mask, sigma)
        assert timesteps.shape == (1, num_tokens, 1)
        # First frame timesteps should be 0
        assert (timesteps[:, :first_frame_tokens, :] == 0.0).all()

        # Test post-processing
        denoised = torch.randn_like(state.latent)
        blended = post_process_latent(denoised, state.denoise_mask, state.clean_latent)
        assert blended.shape == state.latent.shape
        assert blended.dtype == torch.bfloat16

    def test_keyframe_continuation_flow(self):
        """Test keyframe conditioning for video continuation."""
        from llm_dit.conditioning import (
            LatentState,
            VideoConditionByKeyframeIndex,
        )

        # Start with a base video generation
        num_frames = 33
        height = 512
        width = 768
        t_latent = (num_frames - 1) // 8 + 1
        h_latent = height // 32
        w_latent = width // 32
        num_tokens = t_latent * h_latent * w_latent

        state = LatentState.create(
            shape=(1, num_tokens, 128),
            num_frames=num_frames,
            height=height,
            width=width,
            device="cuda",
            dtype=torch.bfloat16,
        )

        # Simulate keyframe from previous generation (last frame)
        keyframe = torch.randn(1, 128, 1, h_latent, w_latent, device="cuda", dtype=torch.bfloat16)

        # Append keyframe at frame position for continuation
        cond = VideoConditionByKeyframeIndex(keyframes=keyframe, frame_idx=t_latent, strength=0.9)
        state = cond.apply_to(state)

        # Verify sequence extended
        expected_tokens = num_tokens + h_latent * w_latent
        assert state.latent.shape == (1, expected_tokens, 128)
        assert state.positions.shape[2] == expected_tokens

        # Appended tokens should have low mask (mostly preserved)
        appended_mask = state.denoise_mask[:, num_tokens:, :]
        assert torch.allclose(appended_mask, torch.tensor(0.1, device="cuda", dtype=torch.bfloat16), atol=0.01)

    def test_denoise_loop_integration(self):
        """Test conditioning through simulated denoising loop."""
        from llm_dit.conditioning import (
            LatentState,
            VideoConditionByLatentIndex,
            timesteps_from_mask,
            post_process_latent,
        )

        # Smaller config for faster test
        num_frames = 9
        height = 128
        width = 128
        t_latent = (num_frames - 1) // 8 + 1  # 2
        h_latent = height // 32  # 4
        w_latent = width // 32  # 4
        num_tokens = t_latent * h_latent * w_latent  # 32

        state = LatentState.create(
            shape=(1, num_tokens, 128),
            num_frames=num_frames,
            height=height,
            width=width,
            device="cuda",
            dtype=torch.bfloat16,
        )

        # Apply I2V conditioning
        image_latent = torch.randn(1, 128, 1, h_latent, w_latent, device="cuda", dtype=torch.bfloat16)
        cond = VideoConditionByLatentIndex(latent=image_latent, latent_idx=0, strength=1.0)
        state = cond.apply_to(state)

        # Add initial noise
        generator = torch.Generator(device="cuda").manual_seed(42)
        state = state.add_noise(generator=generator, noise_scale=1.0)

        # Simulate denoising loop with 5 steps
        sigmas = torch.linspace(1.0, 0.0, 6, device="cuda", dtype=torch.bfloat16)

        for i in range(len(sigmas) - 1):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]

            # Compute per-token timesteps (key conditioning mechanism)
            timesteps = timesteps_from_mask(state.denoise_mask, sigma)

            # Simulate model forward pass
            # In real code: velocity, _ = model(video=modality)
            velocity = torch.randn_like(state.latent)

            # Euler step
            dt = sigma_next - sigma
            denoised = state.latent + velocity * dt

            # Apply post-processing (blend with clean_latent)
            denoised = post_process_latent(denoised, state.denoise_mask, state.clean_latent)

            # Update state for next iteration
            state = LatentState(
                latent=denoised,
                denoise_mask=state.denoise_mask,
                positions=state.positions,
                clean_latent=state.clean_latent,
                _latent_height=state._latent_height,
                _latent_width=state._latent_width,
                _num_frames=state._num_frames,
            )

        # Verify final state
        assert state.latent.shape == (1, num_tokens, 128)
        assert not torch.isnan(state.latent).any()
        assert not torch.isinf(state.latent).any()


# ============================================================================
# TestConditioningGPU - 3 Tests
# ============================================================================


class TestConditioningGPU:
    """Tests for GPU-specific conditioning operations."""

    def test_gpu_device_handling(self):
        """Test conditioning maintains GPU tensors correctly."""
        from llm_dit.conditioning import LatentState, VideoConditionByLatentIndex

        state = LatentState.create(
            shape=(1, 64, 128),
            num_frames=9,
            height=128,
            width=128,
            device="cuda",
            dtype=torch.float32,
        )

        # All tensors should be on GPU
        assert state.latent.device.type == "cuda"
        assert state.denoise_mask.device.type == "cuda"
        assert state.positions.device.type == "cuda"

        # Apply conditioning with GPU tensor
        image_latent = torch.randn(1, 128, 1, 4, 4, device="cuda")
        cond = VideoConditionByLatentIndex(latent=image_latent, latent_idx=0, strength=0.8)
        new_state = cond.apply_to(state)

        # Result should stay on GPU
        assert new_state.latent.device.type == "cuda"
        assert new_state.denoise_mask.device.type == "cuda"
        assert new_state.clean_latent.device.type == "cuda"

    def test_bfloat16_dtype_handling(self):
        """Test conditioning with bfloat16 precision."""
        from llm_dit.conditioning import (
            LatentState,
            VideoConditionByLatentIndex,
            timesteps_from_mask,
            post_process_latent,
        )

        state = LatentState.create(
            shape=(1, 64, 128),
            num_frames=9,
            height=128,
            width=128,
            device="cuda",
            dtype=torch.bfloat16,
        )

        # Apply conditioning
        image_latent = torch.randn(1, 128, 1, 4, 4, device="cuda", dtype=torch.bfloat16)
        cond = VideoConditionByLatentIndex(latent=image_latent, latent_idx=0, strength=1.0)
        state = cond.apply_to(state)

        # Verify dtype preserved
        assert state.latent.dtype == torch.bfloat16
        assert state.denoise_mask.dtype == torch.bfloat16
        assert state.clean_latent.dtype == torch.bfloat16

        # Test utility functions preserve dtype
        sigma = torch.tensor(0.5, device="cuda", dtype=torch.bfloat16)
        timesteps = timesteps_from_mask(state.denoise_mask, sigma)
        assert timesteps.dtype == torch.bfloat16

        denoised = torch.randn_like(state.latent)
        blended = post_process_latent(denoised, state.denoise_mask, state.clean_latent)
        assert blended.dtype == torch.bfloat16

    def test_memory_efficiency(self):
        """Test conditioning doesn't leak memory."""
        from llm_dit.conditioning import LatentState, VideoConditionByLatentIndex

        # Get baseline memory
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        baseline_memory = torch.cuda.memory_allocated()

        # Create large state
        state = LatentState.create(
            shape=(1, 1920, 128),  # 5 frames * 16 * 24 spatial
            num_frames=33,
            height=512,
            width=768,
            device="cuda",
            dtype=torch.bfloat16,
        )

        # Apply multiple conditionings
        for i in range(3):
            image_latent = torch.randn(1, 128, 1, 16, 24, device="cuda", dtype=torch.bfloat16)
            cond = VideoConditionByLatentIndex(latent=image_latent, latent_idx=i, strength=0.8)
            state = cond.apply_to(state)

        # Clean up
        del state, image_latent, cond
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        final_memory = torch.cuda.memory_allocated()

        # Memory should return close to baseline
        memory_leaked = final_memory - baseline_memory
        assert memory_leaked < 1024 * 1024, f"Memory leaked: {memory_leaked / 1024 / 1024:.2f} MB"
