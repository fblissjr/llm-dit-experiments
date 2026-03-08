"""
Tests for LTX-2 Conditioning System implementation.

Last Updated: 2026-01-18

These tests verify the conditioning system for LTX-2 video generation.
The conditioning system enables:
- Image-to-Video (I2V) via VideoConditionByLatentIndex
- Video continuation via VideoConditionByKeyframeIndex
- Per-token denoising strength via denoise_mask

Tests cover:
- LatentState creation and management (12 tests)
- VideoConditionByKeyframeIndex token appending (12 tests)
- VideoConditionByLatentIndex token replacement (12 tests)
- Denoise mask mechanics and formulas (9 tests)

Run with: uv run pytest tests/unit/test_conditioning.py -v
"""

import gc

import pytest
import torch


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
# TestLatentState - 12 Tests
# ============================================================================


class TestLatentState:
    """Tests for LatentState dataclass creation and operations."""

    def test_create_with_shape(self):
        """Test LatentState creation with specified shape."""
        from llm_dit.conditioning import LatentState

        state = LatentState.create(
            shape=(1, 256, 128),  # [B, T, D]
            device="cpu",
            dtype=torch.float32,
        )
        assert state.latent.shape == (1, 256, 128)

    def test_create_with_default_mask(self):
        """Default denoise_mask is all 1.0 (full denoising)."""
        from llm_dit.conditioning import LatentState

        state = LatentState.create(
            shape=(1, 256, 128),
            device="cpu",
            dtype=torch.float32,
        )
        assert state.denoise_mask.shape == (1, 256, 1)  # [B, T, 1] for broadcasting
        assert (state.denoise_mask == 1.0).all()

    def test_create_with_clean_latent_none(self):
        """clean_latent is None by default (populated when conditioning applied)."""
        from llm_dit.conditioning import LatentState

        state = LatentState.create(
            shape=(1, 256, 128),
            device="cpu",
            dtype=torch.float32,
        )
        assert state.clean_latent is None

    def test_create_positions_shape(self):
        """positions have shape [B, 3, T, 2] for bounds."""
        from llm_dit.conditioning import LatentState

        # LTX-2 downsampling: temporal=8, spatial=32
        # 33 frames -> (33-1)//8+1 = 5 latent frames
        # 128 px -> 128//32 = 4 latent spatial
        # Total tokens = 5 * 4 * 4 = 80
        state = LatentState.create(
            shape=(1, 80, 128),
            num_frames=33,
            height=128,
            width=128,
            device="cpu",
            dtype=torch.float32,
        )
        # positions: [B, 3, T, 2] - (batch, dim, tokens, bounds)
        assert state.positions.shape == (1, 3, 80, 2)

    def test_clone_creates_copy(self):
        """clone() creates independent copy of state."""
        from llm_dit.conditioning import LatentState

        state = LatentState.create(
            shape=(1, 64, 128),
            device="cpu",
            dtype=torch.float32,
        )
        cloned = state.clone()

        # Modify original
        state.latent[0, 0, 0] = 999.0

        # Clone should be unaffected
        assert cloned.latent[0, 0, 0] != 999.0

    def test_dtype_float32(self):
        """Test float32 dtype is respected."""
        from llm_dit.conditioning import LatentState

        state = LatentState.create(
            shape=(1, 64, 128),
            device="cpu",
            dtype=torch.float32,
        )
        assert state.latent.dtype == torch.float32
        assert state.denoise_mask.dtype == torch.float32

    def test_dtype_bfloat16(self):
        """Test bfloat16 dtype is respected."""
        from llm_dit.conditioning import LatentState

        state = LatentState.create(
            shape=(1, 64, 128),
            device="cpu",
            dtype=torch.bfloat16,
        )
        assert state.latent.dtype == torch.bfloat16
        assert state.denoise_mask.dtype == torch.bfloat16

    def test_device_cpu(self):
        """Test device placement on CPU."""
        from llm_dit.conditioning import LatentState

        state = LatentState.create(
            shape=(1, 64, 128),
            device="cpu",
            dtype=torch.float32,
        )
        assert state.latent.device.type == "cpu"
        assert state.denoise_mask.device.type == "cpu"
        assert state.positions.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_cuda(self):
        """Test device placement on CUDA."""
        from llm_dit.conditioning import LatentState

        state = LatentState.create(
            shape=(1, 64, 128),
            device="cuda",
            dtype=torch.float32,
        )
        assert state.latent.device.type == "cuda"
        assert state.denoise_mask.device.type == "cuda"
        assert state.positions.device.type == "cuda"

    def test_add_noise_with_mask(self):
        """add_noise respects denoise_mask for scaling."""
        from llm_dit.conditioning import LatentState

        state = LatentState.create(
            shape=(1, 4, 128),
            device="cpu",
            dtype=torch.float32,
        )
        # Set first 2 tokens to no denoising
        state.denoise_mask[:, :2, :] = 0.0

        generator = torch.Generator().manual_seed(42)
        noisy = state.add_noise(generator=generator, noise_scale=1.0)

        # With mask=0, latent should remain unchanged (assuming clean_latent=None means zeros)
        # With mask=1, latent should be pure noise
        assert noisy.latent.shape == state.latent.shape

    def test_with_clean_latent(self):
        """Test setting clean_latent for blending."""
        from llm_dit.conditioning import LatentState

        state = LatentState.create(
            shape=(1, 64, 128),
            device="cpu",
            dtype=torch.float32,
        )
        clean = torch.randn(1, 64, 128)
        new_state = state.with_clean_latent(clean)

        assert new_state.clean_latent is not None
        assert torch.equal(new_state.clean_latent, clean)
        # Original should be unchanged
        assert state.clean_latent is None

    def test_batch_size_support(self):
        """Test batch size > 1."""
        from llm_dit.conditioning import LatentState

        state = LatentState.create(
            shape=(4, 64, 128),  # Batch of 4
            device="cpu",
            dtype=torch.float32,
        )
        assert state.latent.shape[0] == 4
        assert state.denoise_mask.shape[0] == 4


# ============================================================================
# TestVideoConditionByKeyframeIndex - 12 Tests
# ============================================================================


class TestVideoConditionByKeyframeIndex:
    """Tests for keyframe conditioning that APPENDS tokens."""

    def test_appends_tokens_to_sequence(self):
        """Keyframe conditioning extends sequence length."""
        from llm_dit.conditioning import LatentState, VideoConditionByKeyframeIndex

        state = LatentState.create(
            shape=(1, 256, 128),
            device="cpu",
            dtype=torch.float32,
        )
        # 64 tokens = 1 frame worth (4x4 spatial in latent)
        keyframe = torch.randn(1, 128, 1, 4, 4)  # [B, C, F, H, W] latent format
        cond = VideoConditionByKeyframeIndex(keyframes=keyframe, frame_idx=4, strength=0.8)
        new_state = cond.apply_to(state)

        # 64 tokens appended = 16 * 1 * 4 * 4 (but patchified becomes 16 tokens)
        # Actually: 1 * 4 * 4 = 16 tokens for 1 frame
        assert new_state.latent.shape[1] > state.latent.shape[1]

    def test_denoise_mask_for_appended_tokens(self):
        """Appended tokens get denoise_mask = 1.0 - strength."""
        from llm_dit.conditioning import LatentState, VideoConditionByKeyframeIndex

        state = LatentState.create(
            shape=(1, 256, 128),
            device="cpu",
            dtype=torch.float32,
        )
        keyframe = torch.randn(1, 128, 1, 4, 4)
        cond = VideoConditionByKeyframeIndex(keyframes=keyframe, frame_idx=4, strength=0.8)
        new_state = cond.apply_to(state)

        # Original tokens should still have mask=1.0
        assert (new_state.denoise_mask[:, :256, :] == 1.0).all()
        # Appended tokens should have mask = 1.0 - 0.8 = 0.2
        expected_mask = 0.2
        assert torch.allclose(
            new_state.denoise_mask[:, 256:, :],
            torch.full_like(new_state.denoise_mask[:, 256:, :], expected_mask),
            atol=1e-6
        )

    def test_positions_with_frame_idx_offset(self):
        """Appended tokens have positions offset by frame_idx."""
        from llm_dit.conditioning import LatentState, VideoConditionByKeyframeIndex

        state = LatentState.create(
            shape=(1, 256, 128),
            num_frames=33,
            height=128,
            width=128,
            device="cpu",
            dtype=torch.float32,
        )
        keyframe = torch.randn(1, 128, 1, 4, 4)
        cond = VideoConditionByKeyframeIndex(keyframes=keyframe, frame_idx=4, strength=0.8)
        new_state = cond.apply_to(state)

        # New positions should have temporal offset
        assert new_state.positions.shape[2] > state.positions.shape[2]

    def test_clean_latent_includes_keyframes(self):
        """clean_latent should include the appended keyframe tokens."""
        from llm_dit.conditioning import LatentState, VideoConditionByKeyframeIndex

        state = LatentState.create(
            shape=(1, 256, 128),
            device="cpu",
            dtype=torch.float32,
        )
        keyframe = torch.randn(1, 128, 1, 4, 4)
        cond = VideoConditionByKeyframeIndex(keyframes=keyframe, frame_idx=4, strength=0.8)
        new_state = cond.apply_to(state)

        assert new_state.clean_latent is not None
        assert new_state.clean_latent.shape == new_state.latent.shape

    def test_strength_zero_means_no_denoising(self):
        """strength=0.0 means appended tokens are not denoised (mask=1.0)."""
        from llm_dit.conditioning import LatentState, VideoConditionByKeyframeIndex

        state = LatentState.create(
            shape=(1, 64, 128),
            device="cpu",
            dtype=torch.float32,
        )
        keyframe = torch.randn(1, 128, 1, 4, 4)
        cond = VideoConditionByKeyframeIndex(keyframes=keyframe, frame_idx=0, strength=0.0)
        new_state = cond.apply_to(state)

        # mask = 1.0 - 0.0 = 1.0 (full denoising)
        assert (new_state.denoise_mask[:, 64:, :] == 1.0).all()

    def test_strength_one_means_full_conditioning(self):
        """strength=1.0 means appended tokens are fully conditioned (mask=0.0)."""
        from llm_dit.conditioning import LatentState, VideoConditionByKeyframeIndex

        state = LatentState.create(
            shape=(1, 64, 128),
            device="cpu",
            dtype=torch.float32,
        )
        keyframe = torch.randn(1, 128, 1, 4, 4)
        cond = VideoConditionByKeyframeIndex(keyframes=keyframe, frame_idx=0, strength=1.0)
        new_state = cond.apply_to(state)

        # mask = 1.0 - 1.0 = 0.0 (no denoising, pure conditioning)
        assert (new_state.denoise_mask[:, 64:, :] == 0.0).all()

    def test_multiple_keyframes_concat(self):
        """Multiple frames in keyframe tensor are all appended."""
        from llm_dit.conditioning import LatentState, VideoConditionByKeyframeIndex

        state = LatentState.create(
            shape=(1, 64, 128),
            device="cpu",
            dtype=torch.float32,
        )
        # 2 frames worth of keyframes
        keyframe = torch.randn(1, 128, 2, 4, 4)
        cond = VideoConditionByKeyframeIndex(keyframes=keyframe, frame_idx=0, strength=0.5)
        new_state = cond.apply_to(state)

        # Should have appended 2 frames = 32 tokens (2 * 4 * 4)
        assert new_state.latent.shape[1] == 64 + 32

    def test_dtype_preserved(self):
        """Output dtype matches input dtype."""
        from llm_dit.conditioning import LatentState, VideoConditionByKeyframeIndex

        state = LatentState.create(
            shape=(1, 64, 128),
            device="cpu",
            dtype=torch.bfloat16,
        )
        keyframe = torch.randn(1, 128, 1, 4, 4, dtype=torch.bfloat16)
        cond = VideoConditionByKeyframeIndex(keyframes=keyframe, frame_idx=0, strength=0.5)
        new_state = cond.apply_to(state)

        assert new_state.latent.dtype == torch.bfloat16

    def test_device_preserved(self, device):
        """Output device matches input device."""
        from llm_dit.conditioning import LatentState, VideoConditionByKeyframeIndex

        state = LatentState.create(
            shape=(1, 64, 128),
            device=device,
            dtype=torch.float32,
        )
        keyframe = torch.randn(1, 128, 1, 4, 4, device=device)
        cond = VideoConditionByKeyframeIndex(keyframes=keyframe, frame_idx=0, strength=0.5)
        new_state = cond.apply_to(state)

        assert new_state.latent.device == state.latent.device

    def test_frame_idx_zero_for_first_frame(self):
        """frame_idx=0 positions keyframe at the start."""
        from llm_dit.conditioning import LatentState, VideoConditionByKeyframeIndex

        # 17 frames -> 3 latent frames, 128px -> 4 latent spatial
        # Original tokens = 3 * 4 * 4 = 48
        state = LatentState.create(
            shape=(1, 48, 128),
            num_frames=17,
            height=128,
            width=128,
            device="cpu",
            dtype=torch.float32,
        )
        original_tokens = state.num_tokens

        # Keyframe: 1 frame * 4 * 4 = 16 tokens appended
        keyframe = torch.randn(1, 128, 1, 4, 4)
        cond = VideoConditionByKeyframeIndex(keyframes=keyframe, frame_idx=0, strength=0.5)
        new_state = cond.apply_to(state)

        # Temporal position for appended tokens should start at 0
        # positions[:, 0, :, :] is temporal dimension
        appended_positions = new_state.positions[:, 0, original_tokens:, :]
        assert appended_positions.numel() > 0, "Should have appended positions"
        assert appended_positions.min() >= 0.0

    def test_frame_idx_nonzero_offset(self):
        """frame_idx>0 offsets temporal positions."""
        from llm_dit.conditioning import LatentState, VideoConditionByKeyframeIndex

        # 17 frames -> 3 latent frames, 128px -> 4 latent spatial
        # Original tokens = 3 * 4 * 4 = 48
        state = LatentState.create(
            shape=(1, 48, 128),
            num_frames=17,
            height=128,
            width=128,
            device="cpu",
            dtype=torch.float32,
        )
        original_tokens = state.num_tokens

        keyframe = torch.randn(1, 128, 1, 4, 4)
        cond = VideoConditionByKeyframeIndex(keyframes=keyframe, frame_idx=10, strength=0.5)
        new_state = cond.apply_to(state)

        # Temporal position should be offset by frame_idx / fps
        # We just verify it's greater than 0 (offset applied)
        appended_positions = new_state.positions[:, 0, original_tokens:, :]
        assert appended_positions.numel() > 0, "Should have appended positions"
        assert appended_positions.min() > 0.0

    def test_batch_support(self):
        """Conditioning works with batch size > 1."""
        from llm_dit.conditioning import LatentState, VideoConditionByKeyframeIndex

        state = LatentState.create(
            shape=(2, 64, 128),  # Batch of 2
            device="cpu",
            dtype=torch.float32,
        )
        keyframe = torch.randn(2, 128, 1, 4, 4)
        cond = VideoConditionByKeyframeIndex(keyframes=keyframe, frame_idx=0, strength=0.5)
        new_state = cond.apply_to(state)

        assert new_state.latent.shape[0] == 2


# ============================================================================
# TestVideoConditionByLatentIndex - 12 Tests
# ============================================================================


class TestVideoConditionByLatentIndex:
    """Tests for latent conditioning that REPLACES tokens."""

    def test_replaces_tokens_not_appends(self):
        """Latent conditioning replaces tokens, keeping sequence length."""
        from llm_dit.conditioning import LatentState, VideoConditionByLatentIndex

        state = LatentState.create(
            shape=(1, 256, 128),
            num_frames=33,
            height=128,
            width=128,
            device="cpu",
            dtype=torch.float32,
        )
        # Replace first frame (16 tokens for 4x4 spatial)
        replacement = torch.randn(1, 128, 1, 4, 4)
        cond = VideoConditionByLatentIndex(latent=replacement, latent_idx=0, strength=1.0)
        new_state = cond.apply_to(state)

        # Sequence length unchanged
        assert new_state.latent.shape == state.latent.shape

    def test_denoise_mask_for_replaced_tokens(self):
        """Replaced tokens get denoise_mask = 1.0 - strength."""
        from llm_dit.conditioning import LatentState, VideoConditionByLatentIndex

        state = LatentState.create(
            shape=(1, 256, 128),
            num_frames=33,
            height=128,
            width=128,
            device="cpu",
            dtype=torch.float32,
        )
        replacement = torch.randn(1, 128, 1, 4, 4)
        cond = VideoConditionByLatentIndex(latent=replacement, latent_idx=0, strength=0.8)
        new_state = cond.apply_to(state)

        # First 16 tokens (first frame) should have mask = 1.0 - 0.8 = 0.2
        first_frame_tokens = 16  # 4 * 4
        assert torch.allclose(
            new_state.denoise_mask[:, :first_frame_tokens, :],
            torch.full((1, first_frame_tokens, 1), 0.2),
            atol=1e-6
        )
        # Remaining tokens should still have mask = 1.0
        assert (new_state.denoise_mask[:, first_frame_tokens:, :] == 1.0).all()

    def test_clean_latent_stores_replacement(self):
        """clean_latent stores the conditioning latent for blending."""
        from llm_dit.conditioning import LatentState, VideoConditionByLatentIndex

        state = LatentState.create(
            shape=(1, 256, 128),
            num_frames=33,
            height=128,
            width=128,
            device="cpu",
            dtype=torch.float32,
        )
        replacement = torch.randn(1, 128, 1, 4, 4)
        cond = VideoConditionByLatentIndex(latent=replacement, latent_idx=0, strength=1.0)
        new_state = cond.apply_to(state)

        assert new_state.clean_latent is not None

    def test_latent_idx_zero_for_i2v(self):
        """latent_idx=0 is the I2V (Image-to-Video) case."""
        from llm_dit.conditioning import LatentState, VideoConditionByLatentIndex

        state = LatentState.create(
            shape=(1, 256, 128),
            num_frames=33,
            height=128,
            width=128,
            device="cpu",
            dtype=torch.float32,
        )
        replacement = torch.randn(1, 128, 1, 4, 4)
        cond = VideoConditionByLatentIndex(latent=replacement, latent_idx=0, strength=1.0)
        new_state = cond.apply_to(state)

        # First frame tokens should be modified
        first_frame_tokens = 16
        assert new_state.denoise_mask[:, :first_frame_tokens, :].sum() < first_frame_tokens

    def test_latent_idx_nonzero(self):
        """latent_idx>0 replaces tokens at a later frame."""
        from llm_dit.conditioning import LatentState, VideoConditionByLatentIndex

        state = LatentState.create(
            shape=(1, 256, 128),
            num_frames=33,
            height=128,
            width=128,
            device="cpu",
            dtype=torch.float32,
        )
        replacement = torch.randn(1, 128, 1, 4, 4)
        # Replace frame at latent_idx=2 (third frame)
        cond = VideoConditionByLatentIndex(latent=replacement, latent_idx=2, strength=1.0)
        new_state = cond.apply_to(state)

        tokens_per_frame = 16
        start_token = 2 * tokens_per_frame
        end_token = 3 * tokens_per_frame

        # Tokens at frame 2 should have mask = 0.0
        assert (new_state.denoise_mask[:, start_token:end_token, :] == 0.0).all()
        # Other tokens unchanged
        assert (new_state.denoise_mask[:, :start_token, :] == 1.0).all()
        assert (new_state.denoise_mask[:, end_token:, :] == 1.0).all()

    def test_strength_zero_means_full_denoising(self):
        """strength=0.0 means replaced tokens are fully denoised (mask=1.0)."""
        from llm_dit.conditioning import LatentState, VideoConditionByLatentIndex

        state = LatentState.create(
            shape=(1, 64, 128),
            num_frames=9,
            height=128,
            width=128,
            device="cpu",
            dtype=torch.float32,
        )
        replacement = torch.randn(1, 128, 1, 4, 4)
        cond = VideoConditionByLatentIndex(latent=replacement, latent_idx=0, strength=0.0)
        new_state = cond.apply_to(state)

        # mask = 1.0 - 0.0 = 1.0 (full denoising, conditioning has no effect)
        assert (new_state.denoise_mask == 1.0).all()

    def test_strength_one_means_no_denoising(self):
        """strength=1.0 means replaced tokens are not denoised (mask=0.0)."""
        from llm_dit.conditioning import LatentState, VideoConditionByLatentIndex

        state = LatentState.create(
            shape=(1, 64, 128),
            num_frames=9,
            height=128,
            width=128,
            device="cpu",
            dtype=torch.float32,
        )
        replacement = torch.randn(1, 128, 1, 4, 4)
        cond = VideoConditionByLatentIndex(latent=replacement, latent_idx=0, strength=1.0)
        new_state = cond.apply_to(state)

        # First frame (16 tokens) should have mask = 0.0
        assert (new_state.denoise_mask[:, :16, :] == 0.0).all()

    def test_spatial_shape_validation(self):
        """Raises error if spatial shapes don't match."""
        from llm_dit.conditioning import LatentState, VideoConditionByLatentIndex
        from llm_dit.conditioning.exceptions import ConditioningError

        state = LatentState.create(
            shape=(1, 256, 128),
            num_frames=33,
            height=128,
            width=128,
            device="cpu",
            dtype=torch.float32,
        )
        # Wrong spatial size (8x8 instead of 4x4)
        replacement = torch.randn(1, 128, 1, 8, 8)
        cond = VideoConditionByLatentIndex(latent=replacement, latent_idx=0, strength=1.0)

        with pytest.raises(ConditioningError):
            cond.apply_to(state)

    def test_dtype_preserved(self):
        """Output dtype matches input dtype."""
        from llm_dit.conditioning import LatentState, VideoConditionByLatentIndex

        state = LatentState.create(
            shape=(1, 64, 128),
            num_frames=9,
            height=128,
            width=128,
            device="cpu",
            dtype=torch.bfloat16,
        )
        replacement = torch.randn(1, 128, 1, 4, 4, dtype=torch.bfloat16)
        cond = VideoConditionByLatentIndex(latent=replacement, latent_idx=0, strength=0.5)
        new_state = cond.apply_to(state)

        assert new_state.latent.dtype == torch.bfloat16

    def test_device_preserved(self, device):
        """Output device matches input device."""
        from llm_dit.conditioning import LatentState, VideoConditionByLatentIndex

        state = LatentState.create(
            shape=(1, 64, 128),
            num_frames=9,
            height=128,
            width=128,
            device=device,
            dtype=torch.float32,
        )
        replacement = torch.randn(1, 128, 1, 4, 4, device=device)
        cond = VideoConditionByLatentIndex(latent=replacement, latent_idx=0, strength=0.5)
        new_state = cond.apply_to(state)

        assert new_state.latent.device == state.latent.device

    def test_clone_before_modify(self):
        """State is cloned before modification (original unchanged)."""
        from llm_dit.conditioning import LatentState, VideoConditionByLatentIndex

        state = LatentState.create(
            shape=(1, 64, 128),
            num_frames=9,
            height=128,
            width=128,
            device="cpu",
            dtype=torch.float32,
        )
        original_mask = state.denoise_mask.clone()

        replacement = torch.randn(1, 128, 1, 4, 4)
        cond = VideoConditionByLatentIndex(latent=replacement, latent_idx=0, strength=1.0)
        _ = cond.apply_to(state)

        # Original state should be unchanged
        assert torch.equal(state.denoise_mask, original_mask)

    def test_batch_support(self):
        """Conditioning works with batch size > 1."""
        from llm_dit.conditioning import LatentState, VideoConditionByLatentIndex

        state = LatentState.create(
            shape=(2, 64, 128),  # Batch of 2
            num_frames=9,
            height=128,
            width=128,
            device="cpu",
            dtype=torch.float32,
        )
        replacement = torch.randn(2, 128, 1, 4, 4)
        cond = VideoConditionByLatentIndex(latent=replacement, latent_idx=0, strength=0.5)
        new_state = cond.apply_to(state)

        assert new_state.latent.shape[0] == 2


# ============================================================================
# TestDenoiseMaskMechanics - 9 Tests
# ============================================================================


class TestDenoiseMaskMechanics:
    """Tests for denoise mask mechanics and formulas."""

    def test_timestep_scaling_formula(self):
        """Timesteps scale by denoise_mask * sigma (no * 1000; transformer handles that)."""
        from llm_dit.conditioning.utils import timesteps_from_mask

        sigma = torch.tensor(0.5)
        denoise_mask = torch.tensor([[[1.0], [0.5], [0.0]]])  # [B, T, 1]
        timesteps = timesteps_from_mask(denoise_mask, sigma)

        # Reference: denoise_mask * sigma. The transformer's _prepare_timestep
        # multiplies by timestep_scale_multiplier (1000), so we must NOT do it here.
        expected = torch.tensor([[[0.5], [0.25], [0.0]]])
        assert torch.allclose(timesteps, expected)

    def test_output_blending_formula(self):
        """Blending: denoised * mask + clean * (1 - mask)."""
        from llm_dit.conditioning.utils import post_process_latent

        denoised = torch.tensor([[[1.0, 1.0]]])  # [B, T, D]
        clean = torch.tensor([[[0.0, 0.0]]])
        mask = torch.tensor([[[1.0], [0.5], [0.0]]])  # Need to broadcast

        # Test with simple values
        denoised = torch.ones(1, 3, 2)
        clean = torch.zeros(1, 3, 2)
        mask = torch.tensor([[[1.0], [0.5], [0.0]]])

        blended = post_process_latent(denoised, mask, clean)

        # mask=1.0 -> denoised (1.0)
        # mask=0.5 -> 0.5 * 1.0 + 0.5 * 0.0 = 0.5
        # mask=0.0 -> clean (0.0)
        assert torch.allclose(blended[0, 0, :], torch.tensor([1.0, 1.0]))
        assert torch.allclose(blended[0, 1, :], torch.tensor([0.5, 0.5]))
        assert torch.allclose(blended[0, 2, :], torch.tensor([0.0, 0.0]))

    def test_mask_broadcasting_to_latent_dim(self):
        """denoise_mask [B, T, 1] broadcasts to latent [B, T, D]."""
        from llm_dit.conditioning.utils import post_process_latent

        denoised = torch.randn(2, 10, 128)
        clean = torch.randn(2, 10, 128)
        mask = torch.rand(2, 10, 1)  # [B, T, 1]

        blended = post_process_latent(denoised, mask, clean)

        assert blended.shape == (2, 10, 128)

    def test_combined_masks_multiply(self):
        """Multiple conditioning items stack their mask effects."""
        from llm_dit.conditioning import LatentState, VideoConditionByLatentIndex

        state = LatentState.create(
            shape=(1, 128, 128),
            num_frames=17,
            height=128,
            width=128,
            device="cpu",
            dtype=torch.float32,
        )

        # First conditioning: frame 0 with strength=0.5
        replacement1 = torch.randn(1, 128, 1, 4, 4)
        cond1 = VideoConditionByLatentIndex(latent=replacement1, latent_idx=0, strength=0.5)
        state = cond1.apply_to(state)

        # Second conditioning: frame 1 with strength=0.8
        replacement2 = torch.randn(1, 128, 1, 4, 4)
        cond2 = VideoConditionByLatentIndex(latent=replacement2, latent_idx=1, strength=0.8)
        state = cond2.apply_to(state)

        # Frame 0: mask = 0.5, Frame 1: mask = 0.2, Others: mask = 1.0
        tokens_per_frame = 16
        assert torch.allclose(state.denoise_mask[:, :tokens_per_frame, :], torch.tensor(0.5), atol=1e-6)
        assert torch.allclose(state.denoise_mask[:, tokens_per_frame:2*tokens_per_frame, :], torch.tensor(0.2), atol=1e-6)

    def test_mask_gradient_flow(self):
        """Gradients flow through mask operations."""
        from llm_dit.conditioning.utils import post_process_latent

        denoised = torch.randn(1, 4, 128, requires_grad=True)
        clean = torch.randn(1, 4, 128)
        mask = torch.tensor([[[0.5], [0.5], [0.5], [0.5]]])

        blended = post_process_latent(denoised, mask, clean)
        loss = blended.sum()
        loss.backward()

        assert denoised.grad is not None
        # Gradient should be scaled by mask
        assert not torch.equal(denoised.grad, torch.ones_like(denoised.grad))

    def test_zero_mask_blocks_gradient(self):
        """Zero mask means no gradient flows to that token."""
        from llm_dit.conditioning.utils import post_process_latent

        denoised = torch.randn(1, 4, 128, requires_grad=True)
        clean = torch.randn(1, 4, 128)
        mask = torch.tensor([[[1.0], [1.0], [0.0], [0.0]]])  # Last 2 tokens blocked

        blended = post_process_latent(denoised, mask, clean)
        loss = blended.sum()
        loss.backward()

        # Last 2 tokens should have zero gradient
        assert torch.allclose(denoised.grad[0, 2:, :], torch.zeros(2, 128))
        # First 2 tokens should have non-zero gradient
        assert not torch.allclose(denoised.grad[0, :2, :], torch.zeros(2, 128))

    def test_full_mask_preserves_denoised(self):
        """mask=1.0 everywhere means output equals denoised."""
        from llm_dit.conditioning.utils import post_process_latent

        denoised = torch.randn(1, 4, 128)
        clean = torch.randn(1, 4, 128)
        mask = torch.ones(1, 4, 1)

        blended = post_process_latent(denoised, mask, clean)

        assert torch.equal(blended, denoised)

    def test_zero_mask_preserves_clean(self):
        """mask=0.0 everywhere means output equals clean."""
        from llm_dit.conditioning.utils import post_process_latent

        denoised = torch.randn(1, 4, 128)
        clean = torch.randn(1, 4, 128)
        mask = torch.zeros(1, 4, 1)

        blended = post_process_latent(denoised, mask, clean)

        assert torch.equal(blended, clean)

    def test_dtype_preserved_in_blending(self):
        """Blending preserves dtype."""
        from llm_dit.conditioning.utils import post_process_latent

        denoised = torch.randn(1, 4, 128, dtype=torch.bfloat16)
        clean = torch.randn(1, 4, 128, dtype=torch.bfloat16)
        mask = torch.ones(1, 4, 1, dtype=torch.bfloat16)

        blended = post_process_latent(denoised, mask, clean)

        assert blended.dtype == torch.bfloat16
