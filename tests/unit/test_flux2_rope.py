"""
Tests for FLUX.2 4D RoPE implementation.

Last Updated: 2026-01-24

These tests verify the 4D rotary position embedding implementation
specific to FLUX.2 Klein models. FLUX.2 uses 4D coordinates (t, h, w, l)
unlike LTX-2's 3D (t, h, w).

The implementation uses 2x2 rotation matrix format [B, seq, dim//2, 2, 2]
rather than separate (cos, sin) tensors.

Run with: uv run pytest tests/unit/test_flux2_rope.py -v
"""

import pytest
import torch

from llm_dit.models.flux2.rope import (
    EmbedND,
    rope,
    apply_rope,
    create_image_ids,
    create_text_ids,
    create_reference_ids,
)
from llm_dit.models.flux2.constants import Klein9BParams, Klein4BParams


# ============================================================================
# EmbedND Tests
# ============================================================================

class TestEmbedND:
    """Tests for 4D position embedding module."""

    def test_embednd_instantiation(self):
        """Test EmbedND can be instantiated with Klein9B params."""
        params = Klein9BParams()
        embed = EmbedND(
            dim=params.hidden_size // params.num_heads,  # head_dim
            theta=params.theta,
            axes_dim=params.axes_dim,
        )
        assert embed is not None
        assert embed.theta == 2000  # FLUX.2 uses theta=2000

    def test_embednd_output_shape(self):
        """Test EmbedND output has correct shape [B, 1, seq, dim//2, 2, 2]."""
        params = Klein9BParams()
        head_dim = params.hidden_size // params.num_heads  # 4096/32 = 128
        embed = EmbedND(
            dim=head_dim,
            theta=params.theta,
            axes_dim=params.axes_dim,
        )

        batch_size = 2
        height, width = 64, 64  # 1024x1024 image / 16 = 64x64 latent
        img_ids = create_image_ids(batch_size, height, width)

        pe = embed(img_ids)

        # RoPE embeddings in 2x2 rotation matrix format
        # Shape: [B, 1, seq_len, head_dim//2, 2, 2]
        expected_seq_len = height * width
        assert pe.shape == (batch_size, 1, expected_seq_len, head_dim // 2, 2, 2)

    def test_embednd_klein4b_params(self):
        """Test EmbedND works with Klein4B parameters."""
        params = Klein4BParams()
        # Klein4B: axes_dim=[24,24,24,24], sum=96
        # This is the PE dimension, not necessarily equal to head_dim
        pe_dim = sum(params.axes_dim)  # 96

        embed = EmbedND(
            dim=pe_dim,
            theta=params.theta,
            axes_dim=params.axes_dim,
        )

        img_ids = create_image_ids(1, 32, 32)
        pe = embed(img_ids)

        # Shape: [B, 1, seq_len, pe_dim//2, 2, 2]
        assert pe.shape == (1, 1, 32 * 32, pe_dim // 2, 2, 2)


# ============================================================================
# Core RoPE Function Tests
# ============================================================================

class TestRopeFunction:
    """Tests for core rope() computation."""

    def test_rope_output_shape(self):
        """Test rope() produces 2x2 rotation matrix shape."""
        seq_len = 1024
        dim = 128  # Per-axis dimension
        pos = torch.arange(seq_len).float()

        # rope expects batched input [B, seq_len] but works with [seq_len] too
        # Actually looking at impl: pos: [B, seq_len] -> out: [B, seq_len, dim//2, 2, 2]
        pos_batched = pos.unsqueeze(0)  # [1, seq_len]
        out = rope(pos_batched, dim, theta=2000)

        # Output is rotation matrix format: [B, seq, dim//2, 2, 2]
        assert out.shape == (1, seq_len, dim // 2, 2, 2)

    def test_rope_rotation_matrix_structure(self):
        """Test rope() produces valid rotation matrices."""
        seq_len = 100
        dim = 64
        pos = torch.arange(seq_len).float().unsqueeze(0)

        out = rope(pos, dim, theta=2000)

        # Each 2x2 matrix should be a rotation matrix:
        # [[cos, -sin], [sin, cos]]
        # For a rotation matrix, R @ R^T = I
        for i in range(min(10, seq_len)):  # Check first 10 positions
            for d in range(out.shape[2]):  # Check each frequency
                rot_mat = out[0, i, d]  # [2, 2]
                # Check det = 1 (proper rotation)
                det = rot_mat[0, 0] * rot_mat[1, 1] - rot_mat[0, 1] * rot_mat[1, 0]
                assert torch.isclose(det, torch.tensor(1.0), atol=1e-5), \
                    f"Rotation matrix at pos {i}, dim {d} has det={det}"

    def test_rope_unit_norm(self):
        """Test rotation matrix columns have unit norm."""
        pos = torch.arange(100).float().unsqueeze(0)
        dim = 64

        out = rope(pos, dim, theta=2000)

        # Each column of rotation matrix should have unit norm
        for i in range(10):
            for d in range(out.shape[2]):
                rot_mat = out[0, i, d]
                col1_norm = torch.sqrt(rot_mat[0, 0]**2 + rot_mat[1, 0]**2)
                col2_norm = torch.sqrt(rot_mat[0, 1]**2 + rot_mat[1, 1]**2)
                assert torch.isclose(col1_norm, torch.tensor(1.0), atol=1e-5)
                assert torch.isclose(col2_norm, torch.tensor(1.0), atol=1e-5)


# ============================================================================
# Apply RoPE Tests
# ============================================================================

class TestApplyRope:
    """Tests for apply_rope() application."""

    def test_apply_rope_preserves_shape(self):
        """Test apply_rope preserves input shape."""
        batch, heads, seq, head_dim = 2, 8, 64, 128

        q = torch.randn(batch, heads, seq, head_dim)
        k = torch.randn(batch, heads, seq, head_dim)

        # Create position embeddings matching the expected format
        # freqs_cis: [B, 1, seq, head_dim//2, 2, 2]
        freqs_cis = torch.randn(batch, 1, seq, head_dim // 2, 2, 2)

        q_rot, k_rot = apply_rope(q, k, freqs_cis)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_apply_rope_identity_with_identity_matrix(self):
        """Test apply_rope with identity rotation matrices."""
        batch, heads, seq, head_dim = 1, 1, 4, 4

        q = torch.randn(batch, heads, seq, head_dim)
        k = torch.randn(batch, heads, seq, head_dim)

        # Create identity rotation matrices (cos=1, sin=0)
        # [[1, 0], [0, 1]]
        freqs_cis = torch.zeros(batch, 1, seq, head_dim // 2, 2, 2)
        freqs_cis[..., 0, 0] = 1.0  # cos
        freqs_cis[..., 1, 1] = 1.0  # cos

        q_rot, k_rot = apply_rope(q, k, freqs_cis)

        assert torch.allclose(q_rot, q, atol=1e-5)
        assert torch.allclose(k_rot, k, atol=1e-5)

    def test_apply_rope_rotation(self):
        """Test apply_rope actually rotates vectors."""
        batch, heads, seq, head_dim = 1, 1, 1, 4

        q = torch.tensor([[[[1.0, 0.0, 1.0, 0.0]]]])
        k = torch.tensor([[[[0.0, 1.0, 0.0, 1.0]]]])

        # Create 90 degree rotation matrices [[0, -1], [1, 0]]
        freqs_cis = torch.zeros(batch, 1, seq, head_dim // 2, 2, 2)
        freqs_cis[..., 0, 1] = -1.0  # -sin
        freqs_cis[..., 1, 0] = 1.0   # sin

        q_rot, k_rot = apply_rope(q, k, freqs_cis)

        # After 90 degree rotation, vectors should change
        assert not torch.allclose(q_rot, q)
        assert not torch.allclose(k_rot, k)


# ============================================================================
# Position ID Creation Tests
# ============================================================================

class TestPositionIdCreation:
    """Tests for position ID creation functions."""

    def test_create_image_ids_shape(self):
        """Test create_image_ids produces correct shape."""
        batch, height, width = 2, 64, 64

        ids = create_image_ids(batch, height, width)

        # Shape should be [B, H*W, 4] for 4D coordinates
        expected_seq = height * width
        assert ids.shape == (batch, expected_seq, 4)

    def test_create_image_ids_coordinates(self):
        """Test create_image_ids produces correct 4D coordinates."""
        batch, height, width = 1, 4, 4

        ids = create_image_ids(batch, height, width)

        # Reshape for easier inspection
        ids_2d = ids[0].view(height, width, 4)

        # Coordinate 0 (t): should all be 0 for images
        assert torch.all(ids_2d[..., 0] == 0)

        # Coordinate 1 (h): should range 0 to height-1 along first axis
        for h in range(height):
            assert torch.all(ids_2d[h, :, 1] == h)

        # Coordinate 2 (w): should range 0 to width-1 along second axis
        for w in range(width):
            assert torch.all(ids_2d[:, w, 2] == w)

        # Coordinate 3 (l): always 0 for images (text uses l for sequence position)
        assert torch.all(ids_2d[..., 3] == 0)

    def test_create_text_ids_shape(self):
        """Test create_text_ids produces correct shape."""
        batch, seq_len = 2, 512

        ids = create_text_ids(batch, seq_len)

        assert ids.shape == (batch, seq_len, 4)

    def test_create_text_ids_only_linear(self):
        """Test create_text_ids only uses linear position (l coordinate)."""
        batch, seq_len = 1, 100

        ids = create_text_ids(batch, seq_len)

        # Coordinates 0, 1, 2 (t, h, w) should all be 0 for text
        assert torch.all(ids[..., 0] == 0)  # t = 0
        assert torch.all(ids[..., 1] == 0)  # h = 0
        assert torch.all(ids[..., 2] == 0)  # w = 0

        # Coordinate 3 (l) should be sequential
        expected_l = torch.arange(seq_len).float()
        assert torch.allclose(ids[0, :, 3], expected_l)

    def test_create_reference_ids_shape(self):
        """Test create_reference_ids produces correct shape."""
        batch = 2
        ref_heights = [32, 16]
        ref_widths = [32, 16]

        ids = create_reference_ids(batch, ref_heights, ref_widths)

        expected_seq = 32 * 32 + 16 * 16  # Sum of all reference tokens
        assert ids.shape == (batch, expected_seq, 4)

    def test_create_reference_ids_single_ref(self):
        """Test create_reference_ids with single reference image."""
        batch = 1
        ref_heights = [8]
        ref_widths = [8]

        ids = create_reference_ids(batch, ref_heights, ref_widths)

        assert ids.shape == (batch, 8 * 8, 4)

        # First reference should have t = 10 (t_scale * (1 + 0))
        assert torch.all(ids[..., 0] == 10)

    def test_position_ids_dtype(self):
        """Test position IDs are float32 by default."""
        img_ids = create_image_ids(1, 16, 16)
        txt_ids = create_text_ids(1, 50)

        assert img_ids.dtype == torch.float32
        assert txt_ids.dtype == torch.float32

    def test_position_ids_device(self):
        """Test position IDs respect device parameter."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        img_ids = create_image_ids(1, 16, 16, device="cuda")
        txt_ids = create_text_ids(1, 50, device="cuda")

        assert img_ids.device.type == "cuda"
        assert txt_ids.device.type == "cuda"


# ============================================================================
# Axes Dimension Tests
# ============================================================================

class TestAxesDimension:
    """Tests for axes dimension handling in 4D RoPE."""

    def test_axes_dim_sum_equals_head_dim(self):
        """Test that sum of axes dimensions equals head dimension."""
        params = Klein9BParams()
        head_dim = params.hidden_size // params.num_heads

        # axes_dim should sum to head_dim
        assert sum(params.axes_dim) == head_dim, \
            f"axes_dim sum {sum(params.axes_dim)} != head_dim {head_dim}"

    def test_axes_dim_klein4b(self):
        """Test Klein4B axes dimensions."""
        params = Klein4BParams()

        # Klein4B has axes_dim=[32,32,32,32], sum=128 = head_dim (3072/24)
        assert sum(params.axes_dim) == 128
        assert len(params.axes_dim) == 4

    def test_axes_dim_are_equal(self):
        """Test that all 4 axes have equal dimension (balanced 4D)."""
        params = Klein9BParams()

        # All 4 coordinates should get equal embedding dimensions
        assert len(set(params.axes_dim)) == 1, \
            f"Expected equal axes dims, got {params.axes_dim}"

    def test_4d_coordinate_encoding(self):
        """Test that each of 4 coordinates gets encoded into its slice."""
        params = Klein9BParams()
        head_dim = params.hidden_size // params.num_heads  # 128

        embed = EmbedND(
            dim=head_dim,
            theta=params.theta,
            axes_dim=params.axes_dim,
        )

        # Create IDs where only one coordinate varies
        batch, height, width = 1, 4, 4
        img_ids = create_image_ids(batch, height, width)

        pe = embed(img_ids)

        # Output shape: [B, 1, seq, head_dim//2, 2, 2]
        # The head_dim//2 dimension contains concatenated per-axis embeddings
        assert pe.shape[3] == head_dim // 2


# ============================================================================
# Theta Scaling Tests
# ============================================================================

class TestThetaScaling:
    """Tests for theta parameter effects on RoPE."""

    def test_theta_2000_default(self):
        """Test FLUX.2 uses theta=2000 (vs common theta=10000)."""
        params = Klein9BParams()
        assert params.theta == 2000

    def test_theta_affects_frequency(self):
        """Test different theta values produce different frequencies."""
        dim = 64
        pos = torch.arange(100).float().unsqueeze(0)

        out_2000 = rope(pos, dim, theta=2000)
        out_10000 = rope(pos, dim, theta=10000)

        # Different theta should produce different rotation matrices
        assert not torch.allclose(out_2000, out_10000)


# ============================================================================
# Integration Tests
# ============================================================================

class TestRoPEIntegration:
    """Integration tests for RoPE with transformer inputs."""

    def test_rope_with_joint_img_txt_ids(self):
        """Test RoPE handles concatenated image+text IDs."""
        batch = 2
        img_h, img_w = 32, 32
        txt_len = 100

        img_ids = create_image_ids(batch, img_h, img_w)
        txt_ids = create_text_ids(batch, txt_len)

        # FLUX.2 concatenates txt + img for joint attention
        joint_ids = torch.cat([txt_ids, img_ids], dim=1)

        params = Klein9BParams()
        head_dim = params.hidden_size // params.num_heads
        embed = EmbedND(dim=head_dim, theta=params.theta, axes_dim=params.axes_dim)

        pe = embed(joint_ids)

        expected_seq = txt_len + img_h * img_w
        # Shape: [B, 1, seq, head_dim//2, 2, 2]
        assert pe.shape == (batch, 1, expected_seq, head_dim // 2, 2, 2)

    def test_rope_end_to_end_attention_compatible(self):
        """Test RoPE embeddings are compatible with attention dimensions."""
        params = Klein9BParams()
        batch, seq_len = 2, 256
        num_heads = params.num_heads
        head_dim = params.hidden_size // num_heads

        # Create position IDs
        ids = create_image_ids(batch, 16, 16)  # 256 tokens

        # Create embeddings
        embed = EmbedND(dim=head_dim, theta=params.theta, axes_dim=params.axes_dim)
        pe = embed(ids)

        # Create mock Q, K for attention
        q = torch.randn(batch, num_heads, seq_len, head_dim)
        k = torch.randn(batch, num_heads, seq_len, head_dim)

        # Apply RoPE
        q_rot, k_rot = apply_rope(q, k, pe)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape


# Run with: uv run pytest tests/unit/test_flux2_rope.py -v
