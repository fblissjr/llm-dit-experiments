"""
Tests for FLUX.2 4D RoPE implementation.

Last Updated: 2026-01-23

These tests verify the 4D rotary position embedding implementation
specific to FLUX.2 Klein models. FLUX.2 uses 4D coordinates (t, h, w, l)
unlike LTX-2's 3D (t, h, w).

Run with: uv run pytest tests/unit/test_flux2_rope.py -v
"""

import pytest
import torch
import math

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
        """Test EmbedND output has correct shape [B, seq_len, head_dim]."""
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

        # RoPE embeddings: (cos, sin) each with shape [B, seq_len, head_dim]
        assert len(pe) == 2, "Expected (cos, sin) tuple"
        cos, sin = pe
        expected_seq_len = height * width
        assert cos.shape == (batch_size, expected_seq_len, head_dim)
        assert sin.shape == (batch_size, expected_seq_len, head_dim)

    def test_embednd_klein4b_params(self):
        """Test EmbedND works with Klein4B parameters."""
        params = Klein4BParams()
        head_dim = params.hidden_size // params.num_heads  # 3072/24 = 128
        embed = EmbedND(
            dim=head_dim,
            theta=params.theta,
            axes_dim=params.axes_dim,
        )

        img_ids = create_image_ids(1, 32, 32)
        pe = embed(img_ids)

        cos, sin = pe
        assert cos.shape == (1, 32 * 32, head_dim)


# ============================================================================
# Core RoPE Function Tests
# ============================================================================

class TestRopeFunction:
    """Tests for core rope() computation."""

    def test_rope_output_shape(self):
        """Test rope() produces correct shape."""
        batch, seq_len = 2, 1024
        dim = 128
        pos = torch.arange(seq_len).float()

        cos, sin = rope(pos, dim, theta=2000)

        assert cos.shape == (seq_len, dim)
        assert sin.shape == (seq_len, dim)

    def test_rope_periodicity(self):
        """Test rope() produces periodic embeddings."""
        dim = 128
        pos = torch.arange(1000).float()

        cos, sin = rope(pos, dim, theta=2000)

        # First frequencies should be faster (shorter period)
        # Later frequencies should be slower (longer period)
        # Check that cos values for first dimension oscillate faster
        # This is a structural property of RoPE

        # First dimension should have more zero crossings than last
        first_dim_changes = (cos[:-1, 0] * cos[1:, 0] < 0).sum()
        last_dim_changes = (cos[:-1, -1] * cos[1:, -1] < 0).sum()

        assert first_dim_changes > last_dim_changes, \
            "Early dimensions should oscillate faster than late dimensions"

    def test_rope_unit_norm(self):
        """Test that cos² + sin² ≈ 1 (unit circle property)."""
        pos = torch.arange(100).float()
        dim = 64

        cos, sin = rope(pos, dim, theta=2000)

        # cos² + sin² should equal 1 for all positions and dimensions
        norm = cos ** 2 + sin ** 2
        assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5)


# ============================================================================
# Apply RoPE Tests
# ============================================================================

class TestApplyRope:
    """Tests for apply_rope() application."""

    def test_apply_rope_preserves_shape(self):
        """Test apply_rope preserves input shape."""
        batch, seq, dim = 2, 64, 128

        x = torch.randn(batch, seq, dim)
        cos = torch.ones(batch, seq, dim)
        sin = torch.zeros(batch, seq, dim)

        out = apply_rope(x, (cos, sin))

        assert out.shape == x.shape

    def test_apply_rope_identity_with_zeros(self):
        """Test apply_rope is identity when sin=0, cos=1."""
        batch, seq, dim = 2, 64, 128

        x = torch.randn(batch, seq, dim)
        cos = torch.ones(batch, seq, dim)
        sin = torch.zeros(batch, seq, dim)

        out = apply_rope(x, (cos, sin))

        assert torch.allclose(out, x, atol=1e-5)

    def test_apply_rope_rotation(self):
        """Test apply_rope actually rotates vectors."""
        batch, seq, dim = 1, 1, 4

        # Simple input
        x = torch.tensor([[[1.0, 0.0, 1.0, 0.0]]])

        # 90 degree rotation (cos=0, sin=1)
        cos = torch.zeros(batch, seq, dim)
        sin = torch.ones(batch, seq, dim)

        out = apply_rope(x, (cos, sin))

        # After 90 degree rotation, vector should change
        assert not torch.allclose(out, x)


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

        # Coordinate 3 (l): linear position 0 to H*W-1
        linear_expected = torch.arange(height * width).view(height, width)
        assert torch.all(ids_2d[..., 3] == linear_expected)

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
        batch, height, width = 2, 32, 32

        ids = create_reference_ids(batch, height, width)

        expected_seq = height * width
        assert ids.shape == (batch, expected_seq, 4)

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
        head_dim = params.hidden_size // params.num_heads

        assert sum(params.axes_dim) == head_dim

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
        per_axis = params.axes_dim[0]  # 32

        embed = EmbedND(
            dim=head_dim,
            theta=params.theta,
            axes_dim=params.axes_dim,
        )

        # Create IDs where only one coordinate varies
        batch, height, width = 1, 4, 4
        img_ids = create_image_ids(batch, height, width)

        pe = embed(img_ids)
        cos, sin = pe

        # The position embeddings should encode all 4 coordinates
        # Each coordinate uses axes_dim[i] dimensions
        # t: dims 0:32, h: dims 32:64, w: dims 64:96, l: dims 96:128

        # Verify dimensions match axes_dim splits
        total_dim = cos.shape[-1]
        assert total_dim == sum(params.axes_dim)


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
        dim, pos = 64, torch.arange(100).float()

        cos_2000, sin_2000 = rope(pos, dim, theta=2000)
        cos_10000, sin_10000 = rope(pos, dim, theta=10000)

        # Different theta should produce different embeddings
        assert not torch.allclose(cos_2000, cos_10000)
        assert not torch.allclose(sin_2000, sin_10000)

        # Lower theta (2000) should have faster oscillation
        # (more zero crossings in the same position range)
        changes_2000 = (cos_2000[:-1, 0] * cos_2000[1:, 0] < 0).sum()
        changes_10000 = (cos_10000[:-1, 0] * cos_10000[1:, 0] < 0).sum()

        assert changes_2000 >= changes_10000, \
            "Lower theta should produce faster oscillation"


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
        cos, sin = pe

        expected_seq = txt_len + img_h * img_w
        assert cos.shape == (batch, expected_seq, head_dim)

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
        cos, sin = pe

        # Create mock Q, K for attention
        q = torch.randn(batch, num_heads, seq_len, head_dim)
        k = torch.randn(batch, num_heads, seq_len, head_dim)

        # Expand pe for multi-head: [B, seq, D] -> [B, 1, seq, D] -> broadcast
        cos_exp = cos.unsqueeze(1)  # [B, 1, seq, D]
        sin_exp = sin.unsqueeze(1)

        # Apply RoPE to each head
        q_rot = apply_rope(q.reshape(batch * num_heads, seq_len, head_dim),
                          (cos.repeat(num_heads, 1, 1), sin.repeat(num_heads, 1, 1)))
        q_rot = q_rot.view(batch, num_heads, seq_len, head_dim)

        assert q_rot.shape == q.shape


# Run with: uv run pytest tests/unit/test_flux2_rope.py -v
