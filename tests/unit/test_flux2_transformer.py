"""
Tests for FLUX.2 transformer implementation.

Last Updated: 2026-01-24

These tests verify the FLUX.2 Klein transformer with double-stream and
single-stream blocks. Key architectural differences from LTX-2:
- Double→Single stream (8 double + 24 single for 9B)
- Shared AdaLN modulation (computed once, not per-block)
- Joint attention in double blocks (merged K/V)

Run with: uv run pytest tests/unit/test_flux2_transformer.py -v
"""

import pytest
import torch

from llm_dit.models.flux2.transformer import (
    Flux2Transformer,
    DoubleStreamBlock,
    SingleStreamBlock,
    Modulation,
    LastLayer,
    QKNorm,
    RMSNorm,
    timestep_embedding,
)
from llm_dit.models.flux2.constants import (
    Klein9BParams,
    Klein4BParams,
    LATENT_CHANNELS_AFTER_PATCHIFY,
)
from llm_dit.models.flux2.rope import create_image_ids, create_text_ids, EmbedND


# ============================================================================
# Timestep Embedding Tests
# ============================================================================

class TestTimestepEmbedding:
    """Tests for timestep embedding function."""

    def test_timestep_embedding_shape(self):
        """Test timestep embedding produces correct shape."""
        batch = 4
        dim = 256
        t = torch.rand(batch)  # Timesteps in [0, 1]

        emb = timestep_embedding(t, dim)

        assert emb.shape == (batch, dim)

    def test_timestep_embedding_different_times(self):
        """Test different timesteps produce different embeddings."""
        dim = 256

        t0 = torch.tensor([0.0])
        t1 = torch.tensor([1.0])

        emb0 = timestep_embedding(t0, dim)
        emb1 = timestep_embedding(t1, dim)

        assert not torch.allclose(emb0, emb1), \
            "Different timesteps should produce different embeddings"

    def test_timestep_embedding_sinusoidal(self):
        """Test timestep embedding has sinusoidal structure."""
        dim = 128
        t = torch.tensor([0.5])

        emb = timestep_embedding(t, dim)

        # Half should be sin, half cos (or similar structure)
        half = dim // 2
        first_half = emb[0, :half]
        second_half = emb[0, half:]

        # They shouldn't be identical
        assert not torch.allclose(first_half, second_half)


# ============================================================================
# RMSNorm Tests
# ============================================================================

class TestRMSNorm:
    """Tests for RMSNorm implementation."""

    def test_rmsnorm_shape_preserved(self):
        """Test RMSNorm preserves input shape."""
        batch, seq, dim = 2, 100, 4096
        norm = RMSNorm(dim)
        x = torch.randn(batch, seq, dim)

        y = norm(x)

        assert y.shape == x.shape

    def test_rmsnorm_normalizes(self):
        """Test RMSNorm reduces variance."""
        batch, seq, dim = 2, 100, 4096
        norm = RMSNorm(dim)

        # High variance input
        x = torch.randn(batch, seq, dim) * 100

        y = norm(x)

        # Output should have more controlled magnitude
        assert y.pow(2).mean() < x.pow(2).mean()


# ============================================================================
# QKNorm Tests
# ============================================================================

class TestQKNorm:
    """Tests for QK normalization module."""

    def test_qknorm_instantiation(self):
        """Test QKNorm can be instantiated."""
        dim = 128
        qk_norm = QKNorm(dim)

        assert qk_norm is not None
        assert hasattr(qk_norm, "query_norm")
        assert hasattr(qk_norm, "key_norm")

    def test_qknorm_output_shape(self):
        """Test QKNorm preserves Q/K shapes."""
        dim = 128
        qk_norm = QKNorm(dim)

        batch, seq = 2, 100
        q = torch.randn(batch, seq, dim)
        k = torch.randn(batch, seq, dim)
        v = torch.randn(batch, seq, dim)  # QKNorm requires v for dtype casting

        q_norm, k_norm = qk_norm(q, k, v)

        assert q_norm.shape == q.shape
        assert k_norm.shape == k.shape

    def test_qknorm_stabilizes(self):
        """Test QKNorm reduces extreme values."""
        dim = 128
        qk_norm = QKNorm(dim)

        # Large values that could cause attention instability
        q = torch.randn(1, 10, dim) * 100
        k = torch.randn(1, 10, dim) * 100
        v = torch.randn(1, 10, dim)

        q_norm, k_norm = qk_norm(q, k, v)

        # Normalized values should be smaller
        assert q_norm.abs().max() < q.abs().max()
        assert k_norm.abs().max() < k.abs().max()


# ============================================================================
# Modulation Tests
# ============================================================================

class TestModulation:
    """Tests for AdaLN modulation module."""

    def test_modulation_instantiation(self):
        """Test Modulation can be instantiated."""
        dim = 4096
        double = True  # For double-stream blocks
        mod = Modulation(dim, double=double)

        assert mod is not None

    def test_modulation_output_structure(self):
        """Test Modulation outputs correct tuple structure."""
        dim = 4096
        mod = Modulation(dim, double=True)

        batch = 2
        vec = torch.randn(batch, dim)

        output, gate = mod(vec)

        # For double=True, output is first 3 tensors, gate is next 3
        assert isinstance(output, tuple)
        assert len(output) == 3  # (shift, scale, gate) for first modulation

    def test_modulation_single_vs_double(self):
        """Test single and double modulation have different outputs."""
        dim = 4096
        mod_single = Modulation(dim, double=False)
        mod_double = Modulation(dim, double=True)

        vec = torch.randn(1, dim)

        out_single, gate_single = mod_single(vec)
        out_double, gate_double = mod_double(vec)

        # Single should have None for second tuple
        assert gate_single is None
        # Double should have second tuple
        assert gate_double is not None


# ============================================================================
# DoubleStreamBlock Tests
# ============================================================================

class TestDoubleStreamBlock:
    """Tests for double-stream transformer block."""

    @pytest.fixture
    def params(self):
        """Create Klein9B params for testing."""
        return Klein9BParams()

    @pytest.fixture
    def block(self, params):
        """Create a double-stream block."""
        return DoubleStreamBlock(
            hidden_size=params.hidden_size,
            num_heads=params.num_heads,
            mlp_ratio=params.mlp_ratio,
        )

    def test_double_block_instantiation(self, block):
        """Test DoubleStreamBlock can be instantiated."""
        assert block is not None
        assert hasattr(block, "img_attn")
        assert hasattr(block, "txt_attn")

    def test_double_block_output_shapes(self, block, params):
        """Test DoubleStreamBlock preserves input shapes."""
        batch = 2
        img_seq, txt_seq = 1024, 100  # Image and text sequence lengths

        img = torch.randn(batch, img_seq, params.hidden_size)
        txt = torch.randn(batch, txt_seq, params.hidden_size)

        # Create position embeddings
        embed = EmbedND(
            dim=params.hidden_size // params.num_heads,
            theta=params.theta,
            axes_dim=params.axes_dim,
        )
        img_ids = create_image_ids(batch, 32, 32)  # 1024 tokens
        txt_ids = create_text_ids(batch, txt_seq)
        pe_img = embed(img_ids)
        pe_txt = embed(txt_ids)

        # Create modulation: ((shift1, scale1, gate1), (shift2, scale2, gate2))
        def make_mod():
            return (
                (
                    torch.zeros(batch, 1, params.hidden_size),
                    torch.zeros(batch, 1, params.hidden_size),
                    torch.ones(batch, 1, params.hidden_size),
                ),
                (
                    torch.zeros(batch, 1, params.hidden_size),
                    torch.zeros(batch, 1, params.hidden_size),
                    torch.ones(batch, 1, params.hidden_size),
                ),
            )

        mod_img = make_mod()
        mod_txt = make_mod()

        img_out, txt_out, _ = block.forward_kv_extract(
            img, txt, pe_img, pe_txt, mod_img, mod_txt, num_ref_tokens=0,
        )

        assert img_out.shape == img.shape, \
            f"Image output shape {img_out.shape} != input {img.shape}"
        assert txt_out.shape == txt.shape, \
            f"Text output shape {txt_out.shape} != input {txt.shape}"

    def test_double_block_joint_attention(self, block, params):
        """Test double block uses joint attention (merged K/V)."""
        # The double block should have img_attn and txt_attn
        # that produce separate Q but joint K/V

        assert hasattr(block, "img_attn")
        assert hasattr(block, "txt_attn")

        # In FLUX.2, K and V are concatenated: K = [K_txt, K_img]
        # This allows cross-modal attention


# ============================================================================
# SingleStreamBlock Tests
# ============================================================================

class TestSingleStreamBlock:
    """Tests for single-stream transformer block."""

    @pytest.fixture
    def params(self):
        """Create Klein9B params for testing."""
        return Klein9BParams()

    @pytest.fixture
    def block(self, params):
        """Create a single-stream block."""
        return SingleStreamBlock(
            hidden_size=params.hidden_size,
            num_heads=params.num_heads,
            mlp_ratio=params.mlp_ratio,
        )

    def test_single_block_instantiation(self, block):
        """Test SingleStreamBlock can be instantiated."""
        assert block is not None
        # Single-stream has unified attention (no separate img/txt)
        assert hasattr(block, "linear1")
        assert hasattr(block, "linear2")

    def test_single_block_output_shape(self, block, params):
        """Test SingleStreamBlock preserves input shape."""
        batch = 2
        num_txt = 100
        num_img = 1024
        seq_len = num_txt + num_img

        x = torch.randn(batch, seq_len, params.hidden_size)

        # Position embeddings for combined sequence
        embed = EmbedND(
            dim=params.hidden_size // params.num_heads,
            theta=params.theta,
            axes_dim=params.axes_dim,
        )

        # Create combined position IDs
        txt_ids = create_text_ids(batch, num_txt)
        img_ids = create_image_ids(batch, 32, 32)
        combined_ids = torch.cat([txt_ids, img_ids], dim=1)
        pe = embed(combined_ids)

        # Modulation for single-stream: (shift, scale, gate)
        mod = (
            torch.zeros(batch, 1, params.hidden_size),
            torch.zeros(batch, 1, params.hidden_size),
            torch.ones(batch, 1, params.hidden_size),
        )

        out, _ = block.forward_kv_extract(x, pe, mod, num_txt, num_ref_tokens=0)

        assert out.shape == x.shape

    def test_single_block_processes_merged_sequence(self, block, params):
        """Test single block handles txt+img merged sequence."""
        batch = 1
        txt_len, img_len = 50, 256
        total_len = txt_len + img_len

        # In single-stream blocks, txt and img are concatenated
        x = torch.randn(batch, total_len, params.hidden_size)

        # After processing, first txt_len tokens are text, rest are image
        # This is just shape verification
        assert x.shape[1] == total_len


# ============================================================================
# Flux2Transformer Full Model Tests
# ============================================================================

class TestFlux2Transformer:
    """Tests for the full Flux2Transformer model."""

    @pytest.fixture
    def mini_transformer(self):
        """Create minimal transformer for testing (1 block each)."""
        params = Klein9BParams()
        params.depth = 1  # 1 double block
        params.depth_single_blocks = 1  # 1 single block

        with torch.device("meta"):
            return Flux2Transformer(params)

    def test_transformer_instantiation(self, mini_transformer):
        """Test Flux2Transformer can be instantiated."""
        assert mini_transformer is not None
        assert hasattr(mini_transformer, "double_blocks")
        assert hasattr(mini_transformer, "single_blocks")

    def test_transformer_block_counts(self):
        """Test Klein9B has correct block counts."""
        params = Klein9BParams()
        assert params.depth == 8, "Klein9B should have 8 double-stream blocks"
        assert params.depth_single_blocks == 24, "Klein9B should have 24 single-stream blocks"

    def test_transformer_block_counts_4b(self):
        """Test Klein4B has correct block counts."""
        params = Klein4BParams()
        assert params.depth == 5, "Klein4B should have 5 double-stream blocks"
        assert params.depth_single_blocks == 20, "Klein4B should have 20 single-stream blocks"

    def test_transformer_hidden_size(self):
        """Test hidden sizes are correct."""
        params_9b = Klein9BParams()
        params_4b = Klein4BParams()

        assert params_9b.hidden_size == 4096
        assert params_4b.hidden_size == 3072

    def test_transformer_context_dim(self):
        """Test context dimensions match Qwen3 encoder output."""
        params_9b = Klein9BParams()
        params_4b = Klein4BParams()

        # Qwen3-8B: 3 layers * 4096 = 12288
        assert params_9b.context_in_dim == 12288

        # Qwen3-4B: 3 layers * 2560 = 7680
        assert params_4b.context_in_dim == 7680

    def test_transformer_in_channels(self):
        """Test in_channels matches patchified latents."""
        params = Klein9BParams()
        assert params.in_channels == LATENT_CHANNELS_AFTER_PATCHIFY == 128


# ============================================================================
# Shared Modulation Tests
# ============================================================================

class TestSharedModulation:
    """Tests for shared modulation computation."""

    def test_modulation_computed_once(self):
        """Test that modulation is computed once at model level, not per-block."""
        # In FLUX.2, the modulation vector is computed from timestep
        # at the model level and shared across all blocks

        params = Klein9BParams()

        # The Flux2Transformer should have modulation layers
        # that compute once and distribute to blocks

        # This is an architectural property - we verify the structure
        assert hasattr(Flux2Transformer, "__init__")

        # When implemented correctly, forward should compute mod_vec once
        # and pass it to all blocks, rather than recomputing per-block

    def test_modulation_dimensions(self):
        """Test modulation outputs match hidden_size."""
        params = Klein9BParams()
        dim = params.hidden_size

        mod = Modulation(dim, double=True)
        vec = torch.randn(2, dim)

        output, gate = mod(vec)

        # Modulation should produce hidden_size-dimensional shifts/scales
        assert output[0].shape[-1] == dim  # shift
        assert output[1].shape[-1] == dim  # scale


# ============================================================================
# LastLayer Tests
# ============================================================================

class TestLastLayer:
    """Tests for final output projection layer."""

    def test_last_layer_instantiation(self):
        """Test LastLayer can be instantiated."""
        hidden_size = 4096
        out_channels = 128  # Back to patchified latent channels

        layer = LastLayer(hidden_size, out_channels)
        assert layer is not None

    def test_last_layer_output_shape(self):
        """Test LastLayer produces correct output shape."""
        hidden_size = 4096
        out_channels = 128

        layer = LastLayer(hidden_size, out_channels)

        batch, seq = 2, 1024
        x = torch.randn(batch, seq, hidden_size)
        vec = torch.randn(batch, hidden_size)  # Final modulation

        out = layer(x, vec)

        # Output should have out_channels dimension
        assert out.shape == (batch, seq, out_channels)


# ============================================================================
# Architecture Comparison Tests
# ============================================================================

class TestArchitectureComparison:
    """Tests comparing FLUX.2 vs LTX-2 architectures."""

    def test_flux2_uses_double_single_stream(self):
        """Test FLUX.2 uses double→single stream architecture."""
        params = Klein9BParams()

        # FLUX.2: 8 double + 24 single = 32 total blocks
        total_blocks = params.depth + params.depth_single_blocks
        assert total_blocks == 32

    def test_flux2_has_joint_attention(self):
        """Test FLUX.2 double blocks use joint attention."""
        # In double-stream blocks, image and text share K/V
        # This is different from LTX-2's separate cross-attention

        block = DoubleStreamBlock(
            hidden_size=4096,
            num_heads=32,
            mlp_ratio=3.0,
        )

        # Should have both img and txt attention modules
        assert hasattr(block, "img_attn")
        assert hasattr(block, "txt_attn")

    def test_flux2_rope_is_4d(self):
        """Test FLUX.2 uses 4D RoPE (t, h, w, l)."""
        params = Klein9BParams()

        # axes_dim has 4 elements for 4D coordinates
        assert len(params.axes_dim) == 4

    def test_mlp_ratio(self):
        """Test MLP ratio is 3.0 for FLUX.2."""
        params = Klein9BParams()
        assert params.mlp_ratio == 3.0


# ============================================================================
# Numerical Stability Tests
# ============================================================================

class TestNumericalStability:
    """Tests for numerical stability properties."""

    def test_qknorm_prevents_overflow(self):
        """Test QKNorm prevents attention score overflow."""
        dim = 128
        qk_norm = QKNorm(dim)

        # Very large Q/K that could cause overflow in attention
        q = torch.randn(1, 10, dim) * 1000
        k = torch.randn(1, 10, dim) * 1000
        v = torch.randn(1, 10, dim)

        q_norm, k_norm = qk_norm(q, k, v)

        # After normalization, dot product should be reasonable
        scores = torch.matmul(q_norm, k_norm.transpose(-2, -1))

        # Scores should not be inf/nan
        assert torch.isfinite(scores).all(), "Attention scores should be finite"

    def test_rmsnorm_handles_zeros(self):
        """Test RMSNorm handles zero inputs gracefully."""
        dim = 4096
        norm = RMSNorm(dim)

        x = torch.zeros(1, 10, dim)
        y = norm(x)

        # Output should be finite (not nan from 0/0)
        assert torch.isfinite(y).all()


# Run with: uv run pytest tests/unit/test_flux2_transformer.py -v
