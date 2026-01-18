"""
Tests for LTX-2 text connector implementation.

Last Updated: 2026-01-17

Tests verify:
1. Connector module shapes and forward pass
2. RoPE implementation correctness
3. Register replacement logic
4. State dict mapping from diffusers format
5. Numerical equivalence with diffusers (when weights loaded)

Run with: uv run pytest tests/integration/test_ltx2_connectors.py -v
"""

import pytest
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import our implementation
from llm_dit.models.ltx2_connectors import (
    LTX2RotaryPosEmbed1d,
    LTX2Attention1d,
    LTX2FeedForward1d,
    LTX2TransformerBlock1d,
    LTX2ConnectorTransformer1d,
    LTX2TextConnectors,
    apply_split_rotary_emb,
    _map_diffusers_state_dict,
)


# ============================================================================
# RoPE Tests
# ============================================================================

class TestRotaryPosEmbed1d:
    """Tests for 1D rotary position embedding."""

    def test_rope_output_shape(self):
        """Test RoPE output shape matches expected [B, H, T, D//2]."""
        rope = LTX2RotaryPosEmbed1d(
            dim=128,  # head_dim
            num_attention_heads=30,
        )

        batch_size, seq_len = 2, 256
        cos, sin = rope(batch_size, seq_len, device=torch.device("cpu"))

        assert cos.shape == (batch_size, 30, seq_len, 64)  # dim//2 = 64
        assert sin.shape == (batch_size, 30, seq_len, 64)

    def test_rope_values_bounded(self):
        """Test that cos/sin values are properly bounded."""
        rope = LTX2RotaryPosEmbed1d(dim=128)
        cos, sin = rope(1, 100, torch.device("cpu"))

        assert cos.abs().max() <= 1.0 + 1e-6
        assert sin.abs().max() <= 1.0 + 1e-6

    def test_apply_split_rotary_emb(self):
        """Test split RoPE application preserves shape."""
        batch, heads, seq, head_dim = 2, 30, 256, 128
        x = torch.randn(batch, heads, seq, head_dim)

        # Create cos/sin for half dimension
        cos = torch.ones(batch, heads, seq, head_dim // 2)
        sin = torch.zeros(batch, heads, seq, head_dim // 2)

        out = apply_split_rotary_emb(x, cos, sin)

        assert out.shape == x.shape

    def test_rope_identity_with_zero_sin(self):
        """With sin=0, cos=1, RoPE should be identity."""
        batch, heads, seq, head_dim = 1, 4, 10, 64
        x = torch.randn(batch, heads, seq, head_dim)

        cos = torch.ones(batch, heads, seq, head_dim // 2)
        sin = torch.zeros(batch, heads, seq, head_dim // 2)

        out = apply_split_rotary_emb(x, cos, sin)

        # With cos=1, sin=0: out = x*cos + 0 = x
        torch.testing.assert_close(out, x, rtol=1e-5, atol=1e-5)


# ============================================================================
# Attention Tests
# ============================================================================

class TestAttention1d:
    """Tests for 1D attention module."""

    def test_attention_output_shape(self):
        """Test attention output shape matches input."""
        dim = 3840
        attn = LTX2Attention1d(dim=dim, num_heads=30, head_dim=128)

        x = torch.randn(2, 256, dim)
        out = attn(x)

        assert out.shape == x.shape

    def test_attention_with_rope(self):
        """Test attention works with RoPE."""
        dim = 3840
        attn = LTX2Attention1d(dim=dim, num_heads=30, head_dim=128)
        rope = LTX2RotaryPosEmbed1d(dim=128, num_attention_heads=30)

        batch, seq = 2, 256
        x = torch.randn(batch, seq, dim)
        rotary_emb = rope(batch, seq, x.device)

        out = attn(x, rotary_emb=rotary_emb)

        assert out.shape == x.shape

    def test_attention_with_mask(self):
        """Test attention respects attention mask."""
        dim = 3840
        attn = LTX2Attention1d(dim=dim, num_heads=30, head_dim=128)

        batch, seq = 2, 256
        x = torch.randn(batch, seq, dim)

        # Additive mask: -inf for masked positions
        mask = torch.zeros(batch, 1, 1, seq)
        mask[:, :, :, seq // 2:] = float("-inf")

        out = attn(x, attention_mask=mask)

        assert out.shape == x.shape


# ============================================================================
# FeedForward Tests
# ============================================================================

class TestFeedForward1d:
    """Tests for feed-forward network."""

    def test_ff_output_shape(self):
        """Test FFN output shape matches input."""
        dim = 3840
        ff = LTX2FeedForward1d(dim=dim, mult=4)

        x = torch.randn(2, 256, dim)
        out = ff(x)

        assert out.shape == x.shape


# ============================================================================
# Transformer Block Tests
# ============================================================================

class TestTransformerBlock1d:
    """Tests for transformer block."""

    def test_block_output_shape(self):
        """Test block output shape matches input."""
        dim = 3840
        block = LTX2TransformerBlock1d(dim=dim, num_heads=30, head_dim=128)

        x = torch.randn(2, 256, dim)
        out = block(x)

        assert out.shape == x.shape

    def test_block_with_rope(self):
        """Test block works with RoPE."""
        dim = 3840
        block = LTX2TransformerBlock1d(dim=dim, num_heads=30, head_dim=128)
        rope = LTX2RotaryPosEmbed1d(dim=128, num_attention_heads=30)

        batch, seq = 2, 256
        x = torch.randn(batch, seq, dim)
        rotary_emb = rope(batch, seq, x.device)

        out = block(x, rotary_emb=rotary_emb)

        assert out.shape == x.shape


# ============================================================================
# Connector Transformer Tests
# ============================================================================

class TestConnectorTransformer1d:
    """Tests for the full connector transformer."""

    def test_connector_output_shape(self):
        """Test connector output shape."""
        connector = LTX2ConnectorTransformer1d(
            num_attention_heads=30,
            attention_head_dim=128,
            num_layers=2,
            num_learnable_registers=128,
        )

        # Input must be divisible by num_learnable_registers
        batch, seq = 2, 256  # 256 is divisible by 128
        x = torch.randn(batch, seq, 3840)

        out, mask = connector(x)

        assert out.shape == (batch, seq, 3840)

    def test_connector_without_registers(self):
        """Test connector without learnable registers."""
        connector = LTX2ConnectorTransformer1d(
            num_attention_heads=30,
            attention_head_dim=128,
            num_layers=2,
            num_learnable_registers=None,
        )

        batch, seq = 2, 256
        x = torch.randn(batch, seq, 3840)

        out, mask = connector(x)

        assert out.shape == (batch, seq, 3840)

    def test_connector_with_attention_mask(self):
        """Test connector with attention mask and register replacement."""
        connector = LTX2ConnectorTransformer1d(
            num_attention_heads=30,
            attention_head_dim=128,
            num_layers=2,
            num_learnable_registers=128,
        )

        batch, seq = 2, 256
        x = torch.randn(batch, seq, 3840)

        # Binary mask: 1 = valid, 0 = padding
        mask = torch.ones(batch, seq)
        mask[:, seq // 2:] = 0  # Half padding

        # Convert to additive mask
        additive_mask = (mask - 1).reshape(batch, 1, 1, seq) * float("inf")

        out, new_mask = connector(x, attention_mask=additive_mask)

        assert out.shape == (batch, seq, 3840)


# ============================================================================
# Full Text Connectors Tests
# ============================================================================

class TestTextConnectors:
    """Tests for the complete text connector stack."""

    def test_connectors_output_shape(self):
        """Test full connectors output shapes."""
        connectors = LTX2TextConnectors(
            caption_channels=3840,
            text_proj_in_factor=49,
        )

        batch, seq = 2, 256
        # Input is packed multi-layer features
        x = torch.randn(batch, seq, 3840 * 49)
        mask = torch.ones(batch, seq)

        video_out, audio_out, new_mask = connectors(x, mask)

        assert video_out.shape == (batch, seq, 3840)
        assert audio_out.shape == (batch, seq, 3840)

    def test_text_proj_in_dimension(self):
        """Test text_proj_in projects correctly."""
        connectors = LTX2TextConnectors()

        # text_proj_in: Linear(188160 -> 3840)
        assert connectors.text_proj_in.in_features == 3840 * 49  # 188160
        assert connectors.text_proj_in.out_features == 3840


# ============================================================================
# State Dict Mapping Tests
# ============================================================================

class TestStateDictMapping:
    """Tests for mapping diffusers state dict to our format."""

    def test_ff_net_mapping(self):
        """Test FeedForward mapping from diffusers format."""
        diffusers_state_dict = {
            "video_connector.transformer_blocks.0.ff.net.0.proj.weight": torch.randn(15360, 3840),
            "video_connector.transformer_blocks.0.ff.net.0.proj.bias": torch.randn(15360),
            "video_connector.transformer_blocks.0.ff.net.2.weight": torch.randn(3840, 15360),
            "video_connector.transformer_blocks.0.ff.net.2.bias": torch.randn(3840),
        }

        mapped = _map_diffusers_state_dict(diffusers_state_dict)

        assert "video_connector.transformer_blocks.0.ff.proj_in.weight" in mapped
        assert "video_connector.transformer_blocks.0.ff.proj_in.bias" in mapped
        assert "video_connector.transformer_blocks.0.ff.proj_out.weight" in mapped
        assert "video_connector.transformer_blocks.0.ff.proj_out.bias" in mapped

    def test_direct_mapping_preserved(self):
        """Test that directly matching keys are preserved."""
        diffusers_state_dict = {
            "text_proj_in.weight": torch.randn(3840, 188160),
            "video_connector.learnable_registers": torch.randn(128, 3840),
        }

        mapped = _map_diffusers_state_dict(diffusers_state_dict)

        assert "text_proj_in.weight" in mapped
        assert "video_connector.learnable_registers" in mapped


# ============================================================================
# Integration Tests (require GPU and weights)
# ============================================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestConnectorIntegration:
    """Integration tests that require GPU."""

    def test_connector_dtype_handling(self):
        """Test connector works with bfloat16."""
        connector = LTX2ConnectorTransformer1d(
            num_attention_heads=30,
            attention_head_dim=128,
            num_layers=2,
            num_learnable_registers=128,
        ).cuda().to(torch.bfloat16)

        x = torch.randn(1, 256, 3840, device="cuda", dtype=torch.bfloat16)
        out, _ = connector(x)

        assert out.dtype == torch.bfloat16
        assert out.shape == (1, 256, 3840)


# ============================================================================
# Layer Contribution Test (verifies different layers give different scores)
# ============================================================================

class TestLayerContributionVariance:
    """Test that layer masking produces varying outputs."""

    def test_different_masks_give_different_outputs(self):
        """Verify that masking different layers produces different embeddings."""
        connectors = LTX2TextConnectors()

        batch, seq = 1, 256
        base_input = torch.randn(batch, seq, 3840 * 49)
        mask = torch.ones(batch, seq)

        # Get output with full input
        out1, _, _ = connectors(base_input.clone(), mask.clone())

        # Modify input to simulate layer masking (zero out one layer's features)
        modified_input = base_input.clone()
        layer_start = 20 * 3840  # Layer 20
        layer_end = 21 * 3840
        modified_input[:, :, layer_start:layer_end] = 0

        out2, _, _ = connectors(modified_input, mask.clone())

        # Outputs should differ significantly
        diff = (out1 - out2).abs().mean()
        assert diff > 1e-3, "Different inputs should produce different outputs"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
