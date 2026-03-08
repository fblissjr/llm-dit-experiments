"""
Tests for connector gated attention correctness.

Last Updated: 2026-03-08

Regression tests for the V2.3 embeddings connector gated attention fix:
- 2.0x sigmoid multiplier (zero-init gates -> identity)
- Gating applied BEFORE to_out projection (inside Attention.forward)
- Scaling range [0, 2] per-head

Run with: uv run pytest tests/unit/test_connector_gated_attention.py -v
"""

import json
import logging
from unittest.mock import MagicMock, patch

import pytest
import torch

from llm_dit.encoders.embeddings_connector import Attention, BasicTransformerBlock1D

pytestmark = pytest.mark.unit


class TestGatedAttention2xSigmoid:
    """Verify 2.0 * sigmoid(gate) identity property."""

    def test_zero_init_gate_produces_identity(self):
        """With zero-initialized gate bias, 2*sigmoid(0)=1.0 -> output equals non-gated baseline."""
        torch.manual_seed(42)
        heads = 4
        dim_head = 32
        query_dim = heads * dim_head  # 128

        # Create gated and non-gated attention with identical weights
        gated = Attention(
            query_dim=query_dim, heads=heads, dim_head=dim_head,
            apply_gated_attention=True,
        )
        non_gated = Attention(
            query_dim=query_dim, heads=heads, dim_head=dim_head,
            apply_gated_attention=False,
        )

        # Copy all shared weights from gated -> non_gated
        non_gated.to_q.load_state_dict(gated.to_q.state_dict())
        non_gated.to_k.load_state_dict(gated.to_k.state_dict())
        non_gated.to_v.load_state_dict(gated.to_v.state_dict())
        non_gated.to_out.load_state_dict(gated.to_out.state_dict())
        non_gated.q_norm.load_state_dict(gated.q_norm.state_dict())
        non_gated.k_norm.load_state_dict(gated.k_norm.state_dict())

        # Zero-init gate bias (weight already random, but bias=0 -> logit=bias for zero input...
        # Actually logit = W@x + b, so we need to force the full output to 0.
        # Simplest: zero both weight and bias.
        torch.nn.init.zeros_(gated.to_gate_logits.weight)
        torch.nn.init.zeros_(gated.to_gate_logits.bias)

        x = torch.randn(1, 16, query_dim)
        with torch.no_grad():
            out_gated = gated(x)
            out_non_gated = non_gated(x)

        # 2 * sigmoid(0) = 1.0, so gated output should equal non-gated
        torch.testing.assert_close(out_gated, out_non_gated, atol=1e-5, rtol=1e-5)


class TestGatedAttentionPreToOut:
    """Verify gate is applied BEFORE to_out projection (inside Attention, not BasicTransformerBlock1D)."""

    def test_gate_logits_on_attention_not_block(self):
        """to_gate_logits must be an attribute of Attention, not BasicTransformerBlock1D."""
        dim = 128
        heads = 4
        dim_head = 32

        attn = Attention(
            query_dim=dim, heads=heads, dim_head=dim_head,
            apply_gated_attention=True,
        )
        assert hasattr(attn, "to_gate_logits"), "to_gate_logits must be on Attention"
        assert attn.to_gate_logits is not None

        block = BasicTransformerBlock1D(
            dim=dim, heads=heads, dim_head=dim_head,
            apply_gated_attention=True,
        )
        # Gate should be on attn1, not on block directly
        assert hasattr(block.attn1, "to_gate_logits")
        assert block.attn1.to_gate_logits is not None
        assert not hasattr(block, "to_gate_logits"), (
            "to_gate_logits must NOT be on BasicTransformerBlock1D -- "
            "gating must happen inside Attention.forward before to_out"
        )


class TestGatedAttentionScalingRange:
    """Verify gate scaling range is [0, 2] via 2*sigmoid."""

    def test_extreme_negative_gate_zeros_attention(self):
        """Large negative gate logits -> sigmoid ~ 0 -> 2*0 ~ 0 -> attention contribution zeroed.

        Note: to_out has a bias, so the final output is to_out.bias (constant across positions),
        not zero. We verify that the output is position-invariant (constant), meaning the
        attention contribution was fully gated away.
        """
        heads = 4
        dim_head = 32
        query_dim = heads * dim_head

        attn = Attention(
            query_dim=query_dim, heads=heads, dim_head=dim_head,
            apply_gated_attention=True,
        )
        # Force gate to produce large negative logits
        torch.nn.init.zeros_(attn.to_gate_logits.weight)
        attn.to_gate_logits.bias.data.fill_(-100.0)

        x = torch.randn(1, 16, query_dim)
        with torch.no_grad():
            out_gated = attn(x)

        # All positions should produce identical output (just the to_out bias)
        # since the gated attention contribution is zero
        position_std = out_gated[0].std(dim=0).max()
        assert position_std < 1e-5, (
            f"Expected constant output across positions with gate=-100, "
            f"got max position std={position_std}"
        )

    def test_extreme_positive_gate_doubles_output(self):
        """Large positive gate logits -> sigmoid ~ 1 -> 2*1 = 2 -> output doubled vs non-gated."""
        heads = 4
        dim_head = 32
        query_dim = heads * dim_head

        gated = Attention(
            query_dim=query_dim, heads=heads, dim_head=dim_head,
            apply_gated_attention=True,
        )
        non_gated = Attention(
            query_dim=query_dim, heads=heads, dim_head=dim_head,
            apply_gated_attention=False,
        )

        # Copy shared weights
        non_gated.to_q.load_state_dict(gated.to_q.state_dict())
        non_gated.to_k.load_state_dict(gated.to_k.state_dict())
        non_gated.to_v.load_state_dict(gated.to_v.state_dict())
        non_gated.to_out.load_state_dict(gated.to_out.state_dict())
        non_gated.q_norm.load_state_dict(gated.q_norm.state_dict())
        non_gated.k_norm.load_state_dict(gated.k_norm.state_dict())

        # Force gate = +100 -> sigmoid ~ 1.0 -> 2 * 1.0 = 2.0
        torch.nn.init.zeros_(gated.to_gate_logits.weight)
        gated.to_gate_logits.bias.data.fill_(100.0)

        x = torch.randn(1, 16, query_dim)
        with torch.no_grad():
            out_gated = gated(x)
            out_non_gated = non_gated(x)

        # Gate=2.0 is applied BEFORE to_out, so the relationship is:
        # gated = to_out(2.0 * attn_values)
        # non_gated = to_out(attn_values)
        # Since to_out is linear: to_out(2x) = 2 * to_out(x) (for zero-bias to_out)
        # But to_out has bias, so check ratio on individual elements
        # More robust: check that gated output magnitude is roughly 2x non_gated
        ratio = out_gated.abs().mean() / out_non_gated.abs().mean()
        assert 1.5 < ratio < 2.5, (
            f"Expected ~2x scaling with gate=+100, got ratio={ratio:.3f}"
        )


class TestNoGatedAttentionPassthrough:
    """Verify non-gated attention has no gate infrastructure."""

    def test_no_gate_logits_when_disabled(self):
        """apply_gated_attention=False -> to_gate_logits is None."""
        attn = Attention(
            query_dim=128, heads=4, dim_head=32,
            apply_gated_attention=False,
        )
        assert attn.to_gate_logits is None

    def test_gated_vs_non_gated_param_count(self):
        """Gated attention should have extra parameters from to_gate_logits."""
        heads = 4
        dim_head = 32
        query_dim = heads * dim_head

        gated = Attention(
            query_dim=query_dim, heads=heads, dim_head=dim_head,
            apply_gated_attention=True,
        )
        non_gated = Attention(
            query_dim=query_dim, heads=heads, dim_head=dim_head,
            apply_gated_attention=False,
        )

        gated_params = sum(p.numel() for p in gated.parameters())
        non_gated_params = sum(p.numel() for p in non_gated.parameters())

        # to_gate_logits: Linear(query_dim, heads) -> query_dim*heads + heads params
        expected_extra = query_dim * heads + heads
        assert gated_params - non_gated_params == expected_extra, (
            f"Expected {expected_extra} extra params, got {gated_params - non_gated_params}"
        )


class TestConnectorConfigValidation:
    """Tests for _validate_connector_config safetensors metadata check."""

    def _make_encoder(self):
        """Create a minimal Gemma3Encoder-like object with _validate_connector_config."""
        from llm_dit.encoders.gemma3 import Gemma3Encoder
        # We only need the method, not a full encoder. Use __new__ to skip __init__.
        encoder = object.__new__(Gemma3Encoder)
        return encoder

    def test_matching_config_no_warning(self, caplog):
        """Matching configs should produce debug log, no warning."""
        encoder = self._make_encoder()
        our_config = {
            "video_connector_attention_head_dim": 128,
            "video_connector_num_attention_heads": 32,
            "video_connector_num_layers": 8,
            "video_connector_num_learnable_registers": 128,
            "connector_positional_embedding_max_pos": [4096],
            "apply_gated_attention": True,
        }
        # Mock safe_open to return matching metadata
        metadata = {"config": json.dumps(our_config)}
        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_file.metadata.return_value = metadata

        with patch("safetensors.safe_open", return_value=mock_file):
            with caplog.at_level(logging.DEBUG):
                encoder._validate_connector_config(our_config, "/fake/path.safetensors")

        assert "mismatch" not in caplog.text.lower()
        assert "matches" in caplog.text.lower()

    def test_mismatched_config_warns(self, caplog):
        """Mismatched configs should produce a warning log."""
        encoder = self._make_encoder()
        our_config = {
            "video_connector_attention_head_dim": 128,
            "connector_positional_embedding_max_pos": [1],  # Wrong!
            "apply_gated_attention": False,  # Wrong!
        }
        file_config = {
            "video_connector_attention_head_dim": 128,  # Matches
            "connector_positional_embedding_max_pos": [4096],  # Different
            "apply_gated_attention": True,  # Different
        }
        metadata = {"config": json.dumps(file_config)}
        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_file.metadata.return_value = metadata

        with patch("safetensors.safe_open", return_value=mock_file):
            with caplog.at_level(logging.WARNING):
                encoder._validate_connector_config(our_config, "/fake/path.safetensors")

        assert "mismatch" in caplog.text.lower()
        assert "connector_positional_embedding_max_pos" in caplog.text
        assert "apply_gated_attention" in caplog.text

    def test_no_metadata_skips_silently(self, caplog):
        """Missing metadata should not raise, just debug log."""
        encoder = self._make_encoder()
        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_file.metadata.return_value = None

        with patch("safetensors.safe_open", return_value=mock_file):
            with caplog.at_level(logging.DEBUG):
                encoder._validate_connector_config({}, "/fake/path.safetensors")

        assert "skipping validation" in caplog.text.lower()

    def test_file_read_error_skips_silently(self, caplog):
        """File read errors should not raise, just debug log."""
        encoder = self._make_encoder()

        with patch("safetensors.safe_open", side_effect=FileNotFoundError("no file")):
            with caplog.at_level(logging.DEBUG):
                encoder._validate_connector_config({}, "/nonexistent/path.safetensors")

        assert "could not read" in caplog.text.lower()
