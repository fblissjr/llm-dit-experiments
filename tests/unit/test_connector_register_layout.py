"""
Test that _replace_padded_with_learnable_registers compacts valid tokens to the LEFT.

Last Updated: 2026-03-09

The LTX-2 reference implementation places valid tokens at the start of the
sequence, with learnable registers filling the remaining positions on the right.
A previous bug had these reversed (registers on left, valid tokens on right),
which caused wrong RoPE positions for all text tokens.

Reference behavior (from coderef/LTX-2):
    Input (left-padded):  [PAD, PAD, PAD, tok1, tok2, tok3]
    Output:               [tok1, tok2, tok3, reg, reg, reg]

Buggy behavior (before fix):
    Input (left-padded):  [PAD, PAD, PAD, tok1, tok2, tok3]
    Output:               [reg, reg, reg, tok1, tok2, tok3]

Run with: uv run pytest tests/unit/test_connector_register_layout.py -v
"""

import pytest
import torch

from llm_dit.encoders.embeddings_connector import Embeddings1DConnector

pytestmark = pytest.mark.unit


def make_connector(num_registers: int = 4) -> Embeddings1DConnector:
    """Create a minimal connector for testing."""
    connector = Embeddings1DConnector(
        attention_head_dim=8,
        num_attention_heads=2,  # inner_dim = 16
        num_layers=0,           # no transformer blocks needed for register tests
        num_learnable_registers=num_registers,
    )
    # Zero out learnable registers so we can distinguish them from valid tokens
    with torch.no_grad():
        connector.learnable_registers.fill_(0.0)
    return connector


class TestRegisterLayoutMatchesReference:
    """Valid tokens must be compacted to the LEFT, registers fill the RIGHT."""

    def test_left_padded_input_valid_tokens_at_start_of_output(self):
        """
        With left-padded input (valid tokens on right), output should have
        valid tokens at positions 0..N-1 (left-aligned).
        """
        # seq_len=8, 4 valid tokens, 4 padding, 4 registers
        seq_len = 8
        dim = 16  # 2 heads * 8 head_dim
        batch = 1

        # Create input with distinct valid tokens (value 1.0) and padding (0.0)
        # Left-padded: [PAD, PAD, PAD, PAD, tok, tok, tok, tok]
        hidden_states = torch.zeros(batch, seq_len, dim)
        hidden_states[0, 4:, :] = 1.0  # valid tokens on right

        # Additive mask: 0=valid, -10000=padding (left-padded format)
        additive_mask = torch.zeros(batch, 1, 1, seq_len)
        additive_mask[0, 0, 0, :4] = -10000.0  # padding on left

        connector = make_connector(num_registers=4)
        # Set registers to a distinct value so we can identify them
        with torch.no_grad():
            connector.learnable_registers.fill_(-1.0)

        output, _ = connector._replace_padded_with_learnable_registers(
            hidden_states, additive_mask
        )

        # Valid tokens (value 1.0) should be at positions 0..3 (LEFT side)
        # Registers (value -1.0) should be at positions 4..7 (RIGHT side)
        left_values = output[0, :4, :].mean().item()
        right_values = output[0, 4:, :].mean().item()

        assert abs(left_values - 1.0) < 1e-5, (
            f"Expected valid tokens (1.0) at left positions, got {left_values:.4f}. "
            "Registers/padding are on the wrong side."
        )
        assert abs(right_values - (-1.0)) < 1e-5, (
            f"Expected registers (-1.0) at right positions, got {right_values:.4f}. "
            "Register layout is incorrect."
        )

    def test_valid_tokens_preserve_order(self):
        """Valid tokens should maintain their original order after compaction."""
        seq_len = 8
        dim = 16
        batch = 1

        # Left-padded: [PAD, PAD, tok1=1.0, tok2=2.0, tok3=3.0, tok4=4.0, tok5=5.0, tok6=6.0]
        hidden_states = torch.zeros(batch, seq_len, dim)
        for i, val in enumerate([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]):
            hidden_states[0, 2 + i, 0] = val  # use first dim to encode position

        additive_mask = torch.zeros(batch, 1, 1, seq_len)
        additive_mask[0, 0, 0, :2] = -10000.0  # 2 padding tokens

        connector = make_connector(num_registers=4)
        with torch.no_grad():
            connector.learnable_registers.fill_(0.0)

        output, _ = connector._replace_padded_with_learnable_registers(
            hidden_states, additive_mask
        )

        # Positions 0-5 should be tok1..tok6 in order
        for i, expected_val in enumerate([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]):
            actual_val = output[0, i, 0].item()
            assert abs(actual_val - expected_val) < 1e-5, (
                f"Position {i}: expected {expected_val}, got {actual_val:.4f}. "
                "Token order is wrong after register replacement."
            )

    def test_all_valid_no_registers_needed(self):
        """When all tokens are valid (no padding), registers still fill positions after valid tokens."""
        seq_len = 4
        dim = 16
        batch = 1

        hidden_states = torch.ones(batch, seq_len, dim) * 2.0  # all valid
        additive_mask = torch.zeros(batch, 1, 1, seq_len)       # all valid (0 = valid)

        connector = make_connector(num_registers=4)
        with torch.no_grad():
            connector.learnable_registers.fill_(-5.0)

        output, _ = connector._replace_padded_with_learnable_registers(
            hidden_states, additive_mask
        )

        # All positions should have valid tokens (2.0), no registers (-5.0)
        all_vals = output[0, :, :].mean().item()
        assert abs(all_vals - 2.0) < 1e-5, (
            f"All tokens valid: expected 2.0 everywhere, got {all_vals:.4f}"
        )

    def test_attention_mask_becomes_all_valid_after_replacement(self):
        """After register replacement, the returned mask should be all-valid (all zeros additive)."""
        seq_len = 8
        dim = 16
        batch = 1

        hidden_states = torch.randn(batch, seq_len, dim)
        additive_mask = torch.zeros(batch, 1, 1, seq_len)
        additive_mask[0, 0, 0, :3] = -10000.0  # 3 padding

        connector = make_connector(num_registers=4)
        _, out_mask = connector._replace_padded_with_learnable_registers(
            hidden_states, additive_mask
        )

        assert out_mask.max().item() == 0.0, "Returned mask should be all-zeros (all valid)"
        assert out_mask.min().item() == 0.0, "Returned mask should be all-zeros (all valid)"

    def test_batch_independence(self):
        """Each batch item is handled independently with its own valid token count."""
        seq_len = 8
        dim = 16
        batch = 2

        hidden_states = torch.zeros(batch, seq_len, dim)
        # Batch 0: 2 padding, 6 valid (value 1.0)
        hidden_states[0, 2:, :] = 1.0
        # Batch 1: 5 padding, 3 valid (value 2.0)
        hidden_states[1, 5:, :] = 2.0

        additive_mask = torch.zeros(batch, 1, 1, seq_len)
        additive_mask[0, 0, 0, :2] = -10000.0  # batch 0: 2 padding
        additive_mask[1, 0, 0, :5] = -10000.0  # batch 1: 5 padding

        connector = make_connector(num_registers=4)
        with torch.no_grad():
            connector.learnable_registers.fill_(0.0)

        output, _ = connector._replace_padded_with_learnable_registers(
            hidden_states, additive_mask
        )

        # Batch 0: positions 0-5 should be 1.0, positions 6-7 should be 0.0 (registers)
        assert abs(output[0, :6, :].mean().item() - 1.0) < 1e-5, (
            f"Batch 0: first 6 positions should be valid (1.0), "
            f"got {output[0, :6, :].mean().item():.4f}"
        )
        assert abs(output[0, 6:, :].mean().item() - 0.0) < 1e-5, (
            f"Batch 0: last 2 positions should be registers (0.0), "
            f"got {output[0, 6:, :].mean().item():.4f}"
        )

        # Batch 1: positions 0-2 should be 2.0, positions 3-7 should be 0.0 (registers)
        assert abs(output[1, :3, :].mean().item() - 2.0) < 1e-5, (
            f"Batch 1: first 3 positions should be valid (2.0), "
            f"got {output[1, :3, :].mean().item():.4f}"
        )
        assert abs(output[1, 3:, :].mean().item() - 0.0) < 1e-5, (
            f"Batch 1: last 5 positions should be registers (0.0), "
            f"got {output[1, 3:, :].mean().item():.4f}"
        )

    def test_matches_reference_implementation_exactly(self):
        """
        Run both the reference flip-based logic and our implementation
        and verify they produce identical results.

        Reference (from coderef/LTX-2):
          non_zero = hidden[:, mask.squeeze().bool(), :]
          adjusted = F.pad(non_zero, (0,0,0,pad_length))
          flipped_mask = flip(attention_mask_binary, dims=[1])
          out = flipped_mask * adjusted + (1-flipped_mask) * registers
        """
        import torch.nn.functional as F

        seq_len = 8
        dim = 16
        batch = 1

        torch.manual_seed(42)
        hidden_states = torch.randn(batch, seq_len, dim)

        # Left-padded: 3 padding, 5 valid
        additive_mask = torch.zeros(batch, 1, 1, seq_len)
        additive_mask[0, 0, 0, :3] = -10000.0

        connector = make_connector(num_registers=4)
        torch.manual_seed(99)
        with torch.no_grad():
            connector.learnable_registers.copy_(torch.randn(4, dim))

        # Our implementation
        our_output, _ = connector._replace_padded_with_learnable_registers(
            hidden_states.clone(), additive_mask.clone()
        )

        # Reference implementation (batch_size=1 only, from coderef)
        attention_mask_binary = (additive_mask.squeeze(1).squeeze(1).unsqueeze(-1) >= -9000.0).int()
        num_duplications = seq_len // 4
        ref_registers = connector.learnable_registers.repeat(num_duplications, 1).to(hidden_states.dtype)

        non_zero = hidden_states[:, attention_mask_binary.squeeze().bool(), :]
        non_zero_nums = non_zero.shape[1]
        pad_length_ref = seq_len - non_zero_nums
        adjusted = F.pad(non_zero, (0, 0, 0, pad_length_ref), value=0)
        flipped_mask = torch.flip(attention_mask_binary, dims=[1])
        ref_output = flipped_mask * adjusted + (1 - flipped_mask) * ref_registers

        assert torch.allclose(our_output, ref_output, atol=1e-5), (
            f"Our output does not match reference.\n"
            f"Max diff: {(our_output - ref_output).abs().max().item():.6f}\n"
            f"Our output (first 3 pos, first 4 dims):\n{our_output[0, :3, :4]}\n"
            f"Ref output (first 3 pos, first 4 dims):\n{ref_output[0, :3, :4]}"
        )
