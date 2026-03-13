"""
Unit tests for llm_dit.layers.feedforward module.

Last Updated: 2026-02-01

Run with: uv run pytest tests/unit/test_layers_feedforward.py -v
"""

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_dit.layers.feedforward import (
    FeedForward,
    FFNType,
    LTX2_FFN_PRESET,
    ZIMAGE_FFN_PRESET,
    CONNECTOR_FFN_PRESET,
)


class TestFeedForwardBasic:
    """Basic functionality tests for FeedForward class."""

    def test_output_shape_matches_input_standard(self):
        """Standard FFN should preserve input/output shape."""
        ff = FeedForward(768, ffn_type=FFNType.STANDARD)
        x = torch.randn(2, 128, 768)
        out = ff(x)
        assert out.shape == x.shape

    def test_output_shape_matches_input_swiglu(self):
        """SwiGLU FFN should preserve input/output shape."""
        ff = FeedForward(768, ffn_type=FFNType.SWIGLU)
        x = torch.randn(2, 128, 768)
        out = ff(x)
        assert out.shape == x.shape

    def test_output_shape_matches_input_geglu(self):
        """GeGLU FFN should preserve input/output shape."""
        ff = FeedForward(768, ffn_type=FFNType.GEGLU)
        x = torch.randn(2, 128, 768)
        out = ff(x)
        assert out.shape == x.shape

    def test_output_shape_2d_input(self):
        """Works with 2D input (batch, dim)."""
        ff = FeedForward(64, ffn_type=FFNType.STANDARD)
        x = torch.randn(32, 64)
        out = ff(x)
        assert out.shape == x.shape

    def test_output_shape_4d_input(self):
        """Works with 4D input (image-like)."""
        ff = FeedForward(256, ffn_type=FFNType.SWIGLU)
        x = torch.randn(2, 16, 16, 256)
        out = ff(x)
        assert out.shape == x.shape

    def test_custom_hidden_dim(self):
        """Custom hidden_dim should override mult."""
        ff = FeedForward(768, hidden_dim=1024, ffn_type=FFNType.STANDARD)
        assert ff.hidden_dim == 1024

    def test_default_mult_is_4(self):
        """Default mult should be 4."""
        ff = FeedForward(768, ffn_type=FFNType.STANDARD)
        assert ff.hidden_dim == 768 * 4

    def test_custom_mult(self):
        """Custom mult should be respected."""
        ff = FeedForward(768, mult=8 / 3, ffn_type=FFNType.SWIGLU)
        expected = int(768 * 8 / 3)
        assert ff.hidden_dim == expected

    def test_extra_repr(self):
        """extra_repr should include key parameters."""
        ff = FeedForward(768, ffn_type=FFNType.SWIGLU, dropout=0.1)
        repr_str = ff.extra_repr()
        assert "768" in repr_str
        assert "swiglu" in repr_str
        assert "0.1" in repr_str


class TestFeedForwardDtype:
    """dtype preservation tests."""

    def test_preserves_float32_standard(self):
        """Standard FFN should preserve float32 dtype."""
        ff = FeedForward(768, ffn_type=FFNType.STANDARD)
        x = torch.randn(2, 128, 768, dtype=torch.float32)
        out = ff(x)
        assert out.dtype == torch.float32

    def test_preserves_float16_swiglu(self):
        """SwiGLU FFN should preserve float16 dtype."""
        ff = FeedForward(768, ffn_type=FFNType.SWIGLU).half()
        x = torch.randn(2, 128, 768, dtype=torch.float16)
        out = ff(x)
        assert out.dtype == torch.float16

    def test_preserves_bfloat16_geglu(self):
        """GeGLU FFN should preserve bfloat16 dtype."""
        ff = FeedForward(768, ffn_type=FFNType.GEGLU).to(torch.bfloat16)
        x = torch.randn(2, 128, 768, dtype=torch.bfloat16)
        out = ff(x)
        assert out.dtype == torch.bfloat16


class TestFeedForwardNumerical:
    """Numerical correctness tests."""

    def test_zero_input_standard_no_bias(self):
        """Standard FFN with zero input and no bias should return zero."""
        ff = FeedForward(768, ffn_type=FFNType.STANDARD, bias=False)
        x = torch.zeros(2, 128, 768)
        out = ff(x)
        # GELU(0) = 0, and no bias means output is zero
        assert out.abs().max() < 1e-6

    def test_zero_input_swiglu(self):
        """SwiGLU FFN with zero input should return zero."""
        ff = FeedForward(768, ffn_type=FFNType.SWIGLU, bias=False)
        x = torch.zeros(2, 128, 768)
        out = ff(x)
        # silu(0) * anything = 0
        assert out.abs().max() == 0.0

    def test_gradient_flows_standard(self):
        """Gradients should flow through standard FFN."""
        ff = FeedForward(768, ffn_type=FFNType.STANDARD)
        x = torch.randn(2, 128, 768, requires_grad=True)
        out = ff(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_gradient_flows_swiglu(self):
        """Gradients should flow through SwiGLU FFN."""
        ff = FeedForward(768, ffn_type=FFNType.SWIGLU)
        x = torch.randn(2, 128, 768, requires_grad=True)
        out = ff(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert ff.w1.weight.grad is not None

    def test_gradient_flows_geglu(self):
        """Gradients should flow through GeGLU FFN."""
        ff = FeedForward(768, ffn_type=FFNType.GEGLU)
        x = torch.randn(2, 128, 768, requires_grad=True)
        out = ff(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert ff.fc1.weight.grad is not None

    def test_no_nan_with_small_values(self):
        """Should not produce NaN with small input values."""
        ff = FeedForward(768, ffn_type=FFNType.STANDARD)
        x = torch.randn(2, 128, 768) * 1e-6
        out = ff(x)
        assert not torch.isnan(out).any()

    def test_no_nan_with_large_values(self):
        """Should not produce NaN with large input values (float32)."""
        ff = FeedForward(768, ffn_type=FFNType.SWIGLU)
        x = torch.randn(2, 128, 768) * 1e3
        out = ff(x)
        assert not torch.isnan(out).any()


class TestFeedForwardPresets:
    """Tests for preset configurations."""

    def test_ltx2_preset_creates_standard(self):
        """LTX2 preset should create standard FFN."""
        ff = FeedForward(4096, **LTX2_FFN_PRESET)
        assert ff.ffn_type == FFNType.STANDARD
        assert ff.hidden_dim == 4096 * 4
        assert hasattr(ff, "net")

    def test_zimage_preset_creates_swiglu(self):
        """Z-Image preset should create SwiGLU FFN."""
        ff = FeedForward(3072, **ZIMAGE_FFN_PRESET)
        assert ff.ffn_type == FFNType.SWIGLU
        assert ff.hidden_dim == int(3072 * 8 / 3)
        assert hasattr(ff, "w1")
        assert hasattr(ff, "w2")
        assert hasattr(ff, "w3")
        # Check bias=False
        assert not ff.w1.bias

    def test_geglu_ffn_creates_geglu(self):
        """GeGLU FFN should be created with correct parameters."""
        ff = FeedForward(4096, hidden_dim=10240, ffn_type=FFNType.GEGLU, dropout=0.1, bias=False)
        assert ff.ffn_type == FFNType.GEGLU
        assert ff.hidden_dim == 10240
        assert ff.dropout_prob == 0.1

    def test_connector_preset_matches_ltx2(self):
        """Connector preset should match LTX2 preset."""
        ff1 = FeedForward(4096, **LTX2_FFN_PRESET)
        ff2 = FeedForward(4096, **CONNECTOR_FFN_PRESET)
        assert ff1.ffn_type == ff2.ffn_type
        assert ff1.hidden_dim == ff2.hidden_dim


class TestFeedForwardEquivalence:
    """Tests for numerical equivalence with existing implementations."""

    def test_equivalent_to_ltx2_feedforward(self):
        """Should be numerically equivalent to LTX-2 FeedForward."""
        # LTX-2 implementation (from ltx2/components.py)
        class LTX2GELUApprox(nn.Module):
            def __init__(self, dim_in: int, dim_out: int) -> None:
                super().__init__()
                self.proj = nn.Linear(dim_in, dim_out)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return F.gelu(self.proj(x), approximate="tanh")

        class LTX2FeedForward(nn.Module):
            def __init__(self, dim: int, dim_out: int, mult: int = 4) -> None:
                super().__init__()
                inner_dim = int(dim * mult)
                project_in = LTX2GELUApprox(dim, inner_dim)
                self.net = nn.Sequential(
                    project_in, nn.Identity(), nn.Linear(inner_dim, dim_out)
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.net(x)

        dim = 768
        x = torch.randn(2, 128, dim)

        # Our implementation
        our_ff = FeedForward(dim, **LTX2_FFN_PRESET)
        ltx2_ff = LTX2FeedForward(dim, dim, mult=4)

        # Sync weights
        ltx2_ff.net[0].proj.weight.data = our_ff.net[0].proj.weight.data.clone()
        ltx2_ff.net[0].proj.bias.data = our_ff.net[0].proj.bias.data.clone()
        ltx2_ff.net[2].weight.data = our_ff.net[2].weight.data.clone()
        ltx2_ff.net[2].bias.data = our_ff.net[2].bias.data.clone()

        out_ours = our_ff(x)
        out_ltx2 = ltx2_ff(x)

        assert torch.allclose(out_ours, out_ltx2, rtol=1e-5, atol=1e-6)

    def test_equivalent_to_zimage_feedforward(self):
        """Should be numerically equivalent to Z-Image FeedForward."""
        # Z-Image implementation (from z_image/components.py)
        class ZImageFeedForward(nn.Module):
            def __init__(self, dim: int, hidden_dim: int):
                super().__init__()
                self.w1 = nn.Linear(dim, hidden_dim, bias=False)
                self.w2 = nn.Linear(hidden_dim, dim, bias=False)
                self.w3 = nn.Linear(dim, hidden_dim, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.w2(F.silu(self.w1(x)) * self.w3(x))

        dim = 3072
        hidden_dim = int(dim * 8 / 3)
        x = torch.randn(2, 128, dim)

        # Our implementation
        our_ff = FeedForward(dim, **ZIMAGE_FFN_PRESET)
        zimage_ff = ZImageFeedForward(dim, hidden_dim)

        # Sync weights
        zimage_ff.w1.weight.data = our_ff.w1.weight.data.clone()
        zimage_ff.w2.weight.data = our_ff.w2.weight.data.clone()
        zimage_ff.w3.weight.data = our_ff.w3.weight.data.clone()

        out_ours = our_ff(x)
        out_zimage = zimage_ff(x)

        assert torch.allclose(out_ours, out_zimage, rtol=1e-5, atol=1e-6)

    def test_equivalent_to_t5_feedforward(self):
        """Should be numerically equivalent to T5 FeedForward (GeGLU)."""
        # T5 implementation (from wan_text_encoder.py)
        class T5GELU(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return 0.5 * x * (
                    1.0
                    + torch.tanh(
                        math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
                    )
                )

        class T5FeedForward(nn.Module):
            def __init__(self, dim: int, dim_ffn: int, dropout: float = 0.1):
                super().__init__()
                self.gate = nn.Sequential(nn.Linear(dim, dim_ffn, bias=False), T5GELU())
                self.fc1 = nn.Linear(dim, dim_ffn, bias=False)
                self.fc2 = nn.Linear(dim_ffn, dim, bias=False)
                self.dropout = nn.Dropout(dropout)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.fc1(x) * self.gate(x)
                x = self.dropout(x)
                x = self.fc2(x)
                x = self.dropout(x)
                return x

        dim = 4096
        dim_ffn = 10240
        x = torch.randn(2, 128, dim)

        # Our implementation
        our_ff = FeedForward(dim, hidden_dim=dim_ffn, ffn_type=FFNType.GEGLU, dropout=0.1, bias=False)
        t5_ff = T5FeedForward(dim, dim_ffn, dropout=0.1)

        # Sync weights
        t5_ff.gate[0].weight.data = our_ff.gate[0].weight.data.clone()
        t5_ff.fc1.weight.data = our_ff.fc1.weight.data.clone()
        t5_ff.fc2.weight.data = our_ff.fc2.weight.data.clone()

        # Set to inference mode for deterministic comparison
        our_ff.train(False)
        t5_ff.train(False)

        out_ours = our_ff(x)
        out_t5 = t5_ff(x)

        assert torch.allclose(out_ours, out_t5, rtol=1e-5, atol=1e-6)

    def test_equivalent_to_connector_feedforward(self):
        """Should be numerically equivalent to embeddings_connector FeedForward."""
        # Connector implementation (from embeddings_connector.py) - same as LTX2
        class ConnectorGELUApprox(nn.Module):
            def __init__(self, dim_in: int, dim_out: int):
                super().__init__()
                self.proj = nn.Linear(dim_in, dim_out)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return F.gelu(self.proj(x), approximate="tanh")

        class ConnectorFeedForward(nn.Module):
            def __init__(self, dim: int, dim_out: int, mult: int = 4):
                super().__init__()
                inner_dim = int(dim * mult)
                self.net = nn.Sequential(
                    ConnectorGELUApprox(dim, inner_dim),
                    nn.Identity(),
                    nn.Linear(inner_dim, dim_out),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.net(x)

        dim = 768
        x = torch.randn(2, 128, dim)

        # Our implementation
        our_ff = FeedForward(dim, **CONNECTOR_FFN_PRESET)
        conn_ff = ConnectorFeedForward(dim, dim, mult=4)

        # Sync weights
        conn_ff.net[0].proj.weight.data = our_ff.net[0].proj.weight.data.clone()
        conn_ff.net[0].proj.bias.data = our_ff.net[0].proj.bias.data.clone()
        conn_ff.net[2].weight.data = our_ff.net[2].weight.data.clone()
        conn_ff.net[2].bias.data = our_ff.net[2].bias.data.clone()

        out_ours = our_ff(x)
        out_conn = conn_ff(x)

        assert torch.allclose(out_ours, out_conn, rtol=1e-5, atol=1e-6)


class TestFeedForwardActivations:
    """Tests for different activation functions in standard FFN."""

    def test_gelu_tanh_activation(self):
        """GELU(tanh) activation should work."""
        ff = FeedForward(768, ffn_type=FFNType.STANDARD, activation="gelu_tanh")
        x = torch.randn(2, 128, 768)
        out = ff(x)
        assert out.shape == x.shape

    def test_gelu_activation(self):
        """GELU (exact) activation should work."""
        ff = FeedForward(768, ffn_type=FFNType.STANDARD, activation="gelu")
        x = torch.randn(2, 128, 768)
        out = ff(x)
        assert out.shape == x.shape

    def test_silu_activation(self):
        """SiLU activation should work."""
        ff = FeedForward(768, ffn_type=FFNType.STANDARD, activation="silu")
        x = torch.randn(2, 128, 768)
        out = ff(x)
        assert out.shape == x.shape

    def test_different_activations_produce_different_outputs(self):
        """Different activations should produce different outputs."""
        torch.manual_seed(42)
        x = torch.randn(2, 128, 768)

        ff_gelu_tanh = FeedForward(768, ffn_type=FFNType.STANDARD, activation="gelu_tanh")
        ff_gelu = FeedForward(768, ffn_type=FFNType.STANDARD, activation="gelu")

        # Sync weights
        ff_gelu.net[0].proj.weight.data = ff_gelu_tanh.net[0].proj.weight.data.clone()
        ff_gelu.net[0].proj.bias.data = ff_gelu_tanh.net[0].proj.bias.data.clone()
        ff_gelu.net[2].weight.data = ff_gelu_tanh.net[2].weight.data.clone()
        ff_gelu.net[2].bias.data = ff_gelu_tanh.net[2].bias.data.clone()

        out_tanh = ff_gelu_tanh(x)
        out_exact = ff_gelu(x)

        # They should be close but not exactly equal
        # Using rtol=1e-5 to detect meaningful differences
        assert not torch.allclose(out_tanh, out_exact, rtol=1e-5, atol=1e-6)
        # But quite close since tanh is a good approximation
        # Allowing atol=1e-3 to handle small absolute differences
        assert torch.allclose(out_tanh, out_exact, rtol=1e-3, atol=1e-3)


class TestFeedForwardBias:
    """Tests for bias configuration."""

    def test_bias_true_standard(self):
        """Standard FFN with bias=True should have bias."""
        ff = FeedForward(768, ffn_type=FFNType.STANDARD, bias=True)
        assert ff.net[0].proj.bias is not None
        assert ff.net[2].bias is not None

    def test_bias_false_standard(self):
        """Standard FFN with bias=False should not have bias."""
        ff = FeedForward(768, ffn_type=FFNType.STANDARD, bias=False)
        assert ff.net[0].proj.bias is None
        assert ff.net[2].bias is None

    def test_bias_false_swiglu(self):
        """SwiGLU FFN with bias=False should not have bias."""
        ff = FeedForward(768, ffn_type=FFNType.SWIGLU, bias=False)
        assert ff.w1.bias is None
        assert ff.w2.bias is None
        assert ff.w3.bias is None

    def test_bias_true_geglu(self):
        """GeGLU FFN with bias=True should have bias."""
        ff = FeedForward(768, ffn_type=FFNType.GEGLU, bias=True)
        assert ff.gate[0].bias is not None
        assert ff.fc1.bias is not None
        assert ff.fc2.bias is not None


class TestFeedForwardDropout:
    """Tests for dropout behavior."""

    def test_dropout_applied_in_train_mode_geglu(self):
        """GeGLU should apply dropout in train mode."""
        ff = FeedForward(768, ffn_type=FFNType.GEGLU, dropout=0.5)
        ff.train()

        x = torch.ones(10, 100, 768)
        outputs = [ff(x).clone() for _ in range(5)]

        # With 50% dropout, outputs should vary significantly
        all_same = all(torch.allclose(outputs[0], out) for out in outputs[1:])
        assert not all_same, "Outputs should vary with dropout in train mode"

    def test_dropout_not_applied_in_inference_mode_geglu(self):
        """GeGLU should not apply dropout in inference mode."""
        ff = FeedForward(768, ffn_type=FFNType.GEGLU, dropout=0.5)
        ff.train(False)

        x = torch.randn(2, 128, 768)
        outputs = [ff(x) for _ in range(3)]

        # All outputs should be identical in inference mode
        for out in outputs[1:]:
            assert torch.allclose(outputs[0], out)


class TestFeedForwardDevice:
    """Device placement tests."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_forward_standard(self):
        """Standard FFN should work on CUDA device."""
        ff = FeedForward(768, ffn_type=FFNType.STANDARD).cuda()
        x = torch.randn(2, 128, 768, device="cuda")
        out = ff(x)
        assert out.device.type == "cuda"
        assert out.shape == x.shape

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_forward_swiglu(self):
        """SwiGLU FFN should work on CUDA device."""
        ff = FeedForward(768, ffn_type=FFNType.SWIGLU).cuda()
        x = torch.randn(2, 128, 768, device="cuda")
        out = ff(x)
        assert out.device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_forward_geglu(self):
        """GeGLU FFN should work on CUDA device."""
        ff = FeedForward(768, ffn_type=FFNType.GEGLU).cuda()
        x = torch.randn(2, 128, 768, device="cuda")
        out = ff(x)
        assert out.device.type == "cuda"
