"""
Unit tests for llm_dit.layers.normalization module.

Last Updated: 2026-02-01

Run with: uv run pytest tests/unit/test_layers_normalization.py -v
"""

import pytest
import torch
import torch.nn as nn

from llm_dit.layers.normalization import RMSNorm, rms_norm, T5LayerNorm


class TestRMSNormBasic:
    """Basic functionality tests for RMSNorm class."""

    def test_output_shape_matches_input(self):
        """Output shape should match input shape."""
        norm = RMSNorm(768)
        x = torch.randn(2, 128, 768)
        out = norm(x)
        assert out.shape == x.shape

    def test_output_shape_1d(self):
        """Works with 1D input (batch only)."""
        norm = RMSNorm(64)
        x = torch.randn(32, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_output_shape_4d(self):
        """Works with 4D input (image-like)."""
        norm = RMSNorm(256)
        x = torch.randn(2, 16, 16, 256)
        out = norm(x)
        assert out.shape == x.shape

    def test_default_eps_is_1e6(self):
        """Default eps should be 1e-6."""
        norm = RMSNorm(768)
        assert norm.eps == 1e-6

    def test_custom_eps(self):
        """Custom eps should be respected."""
        norm = RMSNorm(768, eps=1e-5)
        assert norm.eps == 1e-5

    def test_weight_parameter_exists(self):
        """Weight parameter should exist by default."""
        norm = RMSNorm(768)
        assert hasattr(norm, "weight")
        assert norm.weight is not None
        assert norm.weight.shape == (768,)

    def test_weight_initialized_to_ones(self):
        """Weight should be initialized to ones."""
        norm = RMSNorm(768)
        assert torch.allclose(norm.weight, torch.ones(768))

    def test_elementwise_affine_false(self):
        """elementwise_affine=False should disable weight."""
        norm = RMSNorm(768, elementwise_affine=False)
        assert norm.weight is None

    def test_extra_repr(self):
        """extra_repr should include key parameters."""
        norm = RMSNorm(768, eps=1e-5)
        repr_str = norm.extra_repr()
        assert "768" in repr_str
        assert "1e-05" in repr_str


class TestRMSNormDtype:
    """dtype preservation tests."""

    def test_preserves_float32(self):
        """Should preserve float32 dtype."""
        norm = RMSNorm(768)
        x = torch.randn(2, 128, 768, dtype=torch.float32)
        out = norm(x)
        assert out.dtype == torch.float32

    def test_preserves_float16(self):
        """Should preserve float16 dtype."""
        norm = RMSNorm(768)
        x = torch.randn(2, 128, 768, dtype=torch.float16)
        out = norm(x)
        assert out.dtype == torch.float16

    def test_preserves_bfloat16(self):
        """Should preserve bfloat16 dtype."""
        norm = RMSNorm(768)
        x = torch.randn(2, 128, 768, dtype=torch.bfloat16)
        out = norm(x)
        assert out.dtype == torch.bfloat16


class TestRMSNormNumerical:
    """Numerical correctness tests."""

    def test_zero_input_returns_zero(self):
        """Zero input should return zero (or near-zero)."""
        norm = RMSNorm(768)
        x = torch.zeros(2, 128, 768)
        out = norm(x)
        # With eps, output won't be exactly zero but should be very small
        assert out.abs().max() < 1e-3

    def test_constant_input_normalized(self):
        """Constant input should be normalized to ~1 (after weight=1)."""
        norm = RMSNorm(768)
        x = torch.ones(2, 128, 768) * 5.0
        out = norm(x)
        # RMS of constant 5 is 5, so normalized is 1
        assert torch.allclose(out, torch.ones_like(out), atol=1e-5)

    def test_gradient_flows(self):
        """Gradients should flow through normalization."""
        norm = RMSNorm(768)
        x = torch.randn(2, 128, 768, requires_grad=True)
        out = norm(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert norm.weight.grad is not None

    def test_no_nan_with_small_values(self):
        """Should not produce NaN with small input values."""
        norm = RMSNorm(768)
        x = torch.randn(2, 128, 768) * 1e-6
        out = norm(x)
        assert not torch.isnan(out).any()

    def test_no_nan_with_large_values(self):
        """Should not produce NaN with large input values."""
        norm = RMSNorm(768)
        x = torch.randn(2, 128, 768) * 1e6
        out = norm(x)
        assert not torch.isnan(out).any()


class TestRMSNormEquivalence:
    """Tests for numerical equivalence with existing implementations."""

    def test_equivalent_to_zimage_implementation(self):
        """Should be numerically equivalent to Z-Image RMSNorm."""
        # Z-Image implementation (from z_image/components.py)
        class ZImageRMSNorm(nn.Module):
            def __init__(self, dim: int, eps: float = 1e-5):
                super().__init__()
                self.eps = eps
                self.weight = nn.Parameter(torch.ones(dim))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
                return x * rms * self.weight

        dim = 768
        x = torch.randn(2, 128, dim)

        # Our implementation (with Z-Image eps)
        our_norm = RMSNorm(dim, eps=1e-5)
        zimage_norm = ZImageRMSNorm(dim, eps=1e-5)

        # Sync weights
        zimage_norm.weight.data = our_norm.weight.data.clone()

        out_ours = our_norm(x)
        out_zimage = zimage_norm(x)

        assert torch.allclose(out_ours, out_zimage, rtol=1e-4, atol=1e-5)

    def test_equivalent_to_wan_implementation(self):
        """Should be numerically equivalent to Wan DiT RMSNorm."""
        # Wan DiT implementation (from wan_dit.py)
        class WanRMSNorm(nn.Module):
            def __init__(self, dim: int, eps: float = 1e-6):
                super().__init__()
                self.eps = eps
                self.weight = nn.Parameter(torch.ones(dim))

            def _norm(self, x: torch.Tensor) -> torch.Tensor:
                return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self._norm(x.float()).to(x.dtype) * self.weight

        dim = 768
        x = torch.randn(2, 128, dim, dtype=torch.bfloat16)

        our_norm = RMSNorm(dim, eps=1e-6)
        wan_norm = WanRMSNorm(dim, eps=1e-6)

        # Sync weights
        wan_norm.weight.data = our_norm.weight.data.clone()

        out_ours = our_norm(x)
        out_wan = wan_norm(x)

        # Compare in float32 to avoid dtype mismatch
        assert torch.allclose(out_ours.float(), out_wan.float(), rtol=1e-3, atol=1e-4)

    def test_equivalent_to_flux2_implementation(self):
        """Should be numerically equivalent to FLUX.2 RMSNorm."""
        # FLUX.2 implementation (from flux2/transformer.py)
        class Flux2RMSNorm(nn.Module):
            def __init__(self, dim: int):
                super().__init__()
                self.scale = nn.Parameter(torch.ones(dim))

            def forward(self, x):
                x_dtype = x.dtype
                x = x.float()
                rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
                return (x * rrms).to(dtype=x_dtype) * self.scale

        dim = 768
        x = torch.randn(2, 128, dim, dtype=torch.bfloat16)

        our_norm = RMSNorm(dim, eps=1e-6)
        flux_norm = Flux2RMSNorm(dim)

        # Sync weights (FLUX.2 uses 'scale' instead of 'weight')
        flux_norm.scale.data = our_norm.weight.data.clone()

        out_ours = our_norm(x)
        out_flux = flux_norm(x)

        # Compare in float32 to avoid dtype mismatch
        assert torch.allclose(out_ours.float(), out_flux.float(), rtol=1e-3, atol=1e-4)


class TestRmsnormFunction:
    """Tests for the functional rms_norm interface."""

    def test_output_shape(self):
        """Functional form should preserve shape."""
        x = torch.randn(2, 128, 768)
        out = rms_norm(x)
        assert out.shape == x.shape

    def test_with_weight(self):
        """Should apply weight when provided."""
        x = torch.randn(2, 128, 768)
        weight = torch.ones(768) * 2.0
        out = rms_norm(x, weight=weight)
        out_no_weight = rms_norm(x)
        # Output with weight=2 should be 2x output with weight=1
        assert torch.allclose(out, out_no_weight * 2.0, rtol=1e-4)

    def test_without_weight(self):
        """Should work without weight."""
        x = torch.randn(2, 128, 768)
        out = rms_norm(x, weight=None)
        assert out.shape == x.shape

    def test_custom_eps(self):
        """Should respect custom eps."""
        x = torch.randn(2, 128, 768)
        out1 = rms_norm(x, eps=1e-5)
        out2 = rms_norm(x, eps=1e-6)
        # Outputs should be slightly different due to different eps
        assert not torch.allclose(out1, out2, rtol=1e-6)

    def test_matches_class_output(self):
        """Functional form should match class output."""
        dim = 768
        x = torch.randn(2, 128, dim)
        weight = torch.randn(dim)

        norm = RMSNorm(dim, eps=1e-6)
        norm.weight.data = weight.clone()

        out_class = norm(x)
        out_func = rms_norm(x, weight=weight, eps=1e-6)

        assert torch.allclose(out_class, out_func, rtol=1e-5)


class TestT5LayerNorm:
    """Tests for T5LayerNorm compatibility alias."""

    def test_is_subclass_of_rmsnorm(self):
        """T5LayerNorm should be a subclass of RMSNorm."""
        assert issubclass(T5LayerNorm, RMSNorm)

    def test_functionally_identical(self):
        """T5LayerNorm should behave identically to RMSNorm."""
        dim = 768
        x = torch.randn(2, 128, dim)

        rms = RMSNorm(dim)
        t5 = T5LayerNorm(dim)
        t5.weight.data = rms.weight.data.clone()

        out_rms = rms(x)
        out_t5 = t5(x)

        assert torch.allclose(out_rms, out_t5)


class TestRMSNormDevice:
    """Device placement tests."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_forward(self):
        """Should work on CUDA device."""
        norm = RMSNorm(768).cuda()
        x = torch.randn(2, 128, 768, device="cuda")
        out = norm(x)
        assert out.device.type == "cuda"
        assert out.shape == x.shape

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_mixed_device_raises(self):
        """Should raise error when input and weight on different devices."""
        norm = RMSNorm(768)  # CPU
        x = torch.randn(2, 128, 768, device="cuda")
        with pytest.raises(RuntimeError):
            norm(x)
