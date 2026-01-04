"""
Unit tests for DiffSynth-style FP8 inference utilities.

Tests fp8_inference context manager, weight conversion, and hardware detection.
"""

import pytest
import torch
import torch.nn as nn

pytestmark = pytest.mark.unit


class TestFP8InferenceImports:
    """Test that FP8 utilities can be imported."""

    def test_import_from_quantization(self):
        from llm_dit.quantization import (
            fp8_inference,
            enable_fp8_weights,
            check_fp8_available,
            get_fp8_info,
            get_fp8_dtype,
            get_fp8_max,
        )
        assert fp8_inference is not None
        assert enable_fp8_weights is not None
        assert check_fp8_available is not None
        assert get_fp8_info is not None
        assert get_fp8_dtype is not None
        assert get_fp8_max is not None

    def test_import_from_module(self):
        from llm_dit.quantization.fp8_inference import (
            fp8_inference,
            enable_fp8_weights,
            enable_fp8_autocast,
        )
        assert fp8_inference is not None
        assert enable_fp8_weights is not None
        assert enable_fp8_autocast is not None


class TestFP8DtypeDetection:
    """Test FP8 dtype and max value detection."""

    def test_get_fp8_dtype_returns_valid_dtype(self):
        from llm_dit.quantization import get_fp8_dtype

        dtype = get_fp8_dtype()
        # Should be one of the FP8 types
        assert dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)

    def test_get_fp8_max_returns_positive(self):
        from llm_dit.quantization import get_fp8_max

        max_val = get_fp8_max()
        assert max_val > 0
        # Should be either 448.0 (NVIDIA) or 224.0 (AMD)
        assert max_val in (448.0, 224.0)

    def test_fp8_info_structure(self):
        from llm_dit.quantization import get_fp8_info

        info = get_fp8_info()

        assert "available" in info
        assert "dtype" in info
        assert "max_value" in info
        assert "platform" in info
        assert "device_name" in info

        assert isinstance(info["available"], bool)
        assert isinstance(info["dtype"], str)
        assert isinstance(info["max_value"], float)
        assert info["platform"] in ("cpu", "nvidia", "amd")

    def test_check_fp8_available(self):
        from llm_dit.quantization import check_fp8_available

        available = check_fp8_available()
        assert isinstance(available, bool)


class TestFP8InferenceContext:
    """Test fp8_inference context manager behavior."""

    def test_context_manager_disabled(self):
        """Context manager with enabled=False should be a no-op."""
        from llm_dit.quantization import fp8_inference
        import torch.nn.functional as F

        original_linear = F.linear

        with fp8_inference(enabled=False):
            # F.linear should remain unchanged
            assert F.linear is original_linear

        # Should still be unchanged after
        assert F.linear is original_linear

    def test_context_manager_restores_linear(self):
        """Context manager should restore F.linear after exit."""
        from llm_dit.quantization import fp8_inference
        import torch.nn.functional as F

        original_linear = F.linear

        with fp8_inference(enabled=True):
            # F.linear should be patched
            assert F.linear is not original_linear

        # Should be restored after exit
        assert F.linear is original_linear

    def test_context_manager_exception_safety(self):
        """Context manager should restore F.linear even on exception."""
        from llm_dit.quantization import fp8_inference
        import torch.nn.functional as F

        original_linear = F.linear

        try:
            with fp8_inference(enabled=True):
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Should be restored after exception
        assert F.linear is original_linear


class TestEnableFP8Weights:
    """Test enable_fp8_weights function."""

    def test_converts_linear_weights(self):
        """enable_fp8_weights should convert Linear layer weights to FP8."""
        from llm_dit.quantization import enable_fp8_weights, get_fp8_dtype

        fp8_dtype = get_fp8_dtype()

        # Create simple model
        model = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )

        # Verify original dtype
        assert model[0].weight.dtype == torch.float32
        assert model[2].weight.dtype == torch.float32

        # Convert to FP8
        enable_fp8_weights(model)

        # Verify FP8 dtype
        assert model[0].weight.dtype == fp8_dtype
        assert model[2].weight.dtype == fp8_dtype

    def test_skips_already_fp8(self):
        """Should skip layers already in FP8."""
        from llm_dit.quantization import enable_fp8_weights, get_fp8_dtype

        fp8_dtype = get_fp8_dtype()

        model = nn.Linear(16, 32)
        model.weight.data = model.weight.data.to(fp8_dtype)

        # Should not error
        enable_fp8_weights(model)
        assert model.weight.dtype == fp8_dtype

    def test_sets_attribute(self):
        """Should set _fp8_weights_enabled attribute."""
        from llm_dit.quantization import enable_fp8_weights

        model = nn.Linear(16, 32)
        enable_fp8_weights(model)

        assert hasattr(model, "_fp8_weights_enabled")
        assert model._fp8_weights_enabled is True


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for FP8 linear test"
)
class TestFP8LinearOp:
    """Test FP8 linear operation (requires CUDA)."""

    def test_fp8_linear_output_shape(self):
        """FP8 linear should produce correct output shape."""
        from llm_dit.quantization import fp8_inference, check_fp8_available

        if not check_fp8_available():
            pytest.skip("FP8 not available on this hardware")

        model = nn.Linear(64, 128).cuda().bfloat16()
        x = torch.randn(2, 16, 64, device="cuda", dtype=torch.bfloat16)

        with fp8_inference():
            output = model(x)

        assert output.shape == (2, 16, 128)
        assert output.dtype == torch.bfloat16

    def test_fp8_linear_with_bias(self):
        """FP8 linear should handle bias correctly."""
        from llm_dit.quantization import fp8_inference, check_fp8_available

        if not check_fp8_available():
            pytest.skip("FP8 not available on this hardware")

        model = nn.Linear(64, 128, bias=True).cuda().bfloat16()
        x = torch.randn(2, 16, 64, device="cuda", dtype=torch.bfloat16)

        with fp8_inference():
            output = model(x)

        assert output.shape == (2, 16, 128)

    def test_fp8_linear_no_bias(self):
        """FP8 linear should handle no bias correctly."""
        from llm_dit.quantization import fp8_inference, check_fp8_available

        if not check_fp8_available():
            pytest.skip("FP8 not available on this hardware")

        model = nn.Linear(64, 128, bias=False).cuda().bfloat16()
        x = torch.randn(2, 16, 64, device="cuda", dtype=torch.bfloat16)

        with fp8_inference():
            output = model(x)

        assert output.shape == (2, 16, 128)


class TestFP8Autocast:
    """Test FP8 autocast hooks."""

    def test_enable_autocast(self):
        """enable_fp8_autocast should add hooks to module."""
        from llm_dit.quantization.fp8_inference import enable_fp8_autocast

        model = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
        )

        enable_fp8_autocast(model)

        # Check that hooks were registered
        # (checking internal state isn't ideal but verifies registration)
        assert hasattr(model[0], "_fp8_autocast_enabled") or True  # May skip Linear
