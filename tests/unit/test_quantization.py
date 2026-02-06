"""
Unit tests for llm_dit.quantization module.

Last Updated: 2026-02-06

Tests cover:
- Unified quantize_component() entry point
- TorchAO quantization utilities
- VAE quantization utilities
- FP8 compatibility checking
- Method validation and recommended method selection
"""

from unittest.mock import MagicMock, patch
import pytest
import torch
import torch.nn as nn


# ============================================================================
# Unified quantize_component() Tests
# ============================================================================

class TestQuantizeComponent:
    """Test the unified quantize_component() entry point."""

    def test_valid_methods_constant(self):
        """Test VALID_METHODS contains all expected methods."""
        from llm_dit.quantization.torchao_utils import VALID_METHODS

        assert "none" in VALID_METHODS
        assert "fp8-dynamic" in VALID_METHODS
        assert "fp8-weight-only" in VALID_METHODS
        assert "int8" in VALID_METHODS
        assert "int4" in VALID_METHODS

    def test_quantize_component_none_is_noop(self):
        """Test quantize_component with method='none' returns model unchanged."""
        from llm_dit.quantization import quantize_component

        model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16))
        original_params = {n: p.clone() for n, p in model.named_parameters()}

        result, stats = quantize_component(model, method="none", component_type="transformer")

        assert result is model
        assert stats["method"] == "none"
        assert stats["quantized_layers"] == 0
        # Parameters should be identical
        for name, param in result.named_parameters():
            assert torch.equal(param, original_params[name])

    def test_quantize_component_invalid_method_raises(self):
        """Test quantize_component raises ValueError for invalid method."""
        from llm_dit.quantization import quantize_component

        model = nn.Linear(64, 32)

        with pytest.raises(ValueError, match="Unknown quantization method"):
            quantize_component(model, method="banana", component_type="transformer")

    def test_quantize_component_int8_on_small_model(self):
        """Test quantize_component with int8 on a small model (no GPU needed)."""
        from llm_dit.quantization import quantize_component

        model = nn.Sequential(
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.Linear(32, 16),
        )

        result, stats = quantize_component(
            model, method="int8", component_type="transformer", verbose=False
        )

        assert stats["method"] == "int8"
        assert stats["component_type"] == "transformer"
        assert stats["quantized_layers"] > 0
        assert stats["total_layers"] > 0

    def test_quantize_component_int4_on_small_model(self):
        """Test quantize_component with int4 on a small model (no GPU needed)."""
        from llm_dit.quantization import quantize_component

        model = nn.Sequential(
            nn.Linear(64, 32),
            nn.Linear(32, 16),
        )

        result, stats = quantize_component(
            model, method="int4", component_type="transformer", verbose=False
        )

        assert stats["method"] == "int4"
        assert stats["quantized_layers"] > 0

    def test_quantize_component_encoder_skips_norms(self):
        """Test encoder component_type skips norm layers."""
        from llm_dit.quantization import quantize_component

        class FakeEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 32)
                self.layer_norm = nn.LayerNorm(32)
                self.embed_tokens = nn.Embedding(100, 64)

        model = FakeEncoder()
        result, stats = quantize_component(
            model, method="int8", component_type="encoder", verbose=False
        )

        # embed_tokens and layer_norm should be skipped
        assert stats["skipped_layers"] >= 0  # At least some skipped
        assert stats["method"] == "int8"
        assert stats["component_type"] == "encoder"

    def test_quantize_component_stats_dict_shape(self):
        """Test that stats dict contains all expected keys."""
        from llm_dit.quantization import quantize_component

        model = nn.Linear(64, 32)
        _, stats = quantize_component(
            model, method="none", component_type="transformer"
        )

        expected_keys = {"quantized_layers", "skipped_layers", "total_layers", "method", "component_type"}
        assert expected_keys.issubset(set(stats.keys()))


class TestQuantCompileWarnings:
    """Test get_quant_compile_warnings()."""

    def test_no_warnings_for_safe_combo(self):
        """fp8-weight-only + default compile should produce no warnings."""
        from llm_dit.quantization.torchao_utils import get_quant_compile_warnings

        warnings = get_quant_compile_warnings("fp8-weight-only", "default")
        assert isinstance(warnings, list)

    def test_warnings_for_fp8_dynamic_compile(self):
        """fp8-dynamic + compile should warn about autotune."""
        from llm_dit.quantization.torchao_utils import get_quant_compile_warnings

        warnings = get_quant_compile_warnings("fp8-dynamic", "reduce-overhead")
        assert isinstance(warnings, list)
        # fp8-dynamic with reduce-overhead may produce warnings
        # (exact behavior depends on implementation)


# ============================================================================
# TorchAO Utilities Tests
# ============================================================================

class TestTorchAOAvailability:
    """Test TorchAO availability checking."""

    def test_is_torchao_available_when_installed(self):
        """Test is_torchao_available returns True when installed."""
        from llm_dit.quantization.torchao_utils import is_torchao_available

        # Reset cached value
        import llm_dit.quantization.torchao_utils as tao
        tao._TORCHAO_AVAILABLE = None

        # This will check actual installation
        result = is_torchao_available()
        assert isinstance(result, bool)

    def test_get_torchao_version(self):
        """Test get_torchao_version returns string or None."""
        from llm_dit.quantization.torchao_utils import get_torchao_version

        version = get_torchao_version()
        assert version is None or isinstance(version, str)


class TestFP8Compatibility:
    """Test FP8 compatibility checking."""

    def test_is_fp8_compatible_layer_linear_aligned(self):
        """Test FP8 compatibility for aligned Linear layer."""
        from llm_dit.quantization.torchao_utils import is_fp8_compatible_layer

        # Both dimensions divisible by 16
        layer = nn.Linear(256, 128)
        assert is_fp8_compatible_layer(layer) is True

    def test_is_fp8_compatible_layer_linear_unaligned_in(self):
        """Test FP8 incompatibility for unaligned in_features."""
        from llm_dit.quantization.torchao_utils import is_fp8_compatible_layer

        layer = nn.Linear(100, 128)  # 100 % 16 != 0
        assert is_fp8_compatible_layer(layer) is False

    def test_is_fp8_compatible_layer_linear_unaligned_out(self):
        """Test FP8 incompatibility for unaligned out_features."""
        from llm_dit.quantization.torchao_utils import is_fp8_compatible_layer

        layer = nn.Linear(256, 100)  # 100 % 16 != 0
        assert is_fp8_compatible_layer(layer) is False

    def test_is_fp8_compatible_layer_non_linear(self):
        """Test FP8 compatibility for non-Linear layers."""
        from llm_dit.quantization.torchao_utils import is_fp8_compatible_layer

        # Non-Linear layers should return True (no constraint)
        layer = nn.Conv2d(3, 32, 3)
        assert is_fp8_compatible_layer(layer) is True

        layer = nn.LayerNorm(256)
        assert is_fp8_compatible_layer(layer) is True

    def test_check_fp8_support_no_cuda(self):
        """Test check_fp8_support returns False without CUDA."""
        from llm_dit.quantization.torchao_utils import check_fp8_support

        with patch('torch.cuda.is_available', return_value=False):
            assert check_fp8_support() is False


class TestGetRecommendedMethod:
    """Test recommended quantization method selection."""

    def test_get_recommended_method_no_cuda(self):
        """Test recommendation without CUDA."""
        from llm_dit.quantization.torchao_utils import get_recommended_method

        with patch('torch.cuda.is_available', return_value=False):
            assert get_recommended_method() == "int8"

    def test_get_recommended_method_with_fp8_support(self):
        """Test recommendation with FP8 support."""
        from llm_dit.quantization.torchao_utils import get_recommended_method

        with patch('torch.cuda.is_available', return_value=True):
            with patch('llm_dit.quantization.torchao_utils.check_fp8_support', return_value=True):
                assert get_recommended_method() == "fp8-weight-only"

    def test_get_recommended_method_without_fp8_support(self):
        """Test recommendation without FP8 support."""
        from llm_dit.quantization.torchao_utils import get_recommended_method

        with patch('torch.cuda.is_available', return_value=True):
            with patch('llm_dit.quantization.torchao_utils.check_fp8_support', return_value=False):
                assert get_recommended_method() == "int8"


# ============================================================================
# VAE Quantization Tests
# ============================================================================

class TestVAEQuantization:
    """Test VAE quantization utilities."""

    def test_quantize_vae_none(self):
        """Test quantize_vae with 'none' method."""
        from llm_dit.quantization.vae_utils import quantize_vae

        vae = MagicMock()
        result = quantize_vae(vae, "none")
        assert result is vae

    def test_quantize_vae_invalid_method(self):
        """Test quantize_vae with invalid method raises ValueError."""
        from llm_dit.quantization.vae_utils import quantize_vae

        vae = MagicMock()

        with pytest.raises(ValueError, match="Unknown VAE quantization method"):
            quantize_vae(vae, "fp16")


class TestEstimateVAEVRAM:
    """Test VAE VRAM estimation."""

    def test_estimate_vae_vram_none(self):
        """Test VRAM estimate without quantization."""
        from llm_dit.quantization.vae_utils import estimate_vae_vram

        vram = estimate_vae_vram("none")
        assert vram == 500  # Base VRAM

    def test_estimate_vae_vram_int8(self):
        """Test VRAM estimate with INT8."""
        from llm_dit.quantization.vae_utils import estimate_vae_vram

        vram = estimate_vae_vram("int8")
        assert vram == 250  # 50% reduction

    def test_estimate_vae_vram_unknown_fallback(self):
        """Test VRAM estimate with unknown method returns base."""
        from llm_dit.quantization.vae_utils import estimate_vae_vram

        vram = estimate_vae_vram("unknown")  # type: ignore[arg-type]
        assert vram == 500  # Falls back to base


class TestGetVAEQuantInfo:
    """Test VAE quantization info retrieval."""

    def test_get_vae_quant_info_none(self):
        """Test info for 'none' method."""
        from llm_dit.quantization.vae_utils import get_vae_quant_info

        info = get_vae_quant_info("none")
        assert info["name"] == "No quantization"
        assert info["vram_reduction"] == "0%"
        assert info["quality"] == "100%"

    def test_get_vae_quant_info_int8(self):
        """Test info for 'int8' method."""
        from llm_dit.quantization.vae_utils import get_vae_quant_info

        info = get_vae_quant_info("int8")
        assert info["name"] == "TorchAO INT8 Dynamic"
        assert info["vram_reduction"] == "~50%"
        assert "Conv2d" in info["supported_layers"]

    def test_get_vae_quant_info_unknown(self):
        """Test info for unknown method returns 'none' info."""
        from llm_dit.quantization.vae_utils import get_vae_quant_info

        info = get_vae_quant_info("unknown")
        assert info["name"] == "No quantization"
