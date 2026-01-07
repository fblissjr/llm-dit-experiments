"""
Unit tests for llm_dit.quantization module.

Tests cover:
- TorchAO quantization utilities
- VAE quantization utilities
- FP8 compatibility checking
"""

from unittest.mock import MagicMock, patch
import pytest
import torch
import torch.nn as nn


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


class TestCreateFP8FilterFn:
    """Test FP8 filter function creation."""

    def test_create_fp8_filter_fn_basic(self):
        """Test basic filter function creation."""
        from llm_dit.quantization.torchao_utils import create_fp8_filter_fn

        filter_fn = create_fp8_filter_fn(skip_incompatible=True, verbose=False)
        assert callable(filter_fn)

    def test_filter_fn_returns_false_for_non_linear(self):
        """Test filter function returns False for non-Linear modules."""
        from llm_dit.quantization.torchao_utils import create_fp8_filter_fn

        filter_fn = create_fp8_filter_fn(skip_incompatible=True, verbose=False)

        # Non-Linear modules should return False (recurse into children)
        conv = nn.Conv2d(3, 32, 3)
        assert filter_fn(conv, "model.conv") is False

        norm = nn.LayerNorm(256)
        assert filter_fn(norm, "model.norm") is False

    def test_filter_fn_returns_true_for_compatible_linear(self):
        """Test filter function returns True for compatible Linear."""
        from llm_dit.quantization.torchao_utils import create_fp8_filter_fn

        filter_fn = create_fp8_filter_fn(skip_incompatible=True, verbose=False)

        linear = nn.Linear(256, 128)  # 16-aligned
        assert filter_fn(linear, "model.linear") is True

    def test_filter_fn_returns_false_for_incompatible_linear(self):
        """Test filter function returns False for incompatible Linear."""
        from llm_dit.quantization.torchao_utils import create_fp8_filter_fn

        filter_fn = create_fp8_filter_fn(skip_incompatible=True, verbose=False)

        linear = nn.Linear(100, 128)  # 100 not divisible by 16
        assert filter_fn(linear, "model.linear") is False


class TestAnalyzeFP8Compatibility:
    """Test FP8 compatibility analysis."""

    def test_analyze_fp8_compatibility(self):
        """Test analyze_fp8_compatibility returns correct stats."""
        from llm_dit.quantization.torchao_utils import analyze_fp8_compatibility

        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.compatible1 = nn.Linear(256, 128)
                self.compatible2 = nn.Linear(512, 256)
                self.incompatible = nn.Linear(100, 50)

        model = TestModel()
        analysis = analyze_fp8_compatibility(model)

        assert analysis["total_linear_layers"] == 3
        assert analysis["compatible_layers"] == 2
        assert analysis["incompatible_layers"] == 1
        assert len(analysis["incompatible_layer_info"]) == 1
        assert analysis["incompatible_layer_info"][0]["name"] == "incompatible"

    def test_analyze_fp8_compatibility_all_compatible(self):
        """Test analysis with all compatible layers."""
        from llm_dit.quantization.torchao_utils import analyze_fp8_compatibility

        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(256, 128)
                self.layer2 = nn.Linear(128, 64)

        model = TestModel()
        analysis = analyze_fp8_compatibility(model)

        assert analysis["total_linear_layers"] == 2
        assert analysis["compatible_layers"] == 2
        assert analysis["incompatible_layers"] == 0
        assert analysis["compatibility_rate"] == 100.0

    def test_analyze_fp8_compatibility_empty_model(self):
        """Test analysis with no Linear layers."""
        from llm_dit.quantization.torchao_utils import analyze_fp8_compatibility

        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 32, 3)

        model = TestModel()
        analysis = analyze_fp8_compatibility(model)

        assert analysis["total_linear_layers"] == 0
        assert analysis["compatibility_rate"] == 0


class TestGetRecommendedMethod:
    """Test recommended quantization method selection."""

    def test_get_recommended_method_no_cuda(self):
        """Test recommendation without CUDA."""
        from llm_dit.quantization.torchao_utils import get_recommended_method

        with patch('torch.cuda.is_available', return_value=False):
            assert get_recommended_method() == "8bit"

    def test_get_recommended_method_with_fp8_support(self):
        """Test recommendation with FP8 support."""
        from llm_dit.quantization.torchao_utils import get_recommended_method

        with patch('torch.cuda.is_available', return_value=True):
            with patch('llm_dit.quantization.torchao_utils.check_fp8_support', return_value=True):
                assert get_recommended_method() == "fp8"

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

    def test_quantize_vae_8bit_warning(self):
        """Test quantize_vae with '8bit' method logs warning."""
        from llm_dit.quantization.vae_utils import quantize_vae

        vae = MagicMock()

        with patch('llm_dit.quantization.vae_utils.logger') as mock_logger:
            result = quantize_vae(vae, "8bit")
            mock_logger.warning.assert_called_once()
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

    def test_estimate_vae_vram_8bit(self):
        """Test VRAM estimate with 8bit."""
        from llm_dit.quantization.vae_utils import estimate_vae_vram

        vram = estimate_vae_vram("8bit")
        assert vram == 250


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
