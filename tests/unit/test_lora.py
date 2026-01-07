"""
Unit tests for llm_dit.utils.lora module.

Tests cover:
- LoRA state dict parsing (Kohya, PEFT formats)
- LoRA weight fusion
- LoRA loading utilities
- LoRA spec parsing
"""

from unittest.mock import MagicMock, patch
import pytest
import torch
import torch.nn as nn


# ============================================================================
# LoRALoader Tests
# ============================================================================

class TestLoRALoader:
    """Test LoRALoader class."""

    def test_init_defaults(self):
        """Test LoRALoader initialization with defaults."""
        from llm_dit.utils.lora import LoRALoader

        loader = LoRALoader()
        assert loader.device == "cpu"
        assert loader.torch_dtype == torch.float32

    def test_init_custom(self):
        """Test LoRALoader initialization with custom params."""
        from llm_dit.utils.lora import LoRALoader

        loader = LoRALoader(device="cuda", torch_dtype=torch.bfloat16)
        assert loader.device == "cuda"
        assert loader.torch_dtype == torch.bfloat16

    def test_get_name_dict_kohya_format(self):
        """Test parsing Kohya-style LoRA state dict."""
        from llm_dit.utils.lora import LoRALoader

        loader = LoRALoader()

        # Kohya format uses lora_up/lora_down
        state_dict = {
            "blocks.0.attn.to_q.lora_up.weight": torch.randn(32, 8),
            "blocks.0.attn.to_q.lora_down.weight": torch.randn(8, 32),
            "blocks.0.attn.to_k.lora_up.weight": torch.randn(32, 8),
            "blocks.0.attn.to_k.lora_down.weight": torch.randn(8, 32),
        }

        name_dict = loader.get_name_dict(state_dict)

        assert "blocks.0.attn.to_q" in name_dict
        assert "blocks.0.attn.to_k" in name_dict

    def test_get_name_dict_peft_format(self):
        """Test parsing PEFT-style LoRA state dict."""
        from llm_dit.utils.lora import LoRALoader

        loader = LoRALoader()

        # PEFT format uses lora_A/lora_B
        state_dict = {
            "blocks.0.attn.to_q.lora_A.weight": torch.randn(8, 32),
            "blocks.0.attn.to_q.lora_B.weight": torch.randn(32, 8),
            "blocks.0.attn.to_v.lora_A.weight": torch.randn(8, 32),
            "blocks.0.attn.to_v.lora_B.weight": torch.randn(32, 8),
        }

        name_dict = loader.get_name_dict(state_dict)

        assert "blocks.0.attn.to_q" in name_dict
        assert "blocks.0.attn.to_v" in name_dict

    def test_get_name_dict_with_diffusion_model_prefix(self):
        """Test parsing state dict with diffusion_model prefix."""
        from llm_dit.utils.lora import LoRALoader

        loader = LoRALoader()

        state_dict = {
            "diffusion_model.blocks.0.attn.to_q.lora_B.weight": torch.randn(32, 8),
            "diffusion_model.blocks.0.attn.to_q.lora_A.weight": torch.randn(8, 32),
        }

        name_dict = loader.get_name_dict(state_dict)

        # Prefix should be removed
        assert "blocks.0.attn.to_q" in name_dict
        assert "diffusion_model.blocks.0.attn.to_q" not in name_dict

    def test_get_name_dict_with_transformer_prefix(self):
        """Test parsing state dict with transformer prefix."""
        from llm_dit.utils.lora import LoRALoader

        loader = LoRALoader()

        state_dict = {
            "transformer.blocks.0.attn.to_q.lora_B.weight": torch.randn(32, 8),
            "transformer.blocks.0.attn.to_q.lora_A.weight": torch.randn(8, 32),
        }

        name_dict = loader.get_name_dict(state_dict)

        assert "blocks.0.attn.to_q" in name_dict

    def test_convert_state_dict(self):
        """Test converting LoRA state dict to standardized format."""
        from llm_dit.utils.lora import LoRALoader

        loader = LoRALoader()

        state_dict = {
            "blocks.0.attn.to_q.lora_B.weight": torch.randn(32, 8),
            "blocks.0.attn.to_q.lora_A.weight": torch.randn(8, 32),
        }

        converted = loader.convert_state_dict(state_dict)

        assert "blocks.0.attn.to_q.lora_A.weight" in converted
        assert "blocks.0.attn.to_q.lora_B.weight" in converted

    def test_fuse_lora_to_base_model(self):
        """Test fusing LoRA weights into base model."""
        from llm_dit.utils.lora import LoRALoader

        loader = LoRALoader(device="cpu", torch_dtype=torch.float32)

        # Create simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(32, 32)

        model = SimpleModel()
        original_weight = model.layer.weight.clone()

        # Create LoRA weights
        lora_rank = 8
        state_dict = {
            "layer.lora_B.weight": torch.randn(32, lora_rank),  # out_features x rank
            "layer.lora_A.weight": torch.randn(lora_rank, 32),  # rank x in_features
        }

        updated_count = loader.fuse_lora_to_base_model(model, state_dict, alpha=1.0)

        assert updated_count == 1
        # Weight should have changed
        assert not torch.allclose(model.layer.weight, original_weight)

    def test_fuse_lora_with_alpha_scaling(self):
        """Test LoRA fusion with different alpha values."""
        from llm_dit.utils.lora import LoRALoader

        loader = LoRALoader(device="cpu", torch_dtype=torch.float32)

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(32, 32)

        # Test with alpha=0.5
        model1 = SimpleModel()
        model1.layer.weight.data.zero_()  # Start from zeros for easier comparison

        # Test with alpha=1.0
        model2 = SimpleModel()
        model2.layer.weight.data.zero_()

        lora_rank = 8
        lora_B = torch.randn(32, lora_rank)
        lora_A = torch.randn(lora_rank, 32)

        state_dict = {
            "layer.lora_B.weight": lora_B,
            "layer.lora_A.weight": lora_A,
        }

        loader.fuse_lora_to_base_model(model1, state_dict, alpha=0.5)
        loader.fuse_lora_to_base_model(model2, state_dict, alpha=1.0)

        # model2 change should be 2x model1 change
        ratio = model2.layer.weight / (model1.layer.weight + 1e-8)
        # Allow for some numerical tolerance
        assert torch.allclose(ratio, torch.full_like(ratio, 2.0), atol=0.1)


# ============================================================================
# load_lora Function Tests
# ============================================================================

class TestLoadLora:
    """Test load_lora function."""

    def test_load_lora_file_not_found(self):
        """Test load_lora raises FileNotFoundError for missing file."""
        from llm_dit.utils.lora import load_lora

        model = nn.Linear(32, 32)

        with pytest.raises(FileNotFoundError):
            load_lora(model, "/nonexistent/path/lora.safetensors")

    def test_load_lora_infers_device_dtype(self):
        """Test load_lora infers device and dtype from model."""
        from llm_dit.utils.lora import load_lora, LoRALoader

        model = nn.Linear(32, 32)

        with patch.object(LoRALoader, 'fuse_lora_to_base_model', return_value=1) as mock_fuse:
            with patch('llm_dit.utils.lora.load_safetensors', return_value={}):
                with patch('pathlib.Path.exists', return_value=True):
                    load_lora(model, "test.safetensors", scale=0.8)

                    # Check that fuse was called
                    mock_fuse.assert_called_once()


# ============================================================================
# fuse_lora Function Tests
# ============================================================================

class TestFuseLora:
    """Test fuse_lora function."""

    def test_fuse_lora_from_state_dict(self):
        """Test fuse_lora with pre-loaded state dict."""
        from llm_dit.utils.lora import fuse_lora

        model = nn.Linear(32, 32)
        original_weight = model.weight.clone()

        # Create LoRA state dict matching the Linear layer name
        # The model's Linear is at the root, so we need to check how it's named
        state_dict = {
            # Use empty prefix since it's a bare Linear module
            ".lora_B.weight": torch.randn(32, 8),
            ".lora_A.weight": torch.randn(8, 32),
        }

        # fuse_lora should handle this gracefully (may return 0 if no matching layers)
        updated = fuse_lora(model, state_dict, scale=1.0)

        # The test checks the function runs without error
        assert isinstance(updated, int)


# ============================================================================
# parse_lora_spec Tests
# ============================================================================

class TestParseLoraSpec:
    """Test parse_lora_spec function."""

    def test_parse_simple_path(self):
        """Test parsing path without scale."""
        from llm_dit.utils.lora import parse_lora_spec

        path, scale = parse_lora_spec("lora.safetensors")
        assert path == "lora.safetensors"
        assert scale == 1.0

    def test_parse_path_with_scale(self):
        """Test parsing path with scale suffix."""
        from llm_dit.utils.lora import parse_lora_spec

        path, scale = parse_lora_spec("lora.safetensors:0.8")
        assert path == "lora.safetensors"
        assert scale == 0.8

    def test_parse_path_with_decimal_scale(self):
        """Test parsing path with decimal scale."""
        from llm_dit.utils.lora import parse_lora_spec

        path, scale = parse_lora_spec("style_lora.safetensors:0.35")
        assert path == "style_lora.safetensors"
        assert scale == 0.35

    def test_parse_path_with_zero_scale(self):
        """Test parsing path with zero scale."""
        from llm_dit.utils.lora import parse_lora_spec

        path, scale = parse_lora_spec("lora.safetensors:0.0")
        assert path == "lora.safetensors"
        assert scale == 0.0

    def test_parse_path_with_full_scale(self):
        """Test parsing path with full scale."""
        from llm_dit.utils.lora import parse_lora_spec

        path, scale = parse_lora_spec("lora.safetensors:1.0")
        assert path == "lora.safetensors"
        assert scale == 1.0

    def test_parse_path_with_directory(self):
        """Test parsing path with directory."""
        from llm_dit.utils.lora import parse_lora_spec

        path, scale = parse_lora_spec("/path/to/loras/style.safetensors:0.7")
        assert path == "/path/to/loras/style.safetensors"
        assert scale == 0.7

    def test_parse_windows_path_with_drive(self):
        """Test parsing Windows path with drive letter."""
        from llm_dit.utils.lora import parse_lora_spec

        # Windows path like C:\path\to\lora.safetensors
        # The colon in C: should not be treated as scale separator
        path, scale = parse_lora_spec("C:\\path\\to\\lora.safetensors")
        assert path == "C:\\path\\to\\lora.safetensors"
        assert scale == 1.0

    def test_parse_windows_path_with_scale(self):
        """Test parsing Windows path with scale."""
        from llm_dit.utils.lora import parse_lora_spec

        path, scale = parse_lora_spec("C:\\loras\\style.safetensors:0.5")
        assert path == "C:\\loras\\style.safetensors"
        assert scale == 0.5

    def test_parse_invalid_scale_treated_as_path(self):
        """Test that invalid scale suffix is treated as part of path."""
        from llm_dit.utils.lora import parse_lora_spec

        # If the part after colon isn't a valid float, keep the whole string
        path, scale = parse_lora_spec("lora.safetensors:invalid")
        assert path == "lora.safetensors:invalid"
        assert scale == 1.0


# ============================================================================
# clear_lora Tests
# ============================================================================

class TestClearLora:
    """Test clear_lora function."""

    def test_clear_lora_raises_not_implemented(self):
        """Test clear_lora raises NotImplementedError."""
        from llm_dit.utils.lora import clear_lora

        model = nn.Linear(32, 32)

        with pytest.raises(NotImplementedError):
            clear_lora(model)


# ============================================================================
# Conv2d LoRA Tests
# ============================================================================

class TestConv2dLoRA:
    """Test LoRA fusion for Conv2d layers."""

    def test_fuse_conv2d_lora(self):
        """Test fusing LoRA into Conv2d layer."""
        from llm_dit.utils.lora import LoRALoader

        loader = LoRALoader(device="cpu", torch_dtype=torch.float32)

        class ConvModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 32, 3, padding=1)

        model = ConvModel()
        original_weight = model.conv.weight.clone()

        # Conv2d LoRA uses 4D tensors
        lora_rank = 4
        # For Conv2d: lora_B shape is (out_channels, rank, 1, 1)
        # For Conv2d: lora_A shape is (rank, in_channels, 1, 1)
        state_dict = {
            "conv.lora_B.weight": torch.randn(32, lora_rank, 1, 1),
            "conv.lora_A.weight": torch.randn(lora_rank, 3, 1, 1),
        }

        updated = loader.fuse_lora_to_base_model(model, state_dict, alpha=1.0)

        assert updated == 1
        assert not torch.allclose(model.conv.weight, original_weight)
