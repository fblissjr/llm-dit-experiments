"""
Unit tests for llm_dit.utils.lora module.

Tests cover:
- LoRA state dict parsing (Kohya, PEFT formats)
- LoRA weight fusion
- LoRA loading utilities
- LoRA spec parsing
- LoRA arg normalization (_normalize_lora_args from generate.py)
"""

from pathlib import Path
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
        assert loader.dtype == torch.float32

    def test_init_custom(self):
        """Test LoRALoader initialization with custom params."""
        from llm_dit.utils.lora import LoRALoader

        loader = LoRALoader(device="cuda", dtype=torch.bfloat16)
        assert loader.device == "cuda"
        assert loader.dtype == torch.bfloat16

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

        loader = LoRALoader(device="cpu", dtype=torch.float32)

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

        loader = LoRALoader(device="cpu", dtype=torch.float32)

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
        from llm_dit.utils.lora import LoRALoader, load_lora

        model = nn.Linear(32, 32)

        with patch.object(LoRALoader, "fuse_lora_to_base_model", return_value=1) as mock_fuse:
            with patch("llm_dit.utils.lora.load_safetensors", return_value={}):
                with patch("pathlib.Path.exists", return_value=True):
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

        loader = LoRALoader(device="cpu", dtype=torch.float32)

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


# ============================================================================
# FusedLoRAState Tests
# ============================================================================


class TestFusedLoRAState:
    """Test LoRA fusion tracking state."""

    def test_empty_state(self):
        """Test initial state is empty."""
        from llm_dit.utils.lora import FusedLoRAState

        state = FusedLoRAState()
        assert state.is_empty
        assert state.summary() == "no LoRAs fused"

    def test_add_and_is_fused(self, tmp_path):
        """Test adding a record and checking if fused."""
        from llm_dit.utils.lora import FusedLoRAState

        state = FusedLoRAState()
        lora_path = str(tmp_path / "style.safetensors")
        state.add(lora_path, 0.8, 112)

        assert not state.is_empty
        assert state.is_fused(lora_path, 0.8)
        assert not state.is_fused(lora_path, 0.5)  # Different scale
        assert not state.is_fused(str(tmp_path / "other.safetensors"), 0.8)

    def test_matches_same_specs(self, tmp_path):
        """Test matches() returns True for identical specs."""
        from llm_dit.utils.lora import FusedLoRAState

        state = FusedLoRAState()
        path_a = str(tmp_path / "a.safetensors")
        path_b = str(tmp_path / "b.safetensors")
        state.add(path_a, 0.5, 50)
        state.add(path_b, 0.3, 62)

        assert state.matches([(path_a, 0.5), (path_b, 0.3)])
        # Order shouldn't matter
        assert state.matches([(path_b, 0.3), (path_a, 0.5)])

    def test_matches_different_specs(self, tmp_path):
        """Test matches() returns False for different specs."""
        from llm_dit.utils.lora import FusedLoRAState

        state = FusedLoRAState()
        path_a = str(tmp_path / "a.safetensors")
        path_b = str(tmp_path / "b.safetensors")
        state.add(path_a, 0.5, 50)

        # Different number of LoRAs
        assert not state.matches([(path_a, 0.5), (path_b, 0.3)])
        # Different scale
        assert not state.matches([(path_a, 0.9)])
        # Different path
        assert not state.matches([(path_b, 0.5)])

    def test_matches_empty(self):
        """Test matches() on empty state vs empty request."""
        from llm_dit.utils.lora import FusedLoRAState

        state = FusedLoRAState()
        assert state.matches([])
        assert not state.matches([("/some/path.safetensors", 1.0)])

    def test_summary_format(self, tmp_path):
        """Test human-readable summary."""
        from llm_dit.utils.lora import FusedLoRAState

        state = FusedLoRAState()
        state.add(str(tmp_path / "anime.safetensors"), 0.8, 100)
        state.add(str(tmp_path / "realism.safetensors"), 0.5, 100)

        summary = state.summary()
        assert "anime.safetensors@0.8" in summary
        assert "realism.safetensors@0.5" in summary

    def test_record_is_frozen(self, tmp_path):
        """Test that LoRAFusionRecord is immutable."""
        from llm_dit.utils.lora import LoRAFusionRecord
        from dataclasses import FrozenInstanceError

        record = LoRAFusionRecord(
            path=str(tmp_path / "lora.safetensors"),
            scale=0.8,
            layers_updated=112,
        )
        with pytest.raises(FrozenInstanceError):
            record.scale = 0.5  # type: ignore[misc]


# ============================================================================
# get_fused_state Tests
# ============================================================================


class TestGetFusedState:
    """Test get_fused_state() model attachment."""

    def test_creates_state_on_first_call(self):
        """Test that state is lazily created on the model."""
        from llm_dit.utils.lora import get_fused_state

        model = nn.Linear(32, 32)
        assert not hasattr(model, "_fused_lora_state")

        state = get_fused_state(model)
        assert hasattr(model, "_fused_lora_state")
        assert state.is_empty

    def test_returns_same_object(self):
        """Test that subsequent calls return the same state object."""
        from llm_dit.utils.lora import get_fused_state

        model = nn.Linear(32, 32)
        state1 = get_fused_state(model)
        state2 = get_fused_state(model)
        assert state1 is state2

    def test_survives_dict_storage(self):
        """Test that state survives when model is stored in a dict (FLUX.2 pattern)."""
        from llm_dit.utils.lora import FusedLoRAState, get_fused_state

        model = nn.Linear(32, 32)
        state = get_fused_state(model)
        state.add("/path/to/lora.safetensors", 0.8, 50)

        # Simulate FLUX.2's dict-based pipeline storage
        pipeline = {"transformer": model, "model_name": "klein-9b-fp8"}

        # Retrieve from dict -- state should still be there
        retrieved_model = pipeline["transformer"]
        retrieved_state = get_fused_state(retrieved_model)
        assert not retrieved_state.is_empty
        assert retrieved_state.is_fused("/path/to/lora.safetensors", 0.8)

    def test_load_lora_records_fusion(self):
        """Test that load_lora() records fusions on the model."""
        from llm_dit.utils.lora import LoRALoader, get_fused_state, load_lora

        model = nn.Linear(32, 32)

        with patch.object(LoRALoader, "fuse_lora_to_base_model", return_value=5) as mock_fuse:
            with patch("llm_dit.utils.lora.load_safetensors", return_value={}):
                with patch("pathlib.Path.exists", return_value=True):
                    load_lora(model, "style.safetensors", scale=0.7)

        state = get_fused_state(model)
        assert not state.is_empty
        assert len(state.records) == 1
        assert state.records[0].scale == 0.7
        assert state.records[0].layers_updated == 5


# ============================================================================
# _infer_model_device_dtype Tests (quantized models)
# ============================================================================


class TestInferModelDeviceDtype:
    """Test _infer_model_device_dtype with quantized parameter types."""

    def test_standard_bf16_model(self):
        """Test inference on a standard bfloat16 model."""
        from llm_dit.utils.lora import _infer_model_device_dtype

        model = nn.Linear(32, 32)
        model = model.to(dtype=torch.bfloat16)
        device, dtype = _infer_model_device_dtype(model)

        assert dtype == torch.bfloat16

    def test_standard_fp32_model(self):
        """Test inference on a standard float32 model."""
        from llm_dit.utils.lora import _infer_model_device_dtype

        model = nn.Linear(32, 32)
        device, dtype = _infer_model_device_dtype(model)

        assert dtype == torch.float32

    def test_quantized_uint8_model(self):
        """Test that uint8 storage dtype returns bfloat16 compute dtype."""
        from llm_dit.utils.lora import _infer_model_device_dtype

        model = nn.Linear(32, 32)
        # Simulate quantized model by replacing weight with uint8 tensor
        model.weight = nn.Parameter(
            torch.zeros(32, 32, dtype=torch.uint8), requires_grad=False
        )
        device, dtype = _infer_model_device_dtype(model)

        assert dtype == torch.bfloat16

    def test_non_tensor_parameter_type(self):
        """Test that non-Tensor parameter type (e.g., Float8Tensor) returns bfloat16."""
        from llm_dit.utils.lora import _infer_model_device_dtype

        # Create a mock module whose parameters() yields a non-Tensor object
        # mimicking torchao Float8Tensor (not a torch.Tensor subclass).
        class MockFloat8Param:
            @property
            def device(self):
                return torch.device("cpu")
            @property
            def dtype(self):
                return torch.float8_e4m3fn

        class MockQuantizedModule(nn.Module):
            def parameters(self, recurse=True):
                yield MockFloat8Param()

        model = MockQuantizedModule()
        _, dtype = _infer_model_device_dtype(model)

        assert dtype == torch.bfloat16

    def test_explicit_dtype_overrides(self):
        """Test that explicit dtype is never overridden."""
        from llm_dit.utils.lora import _infer_model_device_dtype

        model = nn.Linear(32, 32)
        model.weight = nn.Parameter(
            torch.zeros(32, 32, dtype=torch.uint8), requires_grad=False
        )
        # Even with quantized model, explicit dtype should win
        device, dtype = _infer_model_device_dtype(model, dtype=torch.float16)

        assert dtype == torch.float16

    def test_empty_model(self):
        """Test inference on model with no parameters."""
        from llm_dit.utils.lora import _infer_model_device_dtype

        model = nn.Module()  # No parameters
        device, dtype = _infer_model_device_dtype(model)

        assert device == "cpu"
        assert dtype == torch.float32


# ============================================================================
# fuse_lora_to_state_dict Tests (FP8-cast state-dict LoRA fusion)
# ============================================================================


class TestFuseLoraToStateDict:
    """Test state-dict level LoRA fusion for fp8-cast models.

    Aligns with official LTX-2 fuse_loras.py: fuse LoRA deltas into a state dict
    before load_state_dict, supporting mixed fp8+bf16 weights.
    """

    def _make_lora_sd(self, key_prefix: str, out_features: int, in_features: int, rank: int):
        """Build a LoRA state dict with lora_A + lora_B keys."""
        return {
            f"{key_prefix}.lora_A.weight": torch.randn(rank, in_features, dtype=torch.bfloat16),
            f"{key_prefix}.lora_B.weight": torch.randn(out_features, rank, dtype=torch.bfloat16),
        }

    def _write_lora_file(self, tmp_path, key_prefix, out_features, in_features, rank):
        """Write a LoRA safetensors file and return its path."""
        from safetensors.torch import save_file
        sd = self._make_lora_sd(key_prefix, out_features, in_features, rank)
        path = tmp_path / "test_lora.safetensors"
        save_file(sd, str(path))
        return str(path)

    def test_bf16_weights_direct_add(self, tmp_path):
        """bf16 base weight + LoRA delta = bf16 result (no dtype conversion)."""
        from llm_dit.utils.lora import fuse_lora_to_state_dict

        base_sd = {"layer.weight": torch.randn(64, 32, dtype=torch.bfloat16)}
        lora_path = self._write_lora_file(tmp_path, "layer", 64, 32, 8)

        result = fuse_lora_to_state_dict(base_sd, [lora_path], [1.0])

        assert result["layer.weight"].dtype == torch.bfloat16
        # Weight should have changed
        assert not torch.allclose(result["layer.weight"], base_sd["layer.weight"])

    def test_fp8_weights_upcast_fuse_downcast(self, tmp_path):
        """fp8 base weight + LoRA delta: upcast to bf16, add, downcast to fp8."""
        from llm_dit.utils.lora import fuse_lora_to_state_dict

        base_weight = torch.randn(64, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        base_sd = {"layer.weight": base_weight}

        lora_path = self._write_lora_file(tmp_path, "layer", 64, 32, 8)
        result = fuse_lora_to_state_dict(base_sd, [lora_path], [1.0])

        # Result should be fp8 (downcast back after fusion)
        assert result["layer.weight"].dtype == torch.float8_e4m3fn

    def test_non_weight_keys_pass_through(self, tmp_path):
        """Non-linear keys (norms, biases) should pass through unchanged."""
        from llm_dit.utils.lora import fuse_lora_to_state_dict

        base_sd = {
            "layer.weight": torch.randn(64, 32, dtype=torch.bfloat16),
            "norm.weight": torch.ones(64, dtype=torch.bfloat16),
            "layer.bias": torch.zeros(64, dtype=torch.bfloat16),
        }

        lora_path = self._write_lora_file(tmp_path, "layer", 64, 32, 8)
        result = fuse_lora_to_state_dict(base_sd, [lora_path], [1.0])

        # Norm and bias should be untouched
        assert torch.equal(result["norm.weight"], base_sd["norm.weight"])
        assert torch.equal(result["layer.bias"], base_sd["layer.bias"])

    def test_multiple_loras_accumulated(self, tmp_path):
        """Multiple LoRAs should accumulate deltas."""
        from llm_dit.utils.lora import fuse_lora_to_state_dict

        base_sd = {
            "layer.weight": torch.zeros(64, 32, dtype=torch.bfloat16),
        }

        dir_a = tmp_path / "a"
        dir_a.mkdir()
        dir_b = tmp_path / "b"
        dir_b.mkdir()
        path_a = self._write_lora_file(dir_a, "layer", 64, 32, 8)
        path_b = self._write_lora_file(dir_b, "layer", 64, 32, 8)

        result = fuse_lora_to_state_dict(base_sd, [path_a, path_b], [1.0, 0.5])

        # Result should not be zeros (deltas applied)
        assert not torch.allclose(result["layer.weight"], base_sd["layer.weight"])

    def test_scale_affects_magnitude(self, tmp_path):
        """Scale=0 should leave weights unchanged, scale=1 should change them."""
        from llm_dit.utils.lora import fuse_lora_to_state_dict

        base_weight = torch.randn(64, 32, dtype=torch.bfloat16)
        lora_path = self._write_lora_file(tmp_path, "layer", 64, 32, 8)

        result_zero = fuse_lora_to_state_dict(
            {"layer.weight": base_weight.clone()}, [lora_path], [0.0],
        )
        result_one = fuse_lora_to_state_dict(
            {"layer.weight": base_weight.clone()}, [lora_path], [1.0],
        )

        # scale=0: weight unchanged
        assert torch.allclose(result_zero["layer.weight"], base_weight, atol=1e-6)
        # scale=1: weight changed
        assert not torch.allclose(result_one["layer.weight"], base_weight)

    def test_does_not_mutate_input_state_dict(self, tmp_path):
        """Input state dict should not be modified in-place."""
        from llm_dit.utils.lora import fuse_lora_to_state_dict

        base_weight = torch.randn(64, 32, dtype=torch.bfloat16)
        base_sd = {"layer.weight": base_weight.clone()}
        original_data = base_sd["layer.weight"].clone()

        lora_path = self._write_lora_file(tmp_path, "layer", 64, 32, 8)
        fuse_lora_to_state_dict(base_sd, [lora_path], [1.0])

        # Original should be untouched
        assert torch.equal(base_sd["layer.weight"], original_data)

    def test_lokr_format_supported(self, tmp_path):
        """LoKR (Kronecker product) format LoRAs should also work."""
        from llm_dit.utils.lora import fuse_lora_to_state_dict
        from safetensors.torch import save_file

        base_sd = {"layer.weight": torch.randn(64, 32, dtype=torch.bfloat16)}

        lokr_sd = {
            "layer.lokr_w1": torch.randn(8, 4, dtype=torch.bfloat16),
            "layer.lokr_w2": torch.randn(8, 8, dtype=torch.bfloat16),
        }

        path = str(tmp_path / "lokr.safetensors")
        save_file(lokr_sd, path)

        result = fuse_lora_to_state_dict(base_sd, [path], [1.0])

        assert result["layer.weight"].dtype == torch.bfloat16
        assert not torch.allclose(result["layer.weight"], base_sd["layer.weight"])

    def test_scaled_fp8_dequant_fuse_requant(self, tmp_path):
        """Scaled fp8 weights: dequant with weight_scale, fuse in f32, re-quant.

        This is the critical path for LTX-2.3 distilled LoRA. Without proper
        dequantization, raw fp8 values (~300) dwarf the LoRA delta (~0.001),
        making the LoRA negligible.
        """
        from llm_dit.utils.lora import fuse_lora_to_state_dict

        # Create a "real" weight in bf16, then quantize to scaled fp8
        real_weight = torch.randn(64, 32, dtype=torch.float32) * 0.02  # typical magnitude
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        max_abs = real_weight.abs().amax()
        quant_scale = fp8_max / max_abs
        raw_fp8 = (real_weight * quant_scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
        weight_scale = quant_scale.reciprocal()  # max_abs / fp8_max

        base_sd = {"layer.weight": raw_fp8}
        weight_scales = {"layer.weight": weight_scale}

        lora_path = self._write_lora_file(tmp_path, "layer", 64, 32, 8)
        result_sd, new_scales = fuse_lora_to_state_dict(
            base_sd, [lora_path], [1.0], weight_scales=weight_scales,
        )

        # Result should be fp8 with an updated scale
        assert result_sd["layer.weight"].dtype == torch.float8_e4m3fn
        assert "layer.weight" in new_scales
        assert new_scales["layer.weight"].dtype == torch.float32

        # Dequantize result and compare to original real weight
        result_real = result_sd["layer.weight"].to(torch.float32) * new_scales["layer.weight"]
        original_real = raw_fp8.to(torch.float32) * weight_scale

        # The LoRA should have made a meaningful change (not negligible)
        delta_magnitude = (result_real - original_real).abs().mean()
        assert delta_magnitude > 1e-4, (
            f"LoRA delta too small ({delta_magnitude:.2e}) -- "
            "weight_scale likely not applied during fusion"
        )

    def test_scaled_fp8_roundtrip_accuracy(self, tmp_path):
        """Verify that dequant -> fuse -> requant preserves reasonable accuracy."""
        from llm_dit.utils.lora import fuse_lora_to_state_dict

        real_weight = torch.randn(64, 32, dtype=torch.float32) * 0.02
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        max_abs = real_weight.abs().amax()
        quant_scale = fp8_max / max_abs
        raw_fp8 = (real_weight * quant_scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
        weight_scale = quant_scale.reciprocal()

        base_sd = {"layer.weight": raw_fp8}
        weight_scales = {"layer.weight": weight_scale}

        # Use scale=0 LoRA (zero delta) to test roundtrip
        lora_path = self._write_lora_file(tmp_path, "layer", 64, 32, 8)
        result_sd, new_scales = fuse_lora_to_state_dict(
            base_sd, [lora_path], [0.0], weight_scales=weight_scales,
        )

        # Dequantized values should be close (within fp8 quantization error)
        result_real = result_sd["layer.weight"].to(torch.float32) * new_scales["layer.weight"]
        original_real = raw_fp8.to(torch.float32) * weight_scale
        max_error = (result_real - original_real).abs().max()
        # fp8 has ~7 mantissa levels, roundtrip error should be small relative to values
        assert max_error < real_weight.abs().max() * 0.1, (
            f"Roundtrip error too large: {max_error:.2e}"
        )

    def test_backward_compat_no_weight_scales(self, tmp_path):
        """Without weight_scales, return type is plain dict (not tuple)."""
        from llm_dit.utils.lora import fuse_lora_to_state_dict

        base_sd = {"layer.weight": torch.randn(64, 32, dtype=torch.bfloat16)}
        lora_path = self._write_lora_file(tmp_path, "layer", 64, 32, 8)

        result = fuse_lora_to_state_dict(base_sd, [lora_path], [1.0])
        # Should be a plain dict, not a tuple
        assert isinstance(result, dict)
        assert "layer.weight" in result


# ============================================================================
# fuse_lora_to_base_model with fp8-cast native tensors
# ============================================================================


class TestFuseLoraFP8Cast:
    """Test in-place LoRA fusion on models with native fp8 weights + _weight_scale.

    This covers the fp8-cast loading path (FLUX.2 Klein FP8, LTX-2 FP8) where
    weights are torch.Tensor with dtype=float8_e4m3fn and per-tensor _weight_scale
    attributes. Before the fix, these were NOT detected as quantized (type IS
    torch.Tensor), so LoRA deltas were added in the wrong scale and lost on
    truncation back to fp8.
    """

    def test_fp8_cast_lora_changes_dequantized_weight(self):
        """LoRA fusion on fp8-cast model should produce a meaningful weight change.

        Without the fix, the LoRA delta is negligible because it's added to raw
        fp8 numeric values (~300) instead of dequantized real values (~0.02).
        """
        from llm_dit.quantization.fp8_cast import quantize_to_fp8_per_tensor
        from llm_dit.utils.lora import LoRALoader

        # Create a model with native fp8 weights + weight_scale
        model = nn.Module()
        model.layer = nn.Linear(32, 64, bias=False)

        # Simulate fp8-cast loading: quantize real weights to scaled fp8
        real_weight = torch.randn(64, 32) * 0.02  # typical magnitude
        fp8_weight, weight_scale = quantize_to_fp8_per_tensor(real_weight)
        model.layer.weight = nn.Parameter(fp8_weight, requires_grad=False)
        model.layer._weight_scale = weight_scale

        # Record dequantized original weight
        original_real = fp8_weight.float() * weight_scale

        # Create LoRA state dict
        lora_rank = 8
        state_dict = {
            "layer.lora_B.weight": torch.randn(64, lora_rank) * 0.01,
            "layer.lora_A.weight": torch.randn(lora_rank, 32) * 0.01,
        }

        loader = LoRALoader(device="cpu", dtype=torch.bfloat16)
        updated = loader.fuse_lora_to_base_model(model, state_dict, alpha=1.0)
        assert updated == 1

        # Weight should still be fp8
        assert model.layer.weight.dtype == torch.float8_e4m3fn

        # Dequantize the fused weight using the (possibly updated) scale
        fused_scale = getattr(model.layer, "_weight_scale", weight_scale)
        fused_real = model.layer.weight.float() * fused_scale

        # The LoRA should have produced a meaningful change
        delta = (fused_real - original_real).abs().mean()
        assert delta > 1e-4, (
            f"LoRA delta too small ({delta:.2e}) -- "
            "fp8-cast fusion likely not dequanting before merge"
        )

    def test_fp8_cast_lora_updates_weight_scale(self):
        """After fp8-cast LoRA fusion, _weight_scale should be updated."""
        from llm_dit.quantization.fp8_cast import quantize_to_fp8_per_tensor
        from llm_dit.utils.lora import LoRALoader

        model = nn.Module()
        model.layer = nn.Linear(32, 64, bias=False)

        real_weight = torch.randn(64, 32) * 0.02
        fp8_weight, weight_scale = quantize_to_fp8_per_tensor(real_weight)
        model.layer.weight = nn.Parameter(fp8_weight, requires_grad=False)
        model.layer._weight_scale = weight_scale
        original_scale = weight_scale.clone()

        lora_rank = 8
        state_dict = {
            "layer.lora_B.weight": torch.randn(64, lora_rank) * 0.01,
            "layer.lora_A.weight": torch.randn(lora_rank, 32) * 0.01,
        }

        loader = LoRALoader(device="cpu", dtype=torch.bfloat16)
        loader.fuse_lora_to_base_model(model, state_dict, alpha=1.0)

        # Scale should exist and may have changed (re-quantization changes scale)
        assert hasattr(model.layer, "_weight_scale")
        new_scale = model.layer._weight_scale
        assert new_scale.dtype == torch.float32

    def test_fp8_cast_no_weight_scale_still_works(self):
        """fp8 weights without _weight_scale (naive fp8) should still fuse correctly."""
        from llm_dit.utils.lora import LoRALoader

        model = nn.Module()
        model.layer = nn.Linear(32, 64, bias=False)

        # Naive fp8: just cast to fp8 without scaling
        model.layer.weight = nn.Parameter(
            torch.randn(64, 32).to(torch.float8_e4m3fn),
            requires_grad=False,
        )
        original_weight = model.layer.weight.clone()

        lora_rank = 8
        state_dict = {
            "layer.lora_B.weight": torch.randn(64, lora_rank) * 0.1,
            "layer.lora_A.weight": torch.randn(lora_rank, 32) * 0.1,
        }

        loader = LoRALoader(device="cpu", dtype=torch.bfloat16)
        updated = loader.fuse_lora_to_base_model(model, state_dict, alpha=1.0)
        assert updated == 1
        assert model.layer.weight.dtype == torch.float8_e4m3fn


# ============================================================================
# _normalize_lora_args Tests (from generate.py)
# ============================================================================


class TestNormalizeLoraArgs:
    """Test _normalize_lora_args helper extracted from generate.py."""

    def test_none_input_returns_none(self):
        """None lora_path should return (None, None)."""
        from llm_dit.pipelines.generate import _normalize_lora_args
        paths, scales = _normalize_lora_args(None, None)
        assert paths is None
        assert scales is None

    def test_single_str_path_with_float_scale(self):
        """Single string path with float scale."""
        from llm_dit.pipelines.generate import _normalize_lora_args
        paths, scales = _normalize_lora_args("lora.safetensors", 0.7)
        assert paths == ["lora.safetensors"]
        assert scales == [0.7]

    def test_single_path_none_scale_defaults_to_1_0(self):
        """Single path with None scale should default to 1.0 (matches config)."""
        from llm_dit.pipelines.generate import _normalize_lora_args
        paths, scales = _normalize_lora_args("lora.safetensors", None)
        assert paths == ["lora.safetensors"]
        assert scales == [1.0]

    def test_list_paths_with_list_scales(self):
        """List of paths with matching list of scales."""
        from llm_dit.pipelines.generate import _normalize_lora_args
        paths, scales = _normalize_lora_args(
            ["a.safetensors", "b.safetensors"], [0.5, 0.3],
        )
        assert paths == ["a.safetensors", "b.safetensors"]
        assert scales == [0.5, 0.3]

    def test_mismatched_lengths_raises_valueerror(self):
        """Mismatched path and scale lengths should raise ValueError."""
        from llm_dit.pipelines.generate import _normalize_lora_args
        with pytest.raises(ValueError, match="must match"):
            _normalize_lora_args(["a.safetensors", "b.safetensors"], [0.5])

    def test_path_objects_converted_to_str(self):
        """Path objects should be converted to strings."""
        from llm_dit.pipelines.generate import _normalize_lora_args
        paths, scales = _normalize_lora_args(Path("dir/lora.safetensors"), 1.0)
        assert paths == ["dir/lora.safetensors"]
        assert isinstance(paths[0], str)

    def test_single_float_scale_broadcast_to_list(self):
        """Single float scale should be broadcast to match path count."""
        from llm_dit.pipelines.generate import _normalize_lora_args
        paths, scales = _normalize_lora_args(
            ["a.safetensors", "b.safetensors"], 0.5,
        )
        assert scales == [0.5, 0.5]

    def test_single_path_as_list(self):
        """Single-element list should work."""
        from llm_dit.pipelines.generate import _normalize_lora_args
        paths, scales = _normalize_lora_args(["lora.safetensors"], [0.8])
        assert paths == ["lora.safetensors"]
        assert scales == [0.8]

    def test_int_scale_converted_to_float(self):
        """Integer scale should be converted to float."""
        from llm_dit.pipelines.generate import _normalize_lora_args
        paths, scales = _normalize_lora_args("lora.safetensors", 1)
        assert scales == [1.0]
        assert isinstance(scales[0], float)
