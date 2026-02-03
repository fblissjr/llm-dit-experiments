"""
Unit tests for Qwen3 Unified Encoder.

Last Updated: 2026-02-03

Tests the unified Qwen3UnifiedEncoder that supports both Z-Image and FLUX.2 modes.

Run with: uv run pytest tests/unit/test_qwen3_unified_encoder.py -v
"""

import pytest
import torch
from unittest.mock import MagicMock, patch

pytestmark = pytest.mark.unit


class TestQwen3UnifiedEncoderImports:
    """Test imports."""

    def test_import_encoder(self):
        """Test basic import."""
        from llm_dit.encoders.qwen3_unified import Qwen3UnifiedEncoder
        assert Qwen3UnifiedEncoder is not None

    def test_import_config(self):
        """Test config import."""
        from llm_dit.encoders.qwen3_unified import Qwen3EncoderConfig
        assert Qwen3EncoderConfig is not None

    def test_import_presets(self):
        """Test preset imports."""
        from llm_dit.encoders.qwen3_unified import (
            ZIMAGE_CONFIG,
            KLEIN_4B_CONFIG,
            KLEIN_9B_CONFIG,
            PRESETS,
        )
        assert ZIMAGE_CONFIG is not None
        assert KLEIN_4B_CONFIG is not None
        assert KLEIN_9B_CONFIG is not None
        assert "zimage" in PRESETS
        assert "klein-4b" in PRESETS
        assert "klein-9b" in PRESETS

    def test_import_factory(self):
        """Test factory function import."""
        from llm_dit.encoders.qwen3_unified import get_unified_encoder
        assert get_unified_encoder is not None


class TestQwen3EncoderConfig:
    """Test Qwen3EncoderConfig dataclass."""

    def test_default_values(self):
        """Test default config values."""
        from llm_dit.encoders.qwen3_unified import Qwen3EncoderConfig

        config = Qwen3EncoderConfig()

        assert config.layer_indices == [-2]
        assert config.concat_mode == "none"
        assert config.enable_thinking is True
        assert config.max_length == 512
        assert config.pad_to_max is True

    def test_custom_values(self):
        """Test custom config values."""
        from llm_dit.encoders.qwen3_unified import Qwen3EncoderConfig

        config = Qwen3EncoderConfig(
            layer_indices=[9, 18, 27],
            concat_mode="concat",
            enable_thinking=False,
            max_length=1024,
        )

        assert config.layer_indices == [9, 18, 27]
        assert config.concat_mode == "concat"
        assert config.enable_thinking is False
        assert config.max_length == 1024

    def test_invalid_concat_mode_raises(self):
        """Test invalid concat_mode raises error."""
        from llm_dit.encoders.qwen3_unified import Qwen3EncoderConfig

        with pytest.raises(ValueError, match="Invalid concat_mode"):
            Qwen3EncoderConfig(concat_mode="invalid")

    def test_concat_requires_multiple_layers(self):
        """Test concat mode requires multiple layers."""
        from llm_dit.encoders.qwen3_unified import Qwen3EncoderConfig

        with pytest.raises(ValueError, match="requires multiple layers"):
            Qwen3EncoderConfig(
                layer_indices=[-2],  # Only one layer
                concat_mode="concat",
            )


class TestPresetConfigs:
    """Test preset configurations."""

    def test_zimage_preset(self):
        """Test Z-Image preset config."""
        from llm_dit.encoders.qwen3_unified import ZIMAGE_CONFIG

        assert ZIMAGE_CONFIG.layer_indices == [-2]
        assert ZIMAGE_CONFIG.concat_mode == "none"
        assert ZIMAGE_CONFIG.enable_thinking is True
        assert ZIMAGE_CONFIG.output_dim == 2560

    def test_klein_4b_preset(self):
        """Test Klein 4B preset config."""
        from llm_dit.encoders.qwen3_unified import KLEIN_4B_CONFIG

        assert KLEIN_4B_CONFIG.layer_indices == [9, 18, 27]
        assert KLEIN_4B_CONFIG.concat_mode == "concat"
        assert KLEIN_4B_CONFIG.enable_thinking is False
        assert KLEIN_4B_CONFIG.output_dim == 7680

    def test_klein_9b_preset(self):
        """Test Klein 9B preset config."""
        from llm_dit.encoders.qwen3_unified import KLEIN_9B_CONFIG

        assert KLEIN_9B_CONFIG.layer_indices == [9, 18, 27]
        assert KLEIN_9B_CONFIG.concat_mode == "concat"
        assert KLEIN_9B_CONFIG.enable_thinking is False
        assert KLEIN_9B_CONFIG.output_dim == 12288


class TestQwen3UnifiedEncoderInit:
    """Test Qwen3UnifiedEncoder initialization."""

    def test_init_stores_config(self):
        """Test initialization stores config."""
        from llm_dit.encoders.qwen3_unified import (
            Qwen3UnifiedEncoder,
            Qwen3EncoderConfig,
        )

        config = Qwen3EncoderConfig()
        encoder = Qwen3UnifiedEncoder(
            config=config,
            model_path="test-model",
            device="cpu",
        )

        assert encoder.config == config
        assert encoder.model_path == "test-model"
        assert encoder._target_device == torch.device("cpu")

    def test_lazy_loading(self):
        """Test model is not loaded until needed."""
        from llm_dit.encoders.qwen3_unified import (
            Qwen3UnifiedEncoder,
            Qwen3EncoderConfig,
        )

        config = Qwen3EncoderConfig()
        encoder = Qwen3UnifiedEncoder(
            config=config,
            model_path="test-model",
            device="cpu",
        )

        assert encoder._model is None
        assert encoder._tokenizer is None
        assert encoder._is_loaded is False


class TestFromPreset:
    """Test from_preset class method."""

    def test_unknown_preset_raises(self):
        """Test unknown preset raises error."""
        from llm_dit.encoders.qwen3_unified import Qwen3UnifiedEncoder

        with pytest.raises(ValueError, match="Unknown preset"):
            Qwen3UnifiedEncoder.from_preset("unknown")

    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoTokenizer")
    def test_zimage_preset_creates_encoder(self, mock_tokenizer_cls, mock_model_cls):
        """Test zimage preset creates encoder with correct config."""
        from llm_dit.encoders.qwen3_unified import Qwen3UnifiedEncoder, ZIMAGE_CONFIG

        # Setup mocks
        mock_model = MagicMock()
        mock_model.config.hidden_size = 2560
        mock_model.config.num_hidden_layers = 28
        mock_model_cls.from_pretrained.return_value = mock_model

        mock_tokenizer = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        encoder = Qwen3UnifiedEncoder.from_preset(
            "zimage",
            model_path="test-model",
            device="cpu",
        )

        assert encoder.config.layer_indices == ZIMAGE_CONFIG.layer_indices
        assert encoder.config.enable_thinking is True

    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoTokenizer")
    def test_klein_4b_preset_creates_encoder(self, mock_tokenizer_cls, mock_model_cls):
        """Test klein-4b preset creates encoder with correct config."""
        from llm_dit.encoders.qwen3_unified import Qwen3UnifiedEncoder, KLEIN_4B_CONFIG

        # Setup mocks
        mock_model = MagicMock()
        mock_model.config.hidden_size = 2560
        mock_model.config.num_hidden_layers = 28
        mock_model_cls.from_pretrained.return_value = mock_model

        mock_tokenizer = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        encoder = Qwen3UnifiedEncoder.from_preset(
            "klein-4b",
            model_path="test-model",
            device="cpu",
        )

        assert encoder.config.layer_indices == KLEIN_4B_CONFIG.layer_indices
        assert encoder.config.enable_thinking is False


class TestOutputDimensions:
    """Test output dimension calculations."""

    def test_single_layer_output_dim(self):
        """Test output dim for single layer mode."""
        from llm_dit.encoders.qwen3_unified import Qwen3UnifiedEncoder, Qwen3EncoderConfig

        config = Qwen3EncoderConfig(
            layer_indices=[-2],
            concat_mode="none",
            output_dim=2560,
        )
        encoder = Qwen3UnifiedEncoder(config=config, model_path="test", device="cpu")

        # Before loading, uses config.output_dim
        assert encoder.output_dim == 2560

    def test_concat_mode_output_dim(self):
        """Test output dim for concat mode."""
        from llm_dit.encoders.qwen3_unified import Qwen3UnifiedEncoder, Qwen3EncoderConfig

        config = Qwen3EncoderConfig(
            layer_indices=[9, 18, 27],
            concat_mode="concat",
            output_dim=7680,
        )
        encoder = Qwen3UnifiedEncoder(config=config, model_path="test", device="cpu")

        # Before loading, uses config.output_dim
        assert encoder.output_dim == 7680


class TestEnableThinking:
    """Test enable_thinking configuration."""

    def test_zimage_has_thinking_enabled(self):
        """Test Z-Image preset has thinking enabled."""
        from llm_dit.encoders.qwen3_unified import ZIMAGE_CONFIG

        assert ZIMAGE_CONFIG.enable_thinking is True

    def test_klein_has_thinking_disabled(self):
        """Test Klein presets have thinking disabled."""
        from llm_dit.encoders.qwen3_unified import KLEIN_4B_CONFIG, KLEIN_9B_CONFIG

        assert KLEIN_4B_CONFIG.enable_thinking is False
        assert KLEIN_9B_CONFIG.enable_thinking is False

    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoTokenizer")
    def test_thinking_flag_used_in_template(self, mock_tokenizer_cls, mock_model_cls):
        """Test enable_thinking is passed to chat template."""
        from llm_dit.encoders.qwen3_unified import Qwen3UnifiedEncoder

        # Setup mocks
        mock_model = MagicMock()
        mock_model.config.hidden_size = 2560
        mock_model.config.num_hidden_layers = 28
        mock_model.device = torch.device("cpu")
        mock_model_cls.from_pretrained.return_value = mock_model

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "formatted text"
        mock_tokenizer.return_value = {
            "input_ids": torch.ones(1, 10, dtype=torch.long),
            "attention_mask": torch.ones(1, 10),
        }
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        # Create encoder with Klein config (thinking=False)
        encoder = Qwen3UnifiedEncoder.from_preset(
            "klein-4b",
            model_path="test-model",
            device="cpu",
        )

        # Mock forward output
        mock_hidden_states = [torch.randn(1, 10, 2560) for _ in range(28)]
        mock_output = MagicMock()
        mock_output.hidden_states = mock_hidden_states
        mock_model.return_value = mock_output

        # Encode
        encoder.forward(["Test"])

        # Verify enable_thinking=False was used
        call_kwargs = mock_tokenizer.apply_chat_template.call_args[1]
        assert call_kwargs.get("enable_thinking") is False


class TestOffload:
    """Test offload functionality."""

    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoTokenizer")
    def test_offload_moves_to_cpu(self, mock_tokenizer_cls, mock_model_cls):
        """Test offload moves model to CPU."""
        from llm_dit.encoders.qwen3_unified import Qwen3UnifiedEncoder

        mock_model = MagicMock()
        mock_model.config.hidden_size = 2560
        mock_model.config.num_hidden_layers = 28
        mock_model_cls.from_pretrained.return_value = mock_model

        mock_tokenizer = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        encoder = Qwen3UnifiedEncoder.from_preset("zimage", model_path="test", device="cuda")
        encoder.offload()

        mock_model.to.assert_called_with("cpu")
        assert encoder._is_offloaded is True


class TestFactoryFunction:
    """Test get_unified_encoder factory function."""

    def test_unknown_preset_raises(self):
        """Test unknown preset raises error."""
        from llm_dit.encoders.qwen3_unified import get_unified_encoder

        with pytest.raises(ValueError, match="Unknown preset"):
            get_unified_encoder("unknown")

    @patch("llm_dit.encoders.qwen3_unified.Qwen3UnifiedEncoder.from_preset")
    def test_factory_calls_from_preset(self, mock_from_preset):
        """Test factory delegates to from_preset."""
        from llm_dit.encoders.qwen3_unified import get_unified_encoder

        mock_encoder = MagicMock()
        mock_from_preset.return_value = mock_encoder

        result = get_unified_encoder("zimage", model_path="test", device="cuda")

        mock_from_preset.assert_called_once_with(
            preset="zimage",
            model_path="test",
            device="cuda",
        )
        assert result == mock_encoder


class TestQwen3BaseMixin:
    """Test Qwen3EncoderMixin functionality."""

    def test_import_mixin(self):
        """Test mixin can be imported."""
        from llm_dit.encoders.qwen3_base import Qwen3EncoderMixin
        assert Qwen3EncoderMixin is not None

    def test_import_constants(self):
        """Test constants can be imported."""
        from llm_dit.encoders.qwen3_base import (
            QWEN3_4B_HIDDEN_DIM,
            QWEN3_8B_HIDDEN_DIM,
            ZIMAGE_DEFAULT_LAYER,
            KLEIN_DEFAULT_LAYERS,
        )
        assert QWEN3_4B_HIDDEN_DIM == 2560
        assert QWEN3_8B_HIDDEN_DIM == 4096
        assert ZIMAGE_DEFAULT_LAYER == -2
        assert KLEIN_DEFAULT_LAYERS == [9, 18, 27]
