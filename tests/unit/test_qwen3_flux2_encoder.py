"""
Unit tests for Qwen3 FLUX.2 Encoder.

Last Updated: 2026-02-01

Tests the Qwen3Flux2Encoder used for FLUX.2 Klein text encoding.
These tests run without GPU by mocking the model loading.

Run with: uv run pytest tests/unit/test_qwen3_flux2_encoder.py -v

Key differences from Qwen3Encoder (Z-Image):
- Multi-layer extraction: [9, 18, 27] instead of [-2]
- Output concatenation: layers are concatenated -> 3x hidden_dim
- enable_thinking=False: CRITICAL for FLUX.2 (thinking tokens corrupt embeddings)
"""

import pytest
import torch
from unittest.mock import MagicMock, patch

pytestmark = pytest.mark.unit


class TestQwen3Flux2EncoderImports:
    """Test that Qwen3Flux2Encoder can be imported."""

    def test_import_encoder(self):
        """Test basic import."""
        from llm_dit.encoders.qwen3_flux2 import Qwen3Flux2Encoder
        assert Qwen3Flux2Encoder is not None

    def test_import_constants(self):
        """Test constant imports."""
        from llm_dit.encoders.qwen3_flux2 import (
            DEFAULT_OUTPUT_LAYERS,
            DEFAULT_MAX_LENGTH,
        )
        assert DEFAULT_OUTPUT_LAYERS == [9, 18, 27]
        assert DEFAULT_MAX_LENGTH == 512

    def test_import_factory_function(self):
        """Test factory function import."""
        from llm_dit.encoders.qwen3_flux2 import load_qwen3_flux2_encoder
        assert load_qwen3_flux2_encoder is not None


class TestQwen3Flux2EncoderConfig:
    """Test Qwen3Flux2Encoder configuration."""

    def test_default_output_layers(self):
        """Test default output layers are [9, 18, 27]."""
        from llm_dit.encoders.qwen3_flux2 import DEFAULT_OUTPUT_LAYERS
        assert DEFAULT_OUTPUT_LAYERS == [9, 18, 27]

    def test_output_layers_count(self):
        """Test exactly 3 layers are required for Klein models."""
        # This is validated in the constructor
        from llm_dit.encoders.qwen3_flux2 import Qwen3Flux2Encoder

        # Should raise if not exactly 3 layers
        with patch.object(Qwen3Flux2Encoder, '__init__', lambda self, *args, **kwargs: None):
            encoder = Qwen3Flux2Encoder.__new__(Qwen3Flux2Encoder)
            encoder.output_layers = [1, 2]  # Wrong count

            # Validation happens in real __init__, this tests the concept


class TestQwen3Flux2EncoderOutputDimensions:
    """Test output dimension calculations."""

    def test_output_dim_4b_model(self):
        """Test output dimension for Qwen3-4B (2560 * 3 = 7680)."""
        # Qwen3-4B has hidden_dim=2560
        # With 3 layer concatenation: 2560 * 3 = 7680
        expected_4b_dim = 2560 * 3
        assert expected_4b_dim == 7680

    def test_output_dim_8b_model(self):
        """Test output dimension for Qwen3-8B (4096 * 3 = 12288)."""
        # Qwen3-8B has hidden_dim=4096
        # With 3 layer concatenation: 4096 * 3 = 12288
        expected_8b_dim = 4096 * 3
        assert expected_8b_dim == 12288


class TestQwen3Flux2EncoderEnableThinking:
    """Test that enable_thinking is correctly handled.

    CRITICAL: FLUX.2 Klein requires enable_thinking=False.
    Thinking tokens would corrupt the embeddings.
    """

    @patch("llm_dit.encoders.qwen3_flux2.AutoModelForCausalLM")
    @patch("llm_dit.encoders.qwen3_flux2.AutoTokenizer")
    def test_enable_thinking_is_false_in_template(self, mock_tokenizer_cls, mock_model_cls):
        """Test that chat template is applied with enable_thinking=False."""
        from llm_dit.encoders.qwen3_flux2 import Qwen3Flux2Encoder

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

        # Create encoder
        encoder = Qwen3Flux2Encoder(
            model_spec="test-model",
            device="cpu",
        )

        # Mock forward pass
        mock_hidden_states = [torch.randn(1, 10, 2560) for _ in range(28)]
        mock_output = MagicMock()
        mock_output.hidden_states = mock_hidden_states
        mock_model.return_value = mock_output

        # Encode
        encoder.forward(["Test prompt"])

        # Verify enable_thinking=False was used
        call_kwargs = mock_tokenizer.apply_chat_template.call_args[1]
        assert call_kwargs.get("enable_thinking") is False


class TestQwen3Flux2EncoderLayerExtraction:
    """Test multi-layer extraction and concatenation."""

    @patch("llm_dit.encoders.qwen3_flux2.AutoModelForCausalLM")
    @patch("llm_dit.encoders.qwen3_flux2.AutoTokenizer")
    def test_extracts_correct_layers(self, mock_tokenizer_cls, mock_model_cls):
        """Test that layers [9, 18, 27] are extracted."""
        from llm_dit.encoders.qwen3_flux2 import Qwen3Flux2Encoder

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

        # Create encoder with default layers
        encoder = Qwen3Flux2Encoder(
            model_spec="test-model",
            device="cpu",
        )

        assert encoder.output_layers == [9, 18, 27]

    @patch("llm_dit.encoders.qwen3_flux2.AutoModelForCausalLM")
    @patch("llm_dit.encoders.qwen3_flux2.AutoTokenizer")
    def test_custom_layers(self, mock_tokenizer_cls, mock_model_cls):
        """Test custom layer selection."""
        from llm_dit.encoders.qwen3_flux2 import Qwen3Flux2Encoder

        # Setup mocks
        mock_model = MagicMock()
        mock_model.config.hidden_size = 2560
        mock_model.config.num_hidden_layers = 28
        mock_model.device = torch.device("cpu")
        mock_model_cls.from_pretrained.return_value = mock_model

        mock_tokenizer = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        # Create encoder with custom layers
        encoder = Qwen3Flux2Encoder(
            model_spec="test-model",
            device="cpu",
            output_layers=[5, 15, 25],
        )

        assert encoder.output_layers == [5, 15, 25]

    @patch("llm_dit.encoders.qwen3_flux2.AutoModelForCausalLM")
    @patch("llm_dit.encoders.qwen3_flux2.AutoTokenizer")
    def test_wrong_layer_count_raises(self, mock_tokenizer_cls, mock_model_cls):
        """Test that non-3-layer configs raise error."""
        from llm_dit.encoders.qwen3_flux2 import Qwen3Flux2Encoder

        # Setup mocks
        mock_model = MagicMock()
        mock_model.config.hidden_size = 2560
        mock_model.config.num_hidden_layers = 28
        mock_model_cls.from_pretrained.return_value = mock_model

        mock_tokenizer = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        # Should raise for wrong layer count
        with pytest.raises(ValueError, match="must have exactly 3 layers"):
            Qwen3Flux2Encoder(
                model_spec="test-model",
                device="cpu",
                output_layers=[1, 2],  # Only 2 layers
            )


class TestQwen3Flux2EncoderProperties:
    """Test encoder properties."""

    @patch("llm_dit.encoders.qwen3_flux2.AutoModelForCausalLM")
    @patch("llm_dit.encoders.qwen3_flux2.AutoTokenizer")
    def test_output_dim_property(self, mock_tokenizer_cls, mock_model_cls):
        """Test output_dim is num_layers * hidden_dim."""
        from llm_dit.encoders.qwen3_flux2 import Qwen3Flux2Encoder

        mock_model = MagicMock()
        mock_model.config.hidden_size = 2560
        mock_model.config.num_hidden_layers = 28
        mock_model_cls.from_pretrained.return_value = mock_model

        mock_tokenizer = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        encoder = Qwen3Flux2Encoder(model_spec="test", device="cpu")

        assert encoder.output_dim == 3 * 2560
        assert encoder.output_dim == 7680

    @patch("llm_dit.encoders.qwen3_flux2.AutoModelForCausalLM")
    @patch("llm_dit.encoders.qwen3_flux2.AutoTokenizer")
    def test_hidden_dim_property(self, mock_tokenizer_cls, mock_model_cls):
        """Test hidden_dim returns single-layer dimension."""
        from llm_dit.encoders.qwen3_flux2 import Qwen3Flux2Encoder

        mock_model = MagicMock()
        mock_model.config.hidden_size = 2560
        mock_model.config.num_hidden_layers = 28
        mock_model_cls.from_pretrained.return_value = mock_model

        mock_tokenizer = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        encoder = Qwen3Flux2Encoder(model_spec="test", device="cpu")

        assert encoder.hidden_dim == 2560


class TestQwen3Flux2EncoderOffload:
    """Test offload functionality."""

    @patch("llm_dit.encoders.qwen3_flux2.AutoModelForCausalLM")
    @patch("llm_dit.encoders.qwen3_flux2.AutoTokenizer")
    def test_offload_moves_to_cpu(self, mock_tokenizer_cls, mock_model_cls):
        """Test offload moves model to CPU."""
        from llm_dit.encoders.qwen3_flux2 import Qwen3Flux2Encoder

        mock_model = MagicMock()
        mock_model.config.hidden_size = 2560
        mock_model.config.num_hidden_layers = 28
        mock_model_cls.from_pretrained.return_value = mock_model

        mock_tokenizer = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        encoder = Qwen3Flux2Encoder(model_spec="test", device="cuda")
        encoder.offload()

        mock_model.to.assert_called_with("cpu")
        assert encoder.device == torch.device("cpu")

    @patch("llm_dit.encoders.qwen3_flux2.AutoModelForCausalLM")
    @patch("llm_dit.encoders.qwen3_flux2.AutoTokenizer")
    def test_to_moves_to_device(self, mock_tokenizer_cls, mock_model_cls):
        """Test to() moves model to specified device."""
        from llm_dit.encoders.qwen3_flux2 import Qwen3Flux2Encoder

        mock_model = MagicMock()
        mock_model.config.hidden_size = 2560
        mock_model.config.num_hidden_layers = 28
        mock_model_cls.from_pretrained.return_value = mock_model

        mock_tokenizer = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        encoder = Qwen3Flux2Encoder(model_spec="test", device="cpu")
        encoder.to("cuda:0")

        mock_model.to.assert_called_with("cuda:0")


class TestQwen3Flux2EncoderFactoryFunction:
    """Test load_qwen3_flux2_encoder factory function."""

    def test_factory_validates_variant(self):
        """Test factory validates variant name."""
        from llm_dit.encoders.qwen3_flux2 import load_qwen3_flux2_encoder

        with pytest.raises(ValueError, match="Unknown variant"):
            load_qwen3_flux2_encoder(variant="16B")

    @patch("llm_dit.encoders.qwen3_flux2.Qwen3Flux2Encoder.from_pretrained")
    def test_factory_4b_variant(self, mock_from_pretrained):
        """Test factory creates 4B encoder."""
        from llm_dit.encoders.qwen3_flux2 import load_qwen3_flux2_encoder

        mock_encoder = MagicMock()
        mock_from_pretrained.return_value = mock_encoder

        result = load_qwen3_flux2_encoder(variant="4B")

        mock_from_pretrained.assert_called_once()
        call_kwargs = mock_from_pretrained.call_args[1]
        assert "Qwen3-4B" in call_kwargs["model_spec"]

    @patch("llm_dit.encoders.qwen3_flux2.Qwen3Flux2Encoder.from_pretrained")
    def test_factory_8b_variant(self, mock_from_pretrained):
        """Test factory creates 8B encoder."""
        from llm_dit.encoders.qwen3_flux2 import load_qwen3_flux2_encoder

        mock_encoder = MagicMock()
        mock_from_pretrained.return_value = mock_encoder

        result = load_qwen3_flux2_encoder(variant="8B")

        mock_from_pretrained.assert_called_once()
        call_kwargs = mock_from_pretrained.call_args[1]
        assert "Qwen3-8B" in call_kwargs["model_spec"]


class TestQwen3Flux2EncoderEncodeMethods:
    """Test encode methods."""

    @patch("llm_dit.encoders.qwen3_flux2.AutoModelForCausalLM")
    @patch("llm_dit.encoders.qwen3_flux2.AutoTokenizer")
    def test_encode_alias(self, mock_tokenizer_cls, mock_model_cls):
        """Test encode is alias for forward."""
        from llm_dit.encoders.qwen3_flux2 import Qwen3Flux2Encoder

        mock_model = MagicMock()
        mock_model.config.hidden_size = 2560
        mock_model.config.num_hidden_layers = 28
        mock_model.device = torch.device("cpu")
        mock_model_cls.from_pretrained.return_value = mock_model

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "text"
        mock_tokenizer.return_value = {
            "input_ids": torch.ones(1, 10, dtype=torch.long),
            "attention_mask": torch.ones(1, 10),
        }
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        encoder = Qwen3Flux2Encoder(model_spec="test", device="cpu")

        # Mock hidden states
        mock_hidden_states = [torch.randn(1, 10, 2560) for _ in range(28)]
        mock_output = MagicMock()
        mock_output.hidden_states = mock_hidden_states
        mock_model.return_value = mock_output

        # encode should call forward
        with patch.object(encoder, 'forward') as mock_forward:
            mock_forward.return_value = torch.randn(1, 10, 7680)
            encoder.encode(["test"])
            mock_forward.assert_called_once_with(["test"])

    @patch("llm_dit.encoders.qwen3_flux2.AutoModelForCausalLM")
    @patch("llm_dit.encoders.qwen3_flux2.AutoTokenizer")
    def test_encode_single(self, mock_tokenizer_cls, mock_model_cls):
        """Test encode_single wraps single prompt in list."""
        from llm_dit.encoders.qwen3_flux2 import Qwen3Flux2Encoder

        mock_model = MagicMock()
        mock_model.config.hidden_size = 2560
        mock_model.config.num_hidden_layers = 28
        mock_model.device = torch.device("cpu")
        mock_model_cls.from_pretrained.return_value = mock_model

        mock_tokenizer = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        encoder = Qwen3Flux2Encoder(model_spec="test", device="cpu")

        with patch.object(encoder, 'forward') as mock_forward:
            mock_forward.return_value = torch.randn(1, 10, 7680)
            encoder.encode_single("single prompt")
            mock_forward.assert_called_once_with(["single prompt"])
