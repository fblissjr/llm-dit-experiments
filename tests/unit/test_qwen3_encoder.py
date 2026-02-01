"""
Unit tests for Qwen3 Encoder (Z-Image).

Last Updated: 2026-02-01

Tests the Qwen3Encoder wrapper used for Z-Image text encoding.
These tests run without GPU by mocking the TransformersBackend.

Run with: uv run pytest tests/unit/test_qwen3_encoder.py -v
"""

import pytest
import torch
from unittest.mock import MagicMock, patch

pytestmark = pytest.mark.unit


class TestQwen3EncoderImports:
    """Test that Qwen3Encoder can be imported."""

    def test_import_encoder(self):
        """Test basic import."""
        from llm_dit.encoders.qwen3 import Qwen3Encoder
        assert Qwen3Encoder is not None

    def test_import_protocol(self):
        """Test protocol imports."""
        from llm_dit.encoders.protocol import (
            EncoderCapability,
            EncoderInfo,
            EncoderType,
            EncodingOutput,
        )
        assert EncoderCapability is not None
        assert EncoderInfo is not None
        assert EncoderType is not None
        assert EncodingOutput is not None


class TestQwen3EncoderInit:
    """Test Qwen3Encoder initialization."""

    def test_init_with_mock_backend(self):
        """Test initialization with mocked backend."""
        from llm_dit.encoders.qwen3 import Qwen3Encoder

        mock_backend = MagicMock()
        mock_backend.embedding_dim = 2560
        mock_backend.max_sequence_length = 512
        mock_backend.config.quantization = None
        mock_backend.device = torch.device("cpu")
        mock_backend.dtype = torch.bfloat16

        encoder = Qwen3Encoder(backend=mock_backend, model_id="test-model")

        assert encoder._model_id == "test-model"
        assert encoder._backend == mock_backend
        assert encoder._is_offloaded is False


class TestQwen3EncoderProperties:
    """Test Qwen3Encoder properties."""

    def test_embedding_dim(self):
        """Test embedding_dim property."""
        from llm_dit.encoders.qwen3 import Qwen3Encoder

        mock_backend = MagicMock()
        mock_backend.embedding_dim = 2560

        encoder = Qwen3Encoder(backend=mock_backend, model_id="test")
        assert encoder.embedding_dim == 2560

    def test_max_sequence_length(self):
        """Test max_sequence_length property."""
        from llm_dit.encoders.qwen3 import Qwen3Encoder

        mock_backend = MagicMock()
        mock_backend.max_sequence_length = 512

        encoder = Qwen3Encoder(backend=mock_backend, model_id="test")
        assert encoder.max_sequence_length == 512

    def test_device_property(self):
        """Test device property."""
        from llm_dit.encoders.qwen3 import Qwen3Encoder

        mock_backend = MagicMock()
        mock_backend.device = torch.device("cuda:0")

        encoder = Qwen3Encoder(backend=mock_backend, model_id="test")
        assert encoder.device == torch.device("cuda:0")

    def test_dtype_property(self):
        """Test dtype property."""
        from llm_dit.encoders.qwen3 import Qwen3Encoder

        mock_backend = MagicMock()
        mock_backend.dtype = torch.bfloat16

        encoder = Qwen3Encoder(backend=mock_backend, model_id="test")
        assert encoder.dtype == torch.bfloat16


class TestQwen3EncoderInfo:
    """Test Qwen3Encoder info property."""

    def test_info_returns_encoder_info(self):
        """Test info property returns EncoderInfo."""
        from llm_dit.encoders.qwen3 import Qwen3Encoder
        from llm_dit.encoders.protocol import EncoderInfo, EncoderType, EncoderCapability

        mock_backend = MagicMock()
        mock_backend.embedding_dim = 2560
        mock_backend.max_sequence_length = 512
        mock_backend.config.quantization = None
        mock_backend.device = torch.device("cpu")
        mock_backend.dtype = torch.bfloat16

        encoder = Qwen3Encoder(backend=mock_backend, model_id="test-model")
        info = encoder.info

        assert isinstance(info, EncoderInfo)
        assert info.encoder_type == EncoderType.QWEN3
        assert info.model_id == "test-model"
        assert info.hidden_dim == 2560
        assert info.max_sequence_length == 512

    def test_info_capabilities(self):
        """Test info includes correct capabilities."""
        from llm_dit.encoders.qwen3 import Qwen3Encoder
        from llm_dit.encoders.protocol import EncoderCapability

        mock_backend = MagicMock()
        mock_backend.embedding_dim = 2560
        mock_backend.max_sequence_length = 512
        mock_backend.config.quantization = None
        mock_backend.device = torch.device("cpu")
        mock_backend.dtype = torch.bfloat16

        encoder = Qwen3Encoder(backend=mock_backend, model_id="test")
        info = encoder.info

        assert EncoderCapability.TEXT_ENCODING in info.capabilities
        assert EncoderCapability.TEXT_GENERATION in info.capabilities
        assert EncoderCapability.HIDDEN_LAYER_SELECTION in info.capabilities


class TestQwen3EncoderEncode:
    """Test Qwen3Encoder encode method."""

    def test_encode_calls_backend(self):
        """Test encode delegates to backend."""
        from llm_dit.encoders.qwen3 import Qwen3Encoder
        from llm_dit.encoders.protocol import EncodingOutput

        mock_output = EncodingOutput(
            embeddings=[torch.randn(10, 2560)],
            attention_masks=[torch.ones(10)],
            token_counts=[10],
        )

        mock_backend = MagicMock()
        mock_backend.encode.return_value = mock_output

        encoder = Qwen3Encoder(backend=mock_backend, model_id="test")
        result = encoder.encode(["Test prompt"], layer_index=-2)

        mock_backend.encode.assert_called_once_with(
            texts=["Test prompt"],
            return_padded=False,
            layer_index=-2,
        )
        assert result == mock_output

    def test_encode_rejects_images(self):
        """Test encode raises error for images."""
        from llm_dit.encoders.qwen3 import Qwen3Encoder
        from PIL import Image

        mock_backend = MagicMock()
        encoder = Qwen3Encoder(backend=mock_backend, model_id="test")

        with pytest.raises(ValueError, match="does not support vision"):
            encoder.encode(["Test"], images=[Image.new("RGB", (100, 100))])

    def test_encode_default_layer(self):
        """Test encode uses layer -2 by default."""
        from llm_dit.encoders.qwen3 import Qwen3Encoder
        from llm_dit.encoders.protocol import EncodingOutput

        mock_output = EncodingOutput(
            embeddings=[torch.randn(10, 2560)],
            attention_masks=[torch.ones(10)],
            token_counts=[10],
        )

        mock_backend = MagicMock()
        mock_backend.encode.return_value = mock_output

        encoder = Qwen3Encoder(backend=mock_backend, model_id="test")
        encoder.encode(["Test prompt"])

        # Check layer_index=-2 (default)
        call_kwargs = mock_backend.encode.call_args[1]
        assert call_kwargs.get("layer_index") == -2


class TestQwen3EncoderBlended:
    """Test Qwen3Encoder blended encoding."""

    def test_encode_blended_calls_backend(self):
        """Test encode_blended delegates to backend."""
        from llm_dit.encoders.qwen3 import Qwen3Encoder
        from llm_dit.encoders.protocol import EncodingOutput

        mock_output = EncodingOutput(
            embeddings=[torch.randn(10, 2560)],
            attention_masks=[torch.ones(10)],
            token_counts=[10],
        )

        mock_backend = MagicMock()
        mock_backend.encode_blended.return_value = mock_output

        encoder = Qwen3Encoder(backend=mock_backend, model_id="test")
        layer_weights = {-2: 0.5, -4: 0.3, -6: 0.2}
        result = encoder.encode_blended(["Test"], layer_weights=layer_weights)

        mock_backend.encode_blended.assert_called_once_with(
            texts=["Test"],
            layer_weights=layer_weights,
            return_padded=False,
        )


class TestQwen3EncoderGenerate:
    """Test Qwen3Encoder text generation."""

    def test_generate_calls_backend(self):
        """Test generate delegates to backend."""
        from llm_dit.encoders.qwen3 import Qwen3Encoder

        mock_backend = MagicMock()
        mock_backend.generate.return_value = "Generated text"

        encoder = Qwen3Encoder(backend=mock_backend, model_id="test")
        result = encoder.generate(
            prompt="Hello",
            system_prompt="You are helpful",
            max_new_tokens=100,
            temperature=0.7,
        )

        mock_backend.generate.assert_called_once()
        assert result == "Generated text"


class TestQwen3EncoderOffload:
    """Test Qwen3Encoder offload/device management."""

    def test_offload_moves_to_cpu(self):
        """Test offload moves model to CPU."""
        from llm_dit.encoders.qwen3 import Qwen3Encoder

        mock_model = MagicMock()
        mock_backend = MagicMock()
        mock_backend.model = mock_model

        encoder = Qwen3Encoder(backend=mock_backend, model_id="test")
        encoder.offload()

        mock_model.to.assert_called_once_with("cpu")
        assert encoder._is_offloaded is True

    def test_offload_idempotent(self):
        """Test offload is idempotent."""
        from llm_dit.encoders.qwen3 import Qwen3Encoder

        mock_model = MagicMock()
        mock_backend = MagicMock()
        mock_backend.model = mock_model

        encoder = Qwen3Encoder(backend=mock_backend, model_id="test")
        encoder.offload()
        encoder.offload()  # Second call should be no-op

        mock_model.to.assert_called_once_with("cpu")

    def test_to_moves_to_device(self):
        """Test to() moves model to device."""
        from llm_dit.encoders.qwen3 import Qwen3Encoder

        mock_backend = MagicMock()
        mock_backend.to.return_value = mock_backend

        encoder = Qwen3Encoder(backend=mock_backend, model_id="test")
        result = encoder.to(torch.device("cuda:0"))

        mock_backend.to.assert_called_once_with(torch.device("cuda:0"))
        assert result == encoder
        assert encoder._is_offloaded is False


class TestQwen3EncoderCache:
    """Test Qwen3Encoder caching functionality."""

    def test_cache_enabled_property(self):
        """Test cache_enabled delegates to backend."""
        from llm_dit.encoders.qwen3 import Qwen3Encoder

        mock_backend = MagicMock()
        mock_backend.cache_enabled = True

        encoder = Qwen3Encoder(backend=mock_backend, model_id="test")
        assert encoder.cache_enabled is True

    def test_enable_cache(self):
        """Test enable_cache delegates to backend."""
        from llm_dit.encoders.qwen3 import Qwen3Encoder

        mock_cache = MagicMock()
        mock_backend = MagicMock()
        mock_backend.enable_cache.return_value = mock_cache

        encoder = Qwen3Encoder(backend=mock_backend, model_id="test")
        result = encoder.enable_cache(max_size=50)

        mock_backend.enable_cache.assert_called_once_with(50)
        assert result == mock_cache

    def test_disable_cache(self):
        """Test disable_cache delegates to backend."""
        from llm_dit.encoders.qwen3 import Qwen3Encoder

        mock_backend = MagicMock()
        encoder = Qwen3Encoder(backend=mock_backend, model_id="test")
        encoder.disable_cache()

        mock_backend.disable_cache.assert_called_once()

    def test_clear_cache(self):
        """Test clear_cache delegates to backend."""
        from llm_dit.encoders.qwen3 import Qwen3Encoder

        mock_backend = MagicMock()
        encoder = Qwen3Encoder(backend=mock_backend, model_id="test")
        encoder.clear_cache()

        mock_backend.clear_cache.assert_called_once()


class TestQwen3EncoderBackendAccess:
    """Test Qwen3Encoder backend access."""

    def test_backend_property(self):
        """Test backend property exposes underlying backend."""
        from llm_dit.encoders.qwen3 import Qwen3Encoder

        mock_backend = MagicMock()
        encoder = Qwen3Encoder(backend=mock_backend, model_id="test")

        assert encoder.backend == mock_backend
