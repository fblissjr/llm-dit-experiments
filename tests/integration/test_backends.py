"""
Integration tests for backend implementations.

Tests TransformersBackend and APIBackend encoding.
Some tests require model files or API access.
"""

import pytest
import torch

from llm_dit.backends.config import BackendConfig
from llm_dit.backends.protocol import EncodingOutput

pytestmark = pytest.mark.integration


class TestBackendConfig:
    """Test BackendConfig dataclass."""

    def test_default_config(self):
        config = BackendConfig()
        assert config.backend_type == "transformers"
        assert config.max_length == 512
        assert config.torch_dtype == "bfloat16"

    def test_for_z_image_factory(self):
        config = BackendConfig.for_z_image("/path/to/model")
        assert config.model_path == "/path/to/model"
        assert config.subfolder == "text_encoder"
        assert config.max_length == 512

    def test_get_torch_dtype(self):
        config = BackendConfig(torch_dtype="float16")
        assert config.get_torch_dtype() == torch.float16


class TestTransformersBackendMocked:
    """Test TransformersBackend with mocked components."""

    def test_encode_with_mock_backend(self, mock_backend):
        """Test encoding flow with mock backend."""
        texts = ["<|im_start|>user\nA cat<|im_end|>"]
        output = mock_backend.encode(texts)

        assert isinstance(output, EncodingOutput)
        assert len(output.embeddings) == 1
        assert output.embeddings[0].shape[1] == 2560

    def test_batch_encode_with_mock(self, mock_backend):
        """Test batch encoding with mock."""
        texts = [
            "<|im_start|>user\nA cat<|im_end|>",
            "<|im_start|>user\nA dog<|im_end|>",
        ]
        output = mock_backend.encode(texts)

        assert len(output.embeddings) == 2


@pytest.mark.requires_model
class TestTransformersBackendReal:
    """Test TransformersBackend with real model (requires Z_IMAGE_MODEL_PATH)."""

    def test_load_from_pretrained(self, z_image_model_path):
        """Test loading model from pretrained path."""
        from llm_dit.backends.transformers import TransformersBackend

        backend = TransformersBackend.from_pretrained(
            z_image_model_path, device_map="cpu"
        )

        assert backend.embedding_dim == 2560
        assert backend.max_sequence_length == 512

    def test_encode_single_prompt(self, z_image_model_path):
        """Test encoding a single prompt."""
        from llm_dit.backends.transformers import TransformersBackend

        backend = TransformersBackend.from_pretrained(
            z_image_model_path, device_map="cpu"
        )

        formatted = "<|im_start|>user\nA cat sleeping<|im_end|>"
        output = backend.encode([formatted])

        assert isinstance(output, EncodingOutput)
        assert len(output.embeddings) == 1
        assert output.embeddings[0].shape[1] == 2560
        assert output.embeddings[0].dtype in [torch.bfloat16, torch.float32]

    def test_encode_batch(self, z_image_model_path):
        """Test batch encoding multiple prompts."""
        from llm_dit.backends.transformers import TransformersBackend

        backend = TransformersBackend.from_pretrained(
            z_image_model_path, device_map="cpu"
        )

        texts = [
            "<|im_start|>user\nA cat<|im_end|>",
            "<|im_start|>user\nA longer prompt about a dog running in a field<|im_end|>",
        ]
        output = backend.encode(texts)

        assert len(output.embeddings) == 2
        # Different prompts should have different sequence lengths
        assert output.embeddings[0].shape[0] != output.embeddings[1].shape[0]

    def test_encode_with_return_padded(self, z_image_model_path):
        """Test encoding with padded output."""
        from llm_dit.backends.transformers import TransformersBackend

        backend = TransformersBackend.from_pretrained(
            z_image_model_path, device_map="cpu"
        )

        formatted = "<|im_start|>user\nA cat<|im_end|>"
        output = backend.encode([formatted], return_padded=True)

        assert output.padded_embeddings is not None
        assert output.padded_mask is not None
        # Padded should be max_length
        assert output.padded_embeddings.shape[1] == 512


class TestAPIBackendMocked:
    """Test APIBackend with mocked HTTP responses."""

    def test_decode_base64_response(self, mock_api_response):
        """Test decoding base64-encoded embeddings."""
        import base64

        import numpy as np

        from llm_dit.backends.api import APIBackend

        # Decode the mock response
        hidden_states_bytes = base64.b64decode(mock_api_response["hidden_states"])
        shape = mock_api_response["shape"]
        hidden_states = np.frombuffer(hidden_states_bytes, dtype=np.float32).reshape(
            shape
        )

        assert hidden_states.shape == tuple(shape)
        assert hidden_states.shape[1] == 2560

    def test_api_backend_with_mock(self, mock_api_response, monkeypatch):
        """Test API backend encode with mocked HTTP client."""
        from unittest.mock import MagicMock

        from llm_dit.backends.api import APIBackend, APIBackendConfig

        # Create config
        config = APIBackendConfig(
            base_url="http://mock-server:8080",
            model_id="Qwen3-4B",
            encoding_format="base64",
        )
        backend = APIBackend(config)

        # Mock the HTTP client
        mock_response = MagicMock()
        mock_response.json.return_value = mock_api_response
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response

        # Replace client
        monkeypatch.setattr(backend, "_client", mock_client)

        # Test encode
        output = backend.encode(["<|im_start|>user\nTest<|im_end|>"])

        assert isinstance(output, EncodingOutput)
        assert len(output.embeddings) == 1
        assert output.embeddings[0].shape[1] == 2560


@pytest.mark.requires_api
class TestAPIBackendReal:
    """Test APIBackend with real heylookitsanllm server."""

    def test_encode_via_api(self, api_server_url):
        """Test real API encoding."""
        from llm_dit.backends.api import APIBackend, APIBackendConfig

        config = APIBackendConfig(
            base_url=api_server_url,
            model_id="Qwen3-4B-mxfp4-mlx",
            encoding_format="base64",
        )
        backend = APIBackend(config)

        formatted = "<|im_start|>user\nA cat sleeping<|im_end|>"
        output = backend.encode([formatted])

        assert isinstance(output, EncodingOutput)
        assert len(output.embeddings) == 1
        assert output.embeddings[0].shape[1] == 2560

    def test_api_batch_encoding(self, api_server_url):
        """Test batch encoding via API."""
        from llm_dit.backends.api import APIBackend, APIBackendConfig

        config = APIBackendConfig(
            base_url=api_server_url,
            model_id="Qwen3-4B-mxfp4-mlx",
        )
        backend = APIBackend(config)

        texts = [
            "<|im_start|>user\nA cat<|im_end|>",
            "<|im_start|>user\nA dog<|im_end|>",
        ]
        output = backend.encode(texts)

        assert len(output.embeddings) == 2


@pytest.mark.requires_model
@pytest.mark.requires_api
class TestBackendEquivalence:
    """Compare API backend vs local backend outputs."""

    def test_api_vs_local_shape_match(self, z_image_model_path, api_server_url):
        """Verify API and local backends produce same-shaped outputs."""
        from llm_dit.backends.api import APIBackend, APIBackendConfig
        from llm_dit.backends.transformers import TransformersBackend

        # Local backend
        local_backend = TransformersBackend.from_pretrained(
            z_image_model_path, device_map="cpu"
        )

        # API backend
        api_config = APIBackendConfig(
            base_url=api_server_url,
            model_id="Qwen3-4B-mxfp4-mlx",
        )
        api_backend = APIBackend(api_config)

        # Same prompt
        formatted = "<|im_start|>user\nA cat sleeping in sunlight<|im_end|>"

        local_output = local_backend.encode([formatted])
        api_output = api_backend.encode([formatted])

        # Shapes should match
        assert local_output.embeddings[0].shape == api_output.embeddings[0].shape

    @pytest.mark.slow
    def test_api_vs_local_values_similar(self, z_image_model_path, api_server_url):
        """Verify API and local backends produce similar values (within tolerance)."""
        from llm_dit.backends.api import APIBackend, APIBackendConfig
        from llm_dit.backends.transformers import TransformersBackend

        local_backend = TransformersBackend.from_pretrained(
            z_image_model_path, device_map="cpu"
        )

        api_config = APIBackendConfig(
            base_url=api_server_url,
            model_id="Qwen3-4B-mxfp4-mlx",
        )
        api_backend = APIBackend(api_config)

        formatted = "<|im_start|>user\nA simple test prompt<|im_end|>"

        local_output = local_backend.encode([formatted])
        api_output = api_backend.encode([formatted])

        local_emb = local_output.embeddings[0].float()
        api_emb = api_output.embeddings[0].float()

        # Values should be close (tolerance for MLX vs PyTorch differences)
        torch.testing.assert_close(local_emb, api_emb, rtol=1e-2, atol=1e-2)
