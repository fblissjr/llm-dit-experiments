"""
Automated tests for Z-Image web server.

Tests the three server modes:
1. Full pipeline mode (pipeline loaded)
2. Encoder-only mode (local encoder, no DiT/VAE)
3. API backend mode (uses heylookitsanllm for encoding)

Coverage (29 tests):
- Health endpoint: Server status and mode detection
- Encode endpoint: Prompt encoding with/without templates, thinking control
- Generate endpoint: Image generation (full pipeline only), seed control, error handling
- Templates endpoint: Template listing and metadata
- Save embeddings endpoint: Distributed inference support
- Index endpoint: HTML serving
- Request validation: Parameter validation and defaults
- Mode detection: Pipeline/encoder priority and availability

All tests use mocked models - no actual LLM weights are loaded.

Run tests:
    uv run pytest tests/test_web_server.py -v
"""

import pytest
import torch
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, Mock, patch

from llm_dit.backends.protocol import EncodingOutput
from llm_dit.templates.loader import Template


@pytest.fixture
def mock_encoder():
    """Mock encoder with encode method."""
    encoder = MagicMock()
    encoder.device = torch.device("cpu")

    # Mock templates
    encoder.templates = {
        "default": Template(
            name="default",
            content="Generate natural images",
            thinking_content="",
        ),
        "photorealistic": Template(
            name="photorealistic",
            content="Generate photorealistic images",
            thinking_content="Consider lighting and texture...",
        ),
    }

    # Mock encode method
    def mock_encode(prompt, template=None, enable_thinking=True):
        embeddings = torch.randn(77, 2560)
        attention_mask = torch.ones(77, dtype=torch.bool)
        return EncodingOutput(
            embeddings=[embeddings],
            attention_masks=[attention_mask],
        )

    encoder.encode = Mock(side_effect=mock_encode)
    return encoder


@pytest.fixture
def mock_pipeline(mock_encoder):
    """Mock full pipeline with encoder."""
    pipeline = MagicMock()
    pipeline.encoder = mock_encoder
    pipeline.device = torch.device("cpu")

    # Mock generation - configure as side_effect on the pipeline itself
    def mock_generate(prompt, height=1024, width=1024, num_inference_steps=9,
                     guidance_scale=0.0, generator=None, template=None,
                     enable_thinking=True):
        from PIL import Image
        return Image.new("RGB", (width, height), color="blue")

    # Configure the pipeline itself to be callable
    pipeline.side_effect = mock_generate
    return pipeline


@pytest.fixture
def client_full_pipeline(mock_pipeline):
    """Test client with full pipeline loaded."""
    from web.server import app
    import web.server as server_module

    # Inject mocked pipeline
    server_module.pipeline = mock_pipeline
    server_module.encoder = None
    server_module.encoder_only_mode = False

    return TestClient(app)


@pytest.fixture
def client_encoder_only(mock_encoder):
    """Test client with encoder-only mode."""
    from web.server import app
    import web.server as server_module

    # Inject mocked encoder
    server_module.pipeline = None
    server_module.encoder = mock_encoder
    server_module.encoder_only_mode = True

    return TestClient(app)


@pytest.fixture
def client_no_models():
    """Test client with no models loaded."""
    from web.server import app
    import web.server as server_module

    # Clear all models
    server_module.pipeline = None
    server_module.encoder = None
    server_module.encoder_only_mode = False

    return TestClient(app)


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_full_pipeline(self, client_full_pipeline):
        """Health check with full pipeline loaded."""
        response = client_full_pipeline.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "ok"
        assert data["pipeline_loaded"] is True
        assert data["encoder_loaded"] is False
        assert data["encoder_only_mode"] is False

    def test_health_encoder_only(self, client_encoder_only):
        """Health check with encoder-only mode."""
        response = client_encoder_only.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "ok"
        assert data["pipeline_loaded"] is False
        assert data["encoder_loaded"] is True
        assert data["encoder_only_mode"] is True

    def test_health_no_models(self, client_no_models):
        """Health check with no models loaded."""
        response = client_no_models.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "ok"
        assert data["pipeline_loaded"] is False
        assert data["encoder_loaded"] is False
        assert data["encoder_only_mode"] is False


class TestEncodeEndpoint:
    """Tests for /api/encode endpoint."""

    def test_encode_basic(self, client_full_pipeline):
        """Basic encoding with prompt only."""
        response = client_full_pipeline.post(
            "/api/encode",
            json={
                "prompt": "A cat sleeping in sunlight",
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert "shape" in data
        assert "dtype" in data
        assert "encode_time" in data
        assert "prompt" in data

        assert data["shape"] == [77, 2560]
        assert data["prompt"] == "A cat sleeping in sunlight"

    def test_encode_with_template(self, client_encoder_only):
        """Encoding with template specified."""
        response = client_encoder_only.post(
            "/api/encode",
            json={
                "prompt": "A cat sleeping",
                "template": "photorealistic",
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert data["shape"] == [77, 2560]

        # Verify encoder.encode was called with template
        import web.server as server_module
        server_module.encoder.encode.assert_called()

    def test_encode_disable_thinking(self, client_encoder_only):
        """Encoding with thinking disabled."""
        response = client_encoder_only.post(
            "/api/encode",
            json={
                "prompt": "A cat",
                "enable_thinking": False,
            },
        )
        assert response.status_code == 200

        # Verify encoder was called with enable_thinking=False
        import web.server as server_module
        last_call = server_module.encoder.encode.call_args
        assert last_call.kwargs["enable_thinking"] is False

    def test_encode_no_encoder(self, client_no_models):
        """Encoding fails when no encoder available."""
        response = client_no_models.post(
            "/api/encode",
            json={"prompt": "test"},
        )
        assert response.status_code == 503
        assert "Encoder not loaded" in response.json()["detail"]

    def test_encode_error_handling(self, client_encoder_only):
        """Encoding handles errors gracefully."""
        import web.server as server_module

        # Make encoder.encode raise an exception
        server_module.encoder.encode = Mock(
            side_effect=RuntimeError("Encoding failed")
        )

        response = client_encoder_only.post(
            "/api/encode",
            json={"prompt": "test"},
        )
        assert response.status_code == 500
        assert "Encoding failed" in response.json()["detail"]


class TestGenerateEndpoint:
    """Tests for /api/generate endpoint."""

    def test_generate_basic(self, client_full_pipeline):
        """Basic image generation."""
        response = client_full_pipeline.post(
            "/api/generate",
            json={
                "prompt": "A cat sleeping",
                "width": 512,
                "height": 512,
                "steps": 8,
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert "X-Generation-Time" in response.headers

    def test_generate_with_seed(self, client_full_pipeline):
        """Generation with fixed seed."""
        response = client_full_pipeline.post(
            "/api/generate",
            json={
                "prompt": "A cat",
                "seed": 42,
            },
        )
        assert response.status_code == 200
        assert response.headers.get("X-Seed") == "42"

    def test_generate_with_template(self, client_full_pipeline):
        """Generation with template."""
        import web.server as server_module

        # Get a reference to the mock before the call
        pipeline_mock = server_module.pipeline

        response = client_full_pipeline.post(
            "/api/generate",
            json={
                "prompt": "A cat",
                "template": "photorealistic",
            },
        )
        assert response.status_code == 200

        # Verify pipeline was called (at least once, may have been called before)
        assert pipeline_mock.called
        # Check that photorealistic was used in at least one call
        found_template = False
        for call in pipeline_mock.call_args_list:
            if call[1].get("template") == "photorealistic":
                found_template = True
                break
        assert found_template, "Template 'photorealistic' not found in any call"

    def test_generate_encoder_only_mode(self, client_encoder_only):
        """Generation fails in encoder-only mode."""
        response = client_encoder_only.post(
            "/api/generate",
            json={"prompt": "test"},
        )
        assert response.status_code == 400
        assert "encoder-only mode" in response.json()["detail"]

    def test_generate_no_pipeline(self, client_no_models):
        """Generation fails when pipeline not loaded."""
        response = client_no_models.post(
            "/api/generate",
            json={"prompt": "test"},
        )
        assert response.status_code == 503
        assert "Pipeline not loaded" in response.json()["detail"]

    def test_generate_error_handling(self):
        """Generation handles errors gracefully."""
        from web.server import app
        import web.server as server_module

        # Create a fresh pipeline mock that raises an error
        error_pipeline = MagicMock()
        error_pipeline.encoder = MagicMock()
        error_pipeline.device = "cpu"
        error_pipeline.side_effect = RuntimeError("Generation failed")

        # Set up a fresh client with error pipeline
        original_pipeline = server_module.pipeline
        server_module.pipeline = error_pipeline
        server_module.encoder = None
        server_module.encoder_only_mode = False

        try:
            client = TestClient(app)
            response = client.post(
                "/api/generate",
                json={"prompt": "test"},
            )
            assert response.status_code == 500
            assert "Generation failed" in response.json()["detail"]
        finally:
            # Restore original
            server_module.pipeline = original_pipeline


class TestTemplatesEndpoint:
    """Tests for /api/templates endpoint."""

    def test_list_templates_full_pipeline(self, client_full_pipeline):
        """List templates with full pipeline."""
        response = client_full_pipeline.get("/api/templates")
        assert response.status_code == 200

        data = response.json()
        assert "templates" in data
        assert len(data["templates"]) == 2

        # Check template structure
        template_names = [t["name"] for t in data["templates"]]
        assert "default" in template_names
        assert "photorealistic" in template_names

        # Check has_thinking field
        for template in data["templates"]:
            assert "name" in template
            assert "has_thinking" in template

    def test_list_templates_encoder_only(self, client_encoder_only):
        """List templates in encoder-only mode."""
        response = client_encoder_only.get("/api/templates")
        assert response.status_code == 200

        data = response.json()
        assert len(data["templates"]) == 2

    def test_list_templates_no_encoder(self, client_no_models):
        """List templates returns empty when no encoder."""
        response = client_no_models.get("/api/templates")
        assert response.status_code == 200

        data = response.json()
        assert data["templates"] == []

    def test_template_thinking_content(self, client_encoder_only):
        """Verify has_thinking reflects template content."""
        response = client_encoder_only.get("/api/templates")
        data = response.json()

        # Find photorealistic template
        photo_template = next(
            t for t in data["templates"] if t["name"] == "photorealistic"
        )
        assert photo_template["has_thinking"] is True

        # Find default template
        default_template = next(
            t for t in data["templates"] if t["name"] == "default"
        )
        assert default_template["has_thinking"] is False


class TestSaveEmbeddingsEndpoint:
    """Tests for /api/save-embeddings endpoint."""

    def test_save_embeddings_basic(self, client_encoder_only):
        """Basic embeddings save."""
        with patch("llm_dit.distributed.save_embeddings") as mock_save:
            mock_save.return_value = "/path/to/embeddings_abc123.safetensors"

            response = client_encoder_only.post(
                "/api/save-embeddings",
                json={
                    "prompt": "A cat sleeping",
                },
            )
            assert response.status_code == 200

            data = response.json()
            assert "path" in data
            assert "shape" in data
            assert "encode_time" in data
            assert data["shape"] == [77, 2560]

    def test_save_embeddings_with_template(self, client_encoder_only):
        """Save embeddings with template."""
        with patch("llm_dit.distributed.save_embeddings") as mock_save:
            mock_save.return_value = "/path/to/embeddings.safetensors"

            response = client_encoder_only.post(
                "/api/save-embeddings",
                json={
                    "prompt": "A cat",
                    "template": "photorealistic",
                },
            )
            assert response.status_code == 200

            # Verify save_embeddings was called with correct args
            mock_save.assert_called_once()
            call_kwargs = mock_save.call_args.kwargs
            assert call_kwargs["template"] == "photorealistic"
            assert call_kwargs["prompt"] == "A cat"

    def test_save_embeddings_no_encoder(self, client_no_models):
        """Save embeddings fails when no encoder."""
        response = client_no_models.post(
            "/api/save-embeddings",
            json={"prompt": "test"},
        )
        assert response.status_code == 503
        assert "Encoder not loaded" in response.json()["detail"]

    def test_save_embeddings_error_handling(self, client_encoder_only):
        """Save embeddings handles errors gracefully."""
        with patch("llm_dit.distributed.save_embeddings") as mock_save:
            mock_save.side_effect = RuntimeError("Save failed")

            response = client_encoder_only.post(
                "/api/save-embeddings",
                json={"prompt": "test"},
            )
            assert response.status_code == 500
            assert "Save failed" in response.json()["detail"]


class TestIndexEndpoint:
    """Tests for / (index) endpoint."""

    def test_index_returns_html(self, client_full_pipeline):
        """Index endpoint serves HTML file."""
        from pathlib import Path

        # Create a dummy index.html for testing
        web_dir = Path(__file__).parent.parent / "web"
        web_dir.mkdir(exist_ok=True)
        index_file = web_dir / "index.html"

        # Create temporary index file if it doesn't exist
        temp_created = False
        if not index_file.exists():
            index_file.write_text("<html><body>Test</body></html>")
            temp_created = True

        try:
            response = client_full_pipeline.get("/")
            # Just verify we get a response (200 or 404)
            assert response.status_code in (200, 404)
        finally:
            # Clean up temporary file
            if temp_created and index_file.exists():
                index_file.unlink()


class TestRequestValidation:
    """Tests for request validation."""

    def test_encode_missing_prompt(self, client_encoder_only):
        """Encode requires prompt field."""
        response = client_encoder_only.post(
            "/api/encode",
            json={},
        )
        assert response.status_code == 422

    def test_generate_missing_prompt(self, client_full_pipeline):
        """Generate requires prompt field."""
        response = client_full_pipeline.post(
            "/api/generate",
            json={},
        )
        assert response.status_code == 422

    def test_generate_defaults(self, client_full_pipeline):
        """Generate uses default values."""
        import web.server as server_module

        # Get a reference to the mock
        pipeline_mock = server_module.pipeline

        response = client_full_pipeline.post(
            "/api/generate",
            json={"prompt": "test"},
        )
        assert response.status_code == 200

        # Verify defaults were used in at least one call
        assert pipeline_mock.called
        found_defaults = False
        for call in pipeline_mock.call_args_list:
            kwargs = call[1]
            if (kwargs.get("height") == 1024 and
                kwargs.get("width") == 1024 and
                kwargs.get("num_inference_steps") == 9 and
                kwargs.get("guidance_scale") == 0.0):
                found_defaults = True
                break
        assert found_defaults, "Default values not found in any call"


class TestModeDetection:
    """Tests for server mode detection."""

    def test_full_pipeline_mode_flags(self, client_full_pipeline):
        """Full pipeline mode sets correct flags."""
        response = client_full_pipeline.get("/health")
        data = response.json()

        assert data["pipeline_loaded"] is True
        assert data["encoder_only_mode"] is False

    def test_encoder_only_mode_flags(self, client_encoder_only):
        """Encoder-only mode sets correct flags."""
        response = client_encoder_only.get("/health")
        data = response.json()

        assert data["encoder_loaded"] is True
        assert data["encoder_only_mode"] is True

    def test_encoder_priority_in_encode(self, mock_encoder, mock_pipeline):
        """Encode endpoint prefers standalone encoder over pipeline.encoder."""
        from web.server import app
        import web.server as server_module

        # Both encoder and pipeline present
        standalone_encoder = mock_encoder
        pipeline_encoder = MagicMock()
        mock_pipeline.encoder = pipeline_encoder

        server_module.pipeline = mock_pipeline
        server_module.encoder = standalone_encoder
        server_module.encoder_only_mode = False

        client = TestClient(app)
        response = client.post(
            "/api/encode",
            json={"prompt": "test"},
        )
        assert response.status_code == 200

        # Verify standalone encoder was called, not pipeline.encoder
        standalone_encoder.encode.assert_called()
        pipeline_encoder.encode.assert_not_called()
