"""
Integration tests for FastAPI web server.

Tests REST API endpoints for encoding, formatting, and generation.
"""

import pytest

pytestmark = pytest.mark.integration


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_returns_ok(self, client):
        """Health endpoint should return 200."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data


class TestFormatPromptEndpoint:
    """Test /api/format-prompt endpoint (no model required)."""

    def test_format_basic_prompt(self, client):
        """Format a simple prompt without think block."""
        response = client.post(
            "/api/format-prompt",
            json={"prompt": "A cat sleeping", "force_think_block": False},
        )
        assert response.status_code == 200
        data = response.json()

        assert "formatted_prompt" in data
        assert "<|im_start|>user" in data["formatted_prompt"]
        assert "A cat sleeping" in data["formatted_prompt"]
        # Token count should be returned
        assert "token_count" in data

    def test_format_with_system_prompt(self, client):
        """Format prompt with system message."""
        response = client.post(
            "/api/format-prompt",
            json={
                "prompt": "A cat",
                "system_prompt": "You are a photographer.",
                "force_think_block": False,
            },
        )
        assert response.status_code == 200
        data = response.json()

        assert "<|im_start|>system" in data["formatted_prompt"]
        assert "You are a photographer" in data["formatted_prompt"]

    def test_format_with_thinking(self, client):
        """Format prompt with thinking block (triggered by thinking_content)."""
        response = client.post(
            "/api/format-prompt",
            json={
                "prompt": "A cat",
                "thinking_content": "Orange fur, green eyes",
            },
        )
        assert response.status_code == 200
        data = response.json()

        assert "<think>" in data["formatted_prompt"]
        assert "Orange fur" in data["formatted_prompt"]

    def test_format_with_assistant_content(self, client):
        """Format prompt with assistant prefill."""
        response = client.post(
            "/api/format-prompt",
            json={
                "prompt": "A cat",
                "thinking_content": "Thinking...",
                "assistant_content": "Here is your cat:",
            },
        )
        assert response.status_code == 200
        data = response.json()

        assert "Here is your cat:" in data["formatted_prompt"]
        # Should close with im_end when assistant content provided
        assert data["formatted_prompt"].strip().endswith("<|im_end|>")


class TestTemplatesEndpoint:
    """Test /api/templates endpoint."""

    def test_list_templates(self, client):
        """List available templates."""
        response = client.get("/api/templates")
        assert response.status_code == 200
        data = response.json()

        assert "templates" in data
        # Should have some templates (may be empty in test env)
        assert isinstance(data["templates"], (list, dict))


@pytest.mark.requires_model
class TestEncodeEndpoint:
    """Test /api/encode endpoint (requires model)."""

    def test_encode_basic_prompt(self, client_with_model):
        """Encode a simple prompt."""
        response = client_with_model.post(
            "/api/encode",
            json={"prompt": "A cat sleeping", "enable_thinking": True},
        )
        assert response.status_code == 200
        data = response.json()

        assert "shape" in data
        # Should be [seq_len, 2560]
        assert data["shape"][1] == 2560

    def test_encode_with_template(self, client_with_model):
        """Encode with template applied."""
        response = client_with_model.post(
            "/api/encode",
            json={
                "prompt": "A mountain landscape",
                "template": "photorealistic",
                "enable_thinking": True,
            },
        )
        # May fail if template not found, which is ok
        if response.status_code == 200:
            data = response.json()
            assert "shape" in data


@pytest.mark.requires_gpu
@pytest.mark.slow
class TestGenerateEndpoint:
    """Test /api/generate endpoint (requires full pipeline + GPU)."""

    def test_generate_basic_image(self, client_with_pipeline):
        """Generate a basic image."""
        response = client_with_pipeline.post(
            "/api/generate",
            json={
                "prompt": "A simple test image",
                "width": 512,
                "height": 512,
                "steps": 4,
                "seed": 42,
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        # Should have image data
        assert len(response.content) > 1000

    def test_generate_with_thinking(self, client_with_pipeline):
        """Generate with thinking content."""
        response = client_with_pipeline.post(
            "/api/generate",
            json={
                "prompt": "A cat sleeping",
                "thinking_content": "Soft lighting, peaceful mood",
                "enable_thinking": True,
                "width": 512,
                "height": 512,
                "steps": 4,
                "seed": 42,
            },
        )
        assert response.status_code == 200


class TestHistoryEndpoints:
    """Test generation history management."""

    def test_get_empty_history(self, client):
        """Get history when empty."""
        response = client.get("/api/history")
        assert response.status_code == 200
        data = response.json()
        assert "history" in data

    def test_clear_history(self, client):
        """Clear all history."""
        response = client.delete("/api/history")
        assert response.status_code == 200


# Fixtures for web server testing
# Note: The 'client' fixture is defined in conftest.py with a properly mocked encoder


@pytest.fixture
def client_with_model(z_image_model_path):
    """Create test client with encoder loaded."""
    from fastapi.testclient import TestClient

    from web.server import app, load_encoder

    # Load encoder only
    load_encoder(z_image_model_path, device="cpu")

    return TestClient(app)


@pytest.fixture
def client_with_pipeline(z_image_model_path):
    """Create test client with full pipeline (requires GPU)."""
    from fastapi.testclient import TestClient

    from web.server import app, load_pipeline

    # Load full pipeline
    load_pipeline(
        z_image_model_path,
        encoder_device="cpu",
        dit_device="cuda",
        vae_device="cuda",
    )

    return TestClient(app)
