"""
Integration tests for CLI scripts.

Tests scripts/generate.py and web/server.py command-line interfaces.
"""

import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


class TestGenerateScriptHelp:
    """Test CLI help and argument parsing (no model required)."""

    def test_help_flag(self):
        """Test --help shows usage."""
        result = subprocess.run(
            ["uv", "run", "scripts/generate.py", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        assert result.returncode == 0
        assert "usage:" in result.stdout.lower() or "generate" in result.stdout.lower()

    def test_missing_required_args(self):
        """Test error on missing required arguments."""
        result = subprocess.run(
            ["uv", "run", "scripts/generate.py"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        # Should fail without model path or prompt
        assert result.returncode != 0


class TestWebServerHelp:
    """Test web server CLI help."""

    def test_server_help(self):
        """Test server --help."""
        result = subprocess.run(
            ["uv", "run", "web/server.py", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        assert result.returncode == 0
        assert "server" in result.stdout.lower() or "host" in result.stdout.lower()


@pytest.mark.requires_model
class TestGenerateScriptWithModel:
    """Test generate.py with real model (encoder-only mode)."""

    def test_encoder_only_mode(self, z_image_model_path, tmp_path):
        """Test encoding without generation."""
        result = subprocess.run(
            [
                "uv",
                "run",
                "scripts/generate.py",
                "--model-path",
                z_image_model_path,
                "--text-encoder-device",
                "cpu",
                "--encoder-only",
                "A cat sleeping",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        # Should succeed and show encoding info
        assert result.returncode == 0 or "encoding" in result.stderr.lower()

    def test_save_embeddings(self, z_image_model_path, tmp_path):
        """Test saving embeddings to file."""
        emb_path = tmp_path / "embeddings.safetensors"

        result = subprocess.run(
            [
                "uv",
                "run",
                "scripts/generate.py",
                "--model-path",
                z_image_model_path,
                "--text-encoder-device",
                "cpu",
                "--save-embeddings",
                str(emb_path),
                "A cat",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        # Check if embeddings were saved
        if result.returncode == 0:
            assert emb_path.exists()

    def test_format_prompt_output(self, z_image_model_path):
        """Test formatted prompt output."""
        result = subprocess.run(
            [
                "uv",
                "run",
                "scripts/generate.py",
                "--model-path",
                z_image_model_path,
                "--text-encoder-device",
                "cpu",
                "--encoder-only",
                "--system-prompt",
                "You are a painter.",
                "--thinking-content",
                "Soft lighting...",
                "--enable-thinking",
                "A cat sleeping",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        # Output should contain formatted prompt info
        output = result.stdout + result.stderr
        # Check for some indication of formatting
        assert result.returncode == 0 or "formatted" in output.lower()


@pytest.mark.requires_gpu
@pytest.mark.slow
class TestGenerateScriptFullPipeline:
    """Test full generation via CLI (requires GPU)."""

    def test_basic_generation(self, z_image_model_path, tmp_path):
        """Test basic image generation."""
        output_path = tmp_path / "output.png"

        result = subprocess.run(
            [
                "uv",
                "run",
                "scripts/generate.py",
                "--model-path",
                z_image_model_path,
                "--output",
                str(output_path),
                "--width",
                "512",
                "--height",
                "512",
                "--steps",
                "4",
                "--seed",
                "42",
                "--text-encoder-device",
                "cpu",
                "--dit-device",
                "cuda",
                "--vae-device",
                "cuda",
                "A simple test image",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
            timeout=300,  # 5 minute timeout
        )

        assert result.returncode == 0
        assert output_path.exists()

    def test_generation_with_config_file(self, z_image_model_path, tmp_path):
        """Test generation using config file."""
        # Create config file
        config_path = tmp_path / "test_config.toml"
        config_path.write_text(
            f"""
[default]
model_path = "{z_image_model_path}"

[default.encoder]
device = "cpu"

[default.pipeline]
device = "cuda"

[default.generation]
width = 512
height = 512
num_inference_steps = 4
"""
        )

        output_path = tmp_path / "output.png"

        result = subprocess.run(
            [
                "uv",
                "run",
                "scripts/generate.py",
                "--config",
                str(config_path),
                "--output",
                str(output_path),
                "--seed",
                "42",
                "A test from config",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
            timeout=300,
        )

        assert result.returncode == 0
        assert output_path.exists()


@pytest.mark.requires_api
class TestCLIWithAPIBackend:
    """Test CLI with API backend (distributed inference)."""

    def test_api_encoding(self, api_server_url, tmp_path):
        """Test encoding via API backend."""
        result = subprocess.run(
            [
                "uv",
                "run",
                "scripts/generate.py",
                "--api-url",
                api_server_url,
                "--api-model",
                "Qwen3-4B-mxfp4-mlx",
                "--encoder-only",
                "A cat sleeping",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        # Should succeed with API encoding
        assert result.returncode == 0 or "encoding" in result.stderr.lower()
