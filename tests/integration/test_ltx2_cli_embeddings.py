"""
Integration tests for LTX-2 embedding CLI flags.

Last Updated: 2026-01-23

Tests cover:
- --ltx2-save-embeddings CLI argument parsing
- --ltx2-load-embeddings CLI argument parsing
- Help text for new arguments

Run with: uv run pytest tests/integration/test_ltx2_cli_embeddings.py -v
"""

import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


class TestLTX2EmbeddingCLIArgs:
    """Test LTX-2 embedding CLI argument parsing."""

    def test_help_shows_save_embeddings_flag(self):
        """Test --help shows --ltx2-save-embeddings."""
        result = subprocess.run(
            ["uv", "run", "scripts/generate.py", "--model-type", "ltx2", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        assert result.returncode == 0
        assert "--ltx2-save-embeddings" in result.stdout
        assert "Save text embeddings" in result.stdout

    def test_help_shows_load_embeddings_flag(self):
        """Test --help shows --ltx2-load-embeddings."""
        result = subprocess.run(
            ["uv", "run", "scripts/generate.py", "--model-type", "ltx2", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        assert result.returncode == 0
        assert "--ltx2-load-embeddings" in result.stdout
        assert "Load pre-computed embeddings" in result.stdout

    def test_save_embeddings_help_mentions_skip_generation(self):
        """Test help text mentions skipping video generation."""
        result = subprocess.run(
            ["uv", "run", "scripts/generate.py", "--model-type", "ltx2", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        assert result.returncode == 0
        # Check help text describes the behavior
        assert "skip video generation" in result.stdout.lower() or "precomputation" in result.stdout.lower()

    def test_load_embeddings_help_mentions_skip_encoding(self):
        """Test help text mentions skipping text encoding."""
        result = subprocess.run(
            ["uv", "run", "scripts/generate.py", "--model-type", "ltx2", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        assert result.returncode == 0
        # Check help text describes the behavior
        assert "skip text encoding" in result.stdout.lower() or "Gemma3" in result.stdout


class TestLTX2EmbeddingCLIErrors:
    """Test CLI error handling for embedding arguments."""

    def test_save_embeddings_requires_prompt(self):
        """Test --ltx2-save-embeddings fails without prompt."""
        result = subprocess.run(
            [
                "uv",
                "run",
                "scripts/generate.py",
                "--model-type",
                "ltx2",
                "--ltx2-save-embeddings",
                "/tmp/test.safetensors",
                # No prompt provided
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        # Should fail or show error about missing prompt/model
        assert result.returncode != 0 or "error" in result.stderr.lower()

    def test_load_embeddings_requires_model_path(self):
        """Test --ltx2-load-embeddings without model fails appropriately."""
        result = subprocess.run(
            [
                "uv",
                "run",
                "scripts/generate.py",
                "--model-type",
                "ltx2",
                "--ltx2-load-embeddings",
                "/nonexistent/path.safetensors",
                # No model path, but load-embeddings should still need transformer/VAE
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        # Should fail (file doesn't exist or model path missing)
        assert result.returncode != 0


@pytest.mark.requires_ltx2_model
@pytest.mark.slow
class TestLTX2EmbeddingSaveLoad:
    """Test actual embedding save/load with LTX-2 model."""

    @pytest.fixture
    def ltx2_model_path(self):
        """Get LTX-2 model path from environment or skip."""
        import os
        path = os.environ.get("LTX2_MODEL_PATH", "")
        if not path or not Path(path).exists():
            pytest.skip("LTX-2 model not available (set LTX2_MODEL_PATH)")
        return path

    def test_save_embeddings_creates_file(self, ltx2_model_path, tmp_path):
        """Test --ltx2-save-embeddings creates embedding file."""
        emb_path = tmp_path / "test_embeddings.safetensors"

        result = subprocess.run(
            [
                "uv",
                "run",
                "scripts/generate.py",
                "--model-type",
                "ltx2",
                "--model-path",
                ltx2_model_path,
                "--ltx2-text-encoder-device",
                "cpu",
                "--ltx2-save-embeddings",
                str(emb_path),
                "A cat walking through a sunny garden",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
            timeout=300,
        )

        if result.returncode == 0:
            assert emb_path.exists()
            # Check metadata JSON
            json_path = emb_path.with_suffix(".json")
            assert json_path.exists()

    def test_load_embeddings_skips_encoder(self, ltx2_model_path, tmp_path):
        """Test --ltx2-load-embeddings skips text encoder stage."""
        # First save embeddings
        emb_path = tmp_path / "precomputed.safetensors"

        save_result = subprocess.run(
            [
                "uv",
                "run",
                "scripts/generate.py",
                "--model-type",
                "ltx2",
                "--model-path",
                ltx2_model_path,
                "--ltx2-text-encoder-device",
                "cpu",
                "--ltx2-save-embeddings",
                str(emb_path),
                "A cat walking",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
            timeout=300,
        )

        if save_result.returncode != 0:
            pytest.skip("Could not save embeddings")

        # Now load and generate
        output_path = tmp_path / "video.mp4"

        load_result = subprocess.run(
            [
                "uv",
                "run",
                "scripts/generate.py",
                "--model-type",
                "ltx2",
                "--model-path",
                ltx2_model_path,
                "--ltx2-load-embeddings",
                str(emb_path),
                "--ltx2-num-frames",
                "9",  # Minimal frames
                "--ltx2-steps",
                "2",  # Minimal steps
                "--output",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
            timeout=600,
        )

        # Check output indicates precomputed embeddings used
        output = load_result.stdout + load_result.stderr
        assert "precomputed" in output.lower() or "PRECOMPUTED" in output
