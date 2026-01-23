"""
Unit tests for LTX-2 embedding precomputation.

Last Updated: 2026-01-23

Tests cover:
- LTX-2 embedding dimensions (3840-dim vs Z-Image's 2560-dim)
- RuntimeConfig fields for ltx2_save_embeddings/ltx2_load_embeddings
- Embedding save/load roundtrip with LTX-2 dimensions

Run with: uv run pytest tests/unit/test_ltx2_embeddings.py -v
"""

import json
from pathlib import Path

import pytest
import torch


class TestLTX2EmbeddingDimensions:
    """Test embedding infrastructure works with LTX-2's 3840-dim embeddings."""

    def test_save_embeddings_3840_dim(self, tmp_path):
        """Test saving 3840-dim embeddings (LTX-2 Gemma3 output)."""
        from llm_dit.distributed.embeddings import save_embeddings

        # LTX-2 Gemma3 outputs 3840-dim embeddings
        embeddings = torch.randn(128, 3840)  # [seq_len, embed_dim]
        output_path = tmp_path / "ltx2_embeddings.safetensors"

        result_path = save_embeddings(
            embeddings=embeddings,
            path=output_path,
            prompt="A cat walking through a sunny garden",
            model_path="/path/to/LTX-2/text_encoder",
            encoder_device="cpu",
        )

        assert result_path.exists()
        assert result_path.suffix == ".safetensors"

        # Check metadata JSON
        json_path = result_path.with_suffix(".json")
        assert json_path.exists()

        with open(json_path) as f:
            metadata = json.load(f)

        assert metadata["embedding_dim"] == 3840
        assert metadata["sequence_length"] == 128

    def test_load_embeddings_3840_dim(self, tmp_path):
        """Test loading 3840-dim embeddings."""
        from llm_dit.distributed.embeddings import save_embeddings, load_embeddings

        # Save LTX-2 style embeddings
        original = torch.randn(100, 3840)
        save_path = tmp_path / "ltx2_load_test.safetensors"
        save_embeddings(
            embeddings=original,
            path=save_path,
            prompt="Test prompt for LTX-2",
        )

        # Load
        emb_file = load_embeddings(save_path)

        assert emb_file.embeddings.shape == (100, 3840)
        assert torch.allclose(emb_file.embeddings, original)
        assert emb_file.metadata.embedding_dim == 3840

    def test_roundtrip_preserves_values(self, tmp_path):
        """Test complete roundtrip preserves exact tensor values."""
        from llm_dit.distributed.embeddings import save_embeddings, load_embeddings

        torch.manual_seed(42)
        original = torch.randn(150, 3840)
        save_path = tmp_path / "roundtrip.safetensors"

        save_embeddings(
            embeddings=original,
            path=save_path,
            prompt="Roundtrip test",
            model_path="/LTX-2/text_encoder",
        )

        loaded = load_embeddings(save_path)

        # Exact match, not allclose (safetensors is lossless)
        assert torch.equal(loaded.embeddings, original)


class TestRuntimeConfigLTX2Embeddings:
    """Test RuntimeConfig has LTX-2 embedding fields."""

    def test_config_has_save_embeddings_field(self):
        """Test RuntimeConfig has ltx2_save_embeddings field."""
        from llm_dit.cli import RuntimeConfig

        config = RuntimeConfig()
        assert hasattr(config, "ltx2_save_embeddings")
        assert config.ltx2_save_embeddings is None

    def test_config_has_load_embeddings_field(self):
        """Test RuntimeConfig has ltx2_load_embeddings field."""
        from llm_dit.cli import RuntimeConfig

        config = RuntimeConfig()
        assert hasattr(config, "ltx2_load_embeddings")
        assert config.ltx2_load_embeddings is None

    def test_config_fields_are_string_or_none(self):
        """Test embedding path fields accept strings."""
        from llm_dit.cli import RuntimeConfig

        config = RuntimeConfig()
        config.ltx2_save_embeddings = "/path/to/embeddings.safetensors"
        config.ltx2_load_embeddings = "/path/to/load.safetensors"

        assert config.ltx2_save_embeddings == "/path/to/embeddings.safetensors"
        assert config.ltx2_load_embeddings == "/path/to/load.safetensors"


class TestGenerateVideoWithPrecomputedEmbeddings:
    """Test generate_video_with_offloading accepts precomputed embeddings."""

    def test_function_signature_has_precomputed_embeddings(self):
        """Test generate_video_with_offloading has precomputed_embeddings param."""
        import inspect
        from llm_dit.pipelines.generate import generate_video_with_offloading

        sig = inspect.signature(generate_video_with_offloading)
        params = list(sig.parameters.keys())

        assert "precomputed_embeddings" in params

    def test_precomputed_embeddings_default_is_none(self):
        """Test precomputed_embeddings defaults to None."""
        import inspect
        from llm_dit.pipelines.generate import generate_video_with_offloading

        sig = inspect.signature(generate_video_with_offloading)
        param = sig.parameters["precomputed_embeddings"]

        assert param.default is None


class TestEmbeddingDimensionCompatibility:
    """Test embedding infrastructure handles different dimensions."""

    @pytest.mark.parametrize(
        "embed_dim,model_name",
        [
            (2560, "Z-Image (Qwen3-4B)"),
            (3840, "LTX-2 (Gemma3-12B)"),
            (4096, "Future model"),
        ],
    )
    def test_save_load_various_dimensions(self, tmp_path, embed_dim, model_name):
        """Test save/load works for various embedding dimensions."""
        from llm_dit.distributed.embeddings import save_embeddings, load_embeddings

        embeddings = torch.randn(50, embed_dim)
        save_path = tmp_path / f"test_{embed_dim}.safetensors"

        save_embeddings(
            embeddings=embeddings,
            path=save_path,
            prompt=f"Test for {model_name}",
        )

        loaded = load_embeddings(save_path)

        assert loaded.embeddings.shape == (50, embed_dim)
        assert loaded.metadata.embedding_dim == embed_dim
