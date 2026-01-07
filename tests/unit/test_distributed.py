"""
Unit tests for llm_dit.distributed module.

Tests cover:
- Embedding save/load for distributed inference
- Metadata handling
"""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
import json
import pytest
import torch


# ============================================================================
# EmbeddingMetadata Tests
# ============================================================================

class TestEmbeddingMetadata:
    """Test EmbeddingMetadata dataclass."""

    def test_embedding_metadata_creation(self):
        """Test creating EmbeddingMetadata."""
        from llm_dit.distributed.embeddings import EmbeddingMetadata

        metadata = EmbeddingMetadata(
            prompt="A cat sleeping",
            template="photorealistic",
            enable_thinking=True,
            system_prompt="You are an artist",
            thinking_content="Consider lighting",
            sequence_length=100,
            embedding_dim=2560,
            dtype="torch.bfloat16",
            created_at="2024-01-01T00:00:00",
            encoder_device="cuda:0",
            model_path="/path/to/model",
        )

        assert metadata.prompt == "A cat sleeping"
        assert metadata.template == "photorealistic"
        assert metadata.enable_thinking is True
        assert metadata.sequence_length == 100
        assert metadata.embedding_dim == 2560

    def test_embedding_metadata_to_dict(self):
        """Test EmbeddingMetadata.to_dict()."""
        from llm_dit.distributed.embeddings import EmbeddingMetadata

        metadata = EmbeddingMetadata(
            prompt="Test prompt",
            template=None,
            enable_thinking=False,
            system_prompt=None,
            thinking_content=None,
            sequence_length=50,
            embedding_dim=2560,
            dtype="torch.float32",
            created_at="2024-01-01",
            encoder_device="cpu",
            model_path="/model",
        )

        data = metadata.to_dict()
        assert data["prompt"] == "Test prompt"
        assert data["sequence_length"] == 50
        assert data["embedding_dim"] == 2560

    def test_embedding_metadata_from_dict(self):
        """Test EmbeddingMetadata.from_dict()."""
        from llm_dit.distributed.embeddings import EmbeddingMetadata

        data = {
            "prompt": "Test",
            "template": "anime",
            "enable_thinking": True,
            "system_prompt": None,
            "thinking_content": None,
            "sequence_length": 75,
            "embedding_dim": 2560,
            "dtype": "torch.bfloat16",
            "created_at": "2024-01-01",
            "encoder_device": "cuda",
            "model_path": "/model",
        }

        metadata = EmbeddingMetadata.from_dict(data)
        assert metadata.prompt == "Test"
        assert metadata.template == "anime"
        assert metadata.sequence_length == 75


# ============================================================================
# EmbeddingFile Tests
# ============================================================================

class TestEmbeddingFile:
    """Test EmbeddingFile class."""

    @pytest.fixture
    def sample_metadata(self):
        """Create sample metadata."""
        from llm_dit.distributed.embeddings import EmbeddingMetadata

        return EmbeddingMetadata(
            prompt="A beautiful sunset",
            template="photorealistic",
            enable_thinking=True,
            system_prompt=None,
            thinking_content=None,
            sequence_length=100,
            embedding_dim=2560,
            dtype="torch.bfloat16",
            created_at=datetime.now().isoformat(),
            encoder_device="cuda:0",
            model_path="/path/to/model",
        )

    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings tensor."""
        return torch.randn(100, 2560)

    def test_embedding_file_save_and_load(self, tmp_path, sample_embeddings, sample_metadata):
        """Test saving and loading EmbeddingFile."""
        from llm_dit.distributed.embeddings import EmbeddingFile

        emb_file = EmbeddingFile(
            embeddings=sample_embeddings,
            metadata=sample_metadata,
        )

        # Save
        save_path = tmp_path / "embeddings.safetensors"
        emb_file.save(save_path)

        # Check files exist
        assert save_path.exists()
        assert (tmp_path / "embeddings.json").exists()

        # Load
        loaded = EmbeddingFile.load(save_path)

        assert loaded.embeddings.shape == sample_embeddings.shape
        assert loaded.metadata.prompt == sample_metadata.prompt
        assert loaded.metadata.sequence_length == sample_metadata.sequence_length

    def test_embedding_file_load_without_json(self, tmp_path):
        """Test loading embeddings when JSON sidecar is missing."""
        from llm_dit.distributed.embeddings import EmbeddingFile
        from safetensors.torch import save_file

        embeddings = torch.randn(50, 2560)
        save_path = tmp_path / "no_meta.safetensors"

        # Save only the tensor, no JSON
        save_file({"embeddings": embeddings}, save_path)

        # Load should work with minimal metadata
        loaded = EmbeddingFile.load(save_path)

        assert loaded.embeddings.shape == embeddings.shape
        assert loaded.metadata.prompt == "<unknown>"
        assert loaded.metadata.sequence_length == 50
        assert loaded.metadata.embedding_dim == 2560


# ============================================================================
# save_embeddings Function Tests
# ============================================================================

class TestSaveEmbeddings:
    """Test save_embeddings function."""

    def test_save_embeddings_basic(self, tmp_path):
        """Test basic embedding saving."""
        from llm_dit.distributed.embeddings import save_embeddings

        embeddings = torch.randn(100, 2560)
        output_path = tmp_path / "test_embeddings"

        result_path = save_embeddings(
            embeddings=embeddings,
            path=output_path,
            prompt="Test prompt",
            model_path="/path/to/model",
        )

        # Should add .safetensors extension
        assert result_path.suffix == ".safetensors"
        assert result_path.exists()

        # Check JSON sidecar
        json_path = result_path.with_suffix(".json")
        assert json_path.exists()

        with open(json_path) as f:
            metadata = json.load(f)
        assert metadata["prompt"] == "Test prompt"
        assert metadata["sequence_length"] == 100
        assert metadata["embedding_dim"] == 2560

    def test_save_embeddings_with_template(self, tmp_path):
        """Test saving embeddings with template info."""
        from llm_dit.distributed.embeddings import save_embeddings

        embeddings = torch.randn(75, 2560)
        output_path = tmp_path / "template_embeddings.safetensors"

        save_embeddings(
            embeddings=embeddings,
            path=output_path,
            prompt="A cat",
            model_path="/model",
            template="anime",
            enable_thinking=True,
            system_prompt="Draw in anime style",
            thinking_content="Consider character design",
        )

        json_path = output_path.with_suffix(".json")
        with open(json_path) as f:
            metadata = json.load(f)

        assert metadata["template"] == "anime"
        assert metadata["enable_thinking"] is True
        assert metadata["system_prompt"] == "Draw in anime style"
        assert metadata["thinking_content"] == "Consider character design"

    def test_save_embeddings_moves_to_cpu(self, tmp_path):
        """Test that embeddings are moved to CPU before saving."""
        from llm_dit.distributed.embeddings import save_embeddings

        # Create embeddings (will be on CPU in tests)
        embeddings = torch.randn(50, 2560)
        output_path = tmp_path / "cpu_test.safetensors"

        save_embeddings(
            embeddings=embeddings,
            path=output_path,
            prompt="Test",
            encoder_device="cuda:0",  # Simulated device
        )

        # Verify saved embeddings are CPU tensors
        from safetensors.torch import load_file
        loaded = load_file(output_path)
        assert loaded["embeddings"].device.type == "cpu"


# ============================================================================
# load_embeddings Function Tests
# ============================================================================

class TestLoadEmbeddings:
    """Test load_embeddings function."""

    def test_load_embeddings_basic(self, tmp_path):
        """Test basic embedding loading."""
        from llm_dit.distributed.embeddings import save_embeddings, load_embeddings

        # Save first
        embeddings = torch.randn(100, 2560)
        save_path = tmp_path / "load_test.safetensors"
        save_embeddings(
            embeddings=embeddings,
            path=save_path,
            prompt="Load test prompt",
        )

        # Load
        emb_file = load_embeddings(save_path)

        assert emb_file.embeddings.shape == (100, 2560)
        assert emb_file.metadata.prompt == "Load test prompt"

    def test_load_embeddings_to_device(self, tmp_path):
        """Test loading embeddings to specific device."""
        from llm_dit.distributed.embeddings import save_embeddings, load_embeddings

        embeddings = torch.randn(50, 2560)
        save_path = tmp_path / "device_test.safetensors"
        save_embeddings(
            embeddings=embeddings,
            path=save_path,
            prompt="Device test",
        )

        # Load to CPU explicitly
        emb_file = load_embeddings(save_path, device="cpu")
        assert emb_file.embeddings.device.type == "cpu"


# ============================================================================
# Roundtrip Integration Tests
# ============================================================================

class TestDistributedRoundtrip:
    """Test complete save/load roundtrip."""

    def test_full_roundtrip(self, tmp_path):
        """Test complete save and load cycle."""
        from llm_dit.distributed.embeddings import save_embeddings, load_embeddings

        # Create embeddings with specific values
        torch.manual_seed(42)
        original_embeddings = torch.randn(150, 2560)
        save_path = tmp_path / "roundtrip.safetensors"

        # Save
        save_embeddings(
            embeddings=original_embeddings,
            path=save_path,
            prompt="Roundtrip test prompt",
            model_path="/test/model",
            template="test_template",
            enable_thinking=False,
            system_prompt="Test system",
            encoder_device="cpu",
        )

        # Load
        loaded = load_embeddings(save_path)

        # Verify embeddings match
        assert torch.allclose(loaded.embeddings, original_embeddings)

        # Verify metadata
        assert loaded.metadata.prompt == "Roundtrip test prompt"
        assert loaded.metadata.model_path == "/test/model"
        assert loaded.metadata.template == "test_template"
        assert loaded.metadata.enable_thinking is False
        assert loaded.metadata.system_prompt == "Test system"
        assert loaded.metadata.sequence_length == 150
        assert loaded.metadata.embedding_dim == 2560

    def test_timestamp_recorded(self, tmp_path):
        """Test that creation timestamp is recorded."""
        from llm_dit.distributed.embeddings import save_embeddings, load_embeddings

        embeddings = torch.randn(10, 2560)
        save_path = tmp_path / "timestamp_test.safetensors"

        before_save = datetime.now()
        save_embeddings(embeddings=embeddings, path=save_path, prompt="Time test")
        after_save = datetime.now()

        loaded = load_embeddings(save_path)

        # Parse timestamp and verify it's between before and after
        created_at = datetime.fromisoformat(loaded.metadata.created_at)
        # Allow some tolerance for milliseconds
        assert before_save.replace(microsecond=0) <= created_at.replace(microsecond=0)
        assert created_at.replace(microsecond=0) <= after_save.replace(microsecond=0)
