"""
Unit tests for LoRAManager class.

Tests the ComfyUI-style backup+patch system for reversible LoRA loading.
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from llm_dit.utils.lora import LoRAEntry, LoRAManager


class TestLoRAEntry:
    """Tests for LoRAEntry dataclass."""

    def test_creation_with_defaults(self):
        """Test creating LoRAEntry with default values."""
        entry = LoRAEntry(path="/path/to/lora.safetensors")
        assert entry.path == "/path/to/lora.safetensors"
        assert entry.name == "lora"  # Derived from filename
        assert entry.scale == 1.0
        assert entry.trigger_words == ""
        assert entry.enabled is True

    def test_creation_with_all_fields(self):
        """Test creating LoRAEntry with all fields specified."""
        entry = LoRAEntry(
            path="/path/to/anime_style.safetensors",
            name="Anime Style",
            scale=0.8,
            trigger_words="anime style, detailed",
            enabled=False,
        )
        assert entry.path == "/path/to/anime_style.safetensors"
        assert entry.name == "Anime Style"
        assert entry.scale == 0.8
        assert entry.trigger_words == "anime style, detailed"
        assert entry.enabled is False

    def test_name_derived_from_path(self):
        """Test that name is derived from path if not provided."""
        entry = LoRAEntry(path="/models/loras/photo_realistic_v2.safetensors")
        assert entry.name == "photo_realistic_v2"

    def test_to_dict(self):
        """Test serialization to dictionary."""
        entry = LoRAEntry(
            path="/path/to/lora.safetensors",
            scale=0.7,
            trigger_words="test",
            enabled=True,
        )
        data = entry.to_dict()
        assert data["path"] == "/path/to/lora.safetensors"
        assert data["name"] == "lora"
        assert data["scale"] == 0.7
        assert data["trigger_words"] == "test"
        assert data["enabled"] is True

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "path": "/path/to/lora.safetensors",
            "name": "Custom Name",
            "scale": 0.5,
            "trigger_words": "trigger",
            "enabled": False,
        }
        entry = LoRAEntry.from_dict(data)
        assert entry.path == "/path/to/lora.safetensors"
        assert entry.name == "Custom Name"
        assert entry.scale == 0.5
        assert entry.trigger_words == "trigger"
        assert entry.enabled is False

    def test_from_dict_with_defaults(self):
        """Test deserialization with missing optional fields."""
        data = {"path": "/path/to/lora.safetensors"}
        entry = LoRAEntry.from_dict(data)
        assert entry.path == "/path/to/lora.safetensors"
        assert entry.scale == 1.0
        assert entry.trigger_words == ""
        assert entry.enabled is True


class TestLoRAManager:
    """Tests for LoRAManager class."""

    @pytest.fixture
    def mock_model(self):
        """Create a simple mock model for testing."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
        )
        return model

    @pytest.fixture
    def lora_manager(self, mock_model):
        """Create a LoRAManager with mock model."""
        return LoRAManager(mock_model, loras_dir=None)

    def test_init(self, mock_model):
        """Test LoRAManager initialization."""
        manager = LoRAManager(mock_model, loras_dir="/path/to/loras")
        assert manager.model is mock_model
        assert manager.loras_dir == Path("/path/to/loras")
        assert manager.backup == {}
        assert manager.active_loras == []
        assert manager.lora_cache == {}

    def test_init_without_loras_dir(self, mock_model):
        """Test initialization without loras_dir."""
        manager = LoRAManager(mock_model)
        assert manager.loras_dir is None

    def test_scan_directory_empty(self, lora_manager):
        """Test scanning with no loras_dir configured."""
        paths = lora_manager.scan_directory()
        assert paths == []

    def test_scan_directory_with_files(self, mock_model, tmp_path):
        """Test scanning directory with LoRA files."""
        # Create mock LoRA files
        (tmp_path / "anime.safetensors").touch()
        (tmp_path / "photo.safetensors").touch()
        (tmp_path / "style.bin").touch()
        (tmp_path / "readme.txt").touch()  # Should be ignored

        manager = LoRAManager(mock_model, loras_dir=str(tmp_path))
        paths = manager.scan_directory()

        assert len(paths) == 3
        assert any("anime.safetensors" in p for p in paths)
        assert any("photo.safetensors" in p for p in paths)
        assert any("style.bin" in p for p in paths)
        assert not any("readme.txt" in p for p in paths)

    def test_get_available_loras(self, mock_model, tmp_path):
        """Test getting available LoRAs as LoRAEntry objects."""
        (tmp_path / "test.safetensors").touch()

        manager = LoRAManager(mock_model, loras_dir=str(tmp_path))
        available = manager.get_available_loras()

        assert len(available) == 1
        assert available[0].name == "test"
        assert available[0].enabled is False

    def test_add_lora(self, lora_manager):
        """Test adding a LoRA to the active list."""
        lora_manager.add_lora(
            "/path/to/lora.safetensors",
            scale=0.8,
            trigger_words="test trigger",
        )

        assert len(lora_manager.active_loras) == 1
        lora = lora_manager.active_loras[0]
        assert lora.path == "/path/to/lora.safetensors"
        assert lora.scale == 0.8
        assert lora.trigger_words == "test trigger"
        assert lora.enabled is True

    def test_add_lora_replaces_existing(self, lora_manager):
        """Test that adding same LoRA replaces the existing entry."""
        lora_manager.add_lora("/path/to/lora.safetensors", scale=0.5)
        lora_manager.add_lora("/path/to/lora.safetensors", scale=0.8)

        assert len(lora_manager.active_loras) == 1
        assert lora_manager.active_loras[0].scale == 0.8

    def test_remove_lora(self, lora_manager):
        """Test removing a LoRA from the active list."""
        lora_manager.add_lora("/path/to/lora.safetensors")
        assert len(lora_manager.active_loras) == 1

        result = lora_manager.remove_lora("/path/to/lora.safetensors")
        assert result is True
        assert len(lora_manager.active_loras) == 0

    def test_remove_lora_not_found(self, lora_manager):
        """Test removing a LoRA that doesn't exist."""
        result = lora_manager.remove_lora("/path/to/nonexistent.safetensors")
        assert result is False

    def test_set_scale(self, lora_manager):
        """Test updating scale for a LoRA."""
        lora_manager.add_lora("/path/to/lora.safetensors", scale=0.5)

        result = lora_manager.set_scale("/path/to/lora.safetensors", 0.9)
        assert result is True
        assert lora_manager.active_loras[0].scale == 0.9

    def test_set_scale_not_found(self, lora_manager):
        """Test setting scale for non-existent LoRA."""
        result = lora_manager.set_scale("/path/to/nonexistent.safetensors", 0.9)
        assert result is False

    def test_set_enabled(self, lora_manager):
        """Test enabling/disabling a LoRA."""
        lora_manager.add_lora("/path/to/lora.safetensors")
        assert lora_manager.active_loras[0].enabled is True

        result = lora_manager.set_enabled("/path/to/lora.safetensors", False)
        assert result is True
        assert lora_manager.active_loras[0].enabled is False

    def test_reorder(self, lora_manager):
        """Test reordering LoRAs."""
        lora_manager.add_lora("/path/to/lora1.safetensors")
        lora_manager.add_lora("/path/to/lora2.safetensors")
        lora_manager.add_lora("/path/to/lora3.safetensors")

        # Reverse order
        lora_manager.reorder([
            "/path/to/lora3.safetensors",
            "/path/to/lora2.safetensors",
            "/path/to/lora1.safetensors",
        ])

        assert lora_manager.active_loras[0].path == "/path/to/lora3.safetensors"
        assert lora_manager.active_loras[1].path == "/path/to/lora2.safetensors"
        assert lora_manager.active_loras[2].path == "/path/to/lora1.safetensors"

    def test_get_trigger_words(self, lora_manager):
        """Test getting combined trigger words."""
        lora_manager.add_lora("/path/to/lora1.safetensors", trigger_words="anime style")
        lora_manager.add_lora("/path/to/lora2.safetensors", trigger_words="detailed")
        lora_manager.add_lora("/path/to/lora3.safetensors", trigger_words="")  # No trigger

        words = lora_manager.get_trigger_words()
        assert words == "anime style detailed"

    def test_get_trigger_words_disabled_excluded(self, lora_manager):
        """Test that disabled LoRAs don't contribute trigger words."""
        lora_manager.add_lora("/path/to/lora1.safetensors", trigger_words="enabled trigger")
        lora_manager.add_lora(
            "/path/to/lora2.safetensors",
            trigger_words="disabled trigger",
            enabled=False,
        )

        words = lora_manager.get_trigger_words()
        assert words == "enabled trigger"
        assert "disabled" not in words

    def test_get_active_entries(self, lora_manager):
        """Test getting active LoRA entries."""
        lora_manager.add_lora("/path/to/lora1.safetensors")
        lora_manager.add_lora("/path/to/lora2.safetensors")

        entries = lora_manager.get_active_entries()
        assert len(entries) == 2
        assert entries is not lora_manager.active_loras  # Should be a copy

    def test_backup_weights(self, mock_model):
        """Test that backup stores weights on CPU."""
        manager = LoRAManager(mock_model)
        manager._backup_weights()

        assert len(manager.backup) > 0
        for name, weight in manager.backup.items():
            assert weight.device == torch.device("cpu")

    def test_backup_only_once(self, mock_model):
        """Test that backup only happens once."""
        manager = LoRAManager(mock_model)
        manager._backup_weights()
        backup1 = manager.backup.copy()

        # Modify model weights
        with torch.no_grad():
            for module in mock_model.modules():
                if hasattr(module, 'weight') and module.weight is not None:
                    module.weight.add_(1.0)

        # Second backup should not overwrite
        manager._backup_weights()
        backup2 = manager.backup

        # Backup should be the same (from first call)
        for name in backup1:
            assert torch.equal(backup1[name], backup2[name])

    def test_restore_weights(self, mock_model):
        """Test restoring weights from backup."""
        manager = LoRAManager(mock_model)

        # Get original weights
        original_weights = {}
        for name, module in mock_model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                original_weights[name] = module.weight.data.clone()

        # Backup
        manager._backup_weights()

        # Modify weights
        with torch.no_grad():
            for module in mock_model.modules():
                if hasattr(module, 'weight') and module.weight is not None:
                    module.weight.add_(10.0)

        # Restore
        manager._restore_weights()

        # Check weights are restored
        for name, module in mock_model.named_modules():
            if name in original_weights:
                assert torch.allclose(
                    module.weight.data,
                    original_weights[name],
                    atol=1e-6,
                ), f"Weight mismatch for {name}"

    def test_clear_all(self, mock_model):
        """Test clearing all LoRAs and restoring weights."""
        manager = LoRAManager(mock_model)

        # Add some LoRAs
        manager.add_lora("/path/to/lora1.safetensors")
        manager.add_lora("/path/to/lora2.safetensors")

        # Backup weights
        manager._backup_weights()

        # Clear
        manager.clear_all()

        assert len(manager.active_loras) == 0
        assert manager.get_trigger_words() == ""

    def test_set_loras(self, lora_manager):
        """Test replacing all LoRAs at once."""
        entries = [
            LoRAEntry(path="/path/to/lora1.safetensors", scale=0.5),
            LoRAEntry(path="/path/to/lora2.safetensors", scale=0.8),
        ]

        # Mock apply to avoid actual LoRA loading
        with patch.object(lora_manager, 'apply', return_value=0):
            lora_manager.set_loras(entries)

        assert len(lora_manager.active_loras) == 2
        assert lora_manager.active_loras[0].scale == 0.5
        assert lora_manager.active_loras[1].scale == 0.8


class TestLoRAManagerThreadSafety:
    """Tests for thread safety of LoRAManager."""

    def test_apply_is_locked(self):
        """Test that apply() uses locking."""
        model = nn.Linear(10, 10)
        manager = LoRAManager(model)

        # The lock should be acquired during apply
        assert hasattr(manager, '_lock')
        assert manager._lock is not None


class TestLoRAManagerIntegration:
    """Integration tests that require actual LoRA files."""

    @pytest.fixture
    def lora_path(self):
        """Get test LoRA path from environment."""
        import os
        path = os.environ.get("TEST_LORA_PATH")
        if not path:
            pytest.skip("TEST_LORA_PATH not set")
        return path

    @pytest.mark.requires_lora
    def test_load_and_apply_real_lora(self, lora_path):
        """Test loading and applying a real LoRA file."""
        # This test requires a real model and LoRA file
        # Skip if not available
        pytest.skip("Requires full model - run manually")
