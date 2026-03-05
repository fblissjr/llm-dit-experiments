"""
Unit tests for configuration system.

Tests TOML parsing, CLI overrides, device resolution, and v5 quantization API.
These tests run on any platform without GPU or model files.
"""

import pytest
import torch

from llm_dit.config import (
    Config,
    EncoderConfig,
    GenerationConfig,
    LoRAConfig,
    OptimizationConfig,
    SchedulerConfig,
    get_preset,
    load_config,
)

pytestmark = pytest.mark.unit


class TestEncoderConfig:
    """Test EncoderConfig dataclass."""

    def test_default_values(self):
        config = EncoderConfig()
        assert config.device == "auto"
        assert config.dtype == "bfloat16"
        assert config.quantization == "none"
        assert config.cpu_offload is False
        assert config.trust_remote_code is True
        assert config.max_length == 512

    def test_quantization_int8(self):
        config = EncoderConfig(quantization="int8")
        assert config.quantization == "int8"

    def test_quantization_fp8_weight_only(self):
        config = EncoderConfig(quantization="fp8-weight-only")
        assert config.quantization == "fp8-weight-only"

    def test_get_dtype(self):
        config = EncoderConfig(dtype="bfloat16")
        assert config.get_dtype() == torch.bfloat16

        config = EncoderConfig(dtype="float16")
        assert config.get_dtype() == torch.float16

        config = EncoderConfig(dtype="float32")
        assert config.get_dtype() == torch.float32

    def test_get_device_auto_resolution(self):
        config = EncoderConfig(device="auto")
        resolved = config.get_device()
        # Should resolve to one of the available devices
        assert resolved in ["cpu", "cuda", "mps"]

    def test_get_device_explicit(self):
        config = EncoderConfig(device="cpu")
        assert config.get_device() == "cpu"


class TestGenerationConfig:
    """Test GenerationConfig dataclass."""

    def test_default_values(self):
        config = GenerationConfig()
        assert config.height == 1024
        assert config.width == 1024
        assert config.num_inference_steps == 9
        assert config.guidance_scale == 0.0
        assert config.enable_thinking is True
        assert config.default_template is None


class TestSchedulerConfig:
    """Test SchedulerConfig dataclass."""

    def test_default_shift(self):
        config = SchedulerConfig()
        assert config.shift == 3.0

    def test_custom_shift(self):
        config = SchedulerConfig(shift=5.0)
        assert config.shift == 5.0


class TestLoRAConfig:
    """Test LoRAConfig dataclass."""

    def test_default_empty(self):
        config = LoRAConfig()
        assert config.paths == []
        assert config.scales == []

    def test_with_values(self):
        config = LoRAConfig(paths=["lora1.safetensors", "lora2.safetensors"], scales=[0.8, 0.5])
        assert len(config.paths) == 2
        assert len(config.scales) == 2


class TestConfig:
    """Test complete Config dataclass."""

    def test_default_config(self):
        config = Config()
        assert config.model_path == ""
        assert config.templates_dir is None
        assert isinstance(config.encoder, EncoderConfig)
        assert isinstance(config.generation, GenerationConfig)

    def test_from_dict(self):
        data = {
            "model_path": "/path/to/model",
            "templates_dir": "templates",
            "encoder": {"device": "cpu", "quantization": "int8"},
            "generation": {"width": 512, "height": 512},
        }
        config = Config.from_dict(data)

        assert config.model_path == "/path/to/model"
        assert config.encoder.device == "cpu"
        assert config.encoder.quantization == "int8"
        assert config.generation.width == 512

    def test_to_dict(self):
        config = Config(
            model_path="/test/path",
            encoder=EncoderConfig(quantization="int8"),
            generation=GenerationConfig(width=512),
        )
        data = config.to_dict()

        assert data["model_path"] == "/test/path"
        assert data["encoder"]["quantization"] == "int8"
        assert data["generation"]["width"] == 512
        # Should NOT include deprecated fields
        assert "load_in_8bit" not in data["encoder"]
        assert "load_in_4bit" not in data["encoder"]


class TestConfigTOML:
    """Test TOML config file loading."""

    def test_load_from_toml(self, test_config_file):
        config = Config.from_toml(test_config_file, profile="default")

        assert config.model_path == "/test/path"
        assert config.encoder.device == "cpu"
        assert config.encoder.quantization == "none"
        assert config.generation.width == 512
        assert config.generation.height == 512

    def test_load_low_vram_profile(self, test_config_file):
        config = Config.from_toml(test_config_file, profile="low_vram")

        assert config.encoder.device == "cuda"
        assert config.encoder.quantization == "int8"
        assert config.encoder.cpu_offload is True

    def test_missing_profile_raises(self, test_config_file):
        with pytest.raises(KeyError) as exc_info:
            Config.from_toml(test_config_file, profile="nonexistent")

        assert "nonexistent" in str(exc_info.value)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Config.from_toml(tmp_path / "missing.toml")


class TestPresets:
    """Test preset configurations."""

    def test_default_preset(self):
        config = get_preset("default")
        assert config.encoder.device == "auto"
        assert config.encoder.dtype == "bfloat16"

    def test_low_vram_preset(self):
        config = get_preset("low_vram")
        assert config.encoder.quantization == "int8"
        assert config.encoder.cpu_offload is True
        assert config.optimization.cpu_offload is True

    def test_cpu_only_preset(self):
        config = get_preset("cpu_only")
        assert config.encoder.device == "cpu"
        assert config.encoder.dtype == "float32"

    def test_unknown_preset_raises(self):
        with pytest.raises(KeyError):
            get_preset("nonexistent")


class TestLoadConfig:
    """Test load_config function."""

    def test_load_from_toml_file(self, test_config_file):
        config = load_config(path=test_config_file, profile="default")
        assert config.model_path == "/test/path"

    def test_load_from_preset(self):
        config = load_config(preset="low_vram")
        assert config.encoder.quantization == "int8"

    def test_load_default(self):
        config = load_config()
        assert isinstance(config, Config)
        assert config.model_path == ""

    def test_toml_takes_precedence_over_preset(self, test_config_file):
        # When path is provided, preset is ignored
        config = load_config(path=test_config_file, preset="cpu_only")
        # Should use TOML values, not cpu_only preset
        assert config.encoder.dtype == "bfloat16"  # From TOML, not float32


class TestSessionConfigEndpoint:
    """Test that /api/config/session serializes without errors."""

    def test_session_config_none_profile(self):
        """SessionConfigResponse should handle None current_profile."""
        from llm_dit.config import RuntimeConfig
        from web.schemas import SessionConfigResponse

        rc = RuntimeConfig()
        assert rc.current_profile is None

        # This is the fix: getattr returns None (attr exists), not "default"
        profile = getattr(rc, "current_profile", None) or "default"
        resp = SessionConfigResponse(
            values={},
            profile=profile,
            modified=[],
            config_file=None,
        )
        assert resp.profile == "default"

    def test_session_config_with_profile(self):
        """SessionConfigResponse should preserve an explicit profile name."""
        from llm_dit.config import RuntimeConfig
        from web.schemas import SessionConfigResponse

        rc = RuntimeConfig()
        rc.current_profile = "testing"

        profile = getattr(rc, "current_profile", None) or "default"
        resp = SessionConfigResponse(
            values={},
            profile=profile,
            modified=[],
            config_file=None,
        )
        assert resp.profile == "testing"
