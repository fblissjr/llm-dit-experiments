"""
Unit tests for modular configuration system.

Tests TOML parsing, inheritance resolution, component library validation,
and conversion to RuntimeConfig.
"""

import pytest
import torch

from llm_dit.config_modular import (
    ComponentLibrary,
    DevicePlacement,
    DyPEConfig,
    EncoderDefinition,
    FMTTConfig,
    HardwareProfile,
    ModelTypeConfig,
    ModularConfig,
    QuantizationPreset,
    SchedulerDefinition,
    SLGConfig,
    TransformerDefinition,
    VAEDefinition,
    is_modular_config,
)

pytestmark = pytest.mark.unit


class TestDevicePlacement:
    """Test DevicePlacement dataclass."""

    def test_default_values(self):
        dp = DevicePlacement()
        assert dp.encoder == "auto"
        assert dp.dit == "auto"
        assert dp.vae == "auto"
        assert dp.siglip == "cpu"

    def test_resolve_auto(self):
        dp = DevicePlacement()
        resolved = dp.resolve("auto")
        # Should resolve to one of the available devices
        assert resolved in ["cpu", "cuda", "mps"]

    def test_resolve_explicit(self):
        dp = DevicePlacement()
        assert dp.resolve("cpu") == "cpu"
        assert dp.resolve("cuda") == "cuda"

    def test_get_encoder_device(self):
        dp = DevicePlacement(encoder="cpu")
        assert dp.get_encoder_device() == "cpu"

    def test_get_dit_device(self):
        dp = DevicePlacement(dit="cuda")
        assert dp.get_dit_device() == "cuda"


class TestHardwareProfile:
    """Test HardwareProfile dataclass."""

    def test_default_values(self):
        profile = HardwareProfile()
        assert profile.dtype == "bfloat16"
        assert profile.attention_backend == "auto"
        assert profile.compile is False
        assert profile.tiled_vae is False
        assert profile.tile_size == 512
        assert profile.embedding_cache is False
        assert profile.use_custom_scheduler is True

    def test_get_torch_dtype(self):
        profile = HardwareProfile(dtype="bfloat16")
        assert profile.get_torch_dtype() == torch.bfloat16

        profile = HardwareProfile(dtype="float16")
        assert profile.get_torch_dtype() == torch.float16

        profile = HardwareProfile(dtype="float32")
        assert profile.get_torch_dtype() == torch.float32


class TestEncoderDefinition:
    """Test EncoderDefinition dataclass."""

    def test_default_values(self):
        encoder = EncoderDefinition()
        assert encoder.hidden_dim == 2560
        assert encoder.num_layers == 36
        assert encoder.default_hidden_layer == -2
        assert encoder.max_tokens == 2048
        assert "none" in encoder.quantization_options
        assert "8bit" in encoder.quantization_options

    def test_custom_values(self):
        encoder = EncoderDefinition(
            description="Test encoder",
            model_id="test/model",
            hidden_dim=3584,
            num_layers=28,
        )
        assert encoder.hidden_dim == 3584
        assert encoder.num_layers == 28


class TestTransformerDefinition:
    """Test TransformerDefinition dataclass."""

    def test_default_values(self):
        transformer = TransformerDefinition()
        assert transformer.num_layers == 30
        assert transformer.latent_channels == 16


class TestVAEDefinition:
    """Test VAEDefinition dataclass."""

    def test_default_values(self):
        vae = VAEDefinition()
        assert vae.latent_scaling == 0.13025
        assert vae.latent_channels == 4


class TestSchedulerDefinition:
    """Test SchedulerDefinition dataclass."""

    def test_default_values(self):
        scheduler = SchedulerDefinition()
        assert scheduler.type == "flow_matching"
        assert scheduler.shift is None

    def test_with_shift(self):
        scheduler = SchedulerDefinition(shift=3.0)
        assert scheduler.shift == 3.0


class TestModelTypeConfig:
    """Test ModelTypeConfig dataclass."""

    def test_default_values(self):
        model = ModelTypeConfig()
        assert model.default_steps == 9
        assert model.default_cfg_scale == 0.0
        assert model.default_resolution == 1024
        assert model.cpu_offload is True
        assert model.inherits is None

    def test_custom_values(self):
        model = ModelTypeConfig(
            pipeline_class="ZImagePipeline",
            text_encoder="qwen3_4b",
            dit="zimage_dit",
            default_steps=12,
        )
        assert model.pipeline_class == "ZImagePipeline"
        assert model.text_encoder == "qwen3_4b"
        assert model.default_steps == 12


class TestQuantizationPreset:
    """Test QuantizationPreset dataclass."""

    def test_default_values(self):
        quant = QuantizationPreset()
        assert quant.text_encoder == "none"
        assert quant.transformer == "none"

    def test_custom_values(self):
        quant = QuantizationPreset(text_encoder="8bit", transformer="4bit")
        assert quant.text_encoder == "8bit"
        assert quant.transformer == "4bit"


class TestComponentLibrary:
    """Test ComponentLibrary dataclass."""

    def test_empty_library(self):
        library = ComponentLibrary()
        assert len(library.encoders) == 0
        assert len(library.transformers) == 0

    def test_get_encoder_missing(self):
        library = ComponentLibrary()
        with pytest.raises(KeyError) as exc_info:
            library.get_encoder("nonexistent")
        assert "nonexistent" in str(exc_info.value)

    def test_get_encoder_present(self):
        library = ComponentLibrary(
            encoders={"qwen3": EncoderDefinition(description="Qwen3 encoder")}
        )
        encoder = library.get_encoder("qwen3")
        assert encoder.description == "Qwen3 encoder"

    def test_get_transformer_missing(self):
        library = ComponentLibrary()
        with pytest.raises(KeyError) as exc_info:
            library.get_transformer("nonexistent")
        assert "nonexistent" in str(exc_info.value)

    def test_get_vae_missing(self):
        library = ComponentLibrary()
        with pytest.raises(KeyError) as exc_info:
            library.get_vae("nonexistent")
        assert "nonexistent" in str(exc_info.value)

    def test_get_scheduler_missing(self):
        library = ComponentLibrary()
        with pytest.raises(KeyError) as exc_info:
            library.get_scheduler("nonexistent")
        assert "nonexistent" in str(exc_info.value)


class TestDyPEConfig:
    """Test DyPEConfig dataclass."""

    def test_default_values(self):
        config = DyPEConfig()
        assert config.enabled is False
        assert config.method == "vision_yarn"
        assert config.dype_scale == 2.0
        assert config.base_resolution == 1024


class TestSLGConfig:
    """Test SLGConfig dataclass."""

    def test_default_values(self):
        config = SLGConfig()
        assert config.enabled is False
        assert config.scale == 2.5
        assert config.layers == [7, 8, 9, 10, 11, 12]


class TestFMTTConfig:
    """Test FMTTConfig dataclass."""

    def test_default_values(self):
        config = FMTTConfig()
        assert config.enabled is False
        assert config.guidance_scale == 1.0
        assert config.siglip_device == "cpu"


class TestModularConfigFromTOML:
    """Test ModularConfig.from_toml loading."""

    def test_load_basic_config(self, modular_config_file):
        config = ModularConfig.from_toml(modular_config_file, "test_zimage")

        # Check profile loaded
        assert config.profile.dtype == "bfloat16"
        assert config.profile.devices.encoder == "cuda"

        # Check model loaded
        assert config.model.pipeline_class == "ZImagePipeline"
        assert config.model.text_encoder == "qwen3_4b"

        # Check quantization loaded
        assert config.quantization.text_encoder == "none"

    def test_load_with_quantization(self, modular_config_file):
        config = ModularConfig.from_toml(modular_config_file, "test_quantized")

        assert config.quantization.text_encoder == "8bit"
        assert config.quantization.transformer == "none"

    def test_load_with_features(self, modular_config_file):
        config = ModularConfig.from_toml(modular_config_file, "test_with_dype")

        assert config.dype.enabled is True

    def test_load_missing_config_raises(self, modular_config_file):
        with pytest.raises(KeyError) as exc_info:
            ModularConfig.from_toml(modular_config_file, "nonexistent")
        assert "nonexistent" in str(exc_info.value)

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ModularConfig.from_toml(tmp_path / "missing.toml", "test")

    def test_load_legacy_config_raises(self, test_config_file):
        with pytest.raises(ValueError) as exc_info:
            ModularConfig.from_toml(test_config_file, "default")
        assert "modular config" in str(exc_info.value).lower()


class TestModularConfigInheritance:
    """Test model type inheritance resolution."""

    def test_inheritance_merges_values(self, modular_config_file):
        config = ModularConfig.from_toml(modular_config_file, "test_child")

        # Should have values from parent
        assert config.model.text_encoder == "qwen3_4b"
        assert config.model.vae == "zimage_vae"

        # Should have overridden values from child
        assert config.model.default_steps == 12

    def test_circular_inheritance_raises(self, circular_inheritance_config_file):
        with pytest.raises(ValueError) as exc_info:
            ModularConfig.from_toml(circular_inheritance_config_file, "test_circular")
        assert "circular" in str(exc_info.value).lower()


class TestModularConfigValidation:
    """Test ModularConfig.validate()."""

    def test_valid_config_no_errors(self, modular_config_file):
        config = ModularConfig.from_toml(modular_config_file, "test_zimage")
        errors = config.validate()
        assert len(errors) == 0

    def test_missing_encoder_reports_error(self):
        config = ModularConfig(
            model=ModelTypeConfig(text_encoder="nonexistent"),
            components=ComponentLibrary(),
        )
        errors = config.validate()
        assert any("nonexistent" in e for e in errors)


class TestModularConfigToRuntimeConfig:
    """Test conversion to RuntimeConfig."""

    def test_basic_conversion(self, modular_config_file):
        modular = ModularConfig.from_toml(modular_config_file, "test_zimage")
        runtime = modular.to_runtime_config()

        # Check model type
        assert runtime.model_type == "zimage"

        # Check generation defaults
        assert runtime.steps == 9
        assert runtime.guidance_scale == 0.0

        # Check device placement
        assert runtime.encoder_device == "cuda"

    def test_qwen_model_type(self, modular_config_file):
        modular = ModularConfig.from_toml(modular_config_file, "test_qwenimage")
        runtime = modular.to_runtime_config()

        assert runtime.model_type == "qwenimage"
        assert runtime.qwen_image_cfg_scale == 4.0

    def test_overrides_applied(self, modular_config_file):
        modular = ModularConfig.from_toml(modular_config_file, "test_with_overrides")
        runtime = modular.to_runtime_config()

        assert runtime.steps == 15
        assert runtime.width == 2048

    def test_dype_settings_transferred(self, modular_config_file):
        modular = ModularConfig.from_toml(modular_config_file, "test_with_dype")
        runtime = modular.to_runtime_config()

        assert runtime.dype_enabled is True
        assert runtime.dype_method == "vision_yarn"


class TestIsModularConfig:
    """Test is_modular_config() detection."""

    def test_modular_config_detected(self, modular_config_file):
        assert is_modular_config(modular_config_file) is True

    def test_legacy_config_not_detected(self, test_config_file):
        assert is_modular_config(test_config_file) is False

    def test_missing_file_returns_false(self, tmp_path):
        assert is_modular_config(tmp_path / "missing.toml") is False


# Fixtures
@pytest.fixture
def modular_config_file(tmp_path):
    """Create a temporary modular config file."""
    config_path = tmp_path / "modular_config.toml"
    config_path.write_text(
        '''
# Hardware profiles
[profiles.default]
description = "Default profile"
dtype = "bfloat16"
attention_backend = "auto"

[profiles.default.devices]
encoder = "cuda"
dit = "cuda"
vae = "cuda"

# Components
[encoders.qwen3_4b]
description = "Qwen3-4B encoder"
model_id = "Qwen/Qwen3-4B"
hidden_dim = 2560
num_layers = 36

[encoders.qwen2_5_vl]
description = "Qwen2.5-VL encoder"
model_id = "Qwen/Qwen2.5-VL-7B"
hidden_dim = 3584
num_layers = 28

[transformers.zimage_dit]
description = "Z-Image DiT"
num_layers = 40

[transformers.qwenimage_dit]
description = "Qwen-Image DiT"
num_layers = 30

[vaes.zimage_vae]
description = "Z-Image VAE"
latent_channels = 16

[schedulers.flow_turbo]
description = "Flow matching turbo"
type = "flow_matching"
shift = 3.0

# Model types
[models.zimage]
description = "Z-Image pipeline"
pipeline_class = "ZImagePipeline"
text_encoder = "qwen3_4b"
dit = "zimage_dit"
vae = "zimage_vae"
scheduler = "flow_turbo"
default_steps = 9
default_cfg_scale = 0.0
default_resolution = 1024

[models.qwenimage]
description = "Qwen-Image pipeline"
pipeline_class = "QwenImageDiffusersPipeline"
text_encoder = "qwen2_5_vl"
dit = "qwenimage_dit"
vae = "zimage_vae"
scheduler = "flow_turbo"
default_steps = 50
default_cfg_scale = 4.0
default_resolution = 1024

[models.zimage_child]
description = "Child of Z-Image"
inherits = "zimage"
default_steps = 12

# Quantization
[quantization.none]
description = "No quantization"
text_encoder = "none"
transformer = "none"

[quantization.balanced]
description = "Balanced quantization"
text_encoder = "8bit"
transformer = "none"

# Features
[features.dype]
enabled = false
method = "vision_yarn"
dype_scale = 2.0

# Combined configs
[configs.test_zimage]
description = "Test Z-Image config"
profile = "default"
model = "zimage"
quantization = "none"
features = []

[configs.test_quantized]
description = "Test with quantization"
profile = "default"
model = "zimage"
quantization = "balanced"
features = []

[configs.test_with_dype]
description = "Test with DyPE"
profile = "default"
model = "zimage"
quantization = "none"
features = ["dype"]

[configs.test_qwenimage]
description = "Test Qwen-Image"
profile = "default"
model = "qwenimage"
quantization = "none"
features = []

[configs.test_child]
description = "Test inheritance"
profile = "default"
model = "zimage_child"
quantization = "none"
features = []

[configs.test_with_overrides]
description = "Test with overrides"
profile = "default"
model = "zimage"
quantization = "none"
features = []

[configs.test_with_overrides.overrides]
steps = 15
width = 2048
'''
    )
    return config_path


@pytest.fixture
def circular_inheritance_config_file(tmp_path):
    """Create a config file with circular inheritance."""
    config_path = tmp_path / "circular_config.toml"
    config_path.write_text(
        '''
[models.model_a]
description = "Model A"
inherits = "model_b"
pipeline_class = "TestPipeline"

[models.model_b]
description = "Model B"
inherits = "model_a"
pipeline_class = "TestPipeline"

[configs.test_circular]
profile = "default"
model = "model_a"
'''
    )
    return config_path
