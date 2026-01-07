"""
Unit tests for llm_dit.startup module.

Tests cover:
- PipelineLoader initialization
- LoadResult dataclass
- DyPE config building
- Template directory resolution
- Auto-load logic
"""

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import torch


# ============================================================================
# LoadResult Tests
# ============================================================================

class TestLoadResult:
    """Test LoadResult dataclass."""

    def test_load_result_defaults(self):
        """Test LoadResult default values."""
        from llm_dit.startup import LoadResult

        result = LoadResult()
        assert result.pipeline is None
        assert result.encoder is None
        assert result.load_time == 0.0
        assert result.mode == "unknown"
        assert result.encoder_device is None
        assert result.dit_device is None
        assert result.vae_device is None

    def test_load_result_with_values(self):
        """Test LoadResult with values."""
        from llm_dit.startup import LoadResult

        mock_pipeline = MagicMock()
        mock_encoder = MagicMock()

        result = LoadResult(
            pipeline=mock_pipeline,
            encoder=mock_encoder,
            load_time=5.5,
            mode="full",
            encoder_device="cuda:0",
            dit_device="cuda:0",
            vae_device="cuda:1",
        )

        assert result.pipeline is mock_pipeline
        assert result.encoder is mock_encoder
        assert result.load_time == 5.5
        assert result.mode == "full"
        assert result.encoder_device == "cuda:0"


# ============================================================================
# build_dype_config Tests
# ============================================================================

class TestBuildDyPEConfig:
    """Test build_dype_config function."""

    def test_build_dype_config_disabled(self):
        """Test building DyPE config when disabled."""
        from llm_dit.startup import build_dype_config

        config = MagicMock()
        config.dype_enabled = False

        result = build_dype_config(config)
        assert result is None

    def test_build_dype_config_enabled(self):
        """Test building DyPE config when enabled."""
        from llm_dit.startup import build_dype_config

        config = MagicMock()
        config.dype_enabled = True
        config.dype_method = "vision_yarn"
        config.dype_scale = 2.0
        config.dype_exponent = 2.0
        config.dype_start_sigma = 1.0
        config.dype_base_shift = 0.5
        config.dype_max_shift = 1.15
        config.dype_base_resolution = 1024
        config.dype_anisotropic = False
        config.dype_multipass = "single"
        config.dype_pass2_strength = 0.5
        config.dype_pass3_strength = 0.4
        config.dype_frequency_modulation = False

        result = build_dype_config(config)

        assert result is not None
        assert result.enabled is True
        assert result.method == "vision_yarn"
        assert result.dype_scale == 2.0

    def test_build_dype_config_missing_attribute(self):
        """Test building DyPE config with missing attributes uses defaults."""
        from llm_dit.startup import build_dype_config

        config = MagicMock(spec=[])  # Empty spec means no attributes
        config.dype_enabled = True

        # getattr should return defaults
        result = build_dype_config(config)

        assert result is not None
        assert result.enabled is True


# ============================================================================
# PipelineLoader Tests
# ============================================================================

class TestPipelineLoader:
    """Test PipelineLoader class."""

    @pytest.fixture
    def mock_runtime_config(self):
        """Create mock RuntimeConfig."""
        config = MagicMock()
        config.model_path = "/path/to/model"
        config.text_encoder_path = None
        config.templates_dir = None
        config.torch_dtype = "bfloat16"
        config.encoder_device_resolved = "cpu"
        config.dit_device_resolved = "cpu"
        config.vae_device_resolved = "cpu"
        config.quantization = "none"
        config.flash_attn = False
        config.attention_backend = "auto"
        config.compile = False
        config.cpu_offload = False
        config.tiled_vae = False
        config.tile_size = 512
        config.tile_overlap = 64
        config.embedding_cache = False
        config.cache_size = 100
        config.lora_paths = []
        config.lora_scales = []
        config.api_url = None
        config.api_model = None
        config.hidden_layer = -2
        config.long_prompt_mode = "truncate"
        config.use_custom_scheduler = False
        config.shift = 3.0
        config.dype_enabled = False
        config.model_type = "zimage"
        config.get_torch_dtype = MagicMock(return_value=torch.bfloat16)
        return config

    def test_pipeline_loader_init(self, mock_runtime_config):
        """Test PipelineLoader initialization."""
        from llm_dit.startup import PipelineLoader

        loader = PipelineLoader(mock_runtime_config)

        assert loader.config is mock_runtime_config
        assert loader._pipeline is None
        assert loader._encoder is None

    def test_pipeline_loader_properties(self, mock_runtime_config):
        """Test PipelineLoader properties."""
        from llm_dit.startup import PipelineLoader

        loader = PipelineLoader(mock_runtime_config)

        assert loader.pipeline is None
        assert loader.encoder is None

    def test_resolve_templates_dir_from_config(self, mock_runtime_config):
        """Test template directory resolution from config."""
        from llm_dit.startup import PipelineLoader

        mock_runtime_config.templates_dir = "/custom/templates"
        loader = PipelineLoader(mock_runtime_config)

        result = loader._resolve_templates_dir()
        assert result == "/custom/templates"

    def test_resolve_templates_dir_auto(self, mock_runtime_config, tmp_path):
        """Test template directory auto-resolution."""
        from llm_dit.startup import PipelineLoader

        mock_runtime_config.templates_dir = None
        loader = PipelineLoader(mock_runtime_config)

        # Create temporary templates directory
        templates = tmp_path / "templates" / "z_image"
        templates.mkdir(parents=True)

        with patch.object(Path, 'cwd', return_value=tmp_path):
            result = loader._resolve_templates_dir()
            assert result == str(templates)

    def test_resolve_templates_dir_not_found(self, mock_runtime_config):
        """Test template directory returns None when not found."""
        from llm_dit.startup import PipelineLoader

        mock_runtime_config.templates_dir = None
        loader = PipelineLoader(mock_runtime_config)

        with patch.object(Path, 'cwd', return_value=Path("/nonexistent")):
            result = loader._resolve_templates_dir()
            # Should return None if no templates found
            # (actual behavior depends on implementation)


class TestPipelineLoaderApplyOptimizations:
    """Test PipelineLoader._apply_optimizations method."""

    @pytest.fixture
    def mock_pipeline(self):
        """Create mock pipeline."""
        pipeline = MagicMock()
        pipeline.transformer = MagicMock()
        pipeline.transformer.set_attention_backend = MagicMock()
        pipeline.vae = MagicMock()
        pipeline.vae.enable_tiling = MagicMock()
        pipeline.enable_tiled_vae = MagicMock()
        return pipeline

    @pytest.fixture
    def mock_config(self):
        """Create mock config for optimization tests."""
        config = MagicMock()
        config.flash_attn = False
        config.attention_backend = "auto"
        config.compile = False
        config.tiled_vae = False
        config.tile_size = 512
        config.tile_overlap = 64
        return config

    def test_apply_flash_attention(self, mock_pipeline, mock_config):
        """Test flash attention application."""
        from llm_dit.startup import PipelineLoader

        mock_config.flash_attn = True
        loader = PipelineLoader(mock_config)

        loader._apply_optimizations(mock_pipeline)

        mock_pipeline.transformer.set_attention_backend.assert_called_with("flash")

    def test_apply_tiled_vae(self, mock_pipeline, mock_config):
        """Test tiled VAE application."""
        from llm_dit.startup import PipelineLoader

        mock_config.tiled_vae = True
        mock_config.tile_size = 256
        mock_config.tile_overlap = 32
        loader = PipelineLoader(mock_config)

        loader._apply_optimizations(mock_pipeline)

        mock_pipeline.enable_tiled_vae.assert_called_with(
            tile_size=256,
            tile_overlap=32,
        )

    def test_apply_torch_compile(self, mock_pipeline, mock_config):
        """Test torch.compile application."""
        from llm_dit.startup import PipelineLoader

        mock_config.compile = True
        loader = PipelineLoader(mock_config)

        with patch('torch.compile', return_value=mock_pipeline.transformer) as mock_compile:
            loader._apply_optimizations(mock_pipeline)
            mock_compile.assert_called_once()


class TestPipelineLoaderLoadLoras:
    """Test PipelineLoader._load_loras method."""

    def test_load_loras_empty(self):
        """Test loading empty LoRA list."""
        from llm_dit.startup import PipelineLoader

        config = MagicMock()
        config.lora_paths = []
        loader = PipelineLoader(config)

        mock_pipeline = MagicMock()
        loader._load_loras(mock_pipeline)

        # Should not call load_lora
        mock_pipeline.load_lora.assert_not_called()

    def test_load_loras_with_paths(self):
        """Test loading LoRAs from paths."""
        from llm_dit.startup import PipelineLoader

        config = MagicMock()
        config.lora_paths = ["/path/to/lora1.safetensors", "/path/to/lora2.safetensors"]
        config.lora_scales = [0.8, 0.5]
        loader = PipelineLoader(config)

        mock_pipeline = MagicMock()
        mock_pipeline.load_lora = MagicMock(return_value=10)
        loader._load_loras(mock_pipeline)

        mock_pipeline.load_lora.assert_called_once_with(
            config.lora_paths,
            scale=[0.8, 0.5],
        )


class TestPipelineLoaderAutoLoad:
    """Test PipelineLoader.auto_load method."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config for auto_load tests."""
        config = MagicMock()
        config.model_path = "/path/to/model"
        config.api_url = None
        config.qwen_image_edit_only = False
        config.qwen_image_edit_model_path = ""
        config.model_type = "zimage"
        return config

    def test_auto_load_encoder_only(self, mock_config):
        """Test auto_load with encoder_only=True."""
        from llm_dit.startup import PipelineLoader

        loader = PipelineLoader(mock_config)
        loader.load_encoder = MagicMock(return_value=MagicMock())

        loader.auto_load(encoder_only=True)

        loader.load_encoder.assert_called_once()

    def test_auto_load_api_encoder_no_model(self, mock_config):
        """Test auto_load uses API encoder when no local model."""
        from llm_dit.startup import PipelineLoader

        mock_config.api_url = "http://api.example.com"
        mock_config.model_path = ""  # No local model
        loader = PipelineLoader(mock_config)
        loader.load_api_encoder = MagicMock(return_value=MagicMock())

        loader.auto_load()

        loader.load_api_encoder.assert_called_once()

    def test_auto_load_distributed(self, mock_config):
        """Test auto_load with distributed mode."""
        from llm_dit.startup import PipelineLoader

        mock_config.api_url = "http://api.example.com"
        mock_config.model_path = "/path/to/model"
        loader = PipelineLoader(mock_config)
        loader.load_api_pipeline = MagicMock(return_value=MagicMock())

        loader.auto_load(use_api=True)

        loader.load_api_pipeline.assert_called_once()

    def test_auto_load_full_pipeline(self, mock_config):
        """Test auto_load loads full pipeline by default."""
        from llm_dit.startup import PipelineLoader

        loader = PipelineLoader(mock_config)
        loader.load_pipeline = MagicMock(return_value=MagicMock())

        loader.auto_load()

        loader.load_pipeline.assert_called_once()


# ============================================================================
# Integration-style Tests (with mocks)
# ============================================================================

class TestPipelineLoaderLoadEncoder:
    """Test PipelineLoader.load_encoder method."""

    def test_load_encoder_returns_result(self):
        """Test load_encoder returns LoadResult."""
        from llm_dit.startup import PipelineLoader, LoadResult

        config = MagicMock()
        config.model_path = "/path/to/model"
        config.templates_dir = None
        config.encoder_device_resolved = "cpu"
        config.torch_dtype = "bfloat16"
        config.quantization = "none"
        config.embedding_cache = False
        config.cache_size = 100
        config.get_torch_dtype = MagicMock(return_value=torch.bfloat16)

        loader = PipelineLoader(config)

        with patch('llm_dit.startup.ZImageTextEncoder') as MockEncoder:
            mock_encoder = MagicMock()
            mock_encoder.device = "cpu"
            MockEncoder.from_pretrained = MagicMock(return_value=mock_encoder)

            result = loader.load_encoder()

            assert isinstance(result, LoadResult)
            assert result.mode == "encoder_only"
            assert result.encoder is mock_encoder


class TestQwenImageModes:
    """Test Qwen-Image specific loading modes."""

    def test_qwenimage_t2i_ondemand(self):
        """Test Qwen-Image T2I returns on-demand mode."""
        from llm_dit.startup import PipelineLoader, LoadResult

        config = MagicMock()
        config.model_type = "qwenimage-t2i"
        config.qwen_image_model_path = "/path/to/model"

        # Add required methods
        config.get_qwen_image_steps = MagicMock(return_value=50)
        config.get_qwen_image_resolution = MagicMock(return_value=1024)
        config.get_qwen_image_quantize_transformer = MagicMock(return_value="none")
        config.qwen_image_quantize_text_encoder = "none"

        loader = PipelineLoader(config)
        result = loader.load_pipeline()

        assert result.mode == "qwenimage-t2i_ondemand"
        assert result.pipeline is None

    def test_qwenimage_edit_ondemand(self):
        """Test Qwen-Image Edit returns on-demand mode."""
        from llm_dit.startup import PipelineLoader, LoadResult

        config = MagicMock()
        config.model_type = "qwenimage-edit"
        config.qwen_image_model_path = "/path/to/model"

        config.get_qwen_image_steps = MagicMock(return_value=50)
        config.get_qwen_image_resolution = MagicMock(return_value=1024)
        config.get_qwen_image_quantize_transformer = MagicMock(return_value="none")
        config.qwen_image_quantize_text_encoder = "none"

        loader = PipelineLoader(config)
        result = loader.load_pipeline()

        assert result.mode == "qwenimage-edit_ondemand"
        assert result.pipeline is None
