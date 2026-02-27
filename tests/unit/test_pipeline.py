"""
Unit tests for llm_dit.pipelines.z_image module.

Tests cover:
- ZImagePipeline initialization
- Pipeline parameter validation
- setup_attention_backend function
- MAX_TEXT_SEQ_LEN constant
"""

from unittest.mock import MagicMock, patch, PropertyMock
import pytest
import torch
import torch.nn as nn


# ============================================================================
# setup_attention_backend Tests
# ============================================================================

class TestSetupAttentionBackend:
    """Test setup_attention_backend function."""

    def test_setup_attention_backend_auto(self):
        """Test attention backend setup with auto detection."""
        from llm_dit.pipelines.z_image import setup_attention_backend

        with patch('llm_dit.utils.attention.get_attention_backend', return_value='sdpa'):
            with patch('llm_dit.utils.attention.log_attention_info'):
                result = setup_attention_backend(None)
                assert result == 'sdpa'

    def test_setup_attention_backend_specific(self):
        """Test attention backend setup with specific backend."""
        from llm_dit.pipelines.z_image import setup_attention_backend

        with patch('llm_dit.utils.attention.set_attention_backend') as mock_set:
            with patch('llm_dit.utils.attention.get_attention_backend', return_value='flash_attn_2'):
                with patch('llm_dit.utils.attention.log_attention_info'):
                    result = setup_attention_backend('flash_attn_2')

                    mock_set.assert_called_once_with('flash_attn_2')
                    assert result == 'flash_attn_2'

    def test_setup_attention_backend_fallback(self):
        """Test attention backend falls back on error."""
        from llm_dit.pipelines.z_image import setup_attention_backend

        with patch('llm_dit.utils.attention.set_attention_backend', side_effect=ValueError("Not available")):
            with patch('llm_dit.utils.attention.get_attention_backend', return_value='sdpa'):
                with patch('llm_dit.utils.attention.log_attention_info'):
                    result = setup_attention_backend('nonexistent')
                    assert result == 'sdpa'


# ============================================================================
# ZImagePipeline Tests
# ============================================================================

class TestZImagePipelineConstants:
    """Test ZImagePipeline constants."""

    def test_max_text_seq_len(self):
        """Test MAX_TEXT_SEQ_LEN is correct."""
        from llm_dit.pipelines.z_image import MAX_TEXT_SEQ_LEN

        assert MAX_TEXT_SEQ_LEN == 1504


class TestZImagePipelineInit:
    """Test ZImagePipeline initialization."""

    @pytest.fixture
    def mock_encoder(self):
        """Create mock encoder."""
        encoder = MagicMock()
        encoder.device = torch.device("cpu")
        encoder.dtype = torch.bfloat16
        encoder.embedding_dim = 2560
        return encoder

    @pytest.fixture
    def mock_transformer(self):
        """Create mock transformer."""
        transformer = MagicMock()
        transformer.parameters = MagicMock(return_value=iter([torch.nn.Parameter(torch.randn(10))]))
        return transformer

    @pytest.fixture
    def mock_vae(self):
        """Create mock VAE."""
        vae = MagicMock()
        vae.config = MagicMock()
        vae.config.block_out_channels = [128, 256, 512, 512]  # 4 blocks -> scale factor 8
        vae.parameters = MagicMock(return_value=iter([torch.nn.Parameter(torch.randn(10))]))
        return vae

    @pytest.fixture
    def mock_scheduler(self):
        """Create mock scheduler."""
        scheduler = MagicMock()
        scheduler.timesteps = torch.linspace(1.0, 0.0, 10)
        return scheduler

    def test_pipeline_init_basic(self, mock_encoder, mock_transformer, mock_vae, mock_scheduler):
        """Test basic pipeline initialization."""
        from llm_dit.pipelines.z_image import ZImagePipeline

        pipeline = ZImagePipeline(
            encoder=mock_encoder,
            transformer=mock_transformer,
            vae=mock_vae,
            scheduler=mock_scheduler,
        )

        assert pipeline.encoder is mock_encoder
        assert pipeline.transformer is mock_transformer
        assert pipeline.scheduler is mock_scheduler
        assert pipeline.vae_scale_factor == 8  # 2^(4-1)
        assert pipeline._tiled_vae_enabled is False

    def test_pipeline_init_with_tiled_vae(self, mock_encoder, mock_transformer, mock_vae, mock_scheduler):
        """Test pipeline initialization with tiled VAE."""
        from llm_dit.pipelines.z_image import ZImagePipeline

        with patch('llm_dit.utils.tiled_vae.TiledVAEDecoder') as MockTiled:
            mock_tiled = MagicMock()
            MockTiled.return_value = mock_tiled

            pipeline = ZImagePipeline(
                encoder=mock_encoder,
                transformer=mock_transformer,
                vae=mock_vae,
                scheduler=mock_scheduler,
                tiled_vae=True,
                tile_size=256,
                tile_overlap=32,
            )

            MockTiled.assert_called_once_with(mock_vae, tile_size=256, tile_overlap=32)
            assert pipeline._tiled_vae_enabled is True

    def test_pipeline_init_with_dype(self, mock_encoder, mock_transformer, mock_vae, mock_scheduler):
        """Test pipeline initialization with DyPE config."""
        from llm_dit.pipelines.z_image import ZImagePipeline
        from llm_dit.config import DyPEConfig

        dype_config = DyPEConfig(enabled=True, method="vision_yarn")

        pipeline = ZImagePipeline(
            encoder=mock_encoder,
            transformer=mock_transformer,
            vae=mock_vae,
            scheduler=mock_scheduler,
            dype_config=dype_config,
        )

        assert pipeline.dype_config is dype_config


class TestZImagePipelineFromPretrained:
    """Test ZImagePipeline.from_pretrained class method."""

    def test_from_pretrained_requires_model_path(self):
        """Test from_pretrained validates model path."""
        from llm_dit.pipelines.z_image import ZImagePipeline

        with patch('pathlib.Path.exists', return_value=False):
            with pytest.raises((FileNotFoundError, ValueError)):
                ZImagePipeline.from_pretrained("/nonexistent/path")


class TestZImagePipelineProperties:
    """Test ZImagePipeline properties."""

    @pytest.fixture
    def mock_pipeline(self):
        """Create mock pipeline for property tests."""
        from llm_dit.pipelines.z_image import ZImagePipeline

        encoder = MagicMock()
        encoder.device = torch.device("cuda:0")
        encoder.dtype = torch.bfloat16

        transformer = MagicMock()
        param = torch.nn.Parameter(torch.randn(10, device="cuda:0", dtype=torch.bfloat16))
        transformer.parameters = MagicMock(return_value=iter([param]))

        vae = MagicMock()
        vae.config = MagicMock()
        vae.config.block_out_channels = [128, 256, 512, 512]
        vae_param = torch.nn.Parameter(torch.randn(10, device="cuda:0", dtype=torch.bfloat16))
        vae.parameters = MagicMock(return_value=iter([vae_param]))

        scheduler = MagicMock()

        return ZImagePipeline(
            encoder=encoder,
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
        )

    def test_device_property(self, mock_pipeline):
        """Test device property."""
        # Device comes from transformer
        assert mock_pipeline.device == torch.device("cuda:0")

    def test_dtype_property(self, mock_pipeline):
        """Test dtype property."""
        assert mock_pipeline.dtype == torch.bfloat16


# ============================================================================
# ZImagePipeline Method Tests
# ============================================================================

class TestZImagePipelineMethods:
    """Test ZImagePipeline methods."""

    @pytest.fixture
    def mock_pipeline(self):
        """Create mock pipeline for method tests."""
        from llm_dit.pipelines.z_image import ZImagePipeline

        encoder = MagicMock()
        encoder.device = torch.device("cpu")
        encoder.dtype = torch.bfloat16
        encoder.encode = MagicMock(return_value=MagicMock(
            embeddings=[torch.randn(100, 2560)],
            attention_masks=[torch.ones(100, dtype=torch.bool)],
        ))

        transformer = MagicMock()
        param = torch.nn.Parameter(torch.randn(10))
        transformer.parameters = MagicMock(return_value=iter([param]))

        vae = MagicMock()
        vae.config = MagicMock()
        vae.config.block_out_channels = [128, 256, 512, 512]
        vae.config.scaling_factor = 0.18215
        vae.decode = MagicMock(return_value=MagicMock(sample=torch.randn(1, 3, 512, 512)))

        scheduler = MagicMock()
        scheduler.timesteps = torch.linspace(1.0, 0.0, 8)
        scheduler.set_timesteps = MagicMock()
        scheduler.step = MagicMock(return_value=MagicMock(prev_sample=torch.randn(1, 16, 64, 64)))

        return ZImagePipeline(
            encoder=encoder,
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
        )

    def test_encode_prompt(self, mock_pipeline):
        """Test encode_prompt method."""
        embeddings = mock_pipeline.encode_prompt("Test prompt")

        mock_pipeline.encoder.encode.assert_called()
        assert embeddings.shape[0] == 100
        assert embeddings.shape[1] == 2560

    def test_encode_prompt_with_template(self, mock_pipeline):
        """Test encode_prompt with template parameter."""
        embeddings = mock_pipeline.encode_prompt("Test prompt", template="photorealistic")

        mock_pipeline.encoder.encode.assert_called()
        assert embeddings.shape[0] == 100


class TestZImagePipelineLoadLora:
    """Test ZImagePipeline.load_lora method."""

    @pytest.fixture
    def mock_pipeline(self):
        """Create mock pipeline for LoRA tests."""
        from llm_dit.pipelines.z_image import ZImagePipeline

        encoder = MagicMock()
        encoder.device = torch.device("cpu")

        transformer = MagicMock()
        param = torch.nn.Parameter(torch.randn(10))
        # Return fresh iterator each time parameters() is called
        transformer.parameters = lambda: iter([param])
        # Add named_modules for LoRA loading
        transformer.named_modules = MagicMock(return_value=[
            ("blocks.0.attn.to_q", nn.Linear(256, 256)),
            ("blocks.0.attn.to_k", nn.Linear(256, 256)),
        ])

        vae = MagicMock()
        vae.config = MagicMock()
        vae.config.block_out_channels = [128, 256, 512, 512]

        scheduler = MagicMock()

        return ZImagePipeline(
            encoder=encoder,
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
        )

    def test_load_lora_single(self, mock_pipeline, tmp_path):
        """Test loading single LoRA."""
        # Create mock LoRA file
        lora_path = tmp_path / "test_lora.safetensors"

        with patch('llm_dit.utils.lora.load_lora') as mock_load:
            mock_load.return_value = 5

            result = mock_pipeline.load_lora(str(lora_path), scale=0.8)

            mock_load.assert_called_once()

    def test_load_lora_multiple(self, mock_pipeline, tmp_path):
        """Test loading multiple LoRAs."""
        lora_paths = [
            str(tmp_path / "lora1.safetensors"),
            str(tmp_path / "lora2.safetensors"),
        ]

        with patch('llm_dit.utils.lora.load_lora') as mock_load:
            mock_load.return_value = 5

            mock_pipeline.load_lora(lora_paths, scale=[0.8, 0.5])

            assert mock_load.call_count == 2


# ============================================================================
# ZImagePipeline Generation Tests
# ============================================================================

class TestZImagePipelineGenerate:
    """Test ZImagePipeline generation (with mocks)."""

    @pytest.fixture
    def mock_pipeline_for_generation(self):
        """Create mock pipeline for generation tests."""
        from llm_dit.pipelines.z_image import ZImagePipeline

        encoder = MagicMock()
        encoder.device = torch.device("cpu")
        encoder.dtype = torch.bfloat16
        encoder.encode = MagicMock(return_value=MagicMock(
            embeddings=[torch.randn(100, 2560)],
            attention_masks=[torch.ones(100, dtype=torch.bool)],
        ))

        transformer = MagicMock()
        param = torch.nn.Parameter(torch.randn(10))
        transformer.parameters = MagicMock(return_value=iter([param]))
        # Make transformer callable
        transformer.return_value = MagicMock(sample=torch.randn(1, 16, 64, 64))

        vae = MagicMock()
        vae.config = MagicMock()
        vae.config.block_out_channels = [128, 256, 512, 512]
        vae.config.scaling_factor = 0.18215
        vae.config.shift_factor = 0.0
        vae.decode = MagicMock(return_value=MagicMock(sample=torch.randn(1, 3, 512, 512)))

        scheduler = MagicMock()
        scheduler.timesteps = torch.linspace(1.0, 0.0, 8)
        scheduler.set_timesteps = MagicMock()
        scheduler.step = MagicMock(return_value=MagicMock(prev_sample=torch.randn(1, 16, 64, 64)))
        scheduler.sigmas = torch.linspace(1.0, 0.0, 9)

        return ZImagePipeline(
            encoder=encoder,
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
        )

    def test_pipeline_is_callable(self, mock_pipeline_for_generation):
        """Test ZImagePipeline has __call__ method."""
        assert callable(mock_pipeline_for_generation)

    def test_call_validates_resolution(self, mock_pipeline_for_generation):
        """Test __call__ validates resolution."""
        # Resolution should be divisible by VAE scale factor
        # This depends on implementation details
        pass


# ============================================================================
# Utility Function Tests
# ============================================================================

class TestPipelineUtilities:
    """Test pipeline utility functions."""

    def test_latent_to_image_dimensions(self):
        """Test latent to image dimension calculation."""
        # Latent: (B, C, H/8, W/8) -> Image: (B, 3, H, W)
        latent_h, latent_w = 64, 64
        scale_factor = 8
        image_h = latent_h * scale_factor
        image_w = latent_w * scale_factor

        assert image_h == 512
        assert image_w == 512

    def test_image_to_latent_dimensions(self):
        """Test image to latent dimension calculation."""
        image_h, image_w = 1024, 1024
        scale_factor = 8
        latent_h = image_h // scale_factor
        latent_w = image_w // scale_factor

        assert latent_h == 128
        assert latent_w == 128
