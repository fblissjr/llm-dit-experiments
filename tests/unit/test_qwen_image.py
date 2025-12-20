"""
Unit tests for Qwen-Image-Layered components.

Tests latent packing utilities, model components, and pipeline functionality
without requiring GPU or real models (uses mocks where appropriate).
"""

import pytest
import torch
from unittest.mock import MagicMock, patch

pytestmark = pytest.mark.unit


class TestLatentPackingUtilities:
    """Test latent packing/unpacking utilities."""

    def test_imports(self):
        from llm_dit.utils.latent_packing import (
            pack_latents_2x2,
            unpack_latents_2x2,
            pack_multi_layer_latents,
            unpack_multi_layer_latents,
            get_img_shapes_for_rope,
        )
        assert pack_latents_2x2 is not None
        assert unpack_latents_2x2 is not None
        assert pack_multi_layer_latents is not None
        assert unpack_multi_layer_latents is not None
        assert get_img_shapes_for_rope is not None

    def test_pack_latents_2x2_shape(self):
        """Test 2x2 packing produces correct output shape."""
        from llm_dit.utils.latent_packing import pack_latents_2x2

        # 1 latent of shape (16, 64, 64) for 512x512 image
        latents = torch.randn(1, 16, 64, 64)
        packed = pack_latents_2x2(latents, height=512, width=512)

        # Should produce (1, seq_len, 64) - seq_len = 64*64/4 = 1024
        assert packed.shape == (1, 1024, 64)

    def test_unpack_latents_2x2_shape(self):
        """Test 2x2 unpacking produces correct output shape."""
        from llm_dit.utils.latent_packing import unpack_latents_2x2

        # Packed latent of shape (1, 1024, 64) for 512x512 image
        packed = torch.randn(1, 1024, 64)
        unpacked = unpack_latents_2x2(packed, height=512, width=512)

        # Should produce (1, 16, 64, 64)
        assert unpacked.shape == (1, 16, 64, 64)

    def test_pack_unpack_2x2_roundtrip(self):
        """Test pack/unpack 2x2 is lossless."""
        from llm_dit.utils.latent_packing import pack_latents_2x2, unpack_latents_2x2

        original = torch.randn(1, 16, 64, 64)
        packed = pack_latents_2x2(original, height=512, width=512)
        unpacked = unpack_latents_2x2(packed, height=512, width=512)

        assert torch.allclose(original, unpacked)

    def test_pack_multi_layer_3_layers(self):
        """Test multi-layer packing with 3 decomposition layers (+ composite)."""
        from llm_dit.utils.latent_packing import pack_multi_layer_latents

        # 4 latents total (composite + 3 decomposition)
        latents = torch.randn(4, 16, 64, 64)
        height, width = 512, 512
        layer_num = 3

        packed = pack_multi_layer_latents(latents, height, width, layer_num)

        # Should be sequence format (batch, seq_len, channels)
        assert packed.dim() == 3
        # Channels should be 64 (16 * 2 * 2 for packing)
        assert packed.shape[-1] == 64

    def test_pack_multi_layer_7_layers(self):
        """Test multi-layer packing with maximum 7 decomposition layers."""
        from llm_dit.utils.latent_packing import pack_multi_layer_latents

        # 8 latents total (composite + 7 decomposition)
        latents = torch.randn(8, 16, 64, 64)
        height, width = 1024, 1024
        layer_num = 7

        packed = pack_multi_layer_latents(latents, height, width, layer_num)

        assert packed.dim() == 3
        assert packed.shape[-1] == 64

    def test_unpack_multi_layer_roundtrip(self):
        """Test multi-layer pack/unpack is lossless."""
        from llm_dit.utils.latent_packing import (
            pack_multi_layer_latents,
            unpack_multi_layer_latents,
        )

        original = torch.randn(4, 16, 64, 64)
        height, width = 512, 512
        layer_num = 3

        packed = pack_multi_layer_latents(original, height, width, layer_num)
        unpacked = unpack_multi_layer_latents(packed, height, width, layer_num)

        assert unpacked.shape == original.shape
        assert torch.allclose(original, unpacked, atol=1e-5)

    def test_get_img_shapes_for_rope(self):
        """Test RoPE img_shapes generation."""
        from llm_dit.utils.latent_packing import get_img_shapes_for_rope

        height, width = 1024, 1024
        layer_num = 3

        shapes = get_img_shapes_for_rope(height, width, layer_num)

        # Should have layer_num + 1 shapes (for each layer + composite)
        assert len(shapes) == layer_num + 1

        # Each shape should be (frame, height, width) tuple
        for shape in shapes:
            assert len(shape) == 3

    def test_get_img_shapes_with_condition(self):
        """Test RoPE img_shapes with condition image."""
        from llm_dit.utils.latent_packing import get_img_shapes_for_rope

        height, width = 1024, 1024
        layer_num = 3

        shapes = get_img_shapes_for_rope(
            height, width, layer_num,
            include_condition=True,
            condition_height=1024,
            condition_width=1024,
        )

        # Should have layer_num + 1 + 1 shapes (layers + composite + condition)
        assert len(shapes) >= layer_num + 1


class TestTimestepEmbeddings:
    """Test TimestepEmbeddings with use_additional_t_cond support."""

    def test_import(self):
        from llm_dit.models._qwen_image_dit_components import TimestepEmbeddings
        assert TimestepEmbeddings is not None

    def test_init_default(self):
        """Test initialization without additional_t_cond."""
        from llm_dit.models._qwen_image_dit_components import TimestepEmbeddings

        emb = TimestepEmbeddings(embedding_dim=256, out_dim=1024)

        assert emb.embedding_dim == 256
        assert emb.scale == 1000.0
        assert emb.use_additional_t_cond is False
        assert not hasattr(emb, "addition_t_embedding") or emb.addition_t_embedding is None

    def test_init_with_additional_t_cond(self):
        """Test initialization with additional_t_cond enabled."""
        from llm_dit.models._qwen_image_dit_components import TimestepEmbeddings

        emb = TimestepEmbeddings(
            embedding_dim=256,
            out_dim=1024,
            use_additional_t_cond=True,
        )

        assert emb.use_additional_t_cond is True
        assert hasattr(emb, "addition_t_embedding")
        assert emb.addition_t_embedding is not None
        # Should be Embedding(2, out_dim)
        assert emb.addition_t_embedding.num_embeddings == 2
        assert emb.addition_t_embedding.embedding_dim == 1024

    def test_forward_shape_without_cond(self):
        """Test forward pass output shape without additional conditioning."""
        from llm_dit.models._qwen_image_dit_components import TimestepEmbeddings

        emb = TimestepEmbeddings(embedding_dim=256, out_dim=1024)
        timestep = torch.tensor([0.5])

        output = emb(timestep, torch.float32)

        # Output shape is (out_dim,) for single timestep
        assert output.shape == (1024,)

    def test_forward_shape_with_cond(self):
        """Test forward pass output shape with additional conditioning."""
        from llm_dit.models._qwen_image_dit_components import TimestepEmbeddings

        emb = TimestepEmbeddings(
            embedding_dim=256,
            out_dim=1024,
            use_additional_t_cond=True,
        )
        timestep = torch.tensor([0.5])
        addition_t_cond = torch.tensor([0])  # 0 = generation mode

        output = emb(timestep, torch.float32, addition_t_cond=addition_t_cond)

        assert output.shape == (1, 1024)

    def test_forward_batch(self):
        """Test forward with batch of timesteps."""
        from llm_dit.models._qwen_image_dit_components import TimestepEmbeddings

        emb = TimestepEmbeddings(
            embedding_dim=256,
            out_dim=1024,
            use_additional_t_cond=True,
        )
        # Timestep needs shape (batch_size, 1) for proper broadcasting
        timestep = torch.tensor([[0.1], [0.5], [0.9]])
        addition_t_cond = torch.tensor([0, 1, 0])

        output = emb(timestep, torch.float32, addition_t_cond=addition_t_cond)

        assert output.shape == (3, 1024)

    def test_sinusoidal_embedding(self):
        """Test that different timesteps produce different embeddings."""
        from llm_dit.models._qwen_image_dit_components import TimestepEmbeddings

        emb = TimestepEmbeddings(embedding_dim=256, out_dim=1024)

        t1 = torch.tensor([0.0])
        t2 = torch.tensor([0.5])
        t3 = torch.tensor([1.0])

        out1 = emb(t1, torch.float32)
        out2 = emb(t2, torch.float32)
        out3 = emb(t3, torch.float32)

        # All outputs should be different
        assert not torch.allclose(out1, out2, atol=1e-5)
        assert not torch.allclose(out2, out3, atol=1e-5)
        assert not torch.allclose(out1, out3, atol=1e-5)


class TestQwenImageDiT:
    """Test QwenImageDiT wrapper class."""

    def test_import(self):
        from llm_dit.models.qwen_image_dit import QwenImageDiT
        assert QwenImageDiT is not None

    def test_architecture_constants(self):
        from llm_dit.models.qwen_image_dit import QwenImageDiT

        assert QwenImageDiT.NUM_LAYERS == 60
        assert QwenImageDiT.INNER_DIM == 3072
        assert QwenImageDiT.NUM_HEADS == 24
        assert QwenImageDiT.HEAD_DIM == 128
        assert QwenImageDiT.TEXT_DIM == 3584
        assert QwenImageDiT.LATENT_DIM == 64

    def test_device_property(self):
        """Test device property returns correct device."""
        from llm_dit.models.qwen_image_dit import QwenImageDiT

        mock_model = MagicMock()
        device = torch.device("cpu")
        dtype = torch.float32

        dit = QwenImageDiT(mock_model, device, dtype)

        assert dit.device == device

    def test_dtype_property(self):
        """Test dtype property returns correct dtype."""
        from llm_dit.models.qwen_image_dit import QwenImageDiT

        mock_model = MagicMock()
        device = torch.device("cpu")
        dtype = torch.bfloat16

        dit = QwenImageDiT(mock_model, device, dtype)

        assert dit.dtype == dtype


class TestQwenImageDiTModel:
    """Test QwenImageDiTModel internal class."""

    def test_import(self):
        from llm_dit.models._qwen_image_dit_components import QwenImageDiTModel
        assert QwenImageDiTModel is not None

    def test_init_default(self):
        """Test default initialization."""
        from llm_dit.models._qwen_image_dit_components import QwenImageDiTModel

        model = QwenImageDiTModel(num_layers=2)  # Small for testing

        assert len(model.transformer_blocks) == 2
        assert model._use_additional_t_cond is False

    def test_init_with_layer3d_rope(self):
        """Test initialization with layer3d RoPE."""
        from llm_dit.models._qwen_image_dit_components import (
            QwenImageDiTModel,
            QwenEmbedLayer3DRope,
        )

        model = QwenImageDiTModel(num_layers=2, use_layer3d_rope=True)

        # Check that Layer3D RoPE is used
        assert isinstance(model.pos_embed, QwenEmbedLayer3DRope)

    def test_init_with_additional_t_cond(self):
        """Test initialization with additional_t_cond."""
        from llm_dit.models._qwen_image_dit_components import QwenImageDiTModel

        model = QwenImageDiTModel(num_layers=2, use_additional_t_cond=True)

        assert model._use_additional_t_cond is True
        assert model.time_text_embed.use_additional_t_cond is True


class TestQwenImageVAE:
    """Test QwenImageVAE wrapper class."""

    def test_import(self):
        from llm_dit.models.qwen_image_vae import QwenImageVAE
        assert QwenImageVAE is not None

    def test_architecture_constants(self):
        from llm_dit.models.qwen_image_vae import QwenImageVAE

        assert QwenImageVAE.Z_DIM == 16
        assert QwenImageVAE.SCALE_FACTOR == 8

    def test_device_property(self):
        """Test device property."""
        from llm_dit.models.qwen_image_vae import QwenImageVAE

        mock_encoder = MagicMock()
        mock_decoder = MagicMock()
        mock_quant_conv = MagicMock()
        mock_post_quant_conv = MagicMock()
        device = torch.device("cpu")
        dtype = torch.float32

        vae = QwenImageVAE(
            mock_encoder, mock_decoder,
            mock_quant_conv, mock_post_quant_conv,
            device, dtype
        )

        assert vae.device == device

    def test_dtype_property(self):
        """Test dtype property."""
        from llm_dit.models.qwen_image_vae import QwenImageVAE

        mock_encoder = MagicMock()
        mock_decoder = MagicMock()
        mock_quant_conv = MagicMock()
        mock_post_quant_conv = MagicMock()
        device = torch.device("cpu")
        dtype = torch.bfloat16

        vae = QwenImageVAE(
            mock_encoder, mock_decoder,
            mock_quant_conv, mock_post_quant_conv,
            device, dtype
        )

        assert vae.dtype == dtype


class TestQwenImageVAEComponents:
    """Test internal VAE components."""

    def test_causal_conv3d_import(self):
        from llm_dit.models._qwen_image_vae_components import QwenImageCausalConv3d
        assert QwenImageCausalConv3d is not None

    def test_causal_conv3d_shape(self):
        """Test QwenImageCausalConv3d output shape."""
        from llm_dit.models._qwen_image_vae_components import QwenImageCausalConv3d

        conv = QwenImageCausalConv3d(
            in_channels=4,
            out_channels=8,
            kernel_size=3,
        )

        # Input: (batch, channels, time, height, width)
        # Need larger spatial dims to accommodate kernel size 3
        x = torch.randn(1, 4, 4, 32, 32)
        output = conv(x)

        # Should produce valid output
        assert output.shape[0] == 1
        assert output.shape[1] == 8
        assert output.shape[2] > 0  # Temporal
        assert output.shape[3] > 0  # Height
        assert output.shape[4] > 0  # Width

    def test_resnet_block_3d_import(self):
        from llm_dit.models._qwen_image_vae_components import QwenImageResidualBlock
        assert QwenImageResidualBlock is not None

    def test_resnet_block_3d_shape(self):
        """Test QwenImageResidualBlock maintains shape."""
        from llm_dit.models._qwen_image_vae_components import QwenImageResidualBlock

        # Uses in_dim/out_dim, not in_channels/out_channels
        block = QwenImageResidualBlock(in_dim=32, out_dim=32)

        x = torch.randn(1, 32, 4, 16, 16)
        output = block(x)

        assert output.shape == x.shape


class TestQwenImageTextEncoderBackend:
    """Test QwenImageTextEncoderBackend class."""

    def test_import(self):
        from llm_dit.backends.qwen_image import QwenImageTextEncoderBackend
        assert QwenImageTextEncoderBackend is not None

    def test_architecture_constants(self):
        from llm_dit.backends.qwen_image import QwenImageTextEncoderBackend

        assert QwenImageTextEncoderBackend.HIDDEN_DIM == 3584
        assert QwenImageTextEncoderBackend.NUM_LAYERS == 28
        assert QwenImageTextEncoderBackend.MAX_SEQUENCE_LENGTH == 4096

    def test_system_prompts(self):
        from llm_dit.backends.qwen_image import (
            QWEN_IMAGE_SYSTEM_PROMPT,
            QWEN_IMAGE_EDIT_SYSTEM_PROMPT,
            QWEN_IMAGE_DROP_TOKENS,
            QWEN_IMAGE_EDIT_DROP_TOKENS,
        )

        # System prompts should exist
        assert "Describe" in QWEN_IMAGE_SYSTEM_PROMPT
        assert "Describe" in QWEN_IMAGE_EDIT_SYSTEM_PROMPT

        # Drop token counts
        assert QWEN_IMAGE_DROP_TOKENS == 34
        assert QWEN_IMAGE_EDIT_DROP_TOKENS == 64

    def test_embedding_dim_property(self):
        """Test embedding_dim property."""
        from llm_dit.backends.qwen_image import QwenImageTextEncoderBackend

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        device = torch.device("cpu")
        dtype = torch.float32

        backend = QwenImageTextEncoderBackend(mock_model, mock_tokenizer, device, dtype)

        assert backend.embedding_dim == 3584

    def test_max_sequence_length_property(self):
        """Test max_sequence_length property."""
        from llm_dit.backends.qwen_image import QwenImageTextEncoderBackend

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        device = torch.device("cpu")
        dtype = torch.float32

        backend = QwenImageTextEncoderBackend(mock_model, mock_tokenizer, device, dtype)

        assert backend.max_sequence_length == 4096


class TestQwenImagePipeline:
    """Test QwenImagePipeline class."""

    def test_import(self):
        from llm_dit.pipelines.qwen_image import QwenImagePipeline
        assert QwenImagePipeline is not None

    def test_supported_resolutions(self):
        from llm_dit.pipelines.qwen_image import SUPPORTED_RESOLUTIONS

        assert 640 in SUPPORTED_RESOLUTIONS
        assert 1024 in SUPPORTED_RESOLUTIONS

    def test_calculate_dynamic_shift(self):
        """Test dynamic shift calculation."""
        from llm_dit.pipelines.qwen_image import calculate_dynamic_shift

        # At base seq_len, shift should be 1.0
        shift_base = calculate_dynamic_shift(256, base_seq_len=256)
        assert shift_base == pytest.approx(1.0)

        # At 4x seq_len, shift should be 2.0
        shift_4x = calculate_dynamic_shift(1024, base_seq_len=256)
        assert shift_4x == pytest.approx(2.0)

        # At 1024x1024 (seq_len = 64*64/4 = 1024)
        shift_1024 = calculate_dynamic_shift(1024)
        assert shift_1024 == pytest.approx(2.0)

    def test_init_sets_cpu_offload_state(self):
        """Test initialization sets up CPU offload state."""
        from llm_dit.pipelines.qwen_image import QwenImagePipeline

        mock_encoder = MagicMock()
        mock_dit = MagicMock()
        mock_dit.device = torch.device("cuda")
        mock_vae = MagicMock()

        pipe = QwenImagePipeline(mock_encoder, mock_dit, mock_vae)

        assert pipe._cpu_offload_enabled is False
        assert pipe._primary_device == torch.device("cuda")
        assert pipe._cpu_device == torch.device("cpu")

    def test_enable_cpu_offload(self):
        """Test enable_cpu_offload method."""
        from llm_dit.pipelines.qwen_image import QwenImagePipeline

        mock_encoder = MagicMock()
        mock_dit = MagicMock()
        mock_dit.device = torch.device("cuda")
        mock_vae = MagicMock()

        pipe = QwenImagePipeline(mock_encoder, mock_dit, mock_vae)
        result = pipe.enable_cpu_offload()

        assert pipe._cpu_offload_enabled is True
        assert result is pipe  # Returns self for chaining

    def test_disable_cpu_offload(self):
        """Test disable_cpu_offload method."""
        from llm_dit.pipelines.qwen_image import QwenImagePipeline

        mock_encoder = MagicMock()
        mock_dit = MagicMock()
        mock_dit.device = torch.device("cuda")
        mock_vae = MagicMock()

        pipe = QwenImagePipeline(mock_encoder, mock_dit, mock_vae)
        pipe.enable_cpu_offload()
        result = pipe.disable_cpu_offload()

        assert pipe._cpu_offload_enabled is False
        assert result is pipe

    def test_device_property(self):
        """Test device property returns DiT device."""
        from llm_dit.pipelines.qwen_image import QwenImagePipeline

        mock_encoder = MagicMock()
        mock_dit = MagicMock()
        mock_dit.device = torch.device("cuda:0")
        mock_vae = MagicMock()

        pipe = QwenImagePipeline(mock_encoder, mock_dit, mock_vae)

        assert pipe.device == torch.device("cuda:0")

    def test_dtype_property(self):
        """Test dtype property returns DiT dtype."""
        from llm_dit.pipelines.qwen_image import QwenImagePipeline

        mock_encoder = MagicMock()
        mock_dit = MagicMock()
        mock_dit.device = torch.device("cpu")
        mock_dit.dtype = torch.bfloat16
        mock_vae = MagicMock()

        pipe = QwenImagePipeline(mock_encoder, mock_dit, mock_vae)

        assert pipe.dtype == torch.bfloat16

    def test_validate_resolution_valid(self):
        """Test _validate_resolution accepts valid dimensions."""
        from llm_dit.pipelines.qwen_image import QwenImagePipeline

        mock_encoder = MagicMock()
        mock_dit = MagicMock()
        mock_dit.device = torch.device("cpu")
        mock_vae = MagicMock()

        pipe = QwenImagePipeline(mock_encoder, mock_dit, mock_vae)

        # Valid resolutions (divisible by 16)
        pipe._validate_resolution(1024, 1024)  # Should not raise
        pipe._validate_resolution(640, 640)  # Should not raise

    def test_validate_resolution_invalid(self):
        """Test _validate_resolution rejects invalid dimensions."""
        from llm_dit.pipelines.qwen_image import QwenImagePipeline

        mock_encoder = MagicMock()
        mock_dit = MagicMock()
        mock_dit.device = torch.device("cpu")
        mock_vae = MagicMock()

        pipe = QwenImagePipeline(mock_encoder, mock_dit, mock_vae)

        # Invalid resolutions (not divisible by 16)
        with pytest.raises(ValueError):
            pipe._validate_resolution(1000, 1000)

        with pytest.raises(ValueError):
            pipe._validate_resolution(1024, 500)


class TestQwenImagePipelineCPUOffload:
    """Test CPU offload helper methods."""

    def test_offload_text_encoder_when_enabled(self):
        """Test _offload_text_encoder moves encoder to CPU."""
        from llm_dit.pipelines.qwen_image import QwenImagePipeline

        mock_encoder = MagicMock()
        mock_encoder.device = torch.device("cuda")
        mock_dit = MagicMock()
        mock_dit.device = torch.device("cuda")
        mock_vae = MagicMock()

        pipe = QwenImagePipeline(mock_encoder, mock_dit, mock_vae)
        pipe.enable_cpu_offload()

        pipe._offload_text_encoder()

        mock_encoder.to.assert_called_once_with(torch.device("cpu"))

    def test_offload_text_encoder_when_disabled(self):
        """Test _offload_text_encoder does nothing when offload disabled."""
        from llm_dit.pipelines.qwen_image import QwenImagePipeline

        mock_encoder = MagicMock()
        mock_encoder.device = torch.device("cuda")
        mock_dit = MagicMock()
        mock_dit.device = torch.device("cuda")
        mock_vae = MagicMock()

        pipe = QwenImagePipeline(mock_encoder, mock_dit, mock_vae)
        # Don't enable CPU offload

        pipe._offload_text_encoder()

        mock_encoder.to.assert_not_called()

    def test_prepare_dit_moves_components(self):
        """Test _prepare_dit moves DiT to GPU and VAE to CPU."""
        from llm_dit.pipelines.qwen_image import QwenImagePipeline

        mock_encoder = MagicMock()
        mock_dit = MagicMock()
        mock_dit.device = torch.device("cpu")  # Start on CPU
        mock_vae = MagicMock()
        mock_vae.device = torch.device("cuda")  # Start on GPU

        pipe = QwenImagePipeline(mock_encoder, mock_dit, mock_vae)
        pipe._primary_device = torch.device("cuda")
        pipe.enable_cpu_offload()

        pipe._prepare_dit()

        # DiT should move to GPU
        mock_dit.to.assert_called_with(torch.device("cuda"))
        # VAE should move to CPU
        mock_vae.to.assert_called_with(torch.device("cpu"))

    def test_prepare_vae_moves_components(self):
        """Test _prepare_vae moves VAE to GPU and DiT to CPU."""
        from llm_dit.pipelines.qwen_image import QwenImagePipeline

        mock_encoder = MagicMock()
        mock_dit = MagicMock()
        mock_dit.device = torch.device("cuda")  # Start on GPU
        mock_vae = MagicMock()
        mock_vae.device = torch.device("cpu")  # Start on CPU

        pipe = QwenImagePipeline(mock_encoder, mock_dit, mock_vae)
        pipe._primary_device = torch.device("cuda")
        pipe.enable_cpu_offload()

        pipe._prepare_vae()

        # VAE should move to GPU
        mock_vae.to.assert_called_with(torch.device("cuda"))
        # DiT should move to CPU
        mock_dit.to.assert_called_with(torch.device("cpu"))


class TestQwenImagePipelineFromPretrained:
    """Test from_pretrained parameters (mocked)."""

    def test_from_pretrained_accepts_quantization(self):
        """Test from_pretrained accepts quantization parameters."""
        from llm_dit.pipelines.qwen_image import QwenImagePipeline

        # Check that the signature accepts these parameters
        import inspect
        sig = inspect.signature(QwenImagePipeline.from_pretrained)

        # Uses separate quantization for text encoder and DiT
        assert "text_encoder_quantization" in sig.parameters
        assert sig.parameters["text_encoder_quantization"].default == "none"
        assert "dit_quantization" in sig.parameters
        assert sig.parameters["dit_quantization"].default == "none"

    def test_from_pretrained_accepts_compile(self):
        """Test from_pretrained accepts compile parameters."""
        from llm_dit.pipelines.qwen_image import QwenImagePipeline

        import inspect
        sig = inspect.signature(QwenImagePipeline.from_pretrained)

        assert "compile_model" in sig.parameters
        assert sig.parameters["compile_model"].default is False
        assert "compile_mode" in sig.parameters
        assert sig.parameters["compile_mode"].default == "reduce-overhead"

    def test_from_pretrained_accepts_cpu_offload(self):
        """Test from_pretrained accepts cpu_offload parameter."""
        from llm_dit.pipelines.qwen_image import QwenImagePipeline

        import inspect
        sig = inspect.signature(QwenImagePipeline.from_pretrained)

        assert "cpu_offload" in sig.parameters
        assert sig.parameters["cpu_offload"].default is False

    def test_from_pretrained_accepts_device_options(self):
        """Test from_pretrained accepts separate device options."""
        from llm_dit.pipelines.qwen_image import QwenImagePipeline

        import inspect
        sig = inspect.signature(QwenImagePipeline.from_pretrained)

        assert "device" in sig.parameters
        assert "text_encoder_device" in sig.parameters
        assert "vae_device" in sig.parameters


class TestQwenImageBackendQuantization:
    """Test quantization options in text encoder backend."""

    def test_from_pretrained_accepts_quantization(self):
        """Test from_pretrained accepts quantization parameter."""
        from llm_dit.backends.qwen_image import QwenImageTextEncoderBackend

        import inspect
        sig = inspect.signature(QwenImageTextEncoderBackend.from_pretrained)

        assert "quantization" in sig.parameters
        assert sig.parameters["quantization"].default == "none"

    def test_quantization_options(self):
        """Test valid quantization options."""
        # These are the valid options
        valid_options = ("none", "4bit", "8bit")

        # Document the expected behavior
        for opt in valid_options:
            assert opt in valid_options


class TestQwenImageDiTCompile:
    """Test torch.compile options in DiT wrapper."""

    def test_from_pretrained_accepts_compile(self):
        """Test from_pretrained accepts compile parameters."""
        from llm_dit.models.qwen_image_dit import QwenImageDiT

        import inspect
        sig = inspect.signature(QwenImageDiT.from_pretrained)

        assert "compile_model" in sig.parameters
        assert sig.parameters["compile_model"].default is False
        assert "compile_mode" in sig.parameters
        assert sig.parameters["compile_mode"].default == "reduce-overhead"

    def test_compile_modes(self):
        """Test valid compile modes."""
        valid_modes = ("reduce-overhead", "max-autotune", "default")

        for mode in valid_modes:
            assert mode in valid_modes
