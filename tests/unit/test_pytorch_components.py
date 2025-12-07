"""
Unit tests for PyTorch-native components (Phase 1 migration).

Tests FlowMatchScheduler, attention backend, TiledVAEDecoder, and ContextRefiner
without requiring GPU or real models.
"""

import math

import pytest
import torch

pytestmark = pytest.mark.unit


class TestFlowMatchScheduler:
    """Test FlowMatchScheduler pure PyTorch implementation."""

    def test_import(self):
        from llm_dit.schedulers import FlowMatchScheduler
        assert FlowMatchScheduler is not None

    def test_init_default(self):
        from llm_dit.schedulers import FlowMatchScheduler

        scheduler = FlowMatchScheduler()
        assert scheduler.shift == 3.0
        assert scheduler.sigmas is None

    def test_init_custom_shift(self):
        from llm_dit.schedulers import FlowMatchScheduler

        scheduler = FlowMatchScheduler(shift=5.0)
        assert scheduler.shift == 5.0

    def test_set_timesteps(self):
        from llm_dit.schedulers import FlowMatchScheduler

        scheduler = FlowMatchScheduler(shift=3.0)
        scheduler.set_timesteps(num_inference_steps=9, device="cpu")

        assert scheduler.sigmas is not None
        assert len(scheduler.sigmas) == 10  # num_steps + 1
        assert scheduler.timesteps is not None
        assert len(scheduler.timesteps) == 9

    def test_sigma_shift_formula(self):
        """Verify shifted sigma formula: sigma' = shift * sigma / (1 + (shift-1) * sigma)"""
        from llm_dit.schedulers import FlowMatchScheduler

        shift = 3.0
        scheduler = FlowMatchScheduler(shift=shift)
        scheduler.set_timesteps(num_inference_steps=9, device="cpu")

        # First sigma should be close to 1.0 (shifted)
        # sigma_unshifted = 1.0, shifted = 3*1/(1+2*1) = 1.0
        assert scheduler.sigmas[0].item() == pytest.approx(1.0, abs=1e-5)

        # Last sigma should be 0.0
        assert scheduler.sigmas[-1].item() == pytest.approx(0.0, abs=1e-5)

        # Check intermediate values follow the formula
        for i, sigma in enumerate(scheduler.sigmas):
            sigma_unshifted = 1.0 - i / 9  # Linear from 1 to 0
            expected = shift * sigma_unshifted / (1 + (shift - 1) * sigma_unshifted)
            assert sigma.item() == pytest.approx(expected, abs=1e-5)

    def test_step_output_shape(self):
        from llm_dit.schedulers import FlowMatchScheduler

        scheduler = FlowMatchScheduler()
        scheduler.set_timesteps(num_inference_steps=9, device="cpu")

        # Create mock inputs
        model_output = torch.randn(1, 4, 64, 64)
        sample = torch.randn(1, 4, 64, 64)
        timestep_idx = 0

        result = scheduler.step(model_output, timestep_idx, sample)

        # Result is a SchedulerOutput dataclass
        assert hasattr(result, "prev_sample")
        assert result.prev_sample.shape == sample.shape

    def test_add_noise(self):
        from llm_dit.schedulers import FlowMatchScheduler

        scheduler = FlowMatchScheduler()
        scheduler.set_timesteps(num_inference_steps=9, device="cpu")

        original = torch.randn(1, 4, 64, 64)
        noise = torch.randn_like(original)
        timestep_idx = 4

        noisy = scheduler.add_noise(original, noise, timestep_idx)

        assert noisy.shape == original.shape
        # With sigma > 0, noisy should be different from original
        assert not torch.allclose(noisy, original)

    def test_scale_model_input(self):
        from llm_dit.schedulers import FlowMatchScheduler

        scheduler = FlowMatchScheduler()
        scheduler.set_timesteps(num_inference_steps=9, device="cpu")

        sample = torch.randn(1, 4, 64, 64)
        scaled = scheduler.scale_model_input(sample, 0)

        # FlowMatch doesn't scale input, should be unchanged
        assert torch.allclose(scaled, sample)


class TestAttentionBackend:
    """Test attention backend selector."""

    def test_import(self):
        from llm_dit.utils.attention import (
            attention_forward,
            get_attention_backend,
            set_attention_backend,
        )
        assert attention_forward is not None
        assert get_attention_backend is not None
        assert set_attention_backend is not None

    def test_get_backend_default(self):
        from llm_dit.utils.attention import get_attention_backend

        backend = get_attention_backend()
        # Should return something (sdpa is always available with PyTorch 2.0+)
        assert backend in ["flash_attn_3", "flash_attn_2", "sage", "xformers", "sdpa"]

    def test_set_backend_sdpa(self):
        from llm_dit.utils.attention import (
            get_attention_backend,
            set_attention_backend,
        )

        set_attention_backend("sdpa")
        assert get_attention_backend() == "sdpa"

    def test_invalid_backend_raises(self):
        from llm_dit.utils.attention import set_attention_backend
        import pytest

        # Invalid backend should raise ValueError
        with pytest.raises(ValueError):
            set_attention_backend("nonexistent_backend")

    def test_attention_forward_sdpa(self):
        from llm_dit.utils.attention import attention_forward, set_attention_backend

        set_attention_backend("sdpa")

        batch, heads, seq, head_dim = 2, 8, 16, 64
        q = torch.randn(batch, heads, seq, head_dim)
        k = torch.randn(batch, heads, seq, head_dim)
        v = torch.randn(batch, heads, seq, head_dim)

        output = attention_forward(q, k, v)

        assert output.shape == (batch, heads, seq, head_dim)

    def test_attention_forward_with_scale(self):
        from llm_dit.utils.attention import attention_forward, set_attention_backend

        set_attention_backend("sdpa")

        batch, heads, seq, head_dim = 1, 4, 8, 32
        q = torch.randn(batch, heads, seq, head_dim)
        k = torch.randn(batch, heads, seq, head_dim)
        v = torch.randn(batch, heads, seq, head_dim)

        scale = 1.0 / math.sqrt(head_dim)
        output = attention_forward(q, k, v, scale=scale)

        assert output.shape == (batch, heads, seq, head_dim)


class TestContextRefiner:
    """Test ContextRefiner pure PyTorch module."""

    def test_import(self):
        from llm_dit.models import ContextRefiner
        assert ContextRefiner is not None

    def test_init_default(self):
        from llm_dit.models import ContextRefiner

        refiner = ContextRefiner()
        assert refiner.dim == 3840
        assert refiner.n_layers == 2
        assert refiner.n_heads == 30
        assert len(refiner.layers) == 2

    def test_init_custom(self):
        from llm_dit.models import ContextRefiner

        refiner = ContextRefiner(
            dim=1024,
            n_layers=4,
            n_heads=16,
        )
        assert refiner.dim == 1024
        assert refiner.n_layers == 4
        assert refiner.n_heads == 16
        assert len(refiner.layers) == 4

    def test_forward_shape(self):
        from llm_dit.models import ContextRefiner

        # Use smaller dims for faster test
        refiner = ContextRefiner(
            dim=256,
            n_layers=1,
            n_heads=4,
        )

        batch, seq = 2, 16
        x = torch.randn(batch, seq, 256)

        output = refiner(x)

        assert output.shape == (batch, seq, 256)

    def test_forward_with_mask(self):
        from llm_dit.models import ContextRefiner

        refiner = ContextRefiner(
            dim=256,
            n_layers=1,
            n_heads=4,
        )

        batch, seq = 2, 16
        x = torch.randn(batch, seq, 256)
        mask = torch.ones(batch, 1, seq, seq)

        output = refiner(x, attention_mask=mask)

        assert output.shape == (batch, seq, 256)

    def test_gradient_checkpointing(self):
        from llm_dit.models import ContextRefiner

        refiner = ContextRefiner(
            dim=256,
            n_layers=2,
            n_heads=4,
        )

        assert not refiner._gradient_checkpointing

        refiner.enable_gradient_checkpointing()
        assert refiner._gradient_checkpointing

        refiner.enable_gradient_checkpointing(False)
        assert not refiner._gradient_checkpointing

    def test_num_params(self):
        from llm_dit.models import ContextRefiner

        refiner = ContextRefiner(
            dim=256,
            n_layers=2,
            n_heads=4,
        )

        total = refiner.get_num_params()
        trainable = refiner.get_num_params(trainable_only=True)

        assert total > 0
        assert trainable == total  # All params trainable by default

    def test_repr(self):
        from llm_dit.models import ContextRefiner

        refiner = ContextRefiner(dim=256, n_layers=2, n_heads=4)
        repr_str = repr(refiner)

        assert "ContextRefiner" in repr_str
        assert "dim=256" in repr_str
        assert "n_layers=2" in repr_str


class TestContextRefinerComponents:
    """Test individual components of ContextRefiner."""

    def test_rms_norm(self):
        from llm_dit.models.context_refiner import RMSNorm

        norm = RMSNorm(dim=64)
        x = torch.randn(2, 16, 64)
        output = norm(x)

        assert output.shape == x.shape
        # RMSNorm should normalize the input
        rms = torch.sqrt(output.pow(2).mean(-1))
        # Should be close to weight values (initialized to 1)
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)

    def test_rotary_embedding(self):
        from llm_dit.models.context_refiner import RotaryEmbedding

        rope = RotaryEmbedding(dim=64, max_seq_len=256)

        batch, seq, heads, head_dim = 2, 16, 4, 64
        q = torch.randn(batch, seq, heads, head_dim)
        k = torch.randn(batch, seq, heads, head_dim)

        q_rot, k_rot = rope(q, k)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_gated_feedforward(self):
        from llm_dit.models.context_refiner import GatedFeedForward

        ffn = GatedFeedForward(dim=64)
        x = torch.randn(2, 16, 64)
        output = ffn(x)

        assert output.shape == x.shape

    def test_context_refiner_block(self):
        from llm_dit.models.context_refiner import ContextRefinerBlock

        block = ContextRefinerBlock(
            dim=128,
            n_heads=4,
        )

        batch, seq = 2, 16
        x = torch.randn(batch, seq, 128)
        output = block(x)

        assert output.shape == x.shape


class TestTiledVAEDecoder:
    """Test TiledVAEDecoder for large image generation."""

    def test_import(self):
        from llm_dit.utils.tiled_vae import TiledVAEDecoder
        assert TiledVAEDecoder is not None

    def test_init(self):
        from llm_dit.utils.tiled_vae import TiledVAEDecoder
        from unittest.mock import MagicMock

        # Mock VAE with explicit scale_factor
        mock_vae = MagicMock()
        mock_vae.config.scaling_factor = 0.18215

        decoder = TiledVAEDecoder(
            vae=mock_vae,
            tile_size=512,
            tile_overlap=64,
            scale_factor=8,  # Explicitly set scale factor
        )

        assert decoder.tile_size == 512
        assert decoder.tile_overlap == 64
        assert decoder.scale_factor == 8
        assert decoder.latent_tile_size == 64  # 512 / 8

    def test_config_passthrough(self):
        from llm_dit.utils.tiled_vae import TiledVAEDecoder
        from unittest.mock import MagicMock

        mock_vae = MagicMock()
        mock_vae.config.scaling_factor = 0.18215

        decoder = TiledVAEDecoder(
            vae=mock_vae,
            tile_size=256,
            tile_overlap=32,
        )

        # Config passthrough should work
        assert decoder.config == mock_vae.config


class TestRuntimeConfig:
    """Test RuntimeConfig with new PyTorch fields."""

    def test_pytorch_fields_default(self):
        from llm_dit.cli import RuntimeConfig

        config = RuntimeConfig()

        assert config.attention_backend is None
        assert config.use_custom_scheduler is False
        assert config.tiled_vae is False
        assert config.tile_size == 512
        assert config.tile_overlap == 64

    def test_pytorch_fields_custom(self):
        from llm_dit.cli import RuntimeConfig

        config = RuntimeConfig(
            attention_backend="flash_attn_2",
            use_custom_scheduler=True,
            tiled_vae=True,
            tile_size=256,
            tile_overlap=32,
        )

        assert config.attention_backend == "flash_attn_2"
        assert config.use_custom_scheduler is True
        assert config.tiled_vae is True
        assert config.tile_size == 256
        assert config.tile_overlap == 32
