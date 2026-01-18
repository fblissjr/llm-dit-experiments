#!/usr/bin/env python3
"""
LTX-2 End-to-End Video Generation Tests

Last Updated: 2026-01-18

Tests the complete video generation pipeline using prompts from the official
LTX-2 repository to verify end-to-end functionality.

Test prompts sourced from:
- coderef/LTX-2/packages/ltx-trainer/scripts/inference.py
- coderef/LTX-2/packages/ltx-pipelines/README.md

Requirements:
- CUDA GPU with sufficient VRAM (~24GB for full pipeline)
- LTX-2 model weights at models/LTX-2/ (includes text_encoder)

Usage:
    # Run E2E generation tests
    uv run pytest tests/integration/test_ltx2_e2e_generation.py -v

    # Run with verbose output
    uv run pytest tests/integration/test_ltx2_e2e_generation.py -v -s
"""

import gc
import pytest
import torch
from pathlib import Path

# Skip all tests if CUDA not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


# Official LTX-2 test prompts from the repository
LTX2_TEST_PROMPTS = [
    "A cat playing with a ball",
    "A cat walking",
    "A beautiful sunset over the ocean",
]

# Smaller prompt for quick smoke tests
SMOKE_TEST_PROMPT = "A cat walking"


def cleanup_gpu():
    """Free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_gpu_memory_gb() -> float:
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0.0


def models_available() -> bool:
    """Check if all required models are available."""
    transformer_path = Path("models/LTX-2/transformer")
    encoder_path = Path("models/LTX-2/text_encoder")
    return transformer_path.exists() and encoder_path.exists()


class TestGenerationConfig:
    """Test GenerationConfig dataclass."""

    def test_config_latent_dims(self):
        """Test latent dimension calculation."""
        from llm_dit.pipelines.generate import GenerationConfig

        config = GenerationConfig(
            num_frames=33,
            height=512,
            width=768,
        )

        t_latent, h_latent, w_latent = config.latent_dims

        # LTX-2 compression: 8x temporal, 32x spatial
        assert t_latent == (33 - 1) // 8 + 1  # 5
        assert h_latent == 512 // 32  # 16
        assert w_latent == 768 // 32  # 24

    def test_config_num_tokens(self):
        """Test token count calculation."""
        from llm_dit.pipelines.generate import GenerationConfig

        config = GenerationConfig(
            num_frames=33,
            height=512,
            width=768,
        )

        # 5 * 16 * 24 = 1920 tokens
        assert config.num_tokens == 1920


class TestPositionIndices:
    """Test position index creation."""

    def test_create_position_indices_shape(self):
        """Test position indices have correct shape."""
        from llm_dit.pipelines.generate import create_position_indices

        batch_size = 2
        num_frames = 33
        height = 512
        width = 768

        positions = create_position_indices(
            batch_size, num_frames, height, width, torch.device("cpu")
        )

        # Should be [B, 3, T] where T = t_latent * h_latent * w_latent
        t_latent = (num_frames - 1) // 8 + 1  # 5
        h_latent = height // 32  # 16
        w_latent = width // 32  # 24
        num_tokens = t_latent * h_latent * w_latent  # 1920

        assert positions.shape == (batch_size, 3, num_tokens)

    def test_create_position_indices_values(self):
        """Test position indices have correct value ranges."""
        from llm_dit.pipelines.generate import create_position_indices

        num_frames = 17
        height = 256
        width = 384

        positions = create_position_indices(
            1, num_frames, height, width, torch.device("cpu")
        )

        t_latent = (num_frames - 1) // 8 + 1  # 3
        h_latent = height // 32  # 8
        w_latent = width // 32  # 12

        # Check value ranges
        t_indices = positions[0, 0, :]
        h_indices = positions[0, 1, :]
        w_indices = positions[0, 2, :]

        assert t_indices.min() >= 0 and t_indices.max() < t_latent
        assert h_indices.min() >= 0 and h_indices.max() < h_latent
        assert w_indices.min() >= 0 and w_indices.max() < w_latent


class TestVideoModality:
    """Test Modality creation helper."""

    def test_create_video_modality(self):
        """Test Modality dataclass creation."""
        from llm_dit.pipelines.generate import create_video_modality

        batch_size = 1
        num_tokens = 288
        latent_dim = 128
        context_dim = 4096
        context_len = 100

        latent = torch.randn(batch_size, num_tokens, latent_dim)
        timestep = torch.ones(batch_size, num_tokens) * 500
        positions = torch.zeros(batch_size, 3, num_tokens, dtype=torch.long)
        prompt_embeds = torch.randn(batch_size, context_len, context_dim)

        modality = create_video_modality(
            latent, timestep, positions, prompt_embeds
        )

        assert modality.enabled is True
        assert modality.latent.shape == (batch_size, num_tokens, latent_dim)
        assert modality.timesteps.shape == (batch_size, num_tokens)
        assert modality.positions.shape == (batch_size, 3, num_tokens)
        assert modality.context.shape == (batch_size, context_len, context_dim)


def sufficient_vram_bf16() -> bool:
    """Check if GPU has enough VRAM for full model in bf16 (~26GB needed)."""
    if not torch.cuda.is_available():
        return False
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return total_vram >= 32


def sufficient_vram_fp8() -> bool:
    """Check if GPU has enough VRAM for FP8 quantized model (~13GB needed)."""
    if not torch.cuda.is_available():
        return False
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return total_vram >= 16  # FP8 model fits on 24GB GPU


def quanto_available() -> bool:
    """Check if optimum-quanto is available for quantization."""
    try:
        from optimum.quanto import quantize
        return True
    except ImportError:
        return False


# Alias for backwards compatibility
def sufficient_vram() -> bool:
    return sufficient_vram_bf16()


@pytest.mark.skipif(
    not models_available(),
    reason="LTX-2 models not found at models/LTX-2/"
)
@pytest.mark.skipif(
    not sufficient_vram(),
    reason="LTX-2 13B model requires >24GB VRAM (implement CPU offloading for 24GB GPUs)"
)
class TestE2ELatentGeneration:
    """End-to-end latent generation tests (no VAE decode).

    Note: These tests require >24GB VRAM since the full 13B model is ~26GB in bf16.
    For 24GB GPUs, implement CPU offloading (see LTX-2 repo's group offloading).
    """

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and cleanup for each test."""
        cleanup_gpu()
        yield
        cleanup_gpu()

    def test_generate_latents_shape(self):
        """Test that latent generation produces correct shapes."""
        from llm_dit.models import load_ltx2_transformer
        from llm_dit.pipelines.generate import GenerationConfig, generate_video

        # Load transformer to CPU first (RTX 4090 can't fit fp32 weights during load)
        # then transfer to CUDA in bf16
        model = load_ltx2_transformer(
            "models/LTX-2/transformer",
            dtype=torch.bfloat16,
            device="cpu"
        )
        model = model.to("cuda")

        # Create mock embeddings (skip text encoder for this test)
        config = GenerationConfig(
            num_frames=17,  # Small: (17-1)/8+1 = 3 temporal latents
            height=256,
            width=384,
            num_inference_steps=2,  # Minimal steps for testing
            guidance_scale=1.0,  # No CFG for speed
            seed=42,
        )

        # Create dummy prompt embeddings [B, seq_len, context_dim]
        prompt_embeds = torch.randn(
            1, 100, 4096,
            dtype=torch.bfloat16,
            device="cuda"
        )

        # Generate (no VAE = returns latents)
        latents = generate_video(
            model=model,
            prompt_embeds=prompt_embeds,
            config=config,
            vae=None,  # No VAE decode
        )

        # Verify latent shape: [B, D, T_lat, H_lat, W_lat]
        t_lat, h_lat, w_lat = config.latent_dims
        assert latents.shape == (1, 128, t_lat, h_lat, w_lat)

        # Cleanup
        del model, latents
        cleanup_gpu()

    def test_generate_with_cfg(self):
        """Test generation with classifier-free guidance."""
        from llm_dit.models import load_ltx2_transformer
        from llm_dit.pipelines.generate import GenerationConfig, generate_video

        # Load to CPU first, then transfer to CUDA
        model = load_ltx2_transformer(
            "models/LTX-2/transformer",
            dtype=torch.bfloat16,
            device="cpu"
        )
        model = model.to("cuda")

        config = GenerationConfig(
            num_frames=17,
            height=256,
            width=384,
            num_inference_steps=2,
            guidance_scale=3.0,  # With CFG
            seed=42,
        )

        prompt_embeds = torch.randn(
            1, 100, 4096,
            dtype=torch.bfloat16,
            device="cuda"
        )

        latents = generate_video(
            model=model,
            prompt_embeds=prompt_embeds,
            config=config,
            vae=None,
        )

        t_lat, h_lat, w_lat = config.latent_dims
        assert latents.shape == (1, 128, t_lat, h_lat, w_lat)

        del model, latents
        cleanup_gpu()


@pytest.mark.skipif(
    not models_available(),
    reason="LTX-2 models not found"
)
@pytest.mark.skipif(
    not sufficient_vram(),
    reason="LTX-2 13B model requires >24GB VRAM (implement CPU offloading for 24GB GPUs)"
)
@pytest.mark.slow
class TestE2EFullPipeline:
    """Full end-to-end generation with text encoding and VAE decode.

    Note: These tests require >24GB VRAM. For 24GB GPUs like RTX 4090,
    implement CPU/group offloading as done in the official LTX-2 repo.
    """

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and cleanup for each test."""
        cleanup_gpu()
        yield
        cleanup_gpu()

    def test_smoke_generation_with_official_prompt(self):
        """Smoke test: Generate video with official LTX-2 prompt."""
        from llm_dit.models import load_ltx2_transformer
        from llm_dit.models.ltx2 import LTX2TextConnectors
        from llm_dit.encoders.gemma3 import Gemma3TextEncoder
        from llm_dit.pipelines.generate import GenerationConfig, generate_video

        print(f"\nPrompt: {SMOKE_TEST_PROMPT}")

        # Load text encoder
        print("Loading text encoder...")
        text_encoder = Gemma3TextEncoder(
            model_path="models/LTX-2/text_encoder",
            device="cuda",
            dtype=torch.bfloat16,
        )

        # Encode prompt
        print("Encoding prompt...")
        prompt_embeds, attention_mask = text_encoder.encode([SMOKE_TEST_PROMPT])
        print(f"Prompt embeddings shape: {prompt_embeds.shape}")

        # Unload text encoder to free memory
        del text_encoder
        cleanup_gpu()

        # Load transformer to CPU first, then transfer to CUDA (memory-efficient)
        print("Loading transformer...")
        model = load_ltx2_transformer(
            "models/LTX-2/transformer",
            dtype=torch.bfloat16,
            device="cpu"
        )
        model = model.to("cuda")

        connectors = LTX2TextConnectors.from_pretrained(
            "models/LTX-2/transformer",
            dtype=torch.bfloat16,
            device="cpu"
        )
        connectors = connectors.to("cuda")

        # Small config for smoke test
        config = GenerationConfig(
            num_frames=9,  # Minimal: 2 temporal latents
            height=256,
            width=384,
            num_inference_steps=5,  # Few steps
            guidance_scale=3.0,
            seed=42,
        )

        print(f"Generating {config.num_frames} frames at {config.width}x{config.height}...")

        # Generate latents
        latents = generate_video(
            model=model,
            prompt_embeds=prompt_embeds,
            config=config,
            vae=None,  # No VAE for speed
            connectors=connectors,
            attention_mask=attention_mask,
        )

        t_lat, h_lat, w_lat = config.latent_dims
        assert latents.shape == (1, 128, t_lat, h_lat, w_lat)
        print(f"Generated latents shape: {latents.shape}")

        # Verify latents are not all zeros or NaN
        assert not torch.isnan(latents).any(), "Latents contain NaN"
        assert not torch.isinf(latents).any(), "Latents contain Inf"
        assert latents.abs().mean() > 0.01, "Latents are nearly zero"

        print("Smoke test passed!")

        del model, connectors, latents
        cleanup_gpu()

    @pytest.mark.parametrize("prompt", LTX2_TEST_PROMPTS)
    def test_generate_with_official_prompts(self, prompt: str):
        """Test generation with each official LTX-2 prompt."""
        from llm_dit.models import load_ltx2_transformer
        from llm_dit.models.ltx2 import LTX2TextConnectors
        from llm_dit.encoders.gemma3 import Gemma3TextEncoder
        from llm_dit.pipelines.generate import GenerationConfig, generate_video

        print(f"\nTesting prompt: {prompt}")

        # Load encoder, encode, unload
        text_encoder = Gemma3TextEncoder(
            model_path="models/LTX-2/text_encoder",
            device="cuda",
            dtype=torch.bfloat16,
        )
        prompt_embeds, attention_mask = text_encoder.encode([prompt])
        del text_encoder
        cleanup_gpu()

        # Load model to CPU first, then transfer to CUDA
        model = load_ltx2_transformer(
            "models/LTX-2/transformer",
            dtype=torch.bfloat16,
            device="cpu"
        )
        model = model.to("cuda")

        connectors = LTX2TextConnectors.from_pretrained(
            "models/LTX-2/transformer",
            dtype=torch.bfloat16,
            device="cpu"
        )
        connectors = connectors.to("cuda")

        config = GenerationConfig(
            num_frames=9,
            height=256,
            width=384,
            num_inference_steps=3,
            guidance_scale=3.0,
            seed=42,
        )

        latents = generate_video(
            model=model,
            prompt_embeds=prompt_embeds,
            config=config,
            vae=None,
            connectors=connectors,
            attention_mask=attention_mask,
        )

        # Verify output
        assert not torch.isnan(latents).any()
        assert not torch.isinf(latents).any()

        del model, connectors, latents
        cleanup_gpu()


@pytest.mark.skipif(
    not models_available(),
    reason="LTX-2 models not found at models/LTX-2/"
)
@pytest.mark.skipif(
    not sufficient_vram_fp8(),
    reason="Need at least 16GB VRAM for FP8 model"
)
@pytest.mark.skipif(
    not quanto_available(),
    reason="optimum-quanto not installed (pip install optimum-quanto)"
)
class TestE2ELatentGenerationFP8:
    """End-to-end latent generation tests using FP8 quantization.

    These tests work on 24GB GPUs like RTX 4090 by using FP8 quantization
    to reduce the 13B model from ~26GB to ~13GB.
    """

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and cleanup for each test."""
        cleanup_gpu()
        yield
        cleanup_gpu()

    def test_quantized_model_loading(self):
        """Test that FP8 quantized model loads and fits in VRAM."""
        from llm_dit.models.ltx2 import load_ltx2_transformer_quantized

        print("\nLoading model with FP8 quantization...")
        model = load_ltx2_transformer_quantized(
            "models/LTX-2/transformer",
            precision="fp8-quanto",
            dtype=torch.bfloat16,
            video_only=True,
            verbose=True,
        )

        # Move to CUDA
        print("Moving quantized model to CUDA...")
        model = model.to("cuda")

        # Check memory usage
        memory_gb = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU memory used: {memory_gb:.2f} GB")

        # Should be around 13GB for FP8
        assert memory_gb < 20, f"Model using too much memory: {memory_gb:.2f} GB"

        # Verify model works
        num_params = model.get_num_params()
        print(f"Model parameters: {num_params / 1e9:.2f}B")

        del model
        cleanup_gpu()

    def test_generate_latents_fp8(self):
        """Test latent generation with FP8 quantized model."""
        from llm_dit.models.ltx2 import load_ltx2_transformer_quantized
        from llm_dit.pipelines.generate import GenerationConfig, generate_video

        # Load and quantize
        model = load_ltx2_transformer_quantized(
            "models/LTX-2/transformer",
            precision="fp8-quanto",
            dtype=torch.bfloat16,
            verbose=True,
        )
        model = model.to("cuda")

        # Create config for small generation
        config = GenerationConfig(
            num_frames=9,  # Minimal: 2 temporal latents
            height=256,
            width=384,
            num_inference_steps=2,  # Minimal steps
            guidance_scale=1.0,  # No CFG for speed
            seed=42,
        )

        # Create dummy prompt embeddings (3840-dim = raw Gemma3 output)
        # The model's caption_projection transforms 3840 → 4096
        prompt_embeds = torch.randn(
            1, 100, 3840,
            dtype=torch.bfloat16,
            device="cuda"
        )

        # Generate
        latents = generate_video(
            model=model,
            prompt_embeds=prompt_embeds,
            config=config,
            vae=None,
        )

        # Verify output
        t_lat, h_lat, w_lat = config.latent_dims
        assert latents.shape == (1, 128, t_lat, h_lat, w_lat)
        assert not torch.isnan(latents).any()
        assert not torch.isinf(latents).any()

        print(f"Generated latents: {latents.shape}")

        del model, latents
        cleanup_gpu()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
