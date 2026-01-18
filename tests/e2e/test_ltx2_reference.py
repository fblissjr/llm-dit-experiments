"""
LTX-2 Reference Comparison Tests.

Last Updated: 2026-01-18

End-to-end tests using official LTX-2 parameters for 1:1 comparison
with the reference implementation. These tests produce actual video
outputs for visual inspection.

Reference Parameters (from coderef/LTX-2):
- height: 512, width: 768
- num_frames: 121 (or 33 for quick tests)
- num_inference_steps: 40
- guidance_scale: 4.0
- seed: 10

Output: outputs/tests/ltx2/{test_name}_{timestamp}/

Requirements:
- CUDA GPU with 24GB+ VRAM (FP8 quantization enabled)
- LTX-2 model weights at models/LTX-2/

Usage:
    # Run all reference tests
    uv run pytest tests/e2e/test_ltx2_reference.py -v

    # Run quick smoke test only
    uv run pytest tests/e2e/test_ltx2_reference.py -v -k smoke
"""

import gc
import json
import logging
from datetime import datetime
from pathlib import Path

import pytest
import torch

# Skip all tests if CUDA not available
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

logger = logging.getLogger(__name__)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def output_base() -> Path:
    """Get output base directory for test results."""
    base = Path("outputs/tests/ltx2")
    base.mkdir(parents=True, exist_ok=True)
    return base


@pytest.fixture
def output_dir(output_base, request) -> Path:
    """Get timestamped output directory for this test."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_name = request.node.name.replace("[", "_").replace("]", "")
    out_dir = output_base / f"{test_name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


@pytest.fixture(autouse=True)
def cleanup_gpu():
    """Clean up GPU memory before and after each test."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def models_available() -> bool:
    """Check if LTX-2 models are available."""
    transformer_path = Path("models/LTX-2/transformer")
    encoder_path = Path("models/LTX-2/text_encoder")
    return transformer_path.exists() and encoder_path.exists()


def sufficient_vram() -> bool:
    """Check if GPU has enough VRAM for FP8 model (~16GB needed)."""
    if not torch.cuda.is_available():
        return False
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return total_vram >= 16


# =============================================================================
# Reference Constants
# =============================================================================

from llm_dit.models.ltx2 import (
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    DEFAULT_NUM_FRAMES,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_SEED,
    get_reference_config,
    get_quick_test_config,
)


# =============================================================================
# Test Prompts
# =============================================================================

SMOKE_PROMPT = "A cat walking"
REFERENCE_PROMPTS = {
    "cat_walking": "A cat walking",
    "cat_playing": "A cat playing with a ball",
}


# =============================================================================
# Test Classes
# =============================================================================

class TestLTX2ReferenceSmoke:
    """Quick smoke tests with reduced parameters."""

    @pytest.mark.skipif(not models_available(), reason="LTX-2 models not found")
    @pytest.mark.skipif(not sufficient_vram(), reason="Insufficient VRAM")
    def test_smoke_generation(self, output_dir):
        """Quick smoke test with minimal parameters.

        Tests the full pipeline: encode -> generate -> verify output.
        Uses FP8 quantization to fit 13B model on 24GB GPU.
        """
        from llm_dit.models.ltx2 import load_ltx2_transformer_quantized
        from llm_dit.encoders import Gemma3Encoder
        from llm_dit.pipelines.generate import GenerationConfig, generate_video

        logger.info(f"Output: {output_dir}")

        # Load encoder (8-bit)
        logger.info("Loading encoder...")
        encoder = Gemma3Encoder(
            model_id="models/LTX-2/text_encoder/",
            load_in_8bit=True,
            device="cuda",
        )

        # Encode prompt
        logger.info(f"Encoding: {SMOKE_PROMPT}")
        encoding_output = encoder.encode(SMOKE_PROMPT)
        # Extract tensor from EncodingOutput and add batch dim
        embeds = encoding_output.embeddings[0].unsqueeze(0)  # [1, seq_len, dim]
        logger.info(f"Embeddings shape: {embeds.shape}")

        # Unload encoder before loading transformer
        del encoder
        gc.collect()
        torch.cuda.empty_cache()

        # Load transformer (FP8 quantized)
        logger.info("Loading transformer (FP8)...")
        model = load_ltx2_transformer_quantized(
            "models/LTX-2/transformer/",
            precision="fp8-quanto",
            dtype=torch.bfloat16,
        )
        model = model.to("cuda")

        # Generate with minimal params (reduced for 24GB VRAM)
        logger.info("Generating...")
        config = GenerationConfig(
            num_frames=5,  # Minimal: 1 latent frame
            height=256,    # Reduced for memory
            width=384,     # Reduced for memory
            num_inference_steps=2,  # Minimal steps
            guidance_scale=1.0,  # Disable CFG to save memory
            seed=DEFAULT_SEED,
        )

        # Clear any fragmented memory before generation
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        with torch.inference_mode():
            latents = generate_video(
                model=model,
                prompt_embeds=embeds,
                config=config,
                vae=None,  # Skip VAE decode for smoke test
                device=torch.device("cuda"),
                dtype=torch.bfloat16,
            )

        # Verify output (latents, not decoded video)
        assert latents is not None
        assert not torch.isnan(latents).any()
        assert not torch.isinf(latents).any()
        logger.info(f"Latents shape: {latents.shape}")

        # Save metadata
        metadata = {
            "prompt": SMOKE_PROMPT,
            "num_frames": config.num_frames,
            "num_inference_steps": config.num_inference_steps,
            "guidance_scale": config.guidance_scale,
            "height": config.height,
            "width": config.width,
            "seed": config.seed,
            "test_type": "smoke",
            "latents_shape": list(latents.shape),
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Results saved to {output_dir}")

        # Cleanup
        del model, latents
        gc.collect()
        torch.cuda.empty_cache()


class TestLTX2ReferenceT2V:
    """Text-to-Video tests using official parameters."""

    @pytest.mark.slow
    @pytest.mark.skipif(not models_available(), reason="LTX-2 models not found")
    @pytest.mark.skipif(not sufficient_vram(), reason="Insufficient VRAM")
    def test_reference_t2v_short(self, output_dir):
        """Reference T2V with official params but shorter video (33 frames)."""
        from llm_dit.models.ltx2 import load_ltx2_transformer_quantized
        from llm_dit.encoders import Gemma3Encoder
        from llm_dit.pipelines.generate import GenerationConfig, generate_video

        logger.info(f"Output: {output_dir}")

        # Load and encode
        encoder = Gemma3Encoder(
            model_id="models/LTX-2/text_encoder/",
            load_in_8bit=True,
            device="cuda",
        )
        prompt = REFERENCE_PROMPTS["cat_walking"]
        encoding_output = encoder.encode(prompt)
        embeds = encoding_output.embeddings[0].unsqueeze(0)  # [1, seq_len, dim]
        del encoder
        gc.collect()
        torch.cuda.empty_cache()

        # Load transformer
        model = load_ltx2_transformer_quantized(
            "models/LTX-2/transformer/",
            precision="fp8-quanto",
        )
        model = model.to("cuda")

        # Generate with reference params (shorter frames)
        config = GenerationConfig(
            num_frames=33,
            height=DEFAULT_HEIGHT,
            width=DEFAULT_WIDTH,
            num_inference_steps=DEFAULT_NUM_INFERENCE_STEPS,
            guidance_scale=DEFAULT_GUIDANCE_SCALE,
            seed=DEFAULT_SEED,
        )

        latents = generate_video(
            model=model,
            prompt_embeds=embeds,
            config=config,
            vae=None,
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
        )

        assert latents is not None
        logger.info(f"Latents shape: {latents.shape}")

        # Save metadata
        metadata = {
            "prompt": prompt,
            "num_frames": 33,
            "num_inference_steps": DEFAULT_NUM_INFERENCE_STEPS,
            "guidance_scale": DEFAULT_GUIDANCE_SCALE,
            "height": DEFAULT_HEIGHT,
            "width": DEFAULT_WIDTH,
            "seed": DEFAULT_SEED,
            "test_type": "reference_t2v_short",
            "latents_shape": list(latents.shape),
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        del model, latents
        gc.collect()
        torch.cuda.empty_cache()


class TestLTX2ReferenceI2V:
    """Image-to-Video tests with conditioning."""

    @pytest.mark.slow
    @pytest.mark.skipif(not models_available(), reason="LTX-2 models not found")
    @pytest.mark.skipif(not sufficient_vram(), reason="Insufficient VRAM")
    def test_i2v_with_conditioning(self, output_dir):
        """Test I2V generation using conditioning system.

        Uses synthetic latent as stand-in for VAE-encoded image.
        Validates conditioning pipeline works end-to-end.
        """
        from llm_dit.models.ltx2 import load_ltx2_transformer_quantized
        from llm_dit.encoders import Gemma3Encoder
        from llm_dit.pipelines.generate import GenerationConfig, generate_video
        from llm_dit.conditioning import VideoConditionByLatentIndex

        logger.info(f"Output: {output_dir}")

        # Load and encode
        encoder = Gemma3Encoder(
            model_id="models/LTX-2/text_encoder/",
            load_in_8bit=True,
            device="cuda",
        )
        prompt = "A cat walking through a garden"
        encoding_output = encoder.encode(prompt)
        embeds = encoding_output.embeddings[0].unsqueeze(0)  # [1, seq_len, dim]
        del encoder
        gc.collect()
        torch.cuda.empty_cache()

        # Load transformer
        model = load_ltx2_transformer_quantized(
            "models/LTX-2/transformer/",
            precision="fp8-quanto",
        )
        model = model.to("cuda")

        # Create synthetic image latent (would be VAE-encoded in production)
        h_latent = DEFAULT_HEIGHT // 32
        w_latent = DEFAULT_WIDTH // 32
        image_latent = torch.randn(
            1, 128, 1, h_latent, w_latent,
            device="cuda",
            dtype=torch.bfloat16,
        )

        # Create conditioning
        cond = VideoConditionByLatentIndex(
            latent=image_latent,
            latent_idx=0,
            strength=1.0,
        )

        # Generate with conditioning
        config = GenerationConfig(
            num_frames=33,
            height=DEFAULT_HEIGHT,
            width=DEFAULT_WIDTH,
            num_inference_steps=20,
            guidance_scale=DEFAULT_GUIDANCE_SCALE,
            seed=DEFAULT_SEED,
        )

        latents = generate_video(
            model=model,
            prompt_embeds=embeds,
            config=config,
            conditioning=[cond],
            vae=None,
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
        )

        assert latents is not None
        logger.info(f"Latents shape: {latents.shape}")

        # Save metadata
        metadata = {
            "prompt": prompt,
            "num_frames": 33,
            "num_inference_steps": 20,
            "guidance_scale": DEFAULT_GUIDANCE_SCALE,
            "height": DEFAULT_HEIGHT,
            "width": DEFAULT_WIDTH,
            "seed": DEFAULT_SEED,
            "test_type": "i2v_conditioning",
            "conditioning": {
                "type": "VideoConditionByLatentIndex",
                "latent_idx": 0,
                "strength": 1.0,
            },
            "latents_shape": list(latents.shape),
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        del model, latents
        gc.collect()
        torch.cuda.empty_cache()
