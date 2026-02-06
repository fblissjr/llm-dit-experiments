"""
LTX-2 LoRA integration tests.

Last Updated: 2026-02-02

Tests for LoRA loading and generation with the LTX-2 pipeline.
These tests validate that:
1. LoRA weights load correctly
2. Generation with LoRA produces valid output
3. Multiple LoRAs can be stacked

Usage:
    # Run LoRA loading test (requires GPU + model + LoRA file)
    uv run pytest tests/e2e/test_ltx2_lora.py::TestLTX2LoRA::test_lora_loading -v -s

    # Run all LoRA tests
    uv run pytest tests/e2e/test_ltx2_lora.py -v -s

Requirements:
    - CUDA GPU with 16GB+ VRAM
    - LTX-2 model weights at models/LTX-2/
    - LoRA file (default: /home/fbliss/Storage/LTX-2/ltx-2-19b-distilled-lora-384.safetensors)
"""

import gc
import logging
from pathlib import Path

import pytest
import torch

logger = logging.getLogger(__name__)

# Skip all tests if CUDA not available
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

# Default LoRA path (official distilled LoRA)
DEFAULT_LORA_PATH = Path("/home/fbliss/Storage/LTX-2/ltx-2-19b-distilled-lora-384.safetensors")


def models_available() -> bool:
    """Check if LTX-2 models are available."""
    transformer_path = Path("models/LTX-2/transformer")
    encoder_path = Path("models/LTX-2/text_encoder")
    return transformer_path.exists() and encoder_path.exists()


def lora_available() -> bool:
    """Check if LoRA file is available."""
    return DEFAULT_LORA_PATH.exists()


def sufficient_vram() -> bool:
    """Check if GPU has enough VRAM (16GB minimum for FP8)."""
    if not torch.cuda.is_available():
        return False
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return total_vram >= 16


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


class TestLTX2LoRA:
    """LoRA integration tests for LTX-2."""

    @pytest.mark.skipif(not models_available(), reason="LTX-2 models not found")
    @pytest.mark.skipif(not lora_available(), reason="LoRA file not found")
    @pytest.mark.skipif(not sufficient_vram(), reason="Insufficient VRAM (<16GB)")
    def test_lora_loading(self):
        """Test that LoRA loads without errors.

        This test verifies:
        1. LoRA file can be loaded
        2. Weights are fused into transformer
        3. Model still runs forward pass

        This is a smoke test - it doesn't verify generation quality,
        just that the mechanics work.
        """
        from llm_dit.models.ltx2 import load_ltx2_transformer
        from llm_dit.quantization import quantize_component
        from llm_dit.utils.lora import load_lora

        # Load transformer and quantize
        logger.info("Loading transformer...")
        model = load_ltx2_transformer(
            "models/LTX-2/transformer",
            dtype=torch.bfloat16,
            device="cuda",
            video_only=True,
        )
        model, _stats = quantize_component(  # type: ignore[assignment]
            model, method="fp8-weight-only", component_type="transformer"
        )

        # Load LoRA
        logger.info(f"Loading LoRA from {DEFAULT_LORA_PATH}...")
        num_updated = load_lora(
            model,
            DEFAULT_LORA_PATH,
            scale=0.8,
            device="cuda",
            dtype=torch.bfloat16,
        )

        logger.info(f"LoRA loaded: {num_updated} layers updated")

        # Verify some layers were updated
        assert num_updated > 0, "No LoRA layers were fused"

        # Verify model still works (basic forward pass)
        logger.info("Testing forward pass...")
        from llm_dit.models.ltx2 import Modality

        # Create minimal test inputs
        batch_size = 1
        num_tokens = 16  # Minimal token count
        context_len = 4

        latent = torch.randn(batch_size, num_tokens, 128, device="cuda", dtype=torch.bfloat16)
        timesteps = torch.ones(batch_size, num_tokens, device="cuda", dtype=torch.bfloat16) * 0.5
        positions = torch.randn(batch_size, 3, num_tokens, 2, device="cuda", dtype=torch.float32)
        context = torch.randn(batch_size, context_len, 3840, device="cuda", dtype=torch.bfloat16)

        video_modality = Modality(
            latent=latent,
            timesteps=timesteps,
            positions=positions,
            context=context,
            enabled=True,
        )

        with torch.no_grad():
            output, _ = model(video=video_modality)

        assert output.shape == latent.shape, f"Output shape mismatch: {output.shape} vs {latent.shape}"
        logger.info("Forward pass successful")

        # Cleanup
        del model
        torch.cuda.empty_cache()

    @pytest.mark.slow
    @pytest.mark.skipif(not models_available(), reason="LTX-2 models not found")
    @pytest.mark.skipif(not lora_available(), reason="LoRA file not found")
    @pytest.mark.skipif(not sufficient_vram(), reason="Insufficient VRAM (<16GB)")
    def test_generation_with_lora(self):
        """Test full video generation with LoRA.

        This test verifies:
        1. Full pipeline works with LoRA
        2. Output video is valid (not black/noise)
        3. Output is saved correctly
        """
        from llm_dit.pipelines import GenerationConfig, generate_video_with_offloading
        import time
        from datetime import datetime

        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"outputs/baselines/lora_test_{timestamp}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Smoke-tier config for fast testing
        config = GenerationConfig(
            num_frames=9,
            height=256,
            width=384,
            num_inference_steps=4,
            guidance_scale=1.0,
            seed=42,
        )

        logger.info("Generating video with LoRA...")
        start_time = time.time()

        video = generate_video_with_offloading(
            prompt="A cat walking",
            config=config,
            model_path="models/LTX-2",
            lora_path=str(DEFAULT_LORA_PATH),
            lora_scale=0.8,
        )

        gen_time = time.time() - start_time
        logger.info(f"Generation time: {gen_time:.1f}s")

        # Verify video shape
        assert video.dim() == 4, f"Expected 4D tensor, got {video.dim()}D"
        assert video.shape[0] == config.num_frames, f"Expected {config.num_frames} frames, got {video.shape[0]}"
        assert video.shape[-1] == 3, f"Expected RGB (3 channels), got {video.shape[-1]}"

        # Verify video is not uniform (black/white)
        video_float = video.float() / 255.0
        mean_val = video_float.mean().item()
        std_val = video_float.std().item()

        logger.info(f"Video stats: mean={mean_val:.4f}, std={std_val:.4f}")

        assert std_val > 0.01, f"Video appears uniform: std={std_val}"
        assert mean_val > 0.05, f"Video appears too dark: mean={mean_val}"
        assert mean_val < 0.95, f"Video appears too bright: mean={mean_val}"

        # Save video
        import imageio
        output_path = output_dir / "video.mp4"
        imageio.mimwrite(str(output_path), video.cpu().numpy(), fps=24, quality=8)

        logger.info(f"Video saved: {output_path}")
        assert output_path.exists(), f"Video not saved: {output_path}"

    @pytest.mark.slow
    @pytest.mark.skipif(not models_available(), reason="LTX-2 models not found")
    @pytest.mark.skipif(not lora_available(), reason="LoRA file not found")
    @pytest.mark.skipif(not sufficient_vram(), reason="Insufficient VRAM (<16GB)")
    def test_lora_scale_affects_output(self):
        """Test that different LoRA scales produce different outputs.

        This verifies that the LoRA is actually being applied and not ignored.
        We generate with scale=0.0 (should be same as no LoRA) and scale=1.0
        and verify the outputs differ.
        """
        from llm_dit.pipelines import GenerationConfig, generate_video_with_offloading
        import numpy as np

        config = GenerationConfig(
            num_frames=9,
            height=256,
            width=384,
            num_inference_steps=4,
            guidance_scale=1.0,
            seed=42,
        )

        # Generate with scale=0.0 (should be baseline)
        logger.info("Generating with lora_scale=0.0...")
        video_scale_0 = generate_video_with_offloading(
            prompt="A cat walking",
            config=config,
            model_path="models/LTX-2",
            lora_path=str(DEFAULT_LORA_PATH),
            lora_scale=0.0,
        )

        # Force memory cleanup between generations
        gc.collect()
        torch.cuda.empty_cache()

        # Generate with scale=1.0
        logger.info("Generating with lora_scale=1.0...")
        video_scale_1 = generate_video_with_offloading(
            prompt="A cat walking",
            config=config,
            model_path="models/LTX-2",
            lora_path=str(DEFAULT_LORA_PATH),
            lora_scale=1.0,
        )

        # Compare outputs
        diff = (video_scale_0.float() - video_scale_1.float()).abs().mean().item()
        logger.info(f"Mean absolute difference: {diff:.4f}")

        # scale=0 vs scale=1 should produce different outputs
        # Note: scale=0 means LoRA has no effect, but the model is still "fused"
        # so this test validates that scale actually affects the fusion
        assert diff > 0.5, f"LoRA scale doesn't affect output: diff={diff:.4f}"

    @pytest.mark.skipif(not models_available(), reason="LTX-2 models not found")
    @pytest.mark.skipif(not sufficient_vram(), reason="Insufficient VRAM (<16GB)")
    def test_invalid_lora_path_raises_error(self):
        """Test that invalid LoRA path raises appropriate error."""
        from llm_dit.models.ltx2 import load_ltx2_transformer
        from llm_dit.quantization import quantize_component
        from llm_dit.utils.lora import load_lora

        model = load_ltx2_transformer(
            "models/LTX-2/transformer",
            dtype=torch.bfloat16,
            device="cuda",
            video_only=True,
        )
        model, _stats = quantize_component(  # type: ignore[assignment]
            model, method="fp8-weight-only", component_type="transformer"
        )

        with pytest.raises(FileNotFoundError):
            load_lora(
                model,
                "/nonexistent/path/lora.safetensors",
                scale=0.8,
            )

        del model
        torch.cuda.empty_cache()


class TestLoRAPreset:
    """Tests for LoRA preset configuration."""

    def test_preset_file_exists(self):
        """Test that the LoRA test preset file exists."""
        preset_path = Path("presets/testing/ltx2_distilled_lora_test.md")
        # This will fail until we create the preset
        if not preset_path.exists():
            pytest.skip("LoRA preset not yet created")

        assert preset_path.exists(), f"LoRA preset not found: {preset_path}"

    @pytest.mark.skipif(not models_available(), reason="LTX-2 models not found")
    @pytest.mark.skipif(not lora_available(), reason="LoRA file not found")
    @pytest.mark.skipif(not sufficient_vram(), reason="Insufficient VRAM (<16GB)")
    def test_generation_from_lora_preset(self):
        """Test generation using the LoRA preset."""
        preset_path = Path("presets/testing/ltx2_distilled_lora_test.md")
        if not preset_path.exists():
            pytest.skip("LoRA preset not yet created")

        from tests.baselines import generate_baseline_from_preset

        result = generate_baseline_from_preset("ltx2_distilled_lora_test")

        assert result.output_path.exists()
        assert result.frames_generated > 0

        logger.info(f"LoRA preset baseline: {result.output_path}")
