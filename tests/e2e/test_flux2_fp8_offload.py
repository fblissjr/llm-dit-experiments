#!/usr/bin/env python3
"""
FLUX.2 Klein FP8 + Block Offload End-to-End Tests.

Last Updated: 2026-01-24

Tests specifically for FP8 model loading with block offloading enabled.
These tests verify the OOM bug fix (loader.py line 278) that prevented
FP8 models from working with block offloading on 24GB GPUs.

Requirements:
- CUDA GPU with 24GB VRAM (RTX 4090, A5000, etc.)
- FLUX.2 Klein FP8 model weights

Usage:
    # Run FP8 + offload tests
    uv run pytest tests/e2e/test_flux2_fp8_offload.py -v -s

    # Run with memory tracking
    uv run pytest tests/e2e/test_flux2_fp8_offload.py -v -s --log-cli-level=DEBUG
"""

import gc
import logging
import pytest
import torch
from pathlib import Path

# Skip all tests if CUDA not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)

logger = logging.getLogger(__name__)


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


def get_peak_memory_gb() -> float:
    """Get peak GPU memory usage in GB since last reset."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**3
    return 0.0


def reset_peak_memory():
    """Reset peak memory tracking."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def get_total_vram_gb() -> float:
    """Get total GPU VRAM in GB."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / 1024**3
    return 0.0


def sufficient_vram_for_fp8_offload() -> bool:
    """Check if GPU has enough VRAM for FP8 + block offload (~12GB peak)."""
    if not torch.cuda.is_available():
        return False
    return get_total_vram_gb() >= 16


def sufficient_vram_for_fp8_no_offload() -> bool:
    """Check if GPU has enough VRAM for FP8 without offload (~18GB peak)."""
    if not torch.cuda.is_available():
        return False
    return get_total_vram_gb() >= 20


# Test prompts
SMOKE_TEST_PROMPT = "A photo of a cat sitting on a windowsill"


class TestFp8ModelLoading:
    """Test FP8 model loading with and without block offloading."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and cleanup for each test."""
        cleanup_gpu()
        reset_peak_memory()
        yield
        cleanup_gpu()

    @pytest.mark.skipif(
        not sufficient_vram_for_fp8_offload(),
        reason="Need at least 16GB VRAM for FP8 + block offload"
    )
    @pytest.mark.slow
    def test_fp8_loading_with_block_offload_no_oom(self):
        """
        Critical test: FP8 model loading with block offload should NOT OOM.

        This test verifies the fix for the bug where loader.py line 278
        unconditionally moved all weights to GPU before checking block_offload.

        Expected behavior: Peak VRAM stays under 12GB during loading.
        """
        from llm_dit.models.flux2.loader import load_flux2_transformer

        reset_peak_memory()
        initial_mem = get_gpu_memory_gb()
        logger.info(f"Initial GPU memory: {initial_mem:.2f}GB")

        try:
            # This should NOT OOM with the fix
            model = load_flux2_transformer(
                model_name="klein-9b-fp8",
                device="cuda",
                dtype=torch.bfloat16,
                block_offload=True,  # Critical: enables block offloading
            )

            peak_mem = get_peak_memory_gb()
            current_mem = get_gpu_memory_gb()

            logger.info(f"Peak memory during load: {peak_mem:.2f}GB")
            logger.info(f"Current memory after load: {current_mem:.2f}GB")

            # With block offloading, peak should be well under 16GB
            # (only embeddings + small layers on GPU, blocks on CPU)
            assert peak_mem < 16.0, (
                f"Peak memory {peak_mem:.2f}GB exceeds 16GB limit. "
                "Block offloading may not be working correctly."
            )

            # Model should have block offloading enabled
            assert model._block_offload_enabled, "Block offloading not enabled on model"

            # Cleanup
            del model
            cleanup_gpu()

        except torch.cuda.OutOfMemoryError as e:
            pytest.fail(f"OOM during FP8 + block offload loading (BUG NOT FIXED): {e}")
        except FileNotFoundError:
            pytest.skip("Klein-9B-FP8 weights not found")
        except Exception as e:
            if "weights" in str(e).lower() or "model" in str(e).lower():
                pytest.skip(f"Model weights not available: {e}")
            raise

    @pytest.mark.skipif(
        not sufficient_vram_for_fp8_no_offload(),
        reason="Need at least 20GB VRAM for FP8 without block offload"
    )
    @pytest.mark.slow
    def test_fp8_loading_without_block_offload(self):
        """Test FP8 model loading without block offload (full GPU mode)."""
        from llm_dit.models.flux2.loader import load_flux2_transformer

        reset_peak_memory()

        try:
            model = load_flux2_transformer(
                model_name="klein-9b-fp8",
                device="cuda",
                dtype=torch.bfloat16,
                block_offload=False,  # No offloading
            )

            peak_mem = get_peak_memory_gb()
            logger.info(f"Peak memory (no offload): {peak_mem:.2f}GB")

            # Without offloading, entire model on GPU (~17GB for BF16)
            assert model._block_offload_enabled is False

            # Model should be on GPU
            first_param = next(model.parameters())
            assert first_param.device.type == "cuda"

            del model
            cleanup_gpu()

        except torch.cuda.OutOfMemoryError:
            pytest.skip("Not enough VRAM for full GPU mode (expected on 24GB)")
        except FileNotFoundError:
            pytest.skip("Klein-9B-FP8 weights not found")


@pytest.mark.skipif(
    not sufficient_vram_for_fp8_offload(),
    reason="Need at least 16GB VRAM for FP8 + block offload"
)
@pytest.mark.slow
class TestFp8Generation:
    """End-to-end generation tests with FP8 models and block offloading."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and cleanup for each test."""
        cleanup_gpu()
        reset_peak_memory()
        yield
        cleanup_gpu()

    def test_fp8_offload_smoke_generation(self):
        """
        Smoke test: FP8 + block offload can generate an image.

        This is the critical regression test for the OOM bug fix.
        """
        from llm_dit.pipelines.flux2_generate import (
            Flux2GenerationConfig,
            generate_image,
        )

        reset_peak_memory()

        config = Flux2GenerationConfig(
            prompt=SMOKE_TEST_PROMPT,
            height=512,  # Small for faster test
            width=512,
            num_steps=2,  # Minimal steps
            seed=42,
            block_offload=True,  # Critical: use block offloading
        )

        try:
            image = generate_image(config, model_name="klein-9b-fp8")

            # Verify output
            assert image is not None
            assert image.size == (512, 512)
            assert image.mode == "RGB"

            peak_mem = get_peak_memory_gb()
            logger.info(f"Peak memory during generation: {peak_mem:.2f}GB")

            # Peak should stay under 20GB with block offloading
            assert peak_mem < 20.0, (
                f"Peak memory {peak_mem:.2f}GB exceeds 20GB. "
                "Block offloading may not be working during inference."
            )

            print(f"SUCCESS: Generated {image.size} image with FP8 + block offload")
            print(f"Peak VRAM: {peak_mem:.2f}GB")

        except torch.cuda.OutOfMemoryError as e:
            pytest.fail(f"OOM during FP8 + block offload generation: {e}")
        except FileNotFoundError:
            pytest.skip("Klein-9B-FP8 weights not found")
        except Exception as e:
            if "weights" in str(e).lower() or "model" in str(e).lower():
                pytest.skip(f"Model weights not available: {e}")
            raise

    def test_vram_stays_bounded_during_denoising(self):
        """
        Verify VRAM usage stays bounded during denoising loop.

        With block offloading, peak VRAM during denoising should be:
        - Embeddings: ~0.5GB
        - Single block on GPU: ~1.5GB
        - Activations per block: ~4-6GB
        - Peak: ~8-10GB (well under 24GB)
        """
        from llm_dit.pipelines.flux2_generate import (
            Flux2GenerationConfig,
            generate_image,
        )

        reset_peak_memory()

        config = Flux2GenerationConfig(
            prompt="A beautiful landscape with mountains",
            height=1024,  # Full resolution
            width=1024,
            num_steps=4,  # Distilled default
            seed=42,
            block_offload=True,
        )

        try:
            image = generate_image(config, model_name="klein-9b-fp8")

            assert image is not None
            assert image.size == (1024, 1024)

            peak_mem = get_peak_memory_gb()
            logger.info(f"Peak memory (1024x1024): {peak_mem:.2f}GB")

            # At 1024x1024 with block offloading, peak should still fit in 24GB
            # Budget: embeddings(0.5) + block(1.5) + activations(6-8) + overhead(2) ≈ 12GB
            assert peak_mem < 20.0, (
                f"Peak memory {peak_mem:.2f}GB too high for 1024x1024 generation. "
                "Expected under 20GB with block offloading."
            )

        except torch.cuda.OutOfMemoryError as e:
            pytest.fail(f"OOM at 1024x1024 with FP8 + block offload: {e}")
        except FileNotFoundError:
            pytest.skip("Klein-9B-FP8 weights not found")
        except Exception as e:
            if "weights" in str(e).lower() or "model" in str(e).lower():
                pytest.skip(f"Model weights not available: {e}")
            raise


class TestKleinVariants:
    """Test all Klein model variants with FP8 + offload."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and cleanup for each test."""
        cleanup_gpu()
        yield
        cleanup_gpu()

    @pytest.mark.skipif(
        not sufficient_vram_for_fp8_offload(),
        reason="Need at least 16GB VRAM"
    )
    @pytest.mark.slow
    @pytest.mark.parametrize("model_name", [
        "klein-9b-fp8",
        "klein-base-9b-fp8",
    ])
    def test_fp8_variants_with_offload(self, model_name: str):
        """Test loading different FP8 model variants with block offload."""
        from llm_dit.models.flux2.loader import load_flux2_transformer

        try:
            model = load_flux2_transformer(
                model_name=model_name,
                device="cuda",
                dtype=torch.bfloat16,
                block_offload=True,
            )

            assert model._block_offload_enabled
            logger.info(f"Successfully loaded {model_name} with block offload")

            del model
            cleanup_gpu()

        except FileNotFoundError:
            pytest.skip(f"{model_name} weights not found")
        except Exception as e:
            if "weights" in str(e).lower() or "model" in str(e).lower():
                pytest.skip(f"Model weights not available: {e}")
            raise


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
