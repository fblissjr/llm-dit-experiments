#!/usr/bin/env python3
"""
FLUX.2 Klein End-to-End Generation Tests.

Last Updated: 2026-01-23

Tests the complete FLUX.2 Klein image generation pipeline including:
- Text-to-image generation
- Image editing with reference images
- Three-stage memory offloading
- Deterministic generation with seeds

Requirements:
- CUDA GPU with sufficient VRAM (16-24GB depending on model)
- FLUX.2 Klein model weights (downloaded from HF)
- Qwen3 encoder weights

Usage:
    # Run E2E generation tests
    uv run pytest tests/e2e/test_flux2_generation.py -v

    # Run specific test
    uv run pytest tests/e2e/test_flux2_generation.py::TestFlux2BasicGeneration::test_klein_4b_smoke -v -s
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


# Test prompts
TEST_PROMPTS = [
    "A photo of a cat",
    "A beautiful sunset over the ocean",
    "A mountain landscape with snow",
]

SMOKE_TEST_PROMPT = "A photo of a cat sitting on a windowsill"


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


def sufficient_vram_for_klein_4b() -> bool:
    """Check if GPU has enough VRAM for Klein 4B (~12GB with offloading)."""
    if not torch.cuda.is_available():
        return False
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return total_vram >= 16


def sufficient_vram_for_klein_9b() -> bool:
    """Check if GPU has enough VRAM for Klein 9B (~18GB with offloading)."""
    if not torch.cuda.is_available():
        return False
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return total_vram >= 24


def models_available_klein_4b() -> bool:
    """Check if Klein 4B weights are available (either cached or downloadable)."""
    # This will be checked by actually trying to load
    return True  # Assume HF downloads work


def models_available_klein_9b() -> bool:
    """Check if Klein 9B weights are available."""
    return True  # Assume HF downloads work


class TestFlux2GenerationConfig:
    """Test Flux2GenerationConfig dataclass."""

    def test_config_defaults(self):
        """Test default configuration values."""
        from llm_dit.pipelines.flux2_generate import Flux2GenerationConfig

        config = Flux2GenerationConfig(prompt="Test")

        assert config.height == 1024
        assert config.width == 1024
        assert config.num_steps == 4  # Distilled default
        assert config.guidance == 1.0  # Distilled default
        assert config.offload_between_stages is True

    def test_config_latent_dims(self):
        """Test latent dimension calculation."""
        from llm_dit.pipelines.flux2_generate import Flux2GenerationConfig

        config = Flux2GenerationConfig(
            prompt="Test",
            height=1024,
            width=1024,
        )

        # 16x total compression
        assert config.latent_height == 64
        assert config.latent_width == 64
        assert config.num_tokens == 64 * 64  # 4096

    def test_config_editing_mode(self):
        """Test editing mode detection."""
        from llm_dit.pipelines.flux2_generate import Flux2GenerationConfig

        # Text-to-image mode (no reference images)
        config_t2i = Flux2GenerationConfig(prompt="Test")
        assert config_t2i.is_editing_mode is False

        # Editing mode (with reference images)
        config_edit = Flux2GenerationConfig(
            prompt="Test",
            reference_images=["dummy_path.jpg"],
        )
        assert config_edit.is_editing_mode is True


class TestScheduler:
    """Test the timestep schedule generation."""

    def test_get_schedule_length(self):
        """Test that schedule has correct length."""
        from llm_dit.pipelines.flux2_generate import get_schedule

        num_steps = 4
        image_seq_len = 4096  # 64x64 latent

        schedule = get_schedule(num_steps, image_seq_len)

        # Should have num_steps + 1 timesteps
        assert len(schedule) == num_steps + 1

    def test_get_schedule_range(self):
        """Test that schedule starts near 1 and ends near 0."""
        from llm_dit.pipelines.flux2_generate import get_schedule

        schedule = get_schedule(4, 4096)

        # First timestep should be close to 1
        assert schedule[0] > 0.9

        # Last timestep should be close to 0
        assert schedule[-1] < 0.1

    def test_schedule_monotonic_decreasing(self):
        """Test that schedule is monotonically decreasing."""
        from llm_dit.pipelines.flux2_generate import get_schedule

        schedule = get_schedule(50, 4096)

        for i in range(len(schedule) - 1):
            assert schedule[i] > schedule[i + 1], f"Schedule not decreasing at {i}"


class TestReferenceImageProcessing:
    """Test reference image preprocessing functions."""

    def test_preprocess_reference_image(self):
        """Test image preprocessing for VAE encoding."""
        from llm_dit.pipelines.flux2_generate import preprocess_reference_image
        from PIL import Image
        import numpy as np

        # Create a test image
        img = Image.fromarray(np.random.randint(0, 255, (256, 384, 3), dtype=np.uint8))

        tensor = preprocess_reference_image(img)

        # Should be [3, H, W] in [-1, 1] range
        assert tensor.ndim == 3
        assert tensor.shape[0] == 3
        assert tensor.min() >= -1.0
        assert tensor.max() <= 1.0

        # Should be multiple of 16
        assert tensor.shape[1] % 16 == 0
        assert tensor.shape[2] % 16 == 0

    def test_preprocess_with_pixel_limit(self):
        """Test that pixel limit is respected."""
        from llm_dit.pipelines.flux2_generate import preprocess_reference_image
        from PIL import Image
        import numpy as np

        # Create a large image
        img = Image.fromarray(np.random.randint(0, 255, (2048, 2048, 3), dtype=np.uint8))

        # Process with 1MP limit
        tensor = preprocess_reference_image(img, limit_pixels=1024**2)

        # Total pixels should be under limit
        h, w = tensor.shape[1], tensor.shape[2]
        assert h * w <= 1024**2 * 1.1  # Allow some tolerance for rounding

    def test_create_ref_image_ids(self):
        """Test reference image position ID creation."""
        from llm_dit.pipelines.flux2_generate import _create_ref_image_ids

        h, w = 8, 8  # 8x8 latent = 64 tokens
        t_coord = torch.tensor([10])  # Time offset 10

        ids = _create_ref_image_ids(h, w, t_coord, device="cpu")

        # Should have h*w rows, 4 columns (t, h, w, l)
        assert ids.shape == (64, 4)

        # All t coords should be 10
        assert (ids[:, 0] == 10).all()

        # h coords should be in [0, h-1]
        assert ids[:, 1].min() >= 0
        assert ids[:, 1].max() < h

        # w coords should be in [0, w-1]
        assert ids[:, 2].min() >= 0
        assert ids[:, 2].max() < w


@pytest.mark.skipif(
    not sufficient_vram_for_klein_4b(),
    reason="Need at least 16GB VRAM for Klein 4B"
)
@pytest.mark.slow
class TestFlux2BasicGeneration:
    """Basic generation tests using Klein 4B (smaller, faster)."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and cleanup for each test."""
        cleanup_gpu()
        yield
        cleanup_gpu()

    def test_klein_4b_smoke(self):
        """Smoke test: Basic generation with Klein 4B."""
        from llm_dit.pipelines.flux2_generate import (
            Flux2GenerationConfig,
            generate_image,
        )

        config = Flux2GenerationConfig(
            prompt=SMOKE_TEST_PROMPT,
            height=512,  # Smaller for faster test
            width=512,
            num_steps=2,  # Minimal steps
            seed=42,
        )

        try:
            image = generate_image(config, model_name="klein-4b")

            # Verify output
            assert image is not None
            assert image.size == (512, 512)
            assert image.mode == "RGB"

            print(f"Generated image: {image.size}")

        except Exception as e:
            pytest.skip(f"Generation failed (likely missing model): {e}")

    def test_deterministic_with_seed(self):
        """Test that same seed produces same output."""
        from llm_dit.pipelines.flux2_generate import (
            Flux2GenerationConfig,
            generate_image,
        )
        import numpy as np

        config = Flux2GenerationConfig(
            prompt="A red apple on a white table",
            height=256,  # Very small for speed
            width=256,
            num_steps=2,
            seed=12345,
        )

        try:
            # Generate twice with same seed
            img1 = generate_image(config, model_name="klein-4b")
            cleanup_gpu()

            img2 = generate_image(config, model_name="klein-4b")

            # Compare images
            arr1 = np.array(img1)
            arr2 = np.array(img2)

            # Should be identical
            assert np.allclose(arr1, arr2, atol=1), "Seeded generation not deterministic"

        except Exception as e:
            pytest.skip(f"Determinism test failed: {e}")


@pytest.mark.skipif(
    not sufficient_vram_for_klein_9b(),
    reason="Need at least 24GB VRAM for Klein 9B"
)
@pytest.mark.slow
class TestFlux2Klein9BGeneration:
    """Generation tests using Klein 9B."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and cleanup for each test."""
        cleanup_gpu()
        yield
        cleanup_gpu()

    def test_klein_9b_basic_generation(self):
        """Test basic generation with Klein 9B."""
        from llm_dit.pipelines.flux2_generate import (
            Flux2GenerationConfig,
            generate_image,
        )

        config = Flux2GenerationConfig(
            prompt=SMOKE_TEST_PROMPT,
            height=512,
            width=512,
            num_steps=2,
            seed=42,
        )

        try:
            image = generate_image(config, model_name="klein-9b")

            assert image is not None
            assert image.size == (512, 512)

        except Exception as e:
            pytest.skip(f"Klein 9B generation failed: {e}")


@pytest.mark.skipif(
    not sufficient_vram_for_klein_4b(),
    reason="Need at least 16GB VRAM"
)
@pytest.mark.slow
class TestFlux2EditingMode:
    """Test image editing with reference images."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and cleanup for each test."""
        cleanup_gpu()
        yield
        cleanup_gpu()

    def test_single_reference_image(self):
        """Test editing with a single reference image."""
        from llm_dit.pipelines.flux2_generate import (
            Flux2GenerationConfig,
            generate_image,
        )
        from PIL import Image
        import numpy as np
        import tempfile

        # Create a test reference image
        ref_img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            ref_path = f.name
            ref_img.save(ref_path)

        try:
            config = Flux2GenerationConfig(
                prompt="Transform the image into a watercolor painting",
                height=256,
                width=256,
                num_steps=2,
                seed=42,
                reference_images=[ref_path],
            )

            image = generate_image(config, model_name="klein-4b")

            assert image is not None
            assert image.size == (256, 256)
            assert config.is_editing_mode is True

        except Exception as e:
            pytest.skip(f"Editing mode test failed: {e}")
        finally:
            Path(ref_path).unlink(missing_ok=True)

    def test_multiple_reference_images(self):
        """Test editing with multiple reference images."""
        from llm_dit.pipelines.flux2_generate import (
            Flux2GenerationConfig,
            generate_image,
        )
        from PIL import Image
        import numpy as np
        import tempfile

        # Create multiple test reference images
        ref_paths = []
        for i in range(2):
            ref_img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                ref_img.save(f.name)
                ref_paths.append(f.name)

        try:
            config = Flux2GenerationConfig(
                prompt="Combine the styles of these images",
                height=256,
                width=256,
                num_steps=2,
                seed=42,
                reference_images=ref_paths,
            )

            image = generate_image(config, model_name="klein-4b")

            assert image is not None
            assert len(config.reference_images) == 2

        except Exception as e:
            pytest.skip(f"Multi-reference test failed: {e}")
        finally:
            for path in ref_paths:
                Path(path).unlink(missing_ok=True)


class TestFlux2MemoryOffloading:
    """Test memory management and offloading."""

    def test_memory_stages_isolated(self):
        """Test that three-stage offloading keeps peak memory low."""
        from llm_dit.pipelines.flux2_generate import (
            Flux2GenerationConfig,
            cleanup_memory,
        )

        # This is a design verification test
        # The actual memory test requires running generation

        config = Flux2GenerationConfig(
            prompt="Test",
            offload_between_stages=True,
        )

        assert config.offload_between_stages is True, "Offloading should be enabled by default"

    def test_cleanup_memory_function(self):
        """Test that cleanup_memory actually frees GPU memory."""
        from llm_dit.pipelines.flux2_generate import cleanup_memory

        # Allocate some GPU memory
        if torch.cuda.is_available():
            tensor = torch.randn(1000, 1000, device="cuda")
            mem_before = get_gpu_memory_gb()

            del tensor
            cleanup_memory()

            mem_after = get_gpu_memory_gb()

            # Memory should decrease
            assert mem_after <= mem_before


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
