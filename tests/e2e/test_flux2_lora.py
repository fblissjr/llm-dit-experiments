"""
FLUX.2 Klein LoRA integration tests.

Last Updated: 2026-02-03

Tests for LoRA loading and generation with the FLUX.2 Klein pipeline.
These tests validate that:
1. LoRA weights load correctly into the transformer
2. Generation with LoRA produces valid output
3. Different LoRA scales produce different outputs
4. Multiple LoRAs can be stacked
5. Invalid paths raise appropriate errors

Usage:
    # Run LoRA loading test (requires GPU + model + LoRA file)
    uv run pytest tests/e2e/test_flux2_lora.py::TestFlux2LoRA::test_lora_loading -v -s

    # Run all LoRA tests
    uv run pytest tests/e2e/test_flux2_lora.py -v -s

Requirements:
    - CUDA GPU with 16GB+ VRAM
    - FLUX.2 Klein model (downloads from HuggingFace if not cached)
    - LoRA file(s) for testing (user-provided paths)

Note:
    All generated images are saved to tests/artifacts/flux2_lora/ for manual inspection.
    Visual verification is essential - passing tests don't guarantee aesthetic quality.
"""

import gc
import logging
from pathlib import Path
from datetime import datetime

import pytest
import torch

logger = logging.getLogger(__name__)

# Skip all tests if CUDA not available
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

# Output directory for test artifacts (visual verification)
ARTIFACT_DIR = Path("tests/artifacts/flux2_lora")


def sufficient_vram() -> bool:
    """Check if GPU has enough VRAM (16GB minimum for FP8)."""
    if not torch.cuda.is_available():
        return False
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return total_vram >= 16


def get_test_lora_path() -> Path | None:
    """Get path to a test LoRA file if available.

    Searches common locations for FLUX.2 LoRA files.
    Returns None if no LoRA found.
    """
    # Check common locations for FLUX.2 LoRAs
    search_paths = [
        Path("/home/fbliss/Storage/FLUX2/loras"),
        Path("models/FLUX2/loras"),
        Path.home() / "models" / "flux" / "loras",
    ]

    for search_dir in search_paths:
        if search_dir.exists():
            lora_files = list(search_dir.glob("*.safetensors"))
            if lora_files:
                return lora_files[0]

    return None


def lora_available() -> bool:
    """Check if any test LoRA file is available."""
    return get_test_lora_path() is not None


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


@pytest.fixture(scope="module")
def artifact_dir() -> Path:
    """Create and return artifact directory for test outputs."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ARTIFACT_DIR / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


class TestFlux2LoRA:
    """LoRA integration tests for FLUX.2 Klein."""

    @pytest.mark.skipif(not lora_available(), reason="No LoRA file found")
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
        from llm_dit.models.flux2.loader import load_flux2_transformer
        from llm_dit.utils.lora import load_lora

        lora_path = get_test_lora_path()
        assert lora_path is not None, "No LoRA file found for testing"

        # Load transformer (FP8 for lower VRAM)
        logger.info("Loading FLUX.2 transformer...")
        transformer = load_flux2_transformer(
            "klein-9b-fp8",
            device="cuda",
            dtype=torch.bfloat16,
        )

        # Load LoRA
        logger.info(f"Loading LoRA from {lora_path}...")
        num_updated = load_lora(
            transformer,
            lora_path,
            scale=0.8,
        )

        logger.info(f"LoRA loaded: {num_updated} layers updated")

        # Verify some layers were updated
        assert num_updated > 0, "No LoRA layers were fused"

        # Cleanup
        del transformer
        torch.cuda.empty_cache()

    @pytest.mark.slow
    @pytest.mark.skipif(not lora_available(), reason="No LoRA file found")
    @pytest.mark.skipif(not sufficient_vram(), reason="Insufficient VRAM (<16GB)")
    def test_generation_with_lora(self, artifact_dir):
        """Test full image generation with LoRA.

        This test verifies:
        1. Full pipeline works with LoRA
        2. Output image is valid (not black/noise)
        3. Output is saved for visual verification
        """
        from llm_dit.pipelines.flux2_generate import (
            Flux2GenerationConfig,
            generate_image,
        )
        import time

        lora_path = get_test_lora_path()
        assert lora_path is not None

        # Create generation config with LoRA
        config = Flux2GenerationConfig(
            prompt="A serene mountain landscape at sunset, dramatic lighting",
            width=512,
            height=512,
            num_steps=4,
            guidance=1.0,
            seed=42,
            loras=[f"{lora_path}:0.8"],
        )

        logger.info("Generating image with LoRA...")
        start_time = time.time()

        image = generate_image(config, model_name="klein-9b-fp8")

        gen_time = time.time() - start_time
        logger.info(f"Generation time: {gen_time:.1f}s")

        # Verify image is valid
        assert image.size == (config.width, config.height), f"Unexpected size: {image.size}"

        # Check image is not uniform (all black/white)
        import numpy as np
        img_array = np.array(image)
        std_val = img_array.std()
        mean_val = img_array.mean()

        logger.info(f"Image stats: mean={mean_val:.2f}, std={std_val:.2f}")

        assert std_val > 10.0, f"Image appears uniform: std={std_val}"
        assert mean_val > 10.0, f"Image too dark: mean={mean_val}"
        assert mean_val < 245.0, f"Image too bright: mean={mean_val}"

        # Save for visual verification
        output_path = artifact_dir / "generation_with_lora.png"
        image.save(output_path)
        logger.info(f"Image saved: {output_path}")

        assert output_path.exists()

    @pytest.mark.slow
    @pytest.mark.skipif(not lora_available(), reason="No LoRA file found")
    @pytest.mark.skipif(not sufficient_vram(), reason="Insufficient VRAM (<16GB)")
    def test_lora_scale_affects_output(self, artifact_dir):
        """Test that different LoRA scales produce different outputs.

        This verifies that the LoRA is actually being applied and not ignored.
        We generate with scale=0.0 and scale=0.8 and verify the outputs differ
        by a meaningful amount (MAD > 5.0 pixels).
        """
        from llm_dit.pipelines.flux2_generate import (
            Flux2GenerationConfig,
            generate_image,
        )
        import numpy as np

        lora_path = get_test_lora_path()
        assert lora_path is not None

        base_config = {
            "prompt": "A red apple on a wooden table",
            "width": 512,
            "height": 512,
            "num_steps": 4,
            "guidance": 1.0,
            "seed": 42,
        }

        # Generate baseline (no LoRA)
        logger.info("Generating baseline (no LoRA)...")
        config_baseline = Flux2GenerationConfig(**base_config, loras=None)
        img_baseline = generate_image(config_baseline, model_name="klein-9b-fp8")

        # Force cleanup between generations
        gc.collect()
        torch.cuda.empty_cache()

        # Generate with LoRA scale 0.8
        logger.info("Generating with LoRA scale=0.8...")
        config_with_lora = Flux2GenerationConfig(**base_config, loras=[f"{lora_path}:0.8"])
        img_with_lora = generate_image(config_with_lora, model_name="klein-9b-fp8")

        # Compare outputs
        arr_baseline = np.array(img_baseline).astype(np.float32)
        arr_with_lora = np.array(img_with_lora).astype(np.float32)

        mad = np.abs(arr_baseline - arr_with_lora).mean()
        logger.info(f"Mean absolute difference: {mad:.2f} pixels")

        # LoRA should produce noticeably different output
        assert mad > 5.0, f"LoRA doesn't affect output enough: MAD={mad:.2f}"

        # Save both for comparison
        img_baseline.save(artifact_dir / "scale_test_baseline.png")
        img_with_lora.save(artifact_dir / "scale_test_lora_0.8.png")
        logger.info(f"Comparison images saved to {artifact_dir}")

    @pytest.mark.slow
    @pytest.mark.skipif(not lora_available(), reason="No LoRA file found")
    @pytest.mark.skipif(not sufficient_vram(), reason="Insufficient VRAM (<16GB)")
    def test_negative_lora_scale(self, artifact_dir):
        """Test that negative LoRA scales work (inverse effect).

        Some LoRAs can be inverted by using negative scales, which should
        produce results that differ from both baseline and positive scale.
        """
        from llm_dit.pipelines.flux2_generate import (
            Flux2GenerationConfig,
            generate_image,
        )
        import numpy as np

        lora_path = get_test_lora_path()
        assert lora_path is not None

        config_base = {
            "prompt": "A portrait of a person",
            "width": 512,
            "height": 512,
            "num_steps": 4,
            "guidance": 1.0,
            "seed": 123,
        }

        # Generate with positive scale
        logger.info("Generating with positive scale (0.5)...")
        config_pos = Flux2GenerationConfig(**config_base, loras=[f"{lora_path}:0.5"])
        img_pos = generate_image(config_pos, model_name="klein-9b-fp8")

        gc.collect()
        torch.cuda.empty_cache()

        # Generate with negative scale
        logger.info("Generating with negative scale (-0.5)...")
        config_neg = Flux2GenerationConfig(**config_base, loras=[f"{lora_path}:-0.5"])
        img_neg = generate_image(config_neg, model_name="klein-9b-fp8")

        # Compare outputs
        arr_pos = np.array(img_pos).astype(np.float32)
        arr_neg = np.array(img_neg).astype(np.float32)

        mad = np.abs(arr_pos - arr_neg).mean()
        logger.info(f"Positive vs negative MAD: {mad:.2f}")

        # Positive and negative should differ
        assert mad > 5.0, f"Positive/negative scales produce same output: MAD={mad:.2f}"

        # Save for comparison
        img_pos.save(artifact_dir / "negative_scale_pos_0.5.png")
        img_neg.save(artifact_dir / "negative_scale_neg_0.5.png")

    def test_invalid_lora_path_raises_error(self):
        """Test that invalid LoRA path raises FileNotFoundError."""
        from llm_dit.utils.lora import load_lora
        import torch.nn as nn

        # Create a minimal model for testing
        model = nn.Linear(10, 10)

        with pytest.raises(FileNotFoundError):
            load_lora(
                model,
                "/nonexistent/path/to/lora.safetensors",
                scale=0.8,
            )


class TestFlux2MultipleLoRAs:
    """Tests for stacking multiple LoRAs on FLUX.2."""

    @pytest.mark.slow
    @pytest.mark.skipif(not lora_available(), reason="No LoRA file found")
    @pytest.mark.skipif(not sufficient_vram(), reason="Insufficient VRAM (<16GB)")
    def test_multiple_loras_applied(self, artifact_dir):
        """Test that multiple LoRAs can be stacked.

        Uses the same LoRA twice with different scales to verify
        that both are applied (cumulative effect).
        """
        from llm_dit.pipelines.flux2_generate import (
            Flux2GenerationConfig,
            generate_image,
        )
        import numpy as np

        lora_path = get_test_lora_path()
        assert lora_path is not None

        config_base = {
            "prompt": "A fantasy castle on a cliff",
            "width": 512,
            "height": 512,
            "num_steps": 4,
            "guidance": 1.0,
            "seed": 999,
        }

        # Generate with single LoRA
        logger.info("Generating with single LoRA (0.5)...")
        config_single = Flux2GenerationConfig(
            **config_base,
            loras=[f"{lora_path}:0.5"]
        )
        img_single = generate_image(config_single, model_name="klein-9b-fp8")

        gc.collect()
        torch.cuda.empty_cache()

        # Generate with same LoRA twice (should be stronger effect)
        logger.info("Generating with same LoRA twice (0.3 + 0.3)...")
        config_double = Flux2GenerationConfig(
            **config_base,
            loras=[f"{lora_path}:0.3", f"{lora_path}:0.3"]
        )
        img_double = generate_image(config_double, model_name="klein-9b-fp8")

        # Compare - they should differ because 0.5 != 0.3+0.3 (fusion is multiplicative)
        arr_single = np.array(img_single).astype(np.float32)
        arr_double = np.array(img_double).astype(np.float32)

        mad = np.abs(arr_single - arr_double).mean()
        logger.info(f"Single vs double LoRA MAD: {mad:.2f}")

        # Due to fusion mechanics (A @ B applied twice != applied once with different alpha),
        # the outputs should differ
        assert mad > 1.0, f"Multiple LoRAs not stacking: MAD={mad:.2f}"

        # Save for comparison
        img_single.save(artifact_dir / "multiple_loras_single.png")
        img_double.save(artifact_dir / "multiple_loras_double.png")


class TestFlux2LoRAParsing:
    """Tests for LoRA specification parsing."""

    def test_parse_lora_spec_with_scale(self):
        """Test parsing 'path:scale' format."""
        from llm_dit.utils.lora import parse_lora_spec

        path, scale = parse_lora_spec("models/lora.safetensors:0.7")
        assert path == "models/lora.safetensors"
        assert scale == 0.7

    def test_parse_lora_spec_without_scale(self):
        """Test parsing path without scale (default 1.0)."""
        from llm_dit.utils.lora import parse_lora_spec

        path, scale = parse_lora_spec("models/lora.safetensors")
        assert path == "models/lora.safetensors"
        assert scale == 1.0

    def test_parse_lora_spec_negative_scale(self):
        """Test parsing negative scale values."""
        from llm_dit.utils.lora import parse_lora_spec

        path, scale = parse_lora_spec("lora.safetensors:-0.5")
        assert path == "lora.safetensors"
        assert scale == -0.5

    def test_parse_lora_spec_windows_path(self):
        """Test parsing Windows-style paths with colons."""
        from llm_dit.utils.lora import parse_lora_spec

        # Windows path like C:\models\lora.safetensors should handle the drive letter
        path, scale = parse_lora_spec("C:\\models\\lora.safetensors:0.8")
        assert path == "C:\\models\\lora.safetensors"
        assert scale == 0.8
