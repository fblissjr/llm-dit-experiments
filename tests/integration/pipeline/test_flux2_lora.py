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
    # Run LoRA generation test
    uv run pytest tests/integration/pipeline/test_flux2_lora.py::TestFlux2LoRA::test_generation_with_lora -v -s

    # Run all LoRA tests
    uv run pytest tests/integration/pipeline/test_flux2_lora.py -v -s

Requirements:
    - CUDA GPU with 16GB+ VRAM
    - FLUX.2 Klein model (configured in config.toml)
    - LoRA file(s) in loras/ directory

Note:
    All generated images are saved to outputs/tests/runs/ for manual inspection.
    Visual verification is essential - passing tests don't guarantee aesthetic quality.
"""

import gc
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import torch

logger = logging.getLogger(__name__)


def save_generation_metadata(
    output_dir: Path,
    image_path: Path,
    prompt: str,
    seed: int,
    height: int,
    width: int,
    num_steps: int,
    guidance: float,
    model_name: str,
    model_path: str,
    loras: list[str] | None = None,
    generation_time: float | None = None,
    **extra_params,
) -> Path:
    """Save generation metadata alongside the image.

    Creates a JSON file with all parameters used to generate the image,
    enabling reproducibility and regression testing.
    """
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "image_file": image_path.name,
        "prompt": prompt,
        "seed": seed,
        "height": height,
        "width": width,
        "num_inference_steps": num_steps,
        "guidance_scale": guidance,
        "model_name": model_name,
        "model_path": model_path,
        "loras": loras,
        "generation_time_seconds": generation_time,
        **extra_params,
    }

    metadata_path = output_dir / f"{image_path.stem}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata_path

# Skip all tests if CUDA not available
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def get_flux2_config():
    """Load FLUX.2 config from config.toml."""
    from llm_dit.config import load_config

    config = load_config("config.toml")
    return config.flux2


def get_test_lora_path() -> Path | None:
    """Get path to a test LoRA file if available.

    Searches the project's loras/ directory for FLUX.2 compatible LoRAs.
    Priority: loras/FLUX.2-klein/ > loras/
    Returns None if no LoRA found.
    """
    # Check FLUX.2-specific LoRA directory first
    flux2_loras_dir = Path("loras/FLUX.2-klein")
    if flux2_loras_dir.exists():
        # Prefer ghibli style for testing (known good)
        ghibli_lora = flux2_loras_dir / "ghibli_style_klein9b.safetensors"
        if ghibli_lora.exists():
            return ghibli_lora

        # Any Klein-compatible LoRA
        lora_files = list(flux2_loras_dir.glob("*.safetensors"))
        if lora_files:
            return lora_files[0]

    # Fallback to generic loras directory
    loras_dir = Path("loras")
    if loras_dir.exists():
        lora_files = list(loras_dir.glob("*klein*.safetensors"))
        if lora_files:
            return lora_files[0]
        lora_files = list(loras_dir.glob("*.safetensors"))
        if lora_files:
            return lora_files[0]

    return None


def lora_available() -> bool:
    """Check if any test LoRA file is available."""
    return get_test_lora_path() is not None


def flux2_models_available() -> bool:
    """Check if FLUX.2 models are configured and available."""
    try:
        config = get_flux2_config()
        model_path = Path(config.model_path)
        return model_path.exists()
    except Exception:
        return False


class TestFlux2LoRA:
    """LoRA integration tests for FLUX.2 Klein."""

    @pytest.mark.skipif(not flux2_models_available(), reason="FLUX.2 model not configured")
    @pytest.mark.skipif(not lora_available(), reason="No LoRA file found in loras/")
    def test_lora_loading(self):
        """Test that LoRA loads without errors.

        This test verifies:
        1. LoRA file can be loaded
        2. Weights are fused into transformer
        3. No crashes during fusion

        This is a smoke test - it doesn't verify generation quality.
        """
        from llm_dit.models.flux2.loader import load_flux2_transformer
        from llm_dit.utils.lora import load_lora

        config = get_flux2_config()
        lora_path = get_test_lora_path()
        assert lora_path is not None, "No LoRA file found for testing"

        logger.info(f"Loading FLUX.2 transformer from {config.model_path}...")
        transformer = load_flux2_transformer(
            config.default_model,
            device="cuda",
            dtype=torch.bfloat16,
            model_path=config.model_path,
        )

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
    @pytest.mark.skipif(not flux2_models_available(), reason="FLUX.2 model not configured")
    @pytest.mark.skipif(not lora_available(), reason="No LoRA file found in loras/")
    def test_generation_with_lora(self, output_dir):
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

        config = get_flux2_config()
        lora_path = get_test_lora_path()
        assert lora_path is not None

        # Generation parameters
        prompt = "A serene mountain landscape at sunset, dramatic lighting"
        seed = 42
        height = 512
        width = 512
        num_steps = config.default_steps or 4  # Default to 4 for distilled
        guidance = 1.0
        loras = [f"{lora_path}:0.8"]

        # Create generation config with LoRA
        gen_config = Flux2GenerationConfig(
            prompt=prompt,
            width=width,
            height=height,
            num_steps=num_steps,
            guidance=guidance,
            seed=seed,
            loras=loras,
        )

        logger.info("Generating image with LoRA...")
        start_time = time.time()

        image = generate_image(
            gen_config,
            model_name=config.default_model,
            model_path=config.model_path,
        )

        gen_time = time.time() - start_time
        logger.info(f"Generation time: {gen_time:.1f}s")

        # Verify image is valid
        assert image.size == (width, height), f"Unexpected size: {image.size}"

        # Check image is not uniform (all black/white)
        img_array = np.array(image)
        std_val = img_array.std()
        mean_val = img_array.mean()

        logger.info(f"Image stats: mean={mean_val:.2f}, std={std_val:.2f}")

        assert std_val > 10.0, f"Image appears uniform: std={std_val}"
        assert mean_val > 10.0, f"Image too dark: mean={mean_val}"
        assert mean_val < 245.0, f"Image too bright: mean={mean_val}"

        # Save image
        output_path = output_dir / "generation_with_lora.png"
        image.save(output_path)
        logger.info(f"Image saved: {output_path}")

        # Save metadata for reproducibility
        save_generation_metadata(
            output_dir=output_dir,
            image_path=output_path,
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_steps=num_steps,
            guidance=guidance,
            model_name=config.default_model,
            model_path=config.model_path,
            loras=loras,
            generation_time=gen_time,
            image_mean=float(mean_val),
            image_std=float(std_val),
        )

        assert output_path.exists()

    @pytest.mark.slow
    @pytest.mark.skipif(not flux2_models_available(), reason="FLUX.2 model not configured")
    @pytest.mark.skipif(not lora_available(), reason="No LoRA file found in loras/")
    def test_lora_scale_affects_output(self, output_dir):
        """Test that different LoRA scales produce different outputs.

        This verifies that the LoRA is actually being applied and not ignored.
        We generate with no LoRA and with LoRA and verify the outputs differ
        by a meaningful amount (MAD > 5.0 pixels).
        """
        from llm_dit.pipelines.flux2_generate import (
            Flux2GenerationConfig,
            generate_image,
        )

        config = get_flux2_config()
        lora_path = get_test_lora_path()
        assert lora_path is not None

        # Common generation parameters
        prompt = "A red apple on a wooden table"
        width = 512
        height = 512
        num_steps = config.default_steps or 4  # Default for distilled models
        guidance = 1.0
        seed = 42

        # Generate baseline (no LoRA)
        logger.info("Generating baseline (no LoRA)...")
        start_time = time.time()
        config_baseline = Flux2GenerationConfig(
            prompt=prompt, width=width, height=height,
            num_steps=num_steps, guidance=guidance, seed=seed, loras=None
        )
        img_baseline = generate_image(
            config_baseline,
            model_name=config.default_model,
            model_path=config.model_path,
        )
        baseline_time = time.time() - start_time

        # Save baseline
        baseline_path = output_dir / "scale_test_baseline.png"
        img_baseline.save(baseline_path)
        save_generation_metadata(
            output_dir=output_dir,
            image_path=baseline_path,
            prompt=prompt, seed=seed, height=height, width=width,
            num_steps=num_steps, guidance=guidance,
            model_name=config.default_model,
            model_path=config.model_path,
            loras=None,
            generation_time=baseline_time,
        )

        # Force cleanup between generations
        gc.collect()
        torch.cuda.empty_cache()

        # Generate with LoRA scale 0.8
        logger.info("Generating with LoRA scale=0.8...")
        loras = [f"{lora_path}:0.8"]
        start_time = time.time()
        config_with_lora = Flux2GenerationConfig(
            prompt=prompt, width=width, height=height,
            num_steps=num_steps, guidance=guidance, seed=seed, loras=loras
        )
        img_with_lora = generate_image(
            config_with_lora,
            model_name=config.default_model,
            model_path=config.model_path,
        )
        lora_time = time.time() - start_time

        # Save LoRA result
        lora_path_out = output_dir / "scale_test_lora_0.8.png"
        img_with_lora.save(lora_path_out)
        save_generation_metadata(
            output_dir=output_dir,
            image_path=lora_path_out,
            prompt=prompt, seed=seed, height=height, width=width,
            num_steps=num_steps, guidance=guidance,
            model_name=config.default_model,
            model_path=config.model_path,
            loras=loras,
            generation_time=lora_time,
        )

        # Compare outputs
        arr_baseline = np.array(img_baseline).astype(np.float32)
        arr_with_lora = np.array(img_with_lora).astype(np.float32)

        mad = np.abs(arr_baseline - arr_with_lora).mean()
        logger.info(f"Mean absolute difference: {mad:.2f} pixels")

        # LoRA should produce noticeably different output
        assert mad > 5.0, f"LoRA doesn't affect output enough: MAD={mad:.2f}"

        logger.info(f"Comparison images saved to {output_dir}")

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
    @pytest.mark.skipif(not flux2_models_available(), reason="FLUX.2 model not configured")
    @pytest.mark.skipif(not lora_available(), reason="No LoRA file found in loras/")
    def test_multiple_loras_applied(self, output_dir):
        """Test that multiple LoRAs can be stacked.

        Uses the same LoRA twice with different scales to verify
        that both are applied (cumulative effect).
        """
        from llm_dit.pipelines.flux2_generate import (
            Flux2GenerationConfig,
            generate_image,
        )

        config = get_flux2_config()
        lora_path = get_test_lora_path()
        assert lora_path is not None

        # Common generation parameters
        prompt = "A fantasy castle on a cliff"
        width = 512
        height = 512
        num_steps = config.default_steps or 4  # Default for distilled models
        guidance = 1.0
        seed = 999

        # Generate with single LoRA
        logger.info("Generating with single LoRA (0.5)...")
        loras_single = [f"{lora_path}:0.5"]
        start_time = time.time()
        config_single = Flux2GenerationConfig(
            prompt=prompt, width=width, height=height,
            num_steps=num_steps, guidance=guidance, seed=seed,
            loras=loras_single
        )
        img_single = generate_image(
            config_single,
            model_name=config.default_model,
            model_path=config.model_path,
        )
        single_time = time.time() - start_time

        # Save single LoRA result
        single_path = output_dir / "multiple_loras_single.png"
        img_single.save(single_path)
        save_generation_metadata(
            output_dir=output_dir,
            image_path=single_path,
            prompt=prompt, seed=seed, height=height, width=width,
            num_steps=num_steps, guidance=guidance,
            model_name=config.default_model,
            model_path=config.model_path,
            loras=loras_single,
            generation_time=single_time,
        )

        gc.collect()
        torch.cuda.empty_cache()

        # Generate with same LoRA twice (should be stronger effect)
        logger.info("Generating with same LoRA twice (0.3 + 0.3)...")
        loras_double = [f"{lora_path}:0.3", f"{lora_path}:0.3"]
        start_time = time.time()
        config_double = Flux2GenerationConfig(
            prompt=prompt, width=width, height=height,
            num_steps=num_steps, guidance=guidance, seed=seed,
            loras=loras_double
        )
        img_double = generate_image(
            config_double,
            model_name=config.default_model,
            model_path=config.model_path,
        )
        double_time = time.time() - start_time

        # Save double LoRA result
        double_path = output_dir / "multiple_loras_double.png"
        img_double.save(double_path)
        save_generation_metadata(
            output_dir=output_dir,
            image_path=double_path,
            prompt=prompt, seed=seed, height=height, width=width,
            num_steps=num_steps, guidance=guidance,
            model_name=config.default_model,
            model_path=config.model_path,
            loras=loras_double,
            generation_time=double_time,
        )

        # Compare - they should differ because 0.5 != 0.3+0.3 (fusion is multiplicative)
        arr_single = np.array(img_single).astype(np.float32)
        arr_double = np.array(img_double).astype(np.float32)

        mad = np.abs(arr_single - arr_double).mean()
        logger.info(f"Single vs double LoRA MAD: {mad:.2f}")

        # Due to fusion mechanics, the outputs should differ
        assert mad > 1.0, f"Multiple LoRAs not stacking: MAD={mad:.2f}"


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
