"""
Portable LTX-2 baseline tests for 1:1 comparison.

Last Updated: 2026-01-19

These tests work with BOTH:
- llm-dit-experiments repo (uses llm_dit backend)
- Official LTX-2 repo (uses ltx2 backend)

Copy this file and tests/backends/ to the LTX-2 repo to run identical
tests with the official implementation for 1:1 baseline comparison.

Output Structure:
    outputs/tests/runs/{backend}_{test_name}_{timestamp}/
    ├── video.mp4              # Generated video
    ├── metadata.json          # Config + stats
    ├── inputs.json            # Full generation inputs (models, dtypes, shapes)
    └── generation.log         # Step-by-step logs

Usage:
    # Run in llm-dit-experiments repo (uses our implementation)
    uv run pytest tests/e2e/test_baseline_portable.py -v -s

    # Run in LTX-2 repo (uses official implementation)
    pytest tests/e2e/test_baseline_portable.py -v -s

    # Force specific backend
    LLM_DIT_TEST_BACKEND=ltx2 pytest tests/e2e/test_baseline_portable.py -v -s

Requirements:
    - CUDA GPU with 24GB+ VRAM
    - LTX-2 model weights at models/LTX-2/
"""

import gc
import json
import logging
from datetime import datetime
from pathlib import Path

import pytest
import torch

# Import from our backend abstraction
# This auto-detects which backend is available
try:
    from tests.backends import (
        REFERENCE_CONFIG,
        SHORT_CONFIG,
        # Standard configs (single source of truth)
        SMOKE_CONFIG,
        GenerationConfig,
        get_backend,
        get_backend_name,
    )
except ImportError:
    # Fallback for when running in LTX-2 repo with different structure
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from backends import (
        REFERENCE_CONFIG,
        SHORT_CONFIG,
        SMOKE_CONFIG,
        GenerationConfig,
        get_backend,
        get_backend_name,
    )

# Skip all tests if CUDA not available
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

logger = logging.getLogger(__name__)


# =============================================================================
# Fixtures
# =============================================================================


# conftest.py already provides backend_name
@pytest.fixture(scope="module")
def backend():
    """Get the video generation backend (auto-detected)."""
    # Keep this if you need the actual backend OBJECT,
    # but the path generation logic is handled by conftest.py
    backend_name = get_backend_name()
    if backend_name == "none":
        pytest.skip("No video generation backend available")
    logger.info(f"Using backend: {backend_name}")
    return get_backend()


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
    """Check if GPU has enough VRAM (16GB minimum for FP8)."""
    if not torch.cuda.is_available():
        return False
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return total_vram >= 16


# =============================================================================
# Test Prompts (Shared between backends)
# =============================================================================

SMOKE_PROMPT = "A cat walking"

REFERENCE_PROMPTS = {
    "cat_walking": "A cat walking through a sunny garden",
    "sunset": "A beautiful sunset over the ocean with gentle waves",
    "city_night": "A futuristic city at night with neon lights",
}


# =============================================================================
# Test Configurations
# =============================================================================
# Configs are imported from tests.backends (single source of truth)
# SMOKE_CONFIG, SHORT_CONFIG, REFERENCE_CONFIG
#
# | Config           | Frames | Resolution | Steps | CFG | VRAM  | Time    |
# |------------------|--------|------------|-------|-----|-------|---------|
# | SMOKE_CONFIG     | 9      | 256x384    | 8     | 3.0 | 14GB  | ~1min   |
# | SHORT_CONFIG     | 33     | 384x512    | 10    | 3.0 | 16GB  | ~2min   |
# | REFERENCE_CONFIG | 121    | 512x768    | 40    | 4.0 | 20GB  | ~10min  |


# =============================================================================
# Test Classes
# =============================================================================


class TestBaselineSmoke:
    """Quick smoke tests for basic pipeline validation."""

    @pytest.mark.skipif(not models_available(), reason="LTX-2 models not found")
    @pytest.mark.skipif(not sufficient_vram(), reason="Insufficient VRAM (<16GB)")
    def test_smoke_generation(self, backend, output_dir):
        """Minimal smoke test - verify pipeline works end-to-end.

        Uses smallest valid parameters for fastest iteration.
        Validates: model loads, generation runs, video saved.
        """
        config = SMOKE_CONFIG
        prompt = SMOKE_PROMPT

        logger.info(f"Backend: {backend.name}")
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Config: {config.num_frames} frames, {config.height}x{config.width}")

        # Generate video
        result = backend.generate_video(
            prompt=prompt,
            config=config,
            output_dir=output_dir,
            save_video=True,
        )

        # Verify output
        assert result.video is not None
        assert not torch.isnan(result.video.float()).any()
        assert not torch.isinf(result.video.float()).any()

        logger.info(f"Video shape: {result.video.shape}")
        logger.info(f"Total time: {result.stats.total_time:.1f}s")
        logger.info(f"Output: {output_dir}")

        # Verify files saved
        assert (output_dir / "video.mp4").exists()
        assert (output_dir / "metadata.json").exists()
        assert (output_dir / "inputs.json").exists()


class TestBaselineT2V:
    """Text-to-Video baseline tests."""

    @pytest.mark.skipif(not models_available(), reason="LTX-2 models not found")
    @pytest.mark.skipif(not sufficient_vram(), reason="Insufficient VRAM (<16GB)")
    def test_t2v_short(self, backend, output_dir):
        """Short T2V test with reasonable quality (~2min).

        Uses reduced parameters that still produce watchable output.
        Good for iterative debugging.
        """
        config = SHORT_CONFIG
        prompt = REFERENCE_PROMPTS["cat_walking"]

        logger.info(f"Backend: {backend.name}")
        logger.info(f"Prompt: {prompt}")

        result = backend.generate_video(
            prompt=prompt,
            config=config,
            output_dir=output_dir,
            save_video=True,
        )

        # Verify output
        assert result.video is not None
        assert result.video.shape[0] == config.num_frames
        assert result.video.shape[1] == config.height
        assert result.video.shape[2] == config.width

        logger.info(f"Stats: {json.dumps(result.stats.to_dict(), indent=2)}")

    @pytest.mark.slow
    @pytest.mark.skipif(not models_available(), reason="LTX-2 models not found")
    @pytest.mark.skipif(not sufficient_vram(), reason="Insufficient VRAM (<16GB)")
    def test_t2v_reference(self, backend, output_dir):
        """Full reference T2V with official LTX-2 parameters.

        Uses exact parameters from official LTX-2 demos:
        - 121 frames (5 seconds at 24fps)
        - 512x768 resolution
        - 40 inference steps
        - CFG scale 4.0
        - Seed 10

        This is the primary test for 1:1 baseline comparison.
        """
        config = REFERENCE_CONFIG
        prompt = REFERENCE_PROMPTS["cat_walking"]

        logger.info(f"Backend: {backend.name}")
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Config: {config.num_frames} frames, {config.height}x{config.width}")
        logger.info(f"Steps: {config.num_inference_steps}, CFG: {config.guidance_scale}")

        result = backend.generate_video(
            prompt=prompt,
            config=config,
            output_dir=output_dir,
            save_video=True,
        )

        # Verify output dimensions match config
        assert result.video.shape[0] == config.num_frames
        assert result.video.shape[1] == config.height
        assert result.video.shape[2] == config.width
        assert result.video.shape[3] == 3  # RGB

        # Log stats for comparison
        logger.info(f"Generation complete!")
        logger.info(f"Total time: {result.stats.total_time:.1f}s")
        if result.stats.text_encoder_time > 0:
            logger.info(f"  Text encoder: {result.stats.text_encoder_time:.1f}s")
            logger.info(f"  Transformer: {result.stats.transformer_time:.1f}s")
            logger.info(f"  VAE: {result.stats.vae_time:.1f}s")
        logger.info(f"Output: {output_dir}/video.mp4")


class TestBaselineI2V:
    """Image-to-Video baseline tests (requires conditioning support)."""

    @pytest.mark.skip(reason="I2V conditioning not yet implemented in portable backend")
    @pytest.mark.slow
    @pytest.mark.skipif(not models_available(), reason="LTX-2 models not found")
    @pytest.mark.skipif(not sufficient_vram(), reason="Insufficient VRAM (<16GB)")
    def test_i2v_reference(self, backend, output_dir):
        """Image-to-Video with first frame conditioning.

        TODO: Implement when conditioning is available in both backends.
        """
        pass


class TestBaselineComparison:
    """Tests for comparing output between backends."""

    @pytest.mark.skipif(not models_available(), reason="LTX-2 models not found")
    @pytest.mark.skipif(not sufficient_vram(), reason="Insufficient VRAM (<16GB)")
    def test_text_embedding_shape(self, backend, output_dir):
        """Verify text embedding shapes match between backends.

        Both backends should produce identical embedding shapes for the same prompt.
        This is a prerequisite for numerical equivalence.

        Expected shape: [1, seq_len, 3840] (49-layer Gemma3 projection)
        """
        prompt = SMOKE_PROMPT

        embeddings = backend.encode_text(prompt)

        logger.info(f"Backend: {backend.name}")
        logger.info(f"Embeddings shape: {embeddings.shape}")
        logger.info(f"Embeddings dtype: {embeddings.dtype}")

        # Save embedding info for comparison
        info = {
            "backend": backend.name,
            "prompt": prompt,
            "shape": list(embeddings.shape),
            "dtype": str(embeddings.dtype),
            "mean": float(embeddings.float().mean()),
            "std": float(embeddings.float().std()),
        }
        with open(output_dir / "embedding_info.json", "w") as f:
            json.dump(info, f, indent=2)

        # Verify reasonable shape
        assert embeddings.dim() == 3  # [B, seq_len, dim]
        assert embeddings.shape[0] == 1  # Batch size 1
        # Embedding dim should be 3840 (49 Gemma layers) or 4096 (after projection)
        assert embeddings.shape[2] in [3840, 4096]
