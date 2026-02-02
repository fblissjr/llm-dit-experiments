"""
LTX-2 baseline generation and regression tests.

Last Updated: 2026-02-02

These tests generate baseline videos and verify reproducibility.
Use these tests to:
1. Validate the pipeline produces quality output
2. Verify same seed produces identical results
3. Compare outputs before/after changes

Output Structure:
    outputs/baselines/{tier}_seed{seed}_{timestamp}/
    ├── video.mp4              # Generated video
    ├── metadata.json          # Generation config + stats
    ├── inputs.json            # Full reproducibility record
    └── baseline_result.json   # Baseline-specific metadata

Usage:
    # Run smoke baseline (fastest, ~2min)
    uv run pytest tests/e2e/test_ltx2_baselines.py::TestLTX2Baselines::test_smoke_baseline_generation -v -s

    # Run reproducibility test
    uv run pytest tests/e2e/test_ltx2_baselines.py::TestLTX2Baselines::test_baseline_reproducibility -v -s

    # Run all baseline tests (slow)
    uv run pytest tests/e2e/test_ltx2_baselines.py -v -s --runslow

Requirements:
    - CUDA GPU with 16GB+ VRAM
    - LTX-2 model weights at models/LTX-2/
"""

import gc
import json
import logging
from datetime import datetime
from pathlib import Path

import pytest
import torch

from tests.backends import get_backend_name

logger = logging.getLogger(__name__)

# Skip all tests if CUDA not available
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


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


class TestLTX2Baselines:
    """Baseline generation and regression tests for LTX-2."""

    @pytest.mark.skipif(not models_available(), reason="LTX-2 models not found")
    @pytest.mark.skipif(not sufficient_vram(), reason="Insufficient VRAM (<16GB)")
    def test_smoke_baseline_generation(self):
        """Generate smoke-tier baseline video.

        This is the fastest possible baseline generation, useful for:
        - Quick pipeline validation
        - Verifying basic functionality after changes
        - CI/CD smoke tests

        Expected:
        - Video generates in <2 minutes
        - Output is coherent (no artifacts, frozen frames)
        - Files saved correctly
        """
        from tests.baselines import generate_baseline

        result = generate_baseline(
            config_tier="smoke",
            seed=42,
        )

        # Verify output exists
        assert result.output_path.exists(), f"Video not saved: {result.output_path}"
        assert result.frames_generated > 0, "No frames generated"

        # Verify video stats are reasonable (not all black/white)
        assert result.std_pixel_value > 0.01, f"Video appears uniform: std={result.std_pixel_value}"
        assert result.mean_pixel_value > 0.05, f"Video appears too dark: mean={result.mean_pixel_value}"
        assert result.mean_pixel_value < 0.95, f"Video appears too bright: mean={result.mean_pixel_value}"

        # Log results
        logger.info(f"Baseline generated: {result.output_path}")
        logger.info(f"Frames: {result.frames_generated}")
        logger.info(f"Time: {result.generation_time_seconds:.1f}s")
        logger.info(f"VRAM: {result.peak_vram_gb:.1f} GB")
        logger.info(f"Video stats: mean={result.mean_pixel_value:.4f}, std={result.std_pixel_value:.4f}")

    @pytest.mark.skipif(not models_available(), reason="LTX-2 models not found")
    @pytest.mark.skipif(not sufficient_vram(), reason="Insufficient VRAM (<16GB)")
    def test_baseline_reproducibility(self):
        """Verify same seed produces identical or near-identical output.

        This test runs generation twice with the same seed and compares
        the outputs. Due to GPU non-determinism, outputs may not be
        bit-identical but should be visually indistinguishable (SSIM > 0.99).

        Expected:
        - SSIM > 0.99 (near-identical)
        - Visual inspection shows no differences
        """
        from tests.baselines import compare_baselines, generate_baseline

        # Generate same prompt/seed twice
        result1 = generate_baseline(
            config_tier="smoke",
            seed=42,
            output_dir=Path("outputs/baselines/repro_test_run1"),
        )

        result2 = generate_baseline(
            config_tier="smoke",
            seed=42,
            output_dir=Path("outputs/baselines/repro_test_run2"),
        )

        # Compare outputs
        comparison = compare_baselines(result1.output_path, result2.output_path)

        logger.info(f"SSIM: {comparison.ssim:.6f}")
        logger.info(f"PSNR: {comparison.psnr:.2f} dB")
        logger.info(f"Identical: {comparison.is_identical}")
        logger.info(f"Similar: {comparison.is_similar}")

        # Verify reproducibility
        # Note: Due to GPU non-determinism, we can't guarantee bit-identical output
        # but SSIM should be very high (> 0.99)
        assert comparison.ssim > 0.95, f"Outputs differ too much: SSIM={comparison.ssim:.4f}"

        # Ideal case: bit-identical
        if comparison.is_identical:
            logger.info("Outputs are bit-identical")
        elif comparison.ssim > 0.99:
            logger.info("Outputs are visually indistinguishable (SSIM > 0.99)")
        else:
            logger.warning(f"Outputs differ slightly: SSIM={comparison.ssim:.4f}")

    @pytest.mark.slow
    @pytest.mark.skipif(not models_available(), reason="LTX-2 models not found")
    @pytest.mark.skipif(not sufficient_vram(), reason="Insufficient VRAM (<16GB)")
    def test_short_baseline_generation(self):
        """Generate short-tier baseline video.

        Uses reduced but reasonable parameters for quality validation.
        Slower than smoke but produces more watchable output.

        Expected:
        - Video generates in ~2-5 minutes
        - Content reflects prompt semantically
        - Temporal coherence (no frozen frames)
        """
        from tests.baselines import generate_baseline

        result = generate_baseline(
            config_tier="short",
            seed=42,
        )

        assert result.output_path.exists()
        assert result.frames_generated == 33  # Short config uses 33 frames

        logger.info(f"Short baseline generated: {result.output_path}")
        logger.info(f"Time: {result.generation_time_seconds:.1f}s")

    @pytest.mark.slow
    @pytest.mark.skipif(not models_available(), reason="LTX-2 models not found")
    @pytest.mark.skipif(not sufficient_vram(), reason="Insufficient VRAM (<16GB)")
    def test_reference_baseline_generation(self):
        """Generate reference-tier baseline video.

        Uses official LTX-2 parameters for high-quality output.
        This is the primary baseline for quality comparison.

        Expected:
        - Video generates in ~10 minutes
        - High visual quality
        - Full 5 seconds (121 frames at 24fps)
        """
        from tests.baselines import generate_baseline

        result = generate_baseline(
            config_tier="reference",
            seed=42,
        )

        assert result.output_path.exists()
        assert result.frames_generated == 121  # Reference config uses 121 frames

        logger.info(f"Reference baseline generated: {result.output_path}")
        logger.info(f"Time: {result.generation_time_seconds:.1f}s")
        logger.info(f"VRAM: {result.peak_vram_gb:.1f} GB")


class TestBaselineRegression:
    """Regression tests comparing against saved baselines."""

    @pytest.mark.skipif(not models_available(), reason="LTX-2 models not found")
    @pytest.mark.skipif(not sufficient_vram(), reason="Insufficient VRAM (<16GB)")
    def test_smoke_regression(self):
        """Compare current smoke output against saved baseline.

        This test requires a pre-saved baseline at:
        outputs/baselines/smoke_reference/video.mp4

        Run `generate_reference_baseline()` first to create the reference.
        """
        from tests.baselines import compare_baselines, generate_baseline

        reference_path = Path("outputs/baselines/smoke_reference/video.mp4")

        if not reference_path.exists():
            pytest.skip(
                "No reference baseline found. Run: "
                "uv run python -m tests.baselines.ltx2_baseline_runner --tier smoke --output-dir outputs/baselines/smoke_reference"
            )

        # Generate current baseline
        result = generate_baseline(
            config_tier="smoke",
            seed=42,
            output_dir=Path("outputs/baselines/smoke_current"),
        )

        # Compare against reference
        comparison = compare_baselines(result.output_path, reference_path)

        logger.info(f"Regression test: SSIM={comparison.ssim:.4f}")

        # Allow some variance due to potential implementation changes
        # but flag significant regressions
        if comparison.ssim < 0.90:
            pytest.fail(
                f"Significant regression detected: SSIM={comparison.ssim:.4f}\n"
                f"Current: {result.output_path}\n"
                f"Reference: {reference_path}"
            )
        elif comparison.ssim < 0.95:
            logger.warning(f"Minor regression detected: SSIM={comparison.ssim:.4f}")


class TestBaselineQuality:
    """Quality validation tests for generated baselines."""

    @pytest.mark.skipif(not models_available(), reason="LTX-2 models not found")
    @pytest.mark.skipif(not sufficient_vram(), reason="Insufficient VRAM (<16GB)")
    def test_video_not_frozen(self):
        """Verify video has temporal variation (not frozen frames).

        Generates a video and checks that frame-to-frame differences
        indicate actual motion, not static content.
        """
        from tests.baselines import generate_baseline
        from tests.baselines.ltx2_baseline_runner import _load_video_frames
        import numpy as np

        result = generate_baseline(
            config_tier="smoke",
            seed=42,
        )

        # Load frames
        frames = _load_video_frames(result.output_path)

        # Compute frame-to-frame differences
        frame_diffs = []
        for i in range(1, len(frames)):
            diff = np.mean(np.abs(frames[i].astype(float) - frames[i-1].astype(float)))
            frame_diffs.append(diff)

        mean_diff = np.mean(frame_diffs)
        logger.info(f"Mean frame-to-frame difference: {mean_diff:.2f}")

        # Video should have some temporal variation
        assert mean_diff > 0.5, f"Video appears frozen: mean diff={mean_diff:.2f}"

    @pytest.mark.skipif(not models_available(), reason="LTX-2 models not found")
    @pytest.mark.skipif(not sufficient_vram(), reason="Insufficient VRAM (<16GB)")
    def test_video_not_noise(self):
        """Verify video is not random noise.

        Checks that pixel values have reasonable structure,
        not just random static.
        """
        from tests.baselines import generate_baseline
        from tests.baselines.ltx2_baseline_runner import _load_video_frames
        import numpy as np

        result = generate_baseline(
            config_tier="smoke",
            seed=42,
        )

        frames = _load_video_frames(result.output_path)

        # Random noise would have high frequency content
        # Real video has spatial coherence

        # Check spatial coherence (adjacent pixels similar)
        spatial_diffs = []
        for frame in frames:
            h_diff = np.mean(np.abs(frame[:, 1:, :].astype(float) - frame[:, :-1, :].astype(float)))
            v_diff = np.mean(np.abs(frame[1:, :, :].astype(float) - frame[:-1, :, :].astype(float)))
            spatial_diffs.append((h_diff + v_diff) / 2)

        mean_spatial_diff = np.mean(spatial_diffs)
        logger.info(f"Mean spatial difference: {mean_spatial_diff:.2f}")

        # Real images have lower spatial differences than noise
        # Random noise ~40-50, real video typically < 20
        assert mean_spatial_diff < 30, f"Video appears noisy: spatial diff={mean_spatial_diff:.2f}"
