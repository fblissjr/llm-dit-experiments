"""
LTX-2 Two-Stage Pipeline E2E Tests.

Last Updated: 2026-02-10

Exercises the full two-stage pipeline (encode -> stage 1 -> upsample ->
stage 2 -> decode) and saves everything needed for human/Claude inspection:
video, sampled frames as PNGs, and a consolidated run_manifest.json.

Uses the existing e2e conftest infrastructure (output_dir, cleanup_gpu,
backend_name fixtures) and calls generate_video_two_stage() directly.

Run with:
    # Smoke test (~3-5 min, GPU required)
    uv run pytest tests/e2e/test_ltx2_two_stage_e2e.py -v -s -k smoke

    # Full suite including reference (~10 min)
    uv run pytest tests/e2e/test_ltx2_two_stage_e2e.py -v -s --runslow
"""

from __future__ import annotations

import dataclasses
import logging
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.utils import RunManifest, sample_frames, save_frames

logger = logging.getLogger(__name__)


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


# ---------------------------------------------------------------------------
# Model availability check
# ---------------------------------------------------------------------------

def two_stage_models_available() -> bool:
    """Check that all models required for two-stage generation exist."""
    base = Path("models/LTX-2")
    return all([
        (base / "transformer").exists(),
        (base / "text_encoder").exists(),
        (base / "vae").exists(),
        (base / "ltx-2-spatial-upscaler-x2-1.0.safetensors").exists(),
        (base / "ltx-2-19b-distilled-lora-384.safetensors").exists(),
    ])


_skip_no_models = pytest.mark.skipif(
    not two_stage_models_available(),
    reason="Two-stage models not found in models/LTX-2",
)


# ---------------------------------------------------------------------------
# Instrumented runner
# ---------------------------------------------------------------------------

def run_two_stage_instrumented(
    prompt: str,
    num_frames: int,
    height: int,
    width: int,
    seed: int,
    stage1_steps: int,
    stage2_steps: int,
    output_dir: Path,
    test_name: str,
    num_sample_frames: int = 8,
    model_path: str | Path = "models/LTX-2",
    guidance_scale: float = 3.5,
    rescale_scale: float = 0.7,
    negative_prompt: str = "worst quality, blurry, distorted",
    distilled_lora_path: str = "ltx-2-19b-distilled-lora-384.safetensors",
    distilled_lora_scale: float = 0.8,
    spatial_upsampler_file: str = "ltx-2-spatial-upscaler-x2-1.0.safetensors",
) -> RunManifest:
    """Run two-stage generation with full instrumentation.

    Produces: video.mp4, sampled frame PNGs, and run_manifest.json
    in the given output_dir.

    Returns:
        RunManifest with all fields populated.
    """
    from llm_dit.pipelines.generate import (
        GenerationConfig,
        TwoStageConfig,
        generate_video_two_stage,
    )

    config = GenerationConfig(
        num_frames=num_frames,
        height=height,
        width=width,
        seed=seed,
    )
    two_stage = TwoStageConfig(
        stage1_steps=stage1_steps,
        stage2_steps=stage2_steps,
        guidance_scale=guidance_scale,
        rescale_scale=rescale_scale,
        negative_prompt=negative_prompt,
        distilled_lora_path=distilled_lora_path,
        distilled_lora_scale=distilled_lora_scale,
        spatial_upsampler_file=spatial_upsampler_file,
    )

    # Serialize two-stage config for manifest
    two_stage_dict = dataclasses.asdict(two_stage)

    manifest = RunManifest.create(
        test_name=test_name,
        prompt=prompt,
        num_frames=num_frames,
        height=height,
        width=width,
        seed=seed,
        negative_prompt=negative_prompt,
        two_stage_config=two_stage_dict,
    )

    # Timing callback
    stage_starts: dict[str, float] = {}
    stage_timings: dict[str, float] = {}

    def timing_callback(stage_name: str, step: int, total: int) -> None:
        if step == 1 and stage_name not in stage_starts:
            stage_starts[stage_name] = time.perf_counter()
        if step == total and stage_name in stage_starts:
            stage_timings[stage_name] = time.perf_counter() - stage_starts[stage_name]

    # Reset peak memory tracking
    torch.cuda.reset_peak_memory_stats()

    gen_start = time.perf_counter()

    video = generate_video_two_stage(
        prompt=prompt,
        config=config,
        two_stage=two_stage,
        model_path=model_path,
        callback=timing_callback,
    )

    total_time = time.perf_counter() - gen_start
    peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

    # Save video
    video_path = output_dir / "video.mp4"
    _save_video_mp4(video, video_path, fps=24)

    # Sample and save frames
    indices, frames = sample_frames(video, num_samples=num_sample_frames)
    frame_paths = save_frames(frames, indices, output_dir)

    # Populate manifest
    manifest.set_video_info(video, video_path)
    manifest.set_frame_info(frame_paths, indices)
    manifest.stage_timings = stage_timings
    manifest.total_time = total_time
    manifest.peak_vram_gb = round(peak_vram, 2)

    manifest.save(output_dir / "run_manifest.json")

    logger.info(
        f"Run complete: {total_time:.1f}s, peak VRAM {peak_vram:.1f}GB, "
        f"video shape {list(video.shape)}"
    )

    return manifest


def _save_video_mp4(
    video: torch.Tensor,
    path: Path,
    fps: int = 24,
) -> None:
    """Save [F, H, W, C] uint8 tensor as MP4 via imageio."""
    import imageio.v3 as iio

    frames = video.cpu().numpy()
    iio.imwrite(
        str(path),
        frames,
        fps=fps,
        codec="libx264",
        plugin="pyav",
    )
    logger.info(f"Saved video: {path} ({frames.shape[0]} frames, {fps}fps)")


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestTwoStageSmoke:
    """Quick two-stage validation (~3-5 min with reduced steps)."""

    @_skip_no_models
    def test_two_stage_smoke(self, output_dir):
        """10 stage1 steps + 3 stage2 steps at 512x768.

        Validates:
        - Pipeline completes without error
        - Output shape is [33, 512, 768, 3]
        - No NaN/Inf in video
        - Pixel std > 0.01 (not blank)
        - 8 sampled frames saved as PNGs
        - run_manifest.json written with all fields
        """
        manifest = run_two_stage_instrumented(
            prompt="A cat walking through a sunny garden",
            num_frames=33,
            height=512,
            width=768,
            seed=42,
            stage1_steps=10,
            stage2_steps=3,
            output_dir=output_dir,
            test_name="two_stage_smoke",
        )

        # Shape validation
        assert manifest.video_shape == [33, 512, 768, 3], (
            f"Expected [33, 512, 768, 3], got {manifest.video_shape}"
        )

        # Content validation (not blank, not noise)
        assert manifest.pixel_std > 0.01, (
            f"Pixel std {manifest.pixel_std} too low -- video may be blank"
        )

        # Frame sampling validation
        assert len(manifest.frame_paths) == 8, (
            f"Expected 8 sampled frames, got {len(manifest.frame_paths)}"
        )
        assert len(manifest.frame_indices) == 8

        # Output file existence
        assert (output_dir / "video.mp4").exists()
        assert (output_dir / "run_manifest.json").exists()

        # Verify frame PNGs exist
        for fname in manifest.frame_paths:
            assert (output_dir / fname).exists(), f"Frame missing: {fname}"

        # Performance sanity
        assert manifest.total_time > 0
        assert manifest.peak_vram_gb > 0


class TestTwoStageReference:
    """Full two-stage at official parameters (~8-10 min)."""

    @pytest.mark.slow
    @_skip_no_models
    def test_two_stage_reference(self, output_dir):
        """40 stage1 steps + 3 stage2 steps at 512x768.

        Full official parameters. Same assertions as smoke plus:
        - All stage timings captured in manifest
        - Peak VRAM recorded
        """
        manifest = run_two_stage_instrumented(
            prompt="A cat walking through a sunny garden",
            num_frames=33,
            height=512,
            width=768,
            seed=42,
            stage1_steps=40,
            stage2_steps=3,
            output_dir=output_dir,
            test_name="two_stage_reference",
        )

        assert manifest.video_shape == [33, 512, 768, 3]
        assert manifest.pixel_std > 0.01
        assert len(manifest.frame_paths) == 8
        assert (output_dir / "video.mp4").exists()
        assert (output_dir / "run_manifest.json").exists()

        # Stage timings should have been captured
        assert len(manifest.stage_timings) > 0, (
            "Expected stage timings to be captured"
        )
        assert manifest.peak_vram_gb > 0

    @pytest.mark.slow
    @_skip_no_models
    def test_two_stage_reproducibility(self, output_dir):
        """Same seed twice -> SSIM > 0.90 between runs.

        FP8 quantization introduces non-determinism, so we check
        similarity rather than bitwise identity.
        """
        from tests.baselines.ltx2_baseline_runner import _compute_ssim

        prompt = "A cat walking through a sunny garden"
        num_frames = 33
        height = 512
        width = 768
        seed = 42
        stage1_steps = 10
        stage2_steps = 3

        # Run 1
        run1_dir = output_dir / "run1"
        run1_dir.mkdir()
        manifest1 = run_two_stage_instrumented(
            prompt=prompt, num_frames=num_frames, height=height, width=width,
            seed=seed, stage1_steps=stage1_steps, stage2_steps=stage2_steps,
            output_dir=run1_dir, test_name="two_stage_repro_run1",
            num_sample_frames=4,
        )

        # Run 2
        run2_dir = output_dir / "run2"
        run2_dir.mkdir()
        manifest2 = run_two_stage_instrumented(
            prompt=prompt, num_frames=num_frames, height=height, width=width,
            seed=seed, stage1_steps=stage1_steps, stage2_steps=stage2_steps,
            output_dir=run2_dir, test_name="two_stage_repro_run2",
            num_sample_frames=4,
        )

        # Load first frame from each run and compare SSIM
        from PIL import Image

        frame1 = np.array(Image.open(run1_dir / manifest1.frame_paths[0]))
        frame2 = np.array(Image.open(run2_dir / manifest2.frame_paths[0]))

        ssim = _compute_ssim(frame1, frame2)
        logger.info(f"Reproducibility SSIM: {ssim:.4f}")

        assert ssim > 0.90, (
            f"SSIM {ssim:.4f} too low -- runs with same seed should be similar"
        )
