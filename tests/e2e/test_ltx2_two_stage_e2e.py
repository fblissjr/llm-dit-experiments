"""
LTX-2 Two-Stage Pipeline E2E Tests.

Last Updated: 2026-02-10

Exercises the full two-stage pipeline through the API by posting to
/api/ltx2/generate/stream and consuming SSE events. Requires the server
to be running with config.toml (start it before running these tests):

    uv run web/server.py --config config.toml

Then saves everything needed for human/Claude inspection: video,
sampled frames as PNGs, and a consolidated run_manifest.json.

Run with:
    # Smoke test (~3-5 min, server must be running)
    uv run pytest tests/e2e/test_ltx2_two_stage_e2e.py -v -s -k smoke

    # Full suite including reference (~10 min)
    uv run pytest tests/e2e/test_ltx2_two_stage_e2e.py -v -s --runslow

Environment variables:
    TEST_SERVER_URL: Server URL (default: http://localhost:7860)
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path

import numpy as np
import orjson
import pytest
import requests
import torch

from tests.utils import RunManifest, sample_frames, save_frames

logger = logging.getLogger(__name__)

# Server URL -- same convention as z-image API tests
SERVER_URL = os.getenv("TEST_SERVER_URL", "http://localhost:7860")

pytestmark = [
    pytest.mark.e2e,
]


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def consume_sse_stream(
    url: str,
    payload: dict,
    timeout: int = 600,
) -> dict:
    """POST to an SSE endpoint and consume events until completion or error.

    Args:
        url: Full URL to POST to.
        payload: JSON request body.
        timeout: Read timeout in seconds (default 10 min for generation).

    Returns:
        dict with keys:
        - "events": list of all parsed SSE event dicts
        - "progress_events": list of progress-type events
        - "result": the completion event dict (type=complete), or None
        - "error": error message string, or None
    """
    events: list[dict] = []
    progress_events: list[dict] = []
    result: dict | None = None
    error: str | None = None

    resp = requests.post(
        url,
        json=payload,
        stream=True,
        timeout=(10, timeout),  # (connect_timeout, read_timeout)
    )
    resp.raise_for_status()

    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data_str = line[len("data: "):]
        try:
            event = orjson.loads(data_str)
        except Exception:
            logger.warning(f"Failed to parse SSE event: {data_str[:200]}")
            continue

        events.append(event)
        event_type = event.get("type", "")

        if event_type == "progress":
            progress_events.append(event)
            stage = event.get("stage", "")
            step = event.get("step", 0)
            total = event.get("total", 0)
            elapsed = event.get("elapsed", 0)
            logger.info(f"[SSE] {stage}: {step}/{total} ({elapsed:.1f}s)")

        elif event_type == "status":
            logger.info(f"[SSE] status: {event.get('message', '')}")

        elif event_type == "complete":
            result = event
            logger.info(
                f"[SSE] complete: seed={event.get('seed')}, "
                f"time={event.get('generation_time')}s, "
                f"video_url={event.get('video_url')}"
            )

        elif event_type == "error":
            error = event.get("message", "unknown error")
            logger.error(f"[SSE] error: {error}")

    return {
        "events": events,
        "progress_events": progress_events,
        "result": result,
        "error": error,
    }


def resolve_video_path(video_url: str) -> Path:
    """Convert a server video URL to an absolute file path.

    The server saves videos to outputs/videos/ and returns URLs like
    /outputs/videos/video_20260210_170000_abcd1234.mp4. Since the test
    runs on the same machine, we read the file directly from disk.
    """
    # /outputs/videos/video_xxx.mp4 -> outputs/videos/video_xxx.mp4
    relative = video_url.lstrip("/")
    return Path(relative)


def server_reachable() -> bool:
    """Check if the server is running."""
    try:
        resp = requests.get(f"{SERVER_URL}/health", timeout=3)
        return resp.status_code == 200
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        return False


_skip_no_server = pytest.mark.skipif(
    not server_reachable(),
    reason=f"Server not running at {SERVER_URL}",
)


# ---------------------------------------------------------------------------
# Instrumented runner (via API)
# ---------------------------------------------------------------------------

def run_two_stage_via_api(
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
    guidance_scale: float = 3.5,
    rescale_scale: float = 0.7,
    negative_prompt: str = "worst quality, blurry, distorted",
) -> RunManifest:
    """Run two-stage generation via the API with full instrumentation.

    Posts to /api/ltx2/generate/stream, consumes SSE, then reads the
    video from disk for frame sampling and manifest creation.

    Returns:
        RunManifest with all fields populated.
    """
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_frames": num_frames,
        "height": height,
        "width": width,
        "seed": seed,
        "use_two_stage": True,
        "stage1_steps": stage1_steps,
        "stage2_steps": stage2_steps,
        "guidance_scale": guidance_scale,
        "rescale_scale": rescale_scale,
    }

    manifest = RunManifest.create(
        test_name=test_name,
        prompt=prompt,
        num_frames=num_frames,
        height=height,
        width=width,
        seed=seed,
        negative_prompt=negative_prompt,
        two_stage_config={
            "stage1_steps": stage1_steps,
            "stage2_steps": stage2_steps,
            "guidance_scale": guidance_scale,
            "rescale_scale": rescale_scale,
            "use_two_stage": True,
        },
    )

    gen_start = time.perf_counter()

    # Hit the API
    url = f"{SERVER_URL}/api/ltx2/generate/stream"
    logger.info(f"POST {url}")
    logger.info(f"Payload: {orjson.dumps(payload).decode()}")

    sse = consume_sse_stream(url, payload)

    total_time = time.perf_counter() - gen_start

    if sse["error"]:
        raise RuntimeError(f"Generation failed via API: {sse['error']}")

    result = sse["result"]
    if result is None:
        raise RuntimeError("No completion event received from SSE stream")

    # Extract stage timings from progress events
    stage_timings: dict[str, float] = {}
    if result.get("generation_time"):
        stage_timings["total_server"] = result["generation_time"]

    # Read the video from disk
    video_url = result["video_url"]
    video_disk_path = resolve_video_path(video_url)
    if not video_disk_path.exists():
        raise FileNotFoundError(
            f"Video not found at {video_disk_path} "
            f"(from URL: {video_url})"
        )

    # Load video frames for inspection
    video_tensor = _load_video_as_tensor(video_disk_path)

    # Copy video to test output dir
    local_video_path = output_dir / "video.mp4"
    shutil.copy2(video_disk_path, local_video_path)

    # Sample and save frames
    indices, frames = sample_frames(video_tensor, num_samples=num_sample_frames)
    frame_paths = save_frames(frames, indices, output_dir)

    # Populate manifest
    manifest.set_video_info(video_tensor, local_video_path)
    manifest.set_frame_info(frame_paths, indices)
    manifest.stage_timings = stage_timings
    manifest.total_time = total_time

    # Add API-specific metadata
    manifest.environment["server_url"] = SERVER_URL
    manifest.environment["video_url"] = video_url
    manifest.environment["seed_returned"] = result.get("seed")
    manifest.environment["two_stage_returned"] = result.get("two_stage")
    manifest.environment["sse_event_count"] = len(sse["events"])
    manifest.environment["sse_progress_count"] = len(sse["progress_events"])

    manifest.save(output_dir / "run_manifest.json")

    logger.info(
        f"Run complete: {total_time:.1f}s, "
        f"video shape {list(video_tensor.shape)}, "
        f"server reported {result.get('generation_time')}s"
    )

    return manifest


def _load_video_as_tensor(video_path: Path) -> torch.Tensor:
    """Load video from disk as [F, H, W, C] uint8 tensor."""
    import imageio.v3 as iio

    frames = iio.imread(str(video_path), plugin="pyav")
    return torch.from_numpy(np.array(frames))


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestTwoStageSmoke:
    """Quick two-stage validation via API (~3-5 min with reduced steps)."""

    @_skip_no_server
    def test_two_stage_smoke(self, output_dir):
        """Two-stage generation via /api/ltx2/generate/stream.

        Uses reduced stage1 steps for speed. Validates:
        - API returns completion event with video_url
        - Video exists on disk and is loadable
        - Output shape is [33, 512, 768, 3]
        - Pixel std > 0.01 (not blank)
        - 8 sampled frames saved as PNGs
        - run_manifest.json written with all fields
        - SSE stream included progress events
        """
        manifest = run_two_stage_via_api(
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

        # Content validation (not blank)
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

        # API-specific validations
        assert manifest.environment.get("sse_event_count", 0) > 0, (
            "Expected SSE events from the stream"
        )
        assert manifest.environment.get("two_stage_returned") is True, (
            "Server should confirm two_stage=True in completion event"
        )
        assert manifest.total_time > 0


class TestTwoStageReference:
    """Full two-stage at official parameters via API (~8-10 min)."""

    @pytest.mark.slow
    @_skip_no_server
    def test_two_stage_reference(self, output_dir):
        """40 stage1 steps + 3 stage2 steps at 512x768 via API.

        Full official parameters. Same assertions as smoke plus:
        - Server-reported generation_time captured
        """
        manifest = run_two_stage_via_api(
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

        # Server-reported timing
        assert manifest.stage_timings.get("total_server", 0) > 0, (
            "Expected server to report generation_time"
        )

    @pytest.mark.slow
    @_skip_no_server
    def test_two_stage_reproducibility(self, output_dir):
        """Same seed twice via API -> SSIM > 0.90 between runs.

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
        manifest1 = run_two_stage_via_api(
            prompt=prompt, num_frames=num_frames, height=height, width=width,
            seed=seed, stage1_steps=stage1_steps, stage2_steps=stage2_steps,
            output_dir=run1_dir, test_name="two_stage_repro_run1",
            num_sample_frames=4,
        )

        # Run 2
        run2_dir = output_dir / "run2"
        run2_dir.mkdir()
        manifest2 = run_two_stage_via_api(
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
