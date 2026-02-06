"""LTX-2 video generation endpoints: status and streaming generation."""

import asyncio
import hashlib
import json
import logging
import time
import traceback
from pathlib import Path
from typing import AsyncIterator

import torch
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from PIL import Image

from llm_dit.config import RuntimeConfig
from web.dependencies import ConfigDep
from web.schemas import LTX2GenerateRequest

logger = logging.getLogger(__name__)

router = APIRouter()

# Video output directory
VIDEO_OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "videos"
VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


async def cleanup_old_videos(max_age_hours: int = 24) -> int:
    """Delete videos older than max_age_hours.

    Called on startup to prevent unbounded storage growth.
    Returns count of deleted files.
    """
    max_age_seconds = max_age_hours * 3600
    now = time.time()
    deleted_count = 0

    if VIDEO_OUTPUT_DIR.exists():
        for video_file in VIDEO_OUTPUT_DIR.glob("*.mp4"):
            try:
                age = now - video_file.stat().st_mtime
                if age > max_age_seconds:
                    video_file.unlink()
                    deleted_count += 1
                    thumb = video_file.with_suffix(".png")
                    if thumb.exists():
                        thumb.unlink()
            except OSError as e:
                logger.warning(f"Failed to delete old video {video_file}: {e}")

    if deleted_count > 0:
        logger.info(f"[Cleanup] Deleted {deleted_count} videos older than {max_age_hours}h")

    return deleted_count


@router.on_event("startup")
async def startup_video_cleanup():
    """Clean up old videos on server startup."""
    await cleanup_old_videos(max_age_hours=24)


def get_ltx2_model_path(config: RuntimeConfig) -> Path:
    """Get validated LTX-2 model path from config or default location.

    Returns:
        Path to LTX-2 model directory

    Raises:
        ValueError if model path not found or not configured
    """
    # Try to load from config if available
    config_path = getattr(config, "config_path", None)
    profile = getattr(config, "current_profile", "default")

    if config_path:
        from llm_dit.config import Config

        loaded_config = Config.load(config_path, profile=profile)
        if loaded_config.ltx2 and loaded_config.ltx2.model_path:
            model_path = Path(loaded_config.ltx2.model_path).expanduser()
            if model_path.exists():
                return model_path
            raise ValueError(f"LTX-2 model path not found: {model_path}")
        else:
            raise ValueError(
                "LTX-2 not configured. Set ltx2.model_path in config.toml "
                f"under [{profile}.ltx2] section."
            )

    # Fallback: Try default path
    default_path = Path.home() / "Storage" / "LTX-2"
    if default_path.exists():
        return default_path

    raise ValueError(
        f"LTX-2 model not found at {default_path}. "
        "Configure ltx2.model_path in config.toml."
    )


def save_ltx2_video(video: torch.Tensor, path: str, fps: float = 24.0) -> str:
    """Save LTX-2 video tensor to file.

    Args:
        video: Video frames [F, H, W, C] in uint8 format
        path: Output path (.mp4)
        fps: Frame rate

    Returns:
        Path to saved video
    """
    # Convert tensor to numpy
    video_np = video.cpu().numpy()

    try:
        import imageio.v3 as iio

        codec = "libvpx-vp9" if path.endswith(".webm") else "libx264"
        with iio.imopen(path, "w", plugin="FFMPEG") as writer:
            writer.write(video_np, fps=fps, codec=codec)
        logger.info(f"[LTX-2] Saved video to {path}")
    except Exception as e:
        logger.warning(f"[LTX-2] imageio failed: {e}, trying torchvision")
        import torchvision.io as tvio

        tvio.write_video(path, video, fps=fps)
        logger.info(f"[LTX-2] Saved video to {path}")

    return path


@router.get("/api/ltx2/status")
async def ltx2_status(config: ConfigDep):
    """Get LTX-2 pipeline status.

    Returns availability, loaded state, and VRAM usage.
    """
    # Check if LTX-2 config exists in the loaded config file
    ltx2_configured = False

    config_path = getattr(config, "config_path", None)
    profile = getattr(config, "current_profile", "default")

    if config_path:
        try:
            from llm_dit.config import Config

            loaded_config = Config.load(config_path, profile=profile)
            if loaded_config.ltx2 and loaded_config.ltx2.model_path:
                # Check if path actually exists
                model_dir = Path(loaded_config.ltx2.model_path).expanduser()
                ltx2_configured = model_dir.exists()
        except Exception:
            pass

    # Check default path if not configured
    if not ltx2_configured:
        default_path = Path.home() / "Storage" / "LTX-2"
        ltx2_configured = default_path.exists()

    return {
        "available": ltx2_configured,
        # Note: Pure PyTorch pipeline loads/unloads components per request
        # so "loaded" always returns False (no persistent state)
        "loaded": False,
        "vram_used_gb": None,  # TODO: Track actual VRAM usage per model
    }


@router.post("/api/ltx2/generate/stream")
async def ltx2_generate_stream(request: LTX2GenerateRequest, config: ConfigDep):
    """Generate video with SSE progress streaming.

    Returns Server-Sent Events with progress updates during generation,
    then final result with video URL.

    Uses pure PyTorch generation via generate_video_with_offloading().
    """

    async def generate() -> AsyncIterator[str]:
        """Async generator for SSE events."""
        try:
            # Yield initial status
            yield f"data: {json.dumps({'type': 'status', 'message': 'Validating LTX-2 configuration...'})}\n\n"

            # Validate model path exists
            model_path = await asyncio.get_event_loop().run_in_executor(
                None, lambda: get_ltx2_model_path(config)
            )

            yield f"data: {json.dumps({'type': 'status', 'message': 'Starting generation...'})}\n\n"

            # Progress tracking
            progress_state = {"stage": "", "step": 0, "total": request.num_inference_steps}

            def progress_callback(stage: str, step: int, total: int) -> None:
                """Callback to track progress (can't yield directly from here)."""
                progress_state["stage"] = stage
                progress_state["step"] = step
                progress_state["total"] = total

            # Generate video (blocking)
            seed = request.seed if request.seed is not None else int(time.time()) % (2**32)

            start_time = time.time()

            # Run generation in thread pool to not block event loop
            def do_generate():
                from llm_dit.pipelines import generate_video_with_offloading, GenerationConfig

                # Create generation config from request
                gen_config = GenerationConfig(
                    num_frames=request.num_frames,
                    height=request.height,
                    width=request.width,
                    num_inference_steps=request.num_inference_steps,
                    guidance_scale=request.guidance_scale,
                    seed=seed,
                )

                # Generate video with component offloading
                return generate_video_with_offloading(
                    prompt=request.prompt,
                    config=gen_config,
                    model_path=model_path,
                    callback=progress_callback,
                    use_progress=False,  # Disable tqdm, use callback instead
                    lora_path=request.lora_path,
                    lora_scale=request.lora_scale,
                )

            # Start generation in background
            loop = asyncio.get_event_loop()
            gen_task = loop.run_in_executor(None, do_generate)

            # Poll progress while generating
            while not gen_task.done():
                await asyncio.sleep(0.5)
                stage = progress_state["stage"]
                step = progress_state["step"]
                total = progress_state["total"]
                elapsed = time.time() - start_time

                if stage and step > 0:
                    eta = (elapsed / step) * (total - step) if total > 0 else 0
                    its = step / elapsed if elapsed > 0 else 0
                    yield f"data: {json.dumps({'type': 'progress', 'stage': stage, 'step': step, 'total': total, 'elapsed': round(elapsed, 1), 'eta': round(eta, 1), 'its': round(its, 2)})}\n\n"

            # Get result (video tensor [F, H, W, C] uint8)
            video = await gen_task
            generation_time = time.time() - start_time

            yield f"data: {json.dumps({'type': 'status', 'message': 'Saving video...'})}\n\n"

            # Generate unique filename
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            hash_suffix = hashlib.md5(f"{request.prompt}{seed}".encode()).hexdigest()[:8]
            video_filename = f"video_{timestamp}_{hash_suffix}.mp4"
            video_path = VIDEO_OUTPUT_DIR / video_filename

            # Save video
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: save_ltx2_video(video, str(video_path), fps=request.fps)
            )

            # Generate thumbnail (first frame)
            thumb_filename = f"thumb_{timestamp}_{hash_suffix}.png"
            thumb_path = VIDEO_OUTPUT_DIR / thumb_filename

            try:
                # video is [F, H, W, C] uint8 tensor
                first_frame = video[0].cpu().numpy()
                Image.fromarray(first_frame).save(str(thumb_path))
            except Exception as e:
                logger.warning(f"Failed to save thumbnail: {e}")
                thumb_filename = None

            # Return final result
            # Note: Audio generation not yet implemented in pure PyTorch pipeline
            result = {
                "type": "complete",
                "video_url": f"/outputs/videos/{video_filename}",
                "thumbnail_url": f"/outputs/videos/{thumb_filename}" if thumb_filename else None,
                "seed": seed,
                "generation_time": round(generation_time, 1),
                "num_frames": request.num_frames,
                "fps": request.fps,
                "has_audio": False,  # Audio not yet implemented in pure PyTorch pipeline
            }
            yield f"data: {json.dumps(result)}\n\n"

        except Exception as e:
            logger.error(f"[LTX-2] Generation failed: {e}")
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )
