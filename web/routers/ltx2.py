"""LTX-2 video generation endpoints: status and streaming generation."""

import asyncio
import gc
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
from web.param_resolver import csv_to_int_list, resolve_param
from web.schemas import LTX2GenerateRequest, LTX2StatusResponse

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
    """Get validated LTX-2 model path from the injected RuntimeConfig.

    Uses config.ltx2.model_path directly -- no need to re-parse TOML since
    the RuntimeConfig already has the composed sub-configs.

    Returns:
        Path to LTX-2 model directory

    Raises:
        ValueError if model path not found or not configured
    """
    model_path = getattr(config.ltx2, "model_path", "") if config.ltx2 else ""
    if not model_path:
        raise ValueError("LTX-2 not configured. Set ltx2.model_path in config.toml")
    path = Path(model_path).expanduser()
    if not path.exists():
        raise ValueError(f"LTX-2 model path not found: {path}")
    return path


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


@router.get("/api/ltx2/status", response_model=LTX2StatusResponse)
async def ltx2_status(config: ConfigDep) -> LTX2StatusResponse:
    """Get LTX-2 pipeline status.

    Returns availability based on the injected RuntimeConfig -- no TOML re-parsing.
    """
    ltx2_configured = False
    model_path = getattr(config.ltx2, "model_path", "") if config.ltx2 else ""
    if model_path:
        ltx2_configured = Path(model_path).expanduser().exists()

    return LTX2StatusResponse(
        available=ltx2_configured,
        loaded=False,
        vram_used_gb=None,
    )


@router.post("/api/ltx2/generate/stream")
async def ltx2_generate_stream(request: LTX2GenerateRequest, config: ConfigDep):
    """Generate video with SSE progress streaming.

    Supports two-stage generation (default) or single-stage fallback.
    Returns Server-Sent Events with multi-stage progress updates.
    """

    async def generate() -> AsyncIterator[str]:
        """Async generator for SSE events."""
        try:
            yield f"data: {json.dumps({'type': 'status', 'message': 'Validating LTX-2 configuration...'})}\n\n"

            model_path = await asyncio.get_event_loop().run_in_executor(
                None, lambda: get_ltx2_model_path(config)
            )

            mode = "two-stage" if request.use_two_stage else "single-stage"
            yield f"data: {json.dumps({'type': 'status', 'message': f'Starting {mode} generation...'})}\n\n"

            # Progress tracking -- shared dict updated by generation thread
            progress_state = {"stage": "", "step": 0, "total": 0}

            def progress_callback(stage: str, step: int, total: int) -> None:
                progress_state["stage"] = stage
                progress_state["step"] = step
                progress_state["total"] = total

            seed = request.seed if request.seed is not None else int(time.time()) % (2**32)
            start_time = time.time()

            # Merge request params with config defaults
            ltx2_cfg = config.ltx2

            @torch.no_grad()
            def do_generate():
                from llm_dit.pipelines import (
                    GenerationConfig,
                    TwoStageConfig,
                    generate_video_two_stage,
                    generate_video_with_offloading,
                )

                gen_config = GenerationConfig(
                    num_frames=resolve_param(request, "num_frames", ltx2_cfg.num_frames),
                    height=resolve_param(request, "height", ltx2_cfg.height),
                    width=resolve_param(request, "width", ltx2_cfg.width),
                    guidance_scale=resolve_param(request, "guidance_scale", ltx2_cfg.guidance_scale),
                    seed=seed,
                )

                # Resolve text encoder path from config (default: model_path/text_encoder)
                encoder_model_id = ltx2_cfg.encoder_model_id if ltx2_cfg else None
                text_encoder_path = encoder_model_id if encoder_model_id else None

                if request.use_two_stage:
                    # Resolve stg_blocks: client list > config CSV string > default
                    stg_blocks_from_config = csv_to_int_list(ltx2_cfg.stg_blocks) if ltx2_cfg else [29]
                    stg_blocks = resolve_param(request, "stg_blocks", stg_blocks_from_config, skip_none=True)

                    two_stage_cfg = TwoStageConfig(
                        stage1_steps=resolve_param(request, "stage1_steps", ltx2_cfg.stage1_num_inference_steps, skip_none=True),
                        stage2_steps=resolve_param(request, "stage2_steps", ltx2_cfg.stage2_num_inference_steps, skip_none=True),
                        guidance_scale=resolve_param(request, "guidance_scale", ltx2_cfg.guidance_scale),
                        stg_scale=resolve_param(request, "stg_scale", ltx2_cfg.stg_scale),
                        rescale_scale=resolve_param(request, "rescale_scale", ltx2_cfg.rescale_scale),
                        negative_prompt=resolve_param(request, "negative_prompt", ltx2_cfg.negative_prompt),
                        ge_gamma=resolve_param(request, "ge_gamma", ltx2_cfg.ge_gamma),
                        distilled_lora_path=resolve_param(request, "distilled_lora_path", ltx2_cfg.distilled_lora_path, skip_none=True),
                        distilled_lora_scale=resolve_param(request, "distilled_lora_scale", ltx2_cfg.distilled_lora_scale),
                        spatial_upsampler_file=ltx2_cfg.spatial_upsampler_file if ltx2_cfg else "ltx-2-spatial-upscaler-x2-1.0.safetensors",
                    )
                    if stg_blocks is not None:
                        two_stage_cfg.stg_blocks = stg_blocks

                    # Validate distilled LoRA exists before expensive generation
                    distilled_path = two_stage_cfg.distilled_lora_path
                    if distilled_path:
                        resolved = Path(distilled_path)
                        if not resolved.is_absolute():
                            resolved = model_path / resolved
                        if not resolved.exists():
                            raise ValueError(f"Distilled LoRA not found: {resolved}")

                    return generate_video_two_stage(
                        prompt=request.prompt,
                        config=gen_config,
                        two_stage=two_stage_cfg,
                        model_path=model_path,
                        text_encoder_path=text_encoder_path,
                        callback=progress_callback,
                        gemma_variant=ltx2_cfg.gemma_variant if ltx2_cfg else "bf16",
                        lora_path=request.lora_path,
                        lora_scale=request.lora_scale,
                        text_encoder_device=ltx2_cfg.text_encoder_device if ltx2_cfg else "cpu",
                        transformer_device=ltx2_cfg.transformer_device if ltx2_cfg else "cuda",
                        vae_device=ltx2_cfg.vae_device if ltx2_cfg else "cuda",
                        quantize=ltx2_cfg.quantize if ltx2_cfg else "fp8",
                        skip_cleanup=ltx2_cfg.skip_cleanup if ltx2_cfg else False,
                    )
                else:
                    # Single-stage fallback
                    gen_config.num_inference_steps = resolve_param(request, "stage1_steps", ltx2_cfg.num_inference_steps if ltx2_cfg else 40, skip_none=True)
                    return generate_video_with_offloading(
                        prompt=request.prompt,
                        config=gen_config,
                        model_path=model_path,
                        text_encoder_path=text_encoder_path,
                        callback=progress_callback,
                        use_progress=False,
                        lora_path=request.lora_path,
                        lora_scale=request.lora_scale,
                        gemma_variant=ltx2_cfg.gemma_variant if ltx2_cfg else "bf16",
                        text_encoder_device=ltx2_cfg.text_encoder_device if ltx2_cfg else "cpu",
                        transformer_device=ltx2_cfg.transformer_device if ltx2_cfg else "cuda",
                        vae_device=ltx2_cfg.vae_device if ltx2_cfg else "cuda",
                        quantize=ltx2_cfg.quantize if ltx2_cfg else "fp8",
                        skip_cleanup=ltx2_cfg.skip_cleanup if ltx2_cfg else False,
                    )

            loop = asyncio.get_event_loop()
            gen_task = loop.run_in_executor(None, do_generate)

            # Poll progress while generating
            while not gen_task.done():
                await asyncio.sleep(0.5)
                stage = progress_state["stage"]
                step = progress_state["step"]
                total = progress_state["total"]
                elapsed = time.time() - start_time

                if stage and step > 0 and total > 0:
                    eta = (elapsed / step) * (total - step)
                    its = step / elapsed if elapsed > 0 else 0
                    yield f"data: {json.dumps({'type': 'progress', 'stage': stage, 'step': step, 'total': total, 'elapsed': round(elapsed, 1), 'eta': round(eta, 1), 'its': round(its, 2)})}\n\n"

            video = await gen_task
            generation_time = time.time() - start_time

            yield f"data: {json.dumps({'type': 'status', 'message': 'Saving video...'})}\n\n"

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            hash_suffix = hashlib.md5(f"{request.prompt}{seed}".encode()).hexdigest()[:8]
            video_filename = f"video_{timestamp}_{hash_suffix}.mp4"
            video_path = VIDEO_OUTPUT_DIR / video_filename

            resolved_fps = resolve_param(request, "fps", ltx2_cfg.fps if ltx2_cfg else 24.0)
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: save_ltx2_video(video, str(video_path), fps=resolved_fps)
            )

            # Generate thumbnail (first frame)
            thumb_filename = f"thumb_{timestamp}_{hash_suffix}.png"
            thumb_path = VIDEO_OUTPUT_DIR / thumb_filename
            try:
                first_frame = video[0].cpu().numpy()
                Image.fromarray(first_frame).save(str(thumb_path))
            except Exception as e:
                logger.warning(f"Failed to save thumbnail: {e}")
                thumb_filename = None

            result = {
                "type": "complete",
                "urls": [f"/outputs/videos/{video_filename}"],
                "url": f"/outputs/videos/{video_filename}",
                "thumbnail_url": f"/outputs/videos/{thumb_filename}" if thumb_filename else None,
                "seed": seed,
                "generation_time": round(generation_time, 1),
                "num_frames": resolve_param(request, "num_frames", ltx2_cfg.num_frames if ltx2_cfg else 33),
                "fps": resolved_fps,
                "has_audio": False,
                "two_stage": request.use_two_stage,
            }
            yield f"data: {json.dumps(result)}\n\n"

        except Exception as e:
            logger.error(f"[LTX-2] Generation failed: {e}")
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
