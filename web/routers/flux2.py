"""FLUX.2 Klein generation endpoints: status, generation, and streaming."""

import asyncio
import base64
import io
import json
import logging
import time
import traceback
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image

from web.dependencies import ConfigDep
from web.schemas import Flux2GenerateRequest
from web.utils import create_image_response

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Status
# =============================================================================


@router.get("/api/flux2/status")
async def flux2_status():
    """Get FLUX.2 Klein pipeline status.

    Returns availability info. FLUX.2 is always available (downloads from HuggingFace).
    """
    import web.server as srv

    return {
        "available": True,  # FLUX.2 downloads models from HuggingFace as needed
        "loaded": srv.flux2_pipeline is not None,
        "supported_models": [
            "klein-9b", "klein-9b-fp8", "klein-4b", "klein-4b-fp8",
            "klein-base-9b", "klein-base-9b-fp8", "klein-base-4b", "klein-base-4b-fp8"
        ],
    }


# =============================================================================
# Generation
# =============================================================================


@router.post("/api/flux2/generate")
async def flux2_generate(request: Flux2GenerateRequest, config: ConfigDep):
    """Generate image using FLUX.2 Klein.

    Supports both text-to-image and image editing with reference images.
    Returns PNG image as binary response.
    """
    import web.server as srv

    try:
        # Import the FLUX.2 generation pipeline
        from llm_dit.pipelines.flux2_generate import (
            Flux2GenerationConfig,
            generate_image,
        )
        from llm_dit.models.flux2.constants import FLUX2_MODEL_INFO

        # Get model defaults if steps/guidance not specified
        model_info = FLUX2_MODEL_INFO.get(request.model_name.lower(), {})
        defaults = model_info.get("defaults", {"guidance": 1.0, "num_steps": 4})

        num_steps = request.num_steps if request.num_steps is not None else defaults["num_steps"]
        guidance = request.guidance if request.guidance is not None else defaults["guidance"]

        # Process reference images if provided
        ref_images = []
        if request.reference_images:
            for ref_b64 in request.reference_images:
                # Remove data URL prefix if present
                if "," in ref_b64:
                    ref_b64 = ref_b64.split(",", 1)[1]
                img_data = base64.b64decode(ref_b64)
                img = Image.open(io.BytesIO(img_data)).convert("RGB")
                ref_images.append(img)

        # Parse match_image_size from UI string to int
        # "none" -> None, "0 (First Image)" -> 0, etc.
        match_image_size: Optional[int] = None
        if request.match_image_size and request.match_image_size != "none":
            # Extract the number from the start of the string (e.g., "0 (First Image)" -> 0)
            try:
                match_image_size = int(request.match_image_size.split()[0])
            except (ValueError, IndexError):
                match_image_size = None

        # Get offload settings from runtime config (config.toml)
        offload_between_stages = True
        offload_between_stages = getattr(config, "flux2_offload_between_stages", True)

        # Force offload when models aren't pre-loaded (encoder would stay on GPU during transformer load)
        if not isinstance(srv.flux2_pipeline, dict):
            offload_between_stages = True

        # Create generation config
        gen_config = Flux2GenerationConfig(
            prompt=request.prompt,
            height=request.height,
            width=request.width,
            num_steps=num_steps,
            guidance=guidance,
            seed=request.seed,
            reference_images=ref_images,
            match_image_size=match_image_size,
            block_offload=request.block_offload,
            offload_between_stages=offload_between_stages,
            loras=request.loras,
            # Text encoding options
            max_text_length=request.max_text_length,
            pad_to_max=request.pad_to_max,
            output_layers=request.output_layers,
        )

        # Get model/VAE paths - prefer request values, fall back to config
        model_path = request.model_path
        vae_path = request.vae_path

        if not model_path:
            model_path = getattr(config, "flux2_model_path", None)
        if not vae_path:
            vae_path = getattr(config, "flux2_vae_path", None)

        # Generate image
        start_time = time.time()
        logger.info(f"[FLUX.2] Generating {request.width}x{request.height} with {request.model_name}")
        if model_path:
            logger.info(f"[FLUX.2] Using model path: {model_path}")

        # Pass persistent models if pipeline is preloaded
        persistent_encoder = None
        persistent_transformer = None
        persistent_vae = None
        if isinstance(srv.flux2_pipeline, dict):
            persistent_encoder = srv.flux2_pipeline.get("encoder")
            persistent_transformer = srv.flux2_pipeline.get("transformer")
            persistent_vae = srv.flux2_pipeline.get("vae")

        # Run in executor to not block event loop
        loop = asyncio.get_event_loop()
        image = await loop.run_in_executor(
            None,
            lambda: generate_image(
                gen_config,
                model_name=request.model_name,
                encoder=persistent_encoder,
                transformer=persistent_transformer,
                vae=persistent_vae,
                model_path=model_path,
                vae_path=vae_path,
            )
        )

        gen_time = time.time() - start_time
        logger.info(f"[FLUX.2] Generation complete in {gen_time:.1f}s")

        # Return standardized JSON response (same format as Z-Image)
        return create_image_response(
            image=image,
            pipeline_id="flux2",
            seed=gen_config.seed,
            generation_time=gen_time,
        )

    except Exception as e:
        logger.error(f"[FLUX.2] Generation failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Streaming Generation
# =============================================================================


@router.post("/api/flux2/generate/stream")
async def flux2_generate_stream(request: Flux2GenerateRequest, config: ConfigDep):
    """Generate image using FLUX.2 Klein with SSE progress streaming.

    Returns Server-Sent Events with progress updates during generation,
    allowing the frontend to show step-by-step progress.

    Events:
    - {"type": "status", "message": "..."} - Status updates
    - {"type": "progress", "step": N, "total_steps": M} - Step progress
    - {"type": "complete", ...} - Final result with image data
    - {"type": "error", "message": "..."} - Error occurred
    """
    import web.server as srv
    from typing import AsyncIterator

    async def generate_with_progress() -> AsyncIterator[str]:
        """Async generator for SSE events."""
        try:
            # Import the FLUX.2 generation pipeline
            from llm_dit.pipelines.flux2_generate import (
                Flux2GenerationConfig,
                generate_image_with_progress,
            )
            from llm_dit.models.flux2.constants import FLUX2_MODEL_INFO

            # Get model defaults if steps/guidance not specified
            model_info = FLUX2_MODEL_INFO.get(request.model_name.lower(), {})
            defaults = model_info.get("defaults", {"guidance": 1.0, "num_steps": 4})

            num_steps = request.num_steps if request.num_steps is not None else defaults["num_steps"]
            guidance = request.guidance if request.guidance is not None else defaults["guidance"]

            yield f"data: {json.dumps({'type': 'status', 'message': 'Processing request...'})}\n\n"

            # Process reference images if provided
            ref_images = []
            if request.reference_images:
                yield f"data: {json.dumps({'type': 'status', 'message': f'Processing {len(request.reference_images)} reference image(s)...'})}\n\n"
                for ref_b64 in request.reference_images:
                    # Remove data URL prefix if present
                    if "," in ref_b64:
                        ref_b64 = ref_b64.split(",", 1)[1]
                    img_data = base64.b64decode(ref_b64)
                    img = Image.open(io.BytesIO(img_data)).convert("RGB")
                    ref_images.append(img)

            # Parse match_image_size from UI string to int
            match_image_size: Optional[int] = None
            if request.match_image_size and request.match_image_size != "none":
                try:
                    match_image_size = int(request.match_image_size.split()[0])
                except (ValueError, IndexError):
                    match_image_size = None

            # Get offload settings from runtime config (config.toml)
            offload_between_stages = True
            offload_between_stages = getattr(config, "flux2_offload_between_stages", True)

            # Force offload when models aren't pre-loaded (encoder would stay on GPU during transformer load)
            if not isinstance(srv.flux2_pipeline, dict):
                offload_between_stages = True

            # Create generation config
            gen_config = Flux2GenerationConfig(
                prompt=request.prompt,
                height=request.height,
                width=request.width,
                num_steps=num_steps,
                guidance=guidance,
                seed=request.seed,
                reference_images=ref_images,
                match_image_size=match_image_size,
                block_offload=request.block_offload,
                offload_between_stages=offload_between_stages,
                loras=request.loras,
                max_text_length=request.max_text_length,
                pad_to_max=request.pad_to_max,
                output_layers=request.output_layers,
            )

            # Get model/VAE paths
            model_path = request.model_path
            vae_path = request.vae_path
            if not model_path:
                model_path = getattr(config, "flux2_model_path", None)
            if not vae_path:
                vae_path = getattr(config, "flux2_vae_path", None)

            start_time = time.time()
            logger.info(f"[FLUX.2] Generating {request.width}x{request.height} with {request.model_name}")

            # Progress callback that yields SSE events
            def progress_callback(step: int, total: int, stage: str = ""):
                return {
                    "step": step,
                    "total": total,
                    "stage": stage,
                }

            # Run generation with progress in thread pool
            loop = asyncio.get_event_loop()
            progress_queue: asyncio.Queue = asyncio.Queue()

            def run_generation():
                """Run generation and put progress events in queue."""
                def callback(step: int, total: int, stage: str = ""):
                    # Use call_soon_threadsafe to safely put in queue from thread
                    loop.call_soon_threadsafe(
                        progress_queue.put_nowait,
                        {"step": step, "total": total, "stage": stage}
                    )

                # Pass persistent models if pipeline is preloaded
                p_encoder = None
                p_transformer = None
                p_vae = None
                if isinstance(srv.flux2_pipeline, dict):
                    p_encoder = srv.flux2_pipeline.get("encoder")
                    p_transformer = srv.flux2_pipeline.get("transformer")
                    p_vae = srv.flux2_pipeline.get("vae")

                return generate_image_with_progress(
                    gen_config,
                    model_name=request.model_name,
                    encoder=p_encoder,
                    transformer=p_transformer,
                    vae=p_vae,
                    model_path=model_path,
                    vae_path=vae_path,
                    progress_callback=callback,
                )

            # Start generation in background
            gen_future = loop.run_in_executor(None, run_generation)

            # Yield progress events as they come
            last_step = -1
            while not gen_future.done():
                try:
                    # Wait for progress event with timeout
                    progress = await asyncio.wait_for(progress_queue.get(), timeout=0.5)
                    if progress["step"] != last_step:
                        last_step = progress["step"]
                        yield f"data: {json.dumps({'type': 'progress', 'step': progress['step'], 'total_steps': progress['total'], 'message': progress.get('stage', '')})}\n\n"
                except asyncio.TimeoutError:
                    continue

            # Get result
            image = await gen_future

            # Drain any remaining progress events
            while not progress_queue.empty():
                try:
                    progress_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

            gen_time = time.time() - start_time
            logger.info(f"[FLUX.2] Generation complete in {gen_time:.1f}s")

            # Convert image to base64
            img_bytes = io.BytesIO()
            image.save(img_bytes, format="PNG")
            img_b64 = base64.b64encode(img_bytes.getvalue()).decode("ascii")
            data_url = f"data:image/png;base64,{img_b64}"

            # Yield final result
            yield f"data: {json.dumps({'type': 'complete', 'urls': [data_url], 'url': data_url, 'seed': gen_config.seed, 'generation_time': gen_time})}\n\n"

        except Exception as e:
            logger.error(f"[FLUX.2] Stream generation failed: {e}")
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate_with_progress(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
