"""FLUX.2 Klein generation endpoints: status, generation, and streaming."""

import asyncio
import base64
import gc
import io
import json
import logging
import time
import traceback
from typing import Optional

import torch
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image

from web.dependencies import ConfigDep, ManagerDep
from web.param_resolver import resolve_param
from web.schemas import Flux2GenerateRequest, Flux2StatusResponse, ImageGenerationResult
from web.utils import create_image_response

logger = logging.getLogger(__name__)

router = APIRouter()


def _resolve_flux2_params(
    request: Flux2GenerateRequest,
    config,
) -> tuple[int, float, list[str]]:
    """Resolve num_steps and guidance from request/config/model defaults.

    Returns (num_steps, guidance, warnings).
    """
    from llm_dit.models.flux2.constants import FLUX2_MODEL_INFO, get_fixed_params

    model_info = FLUX2_MODEL_INFO.get(request.model_name.lower(), {})
    model_defaults = model_info.get("defaults", {"guidance": 1.0, "num_steps": 4})
    fixed = get_fixed_params(request.model_name)

    flux2_cfg = config.flux2 if hasattr(config, "flux2") and config.flux2 else None
    num_steps = resolve_param(request, "num_steps", flux2_cfg.default_steps if flux2_cfg else None, skip_none=True)
    if num_steps is None:
        num_steps = model_defaults["num_steps"]
    guidance = resolve_param(request, "guidance", flux2_cfg.default_guidance if flux2_cfg else None, skip_none=True)
    if guidance is None:
        guidance = model_defaults["guidance"]

    warnings: list[str] = []
    if "num_steps" in fixed and "num_steps" in request.model_fields_set and request.num_steps != model_defaults["num_steps"]:
        warnings.append(
            f"Distilled model '{request.model_name}' requires num_steps={model_defaults['num_steps']}. "
            f"Overriding requested num_steps={request.num_steps}."
        )
        num_steps = model_defaults["num_steps"]
        logger.warning(f"[FLUX.2] {warnings[-1]}")

    if "guidance" in fixed and "guidance" in request.model_fields_set and request.guidance != model_defaults["guidance"]:
        warnings.append(
            f"Distilled model '{request.model_name}' requires guidance={model_defaults['guidance']}. "
            f"Overriding requested guidance={request.guidance}."
        )
        guidance = model_defaults["guidance"]
        logger.warning(f"[FLUX.2] {warnings[-1]}")

    return num_steps, guidance, warnings


def _upsample_prompt(
    config,
    prompt: str,
    has_reference_images: bool,
) -> tuple[str, str | None]:
    """Upsample a prompt via heylookitsanllm, reading URL/model from config.

    Returns (upsampled_prompt, warning_message_or_none).
    Falls back to original prompt on API error or missing config.
    """
    from llm_dit.utils.prompt_rewriter import Flux2PromptUpsampler

    api_url = getattr(config, "rewriter_api_url", "") or getattr(config, "api_url", None)
    if not api_url:
        return prompt, "Prompt upsampling requested but no API URL configured."

    api_model: str = getattr(config, "rewriter_api_model", "")
    if api_model:
        upsampler = Flux2PromptUpsampler(api_url=api_url, api_model=api_model)
    else:
        upsampler = Flux2PromptUpsampler(api_url=api_url)

    original = prompt
    prompt = upsampler.upsample(prompt, has_reference_images=has_reference_images)
    if prompt != original:
        logger.info(f"[FLUX.2] Prompt upsampled: {len(original)} -> {len(prompt)} chars")
        return prompt, f"Prompt upsampled from: {original[:100]}"
    return prompt, None


def _ensure_correct_model(
    requested_model: str,
    manager,
    requested_loras: list[str] | None = None,
) -> dict | None:
    """Verify loaded FLUX.2 model matches the request; reload if mismatched.

    Returns the (possibly-updated) flux2_pipeline dict, or None if no
    persistent pipeline is available.

    Checks both model name and LoRA specs. When a mismatch is detected,
    this blocks synchronously while the ModelManager unloads the old model
    and loads the new one. The ModelManager's per-pipeline lock prevents
    concurrent reloads.
    """
    if not manager.is_loaded("flux2"):
        return None

    pipeline_dict = manager.get_pipeline("flux2")
    if not isinstance(pipeline_dict, dict):
        return None

    needs_reload = False

    # Check model name mismatch
    loaded_name = pipeline_dict.get("model_name")
    if loaded_name and loaded_name.lower() != requested_model.lower():
        logger.info(
            f"[FLUX.2] Model mismatch: loaded='{loaded_name}', "
            f"requested='{requested_model}'. Reloading..."
        )
        needs_reload = True

    # Check LoRA mismatch on the persistent transformer
    # Filter out empty-path specs (e.g., ":0.80" from unselected UI slots)
    if requested_loras:
        requested_loras = [s for s in requested_loras if not s.startswith(":")]
    if not needs_reload and requested_loras:
        from llm_dit.utils.lora import get_fused_state, parse_lora_spec

        transformer = pipeline_dict.get("transformer")
        if transformer is not None:
            fused_state = get_fused_state(transformer)
            requested_specs = [parse_lora_spec(s) for s in requested_loras]
            if not fused_state.is_empty and not fused_state.matches(requested_specs):
                logger.info(
                    f"[FLUX.2] LoRA mismatch: fused=[{fused_state.summary()}], "
                    f"requested={requested_loras}. Reloading..."
                )
                needs_reload = True
        # Drop local ref so _unload_flux2() can free GPU memory
        del transformer

    if needs_reload:
        result = manager.reload_flux2(requested_model)
        logger.info(
            f"[FLUX.2] Reload complete: now loaded='{requested_model}' "
            f"({result.load_time:.1f}s)"
        )
        pipeline_dict = manager.get_pipeline("flux2")

    return pipeline_dict if isinstance(pipeline_dict, dict) else None


# =============================================================================
# Status
# =============================================================================


@router.get("/api/flux2/status", response_model=Flux2StatusResponse)
async def flux2_status(config: ConfigDep, manager: ManagerDep) -> Flux2StatusResponse:
    """Get FLUX.2 Klein pipeline status.

    Returns availability info. FLUX.2 is always available (downloads from HuggingFace).
    """
    return Flux2StatusResponse(
        available=True,  # FLUX.2 downloads models from HuggingFace as needed
        loaded=manager.is_loaded("flux2"),
        compile_enabled=getattr(config, "flux2_compile", False),
        compile_dynamic=getattr(config, "flux2_compile_dynamic", False),
        compile_vae_enabled=getattr(config, "flux2_compile_vae", False),
        supported_models=[
            "klein-9b", "klein-9b-fp8", "klein-4b", "klein-4b-fp8",
            "klein-base-9b", "klein-base-9b-fp8", "klein-base-4b", "klein-base-4b-fp8",
            "klein-9b-kv", "klein-9b-kv-fp8",
        ],
    )


# =============================================================================
# Model Info
# =============================================================================


@router.get("/api/flux2/models/{model_name}")
async def flux2_model_info(model_name: str):
    """Get model metadata including distilled status and fixed params.

    Used by frontend to conditionally disable controls for distilled models.
    """
    from llm_dit.models.flux2.constants import FLUX2_MODEL_INFO

    model_name_lower = model_name.lower()
    if model_name_lower not in FLUX2_MODEL_INFO:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_name}")

    info = FLUX2_MODEL_INFO[model_name_lower]
    return {
        "model_name": model_name_lower,
        "distilled": info["distilled"],
        "fixed_params": sorted(info.get("fixed_params", set())),
        "defaults": info["defaults"],
        "fp8": info.get("fp8", False),
    }


# =============================================================================
# Generation
# =============================================================================


@router.post("/api/flux2/generate", response_model=ImageGenerationResult)
async def flux2_generate(request: Flux2GenerateRequest, config: ConfigDep, manager: ManagerDep) -> ImageGenerationResult:
    """Generate image using FLUX.2 Klein.

    Supports both text-to-image and image editing with reference images.
    Returns PNG image as binary response.
    """
    try:
        # Auto-load on first request (like LTX-2 pattern)
        if not manager.is_loaded("flux2"):
            logger.info("[FLUX.2] Auto-loading pipeline (first request)...")
            await asyncio.get_event_loop().run_in_executor(None, lambda: manager.load("flux2"))

        # Import the FLUX.2 generation pipeline
        from llm_dit.pipelines.flux2_generate import (
            Flux2GenerationConfig,
            generate_image,
        )
        num_steps, guidance, warnings = _resolve_flux2_params(request, config)

        # Prompt upsampling (optional, requires heylookitsanllm API)
        prompt = request.prompt
        if request.upsample_prompt:
            prompt, upsample_warning = _upsample_prompt(
                config, prompt, has_reference_images=bool(request.reference_images),
            )
            if upsample_warning:
                warnings.append(upsample_warning)

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
        if not manager.is_loaded("flux2"):
            offload_between_stages = True

        # Create generation config
        gen_config = Flux2GenerationConfig(
            prompt=prompt,
            height=request.height,
            width=request.width,
            num_steps=num_steps,
            guidance=guidance,
            seed=request.seed,
            reference_images=ref_images,
            match_image_size=match_image_size,
            block_offload=request.block_offload,
            offload_between_stages=offload_between_stages,
            loras=[s for s in request.loras if not s.startswith(":")] or None if request.loras else None,
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

        # Log compile-aware resolution info
        if getattr(config, "flux2_compile", False):
            latent_tokens = (request.width * request.height) // 256
            compile_dynamic = getattr(config, "flux2_compile_dynamic", False)
            if compile_dynamic:
                logger.info(
                    f"[FLUX.2] compile=true (dynamic shapes), {request.width}x{request.height} "
                    f"({latent_tokens} latent tokens) -- "
                    f"resolution changes do not trigger recompilation"
                )
            else:
                logger.info(
                    f"[FLUX.2] compile=true, {request.width}x{request.height} "
                    f"({latent_tokens} latent tokens) -- "
                    f"first generation at this resolution triggers ~90s warmup"
                )
        if model_path:
            logger.info(f"[FLUX.2] Using model path: {model_path}")

        # Verify loaded model matches request; reload if mismatched
        pipeline_dict = _ensure_correct_model(
            request.model_name, manager, requested_loras=request.loras
        )

        persistent_encoder = None
        persistent_transformer = None
        persistent_vae = None
        if pipeline_dict is not None:
            loaded_name = pipeline_dict.get("model_name", "unknown")
            logger.info(
                f"[FLUX.2] Using persistent models (loaded={loaded_name}, "
                f"requested={request.model_name})"
            )
            persistent_encoder = pipeline_dict.get("encoder")
            persistent_transformer = pipeline_dict.get("transformer")
            persistent_vae = pipeline_dict.get("vae")

        # Run in executor to not block event loop
        loop = asyncio.get_event_loop()

        @torch.no_grad()
        def _run_generate():
            return generate_image(
                gen_config,
                model_name=request.model_name,
                encoder=persistent_encoder,
                transformer=persistent_transformer,
                vae=persistent_vae,
                model_path=model_path,
                vae_path=vae_path,
            )

        image = await loop.run_in_executor(None, _run_generate)

        gen_time = time.time() - start_time
        logger.info(f"[FLUX.2] Generation complete in {gen_time:.1f}s")

        # Return standardized JSON response (same format as Z-Image)
        return create_image_response(
            image=image,
            pipeline_id="flux2",
            seed=gen_config.seed,
            generation_time=gen_time,
            warnings=warnings,
        )

    except RuntimeError as e:
        if "LoRA mismatch" in str(e):
            logger.warning(f"[FLUX.2] {e}")
            raise HTTPException(status_code=409, detail=str(e))
        logger.error(f"[FLUX.2] Generation failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    except (AttributeError, TypeError) as e:
        # Guard against mid-request model unload: if the pipeline
        # becomes None while we hold stale references, operations on
        # partially-freed models raise AttributeError/TypeError.
        if not manager.is_loaded("flux2"):
            logger.warning(
                "[FLUX.2] Model was unloaded during generation. "
                "Returning 503."
            )
            raise HTTPException(
                status_code=503,
                detail="FLUX.2 model was unloaded during generation. Please retry.",
            )
        logger.error(f"[FLUX.2] Generation failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"[FLUX.2] Generation failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# =============================================================================
# Streaming Generation
# =============================================================================


@router.post("/api/flux2/generate/stream")
async def flux2_generate_stream(request: Flux2GenerateRequest, config: ConfigDep, manager: ManagerDep):
    """Generate image using FLUX.2 Klein with SSE progress streaming.

    Returns Server-Sent Events with progress updates during generation,
    allowing the frontend to show step-by-step progress.

    Events:
    - {"type": "status", "message": "..."} - Status updates
    - {"type": "progress", "step": N, "total_steps": M} - Step progress
    - {"type": "complete", ...} - Final result with image data
    - {"type": "error", "message": "..."} - Error occurred
    """
    from typing import AsyncIterator

    async def generate_with_progress() -> AsyncIterator[str]:
        """Async generator for SSE events."""
        try:
            # Auto-load on first request (like LTX-2 pattern)
            if not manager.is_loaded("flux2"):
                yield f"data: {json.dumps({'type': 'status', 'message': 'Loading FLUX.2 models (first request, will be cached)...'})}\n\n"
                await asyncio.get_event_loop().run_in_executor(None, lambda: manager.load("flux2"))

            # Verify loaded model matches request; reload if mismatched
            _ensure_correct_model(request.model_name, manager, requested_loras=request.loras)

            # Import the FLUX.2 generation pipeline
            from llm_dit.pipelines.flux2_generate import (
                Flux2GenerationConfig,
                generate_image_with_progress,
            )
            num_steps, guidance, stream_warnings = _resolve_flux2_params(request, config)

            # Prompt upsampling (optional, requires heylookitsanllm API)
            prompt = request.prompt
            if request.upsample_prompt:
                yield f"data: {json.dumps({'type': 'status', 'message': 'Upsampling prompt...'})}\n\n"
                prompt, upsample_warning = _upsample_prompt(
                    config, prompt, has_reference_images=bool(request.reference_images),
                )
                if upsample_warning:
                    stream_warnings.append(upsample_warning)

            yield f"data: {json.dumps({'type': 'status', 'message': 'Processing request...'})}\n\n"

            # Emit warnings as SSE events
            for w in stream_warnings:
                yield f"data: {json.dumps({'type': 'warning', 'message': w})}\n\n"

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
            if not manager.is_loaded("flux2"):
                offload_between_stages = True

            # Create generation config
            gen_config = Flux2GenerationConfig(
                prompt=prompt,
                height=request.height,
                width=request.width,
                num_steps=num_steps,
                guidance=guidance,
                seed=request.seed,
                reference_images=ref_images,
                match_image_size=match_image_size,
                block_offload=request.block_offload,
                offload_between_stages=offload_between_stages,
                loras=[s for s in request.loras if not s.startswith(":")] or None if request.loras else None,
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

            @torch.no_grad()
            def run_generation():
                """Run generation and put progress events in queue."""
                def callback(step: int, total: int, stage: str = ""):
                    # Use call_soon_threadsafe to safely put in queue from thread
                    loop.call_soon_threadsafe(
                        progress_queue.put_nowait,
                        {"step": step, "total": total, "stage": stage}
                    )

                # Use persistent models (already verified by _ensure_correct_model above)
                p_encoder = None
                p_transformer = None
                p_vae = None
                flux2_dict = manager.get_pipeline("flux2")
                if isinstance(flux2_dict, dict):
                    loaded_name = flux2_dict.get("model_name", "unknown")
                    logger.info(
                        f"[FLUX.2] Stream: using persistent models "
                        f"(loaded={loaded_name}, requested={request.model_name})"
                    )
                    p_encoder = flux2_dict.get("encoder")
                    p_transformer = flux2_dict.get("transformer")
                    p_vae = flux2_dict.get("vae")

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

            # Yield final result (include warnings if any)
            complete_data = {
                'type': 'complete',
                'urls': [data_url],
                'url': data_url,
                'seed': gen_config.seed,
                'generation_time': gen_time,
            }
            if stream_warnings:
                complete_data['warnings'] = stream_warnings
            yield f"data: {json.dumps(complete_data)}\n\n"

        except RuntimeError as e:
            if "LoRA mismatch" in str(e):
                logger.warning(f"[FLUX.2] {e}")
                yield f"data: {json.dumps({'type': 'error', 'message': str(e), 'code': 409})}\n\n"
            else:
                logger.error(f"[FLUX.2] Stream generation failed: {e}")
                traceback.print_exc()
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        except (AttributeError, TypeError) as e:
            if not manager.is_loaded("flux2"):
                logger.warning(
                    "[FLUX.2] Model was unloaded during stream generation. "
                    "Returning error event."
                )
                yield f"data: {json.dumps({'type': 'error', 'message': 'FLUX.2 model was unloaded during generation. Please retry.'})}\n\n"
            else:
                logger.error(f"[FLUX.2] Stream generation failed: {e}")
                traceback.print_exc()
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        except Exception as e:
            logger.error(f"[FLUX.2] Stream generation failed: {e}")
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return StreamingResponse(
        generate_with_progress(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
