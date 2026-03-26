"""Z-Image core generation endpoints: encode, generate, img2img, format-prompt, templates, rewriters, DyPE."""

import asyncio
import base64
import binascii
import gc
import io
import json
import logging
import re
import time
import traceback
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional, Union

import torch
from fastapi import APIRouter, HTTPException
from PIL import Image
from starlette.responses import StreamingResponse

from web.dependencies import ConfigDep, ManagerDep
from web.schemas import (
    DyPEConfigResponse,
    DyPEStatusResponse,
    EncodeRequest,
    EncodeResult,
    FormatPromptResult,
    GenerateRequest,
    ImageGenerationResult,
    Img2ImgRequest,
    RewriteRequest,
    RewriteResult,
    RewriterListResponse,
    SaveEmbeddingsResult,
    TemplateListResponse,
)
from web.utils import create_image_response

import web.server as srv

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Z-Image Variant-Aware Defaults Helper
# =============================================================================


def resolve_zimage_variant(
    request: Union[GenerateRequest, Img2ImgRequest], config
) -> str:
    """Resolve the Z-Image variant from request or config.

    Priority: request.variant (if sent) > config.zimage_variant > "turbo".
    """
    if config is None:
        return "turbo"
    # Use request variant if client sent it explicitly
    if hasattr(request, "variant") and "variant" in request.model_fields_set and request.variant:
        return request.variant
    variant = config.zimage_variant
    if variant == "auto":
        return "turbo"
    return variant


def apply_zimage_variant_defaults(
    request: Union[GenerateRequest, Img2ImgRequest], config, variant: str | None = None
) -> None:
    """Apply Z-Image variant-aware defaults to request in-place.

    Uses model_fields_set to detect whether the client sent a value explicitly.
    Only modifies values the client did NOT send.

    Args:
        request: GenerateRequest or Img2ImgRequest to modify in-place
        config: RuntimeConfig instance
        variant: Resolved variant (if None, resolves from request/config)
    """
    if config is None:
        return

    if variant is None:
        variant = resolve_zimage_variant(request, config)

    from llm_dit.models.zimage.constants import get_variant_defaults

    variant_defaults = get_variant_defaults(variant)

    if hasattr(request, "steps") and "steps" not in request.model_fields_set:
        request.steps = variant_defaults["num_inference_steps"]
        logger.debug(f"Applied variant default: steps={request.steps}")

    if hasattr(request, "guidance_scale") and "guidance_scale" not in request.model_fields_set:
        request.guidance_scale = variant_defaults["guidance_scale"]
        logger.debug(f"Applied variant default: guidance_scale={request.guidance_scale}")

    if hasattr(request, "shift") and "shift" not in request.model_fields_set:
        request.shift = variant_defaults["shift"]
        logger.debug(f"Applied variant default: shift={request.shift}")


def _ensure_correct_zimage_variant(variant: str, manager, config) -> None:
    """Reload Z-Image pipeline if the loaded variant doesn't match the request.

    Updates config.zimage.model_path and config.zimage.variant before reloading.
    """
    if not manager.is_loaded("zimage"):
        return

    current_variant = config.zimage_variant
    if current_variant == variant:
        return

    new_path = config.zimage.resolve_model_path(variant)
    if not new_path:
        raise ValueError(
            f"No model path configured for Z-Image variant '{variant}'. "
            f"Set [zimage].{variant}_model_path in config.toml"
        )

    logger.info(
        f"[Z-Image] Variant mismatch: loaded='{current_variant}', "
        f"requested='{variant}'. Reloading from {new_path}..."
    )

    config.zimage.model_path = new_path
    config.zimage.variant = variant
    manager.reload_zimage()
    logger.info(f"[Z-Image] Reload complete: variant='{variant}'")


# =============================================================================
# Z-Image Encoder Helper
# =============================================================================


def _get_zimage_encoder(manager):
    """Get Z-Image encoder from standalone encoder or Z-Image pipeline.

    Returns None for Qwen-Image pipelines (they don't have a separate encoder).
    """
    if manager.encoder is not None:
        return manager.encoder
    pipeline = manager.get_pipeline("zimage")
    if pipeline is not None and hasattr(pipeline, "encoder"):
        return pipeline.encoder
    return None


def _ensure_zimage_loaded(manager) -> None:
    """Load Z-Image pipeline on-demand via ModelManager."""
    if manager.get_pipeline("zimage") is not None:
        return
    manager.load("zimage")


def _ensure_zimage_lora(
    manager,
    requested_loras: list[str] | None = None,
) -> None:
    """Check LoRA mismatch on Z-Image transformer; reload if needed.

    When the requested LoRAs differ from what's already fused into the
    transformer, we must reload the entire pipeline (LoRA fusion is
    irreversible -- weights are permanently added to the model parameters).
    """
    if not manager.is_loaded("zimage"):
        return

    zimage = manager.get_pipeline("zimage")
    if zimage is None:
        return

    # Filter empty-path specs from UI (e.g., ":0.80" from unselected slots)
    if requested_loras:
        requested_loras = [s for s in requested_loras if not s.startswith(":")]
    if not requested_loras:
        return

    from llm_dit.utils.lora import get_fused_state, parse_lora_spec

    transformer = zimage.transformer
    if transformer is None:
        return

    fused_state = get_fused_state(transformer)
    requested_specs = [parse_lora_spec(s) for s in requested_loras]

    if not fused_state.is_empty and not fused_state.matches(requested_specs):
        logger.info(
            f"[Z-Image] LoRA mismatch: fused=[{fused_state.summary()}], "
            f"requested={requested_loras}. Reloading..."
        )
        # Drop local ref so _unload_zimage() can free GPU memory
        del transformer
        result = manager.reload_zimage()
        logger.info(f"[Z-Image] Reload complete ({result.load_time:.1f}s)")


def _apply_zimage_loras(
    zimage,
    requested_loras: list[str] | None,
) -> None:
    """Fuse requested LoRAs into the Z-Image transformer if not already fused."""
    if not requested_loras:
        return

    clean_loras = [s for s in requested_loras if not s.startswith(":")]
    if not clean_loras:
        return

    transformer = zimage.transformer
    if transformer is None:
        return

    from llm_dit.utils.lora import get_fused_state, load_lora, parse_lora_spec

    fused_state = get_fused_state(transformer)
    requested_specs = [parse_lora_spec(s) for s in clean_loras]

    if fused_state.matches(requested_specs):
        return  # Already fused with exactly these LoRAs

    for path, scale in requested_specs:
        if not fused_state.is_fused(path, scale):
            logger.info(f"[Z-Image] Fusing LoRA: {path} (scale={scale})")
            load_lora(transformer, path, scale=scale)


# =============================================================================
# DyPE Endpoints
# =============================================================================


@router.get("/api/dype/config", response_model=DyPEConfigResponse)
async def dype_config(config: ConfigDep) -> DyPEConfigResponse:
    """Get DyPE configuration defaults from server config.

    Returns default DyPE settings for high-resolution generation.
    """
    if config is None:
        return DyPEConfigResponse()

    # Get DyPE config from runtime config if available
    dype = getattr(config, "dype", None)
    if dype is not None:
        return DyPEConfigResponse(
            enabled=dype.enabled,
            method=dype.method,
            dype_scale=dype.dype_scale,
            dype_exponent=dype.dype_exponent,
            dype_start_sigma=dype.dype_start_sigma,
            base_shift=dype.base_shift,
            max_shift=dype.max_shift,
            base_resolution=dype.base_resolution,
            anisotropic=dype.anisotropic,
        )

    return DyPEConfigResponse()


@router.get("/api/dype/status", response_model=DyPEStatusResponse)
async def dype_status(manager: ManagerDep):
    """Get DyPE feature status and recommendations.

    Returns whether DyPE is recommended for the current pipeline
    and suggested settings based on target resolution.
    """
    pipeline_supports_dype = manager.get_pipeline("zimage") is not None

    return {
        "available": pipeline_supports_dype,
        "supported_methods": ["vision_yarn", "yarn", "ntk"],
        "recommended_for_resolutions": {
            "2K": {"method": "vision_yarn", "multipass": "single"},
            "4K": {"method": "vision_yarn", "multipass": "twopass"},
            "higher": {"method": "vision_yarn", "multipass": "threepass"},
        },
        "notes": [
            "Two-pass is recommended for 4K+ resolutions for better stability",
            "Vision YaRN uses dual-mask frequency blending for best quality",
            "Lower pass2_strength (0.3-0.5) preserves more detail from first pass",
        ],
    }


# =============================================================================
# Z-Image Core Generation Endpoints
# =============================================================================


@router.post("/api/encode", response_model=EncodeResult)
async def encode(request: EncodeRequest, config: ConfigDep, manager: ManagerDep) -> EncodeResult:
    """Encode a prompt to embeddings (for distributed inference)."""
    # Use encoder from pipeline or standalone encoder
    enc = _get_zimage_encoder(manager)
    if enc is None:
        raise HTTPException(status_code=503, detail="Encoder not loaded")

    try:
        start = time.time()
        output = enc.encode(
            request.prompt,
            template=request.template,
            system_prompt=request.system_prompt,
            thinking_content=request.thinking_content,
            assistant_content=request.assistant_content,
            force_think_block=request.force_think_block,
            remove_quotes=request.strip_quotes,
        )
        encode_time = time.time() - start

        embeddings = output.embeddings[0]
        token_count = output.token_counts[0] if output.token_counts else embeddings.shape[0]

        # Get formatted prompt if available
        formatted_prompt = None
        if output.formatted_prompts:
            formatted_prompt = output.formatted_prompts[0]
            if config.logging.log_prompts:
                logger.info(f"Formatted prompt ({len(formatted_prompt)} chars, {token_count} tokens):")
                logger.info(f"---BEGIN FORMATTED PROMPT---")
                logger.info(formatted_prompt)
                logger.info(f"---END FORMATTED PROMPT---")

        return EncodeResult(
            shape=list(embeddings.shape),
            dtype=str(embeddings.dtype),
            encode_time=encode_time,
            token_count=token_count,
            prompt=request.prompt,
            formatted_prompt=formatted_prompt,
        )
    except Exception as e:
        logger.error(f"Encoding failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/generate", response_model=ImageGenerationResult)
async def generate(request: GenerateRequest, config: ConfigDep, manager: ManagerDep) -> ImageGenerationResult:
    """Generate an image from a prompt."""
    if srv.encoder_only_mode:
        raise HTTPException(
            status_code=400, detail="Server running in encoder-only mode. Use /api/encode instead."
        )

    # Resolve variant and apply defaults
    variant = resolve_zimage_variant(request, config)
    apply_zimage_variant_defaults(request, config, variant)

    # Load Z-Image pipeline on-demand if not already loaded
    zimage = manager.get_pipeline("zimage")
    if zimage is None:
        # Set correct model path before first load
        if variant != config.zimage_variant:
            new_path = config.zimage.resolve_model_path(variant)
            if new_path:
                config.zimage.model_path = new_path
                config.zimage.variant = variant
        try:
            _ensure_zimage_loaded(manager)
            zimage = manager.get_pipeline("zimage")
        except Exception as e:
            logger.error(f"[Z-Image] Failed to load pipeline on-demand: {e}")
            raise HTTPException(
                status_code=503,
                detail=f"Z-Image pipeline failed to load: {str(e)}",
            )
    else:
        # Reload if variant changed
        _ensure_correct_zimage_variant(variant, manager, config)

    # LoRA: check mismatch and reload if needed, then fuse
    _ensure_zimage_lora(manager, request.loras)
    zimage = manager.get_pipeline("zimage")  # Re-fetch after potential reload
    _apply_zimage_loras(zimage, request.loras)

    try:
        logger.info("=" * 60)
        logger.info("GENERATION REQUEST")
        logger.info("=" * 60)
        if config.logging.log_prompts:
            logger.info(f"  Prompt: {request.prompt[:80]}...")
            if request.negative_prompt:
                logger.info(f"  Negative: {request.negative_prompt[:80]}...")
        if config.logging.log_generation_params:
            logger.info(f"  Size: {request.width}x{request.height}")
            logger.info(f"  Steps: {request.steps}")
            logger.info(f"  Seed: {request.seed}")
            logger.info(f"  Template: {request.template}")
            logger.info(f"  Force think block: {request.force_think_block}")
            logger.info(f"  Guidance: {request.guidance_scale}")
            logger.info(f"  Long prompt mode: {request.long_prompt_mode}")
            logger.info(f"  Hidden layer: {request.hidden_layer}")
            if request.layer_weights:
                logger.info(f"  Layer weights: {request.layer_weights}")
        logger.info("-" * 60)
        logger.info("Pipeline state:")
        logger.info(f"  pipeline.device: {zimage.device}")
        logger.info(f"  pipeline.dtype: {zimage.dtype}")
        logger.info(
            f"  pipeline.encoder: {type(zimage.encoder).__name__ if zimage.encoder is not None else 'None'}"
        )
        logger.info(f"  pipeline.transformer: {zimage.transformer is not None}")
        logger.info(f"  pipeline.vae: {zimage.vae is not None}")
        if zimage.encoder is not None:
            backend = getattr(zimage.encoder, "backend", None)
            logger.info(f"  encoder.backend: {type(backend).__name__ if backend else 'None'}")
        model_path = config.zimage_model_path or config.model_path
        logger.info(f"  variant: {config.zimage_variant}")
        logger.info(f"  model_path: {model_path}")
        logger.info("-" * 60)

        # Set up generator for reproducibility
        generator = None
        if request.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(request.seed)

        # Negative prompt: only use for base variant (turbo has CFG=0, so it has no effect)
        negative_prompt_to_use = None
        if config.zimage_variant == "base":
            negative_prompt_to_use = request.negative_prompt

        start = time.time()

        # SLG config: "UI always wins" - don't fall back to runtime_config
        # None means disabled, not "use config default"
        slg_scale = request.slg_scale if request.slg_scale is not None else 0.0
        slg_layers = request.slg_layers  # None is valid (means SLG disabled)
        slg_start = request.slg_start if request.slg_start is not None else 0.01
        slg_stop = request.slg_stop if request.slg_stop is not None else 0.2

        # FMTT config: "UI always wins" - don't fall back to runtime_config
        # Only use fmtt_scale if fmtt_enabled is True
        fmtt_scale = (
            request.fmtt_scale if request.fmtt_enabled and request.fmtt_scale is not None else 0.0
        )
        fmtt_start = request.fmtt_start if request.fmtt_start is not None else 0.0
        fmtt_stop = request.fmtt_stop if request.fmtt_stop is not None else 0.5
        fmtt_normalize = request.fmtt_normalize if request.fmtt_normalize is not None else "unit"
        fmtt_decode_scale = (
            request.fmtt_decode_scale if request.fmtt_decode_scale is not None else 0.5
        )
        fmtt_siglip_model = (
            request.fmtt_siglip_model
            if request.fmtt_siglip_model is not None
            else "google/siglip2-giant-opt-patch16-384"
        )
        fmtt_siglip_device = (
            request.fmtt_siglip_device if request.fmtt_siglip_device is not None else "cuda"
        )

        # Convert DyPE request to DyPEConfig if provided
        dype_config = None
        if request.dype is not None and request.dype.enabled:
            from llm_dit.utils.dype import DyPEConfig

            dype_config = DyPEConfig(
                enabled=request.dype.enabled,
                method=request.dype.method,
                dype_scale=request.dype.dype_scale,
                dype_exponent=request.dype.dype_exponent,
                base_shift=request.dype.base_shift,
                max_shift=request.dype.max_shift,
                base_resolution=1024,  # Z-Image base
                multipass=request.dype.multipass,
                pass2_strength=request.dype.pass2_strength,
                pass3_strength=request.dype.pass3_strength,
                frequency_modulation=request.dype.frequency_modulation,
            )

        # Generate image
        logger.info(
            f"Calling pipeline() with long_prompt_mode={request.long_prompt_mode}, hidden_layer={request.hidden_layer}..."
        )
        if negative_prompt_to_use and config.logging.log_prompts:
            neg_display = negative_prompt_to_use[:60] + "..." if len(negative_prompt_to_use) > 60 else negative_prompt_to_use
            logger.info(f"  Negative prompt: {neg_display}")
        if slg_scale > 0 and slg_layers:
            logger.info(
                f"  SLG: scale={slg_scale}, layers={slg_layers}, range=[{slg_start:.0%}, {slg_stop:.0%}]"
            )
        if fmtt_scale > 0:
            logger.info(f"  FMTT: scale={fmtt_scale}, range=[{fmtt_start:.0%}, {fmtt_stop:.0%}]")
        if dype_config is not None:
            logger.info(
                f"  DyPE: method={dype_config.method}, scale={dype_config.dype_scale}, exponent={dype_config.dype_exponent}"
            )

        # Check for multipass generation (for high-res with DyPE)
        multipass_mode = dype_config.multipass if dype_config else "single"
        pass2_strength = dype_config.pass2_strength if dype_config else 0.5
        pass3_strength = dype_config.pass3_strength if dype_config else 0.4

        if multipass_mode != "single" and dype_config and dype_config.enabled:
            # Build passes configuration based on multipass mode
            if multipass_mode == "twopass":
                passes = [
                    {"scale": 0.5, "steps": request.steps},
                    {"scale": 1.0, "steps": request.steps, "strength": pass2_strength},
                ]
            elif multipass_mode == "threepass":
                passes = [
                    {"scale": 0.25, "steps": request.steps},
                    {"scale": 0.5, "steps": request.steps, "strength": pass2_strength},
                    {"scale": 1.0, "steps": request.steps, "strength": pass3_strength},
                ]
            else:
                passes = None  # Use default

            logger.info(
                f"  Multipass: {multipass_mode}, pass2_strength={pass2_strength}, pass3_strength={pass3_strength}"
            )
            image = zimage.generate_multipass(
                request.prompt,
                negative_prompt=negative_prompt_to_use,
                final_width=request.width,
                final_height=request.height,
                passes=passes,
                generator=generator,
                template=request.template,
                system_prompt=request.system_prompt,
                thinking_content=request.thinking_content,
                assistant_content=request.assistant_content,
                force_think_block=request.force_think_block,
                remove_quotes=request.strip_quotes,
                long_prompt_mode=request.long_prompt_mode,
                hidden_layer=request.hidden_layer,
                layer_weights=request.layer_weights,
                # Pass through additional kwargs for each pass
                guidance_scale=request.guidance_scale,
                cfg_normalization=request.cfg_normalization,
                cfg_truncation=request.cfg_truncation,
                shift=None if request.dynamic_shift else request.shift,
                d_noise=request.d_noise,
                skip_layer_guidance_scale=slg_scale,
                skip_layer_indices=slg_layers,
                skip_layer_start=slg_start,
                skip_layer_stop=slg_stop,
                fmtt_guidance_scale=fmtt_scale,
                fmtt_guidance_start=fmtt_start,
                fmtt_guidance_stop=fmtt_stop,
                fmtt_normalize_mode=fmtt_normalize,
                fmtt_decode_scale=fmtt_decode_scale,
                fmtt_siglip_model=fmtt_siglip_model,
                fmtt_siglip_device=fmtt_siglip_device,
                dype_config=dype_config,
                fbcache=request.fbcache,
                fbcache_threshold=request.fbcache_threshold,
                fbcache_log=request.fbcache_log,
            )
        else:
            # Progress callback for console logging
            def progress_callback(step: int, total: int, latents: torch.Tensor) -> None:
                logger.info(f"  Step {step + 1}/{total}")

            # Single pass generation
            image = zimage(
                request.prompt,
                negative_prompt=negative_prompt_to_use,
                height=request.height,
                width=request.width,
                num_inference_steps=request.steps,
                guidance_scale=request.guidance_scale,
                cfg_normalization=request.cfg_normalization,
                cfg_truncation=request.cfg_truncation,
                shift=None if request.dynamic_shift else request.shift,
                d_noise=request.d_noise,
                generator=generator,
                template=request.template,
                system_prompt=request.system_prompt,
                thinking_content=request.thinking_content,
                assistant_content=request.assistant_content,
                force_think_block=request.force_think_block,
                remove_quotes=request.strip_quotes,
                long_prompt_mode=request.long_prompt_mode,
                hidden_layer=request.hidden_layer,
                layer_weights=request.layer_weights,
                skip_layer_guidance_scale=slg_scale,
                skip_layer_indices=slg_layers,
                skip_layer_start=slg_start,
                skip_layer_stop=slg_stop,
                fmtt_guidance_scale=fmtt_scale,
                fmtt_guidance_start=fmtt_start,
                fmtt_guidance_stop=fmtt_stop,
                fmtt_normalize_mode=fmtt_normalize,
                fmtt_decode_scale=fmtt_decode_scale,
                fmtt_siglip_model=fmtt_siglip_model,
                fmtt_siglip_device=fmtt_siglip_device,
                dype_config=dype_config,
                fbcache=request.fbcache,
                fbcache_threshold=request.fbcache_threshold,
                fbcache_log=request.fbcache_log,
                callback=progress_callback,
            )

        gen_time = time.time() - start
        logger.info(f"Generated in {gen_time:.1f}s")
        logger.info("=" * 60)

        # Convert to base64 for response
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        img_b64 = base64.b64encode(img_bytes.getvalue()).decode("ascii")

        # Get formatted prompt for history
        formatted_prompt = None
        enc = _get_zimage_encoder(manager)
        if enc:
            try:
                from llm_dit.conversation import Conversation

                conv = enc._build_conversation(
                    prompt=request.prompt,
                    template=request.template,
                    system_prompt=request.system_prompt,
                    thinking_content=request.thinking_content,
                    assistant_content=request.assistant_content,
                    force_think_block=request.force_think_block,
                    remove_quotes=request.strip_quotes,
                )
                formatted_prompt = enc.formatter.format(conv)
            except Exception as e:
                logger.warning(f"Failed to get formatted prompt: {e}")

        # Store in history
        history_entry = {
            "id": len(srv.generation_history),
            "timestamp": time.time(),
            "model_type": "zimage",  # Z-Image text-to-image
            "prompt": request.prompt,
            "system_prompt": request.system_prompt,
            "thinking_content": request.thinking_content,
            "assistant_content": request.assistant_content,
            "force_think_block": request.force_think_block,
            "strip_quotes": request.strip_quotes,
            "width": request.width,
            "height": request.height,
            "steps": request.steps,
            "seed": request.seed,
            "template": request.template,
            "guidance_scale": request.guidance_scale,
            "shift": request.shift,
            "long_prompt_mode": request.long_prompt_mode,
            "hidden_layer": request.hidden_layer,
            "layer_weights": request.layer_weights,
            "cfg_normalization": request.cfg_normalization,
            "cfg_truncation": request.cfg_truncation,
            "gen_time": gen_time,

            "formatted_prompt": formatted_prompt,
        }
        srv.generation_history.insert(0, history_entry)
        # Trim history
        if len(srv.generation_history) > srv.MAX_HISTORY:
            srv.generation_history.pop()

        # Return standardized JSON response (shared format with FLUX.2, etc.)
        return create_image_response(
            pipeline_id="zimage",
            seed=request.seed,
            generation_time=gen_time,
            history_id=history_entry["id"],
            img_b64=img_b64,  # Reuse already-computed base64
        )

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@router.post("/api/generate/stream")
async def generate_stream(request: GenerateRequest, config: ConfigDep, manager: ManagerDep):
    """Generate an image with SSE progress streaming.

    Returns Server-Sent Events with progress updates during generation,
    allowing the frontend to show step-by-step progress.

    Events:
    - {"type": "status", "message": "..."} - Status updates
    - {"type": "progress", "step": N, "total_steps": M, ...} - Step progress
    - {"type": "complete", ...} - Final result with image data
    - {"type": "error", "message": "..."} - Error occurred
    """
    # Resolve variant and apply defaults
    variant = resolve_zimage_variant(request, config)
    apply_zimage_variant_defaults(request, config, variant)

    if srv.encoder_only_mode:
        raise HTTPException(
            status_code=400, detail="Server running in encoder-only mode. Use /api/encode instead."
        )

    # Load Z-Image pipeline on-demand if not already loaded
    zimage = manager.get_pipeline("zimage")
    if zimage is None:
        if variant != config.zimage_variant:
            new_path = config.zimage.resolve_model_path(variant)
            if new_path:
                config.zimage.model_path = new_path
                config.zimage.variant = variant
        try:
            _ensure_zimage_loaded(manager)
            zimage = manager.get_pipeline("zimage")
        except Exception as e:
            logger.error(f"[Z-Image] Failed to load pipeline on-demand: {e}")
            raise HTTPException(
                status_code=503,
                detail=f"Z-Image pipeline failed to load: {str(e)}",
            )
    else:
        _ensure_correct_zimage_variant(variant, manager, config)

    # LoRA: check mismatch and reload if needed, then fuse
    _ensure_zimage_lora(manager, request.loras)
    zimage = manager.get_pipeline("zimage")  # Re-fetch after potential reload
    _apply_zimage_loras(zimage, request.loras)

    async def generate_with_progress() -> AsyncIterator[str]:
        """Async generator for SSE events."""
        try:
            # Initial status
            yield f"data: {json.dumps({'type': 'status', 'message': 'Starting generation...'})}\n\n"

            # Set up generator for reproducibility
            generator = None
            actual_seed = request.seed if request.seed is not None else int(time.time() * 1000) % (2**32)
            generator = torch.Generator()
            generator.manual_seed(actual_seed)

            # Negative prompt: only use for base variant
            negative_prompt_to_use = None
            if config.zimage_variant == "base":
                negative_prompt_to_use = request.negative_prompt

            # SLG config
            slg_scale = request.slg_scale if request.slg_scale is not None else 0.0
            slg_layers = request.slg_layers
            slg_start = request.slg_start if request.slg_start is not None else 0.01
            slg_stop = request.slg_stop if request.slg_stop is not None else 0.2

            # FMTT config
            fmtt_scale = (
                request.fmtt_scale if request.fmtt_enabled and request.fmtt_scale is not None else 0.0
            )
            fmtt_start = request.fmtt_start if request.fmtt_start is not None else 0.0
            fmtt_stop = request.fmtt_stop if request.fmtt_stop is not None else 0.5
            fmtt_normalize = request.fmtt_normalize if request.fmtt_normalize is not None else "unit"
            fmtt_decode_scale = request.fmtt_decode_scale if request.fmtt_decode_scale is not None else 0.5
            fmtt_siglip_model = request.fmtt_siglip_model or "google/siglip2-giant-opt-patch16-384"
            fmtt_siglip_device = request.fmtt_siglip_device or "cuda"

            # DyPE config
            dype_config = None
            if request.dype is not None and request.dype.enabled:
                from llm_dit.utils.dype import DyPEConfig
                dype_config = DyPEConfig(
                    enabled=request.dype.enabled,
                    method=request.dype.method,
                    dype_scale=request.dype.dype_scale,
                    dype_exponent=request.dype.dype_exponent,
                    base_shift=request.dype.base_shift,
                    max_shift=request.dype.max_shift,
                    base_resolution=1024,
                    multipass=request.dype.multipass,
                    pass2_strength=request.dype.pass2_strength,
                    pass3_strength=request.dype.pass3_strength,
                    frequency_modulation=request.dype.frequency_modulation,
                )

            # Progress tracking state
            progress_state = {"step": 0, "total": request.steps, "start_time": time.time()}

            def progress_callback(step: int, total: int, latents: torch.Tensor) -> None:
                """Update progress state (can't yield from here, but state is shared)."""
                progress_state["step"] = step + 1
                progress_state["total"] = total

            logger.info("=" * 60)
            logger.info("STREAMING GENERATION REQUEST")
            logger.info("=" * 60)
            if config.logging.log_prompts:
                logger.info(f"  Prompt: {request.prompt[:80]}...")
                if negative_prompt_to_use:
                    neg_display = negative_prompt_to_use[:60] + "..." if len(negative_prompt_to_use) > 60 else negative_prompt_to_use
                    logger.info(f"  Negative: {neg_display}")
            if config.logging.log_generation_params:
                logger.info(f"  Size: {request.width}x{request.height}")
                logger.info(f"  Steps: {request.steps}")
                logger.info(f"  Seed: {actual_seed}")

            # Run generation in thread pool (blocking operation)
            loop = asyncio.get_event_loop()

            @torch.no_grad()
            def do_generate():
                return zimage(
                    request.prompt,
                    negative_prompt=negative_prompt_to_use,
                    height=request.height,
                    width=request.width,
                    num_inference_steps=request.steps,
                    guidance_scale=request.guidance_scale,
                    cfg_normalization=request.cfg_normalization,
                    cfg_truncation=request.cfg_truncation,
                    shift=None if request.dynamic_shift else request.shift,
                    d_noise=request.d_noise,
                    generator=generator,
                    template=request.template,
                    system_prompt=request.system_prompt,
                    thinking_content=request.thinking_content,
                    assistant_content=request.assistant_content,
                    force_think_block=request.force_think_block,
                    remove_quotes=request.strip_quotes,
                    long_prompt_mode=request.long_prompt_mode,
                    hidden_layer=request.hidden_layer,
                    layer_weights=request.layer_weights,
                    skip_layer_guidance_scale=slg_scale,
                    skip_layer_indices=slg_layers,
                    skip_layer_start=slg_start,
                    skip_layer_stop=slg_stop,
                    fmtt_guidance_scale=fmtt_scale,
                    fmtt_guidance_start=fmtt_start,
                    fmtt_guidance_stop=fmtt_stop,
                    fmtt_normalize_mode=fmtt_normalize,
                    fmtt_decode_scale=fmtt_decode_scale,
                    fmtt_siglip_model=fmtt_siglip_model,
                    fmtt_siglip_device=fmtt_siglip_device,
                    dype_config=dype_config,
                    fbcache=request.fbcache,
                    fbcache_threshold=request.fbcache_threshold,
                    fbcache_log=request.fbcache_log,
                    callback=progress_callback,
                )

            # Start generation task
            gen_task = loop.run_in_executor(None, do_generate)

            # Poll progress while generating
            last_step = -1
            while not gen_task.done():
                await asyncio.sleep(0.1)  # Poll every 100ms
                step = progress_state["step"]
                total = progress_state["total"]

                if step > last_step and step <= total:
                    elapsed = time.time() - progress_state["start_time"]
                    # Calculate ETA
                    if step > 0:
                        its = step / elapsed  # iterations per second
                        remaining = (total - step) / its if its > 0 else 0
                    else:
                        its = 0
                        remaining = 0

                    yield f"data: {json.dumps({'type': 'progress', 'step': step, 'total_steps': total, 'elapsed': round(elapsed, 1), 'estimated_remaining_ms': int(remaining * 1000), 'message': f'Step {step}/{total}'})}\n\n"
                    last_step = step

            # Get result
            image = await gen_task
            gen_time = time.time() - progress_state["start_time"]

            logger.info(f"[Stream] Generated in {gen_time:.1f}s")
            logger.info("=" * 60)

            # Convert to base64
            img_bytes = io.BytesIO()
            image.save(img_bytes, format="PNG")
            img_b64 = base64.b64encode(img_bytes.getvalue()).decode("ascii")
            data_url = f"data:image/png;base64,{img_b64}"

            # Store in history
            history_entry = {
                "id": len(srv.generation_history),
                "timestamp": time.time(),
                "model_type": "zimage",
                "prompt": request.prompt,
                "width": request.width,
                "height": request.height,
                "steps": request.steps,
                "seed": actual_seed,
                "gen_time": gen_time,

            }
            srv.generation_history.insert(0, history_entry)
            if len(srv.generation_history) > srv.MAX_HISTORY:
                srv.generation_history.pop()

            # Send complete event
            gen_id = f"gen-{int(time.time() * 1000)}"
            complete_event = {
                'type': 'complete',
                'id': gen_id,
                'pipeline_id': 'zimage',
                'output_type': 'image',
                'url': data_url,
                'urls': [data_url],
                'thumbnail_url': data_url,
                'seed': actual_seed,
                'generation_time': gen_time,
            }
            yield f"data: {json.dumps(complete_event)}\n\n"

        except Exception as e:
            logger.error(f"[Stream] Generation failed: {e}")
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
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@router.post("/api/img2img", response_model=ImageGenerationResult)
async def img2img(request: Img2ImgRequest, config: ConfigDep, manager: ManagerDep) -> ImageGenerationResult:
    """Generate an image from an input image with optional differential mask.

    The mask controls per-pixel edit strength:
    - Black (0): Preserve original
    - White (255): Allow full editing
    - Gray: Partial editing
    """
    # Resolve variant and apply defaults
    variant = resolve_zimage_variant(request, config)
    apply_zimage_variant_defaults(request, config, variant)

    if srv.encoder_only_mode:
        raise HTTPException(
            status_code=400, detail="Server running in encoder-only mode. Img2img not available."
        )

    zimage = manager.get_pipeline("zimage")
    if zimage is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")

    _ensure_correct_zimage_variant(variant, manager, config)

    # LoRA: check mismatch and reload if needed, then fuse
    _ensure_zimage_lora(manager, request.loras)
    zimage = manager.get_pipeline("zimage")  # Re-fetch after potential reload
    _apply_zimage_loras(zimage, request.loras)

    try:
        from PIL import UnidentifiedImageError
        PILImage = Image  # Alias for consistency with existing code

        logger.info("=" * 60)
        logger.info("IMG2IMG REQUEST")
        logger.info("=" * 60)
        if config.logging.log_prompts:
            logger.info(f"  Prompt: {request.prompt[:80]}...")
        if config.logging.log_generation_params:
            logger.info(f"  Strength: {request.strength}")
            logger.info(f"  Has mask: {request.mask_image is not None}")
            logger.info(f"  Steps: {request.steps}")
        logger.info(f"  Seed: {request.seed}")
        logger.info("-" * 60)

        # Decode and validate input image from base64
        try:
            # Strip data URL prefix if present
            image_b64 = request.image
            if image_b64.startswith("data:"):
                image_b64 = image_b64.split(",", 1)[1]
            image_data = base64.b64decode(image_b64, validate=True)
            # Size limit: 50MB
            if len(image_data) > 50_000_000:
                raise HTTPException(status_code=413, detail="Image too large (max 50MB)")
            input_image = PILImage.open(io.BytesIO(image_data)).convert("RGB")
        except binascii.Error:
            raise HTTPException(status_code=400, detail="Invalid base64 image data")
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail="Unsupported image format")
        logger.info(f"  Input image size: {input_image.size}")

        # Decode and validate mask if provided
        mask_image = None
        if request.mask_image:
            try:
                mask_b64 = request.mask_image
                if mask_b64.startswith("data:"):
                    mask_b64 = mask_b64.split(",", 1)[1]
                mask_data = base64.b64decode(mask_b64, validate=True)
                if len(mask_data) > 50_000_000:
                    raise HTTPException(status_code=413, detail="Mask image too large (max 50MB)")
                mask_image = PILImage.open(io.BytesIO(mask_data)).convert("L")
            except binascii.Error:
                raise HTTPException(status_code=400, detail="Invalid base64 mask data")
            except UnidentifiedImageError:
                raise HTTPException(status_code=400, detail="Unsupported mask image format")
            logger.info(f"  Mask size: {mask_image.size}")

        # Determine output size
        width = request.width if request.width else input_image.width
        height = request.height if request.height else input_image.height

        # Ensure dimensions are divisible by 16 (VAE constraint)
        width = (width // 16) * 16
        height = (height // 16) * 16

        # Resize input image if needed
        if input_image.size != (width, height):
            input_image = input_image.resize((width, height), PILImage.LANCZOS)
            logger.info(f"  Resized input to: {width}x{height}")

        # Resize mask if needed
        if mask_image and mask_image.size != (width, height):
            mask_image = mask_image.resize((width, height), PILImage.LANCZOS)
            logger.info(f"  Resized mask to: {width}x{height}")

        # Set up generator for reproducibility
        generator = None
        if request.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(request.seed)

        # Negative prompt: only use for base variant (turbo has CFG=0, so it has no effect)
        negative_prompt_to_use = None
        if config.zimage_variant == "base":
            negative_prompt_to_use = request.negative_prompt

        start = time.time()

        # Generate image using img2img
        # Note: SLG, FMTT, DyPE, and layer_weights are not supported in img2img
        logger.info(f"Calling pipeline.img2img with strength={request.strength}...")
        if mask_image:
            logger.info("  Using differential diffusion with mask")

        image = zimage.img2img(
            prompt=request.prompt,
            negative_prompt=negative_prompt_to_use,
            image=input_image,
            mask_image=mask_image,
            strength=request.strength,
            num_inference_steps=request.steps,
            guidance_scale=request.guidance_scale,
            cfg_normalization=request.cfg_normalization,
            cfg_truncation=request.cfg_truncation,
            cfg_norm_mode=request.cfg_norm_mode,
            shift=None if request.dynamic_shift else request.shift,
            d_noise=request.d_noise,
            generator=generator,
            template=request.template,
            system_prompt=request.system_prompt,
            thinking_content=request.thinking_content,
            assistant_content=request.assistant_content,
            force_think_block=request.force_think_block,
            remove_quotes=request.strip_quotes,
            long_prompt_mode=request.long_prompt_mode,
            hidden_layer=request.hidden_layer,
            fbcache=request.fbcache,
            fbcache_threshold=request.fbcache_threshold,
            fbcache_log=request.fbcache_log,
        )

        gen_time = time.time() - start
        logger.info(f"Generated in {gen_time:.1f}s")
        logger.info("=" * 60)

        # Convert to base64 for response
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        img_b64 = base64.b64encode(img_bytes.getvalue()).decode("ascii")

        history_entry = {
            "id": len(srv.generation_history),
            "timestamp": time.time(),
            "model_type": "zimage-img2img",
            "prompt": request.prompt,
            "strength": request.strength,
            "has_mask": request.mask_image is not None,
            "width": width,
            "height": height,
            "steps": request.steps,
            "seed": request.seed,
            "gen_time": gen_time,

        }
        srv.generation_history.insert(0, history_entry)
        if len(srv.generation_history) > srv.MAX_HISTORY:
            srv.generation_history.pop()

        # Return standardized JSON response (shared format with other pipelines)
        return create_image_response(
            pipeline_id="zimage-img2img",
            seed=request.seed,
            generation_time=gen_time,
            history_id=history_entry["id"],
            img_b64=img_b64,  # Reuse already-computed base64
        )

    except Exception as e:
        logger.error(f"Img2img failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# =============================================================================
# Prompt Formatting & Templates
# =============================================================================


@router.post("/api/format-prompt", response_model=FormatPromptResult)
async def format_prompt_endpoint(request: EncodeRequest, manager: ManagerDep) -> FormatPromptResult:
    """Preview the formatted prompt without encoding (fast, no GPU needed)."""
    # Use encoder from pipeline or standalone encoder
    enc = _get_zimage_encoder(manager)
    if enc is None:
        # Try loading Z-Image on-demand to get the encoder
        try:
            _ensure_zimage_loaded(manager)
            enc = _get_zimage_encoder(manager)
        except Exception as e:
            logger.error(f"[Z-Image] Failed to load pipeline on-demand: {e}")

    if enc is None:
        raise HTTPException(
            status_code=503,
            detail="No encoder available. Z-Image pipeline failed to load.",
        )

    try:
        # Build conversation and format without encoding
        from llm_dit.conversation import Conversation

        conv = enc._build_conversation(
            prompt=request.prompt,
            template=request.template,
            system_prompt=request.system_prompt,
            thinking_content=request.thinking_content,
            assistant_content=request.assistant_content,
            force_think_block=request.force_think_block,
            remove_quotes=request.strip_quotes,
        )
        formatted = enc.formatter.format(conv)

        # Get token count if tokenizer is available
        token_count = None
        if hasattr(enc, "backend") and hasattr(enc.backend, "tokenizer"):
            tokens = enc.backend.tokenizer.encode(formatted, add_special_tokens=False)
            token_count = len(tokens)

        return FormatPromptResult(
            formatted_prompt=formatted,
            char_count=len(formatted),
            token_count=token_count,
            max_tokens=1504,
            prompt=request.prompt,
            system_prompt=request.system_prompt,
            thinking_content=request.thinking_content,
            assistant_content=request.assistant_content,
            template=request.template,
            force_think_block=request.force_think_block,
            strip_quotes=request.strip_quotes,
        )
    except Exception as e:
        logger.error(f"Format failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/templates", response_model=TemplateListResponse)
async def list_templates(manager: ManagerDep) -> TemplateListResponse:
    """List available templates with full data for UI population."""
    # Use encoder from pipeline or standalone encoder
    enc = _get_zimage_encoder(manager)
    if enc is None or enc.templates is None:
        return TemplateListResponse()

    from web.schemas import TemplateInfo

    templates = []
    for name in enc.templates:
        tpl = enc.templates.get(name)
        if tpl and tpl.category != "rewriter":  # Exclude rewriter templates
            templates.append(
                TemplateInfo(
                    name=name,
                    description=tpl.description or "",
                    category=tpl.category or "general",
                    system_prompt=tpl.content or "",
                    thinking_content=tpl.thinking_content or "",
                    assistant_content=tpl.assistant_content or "",
                    add_think_block=tpl.add_think_block,
                )
            )

    # Sort by category then name
    templates.sort(key=lambda x: (x.category, x.name))
    return TemplateListResponse(templates=templates)


@router.get("/api/rewriters", response_model=RewriterListResponse)
async def list_rewriters(manager: ManagerDep) -> RewriterListResponse:
    """List available rewriter templates."""
    # Use encoder from pipeline or standalone encoder
    enc = _get_zimage_encoder(manager)
    if enc is None or enc.templates is None:
        return RewriterListResponse()

    from web.schemas import RewriterInfo

    # Get rewriter templates (category == "rewriter")
    rewriters = []
    for tpl in enc.templates.list_by_category("rewriter"):
        rewriters.append(
            RewriterInfo(
                name=tpl.name,
                description=tpl.description,
            )
        )

    return RewriterListResponse(rewriters=rewriters)


# =============================================================================
# Prompt Rewriting
# =============================================================================


@router.post("/api/rewrite", response_model=RewriteResult)
async def rewrite_prompt(request: RewriteRequest, config: ConfigDep, manager: ManagerDep) -> RewriteResult:
    """
    Rewrite/expand a prompt using a rewriter template or custom system prompt.

    Uses the same Qwen3 model loaded for text encoding to generate expanded prompts,
    or a separate API backend if configured.

    Supports two modes:
    1. Template mode: Use `rewriter` to specify a rewriter template
    2. Ad-hoc mode: Use `custom_system_prompt` for custom rewriting instructions

    Backend selection:
    - If rewriter_use_api is True and rewriter_backend is configured, uses API backend
    - Otherwise, uses the local encoder's backend
    """
    # Validate prompt is provided
    if not request.prompt:
        raise HTTPException(
            status_code=400,
            detail="Text prompt is required.",
        )

    # Determine which backend to use for generation
    # Priority: rewriter_backend (if API mode), encoder's backend, pipeline's encoder backend
    backend = None
    backend_name = "local"

    if srv.rewriter_backend is not None:
        backend = srv.rewriter_backend
        backend_name = "api"
        logger.info("[Rewrite] Using API backend for rewriting")
    else:
        # Use encoder from pipeline or standalone encoder
        enc = _get_zimage_encoder(manager)
        if enc is not None:
            backend = getattr(enc, "backend", None)
            backend_name = "local"

    if backend is None:
        raise HTTPException(status_code=503, detail="No backend available for generation")

    if not getattr(backend, "supports_generation", False):
        raise HTTPException(status_code=400, detail="Backend does not support text generation")

    # Get template loader from encoder (for template lookup)
    enc = _get_zimage_encoder(manager)

    # Determine system prompt: custom takes precedence, then template
    system_prompt = None
    rewriter_name = "custom"

    if request.custom_system_prompt:
        # Ad-hoc mode: use custom system prompt directly
        system_prompt = request.custom_system_prompt.strip()
        rewriter_name = "custom"
        logger.info(f"[Rewrite] Using custom system prompt ({len(system_prompt)} chars)")
    elif request.rewriter:
        # Template mode: get system prompt from template
        if enc is None or enc.templates is None:
            raise HTTPException(status_code=400, detail="No templates loaded")

        rewriter_template = enc.templates.get(request.rewriter)
        if rewriter_template is None:
            raise HTTPException(
                status_code=404, detail=f"Rewriter template not found: {request.rewriter}"
            )

        if rewriter_template.category != "rewriter":
            raise HTTPException(
                status_code=400, detail=f"Template '{request.rewriter}' is not a rewriter template"
            )

        system_prompt = rewriter_template.content
        rewriter_name = request.rewriter
    else:
        raise HTTPException(
            status_code=400, detail="Either 'rewriter' or 'custom_system_prompt' must be provided"
        )

    # Get generation parameters from request or config defaults
    # Qwen3 Best Practices (thinking mode): temperature=0.6, top_p=0.95, top_k=20
    max_tokens = request.max_tokens
    temperature = request.temperature
    top_p = request.top_p
    top_k = request.top_k
    min_p = request.min_p
    presence_penalty = request.presence_penalty

    if max_tokens is None:
        max_tokens = config.rewriter_max_tokens
    if temperature is None:
        temperature = config.rewriter_temperature
    if top_p is None:
        top_p = config.rewriter_top_p
    if top_k is None:
        top_k = config.rewriter_top_k
    if min_p is None:
        min_p = config.rewriter_min_p
    if presence_penalty is None:
        presence_penalty = config.rewriter_presence_penalty

    try:
        start = time.time()
        logger.info(f"[Rewrite] Using: {rewriter_name} (backend: {backend_name})")
        logger.info(f"[Rewrite] Input prompt: {request.prompt[:100]}...")
        logger.info(
            f"[Rewrite] Params: max_tokens={max_tokens}, temperature={temperature}, top_p={top_p}, top_k={top_k}, min_p={min_p}, presence_penalty={presence_penalty}"
        )

        # Generate using the backend
        generated = backend.generate(
            prompt=request.prompt,
            system_prompt=system_prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            presence_penalty=presence_penalty,
        )

        gen_time = time.time() - start
        logger.info(f"[Rewrite] Generated {len(generated)} chars in {gen_time:.2f}s")

        # Parse the generated output to separate thinking from the prompt
        # The model may output in several formats:
        # 1. <think>...</think> followed by the prompt
        # 2. Plain reasoning text followed by JSON/structured output
        # 3. Just the rewritten prompt
        thinking_content = None
        rewritten_prompt = generated

        # Try to find <think>...</think> tags first
        think_match = re.search(r"<think>\s*(.*?)\s*</think>", generated, re.DOTALL)
        if think_match:
            thinking_content = think_match.group(1).strip()
            # Remove the think block from the rewritten prompt
            rewritten_prompt = re.sub(
                r"<think>.*?</think>\s*", "", generated, flags=re.DOTALL
            ).strip()
            logger.info(
                f"[Rewrite] Extracted thinking via <think> tags ({len(thinking_content)} chars), prompt ({len(rewritten_prompt)} chars)"
            )
        else:
            # No think tags - try to find JSON at the end and treat preceding text as thinking
            # Look for a JSON object (starts with { and ends with })
            json_match = re.search(r"(\{[\s\S]*\})\s*$", generated)
            if json_match:
                json_text = json_match.group(1)
                # Everything before the JSON is reasoning/thinking
                pre_json = generated[: json_match.start()].strip()
                if pre_json:
                    thinking_content = pre_json
                    rewritten_prompt = json_text
                    logger.info(
                        f"[Rewrite] Extracted thinking via JSON detection ({len(thinking_content)} chars), JSON prompt ({len(rewritten_prompt)} chars)"
                    )
            # If output starts with reasoning patterns like "Okay," "Let me", etc. and has a clear break
            elif re.match(r"^(Okay|Let me|I need|First|The user|Looking)", generated):
                # Look for double newline as separator between thinking and output
                parts = re.split(r"\n\n+", generated, maxsplit=1)
                if len(parts) == 2 and len(parts[1]) > 50:
                    # If second part is substantial, treat first as thinking
                    # But only if second part looks like a prompt (not more reasoning)
                    if not re.match(r"^(Okay|Let me|I need|First|The user|Looking|Now)", parts[1]):
                        thinking_content = parts[0].strip()
                        rewritten_prompt = parts[1].strip()
                        logger.info(
                            f"[Rewrite] Extracted thinking via paragraph split ({len(thinking_content)} chars), prompt ({len(rewritten_prompt)} chars)"
                        )

        # Defense in depth: strip any remaining <think>/<think> tags from both outputs
        # This handles edge cases where tags might be nested or malformed
        if thinking_content:
            thinking_content = re.sub(r"</?think>", "", thinking_content).strip()
        if rewritten_prompt:
            rewritten_prompt = re.sub(r"</?think>", "", rewritten_prompt).strip()
            # Strip surrounding quotes if the entire prompt is wrapped
            if rewritten_prompt.startswith('"') and rewritten_prompt.endswith('"'):
                rewritten_prompt = rewritten_prompt[1:-1].strip()

        # Clear CUDA cache to prevent memory issues when switching back to encoding
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("[Rewrite] Cleared CUDA cache after generation")

        return RewriteResult(
            original_prompt=request.prompt,
            rewritten_prompt=rewritten_prompt,
            thinking_content=thinking_content,
            rewriter=request.rewriter,
            backend=backend_name,
            gen_time=gen_time,
        )

    except Exception as e:
        logger.error(f"Rewrite failed: {e}")
        # Clear CUDA cache even on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Embeddings
# =============================================================================


@router.post("/api/save-embeddings", response_model=SaveEmbeddingsResult)
async def save_embeddings_endpoint(request: EncodeRequest, manager: ManagerDep) -> SaveEmbeddingsResult:
    """Encode and save embeddings to file for distributed inference."""
    # Use encoder from pipeline or standalone encoder
    enc = _get_zimage_encoder(manager)
    if enc is None:
        raise HTTPException(status_code=503, detail="Encoder not loaded")

    try:
        from llm_dit.distributed import save_embeddings as save_emb

        start = time.time()
        output = enc.encode(
            request.prompt,
            template=request.template,
            force_think_block=request.force_think_block,
        )
        encode_time = time.time() - start

        embeddings = output.embeddings[0]

        # Generate filename from prompt
        import hashlib

        prompt_hash = hashlib.md5(request.prompt.encode()).hexdigest()[:8]
        filename = f"embeddings_{prompt_hash}.safetensors"
        output_dir = Path(__file__).parent.parent / "embeddings"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / filename

        # Get device from encoder or pipeline
        device = str(enc.device) if hasattr(enc, "device") else "unknown"

        save_path = save_emb(
            embeddings=embeddings,
            path=str(output_path),
            prompt=request.prompt,
            model_path="unknown",  # Not stored in encoder
            template=request.template,
            force_think_block=request.force_think_block,
            encoder_device=device,
        )

        return SaveEmbeddingsResult(
            path=str(save_path),
            shape=list(embeddings.shape),
            encode_time=encode_time,
        )

    except Exception as e:
        logger.error(f"Save embeddings failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
