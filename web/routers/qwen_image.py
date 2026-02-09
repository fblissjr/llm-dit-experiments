"""Qwen-Image endpoints: edit, multi-edit, and text-to-image generation."""

import base64
import gc
import io
import logging
import time
import traceback

import torch
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image

import web.server as srv
from web.dependencies import ConfigDep, ManagerDep
from web.schemas import (
    QwenImage2512GenerateRequest,
    QwenImageEditLayerRequest,
    QwenImageEditMultiRequest,
    QwenImageEditStatusResponse,
    QwenImageT2IConfigResponse,
    QwenImageT2IStatusResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_HISTORY = 50


def _ensure_qwen_image_loaded(manager, config) -> None:
    """Load Qwen-Image edit pipeline on-demand via ModelManager.

    Auto-unloads Z-Image first if loaded (VRAM constraint).
    """
    if manager.is_loaded("qwen_image"):
        return

    if not config.qwen_image_model_path:
        raise HTTPException(
            status_code=503,
            detail="Qwen-Image pipeline not loaded. Configure qwen_image.model_path in config.",
        )

    # Unload Z-Image first to free VRAM for Qwen-Image-Edit
    if manager.is_loaded("zimage"):
        logger.info("[VRAM] Auto-unloading Z-Image to make room for Qwen-Image-Edit...")
        manager.unload("zimage")

    try:
        manager.load("qwen_image")
    except Exception as e:
        logger.error(f"[Qwen-Image] Failed to load pipeline: {e}")
        raise HTTPException(
            status_code=503, detail=f"Failed to load Qwen-Image pipeline: {e}"
        )


def _ensure_qwen_image_t2i_loaded(manager, config) -> None:
    """Load Qwen-Image T2I pipeline on-demand via ModelManager."""
    if manager.is_loaded("qwen_image_t2i"):
        return

    if not config.qwen_image_model_path:
        raise HTTPException(
            status_code=503,
            detail="Qwen-Image T2I not configured. Use --model-type qwenimage-t2i --qwen-image-model-path",
        )

    try:
        manager.load("qwen_image_t2i")
    except Exception as e:
        logger.error(f"[Qwen-Image T2I] Failed to load pipeline: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=503, detail=f"Failed to load Qwen-Image T2I pipeline: {e}"
        )


# =============================================================================
# Qwen-Image Edit Endpoints
# =============================================================================


@router.post("/api/qwen-image/edit-layer")
async def qwen_image_edit_layer(
    request: QwenImageEditLayerRequest, config: ConfigDep, manager: ManagerDep
):
    """Edit a decomposed layer using text instructions.

    Uses the Qwen-Image-Edit-2511 model to modify a layer based on natural language
    instructions. The edit model is loaded lazily on first use.

    Returns the edited RGBA layer as a PNG image.
    """
    _ensure_qwen_image_loaded(manager, config)
    qwen_pipeline = manager.get_pipeline("qwen_image")

    # Check if pipeline has edit capability
    if not hasattr(qwen_pipeline, "edit_layer"):
        raise HTTPException(
            status_code=400,
            detail="Pipeline does not support layer editing. Use QwenImageDiffusersPipeline.",
        )

    try:
        # Decode base64 layer image
        image_data = request.layer_image
        if image_data.startswith("data:"):
            image_data = image_data.split(",", 1)[1]
        image_bytes = base64.b64decode(image_data)
        layer_image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")

        logger.info("=" * 60)
        logger.info("QWEN-IMAGE LAYER EDIT REQUEST")
        logger.info("=" * 60)
        logger.info(f"  Layer size: {layer_image.size}")
        logger.info(f"  Instruction: {request.instruction[:80]}...")
        logger.info(f"  CFG Scale: {request.cfg_scale}")
        logger.info(f"  Steps: {request.steps}")
        logger.info(f"  Seed: {request.seed}")

        start = time.time()

        # Run layer edit
        edited_layer = qwen_pipeline.edit_layer(
            layer_image=layer_image,
            instruction=request.instruction,
            num_inference_steps=request.steps,
            cfg_scale=request.cfg_scale,
            seed=request.seed,
        )

        edit_time = time.time() - start
        logger.info(f"[Qwen-Image] Edited layer in {edit_time:.1f}s")
        logger.info("=" * 60)

        # Convert to PNG bytes
        img_bytes = io.BytesIO()
        edited_layer.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        return StreamingResponse(
            img_bytes,
            media_type="image/png",
            headers={
                "Content-Disposition": f"attachment; filename=edited_layer_{int(time.time())}.png",
                "X-Edit-Time": str(edit_time),
                "X-Seed": str(request.seed) if request.seed else "random",
            },
        )

    except Exception as e:
        logger.error(f"[Qwen-Image] Layer edit failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/qwen-image/edit-status", response_model=QwenImageEditStatusResponse)
async def qwen_image_edit_status(manager: ManagerDep) -> QwenImageEditStatusResponse:
    """Check if the edit model is loaded and ready."""
    qwen_pipeline = manager.get_pipeline("qwen_image")
    if qwen_pipeline is None:
        return QwenImageEditStatusResponse(available=False)

    has_edit_method = hasattr(qwen_pipeline, "edit_layer")
    has_edit_pipe = (
        hasattr(qwen_pipeline, "has_edit_model") and qwen_pipeline.has_edit_model
    )

    return QwenImageEditStatusResponse(
        available=has_edit_method,
        edit_model_loaded=has_edit_pipe,
        edit_model_path=getattr(qwen_pipeline, "_edit_model_path", None),
        supports_multi_image=hasattr(qwen_pipeline, "edit_multi"),
    )


@router.post("/api/qwen-image/edit-multi")
async def qwen_image_edit_multi(
    request: QwenImageEditMultiRequest, config: ConfigDep, manager: ManagerDep
):
    """Combine multiple images using Qwen-Image-Edit-2511.

    New capability in Edit-2511 for multi-person consistency and creative
    image merging. Supports combining 2+ images into a single coherent output.

    Returns the combined output as a PNG image.
    """
    # Validate input
    if len(request.images) < 2:
        raise HTTPException(
            status_code=400,
            detail=f"edit-multi requires at least 2 images, got {len(request.images)}. "
            "For single-image editing, use /api/qwen-image/edit-layer instead.",
        )

    _ensure_qwen_image_loaded(manager, config)
    qwen_pipeline = manager.get_pipeline("qwen_image")

    # Check if pipeline supports multi-image editing
    if not hasattr(qwen_pipeline, "edit_multi"):
        raise HTTPException(
            status_code=400,
            detail="Pipeline does not support multi-image editing. "
            "Use QwenImageDiffusersPipeline with Edit-2511 model.",
        )

    try:
        # Decode base64 images
        pil_images = []
        for i, img_data in enumerate(request.images):
            try:
                if img_data.startswith("data:"):
                    img_data = img_data.split(",", 1)[1]
                img_bytes = base64.b64decode(img_data)
                img = Image.open(io.BytesIO(img_bytes))
                pil_images.append(img)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to decode image {i}: {e}")

        logger.info("=" * 60)
        logger.info("QWEN-IMAGE MULTI-EDIT REQUEST")
        logger.info("=" * 60)
        logger.info(f"  Number of images: {len(pil_images)}")
        for i, img in enumerate(pil_images):
            logger.info(f"  Image {i}: {img.size}, mode={img.mode}")
        logger.info(f"  Instruction: {request.instruction[:80]}...")
        logger.info(f"  CFG Scale: {request.cfg_scale}")
        logger.info(f"  Steps: {request.steps}")
        logger.info(f"  Seed: {request.seed}")
        logger.info("=" * 60)

        start = time.time()

        # Run multi-image edit
        combined_image = qwen_pipeline.edit_multi(
            images=pil_images,
            instruction=request.instruction,
            num_inference_steps=request.steps,
            cfg_scale=request.cfg_scale,
            seed=request.seed,
        )

        edit_time = time.time() - start
        logger.info(f"[Qwen-Image] Multi-edit completed in {edit_time:.1f}s")
        logger.info(f"  Output size: {combined_image.size}")
        logger.info("=" * 60)

        # Convert to PNG bytes
        img_bytes = io.BytesIO()
        combined_image.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        return StreamingResponse(
            img_bytes,
            media_type="image/png",
            headers={
                "X-Inference-Time": f"{edit_time:.2f}",
                "X-Image-Count": str(len(pil_images)),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Qwen-Image] Multi-edit failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Qwen-Image T2I (Pure Text-to-Image) Endpoints
# Uses unified config: --model-type qwenimage-t2i --qwen-image-model-path
# =============================================================================


def _is_t2i_configured(config) -> bool:
    """Check if T2I is configured via unified config."""
    # T2I uses the unified qwen_image_model_path when model_type is qwenimage-t2i
    return (
        bool(config.qwen_image_model_path) and config.model_type == "qwenimage-t2i"
    )


@router.get("/api/qwen-image-2512/status", response_model=QwenImageT2IStatusResponse)
async def qwen_image_2512_status(
    config: ConfigDep, manager: ManagerDep
) -> QwenImageT2IStatusResponse:
    """Check Qwen-Image T2I pipeline status.

    Note: Uses unified config (--model-type qwenimage-t2i --qwen-image-model-path).
    """
    configured = _is_t2i_configured(config)
    loaded = manager.is_loaded("qwen_image_t2i")

    return QwenImageT2IStatusResponse(
        available=loaded,
        configured=configured,
        model_path=config.qwen_image_model_path,
        quantize_transformer=config.get_qwen_image_quantize_transformer(),
        quantize_text_encoder=config.qwen_image_quantize_text_encoder,
    )


@router.get("/api/qwen-image-2512/config", response_model=QwenImageT2IConfigResponse)
async def qwen_image_2512_config(config: ConfigDep) -> QwenImageT2IConfigResponse:
    """Get Qwen-Image T2I configuration and defaults.

    Note: Uses unified config (--model-type qwenimage-t2i).
    Variant-aware defaults: T2I uses 40 steps, 1024 resolution, fp8 quantization.
    """
    return QwenImageT2IConfigResponse(
        model_path=config.qwen_image_model_path,
        steps=config.get_qwen_image_steps(),
        cfg_scale=config.qwen_image_cfg_scale,
        quantize_transformer=config.get_qwen_image_quantize_transformer(),
        quantize_text_encoder=config.qwen_image_quantize_text_encoder,
        default_width=1024,
        default_height=1024,
        max_sequence_length=512,
    )


@router.post("/api/qwen-image-2512/generate")
async def qwen_image_2512_generate(
    request: QwenImage2512GenerateRequest, config: ConfigDep, manager: ManagerDep
):
    """Generate an image using Qwen-Image T2I (pure text-to-image).

    Uses unified config: --model-type qwenimage-t2i --qwen-image-model-path
    Variant-aware defaults: T2I uses 40 steps, 1024 resolution, fp8 quantization.
    """
    _ensure_qwen_image_t2i_loaded(manager, config)
    t2i_pipeline = manager.get_pipeline("qwen_image_t2i")

    try:
        logger.info("=" * 60)
        logger.info("QWEN-IMAGE T2I GENERATION REQUEST")
        logger.info("=" * 60)
        if config.logging.log_prompts:
            logger.info(f"  Prompt: {request.prompt[:80]}...")
        if config.logging.log_generation_params:
            logger.info(f"  Size: {request.width}x{request.height}")
            logger.info(f"  Steps: {request.steps}")
            logger.info(f"  CFG Scale: {request.cfg_scale}")
            logger.info(f"  Seed: {request.seed}")
        logger.info("=" * 60)

        start = time.time()

        # Generate image
        image = t2i_pipeline(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt or " ",
            height=request.height,
            width=request.width,
            num_inference_steps=request.steps,
            cfg_scale=request.cfg_scale,
            seed=request.seed,
            max_sequence_length=request.max_sequence_length,
        )

        gen_time = time.time() - start
        logger.info(f"[Qwen-Image T2I] Generated in {gen_time:.1f}s")
        logger.info(f"  Output size: {image.size}")
        logger.info("=" * 60)

        # Convert to PNG bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        # Add to history (metadata only -- frontend stores images in IndexedDB)
        history_entry = {
            "id": len(srv.generation_history),
            "timestamp": time.time(),
            "model_type": "qwenimage-t2i",
            "prompt": request.prompt,
            "width": request.width,
            "height": request.height,
            "steps": request.steps,
            "cfg_scale": request.cfg_scale,
            "seed": request.seed,
            "generation_time": gen_time,
        }
        srv.generation_history.append(history_entry)
        if len(srv.generation_history) > MAX_HISTORY:
            srv.generation_history.pop(0)

        # Reset stream position for response
        img_bytes.seek(0)

        return StreamingResponse(
            img_bytes,
            media_type="image/png",
            headers={
                "X-Inference-Time": f"{gen_time:.2f}",
                "X-Model": "qwen-image-2512",
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Qwen-Image T2I] Generation failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
