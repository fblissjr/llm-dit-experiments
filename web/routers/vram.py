"""VRAM and model lifecycle management endpoints.

Handles loading, unloading, and status queries for all pipeline models.
The unified /api/models/ endpoints provide a consistent API for the React
frontend, while /api/vram/ endpoints are the legacy per-pipeline variants.

LoRA discovery endpoints are also included here since they relate to
model configuration.
"""

import gc
import logging
import traceback
from pathlib import Path
from typing import Callable

import torch
from fastapi import APIRouter, HTTPException

from web.dependencies import ConfigDep, ManagerDep
from web.schemas import (
    LoRAFileInfo,
    LoRAListResponse,
    ModelStatusResponse,
    SuccessVramResponse,
    VRAMStatusResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Helper functions
# =============================================================================


def _sync_globals_after_unload(pipeline_ids: list[str]):
    """Sync server.py globals after unloading pipelines via ModelManager.

    Transitional shim -- flux2.py and config_mgmt.py still read these
    globals directly. Once they migrate to ManagerDep, this can be removed.
    """
    import web.server as srv

    for pid in pipeline_ids:
        if pid == "zimage":
            srv.pipeline = None
            srv.encoder = None
        elif pid == "qwen_image":
            srv.qwen_image_pipeline = None
        elif pid == "qwen_image_t2i":
            srv.qwen_image_t2i_pipeline = None
        elif pid == "flux2":
            srv.flux2_pipeline = None
        # ltx2 has no persistent global (loads/unloads per-request)


def _sync_globals_after_load(pipeline_id: str, manager):
    """Sync server.py globals after loading a pipeline via ModelManager.

    Transitional shim -- flux2.py and config_mgmt.py still read these
    globals directly. Once they migrate to ManagerDep, this can be removed.
    """
    import web.server as srv

    if pipeline_id == "zimage":
        srv.pipeline = manager.get_pipeline("zimage")
        srv.encoder = manager.encoder
    elif pipeline_id == "qwen_image":
        srv.qwen_image_pipeline = manager.get_pipeline("qwen_image")
    elif pipeline_id == "qwen_image_t2i":
        srv.qwen_image_t2i_pipeline = manager.get_pipeline("qwen_image_t2i")
    elif pipeline_id == "flux2":
        srv.flux2_pipeline = manager.get_pipeline("flux2")


def _get_pipeline_config_metadata(config, pipeline: str) -> dict:
    """Build config tags and warnings for a pipeline's status display.

    Uses the unified quantization config system. Supports pipeline-specific
    compile and block_offload settings where applicable.
    """
    if config is None:
        return {"config_tags": [], "config_warnings": []}

    config_tags = []
    config_warnings = []

    # Resolve quantization from unified config
    quant_config = config.get_pipeline_quant_config(pipeline)

    # Add quantization tags for each component
    for component in ("encoder", "transformer", "vae"):
        comp_config = getattr(quant_config, component)
        if comp_config.method != "none":
            config_tags.append({
                "key": f"quant_{component}",
                "label": f"{component}: {comp_config.method.upper()}",
                "color": "purple",
            })

    # Pipeline-specific settings
    compile_enabled = False
    compile_mode = "default"
    compile_vae = False
    block_offload = False

    if pipeline == "flux2":
        compile_enabled = getattr(config, "flux2_compile", False)
        compile_vae = getattr(config, "flux2_compile_vae", False)
        compile_mode = getattr(config, "flux2_compile_mode", "default")
        block_offload = getattr(config, "flux2_block_offload", False)

    if compile_enabled:
        config_tags.append({"key": "compile", "label": f"compiled ({compile_mode})", "color": "blue"})

        from llm_dit.quantization import get_quant_compile_warnings

        for warning_msg in get_quant_compile_warnings(quant_config.transformer.method, compile_mode):
            config_warnings.append({"severity": "warning", "message": warning_msg})
        config_warnings.append({
            "severity": "warning",
            "message": "torch.compile active -- first generation at each resolution takes ~90s warmup",
        })
        if block_offload:
            config_warnings.append({
                "severity": "error",
                "message": "compile=true is incompatible with block_offload=true. Loading will fail.",
            })
    if compile_vae:
        config_tags.append({"key": "compile_vae", "label": "VAE compiled", "color": "blue"})
    if block_offload:
        config_tags.append({"key": "block_offload", "label": "block offload", "color": "orange"})
    if quant_config.transformer.method != "none" and block_offload:
        config_warnings.append({
            "severity": "error",
            "message": f"quantization={quant_config.transformer.method} is incompatible with block_offload=true. Loading will fail.",
        })

    return {"config_tags": config_tags, "config_warnings": config_warnings}


# =============================================================================
# VRAM Status
# =============================================================================


@router.get("/api/vram/status", response_model=VRAMStatusResponse)
async def vram_status(manager: ManagerDep) -> VRAMStatusResponse:
    """Get current VRAM usage and loaded models status."""
    data = manager.get_vram_status()
    return VRAMStatusResponse(**data)


# =============================================================================
# Per-Pipeline Load/Unload (internal -- called by unified /api/models/ endpoints)
# =============================================================================


async def vram_load_zimage(config: ConfigDep, manager: ManagerDep):
    """Load Z-Image pipeline on-demand.

    Uses model_path and other settings from config.toml.
    Call this before generating if the pipeline was previously unloaded.
    """
    if manager.is_loaded("zimage"):
        status = manager.get_vram_status()
        return {
            "success": True,
            "message": "Z-Image pipeline already loaded",
            "vram": status.get("vram"),
        }

    model_path = (config.zimage_model_path or config.model_path) if config else None
    if config is None or not model_path:
        raise HTTPException(
            status_code=400,
            detail="Z-Image model_path not configured. Set [zimage].model_path in config.toml",
        )

    try:
        manager.load("zimage")
        _sync_globals_after_load("zimage", manager)
        status = manager.get_vram_status()
        return {
            "success": True,
            "message": "Z-Image pipeline loaded successfully",
            "vram": status.get("vram"),
        }
    except Exception as e:
        logger.error(f"[Z-Image] Failed to load pipeline: {e}")
        raise HTTPException(status_code=503, detail=f"Failed to load Z-Image pipeline: {e}")


async def vram_unload_zimage(manager: ManagerDep):
    """Unload Z-Image pipeline (encoder + DiT + VAE) to free VRAM."""
    unloaded = manager.unload("zimage")
    _sync_globals_after_unload(["zimage"])

    status = manager.get_vram_status()
    return {
        "success": unloaded,
        "message": "Z-Image pipeline unloaded" if unloaded else "Z-Image pipeline was not loaded",
        "vram": status.get("vram"),
    }


async def vram_load_qwen_image(config: ConfigDep, manager: ManagerDep):
    """Load Qwen-Image Edit pipeline on-demand."""
    if manager.is_loaded("qwen_image"):
        status = manager.get_vram_status()
        return {
            "success": True,
            "message": "Qwen-Image Edit pipeline already loaded",
            "vram": status.get("vram"),
        }

    if not config.qwen_image_model_path:
        raise HTTPException(
            status_code=400,
            detail="Qwen-Image model_path not configured. Set qwen_image.model_path in config.toml",
        )

    try:
        manager.load("qwen_image")
        _sync_globals_after_load("qwen_image", manager)
        status = manager.get_vram_status()
        return {
            "success": True,
            "message": "Qwen-Image Edit pipeline loaded successfully",
            "vram": status.get("vram"),
        }
    except Exception as e:
        logger.error(f"[Qwen-Image] Failed to load pipeline: {e}")
        raise HTTPException(status_code=503, detail=f"Failed to load Qwen-Image Edit pipeline: {e}")


async def vram_unload_qwen_image(manager: ManagerDep):
    """Unload Qwen-Image pipeline to free VRAM."""
    unloaded = manager.unload("qwen_image")
    _sync_globals_after_unload(["qwen_image"])

    status = manager.get_vram_status()
    return {
        "success": unloaded,
        "message": "Qwen-Image pipeline unloaded" if unloaded else "Qwen-Image pipeline was not loaded",
        "vram": status.get("vram"),
    }


async def vram_load_qwen_image_t2i(config: ConfigDep, manager: ManagerDep):
    """Load Qwen-Image T2I pipeline on-demand."""
    if manager.is_loaded("qwen_image_t2i"):
        status = manager.get_vram_status()
        return {
            "success": True,
            "message": "Qwen-Image T2I pipeline already loaded",
            "vram": status.get("vram"),
        }

    if not config.qwen_image_model_path:
        raise HTTPException(
            status_code=400,
            detail="Qwen-Image model_path not configured. Set qwen_image.model_path in config.toml",
        )

    try:
        manager.load("qwen_image_t2i")
        _sync_globals_after_load("qwen_image_t2i", manager)
        status = manager.get_vram_status()
        return {
            "success": True,
            "message": "Qwen-Image T2I pipeline loaded successfully",
            "vram": status.get("vram"),
        }
    except Exception as e:
        logger.error(f"[Qwen-Image T2I] Failed to load pipeline: {e}")
        raise HTTPException(status_code=503, detail=f"Failed to load Qwen-Image T2I pipeline: {e}")


async def vram_unload_qwen_image_t2i(manager: ManagerDep):
    """Unload Qwen-Image T2I pipeline to free VRAM."""
    unloaded = manager.unload("qwen_image_t2i")
    _sync_globals_after_unload(["qwen_image_t2i"])

    status = manager.get_vram_status()
    return {
        "success": unloaded,
        "message": "Qwen-Image T2I pipeline unloaded"
        if unloaded
        else "Qwen-Image T2I pipeline was not loaded",
        "vram": status.get("vram"),
    }


async def vram_load_ltx2(config: ConfigDep, manager: ManagerDep):
    """Validate LTX-2 configuration.

    Note: The pure PyTorch pipeline loads components per-request with
    automatic memory offloading. This endpoint validates that the model
    path is configured correctly.
    """
    from web.routers.ltx2 import get_ltx2_model_path

    try:
        model_path = get_ltx2_model_path(config)
        status = manager.get_vram_status()
        return {
            "success": True,
            "message": f"LTX-2 model path validated: {model_path}",
            "vram": status.get("vram"),
        }
    except Exception as e:
        logger.error(f"[LTX-2] Configuration validation failed: {e}")
        raise HTTPException(status_code=503, detail=f"LTX-2 configuration error: {e}")


async def vram_unload_ltx2(manager: ManagerDep):
    """Clean up VRAM after LTX-2 operations.

    Note: The pure PyTorch pipeline automatically unloads components after
    each generation. This endpoint performs a manual memory cleanup.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    status = manager.get_vram_status()
    return {
        "success": True,
        "message": "LTX-2 memory cleanup complete",
        "vram": status.get("vram"),
    }


async def vram_load_flux2(config: ConfigDep, manager: ManagerDep):
    """Load FLUX.2 Klein pipeline on-demand.

    Delegates to ModelManager._load_flux2() which implements the 3-stage
    loading pattern (encoder -> transformer -> VAE) with pinned memory.
    """
    if manager.is_loaded("flux2"):
        status = manager.get_vram_status()
        return {
            "success": True,
            "message": "FLUX.2 pipeline already loaded",
            "vram": status.get("vram"),
        }

    model_path = getattr(config, "flux2_model_path", None)
    if not model_path:
        raise HTTPException(
            status_code=400,
            detail="FLUX.2 model_path not configured. Set flux2.model_path in config.toml",
        )

    try:
        result = manager.load("flux2")
        _sync_globals_after_load("flux2", manager)
        status = manager.get_vram_status()
        return {
            "success": True,
            "message": f"FLUX.2 pipeline loaded ({result.mode})",
            "vram": status.get("vram"),
        }
    except Exception as e:
        logger.error(f"[FLUX.2] Failed to load pipeline: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=503, detail=f"Failed to load FLUX.2 pipeline: {e}")


async def vram_unload_flux2(manager: ManagerDep):
    """Unload FLUX.2 Klein pipeline to free VRAM."""
    unloaded = manager.unload("flux2")
    _sync_globals_after_unload(["flux2"])

    status = manager.get_vram_status()
    return {
        "success": unloaded,
        "message": "FLUX.2 pipeline unloaded" if unloaded else "FLUX.2 pipeline was not loaded",
        "vram": status.get("vram"),
    }


# =============================================================================
# Unified Model Management API
# =============================================================================
# These endpoints provide a consistent API for the React frontend to load/unload
# models by pipeline ID, mapping to the specific load functions above.


# Dispatch tables map pipeline IDs to the router-local handler functions.
# Populated after all handlers are defined (see bottom of this section).
PIPELINE_LOADERS: dict[str, Callable] = {}
PIPELINE_UNLOADERS: dict[str, Callable] = {}


@router.post("/api/models/{pipeline_id}/load", response_model=SuccessVramResponse)
async def load_model_by_id(pipeline_id: str, config: ConfigDep, manager: ManagerDep) -> SuccessVramResponse:
    """Load a model by pipeline ID.

    This is the unified API for the React frontend. Maps pipeline IDs to
    the specific load functions, passing through DI dependencies.
    """
    loader_fn = PIPELINE_LOADERS.get(pipeline_id.lower())
    if not loader_fn:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown pipeline: {pipeline_id}. Available: {list(PIPELINE_LOADERS.keys())}",
        )

    try:
        result = await loader_fn(config=config, manager=manager)
        return SuccessVramResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load {pipeline_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/models/{pipeline_id}/unload", response_model=SuccessVramResponse)
async def unload_model_by_id(pipeline_id: str, manager: ManagerDep) -> SuccessVramResponse:
    """Unload a model by pipeline ID."""
    unloader_fn = PIPELINE_UNLOADERS.get(pipeline_id.lower())
    if not unloader_fn:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown pipeline: {pipeline_id}. Available: {list(PIPELINE_UNLOADERS.keys())}",
        )

    try:
        result = await unloader_fn(manager=manager)
        return SuccessVramResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to unload {pipeline_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/models/unload-all", response_model=SuccessVramResponse)
async def unload_all_models(manager: ManagerDep) -> SuccessVramResponse:
    """Unload all loaded models to free VRAM."""
    manager.unload_all_except(None)
    _sync_globals_after_unload(["zimage", "qwen_image", "qwen_image_t2i", "flux2"])

    status = manager.get_vram_status()
    return SuccessVramResponse(
        success=True,
        message="All models unloaded",
        vram=status.get("vram"),
    )


@router.get("/api/models/{pipeline_id}/status", response_model=ModelStatusResponse)
async def get_model_status(pipeline_id: str, config: ConfigDep, manager: ManagerDep) -> ModelStatusResponse:
    """Get the status of a specific pipeline model."""
    pid = pipeline_id.lower()

    loaded = False
    components = []
    total_vram_mb = 0
    config_meta: dict = {}

    if pid in ("zimage", "z-image"):
        loaded = manager.is_loaded("zimage")
        if loaded:
            components = [
                {"name": "encoder", "vramMB": 8000},
                {"name": "transformer", "vramMB": 8000},
                {"name": "vae", "vramMB": 500},
            ]
            total_vram_mb = sum(c["vramMB"] for c in components)
    elif pid == "qwenimage-edit":
        loaded = manager.is_loaded("qwen_image")
    elif pid == "qwenimage-t2i":
        loaded = manager.is_loaded("qwen_image_t2i")
    elif pid == "ltx2":
        loaded = manager.is_loaded("ltx2")
        if loaded:
            components = [
                {"name": "encoder", "vramMB": 3000},
                {"name": "transformer", "vramMB": 20000},
                {"name": "vae", "vramMB": 1000},
            ]
            total_vram_mb = sum(c["vramMB"] for c in components)
    elif pid == "flux2":
        loaded = manager.is_loaded("flux2")
        config_meta = _get_pipeline_config_metadata(config, "flux2")
        if loaded:
            components = [
                {"name": "encoder", "vramMB": 2000},
                {"name": "transformer", "vramMB": 12000},
                {"name": "vae", "vramMB": 500},
            ]
            total_vram_mb = sum(c["vramMB"] for c in components)
    else:
        raise HTTPException(status_code=404, detail=f"Unknown pipeline: {pipeline_id}")

    # Enrich with model variant and LoRA state
    model_variant = None
    display_name = None
    loras_info: list = []
    lora_summary = None

    if loaded:
        pipeline_obj = manager.get_pipeline(pid)
        if pipeline_obj is not None:
            # Model variant (FLUX.2 stores model_name in its pipeline dict)
            if isinstance(pipeline_obj, dict):
                model_variant = pipeline_obj.get("model_name")
            # LoRA state
            from web.utils import get_lora_info

            loras_info, lora_summary = get_lora_info(pipeline_obj)

    return ModelStatusResponse(
        pipeline_id=pipeline_id,
        status="loaded" if loaded else "unloaded",
        components=components if loaded else [],
        total_vram_mb=total_vram_mb if loaded else 0,
        vram_mb=total_vram_mb if loaded else 0,
        model_variant=model_variant,
        display_name=display_name,
        loras=loras_info,
        lora_summary=lora_summary,
        config_tags=config_meta.get("config_tags", []),
        config_warnings=config_meta.get("config_warnings", []),
    )


# =============================================================================
# LoRA Management API
# =============================================================================


@router.get("/api/loras", response_model=LoRAListResponse)
async def list_available_loras(config: ConfigDep) -> LoRAListResponse:
    """List all available LoRA files from configured directories.

    Scans directories in [lora].paths config for .safetensors files.
    """
    lora_files = []
    lora_dirs = config.lora_paths if config else []

    if not lora_dirs:
        lora_dirs = ["loras"]

    for lora_dir in lora_dirs:
        dir_path = Path(lora_dir)
        if not dir_path.exists():
            logger.debug(f"LoRA directory not found: {lora_dir}")
            continue

        for safetensor_file in dir_path.rglob("*.safetensors"):
            relative_path = str(safetensor_file)
            lora_files.append(LoRAFileInfo(
                path=relative_path,
                name=safetensor_file.stem,
                directory=str(safetensor_file.parent),
                size_mb=round(safetensor_file.stat().st_size / (1024 * 1024), 1),
            ))

    lora_files.sort(key=lambda x: x.name.lower())

    return LoRAListResponse(
        loras=lora_files,
        directories=lora_dirs,
        count=len(lora_files),
    )


@router.get("/api/loras/{pipeline_id}", response_model=LoRAListResponse)
async def list_loras_for_pipeline(pipeline_id: str, config: ConfigDep) -> LoRAListResponse:
    """List LoRA files available for a specific pipeline."""
    pipeline_lora_dirs = {
        "flux2": ["loras/FLUX.2-klein", "loras/flux2"],
        "ltx2": ["loras/LTX-2", "loras/ltx2"],
        "zimage": ["loras/Z-Image", "loras/zimage"],
    }

    dirs = pipeline_lora_dirs.get(
        pipeline_id.lower(),
        config.lora_paths if config else ["loras"],
    )

    lora_files = []
    for lora_dir in dirs:
        dir_path = Path(lora_dir)
        if not dir_path.exists():
            continue

        for safetensor_file in dir_path.rglob("*.safetensors"):
            relative_path = str(safetensor_file)
            lora_files.append(LoRAFileInfo(
                path=relative_path,
                name=safetensor_file.stem,
                directory=str(safetensor_file.parent),
                size_mb=round(safetensor_file.stat().st_size / (1024 * 1024), 1),
            ))

    lora_files.sort(key=lambda x: x.name.lower())

    return LoRAListResponse(
        loras=lora_files,
        pipeline_id=pipeline_id,
        directories=dirs,
        count=len(lora_files),
    )


# =============================================================================
# Dispatch table initialization
# =============================================================================
# Populated after all handler functions are defined so references resolve.

PIPELINE_LOADERS.update({
    "zimage": vram_load_zimage,
    "z-image": vram_load_zimage,
    "qwenimage-edit": vram_load_qwen_image,
    "qwenimage-t2i": vram_load_qwen_image_t2i,
    "ltx2": vram_load_ltx2,
    "flux2": vram_load_flux2,
})

PIPELINE_UNLOADERS.update({
    "zimage": vram_unload_zimage,
    "z-image": vram_unload_zimage,
    "qwenimage-edit": vram_unload_qwen_image,
    "qwenimage-t2i": vram_unload_qwen_image_t2i,
    "ltx2": vram_unload_ltx2,
    "flux2": vram_unload_flux2,
})
