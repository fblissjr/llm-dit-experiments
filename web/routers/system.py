"""System endpoints: health, history, server info, cache management.

Handles non-generation endpoints that provide server status,
generation history management, and system operations.
"""

import gc
import logging
import time
from pathlib import Path

import torch
from fastapi import APIRouter, HTTPException

from web.dependencies import ConfigDep, ManagerDep
from web.schemas import (
    ClearCacheResponse,
    GenerationContextResponse,
    HealthResponse,
    HistoryClearResponse,
    HistoryDeleteResponse,
    HistoryResponse,
    LoRAInfo,
    RestartResponse,
    UnloadFmttResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Health & Status
# =============================================================================


@router.get("/health", response_model=HealthResponse)
async def health(manager: ManagerDep) -> HealthResponse:
    """Health check."""
    return HealthResponse(
        status="ok",
        pipeline_loaded=manager.is_loaded("zimage"),
        encoder_loaded=manager.is_loaded("zimage"),
        encoder_only_mode=False,
        qwen_image_available=manager.is_loaded("qwen_image"),
    )


@router.post("/api/system/unload-fmtt", response_model=UnloadFmttResponse)
async def unload_fmtt(manager: ManagerDep) -> UnloadFmttResponse:
    """Unload cached FMTT reward function (SigLIP) to free GPU memory."""
    zimage = manager.get_pipeline("zimage")
    if zimage is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")

    if not hasattr(zimage, "unload_fmtt"):
        raise HTTPException(
            status_code=501, detail="Pipeline version does not support FMTT unloading"
        )

    was_loaded = zimage.unload_fmtt()

    if was_loaded:
        free_gb = None
        if torch.cuda.is_available():
            free_gb = round(torch.cuda.mem_get_info()[0] / 1024**3, 2)
        return UnloadFmttResponse(success=True, message="FMTT unloaded", free_gb=free_gb)
    else:
        return UnloadFmttResponse(success=False, message="No FMTT was cached")


@router.post("/api/system/clear-cache", response_model=ClearCacheResponse)
async def clear_cache() -> ClearCacheResponse:
    """Clear CUDA cache and Python garbage collection."""
    gc.collect()

    freed_gb = 0.0
    if torch.cuda.is_available():
        before = torch.cuda.memory_reserved() / 1024**3
        torch.cuda.empty_cache()
        after = torch.cuda.memory_reserved() / 1024**3
        freed_gb = before - after

    return ClearCacheResponse(
        success=True,
        freed_gb=round(freed_gb, 2),
        message=f"Freed {freed_gb:.2f} GB of cached memory",
    )


# =============================================================================
# Generation Context (composite status for frontend status bar)
# =============================================================================


# Display name mapping for model variants
_PIPELINE_DISPLAY_NAMES = {
    "flux2": "FLUX.2",
    "zimage": "Z-Image",
    "ltx2": "LTX-2",
    "qwen_image": "Qwen-Image Edit",
    "qwen_image_t2i": "Qwen-Image T2I",
}

# FLUX.2 variant -> friendly display name
_FLUX2_VARIANT_DISPLAY = {
    "klein-4b": "FLUX.2 Klein 4B",
    "klein-9b": "FLUX.2 Klein 9B",
    "klein-base-4b": "FLUX.2 Klein Base 4B",
    "klein-base-9b": "FLUX.2 Klein Base 9B",
    "klein-4b-fp8": "FLUX.2 Klein 4B FP8",
    "klein-9b-fp8": "FLUX.2 Klein 9B FP8",
    "klein-base-4b-fp8": "FLUX.2 Klein Base 4B FP8",
    "klein-base-9b-fp8": "FLUX.2 Klein Base 9B FP8",
}


from web.utils import get_lora_info as _get_lora_info  # noqa: E402


@router.get("/api/context", response_model=GenerationContextResponse)
async def get_generation_context(config: ConfigDep, manager: ManagerDep):
    """Get composite generation context for the frontend status bar.

    Aggregates model variant, LoRA state, VRAM, quantization, compile,
    and session state into a single response. Designed to be polled at
    ~15s intervals by the frontend.
    """
    from web.server import generation_history, pending_restart_changes, server_start_time, session_modified_fields

    # Uptime
    uptime_seconds = None
    if server_start_time:
        uptime_seconds = int(time.time() - server_start_time)

    profile = getattr(config, "current_profile", None) or "default"

    # Determine active pipeline and its details
    active_pipeline = None
    pipeline_display_name = None
    model_variant = None
    loras: list[LoRAInfo] = []
    lora_summary = None
    quantization: dict[str, str] = {}
    compile_enabled = False
    compile_mode = None
    block_offload = False

    # Check each pipeline in priority order (most likely to be loaded first)
    for pid in ("flux2", "zimage", "ltx2", "qwen_image", "qwen_image_t2i"):
        if manager.is_loaded(pid):
            active_pipeline = pid
            pipeline_obj = manager.get_pipeline(pid)

            # Model variant detection
            if pid == "flux2" and isinstance(pipeline_obj, dict):
                model_variant = pipeline_obj.get("model_name")
                if model_variant:
                    pipeline_display_name = _FLUX2_VARIANT_DISPLAY.get(
                        model_variant, f"FLUX.2 {model_variant}"
                    )
                else:
                    pipeline_display_name = "FLUX.2"

                # FLUX.2 compile/offload from config
                compile_enabled = getattr(config, "flux2_compile", False)
                compile_mode = getattr(config, "flux2_compile_mode", None)
                block_offload = getattr(config, "flux2_block_offload", False)
            else:
                pipeline_display_name = _PIPELINE_DISPLAY_NAMES.get(pid, pid)

            # LoRA state
            if pipeline_obj is not None:
                loras, lora_summary = _get_lora_info(pipeline_obj)

            # Quantization (from unified config)
            try:
                quant_config = config.get_pipeline_quant_config(pid)
                for component in ("encoder", "transformer", "vae"):
                    comp_config = getattr(quant_config, component)
                    quantization[component] = comp_config.method
            except Exception:
                pass  # Pipeline may not have quant config

            break  # Only report the first loaded pipeline

    # VRAM
    vram_used_gb = None
    vram_total_gb = None
    vram_percent = None
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        vram_used_gb = round(allocated, 2)
        vram_total_gb = round(total, 2)
        vram_percent = round((allocated / total) * 100, 1) if total > 0 else 0.0

    # FMTT cache status
    fmtt_cached = False
    zimage = manager.get_pipeline("zimage")
    if zimage is not None and hasattr(zimage, "_fmtt_reward_fn"):
        fmtt_cached = zimage._fmtt_reward_fn is not None

    return GenerationContextResponse(
        uptime_seconds=uptime_seconds,
        profile=profile,
        active_pipeline=active_pipeline,
        pipeline_display_name=pipeline_display_name,
        model_variant=model_variant,
        loras=loras,
        lora_summary=lora_summary,
        quantization=quantization,
        compile_enabled=compile_enabled,
        compile_mode=compile_mode,
        block_offload=block_offload,
        vram_used_gb=vram_used_gb,
        vram_total_gb=vram_total_gb,
        vram_percent=vram_percent,
        pending_restart_fields=list(pending_restart_changes.keys()),
        session_modified_fields=list(session_modified_fields),
        fmtt_cached=fmtt_cached,
        history_count=len(generation_history),
    )


@router.post("/api/server/restart", response_model=RestartResponse)
async def restart_server(request: dict = None) -> RestartResponse:
    """Request server restart."""
    import asyncio
    import os
    import sys

    reason = request.get("reason", "user_request") if request else "user_request"
    new_profile = request.get("new_profile") if request else None

    logger.info(f"Server restart requested: reason={reason}, new_profile={new_profile}")

    python = sys.executable
    args = sys.argv.copy()

    if new_profile:
        new_args = []
        skip_next = False
        for arg in args:
            if skip_next:
                skip_next = False
                continue
            if arg == "--profile":
                skip_next = True
                continue
            if arg.startswith("--profile="):
                continue
            new_args.append(arg)
        new_args.extend(["--profile", new_profile])
        args = new_args

    logger.info(f"Restarting server with: {python} {' '.join(args)}")

    async def do_restart():
        await asyncio.sleep(1)
        os.execv(python, [python] + args)

    asyncio.create_task(do_restart())

    return RestartResponse(
        success=True,
        message="Server restarting...",
        new_profile=new_profile,
    )


# =============================================================================
# History
# =============================================================================


@router.get("/api/history", response_model=HistoryResponse)
async def get_history() -> HistoryResponse:
    """Get generation history."""
    from web.server import generation_history
    return HistoryResponse(history=generation_history)


@router.delete("/api/history/{index}", response_model=HistoryDeleteResponse)
async def delete_history_item(index: int) -> HistoryDeleteResponse:
    """Delete a history item."""
    from web.server import generation_history
    if 0 <= index < len(generation_history):
        deleted = generation_history.pop(index)
        return HistoryDeleteResponse(deleted=deleted, remaining=len(generation_history))
    raise HTTPException(status_code=404, detail="History item not found")


@router.delete("/api/history", response_model=HistoryClearResponse)
async def clear_history() -> HistoryClearResponse:
    """Clear all history."""
    import web.server as srv
    count = len(srv.generation_history)
    srv.generation_history = []
    return HistoryClearResponse(cleared=count)
