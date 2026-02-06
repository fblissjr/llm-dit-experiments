"""System endpoints: health, history, server info, cache management.

Handles non-generation endpoints that provide server status,
generation history management, and system operations.
"""

import gc
import logging
import time

import torch
from fastapi import APIRouter, HTTPException

from web.dependencies import ConfigDep, ManagerDep

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Health & Status
# =============================================================================


@router.get("/health")
async def health(manager: ManagerDep):
    """Health check."""
    return {
        "status": "ok",
        "pipeline_loaded": manager.is_loaded("zimage"),
        "encoder_loaded": manager.is_loaded("zimage"),
        "encoder_only_mode": False,
        "qwen_image_available": manager.is_loaded("qwen_image"),
    }


@router.get("/api/system/status")
async def system_status(config: ConfigDep, manager: ManagerDep):
    """Get detailed system status including memory usage and cached models."""
    from web.server import generation_history

    status = {
        "pipeline_loaded": manager.is_loaded("zimage"),
        "encoder_loaded": manager.is_loaded("zimage"),
        "encoder_only_mode": False,
        "qwen_image_available": manager.is_loaded("qwen_image"),
        "qwen_image_t2i_available": manager.is_loaded("qwen_image_t2i"),
        "ltx2_pipeline": manager.is_loaded("ltx2"),
        "flux2_pipeline": manager.is_loaded("flux2"),
        "fmtt_cached": False,
        "history_count": len(generation_history),
    }

    # Check FMTT cache
    zimage = manager.get_pipeline("zimage")
    if zimage is not None and hasattr(zimage, "_fmtt_reward_fn"):
        status["fmtt_cached"] = zimage._fmtt_reward_fn is not None

    # CUDA memory info
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free = total - reserved
        status["cuda"] = {
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "total_gb": round(total, 2),
            "free_gb": round(free, 2),
        }

    # Current configuration info (read-only display)
    config_info = {
        "model_type": config.model_type,
        "attention_backend": config.attention_backend or "auto",
    }

    if hasattr(config, "current_profile"):
        config_info["profile"] = config.current_profile

    # Z-Image specific config
    if config.model_type == "zimage":
        config_info["quantization"] = config.quantization
        config_info["cpu_offload"] = config.cpu_offload
        config_info["flash_attn"] = config.flash_attn
        config_info["torch_compile"] = config.compile
        config_info["tiled_vae"] = getattr(config, "tiled_vae", False)

    # Qwen-Image specific config (all variants)
    if config.model_type.startswith("qwenimage"):
        config_info["quantize_text_encoder"] = config.qwen_image_quantize_text_encoder
        config_info["quantize_transformer"] = (
            config.get_qwen_image_quantize_transformer()
            if hasattr(config, "get_qwen_image_quantize_transformer")
            else config.qwen_image_quantize_transformer or "none"
        )
        config_info["quantize_vae"] = getattr(config, "qwen_image_quantize_vae", "none")
        config_info["cpu_offload"] = config.qwen_image_cpu_offload
        if hasattr(config, "qwen_image_offload_type"):
            config_info["offload_type"] = config.qwen_image_offload_type
        else:
            config_info["offload_type"] = (
                "model" if config.qwen_image_cpu_offload else "none"
            )

    status["config"] = config_info

    return status


@router.post("/api/system/unload-fmtt")
async def unload_fmtt(manager: ManagerDep):
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
        if torch.cuda.is_available():
            free = torch.cuda.mem_get_info()[0] / 1024**3
            return {"success": True, "message": "FMTT unloaded", "free_gb": round(free, 2)}
        return {"success": True, "message": "FMTT unloaded"}
    else:
        return {"success": False, "message": "No FMTT was cached"}


@router.post("/api/system/clear-cache")
async def clear_cache():
    """Clear CUDA cache and Python garbage collection."""
    gc.collect()

    freed_gb = 0
    if torch.cuda.is_available():
        before = torch.cuda.memory_reserved() / 1024**3
        torch.cuda.empty_cache()
        after = torch.cuda.memory_reserved() / 1024**3
        freed_gb = before - after

    return {
        "success": True,
        "freed_gb": round(freed_gb, 2),
        "message": f"Freed {freed_gb:.2f} GB of cached memory",
    }


@router.get("/api/server/status")
async def get_server_status(config: ConfigDep):
    """Get server status including uptime and pending changes."""
    from web.server import pending_restart_changes, server_start_time, session_modified_fields

    uptime_seconds = None
    if server_start_time:
        uptime_seconds = int(time.time() - server_start_time)

    return {
        "status": "running",
        "uptime_seconds": uptime_seconds,
        "profile": getattr(config, "current_profile", "default"),
        "config_file": getattr(config, "config_path", None),
        "pending_restart": pending_restart_changes,
        "session_modified": list(session_modified_fields),
        "can_restart": True,
    }


@router.post("/api/server/restart")
async def restart_server(request: dict = None):
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

    response = {
        "success": True,
        "message": "Server restarting...",
        "new_profile": new_profile,
    }

    async def do_restart():
        await asyncio.sleep(1)
        os.execv(python, [python] + args)

    asyncio.create_task(do_restart())

    return response


# =============================================================================
# History
# =============================================================================


@router.get("/api/history")
async def get_history():
    """Get generation history."""
    from web.server import generation_history
    return {"history": generation_history}


@router.delete("/api/history/{index}")
async def delete_history_item(index: int):
    """Delete a history item."""
    from web.server import generation_history
    if 0 <= index < len(generation_history):
        deleted = generation_history.pop(index)
        return {"deleted": deleted, "remaining": len(generation_history)}
    raise HTTPException(status_code=404, detail="History item not found")


@router.delete("/api/history")
async def clear_history():
    """Clear all history."""
    import web.server as srv
    count = len(srv.generation_history)
    srv.generation_history = []
    return {"cleared": count}
