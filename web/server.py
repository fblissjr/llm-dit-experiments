#!/usr/bin/env python3
"""
Simple web server for Z-Image generation.

Usage:
    uv run web/server.py
    uv run web/server.py --port 8000
    uv run web/server.py --config config.toml --profile default
    uv run web/server.py --encoder-only  # Fast mode, no DiT/VAE
"""

import gc
import logging
import os
import signal
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path so `web.routers` and `llm_dit` resolve
# when invoked directly (e.g., `python web/server.py`).
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

app = FastAPI(title="Z-Image Generator")

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _register_routers():
    """Register all API routers. Called from main() to avoid circular imports.

    Routers import `web.server as srv` at module level to access globals
    (pipeline, encoder, etc). If we imported them at server module level,
    we'd get a circular dependency. Lazy registration breaks the cycle.
    """
    from web.routers import config_mgmt as config_mgmt_router
    from web.routers import core as core_router
    from web.routers import flux2 as flux2_router
    from web.routers import ltx2 as ltx2_router
    from web.routers import qwen_image as qwen_image_router
    from web.routers import system as system_router
    from web.routers import vram as vram_router

    app.include_router(system_router.router)
    app.include_router(vram_router.router)
    app.include_router(config_mgmt_router.router)
    app.include_router(core_router.router)
    app.include_router(flux2_router.router)
    app.include_router(ltx2_router.router)
    app.include_router(qwen_image_router.router)

# Static files (CSS, JS)
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

# Global pipeline/encoder state (loaded on startup)
# Routers access these via `import web.server as srv; srv.pipeline` etc.
# Backed by ModelManager -- see main() for initialization.
pipeline = None  # Z-Image pipeline
encoder = None  # For encoder-only mode
rewriter_backend = None  # API backend for rewriting (if configured)
runtime_config = None  # RuntimeConfig from CLI/TOML
encoder_only_mode = False

# Qwen-Image pipeline (separate from Z-Image)
qwen_image_pipeline = None

# Qwen-Image T2I pipeline (pure text-to-image, separate from Qwen-Image-Layered/Edit)
qwen_image_t2i_pipeline = None

# LTX-2 video generation (deprecated pipeline state)
# Note: Pure PyTorch pipeline loads/unloads components per-request
ltx2_pipeline = None

# FLUX.2 Klein image generation pipeline
flux2_pipeline = None

# Unified model lifecycle manager (replaces per-pipeline load/unload functions)
# Initialized in main(). All load/unload operations delegate to this.
model_manager = None  # type: ignore[assignment]

# In-memory history (cleared on server restart)
generation_history = []
MAX_HISTORY = 50

# Config management - session tracking
session_file_values = {}  # Original values from config file (for detecting changes)
session_modified_fields = set()  # Fields modified during this session
pending_restart_changes = {}  # Changes that require server restart
server_start_time = None  # For uptime tracking


def unload_zimage_pipeline() -> bool:
    """Unload Z-Image pipeline (encoder + DiT + VAE) to free VRAM.

    Delegates to ModelManager when available. Returns True if unloaded.
    """
    global pipeline, encoder

    if model_manager is not None:
        unloaded = model_manager.unload("zimage")
        pipeline = None
        encoder = None
        return unloaded

    # Fallback: pre-ModelManager codepath
    unloaded = False
    if pipeline is not None:
        logger.info("[VRAM] Unloading Z-Image pipeline to free VRAM...")
        try:
            if hasattr(pipeline, "transformer") and pipeline.transformer is not None:
                pipeline.transformer.to("cpu")
            if hasattr(pipeline, "vae") and pipeline.vae is not None:
                pipeline.vae.to("cpu")
        except Exception as e:
            logger.warning(f"[VRAM] Error moving pipeline to CPU: {e}")
        del pipeline
        pipeline = None
        unloaded = True

    if encoder is not None:
        logger.info("[VRAM] Unloading Z-Image encoder...")
        try:
            if hasattr(encoder, "backend") and encoder.backend is not None:
                if hasattr(encoder.backend, "model") and encoder.backend.model is not None:
                    encoder.backend.model.to("cpu")
        except Exception as e:
            logger.warning(f"[VRAM] Error moving encoder to CPU: {e}")
        del encoder
        encoder = None
        unloaded = True

    if unloaded:
        try:
            import torch._dynamo
            torch._dynamo.reset()
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

    return unloaded


def unload_qwen_image_pipeline() -> bool:
    """Unload Qwen-Image pipeline to free VRAM.

    Returns True if unloaded, False if not loaded.
    """
    global qwen_image_pipeline
    import torch

    if qwen_image_pipeline is not None:
        logger.info("[VRAM] Unloading Qwen-Image pipeline to free VRAM...")
        del qwen_image_pipeline
        qwen_image_pipeline = None
        torch.cuda.empty_cache()
        logger.info("[VRAM] Qwen-Image pipeline unloaded, CUDA cache cleared")
        return True
    return False


def unload_qwen_image_t2i_pipeline() -> bool:
    """Unload Qwen-Image T2I pipeline to free VRAM.

    Returns True if unloaded, False if not loaded.
    """
    global qwen_image_t2i_pipeline

    if qwen_image_t2i_pipeline is not None:
        logger.info("[VRAM] Unloading Qwen-Image T2I pipeline to free VRAM...")
        del qwen_image_t2i_pipeline
        qwen_image_t2i_pipeline = None
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"[VRAM] Qwen-Image T2I unloaded. CUDA allocated: {allocated:.2f} GB")
        return True
    return False


def unload_ltx2_pipeline() -> bool:
    """Clean up VRAM after LTX-2 operations.

    Note: Pure PyTorch pipeline loads/unloads components per-request via
    generate_video_with_offloading(). This function performs a general cleanup.

    Returns True after cleanup.
    """
    logger.info("[VRAM] Running LTX-2 memory cleanup...")
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"[VRAM] Cleanup complete. CUDA allocated: {allocated:.2f} GB")
    return True


def get_vram_status() -> dict:
    """Get current VRAM usage and loaded models status.

    Delegates to ModelManager when available, falls back to globals.
    """
    if model_manager is not None:
        return model_manager.get_vram_status()

    # Fallback: pre-ModelManager codepath
    import torch

    status = {
        "cuda_available": torch.cuda.is_available(),
        "models_loaded": {
            "zimage_pipeline": pipeline is not None,
            "zimage_encoder": encoder is not None,
            "qwen_image_pipeline": qwen_image_pipeline is not None,
            "qwen_image_edit": qwen_image_pipeline is not None
            and getattr(qwen_image_pipeline, "edit_pipe", None) is not None,
            "qwen_image_decompose": qwen_image_pipeline is not None
            and getattr(qwen_image_pipeline, "decompose_pipe", None) is not None,
            "qwen_image_t2i_pipeline": qwen_image_t2i_pipeline is not None,
            "ltx2_pipeline": ltx2_pipeline is not None,
            "flux2_pipeline": flux2_pipeline is not None,
        },
        "vram": None,
    }

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        status["vram"] = {
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "total_gb": round(total, 2),
            "free_gb": round(total - allocated, 2),
        }

    return status


@app.get("/")
async def index():
    """Serve the main page."""
    return FileResponse(Path(__file__).parent / "index.html")


def load_zimage_pipeline_on_demand():
    """Load Z-Image pipeline on-demand using ModelManager.

    Returns True if successfully loaded, raises exception on failure.
    Thread-safe: ModelManager handles locking internally.
    Automatically unloads other pipelines first to free VRAM.
    """
    global pipeline

    # Fast path: already loaded
    if pipeline is not None:
        return True

    if model_manager is None or runtime_config is None:
        raise ValueError("Server not initialized (model_manager or runtime_config is None)")

    try:
        model_manager.load("zimage")
        # Sync globals for backward compatibility
        pipeline = model_manager.get_pipeline("zimage")
        logger.info("[Z-Image] Pipeline loaded successfully via ModelManager")
        return True
    except Exception as e:
        logger.error(f"[Z-Image] Failed to load pipeline: {e}")
        pipeline = None
        raise


def main():
    # Register routers (deferred to avoid circular imports with web.routers.*)
    _register_routers()

    # Use shared CLI argument parser
    from llm_dit.cli import create_base_parser, load_runtime_config, setup_logging
    from llm_dit.model_manager import ModelManager

    parser = create_base_parser(
        description="Z-Image web server",
        include_server_args=True,
        include_generation_args=True,
    )

    # Add server-specific arguments
    parser.add_argument(
        "--encoder-only",
        action="store_true",
        help="Load only encoder (fast mode for Mac, no image generation)",
    )
    parser.add_argument(
        "--use-api-encoder",
        action="store_true",
        help="Use API backend for encoding (default: local encoder)",
    )
    parser.add_argument(
        "--no-preload",
        action="store_true",
        help="Don't load any models at startup (all models load on-demand)",
    )

    args = parser.parse_args()

    # Load unified config (handles TOML + CLI overrides)
    global runtime_config, pipeline, encoder, rewriter_backend, encoder_only_mode, model_manager
    runtime_config = load_runtime_config(args)
    setup_logging(runtime_config)

    # Initialize model manager
    model_manager = ModelManager(runtime_config)

    # Debug: Log all pipeline configurations
    logger.debug(f"[Config] Z-Image model: {getattr(runtime_config, 'model_path', None)}")
    logger.debug(f"[Config] FLUX.2 model: {getattr(runtime_config, 'flux2_model_path', None)}")
    logger.debug(f"[Config] FLUX.2 VAE: {getattr(runtime_config, 'flux2_vae_path', None)}")
    logger.debug(f"[Config] LTX-2 model: {getattr(runtime_config, 'ltx2_model_path', None)}")
    logger.debug(f"[Config] Debug mode: {getattr(runtime_config, 'debug', False)}")

    # Store config path for reference
    if hasattr(args, "config") and args.config:
        runtime_config.config_path = args.config
    if hasattr(args, "profile") and args.profile:
        runtime_config.current_profile = args.profile

    # Determine startup behavior from config or CLI flag
    no_preload = getattr(args, "no_preload", False)
    default_pipeline = getattr(runtime_config, "default_pipeline", "none")

    # --no-preload CLI flag overrides config
    if no_preload:
        default_pipeline = "none"

    logger.info("============================================================")
    if default_pipeline == "none":
        logger.info("SERVER STARTING IN ON-DEMAND MODE")
        logger.info("============================================================")
        logger.info("No models loaded at startup. Models will load on first request.")
        pipeline = None
        encoder = None
        encoder_only_mode = False
        mode = "on_demand"
    elif default_pipeline == "z-image":
        logger.info(f"PRELOADING PIPELINE: {default_pipeline}")
        logger.info("============================================================")
        # Validate Z-Image model path (prefer [zimage].model_path, fall back to legacy)
        zimage_path = runtime_config.zimage_model_path or runtime_config.model_path
        if not zimage_path:
            logger.error("default_pipeline='z-image' but [zimage].model_path not set in config.")
            return 1
        # Use ModelManager for Z-Image
        use_api = getattr(args, "use_api_encoder", False)
        result = model_manager.load_pipeline_from_loader(
            encoder_only=args.encoder_only, use_api=use_api
        )
        pipeline = result.pipeline
        encoder = result.encoder
        encoder_only_mode = result.mode in ("encoder_only", "api_encoder")
        mode = result.mode
    elif default_pipeline == "qwen-image":
        logger.info(f"PRELOADING PIPELINE: {default_pipeline}")
        logger.info("============================================================")
        # Validate Qwen-Image model path
        if not runtime_config.qwen_image_model_path:
            logger.error("default_pipeline='qwen-image' but qwen_image.model_path not set in config.")
            return 1
        # Use ModelManager for Qwen-Image
        result = model_manager.load_pipeline_from_loader(
            encoder_only=False, use_api=False
        )
        pipeline = result.pipeline
        encoder = result.encoder
        encoder_only_mode = False
        mode = result.mode
    elif default_pipeline == "flux2":
        logger.info(f"PRELOADING PIPELINE: {default_pipeline}")
        logger.info("============================================================")
        # FLUX.2 loads on first request - just mark it as the intended pipeline
        logger.info("FLUX.2 will load on first generation request.")
        pipeline = None
        encoder = None
        encoder_only_mode = False
        mode = "flux2_on_demand"
    elif default_pipeline == "ltx2":
        logger.info(f"PRELOADING PIPELINE: {default_pipeline}")
        logger.info("============================================================")
        # LTX-2 loads on first request - just mark it as the intended pipeline
        logger.info("LTX-2 will load on first generation request.")
        pipeline = None
        encoder = None
        encoder_only_mode = False
        mode = "ltx2_on_demand"
    else:
        logger.error(f"Unknown default_pipeline: '{default_pipeline}'. Valid options: none, z-image, qwen-image, flux2, ltx2")
        return 1

    # If loaded pipeline is QwenImageDiffusersPipeline, also set qwen_image_pipeline
    global qwen_image_pipeline
    if pipeline is not None:
        from llm_dit.pipelines.qwen_image_diffusers import QwenImageDiffusersPipeline

        if isinstance(pipeline, QwenImageDiffusersPipeline):
            qwen_image_pipeline = pipeline
            logger.info("[Qwen-Image] Pipeline loaded via ModelManager")

    # Log Qwen-Image on-demand modes
    if mode == "qwenimage-t2i_ondemand":
        logger.info("[Qwen-Image T2I] Server started in on-demand mode")
        logger.info("[Qwen-Image T2I] Pipeline will load on first generation request")
    elif mode == "qwenimage-edit_ondemand":
        logger.info("[Qwen-Image Edit] Server started in on-demand mode")
        logger.info("[Qwen-Image Edit] Pipeline will load on first edit request")

    # Initialize rewriter API backend if configured
    if runtime_config.rewriter_use_api:
        # Determine API URL: rewriter-specific or fall back to main API URL
        rewriter_url = runtime_config.rewriter_api_url or runtime_config.api_url
        if rewriter_url:
            from llm_dit.backends.api import APIBackend, APIBackendConfig

            rewriter_api_config = APIBackendConfig(
                base_url=rewriter_url,
                model_id=runtime_config.rewriter_api_model,
            )
            rewriter_backend = APIBackend(rewriter_api_config)
            logger.info(
                f"[Rewriter] API backend configured: {rewriter_url} (model: {runtime_config.rewriter_api_model})"
            )
            logger.info(
                f"[Rewriter] Defaults: temperature={runtime_config.rewriter_temperature}, top_p={runtime_config.rewriter_top_p}, max_tokens={runtime_config.rewriter_max_tokens}"
            )
        else:
            logger.warning("[Rewriter] use_api=True but no API URL configured. Using local model.")

    # Store model_manager in app.state for router access (Phase 3 migration path)
    app.state.model_manager = model_manager
    app.state.runtime_config = runtime_config

    # Run server
    import uvicorn

    # Initialize server start time and save initial config values
    global server_start_time, session_file_values
    server_start_time = time.time()
    session_file_values = runtime_config.to_dict()

    host = runtime_config.host
    port = runtime_config.port

    # Force-exit on second Ctrl-C to avoid hanging on CUDA cleanup / pinned memory
    _shutting_down = False

    def _force_exit(_signum, _frame):
        nonlocal _shutting_down
        if _shutting_down:
            logger.info("Second interrupt received, forcing exit")
            os._exit(1)
        _shutting_down = True
        logger.info("Shutting down (Ctrl-C again to force)...")
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _force_exit)
    signal.signal(signal.SIGTERM, _force_exit)

    uvicorn_kwargs: dict = {"host": host, "port": port}
    ssl_enabled = bool(runtime_config.ssl_certfile and runtime_config.ssl_keyfile)
    if ssl_enabled:
        uvicorn_kwargs["ssl_certfile"] = runtime_config.ssl_certfile
        uvicorn_kwargs["ssl_keyfile"] = runtime_config.ssl_keyfile
        if runtime_config.ssl_ca_certs:
            uvicorn_kwargs["ssl_ca_certs"] = runtime_config.ssl_ca_certs

    protocol = "https" if ssl_enabled else "http"
    logger.info(f"Starting server at {protocol}://{host}:{port} ({mode} mode)")
    uvicorn.run(app, **uvicorn_kwargs)


if __name__ == "__main__":
    main()
