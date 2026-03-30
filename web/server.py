#!/usr/bin/env python3
"""
Multi-pipeline generation server (LTX-2, FLUX.2, Z-Image, Qwen-Image).

Usage:
    uv run web/server.py
    uv run web/server.py --port 8000
    uv run web/server.py --config config.toml --profile default
    uv run web/server.py --encoder-only  # Fast mode, no DiT/VAE
"""

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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.gzip import GZipMiddleware

logger = logging.getLogger(__name__)

app = FastAPI(title="LLM-DiT Studio")

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Compress responses >= 1KB. SSE streams (text/event-stream) are
# automatically excluded because they're sent as chunked, unbuffered.
app.add_middleware(GZipMiddleware, minimum_size=1000)


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

def create_app() -> FastAPI:
    """Create a FastAPI app with all routers registered.

    Used by the OpenAPI export script to extract the spec without running the
    full server (which requires GPU, model paths, etc.).
    """
    _register_routers()
    return app


# Serve generated outputs (videos, thumbnails) at /outputs/*
# LTX-2 returns file paths (not base64) since video files are too large to embed.
_outputs_dir = Path(__file__).resolve().parent.parent / "outputs"
if _outputs_dir.is_dir():
    app.mount("/outputs", StaticFiles(directory=str(_outputs_dir)), name="outputs")

# React frontend build directory (populated by `bun run build` in web/frontend-v2/)
_frontend_dist = Path(__file__).resolve().parent / "frontend-v2" / "dist"

# Global server state (non-pipeline lifecycle).
# Routers access these via `import web.server as srv; srv.rewriter_backend` etc.
# Pipeline state is managed exclusively by ModelManager.
rewriter_backend = None  # API backend for rewriting (if configured)
runtime_config = None  # RuntimeConfig from CLI/TOML
encoder_only_mode = False

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


@app.get("/")
async def index():
    """Serve the main page (React SPA build, or fallback message)."""
    spa_index = _frontend_dist / "index.html"
    if spa_index.is_file():
        return FileResponse(spa_index)
    return {"message": "Frontend not built. Run 'bun run build' in web/frontend-v2/ or use Vite dev server on :5175"}


def main():
    # Register routers (deferred to avoid circular imports with web.routers.*)
    _register_routers()

    from web.routers.ltx2 import cleanup_old_videos

    cleanup_old_videos(max_age_hours=24)

    # Serve React frontend build as SPA (catch-all AFTER API routes)
    if _frontend_dist.is_dir():
        app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="spa")
        logger.info(f"Serving frontend from {_frontend_dist}")

    # Use shared CLI argument parser
    from llm_dit.cli import create_base_parser, load_runtime_config, setup_logging
    from llm_dit.model_manager import ModelManager

    parser = create_base_parser(
        description="LLM-DiT generation server",
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
    global runtime_config, rewriter_backend, encoder_only_mode, model_manager
    runtime_config = load_runtime_config(args)
    setup_logging(runtime_config)

    # Enable TF32 for ~2x faster fp32/bf16 matmul on Ampere+ (RTX 4090, A100, etc.)
    import torch
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

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
        encoder_only_mode = False
        mode = result.mode
    elif default_pipeline == "flux2":
        logger.info(f"PRELOADING PIPELINE: {default_pipeline}")
        logger.info("============================================================")
        # FLUX.2 loads on first request - just mark it as the intended pipeline
        logger.info("FLUX.2 will load on first generation request.")
        encoder_only_mode = False
        mode = "flux2_on_demand"
    elif default_pipeline == "ltx2":
        logger.info(f"PRELOADING PIPELINE: {default_pipeline}")
        logger.info("============================================================")
        # LTX-2 loads on first request - just mark it as the intended pipeline
        logger.info("LTX-2 will load on first generation request.")
        encoder_only_mode = False
        mode = "ltx2_on_demand"
    else:
        logger.error(f"Unknown default_pipeline: '{default_pipeline}'. Valid options: none, z-image, qwen-image, flux2, ltx2")
        return 1

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
