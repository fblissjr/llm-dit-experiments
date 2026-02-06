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
import time
from pathlib import Path
from typing import Optional

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

# Global pipeline/encoder (loaded on startup)
# NOTE: These globals are maintained for backward compatibility with endpoint
# handlers. They are backed by ModelManager -- see main() for initialization.
# Phase 3 (router extraction) will eliminate these in favor of app.state.model_manager.
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


# =============================================================================
# Pipeline Loading Functions (used by main() startup flow)
# =============================================================================


def load_pipeline(
    model_path: str,
    text_encoder_path: Optional[str] = None,
    templates_dir: Optional[str] = None,
    encoder_device: str = "auto",
    dit_device: str = "auto",
    vae_device: str = "auto",
    quantization: str = "none",
    lora_paths: Optional[list] = None,
    lora_scales: Optional[list] = None,
    enable_compile: bool = False,
    attention_backend: str = "auto",
):
    """Load the full generation pipeline."""
    global pipeline

    from llm_dit.pipelines import ZImagePipeline

    logger.info(f"Loading pipeline from {model_path}...")
    if text_encoder_path:
        logger.info(f"  Text encoder: {text_encoder_path}")
    logger.info(f"  Encoder device: {encoder_device}")
    logger.info(f"  DiT device: {dit_device}")
    logger.info(f"  VAE device: {vae_device}")
    logger.info(f"  Quantization: {quantization}")
    logger.info(f"  Attention backend: {attention_backend}")
    logger.info(f"  Torch Compile: {enable_compile}")
    start = time.time()

    pipeline = ZImagePipeline.from_pretrained(
        model_path,
        text_encoder_path=text_encoder_path,
        templates_dir=templates_dir,
        dtype=torch.bfloat16,
        encoder_device=encoder_device,
        dit_device=dit_device,
        vae_device=vae_device,
        quantization=quantization,
        attention_backend=attention_backend,
        # Use our custom FlowMatchScheduler with FLUX-style dynamic shifting
        # This fixes the pure noise bug for Z-Image-Base model
        use_custom_scheduler=True,
    )

    load_time = time.time() - start
    logger.info(f"Pipeline loaded in {load_time:.1f}s")
    logger.info(f"Device: {pipeline.device}")

    # Apply torch.compile for faster inference (slow first run)
    if enable_compile:
        # Use compile_mode from config - "default" avoids CUDA graph issues with Z-Image's RoPE3D
        compile_mode = getattr(runtime_config, "compile_mode", "default") if runtime_config else "default"
        logger.info(f"Compiling transformer with torch.compile (mode={compile_mode})...")
        try:
            pipeline.transformer = torch.compile(pipeline.transformer, mode=compile_mode)
            logger.info("  Transformer compiled (first run will be slow)")
        except Exception as e:
            logger.warning(f"  Failed to compile: {e}")

    # Load LoRAs if configured
    if lora_paths:
        logger.info(f"Loading {len(lora_paths)} LoRA(s)...")
        scales = lora_scales if lora_scales else [1.0] * len(lora_paths)
        try:
            updated = pipeline.load_lora(lora_paths, scale=scales)
            logger.info(f"  {updated} layers updated by LoRA")
        except Exception as e:
            logger.error(f"  Failed to load LoRA: {e}")


def load_encoder_only(
    model_path: str,
    templates_dir: Optional[str] = None,
    encoder_device: str = "auto",
    quantization: str = "none",
):
    """Load only the encoder (fast mode for testing on Mac)."""
    global encoder, encoder_only_mode

    from llm_dit.encoders import ZImageTextEncoder

    logger.info(f"Loading encoder only from {model_path}...")
    logger.info(f"  Encoder device: {encoder_device}")
    logger.info(f"  Quantization: {quantization}")
    start = time.time()

    encoder = ZImageTextEncoder.from_pretrained(
        model_path,
        templates_dir=templates_dir,
        dtype=torch.bfloat16,
        device_map=encoder_device,
        quantization=quantization,
    )

    encoder_only_mode = True
    load_time = time.time() - start
    logger.info(f"Encoder loaded in {load_time:.1f}s (encoder-only mode)")
    logger.info(f"Device: {encoder.device}")


def load_api_encoder(
    api_url: str,
    model_id: str,
    templates_dir: Optional[str] = None,
):
    """Load encoder that uses heylookitsanllm API backend (encoder-only mode)."""
    global encoder, encoder_only_mode

    from llm_dit.backends.api import APIBackend, APIBackendConfig
    from llm_dit.encoders import ZImageTextEncoder
    from llm_dit.templates import TemplateRegistry

    logger.info(f"Connecting to API backend at {api_url}...")

    # Create API backend
    api_config = APIBackendConfig(
        base_url=api_url,
        model_id=model_id,
        encoding_format="base64",
    )
    backend = APIBackend(api_config)

    # Load templates if provided
    templates = None
    if templates_dir:
        templates = TemplateRegistry.from_directory(templates_dir)
        logger.info(f"Loaded {len(templates)} templates")

    # Create encoder with API backend
    encoder = ZImageTextEncoder(
        backend=backend,
        templates=templates,
    )

    encoder_only_mode = True
    logger.info(f"API encoder ready (model: {model_id})")


def load_hybrid_pipeline(
    model_path: str,
    templates_dir: Optional[str] = None,
    enable_cpu_offload: bool = False,
    enable_flash_attn: bool = False,
    enable_compile: bool = False,
    encoder_device: str = "cpu",
    dit_device: str = "cuda",
    vae_device: str = "cuda",
    lora_paths: Optional[list] = None,
    lora_scales: Optional[list] = None,
):
    """Load full pipeline with local encoder + DiT/VAE (for A/B testing vs API)."""
    global pipeline, encoder_only_mode

    from llm_dit.pipelines import ZImagePipeline

    logger.info("=" * 60)
    logger.info("HYBRID MODE SETUP (local encoder + local DiT/VAE)")
    logger.info("=" * 60)
    logger.info(f"  Model path: {model_path}")
    logger.info(f"  Templates: {templates_dir}")
    logger.info(f"  Encoder device: {encoder_device}")
    logger.info(f"  DiT device: {dit_device}")
    logger.info(f"  VAE device: {vae_device}")
    logger.info(f"  CPU Offload: {enable_cpu_offload}")
    logger.info(f"  Flash Attention: {enable_flash_attn}")
    logger.info(f"  Torch Compile: {enable_compile}")
    logger.info("-" * 60)

    start = time.time()

    # Load full pipeline with device placement
    pipeline = ZImagePipeline.from_pretrained(
        model_path,
        templates_dir=templates_dir,
        dtype=torch.bfloat16,
        encoder_device=encoder_device,
        dit_device=dit_device,
        vae_device=vae_device,
        # Use our custom FlowMatchScheduler with FLUX-style dynamic shifting
        # This fixes the pure noise bug for Z-Image-Base model
        use_custom_scheduler=True,
    )

    load_time = time.time() - start
    logger.info(f"Pipeline loaded in {load_time:.1f}s")

    # Apply optimizations
    if enable_flash_attn:
        logger.info("Enabling Flash Attention...")
        try:
            pipeline.transformer.set_attention_backend("flash")
            logger.info("  Flash Attention enabled")
        except Exception as e:
            logger.warning(f"  Failed to enable Flash Attention: {e}")
            logger.warning("  Install with: pip install flash-attn --no-build-isolation")

    if enable_compile:
        # Use compile_mode from config - "default" avoids CUDA graph issues with Z-Image's RoPE3D
        compile_mode = getattr(runtime_config, "compile_mode", "default") if runtime_config else "default"
        logger.info(f"Compiling transformer with torch.compile (mode={compile_mode})...")
        try:
            pipeline.transformer = torch.compile(pipeline.transformer, mode=compile_mode)
            logger.info("  Transformer compiled (first run will be slow)")
        except Exception as e:
            logger.warning(f"  Failed to compile: {e}")

    # Load LoRAs if configured
    if lora_paths:
        logger.info(f"Loading {len(lora_paths)} LoRA(s)...")
        scales = lora_scales if lora_scales else [1.0] * len(lora_paths)
        try:
            updated = pipeline.load_lora(lora_paths, scale=scales)
            logger.info(f"  {updated} layers updated by LoRA")
        except Exception as e:
            logger.error(f"  Failed to load LoRA: {e}")

    encoder_only_mode = False
    logger.info("-" * 60)
    logger.info(f"Hybrid pipeline ready (local encoder on {encoder_device})")
    logger.info(f"  Encoder device: {pipeline.encoder.device}")
    logger.info(f"  DiT device: {next(pipeline.transformer.parameters()).device}")
    logger.info(f"  VAE device: {next(pipeline.vae.parameters()).device}")
    logger.info("=" * 60)


def load_api_pipeline(
    api_url: str,
    model_id: str,
    model_path: str,
    templates_dir: Optional[str] = None,
    enable_cpu_offload: bool = False,
    enable_flash_attn: bool = False,
    enable_compile: bool = False,
    dit_device: str = "auto",
    vae_device: str = "auto",
    lora_paths: Optional[list] = None,
    lora_scales: Optional[list] = None,
):
    """Load full pipeline with API backend for encoding + local DiT/VAE for generation."""
    global pipeline, encoder_only_mode

    from llm_dit.backends.api import APIBackend, APIBackendConfig
    from llm_dit.encoders import ZImageTextEncoder
    from llm_dit.pipelines import ZImagePipeline
    from llm_dit.templates import TemplateRegistry

    logger.info("=" * 60)
    logger.info("DISTRIBUTED MODE SETUP")
    logger.info("=" * 60)
    logger.info(f"  API URL: {api_url}")
    logger.info(f"  API Model: {model_id}")
    logger.info(f"  Local Model: {model_path}")
    logger.info(f"  Templates: {templates_dir}")
    logger.info(f"  CPU Offload: {enable_cpu_offload}")
    logger.info(f"  Flash Attention: {enable_flash_attn}")
    logger.info(f"  Torch Compile: {enable_compile}")
    logger.info("-" * 60)

    # Create API backend for encoding
    logger.info("Creating API backend...")
    api_config = APIBackendConfig(
        base_url=api_url,
        model_id=model_id,
        encoding_format="base64",
    )
    backend = APIBackend(api_config)
    logger.info(f"  Backend created: {backend}")

    # Load templates if provided
    templates = None
    if templates_dir:
        templates = TemplateRegistry.from_directory(templates_dir)
        logger.info(f"  Loaded {len(templates)} templates")

    # Create encoder with API backend
    logger.info("Creating API-backed encoder...")
    api_encoder = ZImageTextEncoder(
        backend=backend,
        templates=templates,
    )
    logger.info(f"  Encoder created: {api_encoder}")
    logger.info(f"  Encoder device: {getattr(api_encoder, 'device', 'N/A')}")

    logger.info("-" * 60)
    logger.info(f"Loading DiT/VAE from {model_path}...")
    logger.info(f"  DiT device: {dit_device}")
    logger.info(f"  VAE device: {vae_device}")
    start = time.time()

    # Load generator-only pipeline, then attach our API encoder
    pipeline = ZImagePipeline.from_pretrained_generator_only(
        model_path,
        dtype=torch.bfloat16,
        enable_cpu_offload=enable_cpu_offload,
        dit_device=dit_device,
        vae_device=vae_device,
        # Use our custom FlowMatchScheduler with FLUX-style dynamic shifting
        # This fixes the pure noise bug for Z-Image-Base model
        use_custom_scheduler=True,
    )

    load_time = time.time() - start
    logger.info(f"  DiT/VAE loaded in {load_time:.1f}s")
    logger.info(
        f"  Transformer device: {pipeline.transformer.device if pipeline.transformer else 'None'}"
    )
    logger.info(
        f"  Transformer dtype: {next(pipeline.transformer.parameters()).dtype if pipeline.transformer else 'None'}"
    )
    logger.info(
        f"  VAE device: {next(pipeline.vae.parameters()).device if pipeline.vae else 'None'}"
    )

    # Apply optimizations
    if enable_flash_attn:
        logger.info("Enabling Flash Attention...")
        try:
            pipeline.transformer.set_attention_backend("flash")
            logger.info("  Flash Attention enabled")
        except Exception as e:
            logger.warning(f"  Failed to enable Flash Attention: {e}")
            logger.warning("  Install with: pip install flash-attn --no-build-isolation")

    if enable_compile:
        # Use compile_mode from config - "default" avoids CUDA graph issues with Z-Image's RoPE3D
        compile_mode = getattr(runtime_config, "compile_mode", "default") if runtime_config else "default"
        logger.info(f"Compiling transformer with torch.compile (mode={compile_mode})...")
        try:
            pipeline.transformer = torch.compile(pipeline.transformer, mode=compile_mode)
            logger.info("  Transformer compiled (first run will be slow)")
        except Exception as e:
            logger.warning(f"  Failed to compile: {e}")

    # Replace the encoder with our API-backed one
    logger.info("Attaching API encoder to pipeline...")
    pipeline.encoder = api_encoder

    # Load LoRAs if configured
    if lora_paths:
        logger.info(f"Loading {len(lora_paths)} LoRA(s)...")
        scales = lora_scales if lora_scales else [1.0] * len(lora_paths)
        try:
            updated = pipeline.load_lora(lora_paths, scale=scales)
            logger.info(f"  {updated} layers updated by LoRA")
        except Exception as e:
            logger.error(f"  Failed to load LoRA: {e}")

    logger.info("-" * 60)
    encoder_only_mode = False
    opts = []
    if enable_cpu_offload:
        opts.append("CPU offload")
    if enable_flash_attn:
        opts.append("Flash Attn")
    if enable_compile:
        opts.append("compiled")
    opts_str = f" ({', '.join(opts)})" if opts else ""
    logger.info(f"Pipeline ready (API encoder + local DiT/VAE{opts_str})")
    logger.info(f"  pipeline.device: {pipeline.device}")
    logger.info(f"  pipeline.dtype: {pipeline.dtype}")
    logger.info(f"  pipeline.encoder: {pipeline.encoder}")
    logger.info(f"  pipeline.transformer: {pipeline.transformer}")
    logger.info(f"  pipeline.vae: {pipeline.vae}")
    logger.info("=" * 60)


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
        result = model_manager.load("zimage")
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
    import time

    import uvicorn

    # Initialize server start time and save initial config values
    global server_start_time, session_file_values
    server_start_time = time.time()
    session_file_values = runtime_config.to_dict()

    host = runtime_config.host
    port = runtime_config.port
    logger.info(f"Starting server at http://{host}:{port} ({mode} mode)")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
