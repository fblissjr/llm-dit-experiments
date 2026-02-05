#!/usr/bin/env python3
"""
Simple web server for Z-Image generation.

Usage:
    uv run web/server.py
    uv run web/server.py --port 8000
    uv run web/server.py --config config.toml --profile default
    uv run web/server.py --encoder-only  # Fast mode, no DiT/VAE
"""

import argparse
import asyncio
import base64
import binascii
import gc
import hashlib
import io
import json
import logging
import re
import time
import traceback
import zipfile
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import httpx
import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

# Basic console logging (file logging added in setup_file_logging)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def setup_file_logging(config: dict) -> None:
    """Set up file logging with rotation based on config.

    Args:
        config: Logging configuration dict with keys:
            - enabled: bool
            - log_dir: str (relative to project root)
            - log_level: str (DEBUG, INFO, WARNING, ERROR)
            - max_bytes: int (max file size before rotation)
            - backup_count: int (number of backup files to keep)
    """
    if not config.get("enabled", False):
        return

    log_dir = Path(__file__).parent.parent / config.get("log_dir", "logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "server.log"
    log_level = getattr(logging, config.get("log_level", "INFO").upper(), logging.INFO)
    max_bytes = config.get("max_bytes", 10 * 1024 * 1024)  # 10MB default
    backup_count = config.get("backup_count", 5)

    # Create rotating file handler
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    # Add to root logger so all loggers write to file
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    root_logger.setLevel(min(root_logger.level, log_level))

    logger.info(f"File logging enabled: {log_file} (max {max_bytes // 1024 // 1024}MB, {backup_count} backups)")

app = FastAPI(title="Z-Image Generator")

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files (CSS, JS)
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

# Global pipeline/encoder (loaded on startup)
pipeline = None  # Z-Image pipeline
encoder = None  # For encoder-only mode
rewriter_backend = None  # API backend for rewriting (if configured)
vl_extractor = None  # Qwen3-VL embedding extractor (if configured)
vl_rewriter = None  # Qwen3-VL instance for vision rewriting (may share with vl_extractor)
vl_embeddings_cache = {}  # Cache for extracted VL embeddings (keyed by hash)
runtime_config = None  # RuntimeConfig from CLI/TOML
encoder_only_mode = False

# Qwen-Image pipeline (separate from Z-Image)
qwen_image_pipeline = None

# Qwen-Image T2I pipeline (pure text-to-image, separate from Qwen-Image-Layered/Edit)
# Uses unified config via --model-type qwenimage-t2i --qwen-image-model-path
qwen_image_t2i_pipeline = None

# LTX-2 video generation (deprecated pipeline state)
# Note: Pure PyTorch pipeline loads/unloads components per-request
# This global is kept for backward compatibility but always remains None
ltx2_pipeline = None

# FLUX.2 Klein image generation pipeline
# Uses Qwen3 encoder with three-stage offloading for 24GB VRAM
flux2_pipeline = None

# Loading locks to prevent concurrent pipeline loading (which causes OOM)
import threading
_zimage_loading_lock = threading.Lock()
_zimage_loading_in_progress = False
_flux2_loading_lock = threading.Lock()

# In-memory history (cleared on server restart)
generation_history = []
MAX_HISTORY = 50

# Config management - session tracking
session_file_values = {}  # Original values from config file (for detecting changes)
session_modified_fields = set()  # Fields modified during this session
pending_restart_changes = {}  # Changes that require server restart
server_start_time = None  # For uptime tracking

# Fields that can be hot-reloaded without restarting
HOT_RELOAD_SAFE = {
    # Scheduler params
    "shift",
    "shift_terminal",
    "dynamic_shift",
    "d_noise",
    # Generation defaults
    "height",
    "width",
    "steps",
    "guidance_scale",
    "cfg_normalization",
    "cfg_truncation",
    "cfg_norm_mode",
    # Prompt handling
    "long_prompt_mode",
    "hidden_layer",
    "layer_weights",
    "enable_thinking",
    "default_template",
    "system_prompt",
    "thinking_content",
    "assistant_content",
    # DyPE feature params
    "dype_enabled",
    "dype_method",
    "dype_scale",
    "dype_exponent",
    "dype_start_sigma",
    "dype_base_shift",
    "dype_max_shift",
    "dype_base_resolution",
    "dype_anisotropic",
    "dype_multipass",
    "dype_pass2_strength",
    "dype_pass3_strength",
    "dype_frequency_modulation",
    # SLG feature params
    "slg_scale",
    "slg_layers",
    "slg_start",
    "slg_stop",
    # FMTT feature params
    "fmtt_scale",
    "fmtt_start",
    "fmtt_stop",
    "fmtt_normalize",
    "fmtt_decode_scale",
    "fmtt_siglip_model",
    "fmtt_siglip_device",
    # Cache settings
    "embedding_cache",
    "cache_size",
    # Tiled VAE (can change between generations)
    "tiled_vae",
    "tile_size",
    "tile_overlap",
    # Seed
    "seed",
    "negative_prompt",
}

# Fields that require server restart (model reload)
REQUIRES_RESTART = {
    # Model paths
    "model_path",
    "text_encoder_path",
    "templates_dir",
    "vl_model_path",
    "qwen_image_model_path",
    "qwen_image_edit_model_path",
    # Device placement
    "encoder_device",
    "dit_device",
    "vae_device",
    # Quantization
    "quantization",
    "dtype",
    "qwen_image_quantize_text_encoder",
    "qwen_image_quantize_transformer",
    # Memory management
    "cpu_offload",
    "qwen_image_cpu_offload",
    # Attention backend
    "attention_backend",
    "flash_attn",
    # Compilation
    "compile",
    "compile_mode",
    # LoRA (requires pipeline reload)
    "lora_paths",
    "lora_scales",
}


# =============================================================================
# Shared Response Utilities
# =============================================================================


def create_image_response(
    image=None,  # PIL Image (optional if img_b64 provided)
    pipeline_id: str = "unknown",
    seed: int | None = None,
    generation_time: float = 0.0,
    history_id: int | None = None,
    img_b64: str | None = None,  # Pre-computed base64 (avoids double-encoding)
) -> dict:
    """Create standardized JSON response for image generation endpoints.

    This shared utility ensures all image endpoints (Z-Image, FLUX.2, etc.)
    return the same format that the React frontend expects.

    Args:
        image: PIL Image object to encode (not needed if img_b64 provided)
        pipeline_id: Pipeline identifier (e.g., "zimage", "flux2")
        seed: Generation seed (or None/-1 for random)
        generation_time: Time taken in seconds
        history_id: Optional history entry ID
        img_b64: Pre-computed base64 string (skips encoding if provided)

    Returns:
        dict with: id, output_type, url, urls, thumbnail_url, seed, generation_time
    """
    # Use pre-computed base64 or encode from PIL Image
    if img_b64 is None:
        if image is None:
            raise ValueError("Either image or img_b64 must be provided")
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        img_b64 = base64.b64encode(img_bytes.getvalue()).decode("ascii")

    data_url = f"data:image/png;base64,{img_b64}"

    return {
        "id": history_id if history_id is not None else f"gen-{int(time.time() * 1000)}",
        "pipeline_id": pipeline_id,
        "output_type": "image",
        "url": data_url,
        "urls": [data_url],
        "thumbnail_url": data_url,
        "seed": seed if seed is not None else -1,
        "generation_time": generation_time,
    }


def unload_zimage_pipeline() -> bool:
    """Unload Z-Image pipeline (encoder + DiT + VAE) to free VRAM.

    Returns True if unloaded, False if not loaded.
    """
    global pipeline, encoder

    unloaded = False
    if pipeline is not None:
        # Log variant and model path before unloading
        if runtime_config is not None:
            model_path = runtime_config.zimage_model_path or runtime_config.model_path
            logger.info(f"[VRAM] Unloading Z-Image pipeline (variant={runtime_config.zimage_variant}, path={model_path})...")
        else:
            logger.info("[VRAM] Unloading Z-Image pipeline to free VRAM...")
        # Move components to CPU before deletion to release CUDA memory
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
        # Move encoder model to CPU before deletion
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
        # Clear torch.compile cache (frees ~3-4GB from compiled kernels)
        try:
            import torch._dynamo

            torch._dynamo.reset()
            logger.info("[VRAM] Cleared torch.compile cache")
        except Exception as e:
            logger.warning(f"[VRAM] Could not clear compile cache: {e}")

        # Force garbage collection before CUDA cache clear
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
        # Log VRAM after cleanup
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"[VRAM] Z-Image unloaded. CUDA allocated: {allocated:.2f} GB")

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
    """Get current VRAM usage and loaded models status."""
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


class DyPEConfigRequest(BaseModel):
    """DyPE configuration for high-resolution generation."""

    enabled: bool = False
    method: str = "vision_yarn"  # vision_yarn, yarn, ntk
    multipass: str = "twopass"  # single, twopass, threepass
    dype_scale: float = 2.0  # Magnitude of DyPE effect
    dype_exponent: float = 2.0  # Decay speed (2.0 = quadratic)
    base_shift: float = 0.5  # Shift at base resolution
    max_shift: float = 1.15  # Shift at max resolution
    pass2_strength: float = 0.5  # img2img strength for pass 2
    pass3_strength: float = 0.4  # img2img strength for pass 3
    frequency_modulation: bool = False  # Timestep-based RoPE frequency scaling (experimental)


class GenerateRequest(BaseModel):
    prompt: str  # User prompt
    negative_prompt: Optional[str] = None  # Negative prompt for CFG (only used with base model)
    system_prompt: Optional[str] = None  # System prompt (optional)
    thinking_content: Optional[str] = (
        None  # Content inside <think>...</think> (triggers think block)
    )
    assistant_content: Optional[str] = None  # Content after </think> (optional)
    force_think_block: bool = False  # If True, add empty think block even without content
    strip_quotes: bool = False  # If True, remove " characters (for JSON-type prompts)
    width: int = 1024
    height: int = 1024
    steps: int = 9
    seed: Optional[int] = None
    template: Optional[str] = None
    guidance_scale: float = 0.0
    cfg_normalization: float = 0.0  # CFG norm clamping (0 = disabled)
    cfg_truncation: float = 1.0  # CFG truncation threshold (1.0 = never)
    shift: float = 3.0  # Scheduler shift parameter
    dynamic_shift: bool = False  # Calculate shift based on resolution (overrides shift)
    d_noise: float = 1.0  # Sigma schedule scaling (<1.0 = sharper, >1.0 = softer)
    long_prompt_mode: str = "interpolate"  # truncate/interpolate/pool/attention_pool
    hidden_layer: int = -2  # Which hidden layer to extract (-1 to -35, Qwen3-4B has 36 layers)
    layer_weights: Optional[Dict[int, float]] = (
        None  # Multi-layer blending weights (overrides hidden_layer)
    )
    # DyPE (high-resolution) options
    dype: Optional[DyPEConfigRequest] = None
    # Skip Layer Guidance (SLG) options
    # None = use config defaults, explicit values override
    slg_scale: Optional[float] = None  # SLG scale (0 = disabled, 2-4 typical)
    slg_layers: Optional[List[int]] = None  # Layer indices to skip (e.g., [7, 8, 9, 10, 11, 12])
    slg_start: Optional[float] = None  # Start SLG at this fraction
    slg_stop: Optional[float] = None  # Stop SLG at this fraction
    # Flow Map Trajectory Tilting (FMTT) options
    # None = use config defaults, explicit values override
    fmtt_enabled: bool = False  # Enable FMTT (must be True for fmtt_scale to be used)
    fmtt_scale: Optional[float] = None  # FMTT scale (0 = disabled, 0.5-2.0 typical)
    fmtt_start: Optional[float] = None  # Start FMTT at this fraction
    fmtt_stop: Optional[float] = None  # Stop FMTT at this fraction
    fmtt_normalize: Optional[str] = None  # Gradient normalization mode: unit, clip, none
    fmtt_decode_scale: Optional[float] = None  # Scale for intermediate VAE decode
    fmtt_siglip_model: Optional[str] = None  # SigLIP model for FMTT
    fmtt_siglip_device: Optional[str] = None  # Device for SigLIP (cuda/cpu)
    # FBCache (Forward Block Cache) options
    fbcache: bool = False  # Enable FBCache acceleration
    fbcache_threshold: Optional[float] = None  # Override threshold (default: adaptive by sigma)
    fbcache_log: bool = False  # Log residual statistics


class Img2ImgRequest(BaseModel):
    """Request for image-to-image generation with optional differential mask.

    Note: SLG, FMTT, DyPE, and layer_weights are not supported for img2img.
    Use text-to-image (/api/generate) for those features.
    """

    prompt: str  # User prompt
    negative_prompt: Optional[str] = None  # Negative prompt for CFG (only used with base model)
    image: str  # Base64-encoded input image
    mask_image: Optional[str] = None  # Base64-encoded grayscale mask (black=preserve, white=edit)
    strength: float = Field(
        0.75, ge=0.0, le=1.0, description="Denoising strength (0=no change, 1=full generation)"
    )
    # Common generation params
    system_prompt: Optional[str] = None
    thinking_content: Optional[str] = None
    assistant_content: Optional[str] = None
    force_think_block: bool = False
    strip_quotes: bool = False
    width: Optional[int] = Field(
        None, ge=64, le=4096, description="Output width (if None, use input image size)"
    )
    height: Optional[int] = Field(
        None, ge=64, le=4096, description="Output height (if None, use input image size)"
    )
    steps: int = Field(9, ge=1, le=500, description="Number of denoising steps")
    seed: Optional[int] = None
    template: Optional[str] = None
    guidance_scale: float = Field(0.0, ge=0.0, le=30.0, description="CFG guidance scale")
    cfg_normalization: float = Field(0.0, ge=0.0, le=10.0, description="CFG normalization strength")
    cfg_truncation: float = Field(
        1.0, ge=0.0, le=1.0, description="Progress threshold for CFG truncation"
    )
    cfg_norm_mode: str = "clamp"  # CFG normalization mode: clamp or match
    shift: float = Field(3.0, ge=0.0, le=10.0, description="Scheduler shift parameter")
    dynamic_shift: bool = False  # Calculate shift based on resolution (overrides shift)
    d_noise: float = Field(
        1.0, ge=0.5, le=2.0, description="Sigma scaling (<1.0 = sharper, >1.0 = softer)"
    )
    long_prompt_mode: str = "interpolate"
    hidden_layer: int = Field(-2, ge=-35, le=-1, description="Hidden layer for text embeddings")
    # FBCache (Forward Block Cache) options
    fbcache: bool = False  # Enable FBCache acceleration
    fbcache_threshold: Optional[float] = None  # Override threshold (default: adaptive by sigma)
    fbcache_log: bool = False  # Log residual statistics


class EncodeRequest(BaseModel):
    prompt: str  # User prompt
    system_prompt: Optional[str] = None  # System prompt (optional)
    thinking_content: Optional[str] = (
        None  # Content inside <think>...</think> (triggers think block)
    )
    assistant_content: Optional[str] = None  # Content after </think> (optional)
    force_think_block: bool = False  # If True, add empty think block even without content
    strip_quotes: bool = False  # If True, remove " characters (for JSON-type prompts)
    template: Optional[str] = None


class RewriteRequest(BaseModel):
    prompt: Optional[str] = None  # User prompt to rewrite/expand (optional if image provided)
    rewriter: Optional[str] = (
        None  # Name of rewriter template (optional if custom_system_prompt provided)
    )
    custom_system_prompt: Optional[str] = None  # Ad-hoc system prompt for rewriting
    max_tokens: Optional[int] = None  # Maximum tokens to generate (default from config: 512)
    temperature: Optional[float] = None  # Sampling temperature (default: 0.6 for Qwen3 thinking)
    top_p: Optional[float] = None  # Nucleus sampling (default: 0.95)
    top_k: Optional[int] = None  # Top-k sampling (default: 20 for Qwen3)
    min_p: Optional[float] = None  # Minimum probability (default: 0.0)
    presence_penalty: Optional[float] = None  # Presence penalty (0-2, default: 0.0)
    # VL rewriter fields
    model: str = "qwen3-4b"  # "qwen3-4b" (text-only) or "qwen3-vl" (vision+text)
    image: Optional[str] = None  # Base64-encoded image (VL model only)


class VLExtractRequest(BaseModel):
    """Request to extract VL embeddings from an image."""

    image: str  # Base64-encoded image
    text: Optional[str] = None  # Optional text description with image
    hidden_layer: int = -2  # Which hidden layer to extract (-2 = penultimate)
    image_tokens_only: bool = False  # Only extract image token embeddings
    scale_to_text: bool = True  # Scale embeddings to match text statistics


class VLGenerateRequest(BaseModel):
    """Request for VL-conditioned generation."""

    prompt: str  # Text prompt
    vl_image: Optional[str] = None  # Base64-encoded reference image (optional)
    vl_embeddings_id: Optional[str] = None  # ID of pre-extracted embeddings (optional)
    vl_alpha: float = 0.3  # VL influence (0.0=text, 1.0=VL)
    vl_hidden_layer: int = -2  # Hidden layer for VL extraction
    vl_image_tokens_only: bool = False  # Only use image tokens
    vl_text: Optional[str] = None  # Text description with reference image
    vl_blend_mode: str = "linear"  # linear, style_only, graduated, attention_weighted
    # Standard generation params
    system_prompt: Optional[str] = None
    thinking_content: Optional[str] = None
    assistant_content: Optional[str] = None
    force_think_block: bool = False
    strip_quotes: bool = False
    width: int = 1024
    height: int = 1024
    steps: int = 9
    seed: Optional[int] = None
    template: Optional[str] = None
    guidance_scale: float = 0.0
    shift: float = 3.0
    dynamic_shift: bool = False  # Calculate shift based on resolution (overrides shift)
    d_noise: float = 1.0  # Sigma schedule scaling (<1.0 = sharper, >1.0 = softer)
    long_prompt_mode: str = "interpolate"
    hidden_layer: int = -2  # For text encoder


class QwenImageDecomposeRequest(BaseModel):
    """Request for Qwen-Image-Layered decomposition."""

    image: str  # Base64-encoded input image
    prompt: str  # Text description of the image
    layer_num: int = 3  # Number of decomposition layers
    resolution: int = 1024  # 640 or 1024 only
    steps: int = 30  # Number of inference steps
    cfg_scale: float = 4.0  # Classifier-free guidance scale
    seed: Optional[int] = None  # Random seed
    shift: Optional[float] = None  # Scheduler shift (auto if None)


class QwenImageEditLayerRequest(BaseModel):
    """Request for Qwen-Image layer editing (single image)."""

    layer_image: str  # Base64-encoded RGBA layer image
    instruction: str  # Text instruction for editing (e.g., "Change color to blue")
    steps: int = 40  # Number of inference steps (40 for Edit-2511)
    cfg_scale: float = 4.0  # Classifier-free guidance scale
    seed: Optional[int] = None  # Random seed


class QwenImageEditMultiRequest(BaseModel):
    """Request for Qwen-Image multi-image editing (2511 feature)."""

    images: List[str]  # Base64-encoded images (2-4 images to combine)
    instruction: str  # Text instruction for combining (e.g., "Place both subjects together")
    steps: int = 40  # Number of inference steps (40 for Edit-2511)
    cfg_scale: float = 4.0  # Classifier-free guidance scale
    seed: Optional[int] = None  # Random seed


class QwenImage2512GenerateRequest(BaseModel):
    """Request for Qwen-Image T2I text-to-image generation."""

    prompt: str  # Text prompt
    negative_prompt: Optional[str] = None  # Negative prompt (optional)
    width: int = 1024
    height: int = 1024
    steps: int = 40  # Diffusion steps
    cfg_scale: float = 4.0  # Classifier-free guidance scale
    seed: Optional[int] = None  # Random seed
    max_sequence_length: int = 512  # Max prompt tokens


class LTX2GenerateRequest(BaseModel):
    """Request for LTX-2 video generation."""

    prompt: str  # Text prompt
    negative_prompt: str = "worst quality, blurry, distorted, inconsistent motion"
    width: int = 768  # Must be multiple of 32
    height: int = 512  # Must be multiple of 32
    num_frames: int = 33  # Must be 8n+1 (9, 17, 25, 33, 41, 49...)
    fps: float = 24.0  # Output framerate
    num_inference_steps: int = 12  # Diffusion steps (12 for distilled)
    guidance_scale: float = 3.5  # CFG scale (3.0-4.0 recommended)
    seed: Optional[int] = None  # Random seed
    enable_audio: bool = False  # Generate audio alongside video
    lora_path: Optional[str] = None  # Path to LoRA weights (.safetensors)
    lora_scale: Optional[float] = None  # LoRA scale (default 0.8)


class Flux2GenerateRequest(BaseModel):
    """Request for FLUX.2 Klein image generation."""

    prompt: str  # Text prompt
    model_name: str = "klein-9b-fp8"  # Model variant
    width: int = 1024  # Image width (must be multiple of 16)
    height: int = 1024  # Image height (must be multiple of 16)
    num_steps: Optional[int] = None  # Denoising steps (4 for distilled, 50 for base)
    guidance: Optional[float] = None  # CFG scale (1.0 for distilled, 4.0 for base)
    seed: Optional[int] = None  # Random seed
    block_offload: bool = False  # Block-by-block GPU offloading for low VRAM
    model_path: Optional[str] = None  # Custom model path (overrides HuggingFace)
    vae_path: Optional[str] = None  # Custom VAE path (overrides HuggingFace)
    reference_images: Optional[List[str]] = None  # Base64 encoded reference images for editing
    match_image_size: Optional[str] = "none"  # "none" or "0 (First Image)", "1 (Second Image)", etc.
    loras: Optional[List[str]] = None  # LoRA weights ["path:scale", ...]

    # Text encoding options
    max_text_length: int = 512  # Max text tokens (512 default, increase for longer prompts)
    pad_to_max: bool = True  # Whether to pad sequences to max_text_length
    output_layers: Optional[List[int]] = None  # Which 3 Qwen3 layers to extract (default [9, 18, 27])

    @field_validator("output_layers")
    @classmethod
    def validate_output_layers(cls, v):
        if v is not None:
            if len(v) != 3:
                raise ValueError("output_layers must have exactly 3 layers")
            for layer in v:
                if not isinstance(layer, int) or layer < 0:
                    raise ValueError(f"Invalid layer index: {layer}")
        return v

    @field_validator("max_text_length")
    @classmethod
    def validate_max_text_length(cls, v):
        if v < 16 or v > 8192:
            raise ValueError("max_text_length must be between 16 and 8192")
        return v


# =============================================================================
# Z-Image Variant-Aware Defaults Helper
# =============================================================================

# Pydantic class defaults (Turbo values) - used to detect if client sent explicit values
_ZIMAGE_PYDANTIC_DEFAULTS = {
    "steps": 9,
    "guidance_scale": 0.0,
    "shift": 3.0,
}


def apply_zimage_variant_defaults(request: Union[GenerateRequest, Img2ImgRequest, "VLGenerateRequest"]) -> None:
    """Apply Z-Image variant-aware defaults to request in-place.

    The Pydantic request classes have hardcoded Turbo defaults (steps=9, shift=3.0,
    guidance_scale=0.0). When the server is running with the Base variant, we need
    to apply the correct defaults if the client didn't explicitly set them.

    This function checks if request values match Pydantic defaults (indicating the
    client didn't override them) and replaces them with variant-appropriate values.

    Args:
        request: GenerateRequest, Img2ImgRequest, or VLGenerateRequest to modify in-place

    Note:
        - Only applies when runtime_config is available and variant is "base"
        - Only modifies values that match Pydantic defaults (client didn't override)
        - Turbo variant needs no changes (Pydantic defaults are already Turbo values)
    """
    global runtime_config

    # Skip if no config or not base variant
    if runtime_config is None:
        return
    if runtime_config.zimage_variant != "base":
        return

    # Get variant defaults from constants
    from llm_dit.models.zimage.constants import get_variant_defaults

    variant_defaults = get_variant_defaults("base")

    # Apply variant defaults only if request has Pydantic defaults
    # (indicating client didn't explicitly set the value)
    if hasattr(request, "steps") and request.steps == _ZIMAGE_PYDANTIC_DEFAULTS["steps"]:
        request.steps = variant_defaults["num_inference_steps"]
        logger.debug(f"Applied variant default: steps={request.steps}")

    if hasattr(request, "guidance_scale") and request.guidance_scale == _ZIMAGE_PYDANTIC_DEFAULTS["guidance_scale"]:
        request.guidance_scale = variant_defaults["guidance_scale"]
        logger.debug(f"Applied variant default: guidance_scale={request.guidance_scale}")

    if hasattr(request, "shift") and request.shift == _ZIMAGE_PYDANTIC_DEFAULTS["shift"]:
        request.shift = variant_defaults["shift"]
        logger.debug(f"Applied variant default: shift={request.shift}")


@app.get("/")
async def index():
    """Serve the main page."""
    return FileResponse(Path(__file__).parent / "index.html")


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "ok",
        "pipeline_loaded": pipeline is not None,
        "encoder_loaded": encoder is not None,
        "encoder_only_mode": encoder_only_mode,
        "vl_available": vl_extractor is not None,
        "qwen_image_available": qwen_image_pipeline is not None,
    }


# =============================================================================
# Qwen-Image-Layered Endpoints
# =============================================================================


@app.get("/api/qwen-image/status")
async def qwen_image_status():
    """Check Qwen-Image model status and configuration."""
    if runtime_config is None:
        return {
            "available": False,
            "reason": "Runtime config not loaded",
        }

    # Check for either layered model or edit-only model
    edit_only = getattr(runtime_config, "qwen_image_edit_only", False)
    has_layered = bool(runtime_config.qwen_image_model_path)
    has_edit = bool(getattr(runtime_config, "qwen_image_edit_model_path", ""))
    configured = has_layered or (edit_only and has_edit)
    loaded = qwen_image_pipeline is not None

    # Determine which model path to show
    if edit_only and has_edit:
        model_path = runtime_config.qwen_image_edit_model_path
    else:
        model_path = runtime_config.qwen_image_model_path

    return {
        "available": loaded,
        "configured": configured,
        "edit_only": edit_only,
        "model_path": model_path if configured else None,
        "default_layer_num": runtime_config.qwen_image_layer_num if configured else 3,
        "default_cfg_scale": runtime_config.qwen_image_cfg_scale if configured else 4.0,
        "default_resolution": runtime_config.qwen_image_resolution if configured else 1024,
        "supported_resolutions": [640, 1024],
    }


@app.get("/api/qwen-image/config")
async def qwen_image_config():
    """Get Qwen-Image default parameters from server config."""
    if runtime_config is None:
        return {
            "layer_num": 3,
            "cfg_scale": 4.0,
            "resolution": 1024,
            "steps": 30,
        }
    return {
        "layer_num": runtime_config.qwen_image_layer_num,
        "cfg_scale": runtime_config.qwen_image_cfg_scale,
        "resolution": runtime_config.qwen_image_resolution,
        "steps": runtime_config.steps,  # Shared step count
    }


@app.post("/api/qwen-image/decompose")
async def qwen_image_decompose(request: QwenImageDecomposeRequest):
    """Decompose an image into multiple RGBA layers.

    Returns a ZIP file containing:
    - composite.png: The original/reconstructed composite
    - layer_1.png through layer_N.png: Decomposed RGBA layers

    The layers can be composited back together to recreate the original image.
    """
    global qwen_image_pipeline

    # Check if pipeline is loaded
    if qwen_image_pipeline is None:
        # Try to load on-demand if configured
        if runtime_config and runtime_config.qwen_image_model_path:
            logger.info("[Qwen-Image] Loading diffusers pipeline on-demand...")
            try:
                from llm_dit.pipelines.qwen_image_diffusers import QwenImageDiffusersPipeline

                qwen_image_pipeline = QwenImageDiffusersPipeline.from_pretrained(
                    runtime_config.qwen_image_model_path,
                    edit_model_path=runtime_config.qwen_image_edit_model_path or None,
                    cpu_offload=True,
                    load_edit_model=False,  # Lazy load on first edit
                )
                logger.info("[Qwen-Image] Diffusers pipeline loaded successfully")
            except Exception as e:
                logger.error(f"[Qwen-Image] Failed to load pipeline: {e}")
                raise HTTPException(
                    status_code=503, detail=f"Failed to load Qwen-Image pipeline: {e}"
                )
        else:
            raise HTTPException(
                status_code=503,
                detail="Qwen-Image pipeline not loaded. Configure qwen_image.model_path in config.",
            )

    # Validate resolution
    if request.resolution not in (640, 1024):
        raise HTTPException(
            status_code=400, detail=f"Resolution must be 640 or 1024. Got: {request.resolution}"
        )

    try:
        # Decode base64 image
        image_data = request.image
        if image_data.startswith("data:"):
            image_data = image_data.split(",", 1)[1]
        image_bytes = base64.b64decode(image_data)
        input_image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")

        logger.info("=" * 60)
        logger.info("QWEN-IMAGE DECOMPOSITION REQUEST")
        logger.info("=" * 60)
        logger.info(f"  Input size: {input_image.size}")
        logger.info(f"  Prompt: {request.prompt[:80]}...")
        logger.info(f"  Resolution: {request.resolution}x{request.resolution}")
        logger.info(f"  Layers: {request.layer_num}")
        logger.info(f"  CFG Scale: {request.cfg_scale}")
        logger.info(f"  Steps: {request.steps}")
        logger.info(f"  Seed: {request.seed}")

        start = time.time()

        # Run decomposition (QwenImageDiffusersPipeline uses resolution param)
        layers = qwen_image_pipeline.decompose(
            image=input_image,
            prompt=request.prompt,
            layer_num=request.layer_num,
            resolution=request.resolution,
            num_inference_steps=request.steps,
            cfg_scale=request.cfg_scale,
            seed=request.seed,
        )

        gen_time = time.time() - start
        logger.info(f"[Qwen-Image] Generated {len(layers)} layers in {gen_time:.1f}s")
        logger.info("=" * 60)

        # Convert layers to base64 for JSON response
        layer_data = []
        for i, layer_img in enumerate(layers):
            layer_bytes = io.BytesIO()
            layer_img.save(layer_bytes, format="PNG")
            layer_b64 = base64.b64encode(layer_bytes.getvalue()).decode("ascii")

            if i == 0:
                layer_name = "Composite"
            else:
                layer_name = f"Layer {i}"

            layer_data.append(
                {
                    "name": layer_name,
                    "image": f"data:image/png;base64,{layer_b64}",
                    "index": i,
                }
            )

        # Create ZIP file for download
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for i, layer_img in enumerate(layers):
                layer_bytes = io.BytesIO()
                layer_img.save(layer_bytes, format="PNG")
                layer_bytes.seek(0)

                if i == 0:
                    zip_name = "composite.png"
                else:
                    zip_name = f"layer_{i}.png"

                zf.writestr(zip_name, layer_bytes.getvalue())

        zip_buffer.seek(0)
        zip_b64 = base64.b64encode(zip_buffer.getvalue()).decode("ascii")

        # Store in history
        history_entry = {
            "id": len(generation_history),
            "timestamp": time.time(),
            "model_type": "qwenimage-layered",
            "prompt": request.prompt,
            "resolution": request.resolution,
            "layer_num": request.layer_num,
            "cfg_scale": request.cfg_scale,
            "steps": request.steps,
            "seed": request.seed,
            "gen_time": gen_time,
            "image_b64": layer_data[0]["image"].split(",")[1] if layer_data else "",
        }
        generation_history.insert(0, history_entry)
        if len(generation_history) > MAX_HISTORY:
            generation_history.pop()

        return {
            "layers": layer_data,
            "zip_data": zip_b64,
            "generation_time": gen_time,
            "layer_count": len(layers),
            "seed": request.seed,
            "resolution": request.resolution,
        }

    except Exception as e:
        logger.error(f"[Qwen-Image] Decomposition failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/qwen-image/edit-layer")
async def qwen_image_edit_layer(request: QwenImageEditLayerRequest):
    """Edit a decomposed layer using text instructions.

    Uses the Qwen-Image-Edit-2511 model to modify a layer based on natural language
    instructions. The edit model is loaded lazily on first use.

    Returns the edited RGBA layer as a PNG image.
    """
    global qwen_image_pipeline

    # Check if pipeline is loaded (we need the diffusers wrapper for editing)
    if qwen_image_pipeline is None:
        # Try to load on-demand if configured
        if runtime_config and runtime_config.qwen_image_model_path:
            # Unload Z-Image first to free VRAM for Qwen-Image-Edit
            if pipeline is not None or encoder is not None:
                logger.info("[VRAM] Auto-unloading Z-Image to make room for Qwen-Image-Edit...")
                unload_zimage_pipeline()

            # Get quantization settings from config
            quant_te = getattr(runtime_config, "qwen_image_quantize_text_encoder", "none")
            quant_tf = getattr(runtime_config, "qwen_image_quantize_transformer", "none")
            quant_te = quant_te if quant_te != "none" else None
            quant_tf = quant_tf if quant_tf != "none" else None

            logger.info(
                f"[Qwen-Image] Loading pipeline in edit-only mode (quantize_text_encoder={quant_te})..."
            )
            try:
                from llm_dit.pipelines.qwen_image_diffusers import QwenImageDiffusersPipeline

                qwen_image_pipeline = QwenImageDiffusersPipeline.from_pretrained(
                    runtime_config.qwen_image_model_path,
                    edit_model_path=runtime_config.qwen_image_edit_model_path or None,
                    cpu_offload=True,
                    edit_only=True,  # Skip decompose model (~12GB saved)
                    quantize_text_encoder=quant_te,
                    quantize_transformer=quant_tf,
                )
                logger.info("[Qwen-Image] Edit pipeline loaded successfully")
            except Exception as e:
                logger.error(f"[Qwen-Image] Failed to load pipeline: {e}")
                raise HTTPException(
                    status_code=503, detail=f"Failed to load Qwen-Image pipeline: {e}"
                )
        else:
            raise HTTPException(
                status_code=503,
                detail="Qwen-Image pipeline not loaded. Configure qwen_image.model_path in config.",
            )

    # Check if pipeline has edit capability
    if not hasattr(qwen_image_pipeline, "edit_layer"):
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
        edited_layer = qwen_image_pipeline.edit_layer(
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


@app.get("/api/qwen-image/edit-status")
async def qwen_image_edit_status():
    """Check if the edit model is loaded and ready."""
    if qwen_image_pipeline is None:
        return {
            "available": False,
            "reason": "Pipeline not loaded",
        }

    has_edit_method = hasattr(qwen_image_pipeline, "edit_layer")
    has_edit_pipe = (
        hasattr(qwen_image_pipeline, "has_edit_model") and qwen_image_pipeline.has_edit_model
    )

    return {
        "available": has_edit_method,
        "edit_model_loaded": has_edit_pipe,
        "edit_model_path": getattr(qwen_image_pipeline, "_edit_model_path", None),
        "supports_multi_image": hasattr(qwen_image_pipeline, "edit_multi"),
    }


@app.post("/api/qwen-image/edit-multi")
async def qwen_image_edit_multi(request: QwenImageEditMultiRequest):
    """Combine multiple images using Qwen-Image-Edit-2511.

    New capability in Edit-2511 for multi-person consistency and creative
    image merging. Supports combining 2+ images into a single coherent output.

    Returns the combined output as a PNG image.
    """
    global qwen_image_pipeline

    # Validate input
    if len(request.images) < 2:
        raise HTTPException(
            status_code=400,
            detail=f"edit-multi requires at least 2 images, got {len(request.images)}. "
            "For single-image editing, use /api/qwen-image/edit-layer instead.",
        )

    # Check if pipeline is loaded
    if qwen_image_pipeline is None:
        if runtime_config and runtime_config.qwen_image_model_path:
            # Unload Z-Image first to free VRAM for Qwen-Image-Edit
            if pipeline is not None or encoder is not None:
                logger.info("[VRAM] Auto-unloading Z-Image to make room for Qwen-Image-Edit...")
                unload_zimage_pipeline()

            # Get quantization settings from config
            quant_te = getattr(runtime_config, "qwen_image_quantize_text_encoder", "none")
            quant_tf = getattr(runtime_config, "qwen_image_quantize_transformer", "none")
            quant_te = quant_te if quant_te != "none" else None
            quant_tf = quant_tf if quant_tf != "none" else None

            logger.info(
                f"[Qwen-Image] Loading pipeline in edit-only mode (quantize_text_encoder={quant_te})..."
            )
            try:
                from llm_dit.pipelines.qwen_image_diffusers import QwenImageDiffusersPipeline

                qwen_image_pipeline = QwenImageDiffusersPipeline.from_pretrained(
                    runtime_config.qwen_image_model_path,
                    edit_model_path=runtime_config.qwen_image_edit_model_path or None,
                    cpu_offload=True,
                    edit_only=True,  # Skip decompose model (~12GB saved)
                    quantize_text_encoder=quant_te,
                    quantize_transformer=quant_tf,
                )
            except Exception as e:
                logger.error(f"[Qwen-Image] Failed to load pipeline: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Failed to load Qwen-Image pipeline: {e}"
                )
        else:
            raise HTTPException(
                status_code=400,
                detail="Qwen-Image pipeline not loaded and no model path configured. "
                "Set qwen_image.model_path in config.toml.",
            )

    # Check if pipeline supports multi-image editing
    if not hasattr(qwen_image_pipeline, "edit_multi"):
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
        combined_image = qwen_image_pipeline.edit_multi(
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


def _is_t2i_configured() -> bool:
    """Check if T2I is configured via unified config."""
    if runtime_config is None:
        return False
    # T2I uses the unified qwen_image_model_path when model_type is qwenimage-t2i
    return (
        bool(runtime_config.qwen_image_model_path) and runtime_config.model_type == "qwenimage-t2i"
    )


@app.get("/api/qwen-image-2512/status")
async def qwen_image_2512_status():
    """Check Qwen-Image T2I pipeline status.

    Note: Uses unified config (--model-type qwenimage-t2i --qwen-image-model-path).
    """
    configured = _is_t2i_configured()
    loaded = qwen_image_t2i_pipeline is not None

    # T2I defaults: steps=40, resolution=1024, quantize_transformer=fp8
    return {
        "available": loaded,
        "configured": configured,
        "model_path": runtime_config.qwen_image_model_path if runtime_config else None,
        "quantize_transformer": runtime_config.get_qwen_image_quantize_transformer()
        if runtime_config
        else "fp8",
        "quantize_text_encoder": runtime_config.qwen_image_quantize_text_encoder
        if runtime_config
        else "none",
    }


@app.get("/api/qwen-image-2512/config")
async def qwen_image_2512_config():
    """Get Qwen-Image T2I configuration and defaults.

    Note: Uses unified config (--model-type qwenimage-t2i).
    Variant-aware defaults: T2I uses 40 steps, 1024 resolution, fp8 quantization.
    """
    # T2I-specific defaults (steps=40, resolution=1024, quantize_transformer=fp8)
    return {
        "model_path": runtime_config.qwen_image_model_path if runtime_config else "",
        "steps": runtime_config.get_qwen_image_steps() if runtime_config else 40,
        "cfg_scale": runtime_config.qwen_image_cfg_scale if runtime_config else 4.0,
        "quantize_transformer": runtime_config.get_qwen_image_quantize_transformer()
        if runtime_config
        else "fp8",
        "quantize_text_encoder": runtime_config.qwen_image_quantize_text_encoder
        if runtime_config
        else "none",
        "default_width": 1024,
        "default_height": 1024,
        "max_sequence_length": 512,
    }


@app.post("/api/qwen-image-2512/generate")
async def qwen_image_2512_generate(request: QwenImage2512GenerateRequest):
    """Generate an image using Qwen-Image T2I (pure text-to-image).

    Uses unified config: --model-type qwenimage-t2i --qwen-image-model-path
    Variant-aware defaults: T2I uses 40 steps, 1024 resolution, fp8 quantization.
    """
    global qwen_image_t2i_pipeline

    # Check if pipeline is loaded, load on-demand if needed
    if qwen_image_t2i_pipeline is None:
        if runtime_config and runtime_config.qwen_image_model_path:
            logger.info("[Qwen-Image T2I] Loading pipeline on-demand...")
            try:
                from llm_dit.pipelines import QwenImage2512Pipeline

                # Get quantization settings from unified config (variant-aware)
                quant_transformer = runtime_config.get_qwen_image_quantize_transformer()
                if quant_transformer == "none":
                    quant_transformer = None

                quant_text_encoder = runtime_config.qwen_image_quantize_text_encoder
                if quant_text_encoder == "none":
                    quant_text_encoder = None

                qwen_image_t2i_pipeline = QwenImage2512Pipeline.from_pretrained(
                    runtime_config.qwen_image_model_path,
                    quantize_transformer=quant_transformer,
                    quantize_text_encoder=quant_text_encoder,
                    cpu_offload=runtime_config.qwen_image_cpu_offload,
                )
                logger.info("[Qwen-Image T2I] Pipeline loaded successfully")
            except Exception as e:
                logger.error(f"[Qwen-Image T2I] Failed to load pipeline: {e}")
                traceback.print_exc()
                raise HTTPException(
                    status_code=503, detail=f"Failed to load Qwen-Image T2I pipeline: {e}"
                )
        else:
            raise HTTPException(
                status_code=503,
                detail="Qwen-Image T2I not configured. Use --model-type qwenimage-t2i --qwen-image-model-path",
            )

    try:
        logger.info("=" * 60)
        logger.info("QWEN-IMAGE T2I GENERATION REQUEST")
        logger.info("=" * 60)
        logger.info(f"  Prompt: {request.prompt[:80]}...")
        logger.info(f"  Size: {request.width}x{request.height}")
        logger.info(f"  Steps: {request.steps}")
        logger.info(f"  CFG Scale: {request.cfg_scale}")
        logger.info(f"  Seed: {request.seed}")
        logger.info("=" * 60)

        start = time.time()

        # Generate image
        image = qwen_image_t2i_pipeline(
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

        # Add to history
        img_b64 = base64.b64encode(img_bytes.getvalue()).decode("ascii")
        history_entry = {
            "id": len(generation_history),
            "timestamp": time.time(),
            "model_type": "qwenimage-t2i",
            "prompt": request.prompt,
            "width": request.width,
            "height": request.height,
            "steps": request.steps,
            "cfg_scale": request.cfg_scale,
            "seed": request.seed,
            "generation_time": gen_time,
            "image": f"data:image/png;base64,{img_b64}",
        }
        generation_history.append(history_entry)
        if len(generation_history) > MAX_HISTORY:
            generation_history.pop(0)

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


# =============================================================================
# Vision Conditioning (Qwen3-VL) Endpoints
# =============================================================================


@app.get("/api/vl/status")
async def vl_status():
    """Check VL conditioning status and configuration."""
    if runtime_config is None:
        return {
            "available": False,
            "reason": "Runtime config not loaded",
        }

    vl_configured = bool(runtime_config.vl_model_path)
    vl_loaded = vl_extractor is not None

    return {
        "available": vl_loaded,
        "configured": vl_configured,
        "model_path": runtime_config.vl_model_path if vl_configured else None,
        "device": runtime_config.vl_device if vl_configured else None,
        "default_alpha": runtime_config.vl_alpha if vl_configured else 0.3,
        "default_hidden_layer": runtime_config.vl_hidden_layer if vl_configured else -2,
        "blend_modes": [
            "interpolate",  # RECOMMENDED: compresses all VL tokens
            "adain_per_dim",  # Best for style transfer
            "adain",  # Transfer VL statistics to text
            "linear",  # WARNING: truncates, loses most VL info
            "style_only",  # Blend only style dimensions
            "graduated",  # Graduated alpha per token
            "attention_weighted",  # Falls back to interpolate
        ],
        "cached_embeddings": list(vl_embeddings_cache.keys()),
    }


@app.get("/api/vl/config")
async def vl_config():
    """Get VL conditioning default parameters from server config."""
    if runtime_config is None:
        return {
            "alpha": 0.3,
            "hidden_layer": -2,
            "auto_unload": True,
            "blend_mode": "linear",
        }
    return {
        "alpha": runtime_config.vl_alpha,
        "hidden_layer": runtime_config.vl_hidden_layer,
        "auto_unload": runtime_config.vl_auto_unload,
        "blend_mode": runtime_config.vl_blend_mode,
    }


@app.post("/api/vl/extract")
async def vl_extract(request: VLExtractRequest):
    """Extract VL embeddings from an uploaded image.

    Returns an embeddings ID that can be used with /api/vl/generate.
    This allows pre-extracting embeddings and reusing them across generations.
    """
    if vl_extractor is None:
        raise HTTPException(
            status_code=503, detail="VL extractor not loaded. Configure vl.model_path in config."
        )

    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        logger.info(f"[VL] Extracting embeddings from {image.size[0]}x{image.size[1]} image")
        logger.info(
            f"[VL] hidden_layer={request.hidden_layer}, image_tokens_only={request.image_tokens_only}"
        )

        start = time.time()

        # Extract embeddings
        result = vl_extractor.extract(
            image=image,
            text=request.text,
            hidden_layer=request.hidden_layer,
            image_tokens_only=request.image_tokens_only,
            scale_to_text=request.scale_to_text,
        )

        extract_time = time.time() - start

        # Generate cache ID
        image_hash = hashlib.md5(image_data).hexdigest()[:8]
        text_hash = hashlib.md5((request.text or "").encode()).hexdigest()[:4]
        cache_id = f"vl_{image_hash}_{text_hash}_L{request.hidden_layer}"

        # Cache the embeddings
        vl_embeddings_cache[cache_id] = {
            "embeddings": result.embeddings,
            "num_tokens": result.num_tokens,
            "hidden_layer": result.hidden_layer,
            "original_std": result.original_std,
            "scaled_std": result.scaled_std,
            "text": request.text,
            "timestamp": time.time(),
        }

        logger.info(
            f"[VL] Extracted {result.num_tokens} tokens in {extract_time:.2f}s -> {cache_id}"
        )

        return {
            "embeddings_id": cache_id,
            "num_tokens": result.num_tokens,
            "shape": list(result.embeddings.shape),
            "hidden_layer": result.hidden_layer,
            "original_std": result.original_std,
            "scaled_std": result.scaled_std,
            "extract_time": extract_time,
        }

    except Exception as e:
        logger.error(f"[VL] Extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/vl/generate")
async def vl_generate(request: VLGenerateRequest):
    """Generate an image with VL conditioning.

    This endpoint supports three modes:
    1. vl_image provided: Extract VL embeddings on-the-fly and generate
    2. vl_embeddings_id provided: Use pre-extracted embeddings
    3. Neither provided: Falls back to standard text-only generation
    """
    # Apply variant-aware defaults (Base vs Turbo)
    apply_zimage_variant_defaults(request)

    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")

    # Get VL embeddings
    vl_emb = None

    if request.vl_embeddings_id:
        # Use cached embeddings
        cached = vl_embeddings_cache.get(request.vl_embeddings_id)
        if cached is None:
            raise HTTPException(
                status_code=404, detail=f"Embeddings not found: {request.vl_embeddings_id}"
            )
        vl_emb = cached["embeddings"]
        logger.info(f"[VL] Using cached embeddings: {request.vl_embeddings_id}")

    elif request.vl_image:
        # Extract on-the-fly
        if vl_extractor is None:
            raise HTTPException(
                status_code=503,
                detail="VL extractor not loaded. Configure vl.model_path in config.",
            )

        try:
            image_data = base64.b64decode(request.vl_image)
            image = Image.open(io.BytesIO(image_data)).convert("RGB")

            logger.info(
                f"[VL] Extracting embeddings on-the-fly from {image.size[0]}x{image.size[1]} image"
            )

            result = vl_extractor.extract(
                image=image,
                text=request.vl_text,
                hidden_layer=request.vl_hidden_layer,
                image_tokens_only=request.vl_image_tokens_only,
            )
            vl_emb = result.embeddings

        except Exception as e:
            logger.error(f"[VL] On-the-fly extraction failed: {e}")
            raise HTTPException(status_code=500, detail=f"VL extraction failed: {e}")

    try:
        logger.info("=" * 60)
        logger.info("VL-CONDITIONED GENERATION REQUEST")
        logger.info("=" * 60)
        logger.info(f"  Prompt: {request.prompt[:80]}...")
        logger.info(f"  VL alpha: {request.vl_alpha}")
        logger.info(f"  VL blend mode: {request.vl_blend_mode}")
        logger.info(f"  Size: {request.width}x{request.height}")
        logger.info(f"  Steps: {request.steps}")

        # Encode text prompt
        enc = encoder if encoder is not None else (pipeline.encoder if pipeline else None)
        if enc is None:
            raise HTTPException(status_code=503, detail="Encoder not loaded")

        text_output = enc.encode(
            request.prompt,
            template=request.template,
            system_prompt=request.system_prompt,
            thinking_content=request.thinking_content,
            assistant_content=request.assistant_content,
            force_think_block=request.force_think_block,
            remove_quotes=request.strip_quotes,
            long_prompt_mode=request.long_prompt_mode,
            hidden_layer=request.hidden_layer,
        )
        text_emb = text_output.embeddings[0]

        # Blend VL and text embeddings
        if vl_emb is not None and request.vl_alpha > 0:
            from llm_dit.vl import (
                blend_adain,
                blend_adain_per_dim,
                blend_embeddings,
                blend_interpolate,
                blend_per_token,
                blend_style_only,
                create_graduated_alpha,
            )

            if request.vl_blend_mode == "linear":
                # WARNING: truncates VL to text length, losing most image info
                blended = blend_embeddings(vl_emb, text_emb, request.vl_alpha)
            elif request.vl_blend_mode == "interpolate":
                # RECOMMENDED: compresses all VL tokens via interpolation
                blended = blend_interpolate(vl_emb, text_emb, request.vl_alpha)
            elif request.vl_blend_mode == "adain":
                # Transfer VL statistics (mean/std) to text structure
                blended = blend_adain(text_emb, vl_emb, request.vl_alpha)
            elif request.vl_blend_mode == "adain_per_dim":
                # Per-dimension AdaIN - best for style transfer
                blended = blend_adain_per_dim(text_emb, vl_emb, request.vl_alpha)
            elif request.vl_blend_mode == "style_only":
                blended = blend_style_only(vl_emb, text_emb, request.vl_alpha)
            elif request.vl_blend_mode == "graduated":
                seq_len = min(vl_emb.shape[0], text_emb.shape[0])
                token_alphas = create_graduated_alpha(seq_len, 0.0, request.vl_alpha * 2)
                blended = blend_per_token(vl_emb, text_emb, token_alphas)
            elif request.vl_blend_mode == "attention_weighted":
                # For now, fall back to interpolate (attention weights not yet available)
                blended = blend_interpolate(vl_emb, text_emb, request.vl_alpha)
            else:
                # Default to interpolate (recommended)
                blended = blend_interpolate(vl_emb, text_emb, request.vl_alpha)

            logger.info(f"[VL] Blended embeddings: shape={blended.shape}, std={blended.std():.2f}")
            prompt_embeds = blended.unsqueeze(0)  # Add batch dim
        else:
            prompt_embeds = text_emb.unsqueeze(0)

        # Set up generator
        generator = None
        if request.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(request.seed)

        start = time.time()

        # Generate using prompt_embeds directly
        image = pipeline(
            prompt_embeds=prompt_embeds,
            height=request.height,
            width=request.width,
            num_inference_steps=request.steps,
            guidance_scale=request.guidance_scale,
            cfg_normalization=request.cfg_normalization,
            cfg_truncation=request.cfg_truncation,
            shift=None if request.dynamic_shift else request.shift,
            d_noise=request.d_noise,
            generator=generator,
        )

        gen_time = time.time() - start
        logger.info(f"[VL] Generated in {gen_time:.1f}s")
        logger.info("=" * 60)

        # Convert to PNG bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        # Store in history with VL info
        img_bytes_copy = io.BytesIO()
        image.save(img_bytes_copy, format="PNG")
        img_b64 = base64.b64encode(img_bytes_copy.getvalue()).decode("ascii")

        history_entry = {
            "id": len(generation_history),
            "timestamp": time.time(),
            "model_type": "zimage",  # Z-Image with VL conditioning
            "prompt": request.prompt,
            "vl_alpha": request.vl_alpha,
            "vl_blend_mode": request.vl_blend_mode,
            "vl_embeddings_id": request.vl_embeddings_id,
            "width": request.width,
            "height": request.height,
            "steps": request.steps,
            "seed": request.seed,
            "gen_time": gen_time,
            "image_b64": img_b64,
        }
        generation_history.insert(0, history_entry)
        if len(generation_history) > MAX_HISTORY:
            generation_history.pop()

        return StreamingResponse(
            img_bytes,
            media_type="image/png",
            headers={
                "X-Generation-Time": str(gen_time),
                "X-Seed": str(request.seed) if request.seed else "random",
                "X-VL-Alpha": str(request.vl_alpha),
                "X-VL-Blend-Mode": request.vl_blend_mode,
            },
        )

    except Exception as e:
        logger.error(f"[VL] Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/vl/cache/{embeddings_id}")
async def vl_clear_cache_entry(embeddings_id: str):
    """Clear a specific cached VL embedding."""
    if embeddings_id in vl_embeddings_cache:
        del vl_embeddings_cache[embeddings_id]
        return {"deleted": embeddings_id}
    raise HTTPException(status_code=404, detail=f"Embeddings not found: {embeddings_id}")


@app.delete("/api/vl/cache")
async def vl_clear_cache():
    """Clear all cached VL embeddings."""
    global vl_embeddings_cache
    count = len(vl_embeddings_cache)
    vl_embeddings_cache = {}
    return {"cleared": count}


# =====================================================================
# DyPE (High-Resolution) Endpoints
# =====================================================================


@app.get("/api/dype/config")
async def dype_config():
    """Get DyPE configuration defaults from server config.

    Returns default DyPE settings for high-resolution generation.
    """
    if runtime_config is None:
        return {
            "enabled": False,
            "method": "vision_yarn",
            "dype_scale": 2.0,
            "dype_exponent": 2.0,
            "dype_start_sigma": 1.0,
            "base_shift": 0.5,
            "max_shift": 1.15,
            "base_resolution": 1024,
            "anisotropic": False,
            "multipass_recommended_threshold": 2048,
        }

    # Get DyPE config from runtime config if available
    dype = getattr(runtime_config, "dype", None)
    if dype is not None:
        return {
            "enabled": dype.enabled,
            "method": dype.method,
            "dype_scale": dype.dype_scale,
            "dype_exponent": dype.dype_exponent,
            "dype_start_sigma": dype.dype_start_sigma,
            "base_shift": dype.base_shift,
            "max_shift": dype.max_shift,
            "base_resolution": dype.base_resolution,
            "anisotropic": dype.anisotropic,
            "multipass_recommended_threshold": 2048,
        }

    return {
        "enabled": False,
        "method": "vision_yarn",
        "dype_scale": 2.0,
        "dype_exponent": 2.0,
        "dype_start_sigma": 1.0,
        "base_shift": 0.5,
        "max_shift": 1.15,
        "base_resolution": 1024,
        "anisotropic": False,
        "multipass_recommended_threshold": 2048,
    }


@app.get("/api/dype/status")
async def dype_status():
    """Get DyPE feature status and recommendations.

    Returns whether DyPE is recommended for the current pipeline
    and suggested settings based on target resolution.
    """
    pipeline_supports_dype = pipeline is not None

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


# =====================================================================
# End DyPE Endpoints
# =====================================================================


# =============================================================================
# LTX-2 Video Generation Endpoints
# =============================================================================

# Video output directory
VIDEO_OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "videos"
VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


async def cleanup_old_videos(max_age_hours: int = 24) -> int:
    """Delete videos older than max_age_hours.

    Called on startup to prevent unbounded storage growth.
    Returns count of deleted files.
    """
    import time

    max_age_seconds = max_age_hours * 3600
    now = time.time()
    deleted_count = 0

    if VIDEO_OUTPUT_DIR.exists():
        for video_file in VIDEO_OUTPUT_DIR.glob("*.mp4"):
            try:
                age = now - video_file.stat().st_mtime
                if age > max_age_seconds:
                    video_file.unlink()
                    deleted_count += 1
                    # Also delete thumbnail if exists
                    thumb = video_file.with_suffix(".png")
                    if thumb.exists():
                        thumb.unlink()
            except OSError as e:
                logger.warning(f"Failed to delete old video {video_file}: {e}")

    if deleted_count > 0:
        logger.info(f"[Cleanup] Deleted {deleted_count} videos older than {max_age_hours}h")

    return deleted_count


@app.on_event("startup")
async def startup_video_cleanup():
    """Clean up old videos on server startup."""
    await cleanup_old_videos(max_age_hours=24)


def get_ltx2_model_path() -> Path:
    """Get validated LTX-2 model path from config or default location.

    Returns:
        Path to LTX-2 model directory

    Raises:
        ValueError if model path not found or not configured
    """
    # Try to load from config if available
    config_path = getattr(runtime_config, "config_path", None) if runtime_config else None
    profile = (
        getattr(runtime_config, "current_profile", "default") if runtime_config else "default"
    )

    if config_path:
        from llm_dit.config import Config

        config = Config.load(config_path, profile=profile)
        if config.ltx2 and config.ltx2.model_path:
            model_path = Path(config.ltx2.model_path).expanduser()
            if model_path.exists():
                return model_path
            raise ValueError(f"LTX-2 model path not found: {model_path}")
        else:
            raise ValueError(
                "LTX-2 not configured. Set ltx2.model_path in config.toml "
                f"under [{profile}.ltx2] section."
            )

    # Fallback: Try default path
    default_path = Path.home() / "Storage" / "LTX-2"
    if default_path.exists():
        return default_path

    raise ValueError(
        f"LTX-2 model not found at {default_path}. "
        "Configure ltx2.model_path in config.toml."
    )


def save_ltx2_video(video: torch.Tensor, path: str, fps: float = 24.0) -> str:
    """Save LTX-2 video tensor to file.

    Args:
        video: Video frames [F, H, W, C] in uint8 format
        path: Output path (.mp4)
        fps: Frame rate

    Returns:
        Path to saved video
    """
    # Convert tensor to numpy
    video_np = video.cpu().numpy()

    try:
        import imageio.v3 as iio

        codec = "libvpx-vp9" if path.endswith(".webm") else "libx264"
        with iio.imopen(path, "w", plugin="FFMPEG") as writer:
            writer.write(video_np, fps=fps, codec=codec)
        logger.info(f"[LTX-2] Saved video to {path}")
    except Exception as e:
        logger.warning(f"[LTX-2] imageio failed: {e}, trying torchvision")
        import torchvision.io as tvio

        tvio.write_video(path, video, fps=fps)
        logger.info(f"[LTX-2] Saved video to {path}")

    return path


@app.get("/api/ltx2/status")
async def ltx2_status():
    """Get LTX-2 pipeline status.

    Returns availability, loaded state, and VRAM usage.
    """
    # Check if LTX-2 config exists in the loaded config file
    ltx2_configured = False

    config_path = getattr(runtime_config, "config_path", None) if runtime_config else None
    profile = getattr(runtime_config, "current_profile", "default") if runtime_config else "default"

    if config_path:
        try:
            from llm_dit.config import Config

            config = Config.load(config_path, profile=profile)
            if config.ltx2 and config.ltx2.model_path:
                # Check if path actually exists
                model_dir = Path(config.ltx2.model_path).expanduser()
                ltx2_configured = model_dir.exists()
        except Exception:
            pass

    # Check default path if not configured
    if not ltx2_configured:
        default_path = Path.home() / "Storage" / "LTX-2"
        ltx2_configured = default_path.exists()

    return {
        "available": ltx2_configured,
        # Note: Pure PyTorch pipeline loads/unloads components per request
        # so "loaded" always returns False (no persistent state)
        "loaded": False,
        "vram_used_gb": None,  # TODO: Track actual VRAM usage per model
    }


@app.post("/api/ltx2/generate/stream")
async def ltx2_generate_stream(request: LTX2GenerateRequest):
    """Generate video with SSE progress streaming.

    Returns Server-Sent Events with progress updates during generation,
    then final result with video URL.

    Uses pure PyTorch generation via generate_video_with_offloading().
    """

    async def generate() -> AsyncIterator[str]:
        """Async generator for SSE events."""
        try:
            # Yield initial status
            yield f"data: {json.dumps({'type': 'status', 'message': 'Validating LTX-2 configuration...'})}\n\n"

            # Validate model path exists
            model_path = await asyncio.get_event_loop().run_in_executor(
                None, get_ltx2_model_path
            )

            yield f"data: {json.dumps({'type': 'status', 'message': 'Starting generation...'})}\n\n"

            # Progress tracking
            progress_state = {"stage": "", "step": 0, "total": request.num_inference_steps}

            def progress_callback(stage: str, step: int, total: int) -> None:
                """Callback to track progress (can't yield directly from here)."""
                progress_state["stage"] = stage
                progress_state["step"] = step
                progress_state["total"] = total

            # Generate video (blocking)
            seed = request.seed if request.seed is not None else int(time.time()) % (2**32)

            start_time = time.time()

            # Run generation in thread pool to not block event loop
            def do_generate():
                from llm_dit.pipelines import generate_video_with_offloading, GenerationConfig

                # Create generation config from request
                gen_config = GenerationConfig(
                    num_frames=request.num_frames,
                    height=request.height,
                    width=request.width,
                    num_inference_steps=request.num_inference_steps,
                    guidance_scale=request.guidance_scale,
                    seed=seed,
                )

                # Generate video with component offloading
                return generate_video_with_offloading(
                    prompt=request.prompt,
                    config=gen_config,
                    model_path=model_path,
                    callback=progress_callback,
                    use_progress=False,  # Disable tqdm, use callback instead
                    lora_path=request.lora_path,
                    lora_scale=request.lora_scale,
                )

            # Start generation in background
            loop = asyncio.get_event_loop()
            gen_task = loop.run_in_executor(None, do_generate)

            # Poll progress while generating
            while not gen_task.done():
                await asyncio.sleep(0.5)
                stage = progress_state["stage"]
                step = progress_state["step"]
                total = progress_state["total"]
                elapsed = time.time() - start_time

                if stage and step > 0:
                    eta = (elapsed / step) * (total - step) if total > 0 else 0
                    its = step / elapsed if elapsed > 0 else 0
                    yield f"data: {json.dumps({'type': 'progress', 'stage': stage, 'step': step, 'total': total, 'elapsed': round(elapsed, 1), 'eta': round(eta, 1), 'its': round(its, 2)})}\n\n"

            # Get result (video tensor [F, H, W, C] uint8)
            video = await gen_task
            generation_time = time.time() - start_time

            yield f"data: {json.dumps({'type': 'status', 'message': 'Saving video...'})}\n\n"

            # Generate unique filename
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            hash_suffix = hashlib.md5(f"{request.prompt}{seed}".encode()).hexdigest()[:8]
            video_filename = f"video_{timestamp}_{hash_suffix}.mp4"
            video_path = VIDEO_OUTPUT_DIR / video_filename

            # Save video
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: save_ltx2_video(video, str(video_path), fps=request.fps)
            )

            # Generate thumbnail (first frame)
            thumb_filename = f"thumb_{timestamp}_{hash_suffix}.png"
            thumb_path = VIDEO_OUTPUT_DIR / thumb_filename

            try:
                # video is [F, H, W, C] uint8 tensor
                first_frame = video[0].cpu().numpy()
                Image.fromarray(first_frame).save(str(thumb_path))
            except Exception as e:
                logger.warning(f"Failed to save thumbnail: {e}")
                thumb_filename = None

            # Return final result
            # Note: Audio generation not yet implemented in pure PyTorch pipeline
            result = {
                "type": "complete",
                "video_url": f"/outputs/videos/{video_filename}",
                "thumbnail_url": f"/outputs/videos/{thumb_filename}" if thumb_filename else None,
                "seed": seed,
                "generation_time": round(generation_time, 1),
                "num_frames": request.num_frames,
                "fps": request.fps,
                "has_audio": False,  # Audio not yet implemented in pure PyTorch pipeline
            }
            yield f"data: {json.dumps(result)}\n\n"

        except Exception as e:
            logger.error(f"[LTX-2] Generation failed: {e}")
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


# Serve video files
app.mount("/outputs/videos", StaticFiles(directory=VIDEO_OUTPUT_DIR), name="videos")


# =============================================================================
# FLUX.2 Klein Image Generation Endpoints
# =============================================================================


@app.get("/api/flux2/status")
async def flux2_status():
    """Get FLUX.2 Klein pipeline status.

    Returns availability info. FLUX.2 is always available (downloads from HuggingFace).
    """
    return {
        "available": True,  # FLUX.2 downloads models from HuggingFace as needed
        "loaded": flux2_pipeline is not None,
        "supported_models": [
            "klein-9b", "klein-9b-fp8", "klein-4b", "klein-4b-fp8",
            "klein-base-9b", "klein-base-9b-fp8", "klein-base-4b", "klein-base-4b-fp8"
        ],
    }


@app.post("/api/flux2/generate")
async def flux2_generate(request: Flux2GenerateRequest):
    """Generate image using FLUX.2 Klein.

    Supports both text-to-image and image editing with reference images.
    Returns PNG image as binary response.
    """
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
        if runtime_config:
            offload_between_stages = getattr(runtime_config, "flux2_offload_between_stages", True)

        # Create generation config
        config = Flux2GenerationConfig(
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

        if not model_path and runtime_config:
            model_path = getattr(runtime_config, "flux2_model_path", None)
        if not vae_path and runtime_config:
            vae_path = getattr(runtime_config, "flux2_vae_path", None)

        # Generate image
        start_time = time.time()
        logger.info(f"[FLUX.2] Generating {request.width}x{request.height} with {request.model_name}")
        if model_path:
            logger.info(f"[FLUX.2] Using model path: {model_path}")

        # Pass persistent models if pipeline is preloaded
        persistent_encoder = None
        persistent_transformer = None
        persistent_vae = None
        if isinstance(flux2_pipeline, dict):
            persistent_encoder = flux2_pipeline.get("encoder")
            persistent_transformer = flux2_pipeline.get("transformer")
            persistent_vae = flux2_pipeline.get("vae")

        # Run in executor to not block event loop
        loop = asyncio.get_event_loop()
        image = await loop.run_in_executor(
            None,
            lambda: generate_image(
                config,
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
            seed=config.seed,
            generation_time=gen_time,
        )

    except Exception as e:
        logger.error(f"[FLUX.2] Generation failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/flux2/generate/stream")
async def flux2_generate_stream(request: Flux2GenerateRequest):
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
            if runtime_config:
                offload_between_stages = getattr(runtime_config, "flux2_offload_between_stages", True)

            # Create generation config
            config = Flux2GenerationConfig(
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
            if not model_path and runtime_config:
                model_path = getattr(runtime_config, "flux2_model_path", None)
            if not vae_path and runtime_config:
                vae_path = getattr(runtime_config, "flux2_vae_path", None)

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
            progress_queue = asyncio.Queue()

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
                if isinstance(flux2_pipeline, dict):
                    p_encoder = flux2_pipeline.get("encoder")
                    p_transformer = flux2_pipeline.get("transformer")
                    p_vae = flux2_pipeline.get("vae")

                return generate_image_with_progress(
                    config,
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
            yield f"data: {json.dumps({'type': 'complete', 'urls': [data_url], 'url': data_url, 'seed': config.seed, 'generation_time': gen_time})}\n\n"

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


# =============================================================================
# Pipeline Schema API (for dynamic UI generation)
# =============================================================================


@app.get("/api/pipelines")
async def get_pipeline_schemas():
    """Return all pipeline schemas for frontend form generation.

    The frontend uses these schemas to dynamically render forms without
    hardcoding pipeline-specific UI. Each schema describes:
    - Pipeline metadata (name, description, output type)
    - Form parameters with types, defaults, and constraints
    - Feature flags (img2img, streaming, reference images)

    Returns:
        dict with:
        - pipelines: Dict of pipeline_id -> PipelineSchema
        - defaults: Current RuntimeConfig values (if loaded)
        - loaded_pipeline: Currently loaded pipeline type (if any)
    """
    from llm_dit.pipelines.schemas import get_all_pipelines

    # Get all registered pipeline schemas
    pipelines = get_all_pipelines()
    pipeline_dicts = {pid: schema.to_dict() for pid, schema in pipelines.items()}

    # Get current defaults from RuntimeConfig if available
    defaults = {}
    if runtime_config is not None:
        try:
            defaults = runtime_config.to_dict()
        except Exception as e:
            logger.warning(f"Failed to serialize RuntimeConfig: {e}")

    # Determine which pipeline is currently loaded
    loaded_pipeline = None
    if pipeline is not None:
        loaded_pipeline = "zimage"
    elif qwen_image_pipeline is not None:
        loaded_pipeline = "qwenimage-layered"
    elif qwen_image_t2i_pipeline is not None:
        loaded_pipeline = "qwenimage-t2i"
    elif ltx2_pipeline is not None:
        loaded_pipeline = "ltx2"
    elif flux2_pipeline is not None:
        loaded_pipeline = "flux2"

    return {
        "pipelines": pipeline_dicts,
        "defaults": defaults,
        "loaded_pipeline": loaded_pipeline,
    }


@app.get("/api/pipelines/{pipeline_id}")
async def get_pipeline_schema(pipeline_id: str):
    """Get schema for a specific pipeline.

    Args:
        pipeline_id: Pipeline identifier (e.g., "zimage", "ltx2")

    Returns:
        PipelineSchema dict for the requested pipeline

    Raises:
        404 if pipeline not found
    """
    from llm_dit.pipelines.schemas import get_pipeline, get_all_pipelines

    schema = get_pipeline(pipeline_id)
    if schema is None:
        raise HTTPException(
            status_code=404,
            detail=f"Pipeline '{pipeline_id}' not found. Available: {list(get_all_pipelines().keys())}",
        )

    return schema.to_dict()


@app.get("/api/pipelines/{pipeline_id}/defaults")
async def get_pipeline_defaults(pipeline_id: str):
    """Get default values for a specific pipeline.

    Merges schema defaults with RuntimeConfig values for the pipeline.

    Args:
        pipeline_id: Pipeline identifier

    Returns:
        Dict of parameter_id -> default_value
    """
    from llm_dit.pipelines.schemas import get_pipeline

    schema = get_pipeline(pipeline_id)
    if schema is None:
        raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_id}' not found")

    # Start with schema defaults
    defaults = schema.get_defaults()

    # Overlay RuntimeConfig values if available
    if runtime_config is not None:
        config_dict = runtime_config.to_dict()

        # Map schema param IDs to RuntimeConfig field names
        # Schema uses clean names, config uses prefixed names for clarity
        param_to_config_map = {
            # FLUX.2 mappings
            "block_offload": "flux2_block_offload",
            "compile": "compile",
            # LTX-2 mappings
            "cpu_offload": "cpu_offload",
            # Z-Image mappings
            "shift": "shift",
            "guidance_scale": "guidance_scale",
            "steps": "steps",
            "width": "width",
            "height": "height",
        }

        for param in schema.params:
            # First check direct match (param.id == config field name)
            if param.id in config_dict:
                defaults[param.id] = config_dict[param.id]
            # Then check mapped name
            elif param.id in param_to_config_map:
                config_key = param_to_config_map[param.id]
                if config_key in config_dict:
                    defaults[param.id] = config_dict[config_key]

        # Add special _variant field for conditional visibility
        # This tells the UI which variant is configured (base/turbo)
        if pipeline_id == "zimage":
            defaults["_variant"] = getattr(runtime_config, "zimage_variant", "turbo")

    return defaults


@app.get("/api/generation-config")
async def get_generation_config():
    """Get generation configuration defaults from server config.

    Returns default values for width, height, steps, shift, long_prompt_mode, hidden_layer, SLG, and FMTT.
    Also returns feature flags indicating which advanced features are enabled in config.
    The UI should call this on load to sync with server config.

    If a default_preset is configured for the current pipeline (e.g., zimage.default_preset),
    the preset's generation params (steps, guidance_scale, shift, negative_prompt) are used.
    """
    if runtime_config is None:
        return {
            "width": 1024,
            "height": 1024,
            "steps": 9,
            "guidance_scale": 0.0,
            "shift": 3.0,
            "dynamic_shift": False,
            "long_prompt_mode": "interpolate",
            "hidden_layer": -2,
            "layer_weights": None,
            "default_preset": "",
            "negative_prompt": "",
            # SLG settings
            "slg_scale": 0.0,
            "slg_layers": None,
            "slg_start": 0.05,
            "slg_stop": 0.5,
            # FMTT settings
            "fmtt_scale": 0.0,
            "fmtt_start": 0.0,
            "fmtt_stop": 0.5,
            "fmtt_normalize": "unit",
            "fmtt_decode_scale": 0.5,
            # Feature flags (all disabled by default)
            "features": {
                "dype_enabled": False,
                "slg_enabled": False,
                "fmtt_enabled": False,
                "differential_img2img_enabled": True,  # Always available for Z-Image
            },
        }

    # Get feature flags from config
    dype_enabled = getattr(runtime_config, "dype_enabled", False)
    slg_enabled = (
        getattr(runtime_config, "slg_enabled", False)
        or getattr(runtime_config, "slg_scale", 0.0) > 0
    )
    fmtt_enabled = (
        getattr(runtime_config, "fmtt_enabled", False)
        or getattr(runtime_config, "fmtt_scale", 0.0) > 0
    )

    # Default generation params from RuntimeConfig (turbo defaults)
    steps = runtime_config.steps
    guidance_scale = runtime_config.guidance_scale
    shift = runtime_config.shift
    d_noise = getattr(runtime_config, "d_noise", 1.0)
    default_preset = ""
    negative_prompt = ""

    # Try to load default_preset from zimage config and use its values
    # This makes the API preset-aware: config.toml holds infrastructure,
    # presets hold generation params (steps, guidance_scale, shift, negative_prompt)
    try:
        config_dict = runtime_config.to_dict() if hasattr(runtime_config, 'to_dict') else {}
        zimage_config = config_dict.get("zimage", {})
        if isinstance(zimage_config, dict):
            default_preset = zimage_config.get("default_preset", "")

        if default_preset:
            from llm_dit.presets import get_preset_registry
            presets_dir = config_dict.get("presets_dir", "presets")
            registry = get_preset_registry(presets_dir)
            preset = registry.get(default_preset)
            if preset:
                # Use preset values for generation params
                if preset.steps is not None:
                    steps = preset.steps
                if preset.guidance_scale is not None:
                    guidance_scale = preset.guidance_scale
                if preset.shift is not None:
                    shift = preset.shift
                if preset.d_noise is not None:
                    d_noise = preset.d_noise
                if preset.negative_prompt:
                    negative_prompt = preset.negative_prompt
    except Exception as e:
        # If preset loading fails, fall back to RuntimeConfig values
        logger.warning(f"Failed to load default preset '{default_preset}': {e}")

    return {
        # Z-Image variant (turbo/base) - determines default parameter values
        "zimage_variant": getattr(runtime_config, "zimage_variant", "turbo"),
        "width": runtime_config.width,
        "height": runtime_config.height,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "shift": shift,
        "dynamic_shift": getattr(runtime_config, "dynamic_shift", False),
        "d_noise": d_noise,
        "long_prompt_mode": runtime_config.long_prompt_mode,
        "hidden_layer": runtime_config.hidden_layer,
        "layer_weights": getattr(runtime_config, "layer_weights", None),
        # Preset info
        "default_preset": default_preset,
        "negative_prompt": negative_prompt,
        # SLG settings
        "slg_scale": getattr(runtime_config, "slg_scale", 0.0),
        "slg_layers": getattr(runtime_config, "slg_layers", None),
        "slg_start": getattr(runtime_config, "slg_start", 0.05),
        "slg_stop": getattr(runtime_config, "slg_stop", 0.5),
        # FMTT settings
        "fmtt_scale": getattr(runtime_config, "fmtt_scale", 0.0),
        "fmtt_start": getattr(runtime_config, "fmtt_start", 0.0),
        "fmtt_stop": getattr(runtime_config, "fmtt_stop", 0.5),
        "fmtt_normalize": getattr(runtime_config, "fmtt_normalize", "unit"),
        "fmtt_decode_scale": getattr(runtime_config, "fmtt_decode_scale", 0.5),
        "fmtt_siglip_model": getattr(
            runtime_config, "fmtt_siglip_model", "google/siglip2-giant-opt-patch16-384"
        ),
        "fmtt_siglip_device": getattr(runtime_config, "fmtt_siglip_device", "cuda"),
        # Feature flags based on config
        "features": {
            "dype_enabled": dype_enabled,
            "slg_enabled": slg_enabled,
            "fmtt_enabled": fmtt_enabled,
            "differential_img2img_enabled": True,  # Always available for Z-Image
        },
    }


# =============================================================================
# Presets API (for generation presets)
# =============================================================================


@app.get("/api/presets")
async def get_all_presets():
    """Return all generation presets with metadata.

    Returns:
        dict with:
        - presets: List of preset objects with name, description, category, params
    """
    from llm_dit.presets import get_preset_registry

    try:
        registry = get_preset_registry()
    except ValueError:
        # Registry not initialized - initialize with default path
        from llm_dit.presets import get_preset_registry
        presets_dir = "presets"
        if runtime_config is not None:
            # Try to get presets_dir from config if available
            config_dict = runtime_config.to_dict() if hasattr(runtime_config, 'to_dict') else {}
            presets_dir = config_dict.get("presets_dir", "presets")
        registry = get_preset_registry(presets_dir)

    all_presets = registry.get_all()
    return {
        "presets": [p.to_api_response() for p in all_presets.values()],
    }


@app.get("/api/presets/{pipeline_id}")
async def get_presets_for_pipeline(pipeline_id: str, variant: Optional[str] = None):
    """Return presets that apply to a specific pipeline.

    Args:
        pipeline_id: Pipeline identifier (e.g., "zimage", "ltx2")
        variant: Optional variant filter (e.g., "base", "turbo")

    Returns:
        dict with:
        - presets: List of preset objects applicable to this pipeline
        - default_preset: Name of the default preset (from config or schema)
    """
    from llm_dit.presets import get_preset_registry

    try:
        registry = get_preset_registry()
    except ValueError:
        # Registry not initialized - initialize with default path
        from llm_dit.presets import get_preset_registry
        presets_dir = "presets"
        if runtime_config is not None:
            config_dict = runtime_config.to_dict() if hasattr(runtime_config, 'to_dict') else {}
            presets_dir = config_dict.get("presets_dir", "presets")
        registry = get_preset_registry(presets_dir)

    # Get presets for this pipeline (and optional variant filter)
    presets = registry.list_for_pipeline(pipeline_id, variant=variant)

    # Determine default preset from config
    default_preset = ""
    if runtime_config is not None:
        config_dict = runtime_config.to_dict() if hasattr(runtime_config, 'to_dict') else {}
        # Check pipeline-specific default_preset
        pipeline_config = config_dict.get(pipeline_id, {})
        if isinstance(pipeline_config, dict):
            default_preset = pipeline_config.get("default_preset", "")

    return {
        "presets": [p.to_api_response() for p in presets],
        "default_preset": default_preset,
    }


@app.get("/api/presets/preset/{name}")
async def get_preset_by_name(name: str):
    """Get full details for a specific preset.

    Args:
        name: Preset name

    Returns:
        Full preset object with all parameters

    Raises:
        404 if preset not found
    """
    from llm_dit.presets import get_preset_registry

    try:
        registry = get_preset_registry()
    except ValueError:
        # Registry not initialized - initialize with default path
        from llm_dit.presets import get_preset_registry
        presets_dir = "presets"
        if runtime_config is not None:
            config_dict = runtime_config.to_dict() if hasattr(runtime_config, 'to_dict') else {}
            presets_dir = config_dict.get("presets_dir", "presets")
        registry = get_preset_registry(presets_dir)

    preset = registry.get(name)
    if preset is None:
        raise HTTPException(
            status_code=404,
            detail=f"Preset '{name}' not found. Available: {registry.list_names()}",
        )

    return preset.to_api_response()


@app.get("/api/resolution-config")
async def get_resolution_config(model: Optional[str] = None):
    """Get resolution constraints for client-side validation.

    Returns VAE multiple, min/max limits, categorized presets, and DyPE config.
    Presets are filtered based on the active model type.

    Args:
        model: Optional model filter ("zimage", "qwenimage-layered", "qwenimage-t2i")
               If not provided, returns presets for all models.

    Model-specific constraints:
    - Z-Image: Flexible resolutions, must be divisible by 16
    - Qwen-Image-Layered: Fixed 640x640 or 1024x1024 only
    - Qwen-Image T2I: Default 1328x1328, flexible with VAE constraints
    """
    from llm_dit.constants import (
        ASPECT_RATIOS,
        DEFAULT_RESOLUTION,
        MAX_RESOLUTION,
        MIN_RESOLUTION,
        VAE_MULTIPLE,
        VAE_SCALE_FACTOR,
    )

    # Detect currently loaded model if not specified
    current_model = model
    if current_model is None:
        if pipeline is not None:
            from llm_dit.pipelines import ZImagePipeline
            from llm_dit.pipelines.qwen_image_diffusers import QwenImageDiffusersPipeline

            if isinstance(pipeline, ZImagePipeline):
                current_model = "zimage"
            elif isinstance(pipeline, QwenImageDiffusersPipeline):
                current_model = "qwenimage-layered"
        if qwen_image_t2i_pipeline is not None:
            current_model = "qwenimage-t2i"
        if qwen_image_pipeline is not None and current_model is None:
            current_model = "qwenimage-layered"

    # DyPE configuration (Z-Image only)
    DYPE_BASE_RESOLUTION = 1024  # Z-Image training resolution

    def get_dype_recommendation(width: int, height: int) -> dict:
        """Get DyPE recommendation based on resolution."""
        max_dim = max(width, height)
        scale = max_dim / DYPE_BASE_RESOLUTION
        if scale <= 1.0:
            return {"recommended": False, "exponent": None}
        if scale >= 3.0:
            exponent = 2.0
        elif scale >= 1.5:
            exponent = 1.0
        else:
            exponent = 0.5
        return {"recommended": True, "exponent": exponent}

    # Model-specific resolution constraints
    model_constraints = {
        "zimage": {
            "vae_multiple": 16,
            "min_resolution": 256,
            "max_resolution": 4096,
            "default_width": 1024,
            "default_height": 1024,
            "flexible": True,
            "supports_dype": True,
            "supports_slg": True,
            "supports_fmtt": True,
        },
        "qwenimage-layered": {
            "vae_multiple": 16,
            "min_resolution": 640,
            "max_resolution": 1024,
            "default_width": 640,
            "default_height": 640,
            "flexible": False,  # Only 640 or 1024
            "fixed_sizes": [640, 1024],
            "supports_dype": False,
            "supports_slg": False,
            "supports_fmtt": False,
        },
        "qwenimage-t2i": {
            "vae_multiple": 16,
            "min_resolution": 256,
            "max_resolution": 2048,
            "default_width": 1328,
            "default_height": 1328,
            "flexible": True,
            "supports_dype": False,
            "supports_slg": False,
            "supports_fmtt": False,
        },
    }

    # Helper to determine aspect category for filter buttons
    def get_aspect_category(width: int, height: int) -> str:
        """Determine aspect category based on ratio for UI filtering."""
        ratio = width / height
        if 0.95 <= ratio <= 1.05:
            return "square"
        elif ratio > 1.05:
            if ratio >= 2.0:  # 19.5:9 = 2.17, 21:9 = 2.33
                return "mobile-landscape"
            return "landscape"
        else:  # ratio < 0.95
            if ratio <= 0.5:  # 9:19.5 = 0.46, 9:20 = 0.45
                return "mobile-portrait"
            return "portrait"

    # Z-Image presets (flexible, all divisible by 16)
    zimage_presets = [
        # Square (1:1)
        {
            "value": "512x512",
            "label": "512",
            "width": 512,
            "height": 512,
            "category": "square",
            "ratio": "1:1",
        },
        {
            "value": "768x768",
            "label": "768",
            "width": 768,
            "height": 768,
            "category": "square",
            "ratio": "1:1",
        },
        {
            "value": "1024x1024",
            "label": "1024",
            "width": 1024,
            "height": 1024,
            "category": "square",
            "ratio": "1:1",
            "default": True,
        },
        {
            "value": "1280x1280",
            "label": "1280",
            "width": 1280,
            "height": 1280,
            "category": "square",
            "ratio": "1:1",
        },
        {
            "value": "1536x1536",
            "label": "1536",
            "width": 1536,
            "height": 1536,
            "category": "square",
            "ratio": "1:1",
        },
        {
            "value": "1920x1920",
            "label": "1920",
            "width": 1920,
            "height": 1920,
            "category": "square",
            "ratio": "1:1",
        },
        {
            "value": "2048x2048",
            "label": "2K",
            "width": 2048,
            "height": 2048,
            "category": "square",
            "ratio": "1:1",
        },
        # Landscape - 16:9
        {
            "value": "1280x720",
            "label": "720p",
            "width": 1280,
            "height": 720,
            "category": "landscape",
            "ratio": "16:9",
        },
        {
            "value": "1920x1088",
            "label": "1080p",
            "width": 1920,
            "height": 1088,
            "category": "landscape",
            "ratio": "16:9",
        },
        {
            "value": "2560x1440",
            "label": "1440p",
            "width": 2560,
            "height": 1440,
            "category": "landscape",
            "ratio": "16:9",
        },
        # Landscape - 3:2
        {
            "value": "1536x1024",
            "label": "1536x1024",
            "width": 1536,
            "height": 1024,
            "category": "landscape",
            "ratio": "3:2",
        },
        {
            "value": "1920x1280",
            "label": "1920x1280",
            "width": 1920,
            "height": 1280,
            "category": "landscape",
            "ratio": "3:2",
        },
        # Landscape - 4:3
        {
            "value": "1024x768",
            "label": "1024x768",
            "width": 1024,
            "height": 768,
            "category": "landscape",
            "ratio": "4:3",
        },
        {
            "value": "1280x960",
            "label": "1280x960",
            "width": 1280,
            "height": 960,
            "category": "landscape",
            "ratio": "4:3",
        },
        {
            "value": "1600x1200",
            "label": "1600x1200",
            "width": 1600,
            "height": 1200,
            "category": "landscape",
            "ratio": "4:3",
        },
        # Mobile Landscape - 21:9, 19.5:9 (phone screens rotated)
        {
            "value": "1792x768",
            "label": "Ultrawide",
            "width": 1792,
            "height": 768,
            "category": "landscape",
            "ratio": "21:9",
        },
        {
            "value": "2560x1088",
            "label": "UW 1080",
            "width": 2560,
            "height": 1088,
            "category": "landscape",
            "ratio": "21:9",
        },
        {
            "value": "2340x1080",
            "label": "Phone HD",
            "width": 2340,
            "height": 1080,
            "category": "landscape",
            "ratio": "19.5:9",
        },
        # Portrait - 9:16
        {
            "value": "720x1280",
            "label": "720p",
            "width": 720,
            "height": 1280,
            "category": "portrait",
            "ratio": "9:16",
        },
        {
            "value": "1088x1920",
            "label": "1080p",
            "width": 1088,
            "height": 1920,
            "category": "portrait",
            "ratio": "9:16",
        },
        {
            "value": "1440x2560",
            "label": "1440p",
            "width": 1440,
            "height": 2560,
            "category": "portrait",
            "ratio": "9:16",
        },
        # Portrait - 2:3
        {
            "value": "1024x1536",
            "label": "1024x1536",
            "width": 1024,
            "height": 1536,
            "category": "portrait",
            "ratio": "2:3",
        },
        {
            "value": "1280x1920",
            "label": "1280x1920",
            "width": 1280,
            "height": 1920,
            "category": "portrait",
            "ratio": "2:3",
        },
        # Portrait - 3:4
        {
            "value": "768x1024",
            "label": "768x1024",
            "width": 768,
            "height": 1024,
            "category": "portrait",
            "ratio": "3:4",
        },
        {
            "value": "960x1280",
            "label": "960x1280",
            "width": 960,
            "height": 1280,
            "category": "portrait",
            "ratio": "3:4",
        },
        {
            "value": "1200x1600",
            "label": "1200x1600",
            "width": 1200,
            "height": 1600,
            "category": "portrait",
            "ratio": "3:4",
        },
        # Mobile Portrait - 9:19.5, 9:20 (phone screens)
        {
            "value": "1080x2340",
            "label": "Phone HD",
            "width": 1080,
            "height": 2340,
            "category": "portrait",
            "ratio": "9:19.5",
        },
        {
            "value": "1284x2778",
            "label": "iPhone Pro",
            "width": 1284,
            "height": 2778,
            "category": "portrait",
            "ratio": "9:19.5",
        },
    ]

    # Qwen-Image-Layered presets (FIXED: only 640 or 1024 square)
    qwenimage_layered_presets = [
        {
            "value": "640x640",
            "label": "640 (Fast)",
            "width": 640,
            "height": 640,
            "category": "square",
            "ratio": "1:1",
            "default": True,
        },
        {
            "value": "1024x1024",
            "label": "1024 (Quality)",
            "width": 1024,
            "height": 1024,
            "category": "square",
            "ratio": "1:1",
        },
    ]

    # Qwen-Image T2I presets (flexible, default 1328)
    qwenimage_t2i_presets = [
        {
            "value": "512x512",
            "label": "512",
            "width": 512,
            "height": 512,
            "category": "square",
            "ratio": "1:1",
        },
        {
            "value": "768x768",
            "label": "768",
            "width": 768,
            "height": 768,
            "category": "square",
            "ratio": "1:1",
        },
        {
            "value": "1024x1024",
            "label": "1024",
            "width": 1024,
            "height": 1024,
            "category": "square",
            "ratio": "1:1",
        },
        {
            "value": "1328x1328",
            "label": "1328 (Default)",
            "width": 1328,
            "height": 1328,
            "category": "square",
            "ratio": "1:1",
            "default": True,
        },
        {
            "value": "1536x1536",
            "label": "1536",
            "width": 1536,
            "height": 1536,
            "category": "square",
            "ratio": "1:1",
        },
        # Landscape
        {
            "value": "1328x1024",
            "label": "1328x1024",
            "width": 1328,
            "height": 1024,
            "category": "landscape",
            "ratio": "4:3",
        },
        {
            "value": "1536x1024",
            "label": "1536x1024",
            "width": 1536,
            "height": 1024,
            "category": "landscape",
            "ratio": "3:2",
        },
        # Portrait
        {
            "value": "1024x1328",
            "label": "1024x1328",
            "width": 1024,
            "height": 1328,
            "category": "portrait",
            "ratio": "3:4",
        },
        {
            "value": "1024x1536",
            "label": "1024x1536",
            "width": 1024,
            "height": 1536,
            "category": "portrait",
            "ratio": "2:3",
        },
    ]

    # Select presets based on model
    if current_model == "qwenimage-layered":
        presets = qwenimage_layered_presets
        constraints = model_constraints["qwenimage-layered"]
    elif current_model == "qwenimage-t2i":
        presets = qwenimage_t2i_presets
        constraints = model_constraints["qwenimage-t2i"]
    else:
        # Default to Z-Image
        presets = zimage_presets
        constraints = model_constraints["zimage"]

    # Add aspect_category and DyPE recommendations to presets
    for preset in presets:
        # Add aspect_category for UI filtering
        preset["aspect_category"] = get_aspect_category(preset["width"], preset["height"])

        # Add DyPE recommendations (Z-Image only)
        if current_model in (None, "zimage"):
            preset["dype"] = get_dype_recommendation(preset["width"], preset["height"])
        else:
            preset["dype"] = {"recommended": False, "exponent": None}

    # Determine available categories
    categories = list(set(p["category"] for p in presets))

    # Use config.toml values if available, otherwise fall back to model defaults
    # This ensures the UI respects user's configured resolution while still
    # providing sensible defaults (1024x1024 for Z-Image)
    if runtime_config is not None and current_model == "zimage":
        default_width = getattr(runtime_config, "width", None) or constraints.get(
            "default_width", 1024
        )
        default_height = getattr(runtime_config, "height", None) or constraints.get(
            "default_height", 1024
        )
    else:
        default_width = constraints.get("default_width", 1024)
        default_height = constraints.get("default_height", 1024)

    return {
        "current_model": current_model,
        "model_constraints": model_constraints,
        "active_constraints": constraints,
        "vae_multiple": VAE_MULTIPLE,
        "vae_scale_factor": VAE_SCALE_FACTOR,
        "min_resolution": constraints.get("min_resolution", MIN_RESOLUTION),
        "max_resolution": constraints.get("max_resolution", MAX_RESOLUTION),
        "default_resolution": DEFAULT_RESOLUTION,
        "default_width": default_width,
        "default_height": default_height,
        "dype_base_resolution": DYPE_BASE_RESOLUTION,
        "aspect_ratios": ASPECT_RATIOS,
        "presets": presets,
        "categories": categories,
        "supports_dype": constraints.get("supports_dype", False),
        "supports_slg": constraints.get("supports_slg", False),
        "supports_fmtt": constraints.get("supports_fmtt", False),
    }


@app.get("/api/rewriter-config")
async def get_rewriter_config():
    """Get rewriter configuration defaults from server config.

    Qwen3 Best Practices (thinking mode):
    - temperature=0.6, top_p=0.95, top_k=20, min_p=0
    - DO NOT use greedy decoding (causes repetition)
    - presence_penalty=0-2 helps reduce endless repetitions
    """
    if runtime_config is None:
        # Return hardcoded defaults matching Qwen3 thinking mode
        return {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "max_tokens": 1024,
            "use_api": False,
        }
    return {
        "temperature": runtime_config.rewriter_temperature,
        "top_p": runtime_config.rewriter_top_p,
        "top_k": runtime_config.rewriter_top_k,
        "min_p": runtime_config.rewriter_min_p,
        "presence_penalty": runtime_config.rewriter_presence_penalty,
        "max_tokens": runtime_config.rewriter_max_tokens,
        "use_api": runtime_config.rewriter_use_api,
    }


@app.get("/api/rewriter-models")
async def get_rewriter_models():
    """Return available rewriter models.

    Models:
    - qwen3-4b: Text-only model (always available)
    - qwen3-vl: Vision+text model (available if vl_model_path is configured)
    - qwen3-vl-api: Vision+text model via API (available if vl_api_model is configured)
    """
    models = [
        {
            "id": "qwen3-4b",
            "name": "Qwen3-4B (Text)",
            "supports_image": False,
            "loaded": True,  # Always available via encoder
        }
    ]

    # Check if VL rewriter via API is available (higher priority than local VL)
    vl_api_available = False
    if (
        runtime_config
        and runtime_config.rewriter_vl_api_model
        and runtime_config.rewriter_vl_enabled
    ):
        vl_api_available = True
        models.append(
            {
                "id": "qwen3-vl-api",
                "name": f"VL via API ({runtime_config.rewriter_vl_api_model})",
                "supports_image": True,
                "loaded": True,  # API is always available
            }
        )

    # Check if local VL rewriter is available
    vl_local_available = False
    vl_loaded = False
    if runtime_config and runtime_config.vl_model_path and runtime_config.rewriter_vl_enabled:
        vl_local_available = True
        vl_loaded = vl_rewriter is not None or vl_extractor is not None
        models.append(
            {
                "id": "qwen3-vl",
                "name": "Qwen3-VL (Vision+Text)",
                "supports_image": True,
                "loaded": vl_loaded,
            }
        )

    return {
        "models": models,
        "default": "qwen3-4b",
        "vl_available": vl_api_available or vl_local_available,
        "vl_enabled": runtime_config.rewriter_vl_enabled if runtime_config else True,
    }


@app.post("/api/encode")
async def encode(request: EncodeRequest):
    """Encode a prompt to embeddings (for distributed inference)."""
    # Use encoder from pipeline or standalone encoder
    enc = encoder if encoder is not None else (pipeline.encoder if pipeline else None)
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
            logger.info(f"Formatted prompt ({len(formatted_prompt)} chars, {token_count} tokens):")
            logger.info(f"---BEGIN FORMATTED PROMPT---")
            logger.info(formatted_prompt)
            logger.info(f"---END FORMATTED PROMPT---")

        return {
            "shape": list(embeddings.shape),
            "dtype": str(embeddings.dtype),
            "encode_time": encode_time,
            "token_count": token_count,
            "prompt": request.prompt,
            "formatted_prompt": formatted_prompt,
        }
    except Exception as e:
        logger.error(f"Encoding failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate")
async def generate(request: GenerateRequest):
    """Generate an image from a prompt."""
    if encoder_only_mode:
        raise HTTPException(
            status_code=400, detail="Server running in encoder-only mode. Use /api/encode instead."
        )

    # Load Z-Image pipeline on-demand if not already loaded
    if pipeline is None:
        try:
            load_zimage_pipeline_on_demand()
        except Exception as e:
            logger.error(f"[Z-Image] Failed to load pipeline on-demand: {e}")
            raise HTTPException(
                status_code=503,
                detail=f"Z-Image pipeline failed to load: {str(e)}",
            )

    # Apply variant-aware defaults (Base vs Turbo)
    apply_zimage_variant_defaults(request)

    try:
        logger.info("=" * 60)
        logger.info("GENERATION REQUEST")
        logger.info("=" * 60)
        logger.info(f"  Prompt: {request.prompt[:80]}...")
        if request.negative_prompt:
            logger.info(f"  Negative: {request.negative_prompt[:80]}...")
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
        logger.info(f"  pipeline.device: {pipeline.device}")
        logger.info(f"  pipeline.dtype: {pipeline.dtype}")
        logger.info(
            f"  pipeline.encoder: {type(pipeline.encoder).__name__ if pipeline.encoder is not None else 'None'}"
        )
        logger.info(f"  pipeline.transformer: {pipeline.transformer is not None}")
        logger.info(f"  pipeline.vae: {pipeline.vae is not None}")
        if pipeline.encoder is not None:
            backend = getattr(pipeline.encoder, "backend", None)
            logger.info(f"  encoder.backend: {type(backend).__name__ if backend else 'None'}")
        if runtime_config is not None:
            model_path = runtime_config.zimage_model_path or runtime_config.model_path
            logger.info(f"  variant: {runtime_config.zimage_variant}")
            logger.info(f"  model_path: {model_path}")
        logger.info("-" * 60)

        # Set up generator for reproducibility
        generator = None
        if request.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(request.seed)

        # Negative prompt: only use for base variant (turbo has CFG=0, so it has no effect)
        negative_prompt_to_use = None
        if runtime_config is not None and runtime_config.zimage_variant == "base":
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
        if negative_prompt_to_use:
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
            image = pipeline.generate_multipass(
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
            image = pipeline(
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

        # Convert to base64 for history storage and response
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        img_b64 = base64.b64encode(img_bytes.getvalue()).decode("ascii")

        # Get formatted prompt for history
        formatted_prompt = None
        enc = encoder if encoder is not None else (pipeline.encoder if pipeline else None)
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
            "id": len(generation_history),
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
            "image_b64": img_b64,
            "formatted_prompt": formatted_prompt,
        }
        generation_history.insert(0, history_entry)
        # Trim history
        if len(generation_history) > MAX_HISTORY:
            generation_history.pop()

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


@app.post("/api/generate/stream")
async def generate_stream(request: GenerateRequest):
    """Generate an image with SSE progress streaming.

    Returns Server-Sent Events with progress updates during generation,
    allowing the frontend to show step-by-step progress.

    Events:
    - {"type": "status", "message": "..."} - Status updates
    - {"type": "progress", "step": N, "total_steps": M, ...} - Step progress
    - {"type": "complete", ...} - Final result with image data
    - {"type": "error", "message": "..."} - Error occurred
    """
    # Apply variant-aware defaults (Base vs Turbo)
    apply_zimage_variant_defaults(request)

    if encoder_only_mode:
        raise HTTPException(
            status_code=400, detail="Server running in encoder-only mode. Use /api/encode instead."
        )

    # Load Z-Image pipeline on-demand if not already loaded
    if pipeline is None:
        try:
            load_zimage_pipeline_on_demand()
        except Exception as e:
            logger.error(f"[Z-Image] Failed to load pipeline on-demand: {e}")
            raise HTTPException(
                status_code=503,
                detail=f"Z-Image pipeline failed to load: {str(e)}",
            )

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
            if runtime_config is not None and runtime_config.zimage_variant == "base":
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
            logger.info(f"  Prompt: {request.prompt[:80]}...")
            if negative_prompt_to_use:
                neg_display = negative_prompt_to_use[:60] + "..." if len(negative_prompt_to_use) > 60 else negative_prompt_to_use
                logger.info(f"  Negative: {neg_display}")
            logger.info(f"  Size: {request.width}x{request.height}")
            logger.info(f"  Steps: {request.steps}")
            logger.info(f"  Seed: {actual_seed}")

            # Run generation in thread pool (blocking operation)
            loop = asyncio.get_event_loop()

            def do_generate():
                return pipeline(
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
                "id": len(generation_history),
                "timestamp": time.time(),
                "model_type": "zimage",
                "prompt": request.prompt,
                "width": request.width,
                "height": request.height,
                "steps": request.steps,
                "seed": actual_seed,
                "gen_time": gen_time,
                "image_b64": img_b64,
            }
            generation_history.insert(0, history_entry)
            if len(generation_history) > MAX_HISTORY:
                generation_history.pop()

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

    return StreamingResponse(
        generate_with_progress(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@app.post("/api/img2img")
async def img2img(request: Img2ImgRequest):
    """Generate an image from an input image with optional differential mask.

    The mask controls per-pixel edit strength:
    - Black (0): Preserve original
    - White (255): Allow full editing
    - Gray: Partial editing
    """
    # Apply variant-aware defaults (Base vs Turbo)
    apply_zimage_variant_defaults(request)

    if encoder_only_mode:
        raise HTTPException(
            status_code=400, detail="Server running in encoder-only mode. Img2img not available."
        )
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")

    try:
        from PIL import UnidentifiedImageError
        PILImage = Image  # Alias for consistency with existing code

        logger.info("=" * 60)
        logger.info("IMG2IMG REQUEST")
        logger.info("=" * 60)
        logger.info(f"  Prompt: {request.prompt[:80]}...")
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
        if runtime_config is not None and runtime_config.zimage_variant == "base":
            negative_prompt_to_use = request.negative_prompt

        start = time.time()

        # Generate image using img2img
        # Note: SLG, FMTT, DyPE, and layer_weights are not supported in img2img
        logger.info(f"Calling pipeline.img2img with strength={request.strength}...")
        if mask_image:
            logger.info("  Using differential diffusion with mask")

        image = pipeline.img2img(
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

        # Convert to base64 for history storage and response
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        img_b64 = base64.b64encode(img_bytes.getvalue()).decode("ascii")

        history_entry = {
            "id": len(generation_history),
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
            "image_b64": img_b64,
        }
        generation_history.insert(0, history_entry)
        if len(generation_history) > MAX_HISTORY:
            generation_history.pop()

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


@app.post("/api/format-prompt")
async def format_prompt_endpoint(request: EncodeRequest):
    """Preview the formatted prompt without encoding (fast, no GPU needed)."""
    global pipeline

    # Use encoder from pipeline or standalone encoder
    enc = encoder if encoder is not None else (pipeline.encoder if pipeline else None)
    if enc is None:
        # Try loading Z-Image on-demand to get the encoder
        try:
            load_zimage_pipeline_on_demand()
            enc = pipeline.encoder if pipeline else None
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

        return {
            "formatted_prompt": formatted,
            "char_count": len(formatted),
            "token_count": token_count,
            "max_tokens": 1504,
            "prompt": request.prompt,
            "system_prompt": request.system_prompt,
            "thinking_content": request.thinking_content,
            "assistant_content": request.assistant_content,
            "template": request.template,
            "force_think_block": request.force_think_block,
            "strip_quotes": request.strip_quotes,
        }
    except Exception as e:
        logger.error(f"Format failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _get_zimage_encoder():
    """Get Z-Image encoder from standalone encoder or Z-Image pipeline.

    Returns None for Qwen-Image pipelines (they don't have a separate encoder).
    """
    if encoder is not None:
        return encoder
    if pipeline is not None and hasattr(pipeline, "encoder"):
        return pipeline.encoder
    return None


@app.get("/api/templates")
async def list_templates():
    """List available templates with full data for UI population."""
    # Use encoder from pipeline or standalone encoder
    enc = _get_zimage_encoder()
    if enc is None or enc.templates is None:
        return {"templates": []}

    templates = []
    for name in enc.templates:
        tpl = enc.templates.get(name)
        if tpl and tpl.category != "rewriter":  # Exclude rewriter templates
            templates.append(
                {
                    "name": name,
                    "description": tpl.description or "",
                    "category": tpl.category or "general",
                    "system_prompt": tpl.content or "",
                    "thinking_content": tpl.thinking_content or "",
                    "assistant_content": tpl.assistant_content or "",
                    "add_think_block": tpl.add_think_block,
                }
            )

    # Sort by category then name
    templates.sort(key=lambda x: (x["category"], x["name"]))
    return {"templates": templates}


@app.get("/api/rewriters")
async def list_rewriters():
    """List available rewriter templates."""
    # Use encoder from pipeline or standalone encoder
    enc = _get_zimage_encoder()
    if enc is None or enc.templates is None:
        return {"rewriters": []}

    # Get rewriter templates (category == "rewriter")
    rewriters = []
    for tpl in enc.templates.list_by_category("rewriter"):
        rewriters.append(
            {
                "name": tpl.name,
                "description": tpl.description,
            }
        )

    return {"rewriters": rewriters}


async def _rewrite_with_vl_api(request: RewriteRequest) -> dict:
    """
    Handle VL-based rewriting via remote API (heylookitsanllm).

    Uses the configured vl_api_model for vision+text generation.
    """
    import re

    # Check if VL API is available
    if not runtime_config or not runtime_config.rewriter_vl_api_model:
        raise HTTPException(
            status_code=400,
            detail="VL API model not configured. Set rewriter.vl_api_model in config.toml.",
        )

    if not runtime_config.rewriter_vl_enabled:
        raise HTTPException(
            status_code=400,
            detail="VL rewriter is disabled. Enable with rewriter.vl_enabled=true in config.",
        )

    # Determine API URL
    api_url = runtime_config.rewriter_api_url or runtime_config.api_url
    if not api_url:
        raise HTTPException(
            status_code=400,
            detail="No API URL configured. Set rewriter.api_url or api.url in config.toml.",
        )

    # Create API backend with VL model
    from llm_dit.backends.api import APIBackend, APIBackendConfig

    timeout = runtime_config.rewriter_timeout if runtime_config else 120.0
    vl_api_config = APIBackendConfig(
        base_url=api_url,
        model_id=runtime_config.rewriter_vl_api_model,
        timeout=timeout,
    )
    vl_api_backend = APIBackend(vl_api_config)

    # Get template loader from encoder (for template lookup)
    enc = encoder if encoder is not None else (pipeline.encoder if pipeline else None)

    # Determine system prompt
    system_prompt = None
    rewriter_name = "custom"

    if request.custom_system_prompt:
        system_prompt = request.custom_system_prompt.strip()
        rewriter_name = "custom"
    elif request.rewriter:
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
        # Use a default system prompt for VL rewriting
        system_prompt = "Describe what you see in this image in detail, suitable for use as an image generation prompt."
        rewriter_name = "default_vl"

    # Get generation parameters (use 'is not None' to preserve 0 values)
    max_tokens = (
        request.max_tokens
        if request.max_tokens is not None
        else (runtime_config.rewriter_max_tokens if runtime_config else 1024)
    )
    temperature = (
        request.temperature
        if request.temperature is not None
        else (runtime_config.rewriter_temperature if runtime_config else 0.6)
    )
    top_p = (
        request.top_p
        if request.top_p is not None
        else (runtime_config.rewriter_top_p if runtime_config else 0.95)
    )
    top_k = (
        request.top_k
        if request.top_k is not None
        else (runtime_config.rewriter_top_k if runtime_config else 20)
    )
    min_p = (
        request.min_p
        if request.min_p is not None
        else (runtime_config.rewriter_min_p if runtime_config else 0.0)
    )
    presence_penalty = (
        request.presence_penalty
        if request.presence_penalty is not None
        else (runtime_config.rewriter_presence_penalty if runtime_config else 0.0)
    )

    try:
        start = time.time()
        logger.info(
            f"[VL API Rewrite] Using: {rewriter_name} (model: {runtime_config.rewriter_vl_api_model})"
        )
        if request.prompt:
            logger.info(f"[VL API Rewrite] Input prompt: {request.prompt[:100]}...")
        logger.info(f"[VL API Rewrite] Has image: {request.image is not None}")
        logger.info(f"[VL API Rewrite] Params: max_tokens={max_tokens}, temperature={temperature}")

        # Generate using VL API backend
        # The image should already be in data URL format from the request
        generated = vl_api_backend.generate(
            prompt=request.prompt,
            image=request.image,  # Pass the data URL directly
            system_prompt=system_prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            presence_penalty=presence_penalty,
        )

        gen_time = time.time() - start
        logger.info(f"[VL API Rewrite] Generated {len(generated)} chars in {gen_time:.2f}s")

        # Parse thinking content (same logic as local rewrite)
        thinking_content = None
        rewritten_prompt = generated

        think_match = re.search(r"<think>\s*(.*?)\s*</think>", generated, re.DOTALL)
        if think_match:
            thinking_content = think_match.group(1).strip()
            rewritten_prompt = re.sub(
                r"<think>.*?</think>\s*", "", generated, flags=re.DOTALL
            ).strip()
            logger.info(f"[VL API Rewrite] Extracted thinking ({len(thinking_content)} chars)")

        # Clean up any remaining tags
        if thinking_content:
            thinking_content = re.sub(r"</?think>", "", thinking_content).strip()
        if rewritten_prompt:
            rewritten_prompt = re.sub(r"</?think>", "", rewritten_prompt).strip()
            # Strip surrounding quotes if the entire prompt is wrapped
            if rewritten_prompt.startswith('"') and rewritten_prompt.endswith('"'):
                rewritten_prompt = rewritten_prompt[1:-1].strip()

        return {
            "original_prompt": request.prompt or "(image only)",
            "rewritten_prompt": rewritten_prompt,
            "thinking_content": thinking_content,
            "rewriter": rewriter_name,
            "backend": "vl-api",
            "model": runtime_config.rewriter_vl_api_model,
            "gen_time": gen_time,
        }

    except httpx.TimeoutException as e:
        logger.error(f"[VL API Rewrite] Timeout after {timeout}s: {e}")
        raise HTTPException(
            status_code=504,
            detail=f"API request timed out after {timeout}s. Try increasing rewriter.timeout in config.toml.",
        )
    except httpx.HTTPStatusError as e:
        # Parse API error details if available
        error_detail = str(e)
        try:
            error_json = e.response.json()
            if "detail" in error_json:
                error_detail = error_json["detail"]
        except Exception:
            pass
        logger.error(f"[VL API Rewrite] HTTP {e.response.status_code}: {error_detail}")
        raise HTTPException(status_code=e.response.status_code, detail=f"API error: {error_detail}")
    except httpx.ConnectError as e:
        logger.error(f"[VL API Rewrite] Connection failed: {e}")
        raise HTTPException(
            status_code=503, detail=f"Cannot connect to API at {api_url}. Is the server running?"
        )
    except Exception as e:
        logger.error(f"[VL API Rewrite] Failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _rewrite_with_vl(request: RewriteRequest) -> dict:
    """
    Handle VL-based rewriting (image+text or image-only).

    Loads Qwen3-VL on-demand if not already loaded.
    """
    global vl_rewriter, vl_extractor

    # Check if VL is available
    if not runtime_config or not runtime_config.vl_model_path:
        raise HTTPException(
            status_code=400, detail="VL model not configured. Set vl.model_path in config.toml."
        )

    if not runtime_config.rewriter_vl_enabled:
        raise HTTPException(
            status_code=400,
            detail="VL rewriter is disabled. Enable with rewriter.vl_enabled=true in config.",
        )

    # Load VL model on-demand if not already loaded
    # Try to reuse vl_extractor if available (same model)
    if vl_rewriter is None:
        if vl_extractor is not None:
            logger.info("[VL Rewrite] Reusing existing VL extractor for rewriting")
            vl_rewriter = vl_extractor
        else:
            logger.info(f"[VL Rewrite] Loading Qwen3-VL from {runtime_config.vl_model_path}")
            from llm_dit.vl import VLEmbeddingExtractor

            vl_dtype = torch.bfloat16 if runtime_config.vl_device == "cuda" else torch.float32
            vl_rewriter = VLEmbeddingExtractor.from_pretrained(
                runtime_config.vl_model_path,
                device=runtime_config.vl_device,
                dtype=vl_dtype,
            )
            logger.info("[VL Rewrite] Qwen3-VL loaded for rewriting")

    # Decode image if provided
    pil_image = None
    if request.image:
        try:
            # Handle data URL format (data:image/png;base64,...)
            image_data = request.image
            if image_data.startswith("data:"):
                # Extract base64 part after the comma
                image_data = image_data.split(",", 1)[1]
            image_bytes = base64.b64decode(image_data)
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            logger.info(f"[VL Rewrite] Decoded image: {pil_image.size}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to decode image: {e}")

    # Get template loader from encoder (for template lookup)
    enc = encoder if encoder is not None else (pipeline.encoder if pipeline else None)

    # Determine system prompt
    system_prompt = None
    rewriter_name = "custom"

    if request.custom_system_prompt:
        system_prompt = request.custom_system_prompt.strip()
        rewriter_name = "custom"
    elif request.rewriter:
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
        # Use a default system prompt for VL rewriting
        system_prompt = "Describe what you see in this image in detail, suitable for use as an image generation prompt."
        rewriter_name = "default_vl"

    # Get generation parameters (use 'is not None' to preserve 0 values)
    max_tokens = (
        request.max_tokens
        if request.max_tokens is not None
        else (runtime_config.rewriter_max_tokens if runtime_config else 1024)
    )
    temperature = (
        request.temperature
        if request.temperature is not None
        else (runtime_config.rewriter_temperature if runtime_config else 0.6)
    )
    top_p = (
        request.top_p
        if request.top_p is not None
        else (runtime_config.rewriter_top_p if runtime_config else 0.95)
    )
    top_k = (
        request.top_k
        if request.top_k is not None
        else (runtime_config.rewriter_top_k if runtime_config else 20)
    )
    min_p = (
        request.min_p
        if request.min_p is not None
        else (runtime_config.rewriter_min_p if runtime_config else 0.0)
    )
    presence_penalty = (
        request.presence_penalty
        if request.presence_penalty is not None
        else (runtime_config.rewriter_presence_penalty if runtime_config else 0.0)
    )

    try:
        start = time.time()
        logger.info(f"[VL Rewrite] Using: {rewriter_name} (model: qwen3-vl)")
        if request.prompt:
            logger.info(f"[VL Rewrite] Input prompt: {request.prompt[:100]}...")
        logger.info(f"[VL Rewrite] Has image: {pil_image is not None}")
        logger.info(f"[VL Rewrite] Params: max_tokens={max_tokens}, temperature={temperature}")

        # Generate using VL model
        generated = vl_rewriter.generate(
            prompt=request.prompt,
            image=pil_image,
            system_prompt=system_prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            presence_penalty=presence_penalty,
        )

        gen_time = time.time() - start
        logger.info(f"[VL Rewrite] Generated {len(generated)} chars in {gen_time:.2f}s")

        # Parse thinking content (same logic as text-only rewrite)
        import re

        thinking_content = None
        rewritten_prompt = generated

        think_match = re.search(r"<think>\s*(.*?)\s*</think>", generated, re.DOTALL)
        if think_match:
            thinking_content = think_match.group(1).strip()
            rewritten_prompt = re.sub(
                r"<think>.*?</think>\s*", "", generated, flags=re.DOTALL
            ).strip()
            logger.info(f"[VL Rewrite] Extracted thinking ({len(thinking_content)} chars)")

        # Clean up any remaining tags
        if thinking_content:
            thinking_content = re.sub(r"</?think>", "", thinking_content).strip()
        if rewritten_prompt:
            rewritten_prompt = re.sub(r"</?think>", "", rewritten_prompt).strip()
            # Strip surrounding quotes if the entire prompt is wrapped
            if rewritten_prompt.startswith('"') and rewritten_prompt.endswith('"'):
                rewritten_prompt = rewritten_prompt[1:-1].strip()

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "original_prompt": request.prompt or "(image only)",
            "rewritten_prompt": rewritten_prompt,
            "thinking_content": thinking_content,
            "rewriter": rewriter_name,
            "backend": "vl",
            "model": "qwen3-vl",
            "gen_time": gen_time,
        }

    except Exception as e:
        logger.error(f"[VL Rewrite] Failed: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rewrite")
async def rewrite_prompt(request: RewriteRequest):
    """
    Rewrite/expand a prompt using a rewriter template or custom system prompt.

    Uses the same Qwen3 model loaded for text encoding to generate expanded prompts,
    or a separate API backend if configured.

    Supports two modes:
    1. Template mode: Use `rewriter` to specify a rewriter template
    2. Ad-hoc mode: Use `custom_system_prompt` for custom rewriting instructions

    Model selection:
    - qwen3-4b: Text-only model (default)
    - qwen3-vl: Vision+text model (requires vl_model_path configured)
    - qwen3-vl-api: Vision+text via remote API (requires vl_api_model configured)

    Backend selection (for qwen3-4b):
    - If rewriter_use_api is True and rewriter_backend is configured, uses API backend
    - Otherwise, uses the local encoder's backend
    """
    global vl_rewriter

    # Validate that at least prompt or image is provided
    if not request.prompt and not request.image:
        raise HTTPException(
            status_code=400, detail="At least one of 'prompt' or 'image' must be provided"
        )

    # Handle VL model selection
    if request.model == "qwen3-vl":
        return await _rewrite_with_vl(request)
    elif request.model == "qwen3-vl-api":
        return await _rewrite_with_vl_api(request)

    # If image provided but model is not VL, warn and ignore
    if request.image:
        logger.warning(
            "[Rewrite] Image provided but model is qwen3-4b (text-only). Image will be ignored."
        )

    # Require prompt for text-only model
    if not request.prompt:
        raise HTTPException(
            status_code=400,
            detail="Text prompt is required for qwen3-4b model. Use qwen3-vl for image-only rewriting.",
        )

    # Determine which backend to use for generation
    # Priority: rewriter_backend (if API mode), encoder's backend, pipeline's encoder backend
    backend = None
    backend_name = "local"

    if rewriter_backend is not None:
        backend = rewriter_backend
        backend_name = "api"
        logger.info("[Rewrite] Using API backend for rewriting")
    else:
        # Use encoder from pipeline or standalone encoder
        enc = encoder if encoder is not None else (pipeline.encoder if pipeline else None)
        if enc is not None:
            backend = getattr(enc, "backend", None)
            backend_name = "local"

    if backend is None:
        raise HTTPException(status_code=503, detail="No backend available for generation")

    if not getattr(backend, "supports_generation", False):
        raise HTTPException(status_code=400, detail="Backend does not support text generation")

    # Get template loader from encoder (for template lookup)
    enc = encoder if encoder is not None else (pipeline.encoder if pipeline else None)

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

    if runtime_config is not None:
        if max_tokens is None:
            max_tokens = runtime_config.rewriter_max_tokens
        if temperature is None:
            temperature = runtime_config.rewriter_temperature
        if top_p is None:
            top_p = runtime_config.rewriter_top_p
        if top_k is None:
            top_k = runtime_config.rewriter_top_k
        if min_p is None:
            min_p = runtime_config.rewriter_min_p
        if presence_penalty is None:
            presence_penalty = runtime_config.rewriter_presence_penalty
    else:
        # Fallback defaults (Qwen3 thinking mode)
        if max_tokens is None:
            max_tokens = 1024
        if temperature is None:
            temperature = 0.6
        if top_p is None:
            top_p = 0.95
        if top_k is None:
            top_k = 20
        if min_p is None:
            min_p = 0.0
        if presence_penalty is None:
            presence_penalty = 0.0

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

        return {
            "original_prompt": request.prompt,
            "rewritten_prompt": rewritten_prompt,
            "thinking_content": thinking_content,
            "rewriter": request.rewriter,
            "backend": backend_name,
            "gen_time": gen_time,
        }

    except Exception as e:
        logger.error(f"Rewrite failed: {e}")
        # Clear CUDA cache even on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/history")
async def get_history():
    """Get generation history."""
    return {"history": generation_history}


@app.delete("/api/history/{index}")
async def delete_history_item(index: int):
    """Delete a history item."""
    if 0 <= index < len(generation_history):
        deleted = generation_history.pop(index)
        return {"deleted": deleted, "remaining": len(generation_history)}
    raise HTTPException(status_code=404, detail="History item not found")


@app.delete("/api/history")
async def clear_history():
    """Clear all history."""
    global generation_history
    count = len(generation_history)
    generation_history = []
    return {"cleared": count}


@app.post("/api/save-embeddings")
async def save_embeddings_endpoint(request: EncodeRequest):
    """Encode and save embeddings to file for distributed inference."""
    # Use encoder from pipeline or standalone encoder
    enc = encoder if encoder is not None else (pipeline.encoder if pipeline else None)
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

        return {
            "path": str(save_path),
            "shape": list(embeddings.shape),
            "encode_time": encode_time,
        }

    except Exception as e:
        logger.error(f"Save embeddings failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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


# =============================================================================
# Model Management Endpoints
# =============================================================================


@app.get("/api/system/status")
async def system_status():
    """Get detailed system status including memory usage and cached models."""
    status = {
        "pipeline_loaded": pipeline is not None,
        "encoder_loaded": encoder is not None,
        "encoder_only_mode": encoder_only_mode,
        "vl_available": vl_extractor is not None,
        "qwen_image_available": qwen_image_pipeline is not None,
        "qwen_image_t2i_available": qwen_image_t2i_pipeline is not None,
        "ltx2_pipeline": ltx2_pipeline is not None,
        "flux2_pipeline": flux2_pipeline is not None,
        "fmtt_cached": False,
        "vl_cache_count": len(vl_embeddings_cache),
        "history_count": len(generation_history),
    }

    # Check FMTT cache
    if pipeline is not None and hasattr(pipeline, "_fmtt_reward_fn"):
        status["fmtt_cached"] = pipeline._fmtt_reward_fn is not None

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
    if runtime_config is not None:
        config_info = {
            "model_type": runtime_config.model_type,
            "attention_backend": runtime_config.attention_backend or "auto",
        }

        # Add profile if available
        if hasattr(runtime_config, "current_profile"):
            config_info["profile"] = runtime_config.current_profile

        # Z-Image specific config
        if runtime_config.model_type == "zimage":
            config_info["quantization"] = runtime_config.quantization
            config_info["cpu_offload"] = runtime_config.cpu_offload
            config_info["flash_attn"] = runtime_config.flash_attn
            config_info["torch_compile"] = runtime_config.compile
            config_info["tiled_vae"] = getattr(runtime_config, "tiled_vae", False)

        # Qwen-Image specific config (all variants)
        if runtime_config.model_type.startswith("qwenimage"):
            config_info["quantize_text_encoder"] = runtime_config.qwen_image_quantize_text_encoder
            config_info["quantize_transformer"] = (
                runtime_config.get_qwen_image_quantize_transformer()
                if hasattr(runtime_config, "get_qwen_image_quantize_transformer")
                else runtime_config.qwen_image_quantize_transformer or "none"
            )
            config_info["quantize_vae"] = getattr(runtime_config, "qwen_image_quantize_vae", "none")
            config_info["cpu_offload"] = runtime_config.qwen_image_cpu_offload
            # Check for new offload_type if available
            if hasattr(runtime_config, "qwen_image_offload_type"):
                config_info["offload_type"] = runtime_config.qwen_image_offload_type
            else:
                config_info["offload_type"] = (
                    "model" if runtime_config.qwen_image_cpu_offload else "none"
                )

        status["config"] = config_info

    return status


@app.post("/api/system/unload-fmtt")
async def unload_fmtt():
    """Unload cached FMTT reward function (SigLIP) to free GPU memory."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")

    if not hasattr(pipeline, "unload_fmtt"):
        raise HTTPException(
            status_code=501, detail="Pipeline version does not support FMTT unloading"
        )

    was_loaded = pipeline.unload_fmtt()

    if was_loaded:
        # Get updated memory stats
        if torch.cuda.is_available():
            free = torch.cuda.mem_get_info()[0] / 1024**3
            return {"success": True, "message": "FMTT unloaded", "free_gb": round(free, 2)}
        return {"success": True, "message": "FMTT unloaded"}
    else:
        return {"success": False, "message": "No FMTT was cached"}


@app.post("/api/system/clear-cache")
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


@app.delete("/api/system/vl-cache")
async def clear_vl_cache():
    """Clear all cached VL embeddings."""
    global vl_embeddings_cache
    count = len(vl_embeddings_cache)
    vl_embeddings_cache = {}
    return {"cleared": count}


# =============================================================================
# VRAM / Model Management API
# =============================================================================


@app.get("/api/vram/status")
async def vram_status():
    """Get current VRAM usage and loaded models status.

    Returns detailed info about which models are loaded and VRAM consumption.
    Useful for understanding memory pressure before loading additional models.
    """
    return get_vram_status()


def unload_all_pipelines_except(keep: str = None):
    """Unload all pipelines except the specified one to free VRAM.

    Args:
        keep: Name of pipeline to keep loaded ('zimage', 'qwen-image', 'qwen-image-t2i', 'ltx2', 'flux2')
              If None, unloads all pipelines.
    """
    unloaded = []

    # Z-Image
    if keep != "zimage":
        if unload_zimage_pipeline():
            unloaded.append("Z-Image")

    # Qwen-Image Edit
    if keep != "qwen-image":
        if unload_qwen_image_pipeline():
            unloaded.append("Qwen-Image")

    # Qwen-Image T2I
    if keep != "qwen-image-t2i":
        if unload_qwen_image_t2i_pipeline():
            unloaded.append("Qwen-Image T2I")

    # LTX-2
    if keep != "ltx2":
        if unload_ltx2_pipeline():
            unloaded.append("LTX-2")

    # FLUX.2
    if keep != "flux2":
        global flux2_pipeline
        if flux2_pipeline is not None:
            if isinstance(flux2_pipeline, dict):
                for key in list(flux2_pipeline.keys()):
                    del flux2_pipeline[key]
            flux2_pipeline = None
            unloaded.append("FLUX.2")

    if unloaded:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"[VRAM] Unloaded pipelines to free memory: {', '.join(unloaded)}")

    return unloaded


def load_zimage_pipeline_on_demand():
    """Load Z-Image pipeline on-demand using runtime config.

    Returns True if successfully loaded, raises exception on failure.
    Thread-safe: uses a lock to prevent concurrent loading attempts.
    Automatically unloads other pipelines first to free VRAM.
    """
    global pipeline, _zimage_loading_in_progress

    # Fast path: already loaded
    if pipeline is not None:
        return True

    # Acquire lock to prevent concurrent loading
    with _zimage_loading_lock:
        # Check again inside lock (another thread may have loaded it)
        if pipeline is not None:
            return True

        # Check if loading is already in progress (shouldn't happen with lock, but safety check)
        if _zimage_loading_in_progress:
            raise ValueError("Z-Image loading already in progress, please wait")

        if runtime_config is None:
            raise ValueError("Runtime config not initialized")

        # Use zimage_model_path if set, fall back to legacy model_path
        model_path = runtime_config.zimage_model_path or runtime_config.model_path
        if not model_path:
            raise ValueError("Z-Image model_path not configured. Set [zimage].model_path in config.toml")

        # Unload other pipelines first to free VRAM
        unload_all_pipelines_except("zimage")

        _zimage_loading_in_progress = True
        logger.info("[Z-Image] Loading pipeline on-demand...")
        logger.info(f"  Model path: {model_path}")
        logger.info(f"  Variant: {runtime_config.zimage_variant}")

        # Debug: log cpu_offload value to diagnose OOM issues
        cpu_offload_value = getattr(runtime_config, "cpu_offload", "NOT_SET")
        logger.info(f"  cpu_offload from config: {cpu_offload_value}")

        # Determine encoder device - use CPU when cpu_offload is enabled to fit in 24GB VRAM
        # cpu_offload maps to [pipeline].enable_model_cpu_offload in config
        encoder_device = runtime_config.encoder_device
        if getattr(runtime_config, "cpu_offload", False):
            encoder_device = "cpu"
            logger.info("  CPU offload enabled - placing encoder on CPU")

        try:
            load_pipeline(
                model_path=model_path,
                text_encoder_path=runtime_config.text_encoder_path,
                templates_dir=runtime_config.templates_dir,
                encoder_device=encoder_device,
                dit_device=runtime_config.dit_device,
                vae_device=runtime_config.vae_device,
                quantization=runtime_config.quantization,
                lora_paths=runtime_config.lora_paths,
                lora_scales=runtime_config.lora_scales,
                enable_compile=getattr(runtime_config, "compile", False),
                attention_backend=getattr(runtime_config, "attention_backend", "auto"),
            )
            logger.info("[Z-Image] Pipeline loaded successfully")
            return True
        except Exception as e:
            logger.error(f"[Z-Image] Failed to load pipeline: {e}")
            # Clean up partial state on failure to avoid VRAM accumulation
            global encoder
            if pipeline is not None:
                del pipeline
                pipeline = None
            if encoder is not None:
                del encoder
                encoder = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("[Z-Image] Cleaned up VRAM after failed load attempt")
            raise
        finally:
            _zimage_loading_in_progress = False


@app.post("/api/vram/load-zimage")
async def vram_load_zimage():
    """Load Z-Image pipeline on-demand.

    Uses model_path and other settings from config.toml.
    Call this before generating if the pipeline was previously unloaded.
    """
    if pipeline is not None:
        status = get_vram_status()
        return {
            "success": True,
            "message": "Z-Image pipeline already loaded",
            "vram": status.get("vram"),
        }

    model_path = (runtime_config.zimage_model_path or runtime_config.model_path) if runtime_config else None
    if runtime_config is None or not model_path:
        raise HTTPException(
            status_code=400,
            detail="Z-Image model_path not configured. Set [zimage].model_path in config.toml",
        )

    try:
        load_zimage_pipeline_on_demand()
        status = get_vram_status()
        return {
            "success": True,
            "message": "Z-Image pipeline loaded successfully",
            "vram": status.get("vram"),
        }
    except Exception as e:
        logger.error(f"[Z-Image] Failed to load pipeline: {e}")
        raise HTTPException(status_code=503, detail=f"Failed to load Z-Image pipeline: {e}")


@app.post("/api/vram/unload-zimage")
async def vram_unload_zimage():
    """Unload Z-Image pipeline (encoder + DiT + VAE) to free VRAM.

    Use this before loading Qwen-Image models if running low on VRAM.
    The pipeline will be reloaded automatically on next Z-Image generation.
    """
    unloaded = unload_zimage_pipeline()

    status = get_vram_status()
    return {
        "success": unloaded,
        "message": "Z-Image pipeline unloaded" if unloaded else "Z-Image pipeline was not loaded",
        "vram": status.get("vram"),
    }


@app.post("/api/vram/load-qwen-image")
async def vram_load_qwen_image():
    """Load Qwen-Image Edit pipeline on-demand.

    Uses qwen_image.model_path from config.toml.
    Call this before editing if the pipeline was previously unloaded.
    """
    global qwen_image_pipeline

    if qwen_image_pipeline is not None:
        status = get_vram_status()
        return {
            "success": True,
            "message": "Qwen-Image Edit pipeline already loaded",
            "vram": status.get("vram"),
        }

    if runtime_config is None or not runtime_config.qwen_image_model_path:
        raise HTTPException(
            status_code=400,
            detail="Qwen-Image model_path not configured. Set qwen_image.model_path in config.toml",
        )

    # Unload other pipelines first to free VRAM
    unload_all_pipelines_except("qwen-image")

    try:
        from llm_dit.pipelines.qwen_image_diffusers import QwenImageDiffusersPipeline

        quant_te = getattr(runtime_config, "qwen_image_quantize_text_encoder", "none")
        quant_tf = getattr(runtime_config, "qwen_image_quantize_transformer", "none")
        quant_te = quant_te if quant_te != "none" else None
        quant_tf = quant_tf if quant_tf != "none" else None

        logger.info(f"[Qwen-Image] Loading pipeline in edit-only mode...")
        qwen_image_pipeline = QwenImageDiffusersPipeline.from_pretrained(
            runtime_config.qwen_image_model_path,
            edit_model_path=runtime_config.qwen_image_edit_model_path or None,
            cpu_offload=True,
            edit_only=True,
            quantize_text_encoder=quant_te,
            quantize_transformer=quant_tf,
        )
        logger.info("[Qwen-Image] Edit pipeline loaded successfully")

        status = get_vram_status()
        return {
            "success": True,
            "message": "Qwen-Image Edit pipeline loaded successfully",
            "vram": status.get("vram"),
        }
    except Exception as e:
        logger.error(f"[Qwen-Image] Failed to load pipeline: {e}")
        raise HTTPException(status_code=503, detail=f"Failed to load Qwen-Image Edit pipeline: {e}")


@app.post("/api/vram/unload-qwen-image")
async def vram_unload_qwen_image():
    """Unload Qwen-Image pipeline (decompose + edit models) to free VRAM.

    Use this before Z-Image generation if running low on VRAM.
    The pipeline will be reloaded automatically on next Qwen-Image operation.
    """
    unloaded = unload_qwen_image_pipeline()

    status = get_vram_status()
    return {
        "success": unloaded,
        "message": "Qwen-Image pipeline unloaded"
        if unloaded
        else "Qwen-Image pipeline was not loaded",
        "vram": status.get("vram"),
    }


@app.post("/api/vram/load-qwen-image-t2i")
async def vram_load_qwen_image_t2i():
    """Load Qwen-Image T2I pipeline on-demand.

    Uses qwen_image.model_path from config.toml with --model-type qwenimage-t2i.
    Call this before generating if the pipeline was previously unloaded.
    """
    global qwen_image_t2i_pipeline

    if qwen_image_t2i_pipeline is not None:
        status = get_vram_status()
        return {
            "success": True,
            "message": "Qwen-Image T2I pipeline already loaded",
            "vram": status.get("vram"),
        }

    if runtime_config is None or not runtime_config.qwen_image_model_path:
        raise HTTPException(
            status_code=400,
            detail="Qwen-Image model_path not configured. Set qwen_image.model_path in config.toml",
        )

    # Unload other pipelines first to free VRAM
    unload_all_pipelines_except("qwen-image-t2i")

    try:
        from llm_dit.pipelines.qwen_image_2512 import QwenImage2512Pipeline

        quant_transformer = runtime_config.get_qwen_image_quantize_transformer()
        if quant_transformer == "none":
            quant_transformer = None

        quant_text_encoder = runtime_config.qwen_image_quantize_text_encoder
        if quant_text_encoder == "none":
            quant_text_encoder = None

        logger.info(f"[Qwen-Image T2I] Loading pipeline...")
        qwen_image_t2i_pipeline = QwenImage2512Pipeline.from_pretrained(
            runtime_config.qwen_image_model_path,
            quantize_transformer=quant_transformer,
            quantize_text_encoder=quant_text_encoder,
            cpu_offload=runtime_config.qwen_image_cpu_offload,
        )
        logger.info("[Qwen-Image T2I] Pipeline loaded successfully")

        status = get_vram_status()
        return {
            "success": True,
            "message": "Qwen-Image T2I pipeline loaded successfully",
            "vram": status.get("vram"),
        }
    except Exception as e:
        logger.error(f"[Qwen-Image T2I] Failed to load pipeline: {e}")
        raise HTTPException(status_code=503, detail=f"Failed to load Qwen-Image T2I pipeline: {e}")


@app.post("/api/vram/unload-qwen-image-t2i")
async def vram_unload_qwen_image_t2i():
    """Unload Qwen-Image T2I pipeline to free VRAM.

    Use this before loading other models if running low on VRAM.
    The pipeline will be reloaded automatically on next generation request.
    """
    unloaded = unload_qwen_image_t2i_pipeline()

    status = get_vram_status()
    return {
        "success": unloaded,
        "message": "Qwen-Image T2I pipeline unloaded"
        if unloaded
        else "Qwen-Image T2I pipeline was not loaded",
        "vram": status.get("vram"),
    }


@app.post("/api/vram/load-ltx2")
async def vram_load_ltx2():
    """Validate LTX-2 configuration.

    Note: The pure PyTorch pipeline loads components per-request with
    automatic memory offloading. This endpoint validates that the model
    path is configured correctly.

    Uses ltx2.model_path from config.toml or default ~/Storage/LTX-2.
    """
    try:
        # Validate model path exists
        model_path = get_ltx2_model_path()
        status = get_vram_status()
        return {
            "success": True,
            "message": f"LTX-2 model path validated: {model_path}",
            "vram": status.get("vram"),
        }
    except Exception as e:
        logger.error(f"[LTX-2] Configuration validation failed: {e}")
        raise HTTPException(status_code=503, detail=f"LTX-2 configuration error: {e}")


@app.post("/api/vram/unload-ltx2")
async def vram_unload_ltx2():
    """Clean up VRAM after LTX-2 operations.

    Note: The pure PyTorch pipeline automatically unloads components after
    each generation. This endpoint performs a manual memory cleanup.
    """
    unload_ltx2_pipeline()

    status = get_vram_status()
    return {
        "success": True,
        "message": "LTX-2 memory cleanup complete",
        "vram": status.get("vram"),
    }


@app.post("/api/vram/load-flux2")
async def vram_load_flux2():
    """Load FLUX.2 Klein pipeline on-demand.

    Uses flux2.model_path and flux2.vae_path from config.toml.
    Call this before generating if the pipeline was previously unloaded.
    """
    global flux2_pipeline

    if flux2_pipeline is not None:
        status = get_vram_status()
        return {
            "success": True,
            "message": "FLUX.2 pipeline already loaded",
            "vram": status.get("vram"),
        }

    # Check config for paths
    model_path = getattr(runtime_config, "flux2_model_path", None) if runtime_config else None
    vae_path = getattr(runtime_config, "flux2_vae_path", None) if runtime_config else None

    if not model_path:
        raise HTTPException(
            status_code=400,
            detail="FLUX.2 model_path not configured. Set flux2.model_path in config.toml",
        )

    with _flux2_loading_lock:
        # Double-check inside lock (another thread may have loaded it)
        if flux2_pipeline is not None:
            status = get_vram_status()
            return {
                "success": True,
                "message": "FLUX.2 pipeline already loaded",
                "vram": status.get("vram"),
            }

        # Unload other pipelines first to free VRAM
        unload_all_pipelines_except("flux2")

        loaded_encoder = None
        loaded_transformer = None
        loaded_vae = None
        try:
            from pathlib import Path
            from llm_dit.models.flux2.loader import load_flux2_transformer, load_flux2_vae
            from llm_dit.encoders.qwen3_unified import Qwen3UnifiedEncoder
            from llm_dit.models.flux2.constants import get_encoder_preset

            model_path_obj = Path(model_path).expanduser()
            if not model_path_obj.exists():
                raise ValueError(f"FLUX.2 model path does not exist: {model_path}")

            model_name = getattr(runtime_config, "flux2_model_name", "klein-9b") if runtime_config else "klein-9b"
            block_offload = getattr(runtime_config, "flux2_block_offload", False) if runtime_config else False
            quantization = getattr(runtime_config, "flux2_quantization", "none") if runtime_config else "none"
            compile_transformer = getattr(runtime_config, "flux2_compile", False) if runtime_config else False
            compile_vae_flag = getattr(runtime_config, "flux2_compile_vae", False) if runtime_config else False
            compile_mode = getattr(runtime_config, "flux2_compile_mode", "max-autotune-no-cudagraphs") if runtime_config else "max-autotune-no-cudagraphs"
            encoder_path = getattr(runtime_config, "flux2_encoder_path", None) if runtime_config else None

            # Validate incompatible settings before loading anything
            if compile_transformer and block_offload:
                raise ValueError(
                    "compile=true is incompatible with block_offload=true. "
                    "Set block_offload=false when using compile=true."
                )

            logger.info(f"[FLUX.2] Loading pipeline from {model_path} (quantization={quantization}, compile={compile_transformer})")

            # Stage 1: Load encoder
            from llm_dit.models.flux2.constants import FLUX2_MODEL_INFO
            model_info = FLUX2_MODEL_INFO.get(model_name.lower(), {})
            preset = get_encoder_preset(model_name)
            text_encoder_spec = encoder_path or model_info.get("text_encoder", "Qwen/Qwen3-8B")

            logger.info(f"[FLUX.2] Loading encoder from: {text_encoder_spec}")
            loaded_encoder = Qwen3UnifiedEncoder.from_preset(preset, model_path=text_encoder_spec, device="cuda")
            # Offload encoder to CPU with pinned memory for fast GPU shuttle via DMA
            loaded_encoder.offload_to_pinned()
            logger.info("[FLUX.2] Encoder loaded and offloaded to CPU with pinned memory")

            # Stage 2: Load transformer
            logger.info(f"[FLUX.2] Loading transformer: {model_name}")
            loaded_transformer = load_flux2_transformer(
                model_name,
                device="cuda",
                model_path=model_path,
                block_offload=block_offload,
                quantize_to=quantization,
            )

            # Apply torch.compile to transformer if configured
            if compile_transformer and not block_offload:
                logger.info(f"[FLUX.2] Compiling transformer with mode={compile_mode}")
                loaded_transformer = torch.compile(loaded_transformer, mode=compile_mode)
                logger.info("[FLUX.2] Transformer compiled (warmup will occur on first forward pass)")

            # Stage 3: Load VAE
            logger.info(f"[FLUX.2] Loading VAE")
            loaded_vae = load_flux2_vae(model_name, device="cuda", vae_path=vae_path)

            # Apply torch.compile to VAE if configured
            if compile_vae_flag:
                logger.info(f"[FLUX.2] Compiling VAE decoder with mode={compile_mode}")
                loaded_vae.decode = torch.compile(loaded_vae.decode, mode=compile_mode)
                logger.info("[FLUX.2] VAE decoder compiled")

            # Store persistent model references (only after all three succeed)
            flux2_pipeline = {
                "encoder": loaded_encoder,
                "transformer": loaded_transformer,
                "vae": loaded_vae,
                "model_name": model_name,
            }

            status = get_vram_status()
            return {
                "success": True,
                "message": f"FLUX.2 pipeline loaded (quantization={quantization}, compile={compile_transformer})",
                "vram": status.get("vram"),
            }
        except Exception as e:
            # Clean up any partially loaded models
            del loaded_encoder, loaded_transformer, loaded_vae
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.error(f"[FLUX.2] Failed to load pipeline: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=503, detail=f"Failed to load FLUX.2 pipeline: {e}")


@app.post("/api/vram/unload-flux2")
async def vram_unload_flux2():
    """Unload FLUX.2 Klein pipeline to free VRAM.

    FLUX.2 uses ~20GB VRAM for 9B models. Unloading is recommended
    before loading other models like Z-Image or Qwen-Image.
    The pipeline will be reloaded automatically on next image generation request.
    """
    global flux2_pipeline

    unloaded = flux2_pipeline is not None
    if unloaded:
        # Explicitly delete model references to free GPU memory
        if isinstance(flux2_pipeline, dict):
            for key in list(flux2_pipeline.keys()):
                del flux2_pipeline[key]
        flux2_pipeline = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("[FLUX.2] Pipeline unloaded, VRAM freed")

    status = get_vram_status()
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

PIPELINE_LOADERS = {
    "zimage": "vram_load_zimage",
    "z-image": "vram_load_zimage",
    "qwenimage-layered": "vram_load_qwen_image",
    "qwenimage-edit": "vram_load_qwen_image",
    "qwenimage-t2i": "vram_load_qwen_image_t2i",
    "ltx2": "vram_load_ltx2",
    "flux2": "vram_load_flux2",
}

PIPELINE_UNLOADERS = {
    "zimage": "vram_unload_zimage",
    "z-image": "vram_unload_zimage",
    "qwenimage-layered": "vram_unload_qwen_image",
    "qwenimage-edit": "vram_unload_qwen_image",
    "qwenimage-t2i": "vram_unload_qwen_image_t2i",
    "ltx2": "vram_unload_ltx2",
    "flux2": "vram_unload_flux2",
}


@app.post("/api/models/{pipeline_id}/load")
async def load_model_by_id(pipeline_id: str):
    """Load a model by pipeline ID.

    This is the unified API for the React frontend. Maps pipeline IDs to
    the specific load functions (e.g., zimage -> vram_load_zimage).

    Args:
        pipeline_id: Pipeline identifier (zimage, ltx2, flux2, qwenimage-t2i, etc.)

    Returns:
        Load result with VRAM status
    """
    loader_name = PIPELINE_LOADERS.get(pipeline_id.lower())
    if not loader_name:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown pipeline: {pipeline_id}. Available: {list(PIPELINE_LOADERS.keys())}",
        )

    # Get the loader function from globals
    loader_fn = globals().get(loader_name)
    if not loader_fn:
        raise HTTPException(status_code=500, detail=f"Loader function not found: {loader_name}")

    try:
        result = await loader_fn()
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load {pipeline_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/models/{pipeline_id}/unload")
async def unload_model_by_id(pipeline_id: str):
    """Unload a model by pipeline ID.

    Args:
        pipeline_id: Pipeline identifier

    Returns:
        Unload result with VRAM status
    """
    unloader_name = PIPELINE_UNLOADERS.get(pipeline_id.lower())
    if not unloader_name:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown pipeline: {pipeline_id}. Available: {list(PIPELINE_UNLOADERS.keys())}",
        )

    unloader_fn = globals().get(unloader_name)
    if not unloader_fn:
        raise HTTPException(status_code=500, detail=f"Unloader function not found: {unloader_name}")

    try:
        result = await unloader_fn()
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to unload {pipeline_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/models/unload-all")
async def unload_all_models():
    """Unload all loaded models to free VRAM."""
    unload_all_pipelines_except(None)

    status = get_vram_status()
    return {
        "success": True,
        "message": "All models unloaded",
        "vram": status.get("vram"),
    }


def _get_flux2_config_metadata() -> dict:
    """Build config tags and warnings for FLUX.2 pipeline status."""
    if runtime_config is None:
        return {"config_tags": [], "config_warnings": []}

    config_tags = []
    config_warnings = []

    quantization = getattr(runtime_config, "flux2_quantization", "none") or "none"
    compile_enabled = getattr(runtime_config, "flux2_compile", False)
    compile_vae = getattr(runtime_config, "flux2_compile_vae", False)
    compile_mode = getattr(runtime_config, "flux2_compile_mode", "reduce-overhead")
    block_offload = getattr(runtime_config, "flux2_block_offload", False)

    if quantization != "none":
        config_tags.append({"key": "quantization", "label": quantization.upper(), "color": "purple"})
    if compile_enabled:
        config_tags.append({"key": "compile", "label": f"compiled ({compile_mode})", "color": "blue"})
    if compile_vae:
        config_tags.append({"key": "compile_vae", "label": "VAE compiled", "color": "blue"})
    if block_offload:
        config_tags.append({"key": "block_offload", "label": "block offload", "color": "orange"})

    if compile_enabled and block_offload:
        config_warnings.append({
            "severity": "error",
            "message": "compile=true is incompatible with block_offload=true. Loading will fail.",
        })
    if quantization != "none" and block_offload:
        config_warnings.append({
            "severity": "error",
            "message": f"quantization={quantization} is incompatible with block_offload=true. Loading will fail.",
        })

    return {"config_tags": config_tags, "config_warnings": config_warnings}


@app.get("/api/models/{pipeline_id}/status")
async def get_model_status(pipeline_id: str):
    """Get the status of a specific pipeline model.

    Returns whether the model is loaded and its VRAM usage.
    """
    pid = pipeline_id.lower()

    # Check if loaded
    loaded = False
    components = []
    total_vram_mb = 0
    config_meta: dict = {}

    if pid in ("zimage", "z-image"):
        loaded = pipeline is not None
        if loaded:
            components = [
                {"name": "encoder", "vramMB": 8000},  # Approximate
                {"name": "transformer", "vramMB": 8000},
                {"name": "vae", "vramMB": 500},
            ]
            total_vram_mb = sum(c["vramMB"] for c in components)
    elif pid in ("qwenimage-layered", "qwenimage-edit"):
        loaded = qwen_image_pipeline is not None
    elif pid == "qwenimage-t2i":
        loaded = qwen_image_t2i_pipeline is not None
    elif pid == "ltx2":
        loaded = ltx2_pipeline is not None
        if loaded:
            components = [
                {"name": "encoder", "vramMB": 3000},
                {"name": "transformer", "vramMB": 20000},
                {"name": "vae", "vramMB": 1000},
            ]
            total_vram_mb = sum(c["vramMB"] for c in components)
    elif pid == "flux2":
        loaded = flux2_pipeline is not None
        config_meta = _get_flux2_config_metadata()
        if loaded:
            components = [
                {"name": "encoder", "vramMB": 2000},
                {"name": "transformer", "vramMB": 12000},
                {"name": "vae", "vramMB": 500},
            ]
            total_vram_mb = sum(c["vramMB"] for c in components)
    else:
        raise HTTPException(status_code=404, detail=f"Unknown pipeline: {pipeline_id}")

    return {
        "pipeline_id": pipeline_id,
        "status": "loaded" if loaded else "unloaded",
        "components": components if loaded else [],
        "total_vram_mb": total_vram_mb if loaded else 0,
        "vram_mb": total_vram_mb if loaded else 0,
        **config_meta,
    }


# =============================================================================
# LoRA Management API
# =============================================================================


@app.get("/api/loras")
async def list_available_loras():
    """List all available LoRA files from configured directories.

    Scans directories in [lora].paths config for .safetensors files.
    Returns sorted list of relative paths.
    """
    from pathlib import Path

    lora_files = []
    lora_dirs = runtime_config.lora_paths if runtime_config else []

    # If no paths configured, check default locations
    if not lora_dirs:
        lora_dirs = ["loras"]

    for lora_dir in lora_dirs:
        dir_path = Path(lora_dir)
        if not dir_path.exists():
            logger.debug(f"LoRA directory not found: {lora_dir}")
            continue

        # Find all .safetensors files recursively
        for safetensor_file in dir_path.rglob("*.safetensors"):
            # Store relative path from the lora directory
            relative_path = str(safetensor_file)
            lora_files.append({
                "path": relative_path,
                "name": safetensor_file.stem,  # Filename without extension
                "directory": str(safetensor_file.parent),
                "size_mb": round(safetensor_file.stat().st_size / (1024 * 1024), 1),
            })

    # Sort by name (case-insensitive)
    lora_files.sort(key=lambda x: x["name"].lower())

    return {
        "loras": lora_files,
        "directories": lora_dirs,
        "count": len(lora_files),
    }


@app.get("/api/loras/{pipeline_id}")
async def list_loras_for_pipeline(pipeline_id: str):
    """List LoRA files available for a specific pipeline.

    Filters LoRAs by pipeline-specific directories.
    """
    from pathlib import Path

    pipeline_lora_dirs = {
        "flux2": ["loras/FLUX.2-klein", "loras/flux2"],
        "ltx2": ["loras/LTX-2", "loras/ltx2"],
        "zimage": ["loras/Z-Image", "loras/zimage"],
    }

    # Get pipeline-specific dirs or fall back to all configured
    dirs = pipeline_lora_dirs.get(pipeline_id.lower(), runtime_config.lora_paths if runtime_config else ["loras"])

    lora_files = []
    for lora_dir in dirs:
        dir_path = Path(lora_dir)
        if not dir_path.exists():
            continue

        for safetensor_file in dir_path.rglob("*.safetensors"):
            relative_path = str(safetensor_file)
            lora_files.append({
                "path": relative_path,
                "name": safetensor_file.stem,
                "directory": str(safetensor_file.parent),
                "size_mb": round(safetensor_file.stat().st_size / (1024 * 1024), 1),
            })

    lora_files.sort(key=lambda x: x["name"].lower())

    return {
        "loras": lora_files,
        "pipeline_id": pipeline_id,
        "directories": dirs,
        "count": len(lora_files),
    }


# =============================================================================
# Configuration Management API
# =============================================================================


@app.get("/api/configs/available")
async def get_available_configs():
    """List all available configs from modular config file.

    Returns configs grouped by model type with descriptions.
    """
    from pathlib import Path

    import tomllib

    # Config listing is not supported with the current config system
    # Use --config config.toml --profile <profile_name> at server startup
    return {
        "configs": [],
        "config_path": getattr(runtime_config, "config_path", None),
        "current_profile": getattr(runtime_config, "current_profile", None),
        "message": "Dynamic config listing not supported. Use --profile at startup.",
    }


@app.post("/api/configs/load")
async def load_config_dynamic(request: dict):
    """Load a config dynamically without server restart.

    NOTE: Dynamic config loading is not supported with the current config system.
    Use --config config.toml --profile <profile_name> at server startup.
    """
    # Dynamic config loading is not supported with current config system
    return {
        "success": False,
        "error": "Dynamic config loading not supported. Restart server with --profile to change config.",
    }


# =============================================================================
# Config Management Endpoints (Phase 1-3)
# =============================================================================


@app.get("/api/config/session")
async def get_session_config():
    """Get current session configuration values.

    Returns the current runtime_config values, the loaded profile name,
    and which fields have been modified during this session.
    """
    if runtime_config is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    # Get all config values
    values = runtime_config.to_dict()

    # Filter to just hot-reload safe fields for the config UI
    ui_values = {k: v for k, v in values.items() if k in HOT_RELOAD_SAFE}

    return {
        "values": ui_values,
        "profile": getattr(runtime_config, "current_profile", "default"),
        "modified": list(session_modified_fields),
        "config_file": getattr(runtime_config, "config_path", None),
    }


@app.put("/api/config/session")
async def update_session_config(request: dict):
    """Update session defaults (hot-reload safe fields only).

    These changes apply immediately but don't persist to file.
    They last until server restart.
    """
    global session_modified_fields

    if runtime_config is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    updated = []
    rejected = []
    pending_restart = []

    for field, value in request.items():
        if field in HOT_RELOAD_SAFE:
            # Hot-reload: apply immediately
            old_value = getattr(runtime_config, field, None)
            setattr(runtime_config, field, value)
            session_modified_fields.add(field)
            updated.append(field)
            logger.info(f"Session config updated: {field} = {value} (was {old_value})")
        elif field in REQUIRES_RESTART:
            # Requires restart: track for later
            pending_restart_changes[field] = value
            pending_restart.append(field)
            logger.info(f"Config change pending restart: {field} = {value}")
        else:
            rejected.append(field)
            logger.warning(f"Unknown config field rejected: {field}")

    return {
        "success": True,
        "updated": updated,
        "pending_restart": pending_restart,
        "rejected": rejected,
    }


@app.get("/api/config/profiles")
async def list_profiles():
    """List available profiles from config.toml."""
    from pathlib import Path

    import tomllib

    config_path = getattr(runtime_config, "config_path", None)
    if not config_path:
        return {
            "profiles": [],
            "current": getattr(runtime_config, "current_profile", "default"),
            "config_file": None,
            "error": "No config file loaded",
        }

    try:
        config_file = Path(config_path)
        if not config_file.exists():
            return {
                "profiles": [],
                "current": getattr(runtime_config, "current_profile", "default"),
                "config_file": str(config_path),
                "error": f"Config file not found: {config_path}",
            }

        with open(config_file, "rb") as f:
            toml_data = tomllib.load(f)

        # Extract profile names (top-level keys that aren't _metadata)
        profiles = [k for k in toml_data.keys() if not k.startswith("_")]

        return {
            "profiles": profiles,
            "current": getattr(runtime_config, "current_profile", "default"),
            "config_file": str(config_path),
        }
    except Exception as e:
        logger.error(f"Error listing profiles: {e}")
        return {
            "profiles": [],
            "current": getattr(runtime_config, "current_profile", "default"),
            "config_file": str(config_path) if config_path else None,
            "error": str(e),
        }


@app.get("/api/server/status")
async def get_server_status():
    """Get server status including uptime and pending changes."""
    import time

    uptime_seconds = None
    if server_start_time:
        uptime_seconds = int(time.time() - server_start_time)

    return {
        "status": "running",
        "uptime_seconds": uptime_seconds,
        "profile": getattr(runtime_config, "current_profile", "default"),
        "config_file": getattr(runtime_config, "config_path", None),
        "pending_restart": pending_restart_changes,
        "session_modified": list(session_modified_fields),
        "can_restart": True,  # We'll implement restart in Phase 3
    }


@app.post("/api/server/restart")
async def restart_server(request: dict = None):
    """Request server restart.

    This endpoint signals the server to restart. The actual restart
    mechanism depends on how the server was launched (systemd, docker, etc.).
    """
    import os
    import sys

    reason = request.get("reason", "user_request") if request else "user_request"
    new_profile = request.get("new_profile") if request else None

    logger.info(f"Server restart requested: reason={reason}, new_profile={new_profile}")

    # For now, we'll use os.execv to restart the process
    # This works when running directly with python/uv run
    # Note: This won't work in all deployment scenarios

    # Build the restart command
    python = sys.executable
    args = sys.argv.copy()

    # If a new profile was requested, update the args
    if new_profile:
        # Remove existing --profile if present
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

    # Log the restart
    logger.info(f"Restarting server with: {python} {' '.join(args)}")

    # Return response before restarting
    response = {
        "success": True,
        "message": "Server restarting...",
        "new_profile": new_profile,
    }

    # Schedule the restart after a short delay
    import asyncio

    async def do_restart():
        await asyncio.sleep(1)  # Give time for response to be sent
        os.execv(python, [python] + args)

    asyncio.create_task(do_restart())

    return response


def main():
    # Use shared CLI argument parser
    from llm_dit.cli import create_base_parser, load_runtime_config, setup_logging
    from llm_dit.startup import PipelineLoader

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
    global runtime_config, pipeline, encoder, rewriter_backend, encoder_only_mode
    runtime_config = load_runtime_config(args)
    setup_logging(runtime_config)

    # Debug: Log all pipeline configurations
    logger.debug(f"[Config] Z-Image model: {getattr(runtime_config, 'model_path', None)}")
    logger.debug(f"[Config] FLUX.2 model: {getattr(runtime_config, 'flux2_model_path', None)}")
    logger.debug(f"[Config] FLUX.2 VAE: {getattr(runtime_config, 'flux2_vae_path', None)}")
    logger.debug(f"[Config] LTX-2 model: {getattr(runtime_config, 'ltx2_model_path', None)}")
    logger.debug(f"[Config] VL model: {getattr(runtime_config, 'vl_model_path', None)}")
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
        # Use PipelineLoader for Z-Image
        loader = PipelineLoader(runtime_config)
        use_api = getattr(args, "use_api_encoder", False)
        result = loader.auto_load(encoder_only=args.encoder_only, use_api=use_api)
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
        # Use PipelineLoader for Qwen-Image
        loader = PipelineLoader(runtime_config)
        result = loader.auto_load(encoder_only=False, use_api=False)
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
            logger.info("[Qwen-Image] Pipeline loaded via PipelineLoader")

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

    # Initialize VL extractor if configured
    global vl_extractor, vl_rewriter
    if runtime_config.vl_model_path:
        logger.info(f"[VL] Loading Qwen3-VL from {runtime_config.vl_model_path}")
        logger.info(
            f"[VL] Device: {runtime_config.vl_device}, default alpha: {runtime_config.vl_alpha}"
        )
        try:
            from llm_dit.vl import VLEmbeddingExtractor

            # Determine torch dtype
            vl_dtype = torch.bfloat16 if runtime_config.vl_device == "cuda" else torch.float32

            vl_extractor = VLEmbeddingExtractor.from_pretrained(
                runtime_config.vl_model_path,
                device=runtime_config.vl_device,
                dtype=vl_dtype,
            )
            logger.info(f"[VL] Qwen3-VL loaded successfully")
            logger.info(f"[VL] Default blend mode: {runtime_config.vl_blend_mode}")

            # If VL rewriter preload is enabled, share the extractor
            if runtime_config.rewriter_preload_vl:
                vl_rewriter = vl_extractor
                logger.info("[VL Rewrite] Preloaded Qwen3-VL for rewriting (shared with extractor)")
        except Exception as e:
            logger.error(f"[VL] Failed to load Qwen3-VL: {e}")
            logger.warning("[VL] Vision conditioning will be disabled")
            vl_extractor = None
    else:
        logger.info("[VL] No vl_model_path configured, vision conditioning disabled")

        # Log VL rewriter status
        if runtime_config.rewriter_vl_enabled:
            logger.info(
                "[VL Rewrite] VL rewriter enabled but no model configured (on-demand loading)"
            )
        else:
            logger.info("[VL Rewrite] VL rewriter disabled")

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
