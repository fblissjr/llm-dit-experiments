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
import io
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import httpx
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
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
    "shift", "shift_terminal", "dynamic_shift", "d_noise",
    # Generation defaults
    "height", "width", "steps", "guidance_scale",
    "cfg_normalization", "cfg_truncation", "cfg_norm_mode",
    # Prompt handling
    "long_prompt_mode", "hidden_layer", "layer_weights",
    "enable_thinking", "default_template",
    "system_prompt", "thinking_content", "assistant_content",
    # DyPE feature params
    "dype_enabled", "dype_method", "dype_scale", "dype_exponent",
    "dype_start_sigma", "dype_base_shift", "dype_max_shift",
    "dype_base_resolution", "dype_anisotropic", "dype_multipass",
    "dype_pass2_strength", "dype_pass3_strength", "dype_frequency_modulation",
    # SLG feature params
    "slg_scale", "slg_layers", "slg_start", "slg_stop",
    # FMTT feature params
    "fmtt_scale", "fmtt_start", "fmtt_stop", "fmtt_normalize",
    "fmtt_decode_scale", "fmtt_siglip_model", "fmtt_siglip_device",
    # Cache settings
    "embedding_cache", "cache_size",
    # Tiled VAE (can change between generations)
    "tiled_vae", "tile_size", "tile_overlap",
    # Seed
    "seed", "negative_prompt",
}

# Fields that require server restart (model reload)
REQUIRES_RESTART = {
    # Model paths
    "model_path", "text_encoder_path", "templates_dir",
    "vl_model_path", "qwen_image_model_path", "qwen_image_edit_model_path",
    # Device placement
    "encoder_device", "dit_device", "vae_device",
    # Quantization
    "quantization", "torch_dtype",
    "qwen_image_quantize_text_encoder", "qwen_image_quantize_transformer",
    # Memory management
    "cpu_offload", "qwen_image_cpu_offload",
    # Attention backend
    "attention_backend", "flash_attn",
    # Compilation
    "compile", "compile_mode",
    # LoRA (requires pipeline reload)
    "lora_paths", "lora_scales",
}


def unload_zimage_pipeline() -> bool:
    """Unload Z-Image pipeline (encoder + DiT + VAE) to free VRAM.

    Returns True if unloaded, False if not loaded.
    """
    global pipeline, encoder
    import gc
    import torch

    unloaded = False
    if pipeline is not None:
        logger.info("[VRAM] Unloading Z-Image pipeline to free VRAM...")
        # Move components to CPU before deletion to release CUDA memory
        try:
            if hasattr(pipeline, 'transformer') and pipeline.transformer is not None:
                pipeline.transformer.to('cpu')
            if hasattr(pipeline, 'vae') and pipeline.vae is not None:
                pipeline.vae.to('cpu')
        except Exception as e:
            logger.warning(f"[VRAM] Error moving pipeline to CPU: {e}")
        del pipeline
        pipeline = None
        unloaded = True

    if encoder is not None:
        logger.info("[VRAM] Unloading Z-Image encoder...")
        # Move encoder model to CPU before deletion
        try:
            if hasattr(encoder, 'backend') and encoder.backend is not None:
                if hasattr(encoder.backend, 'model') and encoder.backend.model is not None:
                    encoder.backend.model.to('cpu')
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
    import gc
    import torch

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


def get_vram_status() -> dict:
    """Get current VRAM usage and loaded models status."""
    import torch

    status = {
        "cuda_available": torch.cuda.is_available(),
        "models_loaded": {
            "zimage_pipeline": pipeline is not None,
            "zimage_encoder": encoder is not None,
            "qwen_image_pipeline": qwen_image_pipeline is not None,
            "qwen_image_edit": qwen_image_pipeline is not None and getattr(qwen_image_pipeline, 'edit_pipe', None) is not None,
            "qwen_image_decompose": qwen_image_pipeline is not None and getattr(qwen_image_pipeline, 'decompose_pipe', None) is not None,
            "qwen_image_t2i_pipeline": qwen_image_t2i_pipeline is not None,
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
    system_prompt: Optional[str] = None  # System prompt (optional)
    thinking_content: Optional[str] = None  # Content inside <think>...</think> (triggers think block)
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
    layer_weights: Optional[Dict[int, float]] = None  # Multi-layer blending weights (overrides hidden_layer)
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
    fmtt_scale: Optional[float] = None  # FMTT scale (0 = disabled, 0.5-2.0 typical)
    fmtt_start: Optional[float] = None  # Start FMTT at this fraction
    fmtt_stop: Optional[float] = None  # Stop FMTT at this fraction
    fmtt_normalize: Optional[str] = None  # Gradient normalization mode: unit, clip, none
    fmtt_decode_scale: Optional[float] = None  # Scale for intermediate VAE decode
    fmtt_siglip_model: Optional[str] = None  # SigLIP model for FMTT
    fmtt_siglip_device: Optional[str] = None  # Device for SigLIP (cuda/cpu)


class Img2ImgRequest(BaseModel):
    """Request for image-to-image generation with optional differential mask.

    Note: SLG, FMTT, DyPE, and layer_weights are not supported for img2img.
    Use text-to-image (/api/generate) for those features.
    """
    prompt: str  # User prompt
    image: str  # Base64-encoded input image
    mask_image: Optional[str] = None  # Base64-encoded grayscale mask (black=preserve, white=edit)
    strength: float = Field(0.75, ge=0.0, le=1.0, description="Denoising strength (0=no change, 1=full generation)")
    # Common generation params
    system_prompt: Optional[str] = None
    thinking_content: Optional[str] = None
    assistant_content: Optional[str] = None
    force_think_block: bool = False
    strip_quotes: bool = False
    width: Optional[int] = Field(None, ge=64, le=4096, description="Output width (if None, use input image size)")
    height: Optional[int] = Field(None, ge=64, le=4096, description="Output height (if None, use input image size)")
    steps: int = Field(9, ge=1, le=500, description="Number of denoising steps")
    seed: Optional[int] = None
    template: Optional[str] = None
    guidance_scale: float = Field(0.0, ge=0.0, le=30.0, description="CFG guidance scale")
    cfg_normalization: float = Field(0.0, ge=0.0, le=10.0, description="CFG normalization strength")
    cfg_truncation: float = Field(1.0, ge=0.0, le=1.0, description="Progress threshold for CFG truncation")
    cfg_norm_mode: str = "clamp"  # CFG normalization mode: clamp or match
    shift: float = Field(3.0, ge=0.0, le=10.0, description="Scheduler shift parameter")
    dynamic_shift: bool = False  # Calculate shift based on resolution (overrides shift)
    d_noise: float = Field(1.0, ge=0.5, le=2.0, description="Sigma scaling (<1.0 = sharper, >1.0 = softer)")
    long_prompt_mode: str = "interpolate"
    hidden_layer: int = Field(-2, ge=-35, le=-1, description="Hidden layer for text embeddings")


class EncodeRequest(BaseModel):
    prompt: str  # User prompt
    system_prompt: Optional[str] = None  # System prompt (optional)
    thinking_content: Optional[str] = None  # Content inside <think>...</think> (triggers think block)
    assistant_content: Optional[str] = None  # Content after </think> (optional)
    force_think_block: bool = False  # If True, add empty think block even without content
    strip_quotes: bool = False  # If True, remove " characters (for JSON-type prompts)
    template: Optional[str] = None


class RewriteRequest(BaseModel):
    prompt: Optional[str] = None  # User prompt to rewrite/expand (optional if image provided)
    rewriter: Optional[str] = None  # Name of rewriter template (optional if custom_system_prompt provided)
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
    edit_only = getattr(runtime_config, 'qwen_image_edit_only', False)
    has_layered = bool(runtime_config.qwen_image_model_path)
    has_edit = bool(getattr(runtime_config, 'qwen_image_edit_model_path', ''))
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
                    status_code=503,
                    detail=f"Failed to load Qwen-Image pipeline: {e}"
                )
        else:
            raise HTTPException(
                status_code=503,
                detail="Qwen-Image pipeline not loaded. Configure qwen_image.model_path in config."
            )

    # Validate resolution
    if request.resolution not in (640, 1024):
        raise HTTPException(
            status_code=400,
            detail=f"Resolution must be 640 or 1024. Got: {request.resolution}"
        )

    try:
        import base64
        import zipfile
        from PIL import Image

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

            layer_data.append({
                "name": layer_name,
                "image": f"data:image/png;base64,{layer_b64}",
                "index": i,
            })

        # Create ZIP file for download
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
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
        import traceback
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
            quant_te = getattr(runtime_config, 'qwen_image_quantize_text_encoder', 'none')
            quant_tf = getattr(runtime_config, 'qwen_image_quantize_transformer', 'none')
            quant_te = quant_te if quant_te != 'none' else None
            quant_tf = quant_tf if quant_tf != 'none' else None

            logger.info(f"[Qwen-Image] Loading pipeline in edit-only mode (quantize_text_encoder={quant_te})...")
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
                    status_code=503,
                    detail=f"Failed to load Qwen-Image pipeline: {e}"
                )
        else:
            raise HTTPException(
                status_code=503,
                detail="Qwen-Image pipeline not loaded. Configure qwen_image.model_path in config."
            )

    # Check if pipeline has edit capability
    if not hasattr(qwen_image_pipeline, 'edit_layer'):
        raise HTTPException(
            status_code=400,
            detail="Pipeline does not support layer editing. Use QwenImageDiffusersPipeline."
        )

    try:
        import base64
        from PIL import Image

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
        import traceback
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

    has_edit_method = hasattr(qwen_image_pipeline, 'edit_layer')
    has_edit_pipe = hasattr(qwen_image_pipeline, 'has_edit_model') and qwen_image_pipeline.has_edit_model

    return {
        "available": has_edit_method,
        "edit_model_loaded": has_edit_pipe,
        "edit_model_path": getattr(qwen_image_pipeline, '_edit_model_path', None),
        "supports_multi_image": hasattr(qwen_image_pipeline, 'edit_multi'),
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
                   "For single-image editing, use /api/qwen-image/edit-layer instead."
        )

    # Check if pipeline is loaded
    if qwen_image_pipeline is None:
        if runtime_config and runtime_config.qwen_image_model_path:
            # Unload Z-Image first to free VRAM for Qwen-Image-Edit
            if pipeline is not None or encoder is not None:
                logger.info("[VRAM] Auto-unloading Z-Image to make room for Qwen-Image-Edit...")
                unload_zimage_pipeline()

            # Get quantization settings from config
            quant_te = getattr(runtime_config, 'qwen_image_quantize_text_encoder', 'none')
            quant_tf = getattr(runtime_config, 'qwen_image_quantize_transformer', 'none')
            quant_te = quant_te if quant_te != 'none' else None
            quant_tf = quant_tf if quant_tf != 'none' else None

            logger.info(f"[Qwen-Image] Loading pipeline in edit-only mode (quantize_text_encoder={quant_te})...")
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
                    status_code=500,
                    detail=f"Failed to load Qwen-Image pipeline: {e}"
                )
        else:
            raise HTTPException(
                status_code=400,
                detail="Qwen-Image pipeline not loaded and no model path configured. "
                       "Set qwen_image.model_path in config.toml."
            )

    # Check if pipeline supports multi-image editing
    if not hasattr(qwen_image_pipeline, 'edit_multi'):
        raise HTTPException(
            status_code=400,
            detail="Pipeline does not support multi-image editing. "
                   "Use QwenImageDiffusersPipeline with Edit-2511 model."
        )

    try:
        # Decode base64 images
        pil_images = []
        for i, img_data in enumerate(request.images):
            try:
                if img_data.startswith('data:'):
                    img_data = img_data.split(',', 1)[1]
                img_bytes = base64.b64decode(img_data)
                img = Image.open(io.BytesIO(img_bytes))
                pil_images.append(img)
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to decode image {i}: {e}"
                )

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
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Qwen-Image] Multi-edit failed: {e}")
        import traceback
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
    return bool(runtime_config.qwen_image_model_path) and runtime_config.model_type == "qwenimage-t2i"


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
        "quantize_transformer": runtime_config.get_qwen_image_quantize_transformer() if runtime_config else "fp8",
        "quantize_text_encoder": runtime_config.qwen_image_quantize_text_encoder if runtime_config else "none",
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
        "quantize_transformer": runtime_config.get_qwen_image_quantize_transformer() if runtime_config else "fp8",
        "quantize_text_encoder": runtime_config.qwen_image_quantize_text_encoder if runtime_config else "none",
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
                import traceback
                traceback.print_exc()
                raise HTTPException(
                    status_code=503,
                    detail=f"Failed to load Qwen-Image T2I pipeline: {e}"
                )
        else:
            raise HTTPException(
                status_code=503,
                detail="Qwen-Image T2I not configured. Use --model-type qwenimage-t2i --qwen-image-model-path"
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
        import base64
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
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Qwen-Image T2I] Generation failed: {e}")
        import traceback
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
            "interpolate",      # RECOMMENDED: compresses all VL tokens
            "adain_per_dim",    # Best for style transfer
            "adain",            # Transfer VL statistics to text
            "linear",           # WARNING: truncates, loses most VL info
            "style_only",       # Blend only style dimensions
            "graduated",        # Graduated alpha per token
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
            status_code=503,
            detail="VL extractor not loaded. Configure vl.model_path in config."
        )

    try:
        import base64
        import hashlib
        from PIL import Image

        # Decode base64 image
        image_data = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        logger.info(f"[VL] Extracting embeddings from {image.size[0]}x{image.size[1]} image")
        logger.info(f"[VL] hidden_layer={request.hidden_layer}, image_tokens_only={request.image_tokens_only}")

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

        logger.info(f"[VL] Extracted {result.num_tokens} tokens in {extract_time:.2f}s -> {cache_id}")

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
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")

    # Get VL embeddings
    vl_emb = None

    if request.vl_embeddings_id:
        # Use cached embeddings
        cached = vl_embeddings_cache.get(request.vl_embeddings_id)
        if cached is None:
            raise HTTPException(
                status_code=404,
                detail=f"Embeddings not found: {request.vl_embeddings_id}"
            )
        vl_emb = cached["embeddings"]
        logger.info(f"[VL] Using cached embeddings: {request.vl_embeddings_id}")

    elif request.vl_image:
        # Extract on-the-fly
        if vl_extractor is None:
            raise HTTPException(
                status_code=503,
                detail="VL extractor not loaded. Configure vl.model_path in config."
            )

        try:
            import base64
            from PIL import Image

            image_data = base64.b64decode(request.vl_image)
            image = Image.open(io.BytesIO(image_data)).convert("RGB")

            logger.info(f"[VL] Extracting embeddings on-the-fly from {image.size[0]}x{image.size[1]} image")

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
                blend_embeddings,
                blend_interpolate,
                blend_style_only,
                blend_per_token,
                blend_adain,
                blend_adain_per_dim,
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
        import base64
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
    dype = getattr(runtime_config, 'dype', None)
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


@app.get("/api/generation-config")
async def get_generation_config():
    """Get generation configuration defaults from server config.

    Returns default values for width, height, steps, shift, long_prompt_mode, hidden_layer, SLG, and FMTT.
    Also returns feature flags indicating which advanced features are enabled in config.
    The UI should call this on load to sync with server config.
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
    dype_enabled = getattr(runtime_config, 'dype_enabled', False)
    slg_enabled = getattr(runtime_config, 'slg_enabled', False) or getattr(runtime_config, 'slg_scale', 0.0) > 0
    fmtt_enabled = getattr(runtime_config, 'fmtt_enabled', False) or getattr(runtime_config, 'fmtt_scale', 0.0) > 0

    return {
        "width": runtime_config.width,
        "height": runtime_config.height,
        "steps": runtime_config.steps,
        "guidance_scale": runtime_config.guidance_scale,
        "shift": runtime_config.shift,
        "dynamic_shift": getattr(runtime_config, 'dynamic_shift', False),
        "d_noise": getattr(runtime_config, 'd_noise', 1.0),
        "long_prompt_mode": runtime_config.long_prompt_mode,
        "hidden_layer": runtime_config.hidden_layer,
        "layer_weights": getattr(runtime_config, 'layer_weights', None),
        # SLG settings
        "slg_scale": getattr(runtime_config, 'slg_scale', 0.0),
        "slg_layers": getattr(runtime_config, 'slg_layers', None),
        "slg_start": getattr(runtime_config, 'slg_start', 0.05),
        "slg_stop": getattr(runtime_config, 'slg_stop', 0.5),
        # FMTT settings
        "fmtt_scale": getattr(runtime_config, 'fmtt_scale', 0.0),
        "fmtt_start": getattr(runtime_config, 'fmtt_start', 0.0),
        "fmtt_stop": getattr(runtime_config, 'fmtt_stop', 0.5),
        "fmtt_normalize": getattr(runtime_config, 'fmtt_normalize', 'unit'),
        "fmtt_decode_scale": getattr(runtime_config, 'fmtt_decode_scale', 0.5),
        "fmtt_siglip_model": getattr(runtime_config, 'fmtt_siglip_model', 'google/siglip2-giant-opt-patch16-384'),
        "fmtt_siglip_device": getattr(runtime_config, 'fmtt_siglip_device', 'cuda'),
        # Feature flags based on config
        "features": {
            "dype_enabled": dype_enabled,
            "slg_enabled": slg_enabled,
            "fmtt_enabled": fmtt_enabled,
            "differential_img2img_enabled": True,  # Always available for Z-Image
        },
    }


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
        VAE_MULTIPLE,
        VAE_SCALE_FACTOR,
        MIN_RESOLUTION,
        MAX_RESOLUTION,
        DEFAULT_RESOLUTION,
        ASPECT_RATIOS,
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
        {"value": "512x512", "label": "512", "width": 512, "height": 512, "category": "square", "ratio": "1:1"},
        {"value": "768x768", "label": "768", "width": 768, "height": 768, "category": "square", "ratio": "1:1"},
        {"value": "1024x1024", "label": "1024", "width": 1024, "height": 1024, "category": "square", "ratio": "1:1", "default": True},
        {"value": "1280x1280", "label": "1280", "width": 1280, "height": 1280, "category": "square", "ratio": "1:1"},
        {"value": "1536x1536", "label": "1536", "width": 1536, "height": 1536, "category": "square", "ratio": "1:1"},
        {"value": "1920x1920", "label": "1920", "width": 1920, "height": 1920, "category": "square", "ratio": "1:1"},
        {"value": "2048x2048", "label": "2K", "width": 2048, "height": 2048, "category": "square", "ratio": "1:1"},
        # Landscape - 16:9
        {"value": "1280x720", "label": "720p", "width": 1280, "height": 720, "category": "landscape", "ratio": "16:9"},
        {"value": "1920x1088", "label": "1080p", "width": 1920, "height": 1088, "category": "landscape", "ratio": "16:9"},
        {"value": "2560x1440", "label": "1440p", "width": 2560, "height": 1440, "category": "landscape", "ratio": "16:9"},
        # Landscape - 3:2
        {"value": "1536x1024", "label": "1536x1024", "width": 1536, "height": 1024, "category": "landscape", "ratio": "3:2"},
        {"value": "1920x1280", "label": "1920x1280", "width": 1920, "height": 1280, "category": "landscape", "ratio": "3:2"},
        # Landscape - 4:3
        {"value": "1024x768", "label": "1024x768", "width": 1024, "height": 768, "category": "landscape", "ratio": "4:3"},
        {"value": "1280x960", "label": "1280x960", "width": 1280, "height": 960, "category": "landscape", "ratio": "4:3"},
        {"value": "1600x1200", "label": "1600x1200", "width": 1600, "height": 1200, "category": "landscape", "ratio": "4:3"},
        # Mobile Landscape - 21:9, 19.5:9 (phone screens rotated)
        {"value": "1792x768", "label": "Ultrawide", "width": 1792, "height": 768, "category": "landscape", "ratio": "21:9"},
        {"value": "2560x1088", "label": "UW 1080", "width": 2560, "height": 1088, "category": "landscape", "ratio": "21:9"},
        {"value": "2340x1080", "label": "Phone HD", "width": 2340, "height": 1080, "category": "landscape", "ratio": "19.5:9"},
        # Portrait - 9:16
        {"value": "720x1280", "label": "720p", "width": 720, "height": 1280, "category": "portrait", "ratio": "9:16"},
        {"value": "1088x1920", "label": "1080p", "width": 1088, "height": 1920, "category": "portrait", "ratio": "9:16"},
        {"value": "1440x2560", "label": "1440p", "width": 1440, "height": 2560, "category": "portrait", "ratio": "9:16"},
        # Portrait - 2:3
        {"value": "1024x1536", "label": "1024x1536", "width": 1024, "height": 1536, "category": "portrait", "ratio": "2:3"},
        {"value": "1280x1920", "label": "1280x1920", "width": 1280, "height": 1920, "category": "portrait", "ratio": "2:3"},
        # Portrait - 3:4
        {"value": "768x1024", "label": "768x1024", "width": 768, "height": 1024, "category": "portrait", "ratio": "3:4"},
        {"value": "960x1280", "label": "960x1280", "width": 960, "height": 1280, "category": "portrait", "ratio": "3:4"},
        {"value": "1200x1600", "label": "1200x1600", "width": 1200, "height": 1600, "category": "portrait", "ratio": "3:4"},
        # Mobile Portrait - 9:19.5, 9:20 (phone screens)
        {"value": "1080x2340", "label": "Phone HD", "width": 1080, "height": 2340, "category": "portrait", "ratio": "9:19.5"},
        {"value": "1284x2778", "label": "iPhone Pro", "width": 1284, "height": 2778, "category": "portrait", "ratio": "9:19.5"},
    ]

    # Qwen-Image-Layered presets (FIXED: only 640 or 1024 square)
    qwenimage_layered_presets = [
        {"value": "640x640", "label": "640 (Fast)", "width": 640, "height": 640, "category": "square", "ratio": "1:1", "default": True},
        {"value": "1024x1024", "label": "1024 (Quality)", "width": 1024, "height": 1024, "category": "square", "ratio": "1:1"},
    ]

    # Qwen-Image T2I presets (flexible, default 1328)
    qwenimage_t2i_presets = [
        {"value": "512x512", "label": "512", "width": 512, "height": 512, "category": "square", "ratio": "1:1"},
        {"value": "768x768", "label": "768", "width": 768, "height": 768, "category": "square", "ratio": "1:1"},
        {"value": "1024x1024", "label": "1024", "width": 1024, "height": 1024, "category": "square", "ratio": "1:1"},
        {"value": "1328x1328", "label": "1328 (Default)", "width": 1328, "height": 1328, "category": "square", "ratio": "1:1", "default": True},
        {"value": "1536x1536", "label": "1536", "width": 1536, "height": 1536, "category": "square", "ratio": "1:1"},
        # Landscape
        {"value": "1328x1024", "label": "1328x1024", "width": 1328, "height": 1024, "category": "landscape", "ratio": "4:3"},
        {"value": "1536x1024", "label": "1536x1024", "width": 1536, "height": 1024, "category": "landscape", "ratio": "3:2"},
        # Portrait
        {"value": "1024x1328", "label": "1024x1328", "width": 1024, "height": 1328, "category": "portrait", "ratio": "3:4"},
        {"value": "1024x1536", "label": "1024x1536", "width": 1024, "height": 1536, "category": "portrait", "ratio": "2:3"},
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
        default_width = getattr(runtime_config, 'width', None) or constraints.get("default_width", 1024)
        default_height = getattr(runtime_config, 'height', None) or constraints.get("default_height", 1024)
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
    if runtime_config and runtime_config.rewriter_vl_api_model and runtime_config.rewriter_vl_enabled:
        vl_api_available = True
        models.append({
            "id": "qwen3-vl-api",
            "name": f"VL via API ({runtime_config.rewriter_vl_api_model})",
            "supports_image": True,
            "loaded": True,  # API is always available
        })

    # Check if local VL rewriter is available
    vl_local_available = False
    vl_loaded = False
    if runtime_config and runtime_config.vl_model_path and runtime_config.rewriter_vl_enabled:
        vl_local_available = True
        vl_loaded = vl_rewriter is not None or vl_extractor is not None
        models.append({
            "id": "qwen3-vl",
            "name": "Qwen3-VL (Vision+Text)",
            "supports_image": True,
            "loaded": vl_loaded,
        })

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
            status_code=400,
            detail="Server running in encoder-only mode. Use /api/encode instead."
        )
    if pipeline is None:
        # Check if it was unloaded for Qwen-Image
        if qwen_image_pipeline is not None:
            raise HTTPException(
                status_code=503,
                detail="Z-Image pipeline was unloaded for Qwen-Image. "
                       "Use the VRAM settings panel to reload Z-Image, or restart the server."
            )
        raise HTTPException(status_code=503, detail="Pipeline not loaded")

    try:
        logger.info("=" * 60)
        logger.info("GENERATION REQUEST")
        logger.info("=" * 60)
        logger.info(f"  Prompt: {request.prompt[:80]}...")
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
        logger.info(f"  pipeline.encoder: {type(pipeline.encoder).__name__ if pipeline.encoder is not None else 'None'}")
        logger.info(f"  pipeline.transformer: {pipeline.transformer is not None}")
        logger.info(f"  pipeline.vae: {pipeline.vae is not None}")
        if pipeline.encoder is not None:
            backend = getattr(pipeline.encoder, 'backend', None)
            logger.info(f"  encoder.backend: {type(backend).__name__ if backend else 'None'}")
        logger.info("-" * 60)

        # Set up generator for reproducibility
        generator = None
        if request.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(request.seed)

        start = time.time()

        # SLG config: "UI always wins" - don't fall back to runtime_config
        # None means disabled, not "use config default"
        slg_scale = request.slg_scale if request.slg_scale is not None else 0.0
        slg_layers = request.slg_layers  # None is valid (means SLG disabled)
        slg_start = request.slg_start if request.slg_start is not None else 0.01
        slg_stop = request.slg_stop if request.slg_stop is not None else 0.2

        # FMTT config: "UI always wins" - don't fall back to runtime_config
        # None means disabled, not "use config default"
        fmtt_scale = request.fmtt_scale if request.fmtt_scale is not None else 0.0
        fmtt_start = request.fmtt_start if request.fmtt_start is not None else 0.0
        fmtt_stop = request.fmtt_stop if request.fmtt_stop is not None else 0.5
        fmtt_normalize = request.fmtt_normalize if request.fmtt_normalize is not None else "unit"
        fmtt_decode_scale = request.fmtt_decode_scale if request.fmtt_decode_scale is not None else 0.5
        fmtt_siglip_model = request.fmtt_siglip_model if request.fmtt_siglip_model is not None else "google/siglip2-giant-opt-patch16-384"
        fmtt_siglip_device = request.fmtt_siglip_device if request.fmtt_siglip_device is not None else "cuda"

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
        logger.info(f"Calling pipeline() with long_prompt_mode={request.long_prompt_mode}, hidden_layer={request.hidden_layer}...")
        if slg_scale > 0 and slg_layers:
            logger.info(f"  SLG: scale={slg_scale}, layers={slg_layers}, range=[{slg_start:.0%}, {slg_stop:.0%}]")
        if fmtt_scale > 0:
            logger.info(f"  FMTT: scale={fmtt_scale}, range=[{fmtt_start:.0%}, {fmtt_stop:.0%}]")
        if dype_config is not None:
            logger.info(f"  DyPE: method={dype_config.method}, scale={dype_config.dype_scale}, exponent={dype_config.dype_exponent}")

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

            logger.info(f"  Multipass: {multipass_mode}, pass2_strength={pass2_strength}, pass3_strength={pass3_strength}")
            image = pipeline.generate_multipass(
                request.prompt,
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
            )
        else:
            # Single pass generation
            image = pipeline(
                request.prompt,
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
            )

        gen_time = time.time() - start
        logger.info(f"Generated in {gen_time:.1f}s")
        logger.info("=" * 60)

        # Convert to PNG bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        # Convert to base64 for history storage
        import base64
        img_bytes_copy = io.BytesIO()
        image.save(img_bytes_copy, format="PNG")
        img_b64 = base64.b64encode(img_bytes_copy.getvalue()).decode("ascii")

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

        return StreamingResponse(
            img_bytes,
            media_type="image/png",
            headers={
                "X-Generation-Time": str(gen_time),
                "X-Seed": str(request.seed) if request.seed else "random",
                "X-History-Id": str(history_entry["id"]),
            },
        )

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/img2img")
async def img2img(request: Img2ImgRequest):
    """Generate an image from an input image with optional differential mask.

    The mask controls per-pixel edit strength:
    - Black (0): Preserve original
    - White (255): Allow full editing
    - Gray: Partial editing
    """
    if encoder_only_mode:
        raise HTTPException(
            status_code=400,
            detail="Server running in encoder-only mode. Img2img not available."
        )
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")

    try:
        import base64
        import binascii
        from PIL import Image as PILImage
        from PIL import UnidentifiedImageError

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

        start = time.time()

        # Generate image using img2img
        # Note: SLG, FMTT, DyPE, and layer_weights are not supported in img2img
        logger.info(f"Calling pipeline.img2img with strength={request.strength}...")
        if mask_image:
            logger.info("  Using differential diffusion with mask")

        image = pipeline.img2img(
            prompt=request.prompt,
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
        )

        gen_time = time.time() - start
        logger.info(f"Generated in {gen_time:.1f}s")
        logger.info("=" * 60)

        # Convert to PNG bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        # Store in history
        img_bytes_copy = io.BytesIO()
        image.save(img_bytes_copy, format="PNG")
        img_b64 = base64.b64encode(img_bytes_copy.getvalue()).decode("ascii")

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

        return StreamingResponse(
            img_bytes,
            media_type="image/png",
            headers={
                "X-Generation-Time": str(gen_time),
                "X-Seed": str(request.seed) if request.seed else "random",
                "X-History-Id": str(history_entry["id"]),
                "X-Mode": "differential" if request.mask_image else "standard",
            },
        )

    except Exception as e:
        logger.error(f"Img2img failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/format-prompt")
async def format_prompt_endpoint(request: EncodeRequest):
    """Preview the formatted prompt without encoding (fast, no GPU needed)."""
    # Use encoder from pipeline or standalone encoder
    enc = encoder if encoder is not None else (pipeline.encoder if pipeline else None)
    if enc is None:
        raise HTTPException(status_code=503, detail="Encoder not loaded")

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
        if hasattr(enc, 'backend') and hasattr(enc.backend, 'tokenizer'):
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
    if pipeline is not None and hasattr(pipeline, 'encoder'):
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
            templates.append({
                "name": name,
                "description": tpl.description or "",
                "category": tpl.category or "general",
                "system_prompt": tpl.content or "",
                "thinking_content": tpl.thinking_content or "",
                "assistant_content": tpl.assistant_content or "",
                "add_think_block": tpl.add_think_block,
            })

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
        rewriters.append({
            "name": tpl.name,
            "description": tpl.description,
        })

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
            detail="VL API model not configured. Set rewriter.vl_api_model in config.toml."
        )

    if not runtime_config.rewriter_vl_enabled:
        raise HTTPException(
            status_code=400,
            detail="VL rewriter is disabled. Enable with rewriter.vl_enabled=true in config."
        )

    # Determine API URL
    api_url = runtime_config.rewriter_api_url or runtime_config.api_url
    if not api_url:
        raise HTTPException(
            status_code=400,
            detail="No API URL configured. Set rewriter.api_url or api.url in config.toml."
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
                status_code=404,
                detail=f"Rewriter template not found: {request.rewriter}"
            )

        if rewriter_template.category != "rewriter":
            raise HTTPException(
                status_code=400,
                detail=f"Template '{request.rewriter}' is not a rewriter template"
            )

        system_prompt = rewriter_template.content
        rewriter_name = request.rewriter
    else:
        # Use a default system prompt for VL rewriting
        system_prompt = "Describe what you see in this image in detail, suitable for use as an image generation prompt."
        rewriter_name = "default_vl"

    # Get generation parameters (use 'is not None' to preserve 0 values)
    max_tokens = request.max_tokens if request.max_tokens is not None else (runtime_config.rewriter_max_tokens if runtime_config else 1024)
    temperature = request.temperature if request.temperature is not None else (runtime_config.rewriter_temperature if runtime_config else 0.6)
    top_p = request.top_p if request.top_p is not None else (runtime_config.rewriter_top_p if runtime_config else 0.95)
    top_k = request.top_k if request.top_k is not None else (runtime_config.rewriter_top_k if runtime_config else 20)
    min_p = request.min_p if request.min_p is not None else (runtime_config.rewriter_min_p if runtime_config else 0.0)
    presence_penalty = request.presence_penalty if request.presence_penalty is not None else (runtime_config.rewriter_presence_penalty if runtime_config else 0.0)

    try:
        start = time.time()
        logger.info(f"[VL API Rewrite] Using: {rewriter_name} (model: {runtime_config.rewriter_vl_api_model})")
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

        think_match = re.search(r'<think>\s*(.*?)\s*</think>', generated, re.DOTALL)
        if think_match:
            thinking_content = think_match.group(1).strip()
            rewritten_prompt = re.sub(r'<think>.*?</think>\s*', '', generated, flags=re.DOTALL).strip()
            logger.info(f"[VL API Rewrite] Extracted thinking ({len(thinking_content)} chars)")

        # Clean up any remaining tags
        if thinking_content:
            thinking_content = re.sub(r'</?think>', '', thinking_content).strip()
        if rewritten_prompt:
            rewritten_prompt = re.sub(r'</?think>', '', rewritten_prompt).strip()
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
            detail=f"API request timed out after {timeout}s. Try increasing rewriter.timeout in config.toml."
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
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"API error: {error_detail}"
        )
    except httpx.ConnectError as e:
        logger.error(f"[VL API Rewrite] Connection failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to API at {api_url}. Is the server running?"
        )
    except Exception as e:
        logger.error(f"[VL API Rewrite] Failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _rewrite_with_vl(request: RewriteRequest) -> dict:
    """
    Handle VL-based rewriting (image+text or image-only).

    Loads Qwen3-VL on-demand if not already loaded.
    """
    import base64
    from PIL import Image

    global vl_rewriter, vl_extractor

    # Check if VL is available
    if not runtime_config or not runtime_config.vl_model_path:
        raise HTTPException(
            status_code=400,
            detail="VL model not configured. Set vl.model_path in config.toml."
        )

    if not runtime_config.rewriter_vl_enabled:
        raise HTTPException(
            status_code=400,
            detail="VL rewriter is disabled. Enable with rewriter.vl_enabled=true in config."
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
                torch_dtype=vl_dtype,
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
            raise HTTPException(
                status_code=400,
                detail=f"Failed to decode image: {e}"
            )

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
                status_code=404,
                detail=f"Rewriter template not found: {request.rewriter}"
            )

        if rewriter_template.category != "rewriter":
            raise HTTPException(
                status_code=400,
                detail=f"Template '{request.rewriter}' is not a rewriter template"
            )

        system_prompt = rewriter_template.content
        rewriter_name = request.rewriter
    else:
        # Use a default system prompt for VL rewriting
        system_prompt = "Describe what you see in this image in detail, suitable for use as an image generation prompt."
        rewriter_name = "default_vl"

    # Get generation parameters (use 'is not None' to preserve 0 values)
    max_tokens = request.max_tokens if request.max_tokens is not None else (runtime_config.rewriter_max_tokens if runtime_config else 1024)
    temperature = request.temperature if request.temperature is not None else (runtime_config.rewriter_temperature if runtime_config else 0.6)
    top_p = request.top_p if request.top_p is not None else (runtime_config.rewriter_top_p if runtime_config else 0.95)
    top_k = request.top_k if request.top_k is not None else (runtime_config.rewriter_top_k if runtime_config else 20)
    min_p = request.min_p if request.min_p is not None else (runtime_config.rewriter_min_p if runtime_config else 0.0)
    presence_penalty = request.presence_penalty if request.presence_penalty is not None else (runtime_config.rewriter_presence_penalty if runtime_config else 0.0)

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

        think_match = re.search(r'<think>\s*(.*?)\s*</think>', generated, re.DOTALL)
        if think_match:
            thinking_content = think_match.group(1).strip()
            rewritten_prompt = re.sub(r'<think>.*?</think>\s*', '', generated, flags=re.DOTALL).strip()
            logger.info(f"[VL Rewrite] Extracted thinking ({len(thinking_content)} chars)")

        # Clean up any remaining tags
        if thinking_content:
            thinking_content = re.sub(r'</?think>', '', thinking_content).strip()
        if rewritten_prompt:
            rewritten_prompt = re.sub(r'</?think>', '', rewritten_prompt).strip()
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
            status_code=400,
            detail="At least one of 'prompt' or 'image' must be provided"
        )

    # Handle VL model selection
    if request.model == "qwen3-vl":
        return await _rewrite_with_vl(request)
    elif request.model == "qwen3-vl-api":
        return await _rewrite_with_vl_api(request)

    # If image provided but model is not VL, warn and ignore
    if request.image:
        logger.warning("[Rewrite] Image provided but model is qwen3-4b (text-only). Image will be ignored.")

    # Require prompt for text-only model
    if not request.prompt:
        raise HTTPException(
            status_code=400,
            detail="Text prompt is required for qwen3-4b model. Use qwen3-vl for image-only rewriting."
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
            backend = getattr(enc, 'backend', None)
            backend_name = "local"

    if backend is None:
        raise HTTPException(status_code=503, detail="No backend available for generation")

    if not getattr(backend, 'supports_generation', False):
        raise HTTPException(
            status_code=400,
            detail="Backend does not support text generation"
        )

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
                status_code=404,
                detail=f"Rewriter template not found: {request.rewriter}"
            )

        if rewriter_template.category != "rewriter":
            raise HTTPException(
                status_code=400,
                detail=f"Template '{request.rewriter}' is not a rewriter template"
            )

        system_prompt = rewriter_template.content
        rewriter_name = request.rewriter
    else:
        raise HTTPException(
            status_code=400,
            detail="Either 'rewriter' or 'custom_system_prompt' must be provided"
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
        logger.info(f"[Rewrite] Params: max_tokens={max_tokens}, temperature={temperature}, top_p={top_p}, top_k={top_k}, min_p={min_p}, presence_penalty={presence_penalty}")

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
        think_match = re.search(r'<think>\s*(.*?)\s*</think>', generated, re.DOTALL)
        if think_match:
            thinking_content = think_match.group(1).strip()
            # Remove the think block from the rewritten prompt
            rewritten_prompt = re.sub(r'<think>.*?</think>\s*', '', generated, flags=re.DOTALL).strip()
            logger.info(f"[Rewrite] Extracted thinking via <think> tags ({len(thinking_content)} chars), prompt ({len(rewritten_prompt)} chars)")
        else:
            # No think tags - try to find JSON at the end and treat preceding text as thinking
            # Look for a JSON object (starts with { and ends with })
            json_match = re.search(r'(\{[\s\S]*\})\s*$', generated)
            if json_match:
                json_text = json_match.group(1)
                # Everything before the JSON is reasoning/thinking
                pre_json = generated[:json_match.start()].strip()
                if pre_json:
                    thinking_content = pre_json
                    rewritten_prompt = json_text
                    logger.info(f"[Rewrite] Extracted thinking via JSON detection ({len(thinking_content)} chars), JSON prompt ({len(rewritten_prompt)} chars)")
            # If output starts with reasoning patterns like "Okay," "Let me", etc. and has a clear break
            elif re.match(r'^(Okay|Let me|I need|First|The user|Looking)', generated):
                # Look for double newline as separator between thinking and output
                parts = re.split(r'\n\n+', generated, maxsplit=1)
                if len(parts) == 2 and len(parts[1]) > 50:
                    # If second part is substantial, treat first as thinking
                    # But only if second part looks like a prompt (not more reasoning)
                    if not re.match(r'^(Okay|Let me|I need|First|The user|Looking|Now)', parts[1]):
                        thinking_content = parts[0].strip()
                        rewritten_prompt = parts[1].strip()
                        logger.info(f"[Rewrite] Extracted thinking via paragraph split ({len(thinking_content)} chars), prompt ({len(rewritten_prompt)} chars)")

        # Defense in depth: strip any remaining <think>/<think> tags from both outputs
        # This handles edge cases where tags might be nested or malformed
        if thinking_content:
            thinking_content = re.sub(r'</?think>', '', thinking_content).strip()
        if rewritten_prompt:
            rewritten_prompt = re.sub(r'</?think>', '', rewritten_prompt).strip()
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
        device = str(enc.device) if hasattr(enc, 'device') else 'unknown'

        save_path = save_emb(
            embeddings=embeddings,
            path=str(output_path),
            prompt=request.prompt,
            model_path='unknown',  # Not stored in encoder
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
    start = time.time()

    pipeline = ZImagePipeline.from_pretrained(
        model_path,
        text_encoder_path=text_encoder_path,
        templates_dir=templates_dir,
        torch_dtype=torch.bfloat16,
        encoder_device=encoder_device,
        dit_device=dit_device,
        vae_device=vae_device,
        quantization=quantization,
    )

    load_time = time.time() - start
    logger.info(f"Pipeline loaded in {load_time:.1f}s")
    logger.info(f"Device: {pipeline.device}")

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
        torch_dtype=torch.bfloat16,
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
        torch_dtype=torch.bfloat16,
        encoder_device=encoder_device,
        dit_device=dit_device,
        vae_device=vae_device,
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
        logger.info("Compiling transformer with torch.compile...")
        try:
            pipeline.transformer = torch.compile(pipeline.transformer, mode="reduce-overhead")
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
        torch_dtype=torch.bfloat16,
        enable_cpu_offload=enable_cpu_offload,
        dit_device=dit_device,
        vae_device=vae_device,
    )

    load_time = time.time() - start
    logger.info(f"  DiT/VAE loaded in {load_time:.1f}s")
    logger.info(f"  Transformer device: {pipeline.transformer.device if pipeline.transformer else 'None'}")
    logger.info(f"  Transformer dtype: {next(pipeline.transformer.parameters()).dtype if pipeline.transformer else 'None'}")
    logger.info(f"  VAE device: {next(pipeline.vae.parameters()).device if pipeline.vae else 'None'}")

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
        logger.info("Compiling transformer with torch.compile...")
        try:
            pipeline.transformer = torch.compile(pipeline.transformer, mode="reduce-overhead")
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
    import gc

    status = {
        "pipeline_loaded": pipeline is not None,
        "encoder_loaded": encoder is not None,
        "encoder_only_mode": encoder_only_mode,
        "vl_available": vl_extractor is not None,
        "qwen_image_available": qwen_image_pipeline is not None,
        "fmtt_cached": False,
        "vl_cache_count": len(vl_embeddings_cache),
        "history_count": len(generation_history),
    }

    # Check FMTT cache
    if pipeline is not None and hasattr(pipeline, '_fmtt_reward_fn'):
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
        if hasattr(runtime_config, 'current_profile'):
            config_info["profile"] = runtime_config.current_profile

        # Z-Image specific config
        if runtime_config.model_type == "zimage":
            config_info["quantization"] = runtime_config.quantization
            config_info["cpu_offload"] = runtime_config.cpu_offload
            config_info["flash_attn"] = runtime_config.flash_attn
            config_info["torch_compile"] = runtime_config.compile
            config_info["tiled_vae"] = getattr(runtime_config, 'tiled_vae', False)

        # Qwen-Image specific config (all variants)
        if runtime_config.model_type.startswith("qwenimage"):
            config_info["quantize_text_encoder"] = runtime_config.qwen_image_quantize_text_encoder
            config_info["quantize_transformer"] = (
                runtime_config.get_qwen_image_quantize_transformer()
                if hasattr(runtime_config, 'get_qwen_image_quantize_transformer')
                else runtime_config.qwen_image_quantize_transformer or "none"
            )
            config_info["quantize_vae"] = getattr(runtime_config, 'qwen_image_quantize_vae', "none")
            config_info["cpu_offload"] = runtime_config.qwen_image_cpu_offload
            # Check for new offload_type if available
            if hasattr(runtime_config, 'qwen_image_offload_type'):
                config_info["offload_type"] = runtime_config.qwen_image_offload_type
            else:
                config_info["offload_type"] = "model" if runtime_config.qwen_image_cpu_offload else "none"

        status["config"] = config_info

    return status


@app.post("/api/system/unload-fmtt")
async def unload_fmtt():
    """Unload cached FMTT reward function (SigLIP) to free GPU memory."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")

    if not hasattr(pipeline, 'unload_fmtt'):
        raise HTTPException(status_code=501, detail="Pipeline version does not support FMTT unloading")

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
    import gc

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
        "message": "Qwen-Image pipeline unloaded" if unloaded else "Qwen-Image pipeline was not loaded",
        "vram": status.get("vram"),
    }


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
        "message": "Qwen-Image T2I pipeline unloaded" if unloaded else "Qwen-Image T2I pipeline was not loaded",
        "vram": status.get("vram"),
    }


# =============================================================================
# Configuration Management API
# =============================================================================


@app.get("/api/configs/available")
async def get_available_configs():
    """List all available configs from modular config file.

    Returns configs grouped by model type with descriptions.
    """
    import tomllib
    from pathlib import Path

    # Config listing is not supported with the current config system
    # Use --config config.toml --profile <profile_name> at server startup
    return {
        "configs": [],
        "config_path": getattr(runtime_config, 'config_path', None),
        "current_profile": getattr(runtime_config, 'current_profile', None),
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
        "profile": getattr(runtime_config, 'current_profile', 'default'),
        "modified": list(session_modified_fields),
        "config_file": getattr(runtime_config, 'config_path', None),
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
    import tomllib
    from pathlib import Path

    config_path = getattr(runtime_config, 'config_path', None)
    if not config_path:
        return {
            "profiles": [],
            "current": getattr(runtime_config, 'current_profile', 'default'),
            "config_file": None,
            "error": "No config file loaded",
        }

    try:
        config_file = Path(config_path)
        if not config_file.exists():
            return {
                "profiles": [],
                "current": getattr(runtime_config, 'current_profile', 'default'),
                "config_file": str(config_path),
                "error": f"Config file not found: {config_path}",
            }

        with open(config_file, "rb") as f:
            toml_data = tomllib.load(f)

        # Extract profile names (top-level keys that aren't _metadata)
        profiles = [k for k in toml_data.keys() if not k.startswith("_")]

        return {
            "profiles": profiles,
            "current": getattr(runtime_config, 'current_profile', 'default'),
            "config_file": str(config_path),
        }
    except Exception as e:
        logger.error(f"Error listing profiles: {e}")
        return {
            "profiles": [],
            "current": getattr(runtime_config, 'current_profile', 'default'),
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
        "profile": getattr(runtime_config, 'current_profile', 'default'),
        "config_file": getattr(runtime_config, 'config_path', None),
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
    import sys
    import os

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

    args = parser.parse_args()

    # Load unified config (handles TOML + CLI overrides)
    global runtime_config, pipeline, encoder, rewriter_backend, encoder_only_mode
    runtime_config = load_runtime_config(args)
    setup_logging(runtime_config)

    # Store config path for reference
    if hasattr(args, 'config') and args.config:
        runtime_config.config_path = args.config
    if hasattr(args, 'profile') and args.profile:
        runtime_config.current_profile = args.profile

    # Validate model path (unless using API-only mode, or Qwen-Image mode)
    has_model = runtime_config.model_path or runtime_config.api_url
    has_qwen_image = bool(runtime_config.qwen_image_model_path)
    if not has_model and not has_qwen_image:
        logger.error("No model path specified. Use --model-path or --config.")
        return 1

    # Use PipelineLoader for unified loading
    loader = PipelineLoader(runtime_config)
    use_api = getattr(args, "use_api_encoder", False)

    # Load using PipelineLoader.auto_load()
    result = loader.auto_load(
        encoder_only=args.encoder_only,
        use_api=use_api,
    )

    # Set global state from load result
    pipeline = result.pipeline
    encoder = result.encoder
    encoder_only_mode = result.mode in ("encoder_only", "api_encoder")
    mode = result.mode

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
            logger.info(f"[Rewriter] API backend configured: {rewriter_url} (model: {runtime_config.rewriter_api_model})")
            logger.info(f"[Rewriter] Defaults: temperature={runtime_config.rewriter_temperature}, top_p={runtime_config.rewriter_top_p}, max_tokens={runtime_config.rewriter_max_tokens}")
        else:
            logger.warning("[Rewriter] use_api=True but no API URL configured. Using local model.")

    # Initialize VL extractor if configured
    global vl_extractor, vl_rewriter
    if runtime_config.vl_model_path:
        logger.info(f"[VL] Loading Qwen3-VL from {runtime_config.vl_model_path}")
        logger.info(f"[VL] Device: {runtime_config.vl_device}, default alpha: {runtime_config.vl_alpha}")
        try:
            from llm_dit.vl import VLEmbeddingExtractor

            # Determine torch dtype
            vl_dtype = torch.bfloat16 if runtime_config.vl_device == "cuda" else torch.float32

            vl_extractor = VLEmbeddingExtractor.from_pretrained(
                runtime_config.vl_model_path,
                device=runtime_config.vl_device,
                torch_dtype=vl_dtype,
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
            logger.info("[VL Rewrite] VL rewriter enabled but no model configured (on-demand loading)")
        else:
            logger.info("[VL Rewrite] VL rewriter disabled")

    # Run server
    import uvicorn
    import time

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
