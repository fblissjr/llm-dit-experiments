"""Pydantic request/response models for the Z-Image generation server.

All API schemas live here, extracted from server.py for shared use
across routers. Response models are generally plain dicts (FastAPI
serializes them automatically), so this file focuses on request models
with validation.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


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
    model: str = "qwen3-4b"  # Rewriter model to use
    image: Optional[str] = None  # Base64-encoded image (for API VL models)


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
    width: int = Field(768, ge=256, le=1280, description="Video width (snapped to multiple of 32)")
    height: int = Field(512, ge=256, le=1280, description="Video height (snapped to multiple of 32)")

    @field_validator("width", "height")
    @classmethod
    def snap_to_32(cls, v: int) -> int:
        """Snap to nearest multiple of 32 (LTX-2 VAE requirement)."""
        snapped = round(v / 32) * 32
        return max(256, min(1280, snapped))
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
    width: int = Field(1024, ge=256, le=2048, description="Image width (snapped to multiple of 16)")
    height: int = Field(1024, ge=256, le=2048, description="Image height (snapped to multiple of 16)")

    @field_validator("width", "height")
    @classmethod
    def snap_to_16(cls, v: int) -> int:
        """Snap to nearest multiple of 16 (FLUX.2 VAE requirement)."""
        snapped = round(v / 16) * 16
        return max(256, min(2048, snapped))
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
