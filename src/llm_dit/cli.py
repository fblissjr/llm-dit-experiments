"""
Shared CLI argument parsing and configuration loading.

This module provides a unified interface for both web/server.py and scripts/generate.py,
ensuring feature parity across all entry points.

Usage:
    from llm_dit.cli import create_base_parser, load_runtime_config, RuntimeConfig

    # In your script's main():
    parser = create_base_parser()
    parser.add_argument("--my-script-specific-arg", ...)
    args = parser.parse_args()
    config = load_runtime_config(args)
"""

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, get_args

import torch

from .config import Config

# Supported model types
ModelType = Literal["zimage", "qwenimage-layered", "qwenimage-t2i", "qwenimage-edit", "ltx2", "wan", "flux2"]
SUPPORTED_MODEL_TYPES: tuple[str, ...] = get_args(ModelType)

logger = logging.getLogger(__name__)


def parse_lora_arg(lora_str: str) -> tuple[str, float]:
    """
    Parse a LoRA argument in the format 'path:scale' or just 'path'.

    Args:
        lora_str: LoRA specification like '/path/to/lora.safetensors:0.8'

    Returns:
        Tuple of (path, scale)

    Examples:
        >>> parse_lora_arg('/path/to/lora.safetensors:0.8')
        ('/path/to/lora.safetensors', 0.8)
        >>> parse_lora_arg('/path/to/lora.safetensors')
        ('/path/to/lora.safetensors', 1.0)
    """
    if ":" in lora_str:
        # Find the last colon (in case path contains colons on Windows)
        last_colon = lora_str.rfind(":")
        path = lora_str[:last_colon]
        try:
            scale = float(lora_str[last_colon + 1 :])
        except ValueError:
            # Not a valid float, treat entire string as path
            path = lora_str
            scale = 1.0
    else:
        path = lora_str
        scale = 1.0
    return path, scale


def parse_layer_weights(weights_str: str) -> dict[int, float]:
    """
    Parse a layer weights string into a dict.

    Args:
        weights_str: Layer weights like '-2:0.7,-6:0.3'

    Returns:
        Dict mapping layer indices to weights

    Examples:
        >>> parse_layer_weights('-2:0.7,-6:0.3')
        {-2: 0.7, -6: 0.3}
        >>> parse_layer_weights('-1:0.33,-2:0.34,-3:0.33')
        {-1: 0.33, -2: 0.34, -3: 0.33}
    """
    weights = {}
    for part in weights_str.split(","):
        part = part.strip()
        if ":" not in part:
            raise ValueError(f"Invalid layer weight format: '{part}'. Expected 'layer:weight'")
        layer_str, weight_str = part.rsplit(":", 1)
        try:
            layer = int(layer_str)
            weight = float(weight_str)
        except ValueError as e:
            raise ValueError(
                f"Invalid layer weight: '{part}'. Layer must be int, weight must be float"
            ) from e
        weights[layer] = weight
    return weights


@dataclass
class RuntimeConfig:
    """
    Unified runtime configuration combining TOML config + CLI overrides.

    This is the single source of truth used by both web server and CLI scripts.
    """

    # Startup pipeline selection
    # Which pipeline to load at server startup (none = on-demand loading)
    default_pipeline: str = "none"  # none, z-image, qwen-image, flux2, ltx2

    # Model type selection (for CLI scripts)
    model_type: str = "zimage"  # zimage, qwenimage-layered, qwenimage-t2i, qwenimage-edit

    # Model paths (Z-Image)
    model_path: str = ""
    text_encoder_path: str | None = None  # If None, uses model_path/text_encoder/
    templates_dir: str | None = None

    # Qwen-Image unified settings (all variants: t2i, edit, layered)
    # Use --model-type to select variant: qwenimage-t2i, qwenimage-edit, qwenimage-layered
    # Variant-specific defaults are applied via get_qwen_variant_defaults()
    qwen_image_model_path: str = ""  # Path to any Qwen-Image model
    qwen_image_edit_model_path: str = ""  # Legacy: Path to Edit model (deprecated, use model_path)
    qwen_image_edit_only: bool = False  # Legacy: load edit model directly
    qwen_image_cpu_offload: bool = True  # Enable CPU offload (required for RTX 4090)
    qwen_image_layer_num: int = 4  # Number of decomposition layers (layered variant only)
    qwen_image_cfg_scale: float = 4.0  # CFG scale (4.0 for all variants)
    qwen_image_steps: int | None = None  # Diffusion steps (None = variant default: 40/25/50)
    qwen_image_resolution: int | None = None  # Resolution (None = variant default: 1024/640/640)
    qwen_image_quantize_text_encoder: str = (
        "none"  # none/4bit/8bit - CPU offload makes quant optional
    )
    qwen_image_quantize_transformer: str | None = None  # None = variant default (fp8/diffsynth-fp8)

    # LTX-2 video generation
    ltx2_model_path: str = ""  # Path to LTX-2 model directory
    ltx2_encoder_model_id: str = "models/LTX-2/text_encoder"  # Gemma 3 encoder
    ltx2_num_frames: int = 33  # Number of frames (33-65 typical for 24GB)
    ltx2_fps: int = 24  # Output framerate
    ltx2_guidance_scale: float = 3.5  # CFG scale (3.0-4.0 recommended)
    ltx2_steps: int | None = None  # Diffusion steps (None = config default: 12 for distilled)
    ltx2_lora_path: str = ""  # Path to LoRA safetensors
    ltx2_lora_scale: float = 1.0  # LoRA blend scale
    ltx2_audio: bool = False  # Enable audio generation
    ltx2_output_path: str = "output.mp4"  # Output video path
    ltx2_save_embeddings: str | None = None  # Save embeddings to file (skip generation)
    ltx2_load_embeddings: str | None = None  # Load embeddings from file (skip encoding)

    # LTX-2 optimization (matching reference repo)
    ltx2_text_encoder_device: str = "cpu"  # cpu recommended for 24GB
    ltx2_transformer_device: str = "cuda"  # DiT on GPU
    ltx2_vae_device: str = "cuda"  # VAE on GPU
    ltx2_quantize: str = "fp8"  # fp8 or none
    ltx2_skip_cleanup: bool = False  # Skip memory cleanup between stages
    ltx2_gemma_variant: str = "bf16"  # bf16, 8bit, q4-qat - Gemma3 backbone variant

    # Wan/HuMo video generation
    wan_humo_path: str = ""  # Path to HuMo transformer (e.g., ~/Storage/HuMo)
    wan_base_path: str = ""  # Path to Wan2.1-T2V for VAE/text encoder
    wan_whisper_path: str = ""  # Path to Whisper (optional, for audio)
    wan_humo_variant: str = "17B"  # "17B" or "1.7B"
    wan_num_frames: int = 97  # Number of frames (97 = ~3.9s at 25fps, HuMo default)
    wan_fps: int = 25  # Output framerate (25 for HuMo)
    wan_height: int = 720  # Video height (multiple of 16)
    wan_width: int = 1280  # Video width (multiple of 16)
    wan_guidance_scale: float = 5.0  # Text guidance (scale_t)
    wan_audio_scale: float = 0.0  # Audio guidance (scale_a), 0 = T2V mode
    wan_steps: int = 50  # Diffusion steps (50 for HuMo)
    wan_offload_mode: str = "model"  # none, model, sequential
    wan_output_path: str = "wan_output.mp4"  # Output video path

    # FLUX.2 Klein image generation
    flux2_model_name: str = "klein-9b"  # klein-4b, klein-9b, klein-base-4b, klein-base-9b
    flux2_num_steps: int | None = None  # None = model default (4 for distilled, 50 for base)
    flux2_guidance: float | None = None  # None = model default (1.0 for distilled, 4.0 for base)
    flux2_seed: int | None = None  # Random seed for reproducibility
    flux2_offload_between_stages: bool = True  # Memory-efficient three-stage offloading
    flux2_block_offload: bool = False  # Block-by-block offloading (slower but uses ~5GB less VRAM)
    flux2_output_path: str = "flux2_output.png"  # Output image path
    flux2_input_images: list[str] | None = None  # Input image paths for editing mode
    flux2_encoder_path: str | None = None  # Custom path for Qwen3 encoder (auto-detects dtype)
    flux2_encoder_device: str = "cuda"  # Device for text encoder (cuda recommended)
    flux2_model_path: str | None = None  # Local path to transformer weights (file or directory)
    flux2_vae_path: str | None = None  # Local path to VAE weights (file or directory)

    # Z-Image variant configuration
    zimage_variant: str = "auto"  # auto, turbo, base
    zimage_model_path: str | None = None  # Path to Z-Image model (overrides config.toml)

    # Device placement
    encoder_device: str = "auto"
    dit_device: str = "auto"
    vae_device: str = "auto"

    # Precision
    dtype: str = "bfloat16"

    # Quantization
    quantization: str = "none"  # none, 4bit, 8bit, int8_dynamic

    # Optimization flags
    cpu_offload: bool = False
    flash_attn: bool = False
    compile: bool = False
    compile_mode: str = "default"  # torch.compile mode (default is CPU-offload safe)

    # PyTorch-native components (Phase 1 migration)
    attention_backend: str | None = None  # auto, flash_attn_2, sdpa, xformers
    use_custom_scheduler: bool = False  # Use our FlowMatchScheduler instead of diffusers
    tiled_vae: bool = False  # Enable tiled VAE decode for large images
    tile_size: int = 512  # Tile size for VAE (pixel space)
    tile_overlap: int = 64  # Overlap between tiles

    # Embedding cache
    embedding_cache: bool = False  # Enable embedding caching
    cache_size: int = 100  # Maximum number of cached embeddings

    # Long prompt handling
    long_prompt_mode: str = "interpolate"  # truncate, interpolate, pool, attention_pool

    # Encoder settings
    hidden_layer: int = -2  # Which layer to extract embeddings from (-1=last, -2=penultimate)
    layer_weights: dict[int, float] | None = (
        None  # Multi-layer blending weights (overrides hidden_layer)
    )

    # Scheduler
    shift: float = 3.0
    shift_terminal: float | None = None  # Stretch sigma to terminal value (0.02 for Qwen-Image)
    dynamic_shift: bool = False  # Use resolution-based shift instead of fixed value
    d_noise: float = 1.0  # Sigma schedule scaling: <1.0 = sharper/detailed, >1.0 = softer

    # CFG settings
    cfg_norm_mode: str = "clamp"  # "clamp" or "match" (DiffSynth-style)

    # Generation defaults
    height: int = 1024
    width: int = 1024
    steps: int = 9
    guidance_scale: float = 0.0
    cfg_normalization: float = 0.0  # CFG norm clamping (0 = disabled)
    cfg_truncation: float = 1.0  # CFG truncation threshold (1.0 = never)
    seed: int | None = None  # Random seed for reproducibility
    negative_prompt: str | None = None  # Negative prompt for CFG
    enable_thinking: bool = True  # DiffSynth always uses empty think block
    default_template: str | None = None

    # Prompt components
    system_prompt: str | None = None
    thinking_content: str | None = None
    assistant_content: str | None = None

    # API backend
    api_url: str | None = None
    api_model: str = "Qwen3-4B-mlx"
    local_encoder: bool = False

    # LoRA
    lora_paths: list[str] = field(default_factory=list)
    lora_scales: list[float] = field(default_factory=list)

    # Server (web only)
    host: str = "127.0.0.1"
    port: int = 7860

    # Rewriter settings (Qwen3 thinking mode recommended defaults)
    # See: https://huggingface.co/Qwen/Qwen3-4B#best-practices
    rewriter_use_api: bool = False  # Use API backend for rewriting
    rewriter_api_url: str = ""  # API URL for rewriter (if different from encoder)
    rewriter_api_model: str = "Qwen3-4B"  # Model ID for rewriter API
    rewriter_temperature: float = 0.6  # Qwen3 thinking mode: 0.6 (NOT greedy!)
    rewriter_top_p: float = 0.95  # Qwen3 thinking mode: 0.95
    rewriter_top_k: int = 20  # Qwen3 thinking mode: 20
    rewriter_min_p: float = 0.0  # Qwen3: 0.0 (disabled)
    rewriter_max_tokens: int = 1024  # Maximum tokens to generate
    rewriter_presence_penalty: float = 0.0  # 0-2, helps reduce endless repetitions
    rewriter_vl_enabled: bool = True  # Allow VL model selection in rewriter UI
    rewriter_preload_vl: bool = False  # Load Qwen3-VL at startup for rewriter
    rewriter_vl_api_model: str = ""  # Model ID for VL via API (e.g., "qwen2.5-vl-72b-mlx")
    rewriter_timeout: float = 120.0  # API request timeout in seconds

    # Vision conditioning (Qwen3-VL)
    vl_model_path: str = ""  # Path to Qwen3-VL model (empty = disabled)
    vl_device: str = "cpu"  # Device for Qwen3-VL
    vl_alpha: float = 0.3  # Default interpolation ratio (0.0=text, 1.0=VL)
    vl_hidden_layer: int = -2  # Hidden layer to extract
    vl_auto_unload: bool = True  # Unload after extraction to save VRAM
    vl_blend_mode: str = "interpolate"  # interpolate (recommended), adain_per_dim, adain, linear

    # DyPE (Dynamic Position Extrapolation) for high-resolution generation
    dype_enabled: bool = False  # Enable DyPE
    dype_method: str = "vision_yarn"  # vision_yarn, yarn, ntk
    dype_scale: float = 2.0  # Magnitude (lambda_s)
    dype_exponent: float = 2.0  # Decay speed (lambda_t)
    dype_start_sigma: float = 1.0  # When to start (0-1)
    dype_base_shift: float = 0.5  # Noise schedule shift at base resolution
    dype_max_shift: float = 1.15  # Noise schedule shift at max resolution
    dype_base_resolution: int = 1024  # Training resolution
    dype_anisotropic: bool = False  # Per-axis scaling for extreme aspect ratios
    dype_multipass: str = "single"  # single, twopass, threepass
    dype_pass2_strength: float = 0.5  # img2img strength for pass 2
    dype_pass3_strength: float = 0.4  # img2img strength for pass 3
    dype_frequency_modulation: bool = False  # Enable timestep-based frequency scaling

    # Skip Layer Guidance (SLG) for improved structure/anatomy
    slg_scale: float = 0.0  # Guidance scale (0 = disabled, 2-3 typical)
    slg_layers: Optional[List[int]] = None  # Layers to skip (e.g., [7, 8, 9, 10, 11, 12])
    slg_start: float = 0.05  # Start SLG at this fraction
    slg_stop: float = 0.5  # Stop SLG at this fraction

    # Flow Map Trajectory Tilting (FMTT) for test-time reward optimization
    fmtt_scale: float = 0.0  # Guidance scale (0 = disabled, 0.5-2.0 typical)
    fmtt_start: float = 0.0  # Start FMTT at this fraction
    fmtt_stop: float = 0.5  # Stop FMTT at this fraction
    fmtt_normalize: str = "unit"  # Gradient normalization mode: unit, clip, none
    fmtt_decode_scale: float = 0.5  # Scale for intermediate VAE decode (0.5 saves VRAM)
    fmtt_siglip_model: str = "google/siglip2-giant-opt-patch16-384"  # SigLIP model
    fmtt_siglip_device: str = "cuda"  # Device for SigLIP (cuda/cpu)

    # Forward Block Cache (FBCache) for inference acceleration
    fbcache: bool = False  # Enable FBCache (30-50% speedup)
    fbcache_threshold: float | None = None  # Override middle threshold (default 0.05 = 5%)
    fbcache_log: bool = False  # Log residual statistics

    # Debug and logging
    debug: bool = False
    verbose: bool = False
    log_dir: str | None = None  # Directory for JSON log files with rotation

    def get_dtype(self) -> torch.dtype:
        """Convert string dtype to torch.dtype."""
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.dtype, torch.bfloat16)

    def resolve_device(self, device: str) -> str:
        """Resolve 'auto' to actual device."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device

    @property
    def encoder_device_resolved(self) -> str:
        return self.resolve_device(self.encoder_device)

    @property
    def dit_device_resolved(self) -> str:
        return self.resolve_device(self.dit_device)

    @property
    def vae_device_resolved(self) -> str:
        return self.resolve_device(self.vae_device)

    def get_qwen_variant_defaults(self) -> dict:
        """
        Return variant-specific defaults for Qwen-Image models based on model_type.

        These defaults are applied when the corresponding field is None.

        Returns:
            Dict with keys: steps, resolution, quantize_transformer
        """
        defaults = {
            "qwenimage-t2i": {
                "steps": 40,
                "resolution": 1024,
                "quantize_transformer": "fp8",  # TorchAO for 60-layer DiT
            },
            "qwenimage-edit": {
                "steps": 25,
                "resolution": 640,
                "quantize_transformer": "diffsynth-fp8",  # DiffSynth-style for 8B DiT
            },
            "qwenimage-layered": {
                "steps": 50,
                "resolution": 640,
                "quantize_transformer": "diffsynth-fp8",
            },
        }
        return defaults.get(self.model_type, {})

    def get_qwen_image_steps(self) -> int:
        """Get effective steps, using variant default if not explicitly set."""
        if self.qwen_image_steps is not None:
            return self.qwen_image_steps
        variant_defaults = self.get_qwen_variant_defaults()
        return variant_defaults.get("steps", 40)

    def get_qwen_image_resolution(self) -> int:
        """Get effective resolution, using variant default if not explicitly set."""
        if self.qwen_image_resolution is not None:
            return self.qwen_image_resolution
        variant_defaults = self.get_qwen_variant_defaults()
        return variant_defaults.get("resolution", 1024)

    def get_qwen_image_quantize_transformer(self) -> str:
        """Get effective transformer quantization, using variant default if not explicitly set."""
        if self.qwen_image_quantize_transformer is not None:
            return self.qwen_image_quantize_transformer
        variant_defaults = self.get_qwen_variant_defaults()
        return variant_defaults.get("quantize_transformer", "none")

    def to_dict(self) -> dict:
        """
        Serialize runtime config to a dictionary for API responses.

        Returns:
            Dict with all config field names and values.
        """
        from dataclasses import fields as dataclass_fields

        result = {}
        for f in dataclass_fields(self):
            value = getattr(self, f.name)
            # Handle non-JSON-serializable types
            if isinstance(value, torch.dtype):
                value = str(value).replace("torch.", "")
            result[f.name] = value
        return result


def create_base_parser(
    description: str = "Z-Image generation",
    include_server_args: bool = False,
    include_generation_args: bool = True,
) -> argparse.ArgumentParser:
    """
    Create the base argument parser with all shared flags.

    Args:
        description: Parser description
        include_server_args: Include --host and --port (for web server)
        include_generation_args: Include generation params like --height, --width

    Returns:
        ArgumentParser with all shared arguments
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Config file
    config_group = parser.add_argument_group("Configuration")
    config_group.add_argument(
        "--config",
        type=str,
        default="config.toml",
        help="Path to TOML config file (default: config.toml)",
    )
    config_group.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Config profile to use. If not specified, auto-detects flat vs profile-based config",
    )

    # Model selection
    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--model-type",
        type=str,
        choices=list(SUPPORTED_MODEL_TYPES),
        default=None,
        help="Model type: zimage, qwenimage-layered, qwenimage-t2i, qwenimage-edit, ltx2, wan. Default: zimage",
    )
    model_group.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model directory. Works for all model types (Z-Image, LTX-2, etc). "
        "Type-specific paths like --ltx2-model-path take precedence if specified.",
    )
    model_group.add_argument(
        "--text-encoder-path",
        type=str,
        default=None,
        help="Path to text encoder (Qwen3-4B). If not specified, uses model-path/text_encoder/",
    )
    model_group.add_argument(
        "--templates-dir",
        type=str,
        default=None,
        help="Path to templates directory",
    )

    # Z-Image variant configuration
    zimage_group = parser.add_argument_group("Z-Image")
    zimage_group.add_argument(
        "--zimage-variant",
        type=str,
        choices=["turbo", "base", "auto"],
        default="auto",
        help="Z-Image variant: turbo (9 steps, CFG baked in), base (35 steps, full CFG), auto (detect from scheduler)",
    )
    zimage_group.add_argument(
        "--zimage-model-path",
        type=str,
        default=None,
        help="Path to Z-Image model (overrides config.toml [zimage].model_path)",
    )

    # Qwen-Image (all variants: t2i, edit, layered)
    # Use --model-type to select: qwenimage-t2i, qwenimage-edit, qwenimage-layered
    qwen_group = parser.add_argument_group("Qwen-Image (all variants)")
    qwen_group.add_argument(
        "--qwen-image-model-path",
        type=str,
        default=None,
        help="Path to any Qwen-Image model (T2I, Edit, or Layered)",
    )
    qwen_group.add_argument(
        "--qwen-image-cpu-offload",
        action="store_true",
        default=None,
        help="Enable CPU offload for Qwen-Image (required for RTX 4090)",
    )
    qwen_group.add_argument(
        "--qwen-image-layers",
        type=int,
        default=None,
        help="Number of decomposition layers (layered variant only, default: 4)",
    )
    qwen_group.add_argument(
        "--qwen-image-steps",
        type=int,
        default=None,
        help="Diffusion steps (variant default: t2i=40, edit=25, layered=50)",
    )
    qwen_group.add_argument(
        "--qwen-image-cfg-scale",
        type=float,
        default=None,
        help="CFG scale for Qwen-Image (default: 4.0)",
    )
    qwen_group.add_argument(
        "--qwen-image-resolution",
        type=int,
        choices=[640, 1024],
        default=None,
        help="Resolution (variant default: t2i=1024, edit=640, layered=640)",
    )
    qwen_group.add_argument(
        "--qwen-image-quantize-text-encoder",
        type=str,
        choices=["none", "4bit", "8bit", "fp8", "int8"],
        default=None,
        help="Quantization for text encoder (Qwen2.5-VL-7B): none (CPU offload), 4bit, 8bit",
    )
    qwen_group.add_argument(
        "--qwen-image-quantize-transformer",
        type=str,
        choices=["none", "4bit", "8bit", "fp8", "int8", "diffsynth-fp8"],
        default=None,
        help="Quantization for DiT (variant default: t2i=fp8, edit/layered=diffsynth-fp8)",
    )

    # LTX-2 video generation
    ltx2_group = parser.add_argument_group("LTX-2 Video Generation")
    ltx2_group.add_argument(
        "--ltx2-model-path",
        type=str,
        default=None,
        help="Path to LTX-2 model directory (e.g., ~/Storage/LTX-2)",
    )
    ltx2_group.add_argument(
        "--ltx2-encoder-model-id",
        type=str,
        default=None,
        help="Path to text encoder (Gemma 3 compatible). Can be any quantized variant "
        "(QAT 4-bit, FP8, etc.) as long as output dimensions match. "
        "Falls back to --text-encoder-path, then model_path/text_encoder.",
    )
    ltx2_group.add_argument(
        "--ltx2-num-frames",
        type=int,
        default=None,
        help="Number of video frames (default: 33, max ~65 for 24GB)",
    )
    ltx2_group.add_argument(
        "--ltx2-fps",
        type=int,
        default=None,
        help="Output framerate (default: 24)",
    )
    ltx2_group.add_argument(
        "--ltx2-guidance-scale",
        type=float,
        default=None,
        help="CFG guidance scale (default: 3.5, range 3.0-4.0)",
    )
    ltx2_group.add_argument(
        "--ltx2-steps",
        type=int,
        default=None,
        help="Diffusion steps (default: 12 for distilled model)",
    )
    ltx2_group.add_argument(
        "--ltx2-lora-path",
        type=str,
        default=None,
        help="Path to LoRA safetensors file",
    )
    ltx2_group.add_argument(
        "--ltx2-lora-scale",
        type=float,
        default=None,
        help="LoRA blend scale (default: 1.0)",
    )
    ltx2_group.add_argument(
        "--ltx2-audio",
        action="store_true",
        default=None,
        help="Enable audio generation",
    )
    ltx2_group.add_argument(
        "--ltx2-output",
        type=str,
        default=None,
        help="Output video path (default: output.mp4)",
    )
    ltx2_group.add_argument(
        "--ltx2-save-embeddings",
        type=str,
        default=None,
        help="Save text embeddings to file (skip video generation, for precomputation). "
        "Useful for encoding prompts once and generating multiple videos with different seeds.",
    )
    ltx2_group.add_argument(
        "--ltx2-load-embeddings",
        type=str,
        default=None,
        help="Load pre-computed embeddings from file (skip text encoding). "
        "Works with any compatible Gemma3 quantization variant.",
    )

    # LTX-2 Optimization (matching reference repo)
    ltx2_opt = parser.add_argument_group("LTX-2 Optimization")
    ltx2_opt.add_argument(
        "--ltx2-text-encoder-device",
        choices=["cpu", "cuda"],
        default=None,
        help="Device for Gemma3 text encoder (cpu recommended for 24GB, default: cpu)",
    )
    ltx2_opt.add_argument(
        "--ltx2-transformer-device",
        choices=["cpu", "cuda"],
        default=None,
        help="Device for DiT transformer (default: cuda)",
    )
    ltx2_opt.add_argument(
        "--ltx2-vae-device",
        choices=["cpu", "cuda"],
        default=None,
        help="Device for VAE decoder (default: cuda)",
    )
    ltx2_opt.add_argument(
        "--ltx2-quantize",
        choices=["none", "fp8"],
        default=None,
        help="Transformer quantization (fp8 for 24GB GPUs, default: fp8)",
    )
    ltx2_opt.add_argument(
        "--ltx2-skip-cleanup",
        action="store_true",
        help="Skip memory cleanup between stages (faster, needs more VRAM)",
    )
    ltx2_opt.add_argument(
        "--ltx2-gemma-variant",
        choices=["bf16", "8bit", "q4-qat"],
        default=None,
        help="Gemma3 backbone variant for text encoding. "
        "bf16: Full precision (~24GB). "
        "8bit: BitsAndBytes 8-bit (~12GB). "
        "q4-qat: Pre-quantized Q4 QAT (~3GB). "
        "Default: bf16",
    )

    # Wan/HuMo video generation
    wan_group = parser.add_argument_group("Wan/HuMo Video Generation")
    wan_group.add_argument(
        "--wan-humo-path",
        type=str,
        default=None,
        help="Path to HuMo transformer (e.g., ~/Storage/HuMo)",
    )
    wan_group.add_argument(
        "--wan-base-path",
        type=str,
        default=None,
        help="Path to Wan2.1-T2V for VAE/text encoder (e.g., ~/Storage/Wan2.1-T2V-1.3B)",
    )
    wan_group.add_argument(
        "--wan-whisper-path",
        type=str,
        default=None,
        help="Path to Whisper for audio (optional, lazy-loads if not set)",
    )
    wan_group.add_argument(
        "--wan-humo-variant",
        type=str,
        choices=["17B", "1.7B"],
        default=None,
        help="HuMo variant (default: 17B)",
    )
    wan_group.add_argument(
        "--wan-num-frames",
        type=int,
        default=None,
        help="Number of video frames (default: 97, ~3.9s at 25fps)",
    )
    wan_group.add_argument(
        "--wan-fps",
        type=int,
        default=None,
        help="Output framerate (default: 25 for HuMo)",
    )
    wan_group.add_argument(
        "--wan-height",
        type=int,
        default=None,
        help="Video height (default: 720, multiple of 16)",
    )
    wan_group.add_argument(
        "--wan-width",
        type=int,
        default=None,
        help="Video width (default: 1280, multiple of 16)",
    )
    wan_group.add_argument(
        "--wan-guidance-scale",
        type=float,
        default=None,
        help="Text guidance scale_t (default: 5.0)",
    )
    wan_group.add_argument(
        "--wan-audio-scale",
        type=float,
        default=None,
        help="Audio guidance scale_a (default: 0.0, set >0 for audio mode)",
    )
    wan_group.add_argument(
        "--wan-steps",
        type=int,
        default=None,
        help="Diffusion steps (default: 50 for HuMo)",
    )
    wan_group.add_argument(
        "--wan-offload-mode",
        type=str,
        choices=["none", "model", "sequential"],
        default=None,
        help="CPU offload mode (default: model for 24GB VRAM)",
    )
    wan_group.add_argument(
        "--wan-output",
        type=str,
        default=None,
        help="Output video path (default: wan_output.mp4)",
    )

    # FLUX.2 Klein image generation
    flux2_group = parser.add_argument_group("FLUX.2 Klein Image Generation")
    flux2_group.add_argument(
        "--flux2-model-name",
        type=str,
        choices=[
            "klein-4b", "klein-9b", "klein-base-4b", "klein-base-9b",
            "klein-4b-fp8", "klein-9b-fp8", "klein-base-4b-fp8", "klein-base-9b-fp8",
        ],
        default=None,
        help="FLUX.2 Klein model variant (default: klein-9b). "
        "Distilled models (klein-*b) use 4 steps, base models use 50 steps. "
        "FP8 variants (-fp8 suffix) use half the memory.",
    )
    flux2_group.add_argument(
        "--flux2-num-steps",
        type=int,
        default=None,
        help="Number of denoising steps (default: 4 for distilled, 50 for base)",
    )
    flux2_group.add_argument(
        "--flux2-guidance",
        type=float,
        default=None,
        help="Guidance scale (default: 1.0 for distilled, 4.0 for base)",
    )
    flux2_group.add_argument(
        "--flux2-seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    flux2_group.add_argument(
        "--flux2-offload",
        action="store_true",
        default=None,
        help="Enable three-stage memory offloading (default: True)",
    )
    flux2_group.add_argument(
        "--flux2-no-offload",
        action="store_true",
        default=None,
        help="Disable memory offloading (requires more VRAM)",
    )
    flux2_group.add_argument(
        "--flux2-block-offload",
        action="store_true",
        default=None,
        help="Enable block-by-block offloading (slower but uses ~5GB less VRAM). "
        "Use this if you're getting OOM errors with full model on GPU.",
    )
    flux2_group.add_argument(
        "--flux2-output",
        type=str,
        default=None,
        help="Output image path (default: flux2_output.png)",
    )
    flux2_group.add_argument(
        "--flux2-input-image",
        type=str,
        nargs="+",
        default=None,
        help="Input image(s) for editing mode. Can specify multiple images.",
    )
    flux2_group.add_argument(
        "--flux2-encoder-path",
        type=str,
        default=None,
        help="Custom path for Qwen3 text encoder (local path or HF model ID). "
        "Auto-detects dtype (BF16, FP8, etc). Default: uses model-specific encoder.",
    )
    flux2_group.add_argument(
        "--flux2-model-path",
        type=str,
        default=None,
        help="Local path to transformer weights (file or directory). "
        "If directory, searches for expected .safetensors file.",
    )
    flux2_group.add_argument(
        "--flux2-vae-path",
        type=str,
        default=None,
        help="Local path to VAE weights (file or directory). "
        "If directory, searches for ae.safetensors in vae/ subdirectory.",
    )

    # Device placement
    device_group = parser.add_argument_group("Devices")
    device_group.add_argument(
        "--text-encoder-device",
        type=str,
        choices=["cpu", "cuda", "mps", "auto"],
        default=None,
        help="Device for text encoder (default: auto)",
    )
    device_group.add_argument(
        "--dit-device",
        type=str,
        choices=["cpu", "cuda", "mps", "auto"],
        default=None,
        help="Device for DiT/transformer (default: auto)",
    )
    device_group.add_argument(
        "--vae-device",
        type=str,
        choices=["cpu", "cuda", "mps", "auto"],
        default=None,
        help="Device for VAE (default: auto)",
    )

    # Optimization
    opt_group = parser.add_argument_group("Optimization")
    opt_group.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Enable CPU offload for transformer",
    )
    opt_group.add_argument(
        "--flash-attn",
        action="store_true",
        help="Enable Flash Attention (requires flash-attn package)",
    )
    opt_group.add_argument(
        "--compile",
        action="store_true",
        help="Compile transformer with torch.compile (slower first run)",
    )
    opt_group.add_argument(
        "--torch-dtype",
        type=str,
        choices=["bfloat16", "float16", "float32"],
        default=None,
        help="Model precision (default: bfloat16)",
    )
    opt_group.add_argument(
        "--quantization",
        type=str,
        choices=["none", "4bit", "8bit", "int8_dynamic"],
        default=None,
        help=(
            "Text encoder quantization: "
            "none (default), "
            "4bit/8bit (BitsAndBytes), "
            "int8_dynamic (torchao, ~50%% VRAM reduction)"
        ),
    )

    # PyTorch-native components
    pytorch_group = parser.add_argument_group("PyTorch Native (Phase 1)")
    pytorch_group.add_argument(
        "--attention-backend",
        type=str,
        choices=["auto", "flash_attn_2", "flash_attn_3", "sage", "xformers", "sdpa"],
        default=None,
        help="Attention backend (default: auto-detect best available)",
    )
    pytorch_group.add_argument(
        "--use-custom-scheduler",
        action="store_true",
        help="Use our pure-PyTorch FlowMatchScheduler instead of diffusers",
    )
    pytorch_group.add_argument(
        "--tiled-vae",
        action="store_true",
        help="Enable tiled VAE decode for large images (2K+)",
    )
    pytorch_group.add_argument(
        "--tile-size",
        type=int,
        default=None,
        help="Tile size for VAE decode in pixels (default: 512)",
    )
    pytorch_group.add_argument(
        "--tile-overlap",
        type=int,
        default=None,
        help="Overlap between VAE tiles in pixels (default: 64)",
    )
    pytorch_group.add_argument(
        "--embedding-cache",
        action="store_true",
        help="Enable embedding cache for repeated prompts",
    )
    pytorch_group.add_argument(
        "--cache-size",
        type=int,
        default=None,
        help="Maximum number of cached embeddings (default: 100)",
    )
    pytorch_group.add_argument(
        "--long-prompt-mode",
        type=str,
        choices=["truncate", "interpolate", "pool", "attention_pool"],
        default=None,
        help=(
            "How to handle prompts exceeding 1504 tokens: "
            "truncate (cut off end), "
            "interpolate (default, smooth resampling), "
            "pool (average pooling), "
            "attention_pool (importance-weighted pooling)"
        ),
    )
    pytorch_group.add_argument(
        "--hidden-layer",
        type=int,
        default=None,
        help=(
            "Which hidden layer to extract embeddings from (default: -2). "
            "-1=last layer, -2=penultimate (default for Z-Image), -3, etc. "
            "Useful for ablation studies comparing different layer outputs."
        ),
    )
    pytorch_group.add_argument(
        "--layer-weights",
        type=str,
        default=None,
        help=(
            "Multi-layer blending weights (overrides --hidden-layer). "
            "Format: 'layer:weight,layer:weight,...' e.g. '-2:0.7,-6:0.3' "
            "Weights are normalized to sum to 1.0."
        ),
    )

    # Scheduler
    sched_group = parser.add_argument_group("Scheduler")
    sched_group.add_argument(
        "--shift",
        type=float,
        default=None,
        help="Scheduler shift parameter (default: 3.0)",
    )
    sched_group.add_argument(
        "--shift-terminal",
        type=float,
        default=None,
        help="Stretch sigma schedule to end at this value instead of 0 (Qwen-Image only, default: None). "
        "Example: 0.02 stretches schedule so final sigma=0.02. Not used by Z-Image.",
    )
    sched_group.add_argument(
        "--dynamic-shift",
        action="store_true",
        help="Calculate shift based on resolution (overrides --shift). "
        "Uses linear interpolation: base_shift=0.5 at 512x512, max_shift=1.15 at 2048x2048.",
    )
    sched_group.add_argument(
        "--d-noise",
        type=float,
        default=None,
        help="Sigma schedule scaling factor. <1.0 = sharper/more detail (try 0.95-0.98), "
        ">1.0 = softer/deeper colors (try 1.02-1.05). Default: 1.0 (no scaling).",
    )

    # LoRA
    lora_group = parser.add_argument_group("LoRA")
    lora_group.add_argument(
        "--lora",
        type=str,
        action="append",
        default=None,
        dest="loras",
        metavar="PATH:SCALE",
        help="Load LoRA weights (repeatable). Format: path/to/lora.safetensors:0.8",
    )

    # Prompt control
    prompt_group = parser.add_argument_group("Prompt Control")
    prompt_group.add_argument(
        "--template",
        type=str,
        default=None,
        help="Template name for encoding",
    )
    prompt_group.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="System prompt for chat template",
    )
    prompt_group.add_argument(
        "--thinking-content",
        type=str,
        default=None,
        help="Content inside <think>...</think> block",
    )
    prompt_group.add_argument(
        "--assistant-content",
        type=str,
        default=None,
        help="Content after </think> (assistant response prefix)",
    )
    prompt_group.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Add <think></think> structure to prompt",
    )

    # API backend
    api_group = parser.add_argument_group("API Backend")
    api_group.add_argument(
        "--api-url",
        type=str,
        default=None,
        help="Use remote API for encoding (e.g., http://mac-ip:8080)",
    )
    api_group.add_argument(
        "--api-model",
        type=str,
        default=None,
        help="Model ID for API backend (default: Qwen3-4B-mlx)",
    )
    api_group.add_argument(
        "--local-encoder",
        action="store_true",
        help="Force local encoder (for A/B testing API vs local)",
    )

    # Vision conditioning (Qwen3-VL)
    vl_group = parser.add_argument_group("Vision Conditioning (Qwen3-VL)")
    vl_group.add_argument(
        "--vl-model-path",
        type=str,
        default=None,
        help="Path to Qwen3-VL model (enables vision conditioning)",
    )
    vl_group.add_argument(
        "--vl-device",
        type=str,
        choices=["cpu", "cuda", "auto"],
        default=None,
        help="Device for Qwen3-VL (default: cpu to save VRAM)",
    )
    vl_group.add_argument(
        "--vl-alpha",
        type=float,
        default=None,
        help="VL influence ratio (0.0=pure text, 1.0=pure VL, default: 0.3)",
    )
    vl_group.add_argument(
        "--vl-hidden-layer",
        type=int,
        default=None,
        help="Hidden layer to extract from Qwen3-VL (default: -2, penultimate)",
    )
    vl_group.add_argument(
        "--vl-no-auto-unload",
        action="store_true",
        help="Keep Qwen3-VL loaded after extraction (uses more VRAM)",
    )
    vl_group.add_argument(
        "--vl-blend-mode",
        type=str,
        choices=[
            "interpolate",
            "adain_per_dim",
            "adain",
            "linear",
            "style_only",
            "graduated",
            "attention_weighted",
        ],
        default=None,
        help=(
            "Blending strategy: "
            "interpolate (recommended, compresses all VL tokens), "
            "adain_per_dim (best for style transfer), "
            "adain (transfer VL statistics to text), "
            "linear (WARNING: truncates, loses most VL info), "
            "style_only (blend only style dimensions), "
            "graduated (more VL for later tokens)"
        ),
    )

    # Rewriter settings
    rewriter_group = parser.add_argument_group("Rewriter")
    rewriter_group.add_argument(
        "--rewriter-use-api",
        action="store_true",
        help="Use API backend for prompt rewriting instead of local model",
    )
    rewriter_group.add_argument(
        "--rewriter-api-url",
        type=str,
        default=None,
        help="API URL for rewriter (defaults to --api-url if not set)",
    )
    rewriter_group.add_argument(
        "--rewriter-api-model",
        type=str,
        default=None,
        help="Model ID for rewriter API (default: Qwen3-4B)",
    )
    rewriter_group.add_argument(
        "--rewriter-temperature",
        type=float,
        default=None,
        help="Sampling temperature for rewriter (default: 0.6 for Qwen3 thinking mode)",
    )
    rewriter_group.add_argument(
        "--rewriter-top-p",
        type=float,
        default=None,
        help="Nucleus sampling threshold for rewriter (default: 0.95)",
    )
    rewriter_group.add_argument(
        "--rewriter-top-k",
        type=int,
        default=None,
        help="Top-k sampling for rewriter (default: 20 for Qwen3)",
    )
    rewriter_group.add_argument(
        "--rewriter-min-p",
        type=float,
        default=None,
        help="Minimum probability threshold for rewriter (default: 0.0, disabled)",
    )
    rewriter_group.add_argument(
        "--rewriter-presence-penalty",
        type=float,
        default=None,
        help="Presence penalty for rewriter (0-2, helps reduce repetition, default: 0.0)",
    )
    rewriter_group.add_argument(
        "--rewriter-max-tokens",
        type=int,
        default=None,
        help="Maximum tokens to generate for rewriter (default: 512)",
    )
    rewriter_group.add_argument(
        "--rewriter-no-vl",
        action="store_true",
        help="Disable VL model selection in rewriter UI",
    )
    rewriter_group.add_argument(
        "--rewriter-preload-vl",
        action="store_true",
        help="Preload Qwen3-VL for rewriter at startup (uses vl.model_path)",
    )
    rewriter_group.add_argument(
        "--rewriter-vl-api-model",
        type=str,
        default=None,
        help="Model ID for VL rewriting via API (e.g., qwen2.5-vl-72b-mlx)",
    )
    rewriter_group.add_argument(
        "--rewriter-timeout",
        type=float,
        default=None,
        help="API request timeout in seconds (default: 120, VL models may need longer)",
    )

    # DyPE (Dynamic Position Extrapolation) for high-resolution generation
    dype_group = parser.add_argument_group("DyPE (High-Resolution)")
    dype_group.add_argument(
        "--dype",
        action="store_true",
        help="Enable DyPE for high-resolution generation (2K-4K+)",
    )
    dype_group.add_argument(
        "--dype-method",
        type=str,
        choices=["vision_yarn", "yarn", "ntk"],
        default=None,
        help="RoPE extrapolation method (default: vision_yarn)",
    )
    dype_group.add_argument(
        "--dype-scale",
        type=float,
        default=None,
        help="DyPE magnitude (lambda_s, default: 2.0)",
    )
    dype_group.add_argument(
        "--dype-exponent",
        type=float,
        default=None,
        help="DyPE decay speed (lambda_t, default: 2.0 = quadratic)",
    )
    dype_group.add_argument(
        "--dype-start-sigma",
        type=float,
        default=None,
        help="When to start DyPE decay (0-1, 1.0 = from start, default: 1.0)",
    )
    dype_group.add_argument(
        "--dype-base-shift",
        type=float,
        default=None,
        help="Noise schedule shift at base resolution (default: 0.5)",
    )
    dype_group.add_argument(
        "--dype-max-shift",
        type=float,
        default=None,
        help="Noise schedule shift at max resolution (default: 1.15)",
    )
    dype_group.add_argument(
        "--dype-base-resolution",
        type=int,
        default=None,
        help="Training resolution (Z-Image: 1024, Qwen: 1328, default: 1024)",
    )
    dype_group.add_argument(
        "--dype-anisotropic",
        action="store_true",
        help="Use per-axis scaling for extreme aspect ratios (16:9, 9:16)",
    )
    dype_group.add_argument(
        "--dype-multipass",
        type=str,
        choices=["single", "twopass", "threepass"],
        default=None,
        help="Generation mode: single (direct), twopass (512->target), threepass (256->512->target)",
    )
    dype_group.add_argument(
        "--dype-pass2-strength",
        type=float,
        default=None,
        help="img2img strength for second pass (0.0-1.0, default: 0.5)",
    )
    dype_group.add_argument(
        "--dype-pass3-strength",
        type=float,
        default=None,
        help="img2img strength for third pass (0.0-1.0, default: 0.4)",
    )
    dype_group.add_argument(
        "--dype-frequency-modulation",
        action="store_true",
        help="Enable timestep-based RoPE frequency scaling (experimental)",
    )

    # Skip Layer Guidance (SLG)
    slg_group = parser.add_argument_group("Skip Layer Guidance (SLG)")
    slg_group.add_argument(
        "--slg-scale",
        type=float,
        default=None,
        help="SLG scale (0 = disabled, 2-4 typical for improved anatomy/structure)",
    )
    slg_group.add_argument(
        "--slg-layers",
        type=str,
        default=None,
        help=(
            "Comma-separated layer indices to skip (e.g., '15,16,17,18,19'). "
            "Z-Image has 40 layers (0-39). Middle layers recommended."
        ),
    )
    slg_group.add_argument(
        "--slg-start",
        type=float,
        default=None,
        help="Start SLG at this fraction of steps (default: 0.01)",
    )
    slg_group.add_argument(
        "--slg-stop",
        type=float,
        default=None,
        help="Stop SLG at this fraction of steps (default: 0.2)",
    )

    # Flow Map Trajectory Tilting (FMTT)
    fmtt_group = parser.add_argument_group("Flow Map Trajectory Tilting (FMTT)")
    fmtt_group.add_argument(
        "--fmtt-scale",
        type=float,
        default=None,
        help=(
            "FMTT guidance scale (0 = disabled, 0.5-2.0 typical). "
            "Uses SigLIP2 reward to guide generation toward text-aligned images. "
            "Note: Requires ~4GB extra VRAM for SigLIP model."
        ),
    )
    fmtt_group.add_argument(
        "--fmtt-start",
        type=float,
        default=None,
        help="Start FMTT at this fraction of steps (default: 0.0)",
    )
    fmtt_group.add_argument(
        "--fmtt-stop",
        type=float,
        default=None,
        help="Stop FMTT at this fraction of steps (default: 0.5)",
    )
    fmtt_group.add_argument(
        "--fmtt-normalize",
        type=str,
        choices=["unit", "clip", "none"],
        default=None,
        help="Gradient normalization mode (default: unit)",
    )
    fmtt_group.add_argument(
        "--fmtt-decode-scale",
        type=float,
        default=None,
        help=(
            "Scale for intermediate VAE decode (default: 0.5 = 512px for 1024px input). "
            "Lower values save VRAM but reduce gradient precision."
        ),
    )
    fmtt_group.add_argument(
        "--fmtt-siglip-model",
        type=str,
        default=None,
        help="SigLIP model ID (default: google/siglip2-giant-opt-patch16-384)",
    )
    fmtt_group.add_argument(
        "--fmtt-siglip-device",
        type=str,
        choices=["cuda", "cpu", "auto"],
        default=None,
        help="Device for SigLIP model (default: cuda). Use cpu to save VRAM.",
    )

    # Forward Block Cache (FBCache)
    fbcache_group = parser.add_argument_group("Forward Block Cache (FBCache)")
    fbcache_group.add_argument(
        "--fbcache",
        action="store_true",
        help=(
            "Enable FBCache for 30-50%% inference speedup. "
            "Skips redundant transformer blocks when residuals are similar."
        ),
    )
    fbcache_group.add_argument(
        "--fbcache-threshold",
        type=float,
        default=None,
        help=(
            "Override middle-phase threshold (default: 0.05 = 5%%). "
            "Lower = more conservative (fewer skips), higher = more aggressive."
        ),
    )
    fbcache_group.add_argument(
        "--fbcache-log",
        action="store_true",
        help="Log FBCache residual statistics for analysis",
    )

    # Generation parameters (optional)
    if include_generation_args:
        gen_group = parser.add_argument_group("Generation")
        gen_group.add_argument(
            "--height",
            type=int,
            default=None,
            help="Image height (default: 1024, must be divisible by 16)",
        )
        gen_group.add_argument(
            "--width",
            type=int,
            default=None,
            help="Image width (default: 1024, must be divisible by 16)",
        )
        gen_group.add_argument(
            "--steps",
            type=int,
            default=None,
            help="Number of inference steps (default: 9 for turbo)",
        )
        gen_group.add_argument(
            "--guidance-scale",
            type=float,
            default=None,
            help="CFG scale (default: 0.0, not needed for Z-Image-Turbo)",
        )
        gen_group.add_argument(
            "--negative-prompt",
            type=str,
            default=None,
            help="Negative prompt for CFG",
        )
        gen_group.add_argument(
            "--cfg-normalization",
            type=float,
            default=None,
            help="CFG norm clamping factor (0.0 = disabled, typical: 1.0-2.0). "
            "Clamps combined prediction norm relative to positive prediction. "
            "Prevents CFG from over-amplifying. Useful for non-distilled models.",
        )
        gen_group.add_argument(
            "--cfg-truncation",
            type=float,
            default=None,
            help="CFG truncation threshold (1.0 = never, typical: 0.5-0.8). "
            "Stops applying CFG after this fraction of denoising progress. "
            "E.g., 0.7 means no CFG for the final 30%%. Reduces late-stage artifacts.",
        )
        gen_group.add_argument(
            "--cfg-norm-mode",
            type=str,
            choices=["clamp", "match"],
            default=None,
            help="CFG normalization mode (default: clamp). "
            "'clamp' limits combined prediction norm to factor * pos_norm. "
            "'match' scales combined prediction to exactly match pos_norm (DiffSynth-style). "
            "Only relevant when guidance_scale > 0 (not used by Z-Image-Turbo).",
        )
        gen_group.add_argument(
            "--seed",
            type=int,
            default=None,
            help="Random seed for reproducibility",
        )
        gen_group.add_argument(
            "--embeddings-file",
            type=str,
            default=None,
            help="Path to pre-computed embeddings file (.safetensors). Skips text encoding.",
        )

    # Server args (optional)
    if include_server_args:
        server_group = parser.add_argument_group("Server")
        server_group.add_argument(
            "--host",
            type=str,
            default=None,
            help="Host to bind to (default: 127.0.0.1)",
        )
        server_group.add_argument(
            "--port",
            type=int,
            default=None,
            help="Port to bind to (default: 7860)",
        )

    # Debug
    debug_group = parser.add_argument_group("Debug")
    debug_group.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    debug_group.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    debug_group.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory for JSON log files with rotation (enables file logging)",
    )

    return parser


def _apply_cli_overrides(args: argparse.Namespace, config: RuntimeConfig) -> RuntimeConfig:
    """
    Apply CLI argument overrides to a RuntimeConfig.

    This is called after loading config from TOML.
    CLI arguments take precedence over config file values.

    Args:
        args: Parsed CLI arguments
        config: RuntimeConfig to modify

    Returns:
        Modified RuntimeConfig with CLI overrides applied
    """
    # Model overrides
    if getattr(args, "model_type", None) is not None:
        config.model_type = args.model_type
    if getattr(args, "model_path", None) is not None:
        config.model_path = args.model_path
    if getattr(args, "text_encoder_path", None) is not None:
        config.text_encoder_path = args.text_encoder_path
    if getattr(args, "templates_dir", None) is not None:
        config.templates_dir = args.templates_dir

    # Qwen-Image overrides
    if getattr(args, "qwen_image_model_path", None) is not None:
        config.qwen_image_model_path = args.qwen_image_model_path
    if getattr(args, "qwen_image_edit_model_path", None) is not None:
        config.qwen_image_edit_model_path = args.qwen_image_edit_model_path
    if getattr(args, "qwen_image_cpu_offload", None) is not None:
        config.qwen_image_cpu_offload = args.qwen_image_cpu_offload
    if getattr(args, "qwen_image_layers", None) is not None:
        config.qwen_image_layer_num = args.qwen_image_layers
    if getattr(args, "qwen_image_steps", None) is not None:
        config.qwen_image_steps = args.qwen_image_steps
    if getattr(args, "qwen_image_cfg_scale", None) is not None:
        config.qwen_image_cfg_scale = args.qwen_image_cfg_scale
    if getattr(args, "qwen_image_resolution", None) is not None:
        config.qwen_image_resolution = args.qwen_image_resolution
    if getattr(args, "qwen_image_edit_only", False):
        config.qwen_image_edit_only = args.qwen_image_edit_only
    if getattr(args, "qwen_image_quantize_text_encoder", None) is not None:
        config.qwen_image_quantize_text_encoder = args.qwen_image_quantize_text_encoder
    if getattr(args, "qwen_image_quantize_transformer", None) is not None:
        config.qwen_image_quantize_transformer = args.qwen_image_quantize_transformer

    # LTX-2 video overrides
    if getattr(args, "ltx2_model_path", None) is not None:
        config.ltx2_model_path = args.ltx2_model_path
    if getattr(args, "ltx2_encoder_model_id", None) is not None:
        config.ltx2_encoder_model_id = args.ltx2_encoder_model_id
    if getattr(args, "ltx2_num_frames", None) is not None:
        config.ltx2_num_frames = args.ltx2_num_frames
    if getattr(args, "ltx2_fps", None) is not None:
        config.ltx2_fps = args.ltx2_fps
    if getattr(args, "ltx2_guidance_scale", None) is not None:
        config.ltx2_guidance_scale = args.ltx2_guidance_scale
    if getattr(args, "ltx2_steps", None) is not None:
        config.ltx2_steps = args.ltx2_steps
    if getattr(args, "ltx2_lora_path", None) is not None:
        config.ltx2_lora_path = args.ltx2_lora_path
    if getattr(args, "ltx2_lora_scale", None) is not None:
        config.ltx2_lora_scale = args.ltx2_lora_scale
    if getattr(args, "ltx2_audio", False):
        config.ltx2_audio = True
    if getattr(args, "ltx2_output", None) is not None:
        config.ltx2_output_path = args.ltx2_output

    # LTX-2 optimization overrides
    if getattr(args, "ltx2_text_encoder_device", None) is not None:
        config.ltx2_text_encoder_device = args.ltx2_text_encoder_device
    if getattr(args, "ltx2_transformer_device", None) is not None:
        config.ltx2_transformer_device = args.ltx2_transformer_device
    if getattr(args, "ltx2_vae_device", None) is not None:
        config.ltx2_vae_device = args.ltx2_vae_device
    if getattr(args, "ltx2_quantize", None) is not None:
        config.ltx2_quantize = args.ltx2_quantize
    if getattr(args, "ltx2_skip_cleanup", False):
        config.ltx2_skip_cleanup = True
    if getattr(args, "ltx2_save_embeddings", None) is not None:
        config.ltx2_save_embeddings = args.ltx2_save_embeddings
    if getattr(args, "ltx2_load_embeddings", None) is not None:
        config.ltx2_load_embeddings = args.ltx2_load_embeddings
    if getattr(args, "ltx2_gemma_variant", None) is not None:
        config.ltx2_gemma_variant = args.ltx2_gemma_variant

    # Wan/HuMo video overrides
    if getattr(args, "wan_humo_path", None) is not None:
        config.wan_humo_path = args.wan_humo_path
    if getattr(args, "wan_base_path", None) is not None:
        config.wan_base_path = args.wan_base_path
    if getattr(args, "wan_whisper_path", None) is not None:
        config.wan_whisper_path = args.wan_whisper_path
    if getattr(args, "wan_humo_variant", None) is not None:
        config.wan_humo_variant = args.wan_humo_variant
    if getattr(args, "wan_num_frames", None) is not None:
        config.wan_num_frames = args.wan_num_frames
    if getattr(args, "wan_fps", None) is not None:
        config.wan_fps = args.wan_fps
    if getattr(args, "wan_height", None) is not None:
        config.wan_height = args.wan_height
    if getattr(args, "wan_width", None) is not None:
        config.wan_width = args.wan_width
    if getattr(args, "wan_guidance_scale", None) is not None:
        config.wan_guidance_scale = args.wan_guidance_scale
    if getattr(args, "wan_audio_scale", None) is not None:
        config.wan_audio_scale = args.wan_audio_scale
    if getattr(args, "wan_steps", None) is not None:
        config.wan_steps = args.wan_steps
    if getattr(args, "wan_offload_mode", None) is not None:
        config.wan_offload_mode = args.wan_offload_mode
    if getattr(args, "wan_output", None) is not None:
        config.wan_output_path = args.wan_output

    # FLUX.2 Klein overrides
    if getattr(args, "flux2_model_name", None) is not None:
        config.flux2_model_name = args.flux2_model_name
    if getattr(args, "flux2_num_steps", None) is not None:
        config.flux2_num_steps = args.flux2_num_steps
    if getattr(args, "flux2_guidance", None) is not None:
        config.flux2_guidance = args.flux2_guidance
    if getattr(args, "flux2_seed", None) is not None:
        config.flux2_seed = args.flux2_seed
    if getattr(args, "flux2_offload", False):
        config.flux2_offload_between_stages = True
    if getattr(args, "flux2_no_offload", False):
        config.flux2_offload_between_stages = False
    if getattr(args, "flux2_block_offload", False):
        config.flux2_block_offload = True
    if getattr(args, "flux2_output", None) is not None:
        config.flux2_output_path = args.flux2_output
    if getattr(args, "flux2_input_image", None) is not None:
        config.flux2_input_images = args.flux2_input_image
    if getattr(args, "flux2_encoder_path", None) is not None:
        config.flux2_encoder_path = args.flux2_encoder_path
    if getattr(args, "flux2_model_path", None) is not None:
        config.flux2_model_path = args.flux2_model_path
    if getattr(args, "flux2_vae_path", None) is not None:
        config.flux2_vae_path = args.flux2_vae_path

    # Z-Image variant overrides
    if getattr(args, "zimage_variant", None) is not None:
        config.zimage_variant = args.zimage_variant
    if getattr(args, "zimage_model_path", None) is not None:
        config.zimage_model_path = args.zimage_model_path

    # Apply Z-Image variant-aware defaults
    # Auto-detect variant from model path if set to "auto"
    if config.zimage_variant == "auto":
        from llm_dit.models.zimage.constants import detect_zimage_variant
        model_path = config.zimage_model_path or config.model_path
        if model_path:
            config.zimage_variant = detect_zimage_variant(model_path)

    # Apply variant defaults when variant is known (explicit or auto-detected)
    if config.zimage_variant in ("base", "turbo"):
        from llm_dit.models.zimage.constants import get_variant_defaults

        variant_defaults = get_variant_defaults(config.zimage_variant)
        # Apply variant defaults for parameters not explicitly set by user
        if getattr(args, "shift", None) is None:
            config.shift = variant_defaults["shift"]
        if getattr(args, "steps", None) is None:
            config.steps = variant_defaults["num_inference_steps"]
        if getattr(args, "guidance_scale", None) is None:
            config.guidance_scale = variant_defaults["guidance_scale"]

    # Device overrides
    if getattr(args, "text_encoder_device", None) is not None:
        config.encoder_device = args.text_encoder_device
    if getattr(args, "dit_device", None) is not None:
        config.dit_device = args.dit_device
    if getattr(args, "vae_device", None) is not None:
        config.vae_device = args.vae_device

    # Optimization overrides
    if getattr(args, "cpu_offload", False):
        config.cpu_offload = True
    if getattr(args, "flash_attn", False):
        config.flash_attn = True
    if getattr(args, "compile", False):
        config.compile = True
    if getattr(args, "dtype", None) is not None:
        config.dtype = args.dtype
    if getattr(args, "quantization", None) is not None:
        config.quantization = args.quantization

    # Scheduler overrides
    if getattr(args, "shift", None) is not None:
        config.shift = args.shift
    if getattr(args, "shift_terminal", None) is not None:
        config.shift_terminal = args.shift_terminal
    if getattr(args, "dynamic_shift", False):
        config.dynamic_shift = True
    if getattr(args, "d_noise", None) is not None:
        config.d_noise = args.d_noise

    # CFG mode override
    if getattr(args, "cfg_norm_mode", None) is not None:
        config.cfg_norm_mode = args.cfg_norm_mode

    # PyTorch-native component overrides
    if getattr(args, "attention_backend", None) is not None:
        config.attention_backend = args.attention_backend
    if getattr(args, "use_custom_scheduler", False):
        config.use_custom_scheduler = True
    if getattr(args, "tiled_vae", False):
        config.tiled_vae = True
    if getattr(args, "tile_size", None) is not None:
        config.tile_size = args.tile_size
    if getattr(args, "tile_overlap", None) is not None:
        config.tile_overlap = args.tile_overlap
    if getattr(args, "embedding_cache", False):
        config.embedding_cache = True
    if getattr(args, "cache_size", None) is not None:
        config.cache_size = args.cache_size
    if getattr(args, "long_prompt_mode", None) is not None:
        config.long_prompt_mode = args.long_prompt_mode
    if getattr(args, "hidden_layer", None) is not None:
        config.hidden_layer = args.hidden_layer
    if getattr(args, "layer_weights", None) is not None:
        config.layer_weights = parse_layer_weights(args.layer_weights)

    # LoRA overrides
    if getattr(args, "loras", None):
        config.lora_paths = []
        config.lora_scales = []
        for lora_str in args.loras:
            path, scale = parse_lora_arg(lora_str)
            config.lora_paths.append(path)
            config.lora_scales.append(scale)

    # Prompt control overrides
    if getattr(args, "template", None) is not None:
        config.default_template = args.template
    if getattr(args, "system_prompt", None) is not None:
        config.system_prompt = args.system_prompt
    if getattr(args, "thinking_content", None) is not None:
        config.thinking_content = args.thinking_content
    if getattr(args, "assistant_content", None) is not None:
        config.assistant_content = args.assistant_content
    if getattr(args, "enable_thinking", False):
        config.enable_thinking = True

    # Generation overrides
    if getattr(args, "height", None) is not None:
        config.height = args.height
    if getattr(args, "width", None) is not None:
        config.width = args.width
    if getattr(args, "steps", None) is not None:
        config.steps = args.steps
    if getattr(args, "guidance_scale", None) is not None:
        config.guidance_scale = args.guidance_scale
    if getattr(args, "seed", None) is not None:
        config.seed = args.seed
    if getattr(args, "negative_prompt", None) is not None:
        config.negative_prompt = args.negative_prompt
    if getattr(args, "cfg_normalization", None) is not None:
        config.cfg_normalization = args.cfg_normalization
    if getattr(args, "cfg_truncation", None) is not None:
        config.cfg_truncation = args.cfg_truncation

    # DyPE overrides
    if getattr(args, "dype", False):
        config.dype_enabled = True
    if getattr(args, "dype_method", None) is not None:
        config.dype_method = args.dype_method
    if getattr(args, "dype_scale", None) is not None:
        config.dype_scale = args.dype_scale

    # SLG overrides
    if getattr(args, "slg_scale", None) is not None:
        config.slg_scale = args.slg_scale
    if getattr(args, "slg_layers", None) is not None:
        config.slg_layers = [int(x.strip()) for x in args.slg_layers.split(",")]

    # Server overrides
    if getattr(args, "host", None) is not None:
        config.host = args.host
    if getattr(args, "port", None) is not None:
        config.port = args.port

    # Debug overrides
    if getattr(args, "debug", False):
        config.debug = True
    if getattr(args, "verbose", False):
        config.verbose = True
    if getattr(args, "log_dir", None) is not None:
        config.log_dir = args.log_dir

    return config


def load_runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    """
    Load runtime configuration from TOML file + CLI overrides.

    Priority (highest to lowest):
    1. CLI arguments
    2. TOML config file (with profile)
    3. Defaults

    Args:
        args: Parsed CLI arguments

    Returns:
        RuntimeConfig with all settings resolved
    """
    # Start with defaults
    config = RuntimeConfig()
    from pathlib import Path

    # Load TOML config if provided
    toml_config: Config | None = None
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            # Only warn if using default path; error if user explicitly specified
            if args.config == "config.toml":
                logger.warning(f"Default config file not found: {args.config}, using defaults")
            else:
                raise FileNotFoundError(f"Config file not found: {args.config}")
        else:
            try:
                # Pass profile to from_toml - if None, it auto-detects flat vs profile-based config
                toml_config = Config.from_toml(args.config, args.profile)
                if args.profile:
                    logger.info(f"Loaded config profile: {args.profile}")
                # Note: from_toml logs "Loaded flat config" when auto-detecting

                # Apply TOML values to runtime config
                config.default_pipeline = toml_config.default_pipeline or config.default_pipeline
                config.model_path = toml_config.model_path or config.model_path
                config.templates_dir = toml_config.templates_dir or config.templates_dir
                config.encoder_device = toml_config.encoder.device
                config.dtype = toml_config.encoder.dtype
                config.hidden_layer = toml_config.encoder.hidden_layer
                config.quantization = toml_config.encoder.quantization

                # Generation defaults from config
                config.height = toml_config.generation.height
                config.width = toml_config.generation.width
                config.steps = toml_config.generation.num_inference_steps
                config.guidance_scale = toml_config.generation.guidance_scale
                config.cfg_normalization = getattr(toml_config.generation, "cfg_normalization", 0.0)
                config.cfg_truncation = getattr(toml_config.generation, "cfg_truncation", 1.0)
                config.enable_thinking = toml_config.generation.enable_thinking
                config.default_template = toml_config.generation.default_template

                # Pipeline settings
                config.cpu_offload = toml_config.pipeline.enable_model_cpu_offload

                # Check for optimization section
                if hasattr(toml_config, "optimization"):
                    opt = toml_config.optimization
                    config.flash_attn = getattr(opt, "flash_attn", False)
                    config.compile = getattr(opt, "compile", False)
                    config.compile_mode = getattr(opt, "compile_mode", "max-autotune-no-cudagraphs")
                    # Note: cpu_offload is set from [pipeline].enable_model_cpu_offload (line 1822)
                    # Do NOT override from [optimization].cpu_offload - that was a legacy conflict

                # Check for scheduler section
                if hasattr(toml_config, "scheduler"):
                    sched = toml_config.scheduler
                    config.shift = getattr(sched, "shift", 3.0)
                    config.dynamic_shift = getattr(sched, "dynamic_shift", False)
                    config.d_noise = getattr(sched, "d_noise", 1.0)

                # Check for LoRA section
                if hasattr(toml_config, "lora"):
                    lora = toml_config.lora
                    config.lora_paths = getattr(lora, "paths", [])
                    config.lora_scales = getattr(lora, "scales", [])

                # Check for PyTorch-native section
                if hasattr(toml_config, "pytorch"):
                    pytorch = toml_config.pytorch
                    config.attention_backend = getattr(pytorch, "attention_backend", None)
                    config.use_custom_scheduler = getattr(pytorch, "use_custom_scheduler", False)
                    config.tiled_vae = getattr(pytorch, "tiled_vae", False)
                    config.tile_size = getattr(pytorch, "tile_size", 512)
                    config.tile_overlap = getattr(pytorch, "tile_overlap", 64)
                    config.embedding_cache = getattr(pytorch, "embedding_cache", False)
                    config.cache_size = getattr(pytorch, "cache_size", 100)
                    config.long_prompt_mode = getattr(pytorch, "long_prompt_mode", "interpolate")

                # Check for rewriter section
                if hasattr(toml_config, "rewriter"):
                    rewriter = toml_config.rewriter
                    config.rewriter_use_api = getattr(rewriter, "use_api", False)
                    config.rewriter_api_url = getattr(rewriter, "api_url", "")
                    config.rewriter_api_model = getattr(rewriter, "api_model", "Qwen3-4B")
                    config.rewriter_temperature = getattr(rewriter, "temperature", 0.6)
                    config.rewriter_top_p = getattr(rewriter, "top_p", 0.95)
                    config.rewriter_top_k = getattr(rewriter, "top_k", 20)
                    config.rewriter_min_p = getattr(rewriter, "min_p", 0.0)
                    config.rewriter_presence_penalty = getattr(rewriter, "presence_penalty", 0.0)
                    config.rewriter_max_tokens = getattr(rewriter, "max_tokens", 512)
                    config.rewriter_vl_enabled = getattr(rewriter, "vl_enabled", True)
                    config.rewriter_preload_vl = getattr(rewriter, "preload_vl", False)
                    config.rewriter_vl_api_model = getattr(rewriter, "vl_api_model", "")
                    config.rewriter_timeout = getattr(rewriter, "timeout", 120.0)

                # Check for VL section
                if hasattr(toml_config, "vl"):
                    vl = toml_config.vl
                    config.vl_model_path = getattr(vl, "model_path", "")
                    config.vl_device = getattr(vl, "device", "cpu")
                    config.vl_alpha = getattr(vl, "default_alpha", 0.3)
                    config.vl_hidden_layer = getattr(vl, "default_hidden_layer", -2)
                    config.vl_auto_unload = getattr(vl, "auto_unload", True)

                # Check for Qwen-Image section
                if hasattr(toml_config, "qwen_image"):
                    qi = toml_config.qwen_image
                    config.qwen_image_model_path = getattr(qi, "model_path", "")
                    config.qwen_image_edit_model_path = getattr(qi, "edit_model_path", "")
                    config.qwen_image_cpu_offload = getattr(qi, "cpu_offload", True)
                    config.qwen_image_layer_num = getattr(qi, "layer_num", 4)
                    config.qwen_image_steps = getattr(qi, "num_inference_steps", 25)
                    config.qwen_image_cfg_scale = getattr(qi, "cfg_scale", 4.0)
                    config.qwen_image_resolution = getattr(qi, "resolution", 640)
                    config.qwen_image_quantize_text_encoder = getattr(
                        qi, "quantize_text_encoder", "none"
                    )
                    config.qwen_image_quantize_transformer = getattr(qi, "quantize_transformer", "none")

                # Check for LTX-2 section
                if hasattr(toml_config, "ltx2"):
                    ltx2 = toml_config.ltx2
                    config.ltx2_model_path = getattr(ltx2, "model_path", "")
                    config.ltx2_num_frames = getattr(ltx2, "num_frames", 33)
                    config.ltx2_fps = getattr(ltx2, "fps", 24)
                    config.ltx2_guidance_scale = getattr(ltx2, "guidance_scale", 3.5)
                    config.ltx2_steps = getattr(ltx2, "num_inference_steps", None)
                    config.ltx2_lora_path = getattr(ltx2, "lora_path", "")
                    config.ltx2_lora_scale = getattr(ltx2, "lora_scale", 1.0)
                    config.ltx2_audio = getattr(ltx2, "audio_enabled", False)
                    # LTX-2 optimization settings
                    config.ltx2_text_encoder_device = getattr(ltx2, "text_encoder_device", "cpu")
                    config.ltx2_transformer_device = getattr(ltx2, "transformer_device", "cuda")
                    config.ltx2_vae_device = getattr(ltx2, "vae_device", "cuda")
                    config.ltx2_quantize = getattr(ltx2, "quantize", "fp8")
                    config.ltx2_skip_cleanup = getattr(ltx2, "skip_cleanup", False)

                # Check for FLUX.2 section
                if hasattr(toml_config, "flux2"):
                    flux2 = toml_config.flux2
                    # Use 'or' to handle empty strings as falsy
                    config.flux2_model_path = flux2.model_path or config.flux2_model_path
                    config.flux2_vae_path = flux2.vae_path or config.flux2_vae_path
                    config.flux2_encoder_path = flux2.encoder_path or config.flux2_encoder_path
                    config.flux2_model_name = flux2.default_model or config.flux2_model_name
                    config.flux2_block_offload = flux2.block_offload
                    # Read offload_between_stages from TOML (defaults to True for memory efficiency)
                    config.flux2_offload_between_stages = getattr(flux2, "offload_between_stages", True)
                    # Read encoder_device from TOML (defaults to cuda)
                    config.flux2_encoder_device = getattr(flux2, "encoder_device", "cuda")
                    if flux2.default_steps is not None:
                        config.flux2_num_steps = flux2.default_steps
                    if flux2.default_guidance is not None:
                        config.flux2_guidance = flux2.default_guidance

                # Check for Z-Image section
                if hasattr(toml_config, "zimage"):
                    zimage = toml_config.zimage
                    config.zimage_model_path = getattr(zimage, "model_path", "") or config.zimage_model_path
                    config.zimage_variant = getattr(zimage, "variant", "auto")
                    # Generation params (steps, guidance_scale, shift, negative_prompt) are now
                    # handled by presets - see presets/zimage/. CLI flags still override.

                # Check for Wan section
                if hasattr(toml_config, "wan"):
                    wan = toml_config.wan
                    config.wan_humo_path = getattr(wan, "humo_path", "")
                    config.wan_base_path = getattr(wan, "base_path", "")
                    config.wan_whisper_path = getattr(wan, "whisper_path", "")
                    config.wan_humo_variant = getattr(wan, "humo_variant", "17B")
                    config.wan_num_frames = getattr(wan, "num_frames", 97)
                    config.wan_fps = getattr(wan, "fps", 25)
                    config.wan_height = getattr(wan, "height", 720)
                    config.wan_width = getattr(wan, "width", 1280)
                    config.wan_guidance_scale = getattr(wan, "guidance_scale", 5.0)
                    config.wan_audio_scale = getattr(wan, "audio_scale", 0.0)
                    config.wan_steps = getattr(wan, "num_inference_steps", 50)
                    config.wan_offload_mode = getattr(wan, "offload_mode", "model")

                # Check for DyPE section
                if hasattr(toml_config, "dype"):
                    dype = toml_config.dype
                    config.dype_enabled = getattr(dype, "enabled", False)
                    config.dype_method = getattr(dype, "method", "vision_yarn")
                    config.dype_scale = getattr(dype, "dype_scale", 2.0)
                    config.dype_exponent = getattr(dype, "dype_exponent", 2.0)
                    config.dype_start_sigma = getattr(dype, "dype_start_sigma", 1.0)
                    config.dype_base_shift = getattr(dype, "base_shift", 0.5)
                    config.dype_max_shift = getattr(dype, "max_shift", 1.15)
                    config.dype_base_resolution = getattr(dype, "base_resolution", 1024)
                    config.dype_anisotropic = getattr(dype, "anisotropic", False)
                    config.dype_multipass = getattr(dype, "multipass", "single")
                    config.dype_pass2_strength = getattr(dype, "pass2_strength", 0.5)
                    config.dype_pass3_strength = getattr(dype, "pass3_strength", 0.4)
                    config.dype_frequency_modulation = getattr(dype, "frequency_modulation", False)

                # Check for SLG (Skip Layer Guidance) section
                if hasattr(toml_config, "slg"):
                    slg = toml_config.slg
                    if getattr(slg, "enabled", False):
                        config.slg_scale = getattr(slg, "scale", 2.5)
                        config.slg_layers = getattr(slg, "layers", [7, 8, 9, 10, 11, 12])
                        config.slg_start = getattr(slg, "start", 0.05)
                        config.slg_stop = getattr(slg, "stop", 0.5)

                # Check for FMTT (Flow Map Trajectory Tilting) section
                if hasattr(toml_config, "fmtt"):
                    fmtt = toml_config.fmtt
                    # Always load siglip_model and siglip_device from config
                    config.fmtt_siglip_model = getattr(
                        fmtt, "siglip_model", "google/siglip2-giant-opt-patch16-384"
                    )
                    config.fmtt_siglip_device = getattr(fmtt, "siglip_device", "cuda")
                    if getattr(fmtt, "enabled", False):
                        config.fmtt_scale = getattr(fmtt, "guidance_scale", 1.0)
                        config.fmtt_start = getattr(fmtt, "guidance_start", 0.0)
                        config.fmtt_stop = getattr(fmtt, "guidance_stop", 0.5)
                        config.fmtt_normalize = getattr(fmtt, "normalize_mode", "unit")
                        config.fmtt_decode_scale = getattr(fmtt, "decode_scale", 0.5)

                # Check for FBCache (Forward Block Cache) section
                if hasattr(toml_config, "fbcache"):
                    fbcache = toml_config.fbcache
                    if getattr(fbcache, "enabled", False):
                        config.fbcache = True
                        config.fbcache_threshold = getattr(fbcache, "middle_threshold", 0.05)
                        config.fbcache_log = getattr(fbcache, "log_residuals", False)

            except Exception as e:
                logger.warning(f"Failed to load config: {e}")

    # Also check for server section in TOML (only if file exists)
    if args.config and Path(args.config).exists():
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib

        with open(args.config, "rb") as f:
            raw_toml = tomllib.load(f)
        server_cfg = raw_toml.get("server", {})
        config.host = server_cfg.get("host", config.host)
        config.port = server_cfg.get("port", config.port)

    # Apply CLI overrides using shared function
    config = _apply_cli_overrides(args, config)

    return config


def setup_logging(config: RuntimeConfig) -> None:
    """Configure logging based on runtime config.

    Uses structured JSON file logging with rotation when log_dir is set.
    Console output remains human-readable.
    """
    from llm_dit.utils.logging_config import setup_logging as configure_logging

    level = logging.DEBUG if config.debug or config.verbose else logging.INFO

    # Check if JSON file logging is requested
    log_dir = getattr(config, "log_dir", None)
    enable_json = log_dir is not None

    configure_logging(
        level=level,
        log_dir=Path(log_dir) if log_dir else None,
        enable_json_file=enable_json,
    )

    if config.debug:
        # Enable debug for all project modules
        logging.getLogger("llm_dit").setLevel(logging.DEBUG)
        logging.getLogger("llm_dit.backends").setLevel(logging.DEBUG)
        logging.getLogger("llm_dit.pipelines").setLevel(logging.DEBUG)
        logging.getLogger("web").setLevel(logging.DEBUG)
        logging.getLogger("__main__").setLevel(logging.DEBUG)  # For direct script execution
