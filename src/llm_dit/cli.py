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

last updated: 2026-02-06
"""

import argparse
import logging
from pathlib import Path
from typing import Literal, get_args

from .config import Config, RuntimeConfig

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
        choices=["none", "fp8-dynamic", "fp8-weight-only", "int8", "int4"],
        default=None,
        help="Quantization for text encoder (Qwen2.5-VL-7B): torchao methods",
    )
    qwen_group.add_argument(
        "--qwen-image-quantize-transformer",
        type=str,
        choices=["none", "fp8-dynamic", "fp8-weight-only", "int8", "int4"],
        default=None,
        help="Quantization for DiT (variant default: fp8-weight-only)",
    )

    # LTX-2 video generation
    ltx2_group = parser.add_argument_group("LTX-2 Video Generation")
    ltx2_group.add_argument("--ltx2-model-path", type=str, default=None,
        help="Path to LTX-2 model directory (e.g., ~/Storage/LTX-2)")
    ltx2_group.add_argument("--ltx2-encoder-model-id", type=str, default=None,
        help="Path to text encoder (Gemma 3 compatible).")
    ltx2_group.add_argument("--ltx2-num-frames", type=int, default=None,
        help="Number of video frames (default: 33, max ~65 for 24GB)")
    ltx2_group.add_argument("--ltx2-fps", type=int, default=None,
        help="Output framerate (default: 24)")
    ltx2_group.add_argument("--ltx2-guidance-scale", type=float, default=None,
        help="CFG guidance scale (default: 3.5, range 3.0-4.0)")
    ltx2_group.add_argument("--ltx2-steps", type=int, default=None,
        help="Diffusion steps (default: 12 for distilled model)")
    ltx2_group.add_argument("--ltx2-lora-path", type=str, default=None,
        help="Path to LoRA safetensors file")
    ltx2_group.add_argument("--ltx2-lora-scale", type=float, default=None,
        help="LoRA blend scale (default: 1.0)")
    ltx2_group.add_argument("--ltx2-audio", action="store_true", default=None,
        help="Enable audio generation")
    ltx2_group.add_argument("--ltx2-output", type=str, default=None,
        help="Output video path (default: output.mp4)")
    ltx2_group.add_argument("--ltx2-save-embeddings", type=str, default=None,
        help="Save text embeddings to file (skip video generation, for precomputation).")
    ltx2_group.add_argument("--ltx2-load-embeddings", type=str, default=None,
        help="Load pre-computed embeddings from file (skip text encoding).")

    # LTX-2 Optimization
    ltx2_opt = parser.add_argument_group("LTX-2 Optimization")
    ltx2_opt.add_argument("--ltx2-text-encoder-device", choices=["cpu", "cuda"], default=None,
        help="Device for Gemma3 text encoder (cpu recommended for 24GB, default: cpu)")
    ltx2_opt.add_argument("--ltx2-transformer-device", choices=["cpu", "cuda"], default=None,
        help="Device for DiT transformer (default: cuda)")
    ltx2_opt.add_argument("--ltx2-vae-device", choices=["cpu", "cuda"], default=None,
        help="Device for VAE decoder (default: cuda)")
    ltx2_opt.add_argument("--ltx2-quantize", choices=["none", "fp8"], default=None,
        help="Transformer quantization (fp8 for 24GB GPUs, default: fp8)")
    ltx2_opt.add_argument("--ltx2-skip-cleanup", action="store_true",
        help="Skip memory cleanup between stages (faster, needs more VRAM)")
    ltx2_opt.add_argument("--ltx2-gemma-variant", choices=["bf16", "8bit", "q4-qat"], default=None,
        help="Gemma3 backbone variant: bf16 (full), 8bit (TorchAO), q4-qat (pre-quantized). Default: bf16")

    # Wan/HuMo video generation
    wan_group = parser.add_argument_group("Wan/HuMo Video Generation")
    wan_group.add_argument("--wan-humo-path", type=str, default=None,
        help="Path to HuMo transformer (e.g., ~/Storage/HuMo)")
    wan_group.add_argument("--wan-base-path", type=str, default=None,
        help="Path to Wan2.1-T2V for VAE/text encoder")
    wan_group.add_argument("--wan-whisper-path", type=str, default=None,
        help="Path to Whisper for audio (optional)")
    wan_group.add_argument("--wan-humo-variant", type=str, choices=["17B", "1.7B"], default=None,
        help="HuMo variant (default: 17B)")
    wan_group.add_argument("--wan-num-frames", type=int, default=None,
        help="Number of video frames (default: 97, ~3.9s at 25fps)")
    wan_group.add_argument("--wan-fps", type=int, default=None,
        help="Output framerate (default: 25 for HuMo)")
    wan_group.add_argument("--wan-height", type=int, default=None,
        help="Video height (default: 720, multiple of 16)")
    wan_group.add_argument("--wan-width", type=int, default=None,
        help="Video width (default: 1280, multiple of 16)")
    wan_group.add_argument("--wan-guidance-scale", type=float, default=None,
        help="Text guidance scale_t (default: 5.0)")
    wan_group.add_argument("--wan-audio-scale", type=float, default=None,
        help="Audio guidance scale_a (default: 0.0, set >0 for audio mode)")
    wan_group.add_argument("--wan-steps", type=int, default=None,
        help="Diffusion steps (default: 50 for HuMo)")
    wan_group.add_argument("--wan-offload-mode", type=str, choices=["none", "model", "sequential"],
        default=None, help="CPU offload mode (default: model for 24GB VRAM)")
    wan_group.add_argument("--wan-output", type=str, default=None,
        help="Output video path (default: wan_output.mp4)")

    # FLUX.2 Klein image generation
    flux2_group = parser.add_argument_group("FLUX.2 Klein Image Generation")
    flux2_group.add_argument("--flux2-model-name", type=str,
        choices=["klein-4b", "klein-9b", "klein-base-4b", "klein-base-9b",
                 "klein-4b-fp8", "klein-9b-fp8", "klein-base-4b-fp8", "klein-base-9b-fp8"],
        default=None,
        help="FLUX.2 Klein model variant (default: klein-9b). "
        "Distilled models use 4 steps, base models use 50 steps. FP8 variants use half memory.")
    flux2_group.add_argument("--flux2-num-steps", type=int, default=None,
        help="Number of denoising steps (default: 4 for distilled, 50 for base)")
    flux2_group.add_argument("--flux2-guidance", type=float, default=None,
        help="Guidance scale (default: 1.0 for distilled, 4.0 for base)")
    flux2_group.add_argument("--flux2-seed", type=int, default=None,
        help="Random seed for reproducibility")
    flux2_group.add_argument("--flux2-offload", action="store_true", default=None,
        help="Enable three-stage memory offloading (default: True)")
    flux2_group.add_argument("--flux2-no-offload", action="store_true", default=None,
        help="Disable memory offloading (requires more VRAM)")
    flux2_group.add_argument("--flux2-block-offload", action="store_true", default=None,
        help="Enable block-by-block offloading (slower but uses ~5GB less VRAM)")
    flux2_group.add_argument("--flux2-output", type=str, default=None,
        help="Output image path (default: flux2_output.png)")
    flux2_group.add_argument("--flux2-input-image", type=str, nargs="+", default=None,
        help="Input image(s) for editing mode. Can specify multiple images.")
    flux2_group.add_argument("--flux2-encoder-path", type=str, default=None,
        help="Custom path for Qwen3 text encoder (local path or HF model ID).")
    flux2_group.add_argument("--flux2-model-path", type=str, default=None,
        help="Local path to transformer weights (file or directory).")
    flux2_group.add_argument("--flux2-vae-path", type=str, default=None,
        help="Local path to VAE weights (file or directory).")

    # Device placement
    device_group = parser.add_argument_group("Devices")
    device_group.add_argument("--text-encoder-device", type=str,
        choices=["cpu", "cuda", "mps", "auto"], default=None,
        help="Device for text encoder (default: auto)")
    device_group.add_argument("--dit-device", type=str,
        choices=["cpu", "cuda", "mps", "auto"], default=None,
        help="Device for DiT/transformer (default: auto)")
    device_group.add_argument("--vae-device", type=str,
        choices=["cpu", "cuda", "mps", "auto"], default=None,
        help="Device for VAE (default: auto)")

    # Optimization
    opt_group = parser.add_argument_group("Optimization")
    opt_group.add_argument("--cpu-offload", action="store_true",
        help="Enable CPU offload for transformer")
    opt_group.add_argument("--flash-attn", action="store_true",
        help="Enable Flash Attention (requires flash-attn package)")
    opt_group.add_argument("--compile", action="store_true",
        help="Compile transformer with torch.compile (slower first run)")
    opt_group.add_argument("--torch-dtype", type=str,
        choices=["bfloat16", "float16", "float32"], default=None,
        help="Model precision (default: bfloat16)")
    opt_group.add_argument("--quantization", type=str,
        choices=["none", "fp8-dynamic", "fp8-weight-only", "int8", "int4"], default=None,
        help="Text encoder quantization: none (default), fp8-dynamic/fp8-weight-only (RTX 4090+), int8/int4 (torchao)")

    # PyTorch-native components
    pytorch_group = parser.add_argument_group("PyTorch Native")
    pytorch_group.add_argument("--attention-backend", type=str,
        choices=["auto", "flash_attn_2", "flash_attn_3", "sage", "xformers", "sdpa"], default=None,
        help="Attention backend (default: auto-detect best available)")
    pytorch_group.add_argument("--use-custom-scheduler", action="store_true",
        help="Use our pure-PyTorch FlowMatchScheduler instead of diffusers")
    pytorch_group.add_argument("--tiled-vae", action="store_true",
        help="Enable tiled VAE decode for large images (2K+)")
    pytorch_group.add_argument("--tile-size", type=int, default=None,
        help="Tile size for VAE decode in pixels (default: 512)")
    pytorch_group.add_argument("--tile-overlap", type=int, default=None,
        help="Overlap between VAE tiles in pixels (default: 64)")
    pytorch_group.add_argument("--embedding-cache", action="store_true",
        help="Enable embedding cache for repeated prompts")
    pytorch_group.add_argument("--cache-size", type=int, default=None,
        help="Maximum number of cached embeddings (default: 100)")
    pytorch_group.add_argument("--long-prompt-mode", type=str,
        choices=["truncate", "interpolate", "pool", "attention_pool"], default=None,
        help="How to handle prompts exceeding 1504 tokens (default: interpolate)")
    pytorch_group.add_argument("--hidden-layer", type=int, default=None,
        help="Which hidden layer to extract embeddings from (default: -2). "
        "-1=last layer, -2=penultimate (default for Z-Image).")
    pytorch_group.add_argument("--layer-weights", type=str, default=None,
        help="Multi-layer blending weights (overrides --hidden-layer). "
        "Format: 'layer:weight,layer:weight,...' e.g. '-2:0.7,-6:0.3'")

    # Scheduler
    sched_group = parser.add_argument_group("Scheduler")
    sched_group.add_argument("--shift", type=float, default=None,
        help="Scheduler shift parameter (default: 3.0)")
    sched_group.add_argument("--shift-terminal", type=float, default=None,
        help="Stretch sigma schedule to end at this value instead of 0 (Qwen-Image only, default: None)")
    sched_group.add_argument("--dynamic-shift", action="store_true",
        help="Calculate shift based on resolution (overrides --shift)")
    sched_group.add_argument("--d-noise", type=float, default=None,
        help="Sigma schedule scaling factor. <1.0 = sharper, >1.0 = softer. Default: 1.0")

    # LoRA
    lora_group = parser.add_argument_group("LoRA")
    lora_group.add_argument("--lora", type=str, action="append", default=None,
        dest="loras", metavar="PATH:SCALE",
        help="Load LoRA weights (repeatable). Format: path/to/lora.safetensors:0.8")

    # Prompt control
    prompt_group = parser.add_argument_group("Prompt Control")
    prompt_group.add_argument("--template", type=str, default=None,
        help="Template name for encoding")
    prompt_group.add_argument("--system-prompt", type=str, default=None,
        help="System prompt for chat template")
    prompt_group.add_argument("--thinking-content", type=str, default=None,
        help="Content inside <think>...</think> block")
    prompt_group.add_argument("--assistant-content", type=str, default=None,
        help="Content after </think> (assistant response prefix)")
    prompt_group.add_argument("--enable-thinking", action="store_true",
        help="Add <think></think> structure to prompt")

    # API backend
    api_group = parser.add_argument_group("API Backend")
    api_group.add_argument("--api-url", type=str, default=None,
        help="Use remote API for encoding (e.g., http://mac-ip:8080)")
    api_group.add_argument("--api-model", type=str, default=None,
        help="Model ID for API backend (default: Qwen3-4B-mlx)")
    api_group.add_argument("--local-encoder", action="store_true",
        help="Force local encoder (for A/B testing API vs local)")

    # Rewriter settings
    rewriter_group = parser.add_argument_group("Rewriter")
    rewriter_group.add_argument("--rewriter-use-api", action="store_true",
        help="Use API backend for prompt rewriting")
    rewriter_group.add_argument("--rewriter-api-url", type=str, default=None,
        help="API URL for rewriter (defaults to --api-url if not set)")
    rewriter_group.add_argument("--rewriter-api-model", type=str, default=None,
        help="Model ID for rewriter API (default: Qwen3-4B)")
    rewriter_group.add_argument("--rewriter-temperature", type=float, default=None,
        help="Sampling temperature for rewriter (default: 0.6)")
    rewriter_group.add_argument("--rewriter-top-p", type=float, default=None,
        help="Nucleus sampling threshold for rewriter (default: 0.95)")
    rewriter_group.add_argument("--rewriter-top-k", type=int, default=None,
        help="Top-k sampling for rewriter (default: 20)")
    rewriter_group.add_argument("--rewriter-min-p", type=float, default=None,
        help="Minimum probability threshold for rewriter (default: 0.0)")
    rewriter_group.add_argument("--rewriter-presence-penalty", type=float, default=None,
        help="Presence penalty for rewriter (0-2, default: 0.0)")
    rewriter_group.add_argument("--rewriter-max-tokens", type=int, default=None,
        help="Maximum tokens for rewriter (default: 512)")
    rewriter_group.add_argument("--rewriter-timeout", type=float, default=None,
        help="API request timeout in seconds (default: 120)")

    # DyPE (Dynamic Position Extrapolation) for high-resolution generation
    dype_group = parser.add_argument_group("DyPE (High-Resolution)")
    dype_group.add_argument("--dype", action="store_true",
        help="Enable DyPE for high-resolution generation (2K-4K+)")
    dype_group.add_argument("--dype-method", type=str,
        choices=["vision_yarn", "yarn", "ntk"], default=None,
        help="RoPE extrapolation method (default: vision_yarn)")
    dype_group.add_argument("--dype-scale", type=float, default=None,
        help="DyPE magnitude (lambda_s, default: 2.0)")
    dype_group.add_argument("--dype-exponent", type=float, default=None,
        help="DyPE decay speed (lambda_t, default: 2.0 = quadratic)")
    dype_group.add_argument("--dype-start-sigma", type=float, default=None,
        help="When to start DyPE decay (0-1, 1.0 = from start, default: 1.0)")
    dype_group.add_argument("--dype-base-shift", type=float, default=None,
        help="Noise schedule shift at base resolution (default: 0.5)")
    dype_group.add_argument("--dype-max-shift", type=float, default=None,
        help="Noise schedule shift at max resolution (default: 1.15)")
    dype_group.add_argument("--dype-base-resolution", type=int, default=None,
        help="Training resolution (default: 1024)")
    dype_group.add_argument("--dype-anisotropic", action="store_true",
        help="Use per-axis scaling for extreme aspect ratios")
    dype_group.add_argument("--dype-multipass", type=str,
        choices=["single", "twopass", "threepass"], default=None,
        help="Generation mode: single, twopass (512->target), threepass (256->512->target)")
    dype_group.add_argument("--dype-pass2-strength", type=float, default=None,
        help="img2img strength for second pass (0.0-1.0, default: 0.5)")
    dype_group.add_argument("--dype-pass3-strength", type=float, default=None,
        help="img2img strength for third pass (0.0-1.0, default: 0.4)")
    dype_group.add_argument("--dype-frequency-modulation", action="store_true",
        help="Enable timestep-based RoPE frequency scaling (experimental)")

    # Forward Block Cache (FBCache)
    fbcache_group = parser.add_argument_group("Forward Block Cache (FBCache)")
    fbcache_group.add_argument("--fbcache", action="store_true",
        help="Enable FBCache for 30-50%% inference speedup")
    fbcache_group.add_argument("--fbcache-threshold", type=float, default=None,
        help="Override middle-phase threshold (default: 0.05 = 5%%)")
    fbcache_group.add_argument("--fbcache-log", action="store_true",
        help="Log FBCache residual statistics for analysis")

    # Generation parameters (optional)
    if include_generation_args:
        gen_group = parser.add_argument_group("Generation")
        gen_group.add_argument("--height", type=int, default=None,
            help="Image height (default: 1024, must be divisible by 16)")
        gen_group.add_argument("--width", type=int, default=None,
            help="Image width (default: 1024, must be divisible by 16)")
        gen_group.add_argument("--steps", type=int, default=None,
            help="Number of inference steps (default: 9 for turbo)")
        gen_group.add_argument("--guidance-scale", type=float, default=None,
            help="CFG scale (default: 0.0, not needed for Z-Image-Turbo)")
        gen_group.add_argument("--negative-prompt", type=str, default=None,
            help="Negative prompt for CFG")
        gen_group.add_argument("--cfg-normalization", type=float, default=None,
            help="CFG norm clamping factor (0.0 = disabled, typical: 1.0-2.0)")
        gen_group.add_argument("--cfg-truncation", type=float, default=None,
            help="CFG truncation threshold (1.0 = never, typical: 0.5-0.8)")
        gen_group.add_argument("--cfg-norm-mode", type=str,
            choices=["clamp", "match"], default=None,
            help="CFG normalization mode (default: clamp)")
        gen_group.add_argument("--seed", type=int, default=None,
            help="Random seed for reproducibility")
        gen_group.add_argument("--embeddings-file", type=str, default=None,
            help="Path to pre-computed embeddings file (.safetensors). Skips text encoding.")

    # Server args (optional)
    if include_server_args:
        server_group = parser.add_argument_group("Server")
        server_group.add_argument("--host", type=str, default=None,
            help="Host to bind to (default: 127.0.0.1)")
        server_group.add_argument("--port", type=int, default=None,
            help="Port to bind to (default: 7860)")

    # Debug
    debug_group = parser.add_argument_group("Debug")
    debug_group.add_argument("--debug", action="store_true",
        help="Enable DEBUG-level logging for all project modules (very noisy)")
    debug_group.add_argument("--log-dir", type=str, default=None,
        help="Directory for JSON log files with rotation (enables file logging)")

    return parser


# ---------------------------------------------------------------------------
# CLI override helpers
# ---------------------------------------------------------------------------

def _set_if(args: argparse.Namespace, arg_name: str, obj: object, field: str) -> None:
    """Set obj.field = args.arg_name if the CLI arg was explicitly provided (not None)."""
    val = getattr(args, arg_name, None)
    if val is not None:
        setattr(obj, field, val)


def _set_flag(args: argparse.Namespace, arg_name: str, obj: object, field: str) -> None:
    """Set obj.field = True if a store_true CLI flag was passed."""
    if getattr(args, arg_name, False):
        setattr(obj, field, True)


def _apply_cli_overrides(args: argparse.Namespace, config: RuntimeConfig) -> RuntimeConfig:
    """
    Apply CLI argument overrides to a RuntimeConfig.

    CLI arguments take precedence over config file values.
    Only values explicitly provided (not None) override the config.
    """
    # Top-level overrides
    _set_if(args, "model_type", config, "model_type")
    _set_if(args, "model_path", config, "model_path")
    _set_if(args, "text_encoder_path", config, "text_encoder_path")
    _set_if(args, "templates_dir", config, "templates_dir")

    # Z-Image overrides
    _set_if(args, "zimage_variant", config.zimage, "variant")
    _set_if(args, "zimage_model_path", config.zimage, "model_path")

    # Apply Z-Image variant-aware defaults
    if config.zimage.variant == "auto":
        from llm_dit.models.zimage.constants import detect_zimage_variant
        model_path = config.zimage.model_path or config.model_path
        if model_path:
            config.zimage.variant = detect_zimage_variant(model_path)

    if config.zimage.variant in ("base", "turbo"):
        from llm_dit.models.zimage.constants import get_variant_defaults
        variant_defaults = get_variant_defaults(config.zimage.variant)
        if getattr(args, "shift", None) is None:
            config.scheduler.shift = variant_defaults["shift"]
        if getattr(args, "steps", None) is None:
            config.generation.num_inference_steps = variant_defaults["num_inference_steps"]
        if getattr(args, "guidance_scale", None) is None:
            config.generation.guidance_scale = variant_defaults["guidance_scale"]

    # Qwen-Image overrides
    _set_if(args, "qwen_image_model_path", config.qwen_image, "model_path")
    _set_if(args, "qwen_image_edit_model_path", config.qwen_image, "edit_model_path")
    _set_flag(args, "qwen_image_cpu_offload", config.qwen_image, "cpu_offload")
    _set_if(args, "qwen_image_layers", config.qwen_image, "layer_num")
    _set_if(args, "qwen_image_steps", config.qwen_image, "num_inference_steps")
    _set_if(args, "qwen_image_cfg_scale", config.qwen_image, "cfg_scale")
    _set_if(args, "qwen_image_resolution", config.qwen_image, "resolution")
    _set_if(args, "qwen_image_quantize_text_encoder", config.qwen_image, "quantize_text_encoder")
    _set_if(args, "qwen_image_quantize_transformer", config.qwen_image, "quantize_transformer")

    # LTX-2 overrides
    _set_if(args, "ltx2_model_path", config.ltx2, "model_path")
    _set_if(args, "ltx2_encoder_model_id", config.ltx2, "encoder_model_id")
    _set_if(args, "ltx2_num_frames", config.ltx2, "num_frames")
    _set_if(args, "ltx2_fps", config.ltx2, "fps")
    _set_if(args, "ltx2_guidance_scale", config.ltx2, "guidance_scale")
    _set_if(args, "ltx2_steps", config.ltx2, "num_inference_steps")
    _set_if(args, "ltx2_lora_path", config.ltx2, "lora_path")
    _set_if(args, "ltx2_lora_scale", config.ltx2, "lora_scale")
    _set_flag(args, "ltx2_audio", config.ltx2, "audio_enabled")
    _set_if(args, "ltx2_output", config.ltx2, "output_path")
    _set_if(args, "ltx2_save_embeddings", config.ltx2, "save_embeddings")
    _set_if(args, "ltx2_load_embeddings", config.ltx2, "load_embeddings")
    _set_if(args, "ltx2_text_encoder_device", config.ltx2, "text_encoder_device")
    _set_if(args, "ltx2_transformer_device", config.ltx2, "transformer_device")
    _set_if(args, "ltx2_vae_device", config.ltx2, "vae_device")
    _set_if(args, "ltx2_quantize", config.ltx2, "quantize")
    _set_flag(args, "ltx2_skip_cleanup", config.ltx2, "skip_cleanup")
    _set_if(args, "ltx2_gemma_variant", config.ltx2, "gemma_variant")

    # Wan/HuMo overrides
    _set_if(args, "wan_humo_path", config.wan, "humo_path")
    _set_if(args, "wan_base_path", config.wan, "base_path")
    _set_if(args, "wan_whisper_path", config.wan, "whisper_path")
    _set_if(args, "wan_humo_variant", config.wan, "humo_variant")
    _set_if(args, "wan_num_frames", config.wan, "num_frames")
    _set_if(args, "wan_fps", config.wan, "fps")
    _set_if(args, "wan_height", config.wan, "height")
    _set_if(args, "wan_width", config.wan, "width")
    _set_if(args, "wan_guidance_scale", config.wan, "guidance_scale")
    _set_if(args, "wan_audio_scale", config.wan, "audio_scale")
    _set_if(args, "wan_steps", config.wan, "num_inference_steps")
    _set_if(args, "wan_offload_mode", config.wan, "offload_mode")
    _set_if(args, "wan_output", config.wan, "output_path")

    # FLUX.2 Klein overrides
    _set_if(args, "flux2_model_name", config.flux2, "default_model")
    _set_if(args, "flux2_num_steps", config.flux2, "default_steps")
    _set_if(args, "flux2_guidance", config.flux2, "default_guidance")
    _set_if(args, "flux2_seed", config.flux2, "seed")
    _set_if(args, "flux2_encoder_path", config.flux2, "encoder_path")
    _set_if(args, "flux2_model_path", config.flux2, "model_path")
    _set_if(args, "flux2_vae_path", config.flux2, "vae_path")
    _set_if(args, "flux2_output", config.flux2, "output_path")
    _set_if(args, "flux2_input_image", config.flux2, "input_images")
    if getattr(args, "flux2_offload", False):
        config.flux2.offload_between_stages = True
    if getattr(args, "flux2_no_offload", False):
        config.flux2.offload_between_stages = False
    _set_flag(args, "flux2_block_offload", config.flux2, "block_offload")

    # Device overrides
    _set_if(args, "text_encoder_device", config.encoder, "device")
    _set_if(args, "dit_device", config.optimization, "dit_device")
    _set_if(args, "vae_device", config.optimization, "vae_device")

    # Optimization overrides
    _set_flag(args, "cpu_offload", config.optimization, "cpu_offload")
    _set_flag(args, "flash_attn", config.optimization, "flash_attn")
    _set_flag(args, "compile", config.optimization, "compile")
    _set_if(args, "torch_dtype", config.encoder, "dtype")
    _set_if(args, "quantization", config.encoder, "quantization")

    # Scheduler overrides
    _set_if(args, "shift", config.scheduler, "shift")
    _set_if(args, "shift_terminal", config.scheduler, "shift_terminal")
    _set_flag(args, "dynamic_shift", config.scheduler, "dynamic_shift")
    _set_if(args, "d_noise", config.scheduler, "d_noise")

    # PyTorch-native overrides
    _set_if(args, "attention_backend", config.pytorch, "attention_backend")
    _set_flag(args, "use_custom_scheduler", config.pytorch, "use_custom_scheduler")
    _set_flag(args, "tiled_vae", config.pytorch, "tiled_vae")
    _set_if(args, "tile_size", config.pytorch, "tile_size")
    _set_if(args, "tile_overlap", config.pytorch, "tile_overlap")
    _set_flag(args, "embedding_cache", config.pytorch, "embedding_cache")
    _set_if(args, "cache_size", config.pytorch, "cache_size")
    _set_if(args, "long_prompt_mode", config.pytorch, "long_prompt_mode")
    _set_if(args, "hidden_layer", config.encoder, "hidden_layer")
    if getattr(args, "layer_weights", None) is not None:
        config.encoder.layer_weights = parse_layer_weights(args.layer_weights)

    # CFG mode override
    _set_if(args, "cfg_norm_mode", config.generation, "cfg_norm_mode")

    # LoRA overrides
    if getattr(args, "loras", None):
        paths: list[str] = []
        scales: list[float] = []
        for lora_str in args.loras:
            path, scale = parse_lora_arg(lora_str)
            paths.append(path)
            scales.append(scale)
        config.lora.paths = paths
        config.lora.scales = scales

    # Prompt control overrides
    _set_if(args, "template", config.generation, "default_template")
    _set_if(args, "system_prompt", config.generation, "system_prompt")
    _set_if(args, "thinking_content", config.generation, "thinking_content")
    _set_if(args, "assistant_content", config.generation, "assistant_content")
    _set_flag(args, "enable_thinking", config.generation, "enable_thinking")

    # Generation overrides
    _set_if(args, "height", config.generation, "height")
    _set_if(args, "width", config.generation, "width")
    _set_if(args, "steps", config.generation, "num_inference_steps")
    _set_if(args, "guidance_scale", config.generation, "guidance_scale")
    _set_if(args, "seed", config.generation, "seed")
    _set_if(args, "negative_prompt", config.generation, "negative_prompt")
    _set_if(args, "cfg_normalization", config.generation, "cfg_normalization")
    _set_if(args, "cfg_truncation", config.generation, "cfg_truncation")

    # DyPE overrides
    _set_flag(args, "dype", config.dype, "enabled")
    _set_if(args, "dype_method", config.dype, "method")
    _set_if(args, "dype_scale", config.dype, "dype_scale")
    _set_if(args, "dype_exponent", config.dype, "dype_exponent")
    _set_if(args, "dype_start_sigma", config.dype, "dype_start_sigma")
    _set_if(args, "dype_base_shift", config.dype, "base_shift")
    _set_if(args, "dype_max_shift", config.dype, "max_shift")
    _set_if(args, "dype_base_resolution", config.dype, "base_resolution")
    _set_flag(args, "dype_anisotropic", config.dype, "anisotropic")
    _set_if(args, "dype_multipass", config.dype, "multipass")
    _set_if(args, "dype_pass2_strength", config.dype, "pass2_strength")
    _set_if(args, "dype_pass3_strength", config.dype, "pass3_strength")
    _set_flag(args, "dype_frequency_modulation", config.dype, "frequency_modulation")

    # FBCache overrides
    _set_flag(args, "fbcache", config.fbcache, "enabled")
    _set_if(args, "fbcache_threshold", config.fbcache, "middle_threshold")
    _set_flag(args, "fbcache_log", config.fbcache, "log_residuals")

    # Rewriter overrides
    _set_flag(args, "rewriter_use_api", config.rewriter, "use_api")
    _set_if(args, "rewriter_api_url", config.rewriter, "api_url")
    _set_if(args, "rewriter_api_model", config.rewriter, "api_model")
    _set_if(args, "rewriter_temperature", config.rewriter, "temperature")
    _set_if(args, "rewriter_top_p", config.rewriter, "top_p")
    _set_if(args, "rewriter_top_k", config.rewriter, "top_k")
    _set_if(args, "rewriter_min_p", config.rewriter, "min_p")
    _set_if(args, "rewriter_presence_penalty", config.rewriter, "presence_penalty")
    _set_if(args, "rewriter_max_tokens", config.rewriter, "max_tokens")
    _set_if(args, "rewriter_timeout", config.rewriter, "timeout")

    # API overrides
    _set_if(args, "api_url", config.api, "url")
    _set_if(args, "api_model", config.api, "model")
    _set_flag(args, "local_encoder", config.api, "local_encoder")

    # Server overrides
    _set_if(args, "host", config, "host")
    _set_if(args, "port", config, "port")

    # Debug overrides
    _set_flag(args, "debug", config, "debug")
    _set_if(args, "log_dir", config.logging, "log_dir")

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
    config = RuntimeConfig()

    # Load TOML config if provided
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            if args.config == "config.toml":
                logger.warning(f"Default config file not found: {args.config}, using defaults")
            else:
                raise FileNotFoundError(f"Config file not found: {args.config}")
        else:
            try:
                # Parse TOML into Config dataclass, then build RuntimeConfig
                toml_config = Config.from_toml(args.config, args.profile)
                config = RuntimeConfig.from_toml_config(toml_config)

                if args.profile:
                    logger.info(f"Loaded config profile: {args.profile}")

                # Store metadata
                config.config_path = str(config_path)
                config.current_profile = args.profile

            except Exception as e:
                logger.warning(f"Failed to load config: {e}")

    # Parse server section from raw TOML (not in Config dataclass)
    if args.config and Path(args.config).exists():
        try:
            try:
                import tomllib
            except ImportError:
                import tomli as tomllib  # type: ignore[no-redef]

            with open(args.config, "rb") as f:
                raw_toml = tomllib.load(f)

            server_cfg = raw_toml.get("server", {})
            config.host = server_cfg.get("host", config.host)
            config.port = server_cfg.get("port", config.port)

            # Parse unified [quantization] section into quant sub-config
            quant_cfg = raw_toml.get("quantization", {})
            if quant_cfg:
                from .config import ComponentQuantConfig, PipelineQuantConfig
                g = quant_cfg.get("granularity", "per-tensor")
                config.quant = PipelineQuantConfig(
                    encoder=ComponentQuantConfig(
                        method=quant_cfg.get("encoder", "none"), granularity=g),
                    transformer=ComponentQuantConfig(
                        method=quant_cfg.get("transformer", "fp8-weight-only"), granularity=g),
                    vae=ComponentQuantConfig(
                        method=quant_cfg.get("vae", "none"), granularity=g),
                )

        except Exception as e:
            logger.debug(f"Could not parse raw TOML sections: {e}")

    # Apply CLI overrides
    config = _apply_cli_overrides(args, config)

    return config


def setup_logging(config: RuntimeConfig) -> None:
    """Configure logging based on runtime config.

    Log level priority (highest wins):
      --debug     -> DEBUG (all project modules)
      config.toml -> [logging] log_level
      default     -> INFO

    Uses structured JSON file logging with rotation when log_dir is set.
    Console output remains human-readable.
    """
    from llm_dit.utils.logging_config import setup_logging as configure_logging

    # Resolve log level: --debug overrides config.toml
    if config.debug:
        level = logging.DEBUG
    else:
        level = getattr(logging, config.logging.log_level.upper(), logging.INFO)

    # File logging: enabled if log_dir is set AND logging.enabled is true
    log_cfg = config.logging
    log_dir = log_cfg.log_dir
    enable_json = log_cfg.enabled and bool(log_dir)

    configure_logging(
        level=level,
        log_dir=Path(log_dir) if enable_json else None,
        enable_json_file=enable_json,
        max_bytes=log_cfg.max_bytes,
        backup_count=log_cfg.backup_count,
    )

    if config.debug:
        # Enable debug for all project modules
        logging.getLogger("llm_dit").setLevel(logging.DEBUG)
        logging.getLogger("llm_dit.backends").setLevel(logging.DEBUG)
        logging.getLogger("llm_dit.pipelines").setLevel(logging.DEBUG)
        logging.getLogger("web").setLevel(logging.DEBUG)
        logging.getLogger("__main__").setLevel(logging.DEBUG)
