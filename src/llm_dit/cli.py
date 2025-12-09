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
from typing import Literal

import torch

from .config import Config

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
    if ':' in lora_str:
        # Find the last colon (in case path contains colons on Windows)
        last_colon = lora_str.rfind(':')
        path = lora_str[:last_colon]
        try:
            scale = float(lora_str[last_colon + 1:])
        except ValueError:
            # Not a valid float, treat entire string as path
            path = lora_str
            scale = 1.0
    else:
        path = lora_str
        scale = 1.0
    return path, scale


@dataclass
class RuntimeConfig:
    """
    Unified runtime configuration combining TOML config + CLI overrides.

    This is the single source of truth used by both web server and CLI scripts.
    """

    # Model paths
    model_path: str = ""
    text_encoder_path: str | None = None  # If None, uses model_path/text_encoder/
    templates_dir: str | None = None

    # Device placement
    encoder_device: str = "auto"
    dit_device: str = "auto"
    vae_device: str = "auto"

    # Precision
    torch_dtype: str = "bfloat16"

    # Optimization flags
    cpu_offload: bool = False
    flash_attn: bool = False
    compile: bool = False

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
    long_prompt_mode: str = "truncate"  # truncate, interpolate, pool, attention_pool

    # Encoder settings
    hidden_layer: int = -2  # Which layer to extract embeddings from (-1=last, -2=penultimate)

    # Scheduler
    shift: float = 3.0

    # Generation defaults
    height: int = 1024
    width: int = 1024
    steps: int = 9
    guidance_scale: float = 0.0
    seed: int | None = None  # Random seed for reproducibility
    negative_prompt: str | None = None  # Negative prompt for CFG
    enable_thinking: bool = False
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

    # Rewriter settings
    rewriter_use_api: bool = False  # Use API backend for rewriting
    rewriter_api_url: str = ""  # API URL for rewriter (if different from encoder)
    rewriter_api_model: str = "Qwen3-4B"  # Model ID for rewriter API
    rewriter_temperature: float = 1.0  # Sampling temperature
    rewriter_top_p: float = 0.95  # Nucleus sampling threshold
    rewriter_min_p: float = 0.0  # Minimum probability threshold (0.0 = disabled)
    rewriter_max_tokens: int = 512  # Maximum tokens to generate

    # Debug
    debug: bool = False
    verbose: bool = False

    def get_torch_dtype(self) -> torch.dtype:
        """Convert string dtype to torch.dtype."""
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.torch_dtype, torch.bfloat16)

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
        default=None,
        help="Path to TOML config file",
    )
    config_group.add_argument(
        "--profile",
        type=str,
        default="default",
        help="Config profile to use (default: default)",
    )

    # Model paths
    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to Z-Image model (DiT + VAE) or HuggingFace ID",
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
            "How to handle prompts exceeding 1024 tokens: "
            "truncate (default, cut off), "
            "interpolate (resample embeddings), "
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

    # Scheduler
    sched_group = parser.add_argument_group("Scheduler")
    sched_group.add_argument(
        "--shift",
        type=float,
        default=None,
        help="Scheduler shift parameter (default: 3.0)",
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
        help="Sampling temperature for rewriter (default: 1.0)",
    )
    rewriter_group.add_argument(
        "--rewriter-top-p",
        type=float,
        default=None,
        help="Nucleus sampling threshold for rewriter (default: 0.95)",
    )
    rewriter_group.add_argument(
        "--rewriter-min-p",
        type=float,
        default=None,
        help="Minimum probability threshold for rewriter (default: 0.0, disabled)",
    )
    rewriter_group.add_argument(
        "--rewriter-max-tokens",
        type=int,
        default=None,
        help="Maximum tokens to generate for rewriter (default: 512)",
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
            "--seed",
            type=int,
            default=None,
            help="Random seed for reproducibility",
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

    return parser


def load_runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    """
    Load runtime configuration from TOML file + CLI overrides.

    Priority (highest to lowest):
    1. CLI arguments
    2. TOML config file
    3. Defaults

    Args:
        args: Parsed CLI arguments

    Returns:
        RuntimeConfig with all settings resolved
    """
    # Start with defaults
    config = RuntimeConfig()

    # Load TOML config if provided
    toml_config: Config | None = None
    if args.config:
        try:
            toml_config = Config.from_toml(args.config, args.profile)
            logger.info(f"Loaded config profile: {args.profile}")

            # Apply TOML values to runtime config
            config.model_path = toml_config.model_path or config.model_path
            config.templates_dir = toml_config.templates_dir or config.templates_dir
            config.encoder_device = toml_config.encoder.device
            config.torch_dtype = toml_config.encoder.torch_dtype
            config.hidden_layer = toml_config.encoder.hidden_layer

            # Generation defaults from config
            config.height = toml_config.generation.height
            config.width = toml_config.generation.width
            config.steps = toml_config.generation.num_inference_steps
            config.guidance_scale = toml_config.generation.guidance_scale
            config.enable_thinking = toml_config.generation.enable_thinking
            config.default_template = toml_config.generation.default_template

            # Pipeline settings
            config.cpu_offload = toml_config.pipeline.enable_model_cpu_offload

            # Check for optimization section
            if hasattr(toml_config, 'optimization'):
                opt = toml_config.optimization
                config.flash_attn = getattr(opt, 'flash_attn', False)
                config.compile = getattr(opt, 'compile', False)
                config.cpu_offload = getattr(opt, 'cpu_offload', config.cpu_offload)

            # Check for scheduler section
            if hasattr(toml_config, 'scheduler'):
                sched = toml_config.scheduler
                config.shift = getattr(sched, 'shift', 3.0)

            # Check for LoRA section
            if hasattr(toml_config, 'lora'):
                lora = toml_config.lora
                config.lora_paths = getattr(lora, 'paths', [])
                config.lora_scales = getattr(lora, 'scales', [])

            # Check for PyTorch-native section
            if hasattr(toml_config, 'pytorch'):
                pytorch = toml_config.pytorch
                config.attention_backend = getattr(pytorch, 'attention_backend', None)
                config.use_custom_scheduler = getattr(pytorch, 'use_custom_scheduler', False)
                config.tiled_vae = getattr(pytorch, 'tiled_vae', False)
                config.tile_size = getattr(pytorch, 'tile_size', 512)
                config.tile_overlap = getattr(pytorch, 'tile_overlap', 64)
                config.embedding_cache = getattr(pytorch, 'embedding_cache', False)
                config.cache_size = getattr(pytorch, 'cache_size', 100)
                config.long_prompt_mode = getattr(pytorch, 'long_prompt_mode', 'truncate')

            # Check for rewriter section
            if hasattr(toml_config, 'rewriter'):
                rewriter = toml_config.rewriter
                config.rewriter_use_api = getattr(rewriter, 'use_api', False)
                config.rewriter_api_url = getattr(rewriter, 'api_url', '')
                config.rewriter_api_model = getattr(rewriter, 'api_model', 'Qwen3-4B')
                config.rewriter_temperature = getattr(rewriter, 'temperature', 1.0)
                config.rewriter_top_p = getattr(rewriter, 'top_p', 0.95)
                config.rewriter_min_p = getattr(rewriter, 'min_p', 0.0)
                config.rewriter_max_tokens = getattr(rewriter, 'max_tokens', 512)

        except Exception as e:
            logger.warning(f"Failed to load config: {e}")

    # Also check for server section in TOML
    if args.config:
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib

        with open(args.config, "rb") as f:
            raw_toml = tomllib.load(f)
        server_cfg = raw_toml.get("server", {})
        config.host = server_cfg.get("host", config.host)
        config.port = server_cfg.get("port", config.port)

    # Apply CLI overrides (only if explicitly provided)
    if args.model_path is not None:
        config.model_path = args.model_path
    if getattr(args, 'text_encoder_path', None) is not None:
        config.text_encoder_path = args.text_encoder_path
    if args.templates_dir is not None:
        config.templates_dir = args.templates_dir

    # Device overrides
    if getattr(args, 'text_encoder_device', None) is not None:
        config.encoder_device = args.text_encoder_device
    if getattr(args, 'dit_device', None) is not None:
        config.dit_device = args.dit_device
    if getattr(args, 'vae_device', None) is not None:
        config.vae_device = args.vae_device

    # Optimization overrides
    if args.cpu_offload:
        config.cpu_offload = True
    if args.flash_attn:
        config.flash_attn = True
    if args.compile:
        config.compile = True
    if getattr(args, 'torch_dtype', None) is not None:
        config.torch_dtype = args.torch_dtype

    # Scheduler overrides
    if getattr(args, 'shift', None) is not None:
        config.shift = args.shift

    # PyTorch-native component overrides
    if getattr(args, 'attention_backend', None) is not None:
        config.attention_backend = args.attention_backend
    if getattr(args, 'use_custom_scheduler', False):
        config.use_custom_scheduler = True
    if getattr(args, 'tiled_vae', False):
        config.tiled_vae = True
    if getattr(args, 'tile_size', None) is not None:
        config.tile_size = args.tile_size
    if getattr(args, 'tile_overlap', None) is not None:
        config.tile_overlap = args.tile_overlap
    if getattr(args, 'embedding_cache', False):
        config.embedding_cache = True
    if getattr(args, 'cache_size', None) is not None:
        config.cache_size = args.cache_size
    if getattr(args, 'long_prompt_mode', None) is not None:
        config.long_prompt_mode = args.long_prompt_mode
    if getattr(args, 'hidden_layer', None) is not None:
        config.hidden_layer = args.hidden_layer

    # LoRA overrides
    if getattr(args, 'loras', None):
        config.lora_paths = []
        config.lora_scales = []
        for lora_str in args.loras:
            path, scale = parse_lora_arg(lora_str)
            config.lora_paths.append(path)
            config.lora_scales.append(scale)

    # Prompt control overrides
    if getattr(args, 'template', None) is not None:
        config.default_template = args.template
    if getattr(args, 'system_prompt', None) is not None:
        config.system_prompt = args.system_prompt
    if getattr(args, 'thinking_content', None) is not None:
        config.thinking_content = args.thinking_content
    if getattr(args, 'assistant_content', None) is not None:
        config.assistant_content = args.assistant_content
    if getattr(args, 'enable_thinking', False):
        config.enable_thinking = True

    # API backend overrides
    if getattr(args, 'api_url', None) is not None:
        config.api_url = args.api_url
    if getattr(args, 'api_model', None) is not None:
        config.api_model = args.api_model
    if getattr(args, 'local_encoder', False):
        config.local_encoder = True

    # Rewriter overrides
    if getattr(args, 'rewriter_use_api', False):
        config.rewriter_use_api = True
    if getattr(args, 'rewriter_api_url', None) is not None:
        config.rewriter_api_url = args.rewriter_api_url
    if getattr(args, 'rewriter_api_model', None) is not None:
        config.rewriter_api_model = args.rewriter_api_model
    if getattr(args, 'rewriter_temperature', None) is not None:
        config.rewriter_temperature = args.rewriter_temperature
    if getattr(args, 'rewriter_top_p', None) is not None:
        config.rewriter_top_p = args.rewriter_top_p
    if getattr(args, 'rewriter_min_p', None) is not None:
        config.rewriter_min_p = args.rewriter_min_p
    if getattr(args, 'rewriter_max_tokens', None) is not None:
        config.rewriter_max_tokens = args.rewriter_max_tokens

    # Generation overrides
    if getattr(args, 'height', None) is not None:
        config.height = args.height
    if getattr(args, 'width', None) is not None:
        config.width = args.width
    if getattr(args, 'steps', None) is not None:
        config.steps = args.steps
    if getattr(args, 'guidance_scale', None) is not None:
        config.guidance_scale = args.guidance_scale
    if getattr(args, 'seed', None) is not None:
        config.seed = args.seed
    if getattr(args, 'negative_prompt', None) is not None:
        config.negative_prompt = args.negative_prompt

    # Server overrides
    if getattr(args, 'host', None) is not None:
        config.host = args.host
    if getattr(args, 'port', None) is not None:
        config.port = args.port

    # Debug overrides
    if args.debug:
        config.debug = True
    if getattr(args, 'verbose', False):
        config.verbose = True

    return config


def setup_logging(config: RuntimeConfig) -> None:
    """Configure logging based on runtime config."""
    level = logging.DEBUG if config.debug or config.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    if config.debug:
        # Enable debug for llm_dit modules
        logging.getLogger("llm_dit").setLevel(logging.DEBUG)
        logging.getLogger("llm_dit.backends").setLevel(logging.DEBUG)
        logging.getLogger("llm_dit.pipelines").setLevel(logging.DEBUG)
