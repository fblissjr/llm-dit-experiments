"""
TOML-based configuration for llm-dit-experiments.

Supports profiles for different hardware configurations, including:
- Quantization via BitsAndBytesConfig (4-bit or 8-bit)
- CPU offloading for memory-constrained systems
- Device selection (cuda, mps, cpu)

Transformers v5 Migration:
- load_in_8bit/load_in_4bit are DEPRECATED in transformers v5
- Use quantization="8bit" or quantization="4bit" instead
- The config automatically builds BitsAndBytesConfig internally

Example config (config.toml):

    [default]
    model_path = "/path/to/z-image"
    templates_dir = "templates/z_image"
    torch_dtype = "bfloat16"

    [default.encoder]
    device = "cuda"
    quantization = "none"
    cpu_offload = false

    [default.pipeline]
    device = "cuda"

    [low_vram]
    model_path = "/path/to/z-image"

    [low_vram.encoder]
    device = "cpu"
    quantization = "8bit"
    cpu_offload = true

    [low_vram.pipeline]
    device = "cuda"
"""

import logging
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch

logger = logging.getLogger(__name__)

# Try to import tomllib (Python 3.11+) or tomli
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None


@dataclass
class EncoderConfig:
    """Configuration for the text encoder (LLM).

    Transformers v5 Migration Notes:
    - load_in_8bit/load_in_4bit are DEPRECATED
    - Use quantization="8bit" or quantization="4bit" instead
    - Config will auto-migrate legacy fields with a deprecation warning
    """

    device: str = "auto"  # auto, cuda, mps, cpu
    torch_dtype: str = "bfloat16"  # bfloat16, float16, float32
    quantization: str = "none"  # none, 4bit, 8bit (v5 API)
    cpu_offload: bool = False  # Offload to CPU after encoding
    trust_remote_code: bool = True
    max_length: int = 512
    hidden_layer: int = -2  # Which layer to extract embeddings from (-1=last, -2=penultimate)

    # DEPRECATED: These fields are kept for backwards compatibility only
    # They will be removed in a future version
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    def __post_init__(self):
        """Handle deprecation migration from load_in_8bit/load_in_4bit to quantization."""
        # Migrate legacy fields if used
        if self.load_in_8bit and self.quantization == "none":
            warnings.warn(
                "load_in_8bit is deprecated in transformers v5. "
                "Use quantization='8bit' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.quantization = "8bit"
        elif self.load_in_4bit and self.quantization == "none":
            warnings.warn(
                "load_in_4bit is deprecated in transformers v5. "
                "Use quantization='4bit' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.quantization = "4bit"

    def get_torch_dtype(self) -> torch.dtype:
        """Convert string dtype to torch.dtype."""
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.torch_dtype, torch.bfloat16)

    def get_device(self) -> str:
        """Get resolved device string."""
        if self.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return self.device

    def get_quantization_config(self) -> "BitsAndBytesConfig | None":
        """Get BitsAndBytesConfig for transformers v5.

        Returns:
            BitsAndBytesConfig if quantization is enabled, None otherwise.

        Note:
            This is the v5-compliant way to configure quantization.
            The config should be passed to from_pretrained() as:

                model = AutoModel.from_pretrained(
                    model_path,
                    quantization_config=config.encoder.get_quantization_config(),
                )
        """
        if self.quantization == "none":
            return None

        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            raise ImportError(
                "BitsAndBytesConfig requires transformers>=4.30.0. "
                "Install with: pip install transformers>=4.30.0"
            )

        if self.quantization == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True)
        elif self.quantization == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.get_torch_dtype(),
            )
        else:
            raise ValueError(
                f"Unknown quantization: {self.quantization}. "
                f"Valid options: none, 4bit, 8bit"
            )


@dataclass
class PipelineConfig:
    """Configuration for the diffusers pipeline (transformer + VAE)."""

    device: str = "auto"  # auto, cuda, mps, cpu
    torch_dtype: str = "bfloat16"
    enable_model_cpu_offload: bool = False  # Sequential CPU offload
    enable_sequential_cpu_offload: bool = False  # More aggressive offload

    def get_torch_dtype(self) -> torch.dtype:
        """Convert string dtype to torch.dtype."""
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.torch_dtype, torch.bfloat16)

    def get_device(self) -> str:
        """Get resolved device string."""
        if self.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return self.device


@dataclass
class GenerationConfig:
    """Default generation parameters."""

    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 9
    guidance_scale: float = 0.0
    enable_thinking: bool = True
    default_template: str | None = None


@dataclass
class OptimizationConfig:
    """Optimization settings for pipeline execution."""

    flash_attn: bool = False  # Enable Flash Attention
    compile: bool = False  # Enable torch.compile
    cpu_offload: bool = False  # Enable CPU offload for transformer


@dataclass
class SchedulerConfig:
    """Scheduler settings."""

    shift: float = 3.0  # Flow matching scheduler shift parameter


@dataclass
class LoRAConfig:
    """LoRA configuration."""

    paths: list[str] = field(default_factory=list)  # Paths to LoRA files
    scales: list[float] = field(default_factory=list)  # Scale for each LoRA


@dataclass
class PyTorchConfig:
    """PyTorch-native component configuration.

    These settings control the Phase 1 migration components that reduce
    diffusers dependency and optimize for RTX 4090.
    """

    attention_backend: str = "auto"  # auto, flash_attn_2, flash_attn_3, sage, xformers, sdpa
    use_custom_scheduler: bool = False  # Use pure PyTorch FlowMatchScheduler
    tiled_vae: bool = False  # Enable tiled VAE decode for 2K+ images
    tile_size: int = 512  # Tile size in pixels (latent = tile_size / 8)
    tile_overlap: int = 64  # Overlap between tiles for smooth blending
    embedding_cache: bool = False  # Cache embeddings for repeated prompts
    cache_size: int = 100  # Max cached embeddings (LRU eviction)
    long_prompt_mode: str = "interpolate"  # truncate, interpolate, pool, attention_pool


@dataclass
class VLConfig:
    """Configuration for Qwen3-VL vision conditioning.

    This enables zero-shot vision conditioning by extracting embeddings from
    Qwen3-VL and blending them with text embeddings.

    Key insight: Qwen3-VL-4B's text model shares architecture with Qwen3-4B
    (hidden_size=2560), enabling direct embedding transfer without training.
    """

    model_path: str = ""  # Path to Qwen3-VL model (empty = disabled)
    device: str = "cpu"  # Device for Qwen3-VL (cpu recommended to save VRAM)
    default_alpha: float = 1.0  # Default interpolation ratio (0.0=text, 1.0=VL) - use 1.0 for pure VL
    default_hidden_layer: int = -8  # Layer -8 produces cleaner results than -2 (penultimate)
    text_tokens_only: bool = True  # Use only text token positions (image tokens cause artifacts)
    auto_unload: bool = True  # Unload after extraction to save VRAM
    target_std: float = 70.0  # Target std for scaling (measured from Qwen3-4B text embeddings)


@dataclass
class RewriterConfig:
    """Configuration for prompt rewriting using LLM generation.

    The rewriter can use either the local model or a remote API backend
    for text generation. When use_api is True and api_url is set,
    the rewriter will use the API backend instead of the local model.

    VL Rewriting:
    - When vl_enabled is True and vl.model_path is configured, users can
      select Qwen3-VL for vision-enabled prompt rewriting in the web UI.
    - VL model is loaded on-demand when first selected (unless preload_vl is True).
    - Supports image-only, text-only, or combined image+text rewriting.

    Qwen3 Best Practices (thinking mode):
    - temperature=0.6, top_p=0.95, top_k=20, min_p=0 (default)
    - DO NOT use greedy decoding (causes repetition)
    - presence_penalty=0-2 helps reduce endless repetitions
    See: https://huggingface.co/Qwen/Qwen3-4B#best-practices
    """

    # Whether to use API backend for rewriting (default: use local model)
    use_api: bool = False
    # API backend settings (only used when use_api=True)
    api_url: str = ""  # URL for heylookitsanllm API (falls back to --api-url if empty)
    api_model: str = "Qwen3-4B"  # Model ID for API backend
    # Generation parameters (Qwen3 thinking mode defaults)
    temperature: float = 0.6  # Qwen3 thinking mode: 0.6 (NOT greedy!)
    top_p: float = 0.95  # Qwen3 thinking mode: 0.95
    top_k: int = 20  # Qwen3 thinking mode: 20
    min_p: float = 0.0  # Qwen3: 0.0 (disabled)
    presence_penalty: float = 0.0  # 0-2, helps reduce endless repetitions
    max_tokens: int = 512  # Maximum tokens to generate
    # VL rewriter settings
    vl_enabled: bool = True  # Allow VL model selection in rewriter UI
    preload_vl: bool = False  # Load Qwen3-VL at startup for rewriter (uses vl.model_path)
    vl_api_model: str = ""  # Model ID for VL via API (e.g., "qwen2.5-vl-72b-mlx"). Empty = use local VL
    # API timeout settings
    timeout: float = 120.0  # API request timeout in seconds (VL models need longer)


@dataclass
class Config:
    """Complete configuration for Z-Image generation."""

    model_path: str = ""
    templates_dir: str | None = None

    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    pytorch: PyTorchConfig = field(default_factory=PyTorchConfig)
    rewriter: RewriterConfig = field(default_factory=RewriterConfig)
    vl: VLConfig = field(default_factory=VLConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """Create config from dictionary."""
        encoder_data = data.pop("encoder", {})
        pipeline_data = data.pop("pipeline", {})
        generation_data = data.pop("generation", {})
        optimization_data = data.pop("optimization", {})
        scheduler_data = data.pop("scheduler", {})
        lora_data = data.pop("lora", {})
        pytorch_data = data.pop("pytorch", {})
        rewriter_data = data.pop("rewriter", {})
        vl_data = data.pop("vl", {})

        return cls(
            model_path=data.get("model_path", ""),
            templates_dir=data.get("templates_dir"),
            encoder=EncoderConfig(**encoder_data),
            pipeline=PipelineConfig(**pipeline_data),
            generation=GenerationConfig(**generation_data),
            optimization=OptimizationConfig(**optimization_data),
            scheduler=SchedulerConfig(**scheduler_data),
            lora=LoRAConfig(**lora_data),
            pytorch=PyTorchConfig(**pytorch_data),
            rewriter=RewriterConfig(**rewriter_data),
            vl=VLConfig(**vl_data),
        )

    @classmethod
    def from_toml(cls, path: str | Path, profile: str = "default") -> "Config":
        """
        Load config from TOML file.

        Args:
            path: Path to TOML config file
            profile: Profile name to load (default: "default")

        Returns:
            Loaded Config

        Example TOML:
            [default]
            model_path = "/path/to/model"

            [default.encoder]
            quantization = "8bit"

            [low_vram]
            model_path = "/path/to/model"

            [low_vram.encoder]
            quantization = "8bit"
            cpu_offload = true
        """
        if tomllib is None:
            raise ImportError(
                "tomllib/tomli required for TOML config. "
                "Install with: pip install tomli (Python <3.11)"
            )

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "rb") as f:
            data = tomllib.load(f)

        if profile not in data:
            available = list(data.keys())
            raise KeyError(
                f"Profile '{profile}' not found in config. "
                f"Available: {available}"
            )

        profile_data = data[profile]
        logger.info(f"Loaded config profile: {profile}")
        return cls.from_dict(profile_data)

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to dictionary."""
        return {
            "model_path": self.model_path,
            "templates_dir": self.templates_dir,
            "encoder": {
                "device": self.encoder.device,
                "torch_dtype": self.encoder.torch_dtype,
                "quantization": self.encoder.quantization,
                "cpu_offload": self.encoder.cpu_offload,
                "trust_remote_code": self.encoder.trust_remote_code,
                "max_length": self.encoder.max_length,
                "hidden_layer": self.encoder.hidden_layer,
            },
            "pipeline": {
                "device": self.pipeline.device,
                "torch_dtype": self.pipeline.torch_dtype,
                "enable_model_cpu_offload": self.pipeline.enable_model_cpu_offload,
                "enable_sequential_cpu_offload": self.pipeline.enable_sequential_cpu_offload,
            },
            "generation": {
                "height": self.generation.height,
                "width": self.generation.width,
                "num_inference_steps": self.generation.num_inference_steps,
                "guidance_scale": self.generation.guidance_scale,
                "enable_thinking": self.generation.enable_thinking,
                "default_template": self.generation.default_template,
            },
            "optimization": {
                "flash_attn": self.optimization.flash_attn,
                "compile": self.optimization.compile,
                "cpu_offload": self.optimization.cpu_offload,
            },
            "scheduler": {
                "shift": self.scheduler.shift,
            },
            "lora": {
                "paths": self.lora.paths,
                "scales": self.lora.scales,
            },
            "pytorch": {
                "attention_backend": self.pytorch.attention_backend,
                "use_custom_scheduler": self.pytorch.use_custom_scheduler,
                "tiled_vae": self.pytorch.tiled_vae,
                "tile_size": self.pytorch.tile_size,
                "tile_overlap": self.pytorch.tile_overlap,
                "embedding_cache": self.pytorch.embedding_cache,
                "cache_size": self.pytorch.cache_size,
                "long_prompt_mode": self.pytorch.long_prompt_mode,
            },
            "rewriter": {
                "use_api": self.rewriter.use_api,
                "api_url": self.rewriter.api_url,
                "api_model": self.rewriter.api_model,
                "temperature": self.rewriter.temperature,
                "top_p": self.rewriter.top_p,
                "top_k": self.rewriter.top_k,
                "min_p": self.rewriter.min_p,
                "presence_penalty": self.rewriter.presence_penalty,
                "max_tokens": self.rewriter.max_tokens,
                "vl_enabled": self.rewriter.vl_enabled,
                "preload_vl": self.rewriter.preload_vl,
            },
            "vl": {
                "model_path": self.vl.model_path,
                "device": self.vl.device,
                "default_alpha": self.vl.default_alpha,
                "default_hidden_layer": self.vl.default_hidden_layer,
                "auto_unload": self.vl.auto_unload,
                "target_std": self.vl.target_std,
            },
        }


# Preset configurations
PRESETS = {
    "default": Config(
        encoder=EncoderConfig(device="auto", torch_dtype="bfloat16"),
        pipeline=PipelineConfig(device="auto", torch_dtype="bfloat16"),
    ),
    "low_vram": Config(
        encoder=EncoderConfig(
            device="cuda",
            torch_dtype="bfloat16",
            quantization="8bit",  # v5 API
            cpu_offload=True,
        ),
        pipeline=PipelineConfig(
            device="cuda",
            torch_dtype="bfloat16",
            enable_model_cpu_offload=True,
        ),
    ),
    "cpu_only": Config(
        encoder=EncoderConfig(device="cpu", torch_dtype="float32"),
        pipeline=PipelineConfig(device="cpu", torch_dtype="float32"),
    ),
}


def get_preset(name: str) -> Config:
    """Get a preset configuration by name."""
    if name not in PRESETS:
        raise KeyError(f"Unknown preset: {name}. Available: {list(PRESETS.keys())}")
    return PRESETS[name]


def load_config(
    path: str | Path | None = None,
    profile: str = "default",
    preset: str | None = None,
) -> Config:
    """
    Load configuration from file or preset.

    Priority:
    1. If path is provided, load from TOML file
    2. If preset is provided, use preset
    3. Otherwise, use default config

    Args:
        path: Optional path to TOML config file
        profile: Profile name within TOML file
        preset: Preset name ("default", "low_vram", "cpu_only")

    Returns:
        Loaded Config
    """
    if path is not None:
        return Config.from_toml(path, profile)
    elif preset is not None:
        return get_preset(preset)
    else:
        return Config()
