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
class RewriterConfig:
    """Configuration for prompt rewriting using LLM generation.

    The rewriter can use either the local model or a remote API backend
    for text generation. When use_api is True and api_url is set,
    the rewriter will use the API backend instead of the local model.
    """

    # Whether to use API backend for rewriting (default: use local model)
    use_api: bool = False
    # API backend settings (only used when use_api=True)
    api_url: str = ""  # URL for heylookitsanllm API
    api_model: str = "Qwen3-4B"  # Model ID for API backend
    # Generation parameters
    temperature: float = 1.0  # Sampling temperature (default 1.0)
    top_p: float = 0.95  # Nucleus sampling threshold
    max_tokens: int = 512  # Maximum tokens to generate


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
    rewriter: RewriterConfig = field(default_factory=RewriterConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """Create config from dictionary."""
        encoder_data = data.pop("encoder", {})
        pipeline_data = data.pop("pipeline", {})
        generation_data = data.pop("generation", {})
        optimization_data = data.pop("optimization", {})
        scheduler_data = data.pop("scheduler", {})
        lora_data = data.pop("lora", {})
        rewriter_data = data.pop("rewriter", {})

        return cls(
            model_path=data.get("model_path", ""),
            templates_dir=data.get("templates_dir"),
            encoder=EncoderConfig(**encoder_data),
            pipeline=PipelineConfig(**pipeline_data),
            generation=GenerationConfig(**generation_data),
            optimization=OptimizationConfig(**optimization_data),
            scheduler=SchedulerConfig(**scheduler_data),
            lora=LoRAConfig(**lora_data),
            rewriter=RewriterConfig(**rewriter_data),
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
            "rewriter": {
                "use_api": self.rewriter.use_api,
                "api_url": self.rewriter.api_url,
                "api_model": self.rewriter.api_model,
                "temperature": self.rewriter.temperature,
                "top_p": self.rewriter.top_p,
                "max_tokens": self.rewriter.max_tokens,
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
