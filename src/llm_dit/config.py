"""
TOML-based configuration for llm-dit-experiments.

Supports profiles for different hardware configurations, including:
- 8-bit quantization for LLM text encoder
- CPU offloading for memory-constrained systems
- Device selection (cuda, mps, cpu)

Example config (config.toml):

    [default]
    model_path = "/path/to/z-image"
    templates_dir = "templates/z_image"
    torch_dtype = "bfloat16"

    [default.encoder]
    device = "cuda"
    load_in_8bit = false
    cpu_offload = false

    [default.pipeline]
    device = "cuda"

    [low_vram]
    model_path = "/path/to/z-image"

    [low_vram.encoder]
    device = "cpu"
    load_in_8bit = true
    cpu_offload = true

    [low_vram.pipeline]
    device = "cuda"
"""

import logging
import os
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
    """Configuration for the text encoder (LLM)."""

    device: str = "auto"  # auto, cuda, mps, cpu
    torch_dtype: str = "bfloat16"  # bfloat16, float16, float32
    load_in_8bit: bool = False  # Use bitsandbytes 8-bit quantization
    load_in_4bit: bool = False  # Use bitsandbytes 4-bit quantization
    cpu_offload: bool = False  # Offload to CPU after encoding
    trust_remote_code: bool = True
    max_length: int = 512

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
class Config:
    """Complete configuration for Z-Image generation."""

    model_path: str = ""
    templates_dir: str | None = None

    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """Create config from dictionary."""
        encoder_data = data.pop("encoder", {})
        pipeline_data = data.pop("pipeline", {})
        generation_data = data.pop("generation", {})

        return cls(
            model_path=data.get("model_path", ""),
            templates_dir=data.get("templates_dir"),
            encoder=EncoderConfig(**encoder_data),
            pipeline=PipelineConfig(**pipeline_data),
            generation=GenerationConfig(**generation_data),
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
            load_in_8bit = true

            [low_vram]
            model_path = "/path/to/model"

            [low_vram.encoder]
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
                "load_in_8bit": self.encoder.load_in_8bit,
                "load_in_4bit": self.encoder.load_in_4bit,
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
            load_in_8bit=True,
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
