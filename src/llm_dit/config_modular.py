"""
Modular configuration system for llm-dit-experiments.

Provides a component-based configuration that eliminates duplication by:
- Defining components once in a library (encoders, transformers, VAEs, schedulers)
- Referencing components by ID in model type definitions
- Supporting inheritance between model types
- Separating hardware profiles (WHERE to run) from model configs (WHAT to run)

Usage:
    from llm_dit.config_modular import ModularConfig

    # Load modular config
    config = ModularConfig.from_toml("config.toml", "rtx4090_zimage")

    # Convert to RuntimeConfig for backward compatibility
    runtime_config = config.to_runtime_config()

last updated: 2025-12-27
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

import torch

try:
    import tomllib
except ImportError:
    import tomli as tomllib


# =============================================================================
# Device Placement
# =============================================================================


@dataclass
class DevicePlacement:
    """Device placement for pipeline components."""

    encoder: str = "auto"
    dit: str = "auto"
    vae: str = "auto"
    siglip: str = "cpu"  # For FMTT reward model

    def resolve(self, device_str: str) -> str:
        """Resolve 'auto' to actual device."""
        if device_str == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device_str

    def get_encoder_device(self) -> str:
        return self.resolve(self.encoder)

    def get_dit_device(self) -> str:
        return self.resolve(self.dit)

    def get_vae_device(self) -> str:
        return self.resolve(self.vae)


# =============================================================================
# Component Definitions
# =============================================================================


@dataclass
class EncoderDefinition:
    """Definition of a text encoder component."""

    description: str = ""
    model_id: str = ""  # HuggingFace model ID
    hidden_dim: int = 2560
    num_layers: int = 36
    default_hidden_layer: int = -2
    max_tokens: int = 2048
    quantization_options: list[str] = field(
        default_factory=lambda: ["none", "4bit", "8bit", "fp8", "int8"]
    )


@dataclass
class TransformerDefinition:
    """Definition of a DiT/transformer component."""

    description: str = ""
    num_layers: int = 30
    latent_channels: int = 16
    quantization_options: list[str] = field(
        default_factory=lambda: ["none", "4bit", "8bit", "fp8", "int8"]
    )


@dataclass
class VAEDefinition:
    """Definition of a VAE component."""

    description: str = ""
    latent_scaling: float = 0.13025
    latent_channels: int = 4


@dataclass
class SchedulerDefinition:
    """Definition of a scheduler component."""

    description: str = ""
    type: str = "flow_matching"  # flow_matching, euler, ddpm
    shift: Optional[float] = None
    shift_mode: Optional[str] = None  # "dynamic" or fixed value


# =============================================================================
# Hardware Profile
# =============================================================================


@dataclass
class HardwareProfile:
    """Hardware-specific settings (device placement + optimizations)."""

    description: str = ""
    devices: DevicePlacement = field(default_factory=DevicePlacement)
    dtype: str = "bfloat16"

    # Optimization flags
    attention_backend: str = "auto"
    compile: bool = False
    tiled_vae: bool = False
    tile_size: int = 512
    tile_overlap: int = 64

    # Embedding cache
    embedding_cache: bool = False
    cache_size: int = 100
    long_prompt_mode: str = "interpolate"

    # Custom scheduler
    use_custom_scheduler: bool = True

    def get_torch_dtype(self) -> torch.dtype:
        """Convert string dtype to torch.dtype."""
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.dtype, torch.bfloat16)


# =============================================================================
# Model Type Configuration
# =============================================================================


@dataclass
class ModelTypeConfig:
    """Configuration for a specific model type."""

    description: str = ""
    pipeline_class: str = ""  # e.g., "ZImagePipeline", "QwenImageDiffusersPipeline"

    # Component references (IDs that look up in component library)
    text_encoder: str = ""
    dit: str = ""
    vae: str = ""
    scheduler: str = ""

    # Model-specific paths
    model_path: str = ""
    edit_model_path: str = ""  # For Qwen-Image-Edit models
    templates_dir: str = ""

    # Generation defaults
    default_steps: int = 9
    default_cfg_scale: float = 0.0
    default_resolution: int = 1024

    # Optional model-specific settings
    layer_num: Optional[int] = None  # For Qwen-Image decomposition
    cpu_offload: bool = True

    # Inheritance
    inherits: Optional[str] = None


# =============================================================================
# Quantization Preset
# =============================================================================


@dataclass
class QuantizationPreset:
    """Quantization settings for multiple components."""

    description: str = ""
    text_encoder: str = "none"  # none, 4bit, 8bit, fp8, int8
    transformer: str = "none"


# =============================================================================
# Feature Configurations
# =============================================================================


@dataclass
class DyPEConfig:
    """DyPE (Dynamic Position Extrapolation) configuration."""

    enabled: bool = False
    method: str = "vision_yarn"
    dype_scale: float = 2.0
    dype_exponent: float = 2.0
    dype_start_sigma: float = 1.0
    base_shift: float = 0.5
    max_shift: float = 1.15
    base_resolution: int = 1024
    anisotropic: bool = False


@dataclass
class SLGConfig:
    """Skip Layer Guidance configuration."""

    enabled: bool = False
    scale: float = 2.5
    layers: list[int] = field(default_factory=lambda: [7, 8, 9, 10, 11, 12])
    start: float = 0.05
    stop: float = 0.5


@dataclass
class FMTTConfig:
    """Flow Map Trajectory Tilting configuration."""

    enabled: bool = False
    guidance_scale: float = 1.0
    guidance_start: float = 0.0
    guidance_stop: float = 0.5
    normalize_mode: str = "unit"
    decode_scale: float = 0.5
    siglip_model: str = "google/siglip2-giant-opt-patch16-384"
    siglip_device: str = "cpu"


# =============================================================================
# Component Library
# =============================================================================


@dataclass
class ComponentLibrary:
    """Library of reusable component definitions."""

    encoders: dict[str, EncoderDefinition] = field(default_factory=dict)
    transformers: dict[str, TransformerDefinition] = field(default_factory=dict)
    vaes: dict[str, VAEDefinition] = field(default_factory=dict)
    schedulers: dict[str, SchedulerDefinition] = field(default_factory=dict)

    def get_encoder(self, encoder_id: str) -> EncoderDefinition:
        """Get encoder definition by ID."""
        if encoder_id not in self.encoders:
            available = list(self.encoders.keys())
            raise KeyError(
                f"Encoder '{encoder_id}' not found. Available: {available}"
            )
        return self.encoders[encoder_id]

    def get_transformer(self, transformer_id: str) -> TransformerDefinition:
        """Get transformer definition by ID."""
        if transformer_id not in self.transformers:
            available = list(self.transformers.keys())
            raise KeyError(
                f"Transformer '{transformer_id}' not found. Available: {available}"
            )
        return self.transformers[transformer_id]

    def get_vae(self, vae_id: str) -> VAEDefinition:
        """Get VAE definition by ID."""
        if vae_id not in self.vaes:
            available = list(self.vaes.keys())
            raise KeyError(f"VAE '{vae_id}' not found. Available: {available}")
        return self.vaes[vae_id]

    def get_scheduler(self, scheduler_id: str) -> SchedulerDefinition:
        """Get scheduler definition by ID."""
        if scheduler_id not in self.schedulers:
            available = list(self.schedulers.keys())
            raise KeyError(
                f"Scheduler '{scheduler_id}' not found. Available: {available}"
            )
        return self.schedulers[scheduler_id]


# =============================================================================
# Modular Config
# =============================================================================


@dataclass
class ModularConfig:
    """
    Top-level modular configuration.

    Combines:
    - Hardware profile (device placement, optimizations)
    - Model type (pipeline class, component references, defaults)
    - Component library (encoder, transformer, VAE, scheduler definitions)
    - Quantization preset
    - Feature configs (DyPE, SLG, FMTT)
    """

    # Core configuration
    profile: HardwareProfile = field(default_factory=HardwareProfile)
    model: ModelTypeConfig = field(default_factory=ModelTypeConfig)
    components: ComponentLibrary = field(default_factory=ComponentLibrary)
    quantization: QuantizationPreset = field(default_factory=QuantizationPreset)

    # Feature configurations
    dype: DyPEConfig = field(default_factory=DyPEConfig)
    slg: SLGConfig = field(default_factory=SLGConfig)
    fmtt: FMTTConfig = field(default_factory=FMTTConfig)

    # Server settings
    server_host: str = "127.0.0.1"
    server_port: int = 7860

    # Overrides from combined config
    overrides: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_toml(cls, path: Path | str, config_name: str) -> "ModularConfig":
        """
        Load modular config from TOML file.

        Args:
            path: Path to config.toml
            config_name: Name of combined config (e.g., "rtx4090_zimage")

        Returns:
            Fully resolved ModularConfig

        Raises:
            KeyError: If config_name or referenced components not found
            FileNotFoundError: If config file doesn't exist
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "rb") as f:
            data = tomllib.load(f)

        # Check if this is a modular config (has [configs] section)
        if "configs" not in data:
            raise ValueError(
                f"Not a modular config file (missing [configs] section). "
                f"Use legacy Config.from_toml() instead."
            )

        # Get combined config
        if config_name not in data["configs"]:
            available = list(data["configs"].keys())
            raise KeyError(
                f"Config '{config_name}' not found. Available: {available}"
            )

        combined = data["configs"][config_name]

        # Extract references
        profile_id = combined.get("profile", "default")
        model_id = combined.get("model", "zimage")
        quant_id = combined.get("quantization", "none")
        feature_ids = combined.get("features", [])
        overrides = combined.get("overrides", {})

        # Load hardware profile
        profile = cls._load_profile(data, profile_id)

        # Load model type (with inheritance)
        model = cls._load_model(data, model_id)

        # Load component library
        components = cls._load_components(data)

        # Load quantization preset
        quantization = cls._load_quantization(data, quant_id)

        # Load feature configs
        dype = cls._load_dype(data, "dype" in feature_ids)
        slg = cls._load_slg(data, "slg" in feature_ids)
        fmtt = cls._load_fmtt(data, "fmtt" in feature_ids)

        # Server settings
        server_data = data.get("server", {})

        return cls(
            profile=profile,
            model=model,
            components=components,
            quantization=quantization,
            dype=dype,
            slg=slg,
            fmtt=fmtt,
            server_host=server_data.get("host", "127.0.0.1"),
            server_port=server_data.get("port", 7860),
            overrides=overrides,
        )

    @classmethod
    def _load_profile(cls, data: dict, profile_id: str) -> HardwareProfile:
        """Load hardware profile from TOML data."""
        if "profiles" not in data or profile_id not in data["profiles"]:
            # Return default profile
            return HardwareProfile(description=f"Default profile ('{profile_id}' not found)")

        profile_data = data["profiles"][profile_id]

        # Parse devices
        devices_data = profile_data.get("devices", {})
        devices = DevicePlacement(
            encoder=devices_data.get("encoder", "auto"),
            dit=devices_data.get("dit", "auto"),
            vae=devices_data.get("vae", "auto"),
            siglip=devices_data.get("siglip", "cpu"),
        )

        return HardwareProfile(
            description=profile_data.get("description", ""),
            devices=devices,
            dtype=profile_data.get("dtype", "bfloat16"),
            attention_backend=profile_data.get("attention_backend", "auto"),
            compile=profile_data.get("compile", False),
            tiled_vae=profile_data.get("tiled_vae", False),
            tile_size=profile_data.get("tile_size", 512),
            tile_overlap=profile_data.get("tile_overlap", 64),
            embedding_cache=profile_data.get("embedding_cache", False),
            cache_size=profile_data.get("cache_size", 100),
            long_prompt_mode=profile_data.get("long_prompt_mode", "interpolate"),
            use_custom_scheduler=profile_data.get("use_custom_scheduler", True),
        )

    @classmethod
    def _load_model(cls, data: dict, model_id: str) -> ModelTypeConfig:
        """Load model type with inheritance resolution."""
        if "models" not in data or model_id not in data["models"]:
            raise KeyError(f"Model type '{model_id}' not found in [models] section")

        model_data = data["models"][model_id].copy()

        # Resolve inheritance
        if "inherits" in model_data:
            parent_id = model_data.pop("inherits")
            parent_data = cls._load_model_data_recursive(data, parent_id, set())
            # Merge: child overrides parent
            merged = {**parent_data, **model_data}
            model_data = merged

        return ModelTypeConfig(
            description=model_data.get("description", ""),
            pipeline_class=model_data.get("pipeline_class", ""),
            text_encoder=model_data.get("text_encoder", ""),
            dit=model_data.get("dit", ""),
            vae=model_data.get("vae", ""),
            scheduler=model_data.get("scheduler", ""),
            model_path=model_data.get("model_path", ""),
            edit_model_path=model_data.get("edit_model_path", ""),
            templates_dir=model_data.get("templates_dir", ""),
            default_steps=model_data.get("default_steps", 9),
            default_cfg_scale=model_data.get("default_cfg_scale", 0.0),
            default_resolution=model_data.get("default_resolution", 1024),
            layer_num=model_data.get("layer_num"),
            cpu_offload=model_data.get("cpu_offload", True),
            inherits=None,  # Already resolved
        )

    @classmethod
    def _load_model_data_recursive(
        cls, data: dict, model_id: str, visited: set
    ) -> dict:
        """Recursively load model data with inheritance."""
        if model_id in visited:
            raise ValueError(f"Circular inheritance detected: {model_id}")
        visited.add(model_id)

        if model_id not in data["models"]:
            raise KeyError(f"Model type '{model_id}' not found (referenced in inheritance)")

        model_data = data["models"][model_id].copy()

        if "inherits" in model_data:
            parent_id = model_data.pop("inherits")
            parent_data = cls._load_model_data_recursive(data, parent_id, visited)
            return {**parent_data, **model_data}

        return model_data

    @classmethod
    def _load_components(cls, data: dict) -> ComponentLibrary:
        """Load component library from TOML data."""
        library = ComponentLibrary()

        # Load encoders
        for enc_id, enc_data in data.get("encoders", {}).items():
            library.encoders[enc_id] = EncoderDefinition(
                description=enc_data.get("description", ""),
                model_id=enc_data.get("model_id", ""),
                hidden_dim=enc_data.get("hidden_dim", 2560),
                num_layers=enc_data.get("num_layers", 36),
                default_hidden_layer=enc_data.get("default_hidden_layer", -2),
                max_tokens=enc_data.get("max_tokens", 2048),
                quantization_options=enc_data.get(
                    "quantization_options", ["none", "4bit", "8bit", "fp8", "int8"]
                ),
            )

        # Load transformers
        for tf_id, tf_data in data.get("transformers", {}).items():
            library.transformers[tf_id] = TransformerDefinition(
                description=tf_data.get("description", ""),
                num_layers=tf_data.get("num_layers", 30),
                latent_channels=tf_data.get("latent_channels", 16),
                quantization_options=tf_data.get(
                    "quantization_options", ["none", "4bit", "8bit"]
                ),
            )

        # Load VAEs
        for vae_id, vae_data in data.get("vaes", {}).items():
            library.vaes[vae_id] = VAEDefinition(
                description=vae_data.get("description", ""),
                latent_scaling=vae_data.get("latent_scaling", 0.13025),
                latent_channels=vae_data.get("latent_channels", 4),
            )

        # Load schedulers
        for sched_id, sched_data in data.get("schedulers", {}).items():
            library.schedulers[sched_id] = SchedulerDefinition(
                description=sched_data.get("description", ""),
                type=sched_data.get("type", "flow_matching"),
                shift=sched_data.get("shift"),
                shift_mode=sched_data.get("shift_mode"),
            )

        return library

    @classmethod
    def _load_quantization(cls, data: dict, quant_id: str) -> QuantizationPreset:
        """Load quantization preset."""
        if "quantization" not in data or quant_id not in data["quantization"]:
            return QuantizationPreset()

        quant_data = data["quantization"][quant_id]
        return QuantizationPreset(
            description=quant_data.get("description", ""),
            text_encoder=quant_data.get("text_encoder", "none"),
            transformer=quant_data.get("transformer", "none"),
        )

    @classmethod
    def _load_dype(cls, data: dict, enabled: bool) -> DyPEConfig:
        """Load DyPE configuration."""
        dype_data = data.get("features", {}).get("dype", {})
        return DyPEConfig(
            enabled=enabled or dype_data.get("enabled", False),
            method=dype_data.get("method", "vision_yarn"),
            dype_scale=dype_data.get("dype_scale", 2.0),
            dype_exponent=dype_data.get("dype_exponent", 2.0),
            dype_start_sigma=dype_data.get("dype_start_sigma", 1.0),
            base_shift=dype_data.get("base_shift", 0.5),
            max_shift=dype_data.get("max_shift", 1.15),
            base_resolution=dype_data.get("base_resolution", 1024),
            anisotropic=dype_data.get("anisotropic", False),
        )

    @classmethod
    def _load_slg(cls, data: dict, enabled: bool) -> SLGConfig:
        """Load SLG configuration."""
        slg_data = data.get("features", {}).get("slg", {})
        return SLGConfig(
            enabled=enabled or slg_data.get("enabled", False),
            scale=slg_data.get("scale", 2.5),
            layers=slg_data.get("layers", [7, 8, 9, 10, 11, 12]),
            start=slg_data.get("start", 0.05),
            stop=slg_data.get("stop", 0.5),
        )

    @classmethod
    def _load_fmtt(cls, data: dict, enabled: bool) -> FMTTConfig:
        """Load FMTT configuration."""
        fmtt_data = data.get("features", {}).get("fmtt", {})
        return FMTTConfig(
            enabled=enabled or fmtt_data.get("enabled", False),
            guidance_scale=fmtt_data.get("guidance_scale", 1.0),
            guidance_start=fmtt_data.get("guidance_start", 0.0),
            guidance_stop=fmtt_data.get("guidance_stop", 0.5),
            normalize_mode=fmtt_data.get("normalize_mode", "unit"),
            decode_scale=fmtt_data.get("decode_scale", 0.5),
            siglip_model=fmtt_data.get("siglip_model", "google/siglip2-giant-opt-patch16-384"),
            siglip_device=fmtt_data.get("siglip_device", "cpu"),
        )

    def validate(self) -> list[str]:
        """
        Validate configuration and return list of errors.

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        # Validate component references
        if self.model.text_encoder and self.model.text_encoder not in self.components.encoders:
            errors.append(
                f"Text encoder '{self.model.text_encoder}' not found in component library"
            )

        if self.model.dit and self.model.dit not in self.components.transformers:
            errors.append(
                f"Transformer '{self.model.dit}' not found in component library"
            )

        if self.model.vae and self.model.vae not in self.components.vaes:
            errors.append(f"VAE '{self.model.vae}' not found in component library")

        if self.model.scheduler and self.model.scheduler not in self.components.schedulers:
            errors.append(
                f"Scheduler '{self.model.scheduler}' not found in component library"
            )

        # Validate quantization options
        if self.model.text_encoder and self.quantization.text_encoder != "none":
            encoder_def = self.components.encoders.get(self.model.text_encoder)
            if encoder_def and self.quantization.text_encoder not in encoder_def.quantization_options:
                errors.append(
                    f"Quantization '{self.quantization.text_encoder}' not supported for "
                    f"encoder '{self.model.text_encoder}'. "
                    f"Supported: {encoder_def.quantization_options}"
                )

        return errors

    def to_runtime_config(self):
        """
        Convert to legacy RuntimeConfig for backward compatibility.

        Returns:
            RuntimeConfig instance with all settings applied
        """
        from llm_dit.cli import RuntimeConfig

        # Determine model type string
        model_type = "zimage"
        if "qwen" in self.model.pipeline_class.lower():
            model_type = "qwenimage"

        # Get scheduler shift
        scheduler_shift = 3.0
        if self.model.scheduler and self.model.scheduler in self.components.schedulers:
            sched = self.components.schedulers[self.model.scheduler]
            if sched.shift is not None:
                scheduler_shift = sched.shift

        # Build RuntimeConfig
        config = RuntimeConfig(
            # Model type
            model_type=model_type,
            # Paths
            model_path=self.model.model_path,
            templates_dir=self.model.templates_dir,
            # Devices
            encoder_device=self.profile.devices.get_encoder_device(),
            dit_device=self.profile.devices.get_dit_device(),
            vae_device=self.profile.devices.get_vae_device(),
            # Dtype
            torch_dtype=self.profile.dtype,
            # Quantization
            quantization=self.quantization.text_encoder,
            # Generation defaults
            width=self.model.default_resolution,
            height=self.model.default_resolution,
            steps=self.model.default_steps,
            guidance_scale=self.model.default_cfg_scale,
            # Scheduler
            shift=scheduler_shift,
            use_custom_scheduler=self.profile.use_custom_scheduler,
            # Optimizations
            attention_backend=self.profile.attention_backend,
            compile=self.profile.compile,
            tiled_vae=self.profile.tiled_vae,
            tile_size=self.profile.tile_size,
            tile_overlap=self.profile.tile_overlap,
            embedding_cache=self.profile.embedding_cache,
            cache_size=self.profile.cache_size,
            long_prompt_mode=self.profile.long_prompt_mode,
            # DyPE
            dype_enabled=self.dype.enabled,
            dype_method=self.dype.method,
            dype_scale=self.dype.dype_scale,
            dype_exponent=self.dype.dype_exponent,
            dype_start_sigma=self.dype.dype_start_sigma,
            dype_base_shift=self.dype.base_shift,
            dype_max_shift=self.dype.max_shift,
            dype_base_resolution=self.dype.base_resolution,
            dype_anisotropic=self.dype.anisotropic,
            # SLG (RuntimeConfig uses slg_scale=0 as disabled indicator)
            slg_scale=self.slg.scale if self.slg.enabled else 0.0,
            slg_layers=self.slg.layers if self.slg.enabled else None,
            slg_start=self.slg.start,
            slg_stop=self.slg.stop,
            # FMTT (RuntimeConfig uses fmtt_scale=0 as disabled indicator)
            fmtt_scale=self.fmtt.guidance_scale if self.fmtt.enabled else 0.0,
            fmtt_start=self.fmtt.guidance_start,
            fmtt_stop=self.fmtt.guidance_stop,
            fmtt_normalize=self.fmtt.normalize_mode,
            fmtt_decode_scale=self.fmtt.decode_scale,
            fmtt_siglip_model=self.fmtt.siglip_model,
            fmtt_siglip_device=self.fmtt.siglip_device,
            # Qwen-Image specific
            qwen_image_model_path=self.model.model_path if model_type == "qwenimage" else "",
            qwen_image_edit_model_path=self.model.edit_model_path if model_type == "qwenimage" else "",
            qwen_image_cpu_offload=self.model.cpu_offload,
            qwen_image_layer_num=self.model.layer_num or 4,
            qwen_image_cfg_scale=self.model.default_cfg_scale,
            qwen_image_steps=self.model.default_steps,
            qwen_image_resolution=self.model.default_resolution,
            qwen_image_quantize_text_encoder=self.quantization.text_encoder,
            qwen_image_quantize_transformer=self.quantization.transformer,
            # Server
            host=self.server_host,
            port=self.server_port,
        )

        # Apply overrides
        for key, value in self.overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config


def is_modular_config(path: Path | str) -> bool:
    """
    Check if a TOML file uses the modular config format.

    Args:
        path: Path to config file

    Returns:
        True if file has [configs] section (modular format)
    """
    path = Path(path)
    if not path.exists():
        return False

    with open(path, "rb") as f:
        data = tomllib.load(f)

    return "configs" in data
