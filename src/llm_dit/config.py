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
from typing import Any, List, Literal

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

    Quantization Options:
    - "none": No quantization (full precision)
    - "4bit": BitsAndBytes 4-bit quantization (NF4)
    - "8bit": BitsAndBytes 8-bit quantization (INT8)
    - "int8_dynamic": PyTorch native int8 dynamic quantization (torchao)
        - Uses torch.ao.quantization.quantize_dynamic()
        - ~50% VRAM reduction, well-validated for LLMs
        - Applied post-load, no BitsAndBytes dependency
    """

    device: str = "auto"  # auto, cuda, mps, cpu
    torch_dtype: str = "bfloat16"  # bfloat16, float16, float32
    quantization: str = "none"  # none, 4bit, 8bit, int8_dynamic
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
            BitsAndBytesConfig if BitsAndBytes quantization is enabled, None otherwise.
            Returns None for int8_dynamic (handled separately via post-load quantization).

        Note:
            This is the v5-compliant way to configure quantization.
            The config should be passed to from_pretrained() as:

                model = AutoModel.from_pretrained(
                    model_path,
                    quantization_config=config.encoder.get_quantization_config(),
                )

            For int8_dynamic, use needs_post_load_quantization() and
            apply_post_load_quantization() instead.
        """
        if self.quantization in ("none", "int8_dynamic"):
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
                f"Valid options: none, 4bit, 8bit, int8_dynamic"
            )

    def needs_post_load_quantization(self) -> bool:
        """Check if post-load quantization is needed.

        Returns:
            True if int8_dynamic quantization should be applied after model loading.
        """
        return self.quantization == "int8_dynamic"

    def apply_post_load_quantization(self, model: "torch.nn.Module") -> "torch.nn.Module":
        """Apply post-load quantization (int8_dynamic) to the model.

        Uses torch.ao.quantization.quantize_dynamic() to apply int8 dynamic
        quantization to all Linear layers. This provides ~50% VRAM reduction
        with minimal quality impact for LLMs.

        Args:
            model: The loaded model to quantize

        Returns:
            Quantized model (in-place modification)

        Raises:
            ValueError: If quantization mode is not int8_dynamic
        """
        if self.quantization != "int8_dynamic":
            raise ValueError(
                f"apply_post_load_quantization() only valid for int8_dynamic, "
                f"got {self.quantization}"
            )

        import torch.ao.quantization as tq

        logger.info("Applying int8 dynamic quantization (torchao)...")

        # Quantize all Linear layers to int8
        model = tq.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8,
        )

        logger.info("  int8 dynamic quantization applied successfully")
        return model


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
    cfg_normalization: float = 0.0  # CFG norm clamping (0.0 = disabled)
    cfg_truncation: float = 1.0  # CFG truncation threshold (1.0 = no truncation)
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
class QwenImageConfig:
    """Configuration for Qwen-Image-Layered model.

    This enables image-to-layers decomposition using the Qwen-Image-Layered model,
    which is separate from the Z-Image text-to-image pipeline.

    Key differences from Z-Image:
    - Uses Qwen2.5-VL-7B-Instruct as text encoder (3584 dim vs 2560)
    - 60-layer DiT (vs 28+2 context refiner)
    - 2x2 latent packing (16 channels -> 64 packed)
    - Outputs multiple RGBA layers for decomposition
    - Only supports 640 or 1024 base resolutions

    The model takes an input image and decomposes it into N+1 layers:
    - Layer 0: Composite (input) layer
    - Layers 1-N: Decomposed RGBA layers
    """

    model_path: str = ""  # Path to Qwen-Image-Layered model directory
    edit_model_path: str = ""  # Path to Qwen-Image-Edit model (or HuggingFace ID)
    device: str = "cuda"  # Device for DiT and VAE
    text_encoder_device: str = "cuda"  # Device for text encoder (7B model)
    torch_dtype: str = "bfloat16"  # Model dtype
    cpu_offload: bool = True  # Enable sequential CPU offload for memory efficiency

    # Quantization (for VRAM-constrained GPUs like RTX 4090)
    quantize_text_encoder: str = "none"  # none/4bit/8bit - Qwen2.5-VL-7B: 14GB -> 3.5GB (4bit)
    quantize_transformer: str = "none"  # none/4bit/8bit - DiT: 8GB -> 2GB (4bit)

    # Generation settings
    num_inference_steps: int = 40  # Denoising steps (40 for Edit-2511, was 50 for 2509)
    cfg_scale: float = 4.0  # Classifier-free guidance scale
    layer_num: int = 4  # Number of decomposition layers (outputs layer_num+1 images)

    # Resolution (only 640 or 1024 supported)
    resolution: int = 1024  # Base resolution (enforced to 640 or 1024)

    # Flow matching scheduler
    shift: float | None = None  # Dynamic shift computed from latent size if None

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

    def validate_resolution(self) -> None:
        """Validate and enforce supported resolutions."""
        if self.resolution not in (640, 1024):
            raise ValueError(
                f"Qwen-Image-Layered only supports resolutions 640 or 1024, "
                f"got {self.resolution}. The model was trained on these specific "
                f"resolutions and other values may produce poor results."
            )


@dataclass
class DyPEConfig:
    """Configuration for DyPE (Dynamic Position Extrapolation).

    DyPE is a training-free technique that enables high-resolution generation
    (2K-4K+) by dynamically adjusting RoPE frequencies based on the diffusion
    timestep. The core insight: early diffusion steps establish low-frequency
    structure while late steps add high-frequency details.

    Based on ComfyUI-DyPE implementation.

    Attributes:
        enabled: Whether DyPE is enabled (default: False)
        method: RoPE extrapolation method (vision_yarn, yarn, ntk)
        dype_scale: Magnitude of DyPE effect (lambda_s, default: 2.0)
        dype_exponent: Decay speed of DyPE (lambda_t, default: 2.0 = quadratic)
        dype_start_sigma: When to start DyPE decay (0-1, 1.0 = from start)
        base_shift: Noise schedule shift at base resolution
        max_shift: Noise schedule shift at max resolution
        base_resolution: Training resolution (Z-Image: 1024, Qwen: 1328)
        anisotropic: Use per-axis scaling for extreme aspect ratios
    """

    enabled: bool = False
    method: Literal["vision_yarn", "yarn", "ntk"] = "vision_yarn"
    dype_scale: float = 2.0
    dype_exponent: float = 2.0
    dype_start_sigma: float = 1.0
    base_shift: float = 0.5
    max_shift: float = 1.15
    base_resolution: int = 1024
    anisotropic: bool = False

    def __post_init__(self):
        """Validate and clamp parameters."""
        self.dype_start_sigma = max(0.001, min(1.0, self.dype_start_sigma))

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "method": self.method,
            "dype_scale": self.dype_scale,
            "dype_exponent": self.dype_exponent,
            "dype_start_sigma": self.dype_start_sigma,
            "base_shift": self.base_shift,
            "max_shift": self.max_shift,
            "base_resolution": self.base_resolution,
            "anisotropic": self.anisotropic,
        }


@dataclass
class SLGConfig:
    """Configuration for Skip Layer Guidance (SLG).

    SLG improves structure and anatomy, especially for human/animal subjects,
    by selectively skipping layers during the denoising process and applying
    guidance based on the difference. This requires approximately 2x inference
    time since it runs two forward passes per step where SLG is active.

    Attributes:
        enabled: Whether SLG is enabled (default: False)
        scale: Guidance scale, typical values 2.0-4.0 (default: 2.8)
        layers: List of layer indices to skip, e.g., [15, 16, 17, 18, 19]
        start: Start SLG at this fraction of total steps (default: 0.01 = 1%)
        stop: Stop SLG at this fraction of total steps (default: 0.2 = 20%)

    Example config.toml:
        [rtx4090.slg]
        enabled = true
        scale = 2.5
        layers = [7, 8, 9, 10, 11, 12]
        start = 0.05
        stop = 0.5

    Note on Z-Image defaults:
        - Z-Image DiT has 30 layers (middle = ~10-20)
        - Turbo-distilled (8-9 steps) with shift 3.0-7.0
        - Structure established in first ~4 steps
        - Layers [7-12] target middle layers for structure
        - Range 5%-50% catches steps 0-4 at 8 total steps
        - Scale 2.5 (lower than SD3.5's 2.8 since more steps affected)
    """

    enabled: bool = False
    scale: float = 2.5
    layers: List[int] = field(default_factory=lambda: [7, 8, 9, 10, 11, 12])
    start: float = 0.05
    stop: float = 0.5

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "scale": self.scale,
            "layers": self.layers,
            "start": self.start,
            "stop": self.stop,
        }


@dataclass
class FMTTConfig:
    """Configuration for Flow Map Trajectory Tilting (FMTT).

    FMTT is a test-time optimization technique that guides diffusion sampling
    toward higher-reward regions using gradient-based trajectory modification.
    At each step, it predicts where the trajectory will end (via flow map),
    evaluates a reward, and nudges the velocity toward higher reward.

    Memory considerations:
        - Loads SigLIP2-Giant (~4GB VRAM)
        - For 24GB cards, encoder must be on CPU when FMTT is enabled
        - Adds ~3-4GB overhead during guided steps

    Attributes:
        enabled: Whether FMTT is enabled (default: False)
        guidance_scale: Scale for reward gradients (0.5-2.0 typical, default: 1.0)
        guidance_start: Start guidance at this fraction of steps (default: 0.0)
        guidance_stop: Stop guidance at this fraction of steps (default: 0.5)
        normalize_mode: Gradient normalization mode:
            - "unit": Normalize to unit norm (default, most stable)
            - "clip": Clip to max norm
            - "none": No normalization
        decode_scale: Scale factor for intermediate VAE decode (default: 0.5)
            - 0.5 = 512px for 1024px input (saves VRAM)
            - 1.0 = full resolution (more precise but uses more VRAM)
        reward_model: Reward model to use (default: "siglip")
            - "siglip": SigLIP2-Giant for text-image alignment
        siglip_model: HuggingFace model ID for SigLIP (default: google/siglip2-giant-opt-patch16-384)
        siglip_device: Device for SigLIP (default: "cuda")
            - "cuda": Run on GPU (requires ~4GB VRAM)
            - "cpu": Run on CPU (slower but saves VRAM)

    Example config.toml:
        [rtx4090.fmtt]
        enabled = false
        guidance_scale = 1.0
        guidance_start = 0.0
        guidance_stop = 0.5
        normalize_mode = "unit"
        decode_scale = 0.5
        siglip_model = "google/siglip2-giant-opt-patch16-384"
        siglip_device = "cuda"

    Reference: arXiv:2511.22688 (Test-Time Scaling of Diffusion Models with Flow Maps)
    """

    enabled: bool = False
    guidance_scale: float = 1.0
    guidance_start: float = 0.0
    guidance_stop: float = 0.5
    normalize_mode: str = "unit"
    decode_scale: float = 0.5
    reward_model: str = "siglip"
    siglip_model: str = "google/siglip2-giant-opt-patch16-384"
    siglip_device: str = "cuda"

    def __post_init__(self):
        """Validate normalize_mode."""
        valid_modes = ("unit", "clip", "none")
        if self.normalize_mode not in valid_modes:
            raise ValueError(
                f"normalize_mode must be one of {valid_modes}, got {self.normalize_mode}"
            )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "guidance_scale": self.guidance_scale,
            "guidance_start": self.guidance_start,
            "guidance_stop": self.guidance_stop,
            "normalize_mode": self.normalize_mode,
            "decode_scale": self.decode_scale,
            "reward_model": self.reward_model,
            "siglip_model": self.siglip_model,
            "siglip_device": self.siglip_device,
        }


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
    """Complete configuration for Z-Image and Qwen-Image generation."""

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
    qwen_image: QwenImageConfig = field(default_factory=QwenImageConfig)
    dype: DyPEConfig = field(default_factory=DyPEConfig)
    slg: SLGConfig = field(default_factory=SLGConfig)
    fmtt: FMTTConfig = field(default_factory=FMTTConfig)

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
        qwen_image_data = data.pop("qwen_image", {})
        dype_data = data.pop("dype", {})
        slg_data = data.pop("slg", {})
        fmtt_data = data.pop("fmtt", {})

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
            qwen_image=QwenImageConfig(**qwen_image_data),
            dype=DyPEConfig(**dype_data),
            slg=SLGConfig(**slg_data),
            fmtt=FMTTConfig(**fmtt_data),
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
            "qwen_image": {
                "model_path": self.qwen_image.model_path,
                "edit_model_path": self.qwen_image.edit_model_path,
                "device": self.qwen_image.device,
                "text_encoder_device": self.qwen_image.text_encoder_device,
                "torch_dtype": self.qwen_image.torch_dtype,
                "cpu_offload": self.qwen_image.cpu_offload,
                "num_inference_steps": self.qwen_image.num_inference_steps,
                "cfg_scale": self.qwen_image.cfg_scale,
                "layer_num": self.qwen_image.layer_num,
                "resolution": self.qwen_image.resolution,
                "shift": self.qwen_image.shift,
            },
            "dype": self.dype.to_dict(),
            "slg": self.slg.to_dict(),
            "fmtt": self.fmtt.to_dict(),
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
