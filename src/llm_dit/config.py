"""
TOML-based configuration for llm-dit-experiments.

Flat config format with per-pipeline sections. Quantization uses unified
torchao methods: none, fp8-dynamic, fp8-weight-only, int8, int4.

Example config (config.toml):

    default_pipeline = "none"
    model_path = "/path/to/model"

    [quantization]
    encoder = "none"
    transformer = "fp8-weight-only"
    vae = "none"

    [encoder]
    device = "cuda"
    quantization = "none"

    [flux2]
    model_path = "/path/to/FLUX.2-klein"
    quantization = "fp8-weight-only"
    compile = true
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, List, Literal

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


# ---------------------------------------------------------------------------
# Unified quantization config (shared across all pipelines)
# ---------------------------------------------------------------------------

# Canonical method strings accepted by quantize_component()
VALID_QUANT_METHODS = ("none", "fp8-dynamic", "fp8-weight-only", "int8", "int4")


@dataclass
class ComponentQuantConfig:
    """Quantization config for a single model component (encoder, transformer, VAE)."""

    method: str = "none"  # none, fp8-dynamic, fp8-weight-only, int8, int4
    granularity: str = "per-tensor"  # per-tensor, per-row (FP8 only)

    def __post_init__(self) -> None:
        if self.method not in VALID_QUANT_METHODS:
            raise ValueError(
                f"Invalid quantization method: '{self.method}'. "
                f"Valid options: {', '.join(VALID_QUANT_METHODS)}"
            )
        if self.granularity not in ("per-tensor", "per-row"):
            raise ValueError(
                f"Invalid granularity: '{self.granularity}'. "
                f"Valid options: per-tensor, per-row"
            )

    @property
    def is_none(self) -> bool:
        return self.method == "none"

    @property
    def is_fp8(self) -> bool:
        return self.method in ("fp8-dynamic", "fp8-weight-only")


@dataclass
class PipelineQuantConfig:
    """Resolved quantization config for all components of a pipeline."""

    encoder: ComponentQuantConfig = field(default_factory=ComponentQuantConfig)
    transformer: ComponentQuantConfig = field(default_factory=ComponentQuantConfig)
    vae: ComponentQuantConfig = field(default_factory=ComponentQuantConfig)

    @classmethod
    def from_dict(cls, data: dict) -> "PipelineQuantConfig":
        """Parse from dict like {encoder: "fp8-weight-only", transformer: "int8", vae: "none"}."""
        granularity = data.get("granularity", "per-tensor")
        return cls(
            encoder=ComponentQuantConfig(
                method=data.get("encoder", "none"), granularity=granularity,
            ),
            transformer=ComponentQuantConfig(
                method=data.get("transformer", "none"), granularity=granularity,
            ),
            vae=ComponentQuantConfig(
                method=data.get("vae", "none"), granularity=granularity,
            ),
        )

    def to_dict(self) -> dict:
        return {
            "encoder": self.encoder.method,
            "transformer": self.transformer.method,
            "vae": self.vae.method,
            "granularity": self.encoder.granularity,
        }


# Qwen-Image optimization presets for different hardware/memory configurations
QWEN_IMAGE_PRESETS = {
    "balanced": {
        "quantize_text_encoder": "int8",
        "quantize_transformer": "none",
        "quantize_vae": "none",
        "offload_type": "model",
        "cpu_offload": True,
    },
    "rtx4090_fp8": {
        "quantize_text_encoder": "fp8-weight-only",
        "quantize_transformer": "fp8-weight-only",
        "quantize_vae": "int8",
        "offload_type": "none",
        "cpu_offload": False,
    },
    "rtx4090_group": {
        "quantize_text_encoder": "int8",
        "quantize_transformer": "none",
        "quantize_vae": "int8",
        "offload_type": "group",
        "num_blocks_per_group": 2,
        "cpu_offload": True,
    },
    "max_vram_savings": {
        "quantize_text_encoder": "int4",
        "quantize_transformer": "int4",
        "quantize_vae": "int8",
        "offload_type": "group",
        "num_blocks_per_group": 1,
        "cpu_offload": True,
    },
    "amd_mi300": {
        "quantize_text_encoder": "int8",
        "quantize_transformer": "fp8-dynamic",
        "quantize_vae": "int8",
        "offload_type": "model",
        "cpu_offload": True,
    },
}


@dataclass
class EncoderConfig:
    """Configuration for the text encoder (LLM).

    Quantization Options (torchao unified):
    - "none": No quantization (full precision)
    - "fp8-dynamic": FP8 weights + FP8 activations (RTX 4090+)
    - "fp8-weight-only": FP8 weights, BF16 activations
    - "int8": INT8 weight-only
    - "int4": INT4 weight-only (max compression)
    """

    device: str = "auto"  # auto, cuda, mps, cpu
    dtype: str = "bfloat16"  # bfloat16, float16, float32
    quantization: str = "none"  # none, fp8-dynamic, fp8-weight-only, int8, int4
    cpu_offload: bool = False  # Offload to CPU after encoding
    trust_remote_code: bool = True
    max_length: int = 512
    hidden_layer: int = -2  # Which layer to extract embeddings from (-1=last, -2=penultimate)
    layer_weights: dict[int, float] | None = None  # Multi-layer blending weights (overrides hidden_layer)

    def get_dtype(self) -> torch.dtype:
        """Convert string dtype to torch.dtype."""
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.dtype, torch.bfloat16)

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
    dtype: str = "bfloat16"
    enable_model_cpu_offload: bool = False  # Sequential CPU offload
    enable_sequential_cpu_offload: bool = False  # More aggressive offload

    def get_dtype(self) -> torch.dtype:
        """Convert string dtype to torch.dtype."""
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.dtype, torch.bfloat16)

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
    cfg_norm_mode: str = "clamp"  # CFG norm mode: "clamp" or "match" (DiffSynth-style)
    enable_thinking: bool = True
    default_template: str | None = None
    seed: int | None = None  # Random seed for reproducibility
    negative_prompt: str | None = None  # Negative prompt for CFG
    system_prompt: str | None = None  # System prompt override
    thinking_content: str | None = None  # Thinking content override
    assistant_content: str | None = None  # Assistant content override


@dataclass
class OptimizationConfig:
    """Optimization settings for pipeline execution."""

    flash_attn: bool = False  # Enable Flash Attention
    compile: bool = False  # Enable torch.compile
    compile_mode: str = "default"  # torch.compile mode (default is CPU-offload safe)
    cpu_offload: bool = False  # Enable CPU offload for transformer
    dit_device: str = "auto"  # DiT device placement
    vae_device: str = "auto"  # VAE device placement


@dataclass
class SchedulerConfig:
    """Scheduler settings."""

    shift: float = 3.0  # Flow matching scheduler shift parameter
    shift_terminal: float | None = None  # Terminal sigma value (Qwen-Image only, None for Z-Image)
    dynamic_shift: bool = False  # Calculate shift based on resolution (overrides shift)
    d_noise: float = 1.0  # Sigma schedule scaling: <1.0 = sharper, >1.0 = softer


@dataclass
class APIConfig:
    """Configuration for remote text encoder API (distributed inference).

    When configured, the text encoder runs on a remote server (e.g., Mac)
    while the DiT runs locally on GPU. This enables distributed inference
    across machines.
    """

    url: str = ""  # API URL, e.g., "http://mac-host:8080" (empty = use local encoder)
    model: str = "Qwen3-4B"  # Model ID for embedding extraction
    local_encoder: bool = False  # Use local encoder even when API is configured


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
class LTX2Config:
    """Configuration for LTX-2 video generation model.

    LTX-2 is a 19B parameter video+audio generation model from Lightricks:
    - 14B video transformer + 5B audio transformer (asymmetric dual-stream DiT)
    - Text encoder: Gemma 3-12B (NOT Qwen3)
    - Output: Video (512x768 typical) + optional audio
    - Memory: FP8 model (25GB) + CPU offload for RTX 4090

    Key differences from Z-Image:
    - Uses Gemma 3-12B text encoder (4096 dim) vs Qwen3-4B (2560 dim)
    - Outputs video+audio vs image only
    - 19B params vs ~2B params
    - Requires model-level CPU offloading for 24GB VRAM

    IMPORTANT: RTX 4090 (SM89) has NATIVE FP8 tensor core support but FP4 is EMULATED.
    FP8 is faster than FP4 on RTX 4090 despite larger size. Use FP4 only on Hopper+.

    Offload modes:
    - none: No offloading (requires >48GB VRAM)
    - model: Model-level offload (~20-50% slower, enables 24GB inference)
    - sequential: Layer-by-layer streaming (~3-5x slower)
    - group: Stream DiT blocks in groups (configurable VRAM vs speed)

    Memory budget (24GB):
    - Encoder (Gemma 3-12B Q4): ~6GB → offload after encoding
    - Transformer (FP8): ~19GB peak during generation
    - VAE: ~2GB for tiled decode
    - Peak: ~23-24GB with proper offloading
    """

    # Model paths
    model_path: str = ""  # Directory containing LTX-2 model files
    transformer_file: str = "ltx-2-19b-distilled-fp8.safetensors"  # Native FP8 recommended

    # Text encoder (Gemma 3-12B)
    encoder_model_id: str = "models/LTX-2/text_encoder"
    encoder_quantization: str = "none"  # Gemma 3 already ships with Q4 QAT
    encoder_cpu_offload: bool = True  # Offload after encoding (REQUIRED for 24GB)

    # LoRA configuration
    lora_path: str = ""  # Path to LoRA safetensors
    lora_scale: float = 1.0  # LoRA blend scale (0.0-1.0)

    # CPU offloading during generation
    offload_mode: str = "model"  # none, model, sequential, group
    num_blocks_per_group: int = 2  # For group offload: DiT blocks to keep on GPU

    # Video generation defaults
    height: int = 768  # Video height (multiple of 32)
    width: int = 512  # Video width (multiple of 32)
    num_frames: int = 33  # Number of frames (33-65 typical for 24GB)
    fps: int = 24  # Output FPS
    num_inference_steps: int = 12  # Diffusion steps (8+4 for distilled, 40 for full)
    guidance_scale: float = 3.5  # CFG scale (3.0-4.0 recommended)

    # Audio generation
    audio_enabled: bool = False  # Enable audio stream generation
    audio_negative_prompt: str = ""  # Negative prompt for audio CFG

    # Distillation settings
    use_distilled: bool = True  # Use distilled model (8+4 step)
    distilled_steps_stage1: int = 8  # First stage steps (distilled)
    distilled_steps_stage2: int = 4  # Second stage steps (distilled)

    # Image-to-video (optional)
    input_image: str = ""  # Path to input image for I2V mode
    image_weight: float = 0.7  # Blend weight for input image (0.0-1.0)

    # VAE settings
    tiled_vae: bool = True  # REQUIRED for video decode on 24GB
    tile_size_temporal: int = 8  # Temporal tile size (frames)
    tile_size_spatial: int = 256  # Spatial tile size (pixels)
    tile_overlap_temporal: int = 4  # Temporal overlap
    tile_overlap_spatial: int = 32  # Spatial overlap

    # Embedding precomputation
    save_embeddings: str | None = None  # Save embeddings to this path (skip generation)
    load_embeddings: str | None = None  # Load precomputed embeddings from this path

    # Output
    output_path: str = "output.mp4"  # Output video path

    # Device placement
    text_encoder_device: str = "cpu"  # Device for Gemma3 (cpu recommended for 24GB)
    transformer_device: str = "cuda"  # Device for DiT transformer
    vae_device: str = "cuda"  # Device for VAE decoder

    # Quantization
    quantize: str = "fp8"  # Transformer quantization (none, fp8)
    skip_cleanup: bool = False  # Skip memory cleanup between stages
    gemma_variant: str = "bf16"  # Gemma3 backbone: bf16, 8bit, q4-qat

    # Preset configuration
    default_preset: str = ""  # Default preset to load (e.g., "cinematic")

    def get_dtype(self) -> torch.dtype:
        """Get torch dtype for computation.

        LTX-2 always uses bfloat16 for computation regardless of weight quantization.
        FP8/FP4 refers to weight storage format, not computation dtype.
        The transformer loads quantized weights but computes in bfloat16.
        """
        return torch.bfloat16

    def get_model_file_path(self) -> str:
        """Get full path to transformer model file."""
        if not self.model_path:
            raise ValueError("model_path is required")
        return os.path.join(self.model_path, self.transformer_file)

    def get_total_steps(self) -> int:
        """Get total inference steps based on distillation mode."""
        if self.use_distilled:
            return self.distilled_steps_stage1 + self.distilled_steps_stage2
        return self.num_inference_steps

    def validate(self) -> None:
        """Validate configuration settings."""
        valid_offload_modes = ("none", "model", "sequential", "group")
        if self.offload_mode not in valid_offload_modes:
            raise ValueError(
                f"Invalid offload_mode='{self.offload_mode}'. "
                f"Valid options: {', '.join(valid_offload_modes)}"
            )

        # Validate num_blocks_per_group for group offload mode
        if self.offload_mode == "group" and self.num_blocks_per_group <= 0:
            raise ValueError(
                f"num_blocks_per_group must be > 0 for group offload mode, "
                f"got {self.num_blocks_per_group}"
            )

        # Validate dimensions are multiples of 32
        if self.height % 32 != 0:
            raise ValueError(f"height must be multiple of 32, got {self.height}")
        if self.width % 32 != 0:
            raise ValueError(f"width must be multiple of 32, got {self.width}")

        # Validate LoRA scale
        if not 0.0 <= self.lora_scale <= 1.0:
            raise ValueError(f"lora_scale must be 0.0-1.0, got {self.lora_scale}")

        # Validate image weight
        if not 0.0 <= self.image_weight <= 1.0:
            raise ValueError(f"image_weight must be 0.0-1.0, got {self.image_weight}")

        # Warn if num_inference_steps doesn't match distilled mode
        expected_steps = self.distilled_steps_stage1 + self.distilled_steps_stage2
        if self.use_distilled and self.num_inference_steps != expected_steps:
            logger.warning(
                f"use_distilled=True but num_inference_steps={self.num_inference_steps} "
                f"doesn't match distilled steps ({expected_steps}). "
                f"get_total_steps() will return {expected_steps} for distilled mode."
            )

        # Warn about FP4 on RTX 4090
        if "fp4" in self.transformer_file.lower():
            logger.warning(
                "FP4 model selected. Note: RTX 4090 (SM89) emulates FP4, which is slower "
                "than native FP8. Consider using FP8 model for better performance on RTX 4090."
            )

    # LTX-2 architecture constants for VRAM estimation (ClassVar to exclude from dataclass fields)
    _LTX2_NUM_BLOCKS: ClassVar[int] = 48  # Total transformer blocks (14B video DiT)
    _LTX2_VRAM_FP8_GB: ClassVar[float] = 19.0  # FP8 quantized transformer
    _LTX2_VRAM_FP4_GB: ClassVar[float] = 14.0  # FP4 quantized transformer
    _LTX2_VRAM_BF16_GB: ClassVar[float] = 38.0  # Full precision transformer
    _GEMMA3_VRAM_Q4_GB: ClassVar[float] = 6.0  # Q4 quantized Gemma 3-12B
    _GEMMA3_VRAM_FULL_GB: ClassVar[float] = 24.0  # Full precision Gemma 3-12B
    _VAE_VRAM_GB: ClassVar[float] = 2.0  # Video VAE
    _OVERHEAD_GB: ClassVar[float] = 2.0  # CUDA overhead, activations, etc.
    _GROUP_OVERHEAD_GB: ClassVar[float] = 4.0  # Additional overhead for group offload

    def estimate_vram_usage(self) -> dict[str, float]:
        """Estimate VRAM usage for different components (in GB).

        Returns:
            Dictionary with estimated VRAM for each component and total peak.

        Note:
            These are rough estimates based on model architecture.
            Actual usage depends on batch size, resolution, and runtime factors.
        """
        estimates = {}

        # Transformer VRAM based on file type
        if "fp8" in self.transformer_file.lower():
            estimates["transformer"] = self._LTX2_VRAM_FP8_GB
        elif "fp4" in self.transformer_file.lower():
            estimates["transformer"] = self._LTX2_VRAM_FP4_GB
        else:
            estimates["transformer"] = self._LTX2_VRAM_BF16_GB

        # Encoder (loaded temporarily, offloaded before transformer)
        if "q4" in self.encoder_model_id.lower():
            estimates["encoder"] = self._GEMMA3_VRAM_Q4_GB
        else:
            estimates["encoder"] = self._GEMMA3_VRAM_FULL_GB

        # VAE
        estimates["vae"] = self._VAE_VRAM_GB

        # Peak depends on offload mode
        if self.offload_mode == "none":
            # All components loaded simultaneously
            estimates["peak"] = estimates["transformer"] + estimates["vae"]
        elif self.offload_mode in ("model", "sequential"):
            # Sequential loading: max of any single component + overhead
            estimates["peak"] = max(estimates.values()) + self._OVERHEAD_GB
        else:  # group offload
            # Reduced transformer based on blocks per group
            # Clamp to valid range to avoid division errors
            blocks = max(1, min(self.num_blocks_per_group, self._LTX2_NUM_BLOCKS))
            block_fraction = blocks / self._LTX2_NUM_BLOCKS
            estimates["peak"] = (
                estimates["transformer"] * block_fraction
                + estimates["vae"]
                + self._GROUP_OVERHEAD_GB
            )

        return estimates

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "model_path": self.model_path,
            "transformer_file": self.transformer_file,
            "encoder_model_id": self.encoder_model_id,
            "encoder_quantization": self.encoder_quantization,
            "encoder_cpu_offload": self.encoder_cpu_offload,
            "lora_path": self.lora_path,
            "lora_scale": self.lora_scale,
            "offload_mode": self.offload_mode,
            "num_blocks_per_group": self.num_blocks_per_group,
            "height": self.height,
            "width": self.width,
            "num_frames": self.num_frames,
            "fps": self.fps,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "audio_enabled": self.audio_enabled,
            "audio_negative_prompt": self.audio_negative_prompt,
            "use_distilled": self.use_distilled,
            "distilled_steps_stage1": self.distilled_steps_stage1,
            "distilled_steps_stage2": self.distilled_steps_stage2,
            "input_image": self.input_image,
            "image_weight": self.image_weight,
            "tiled_vae": self.tiled_vae,
            "tile_size_temporal": self.tile_size_temporal,
            "tile_size_spatial": self.tile_size_spatial,
            "tile_overlap_temporal": self.tile_overlap_temporal,
            "tile_overlap_spatial": self.tile_overlap_spatial,
        }


@dataclass
class Flux2Config:
    """Configuration for FLUX.2 Klein image generation.

    FLUX.2 Klein is a family of 8 model variants:
    - Distilled (4 steps, CFG=1.0): klein-9b, klein-9b-fp8, klein-4b, klein-4b-fp8
    - Base (50 steps, CFG=4.0): klein-base-9b, klein-base-9b-fp8, klein-base-4b, klein-base-4b-fp8
    """

    model_path: str = ""  # Path to transformer weights (file or directory)
    vae_path: str = ""  # Path to VAE weights (file or directory)
    encoder_path: str = ""  # Path to Qwen3 encoder (optional, uses HuggingFace if empty)
    default_model: str = "klein-9b-fp8"  # Default model variant
    block_offload: bool = False  # Enable block-by-block GPU offloading (slower but uses ~5GB less VRAM)
    offload_between_stages: bool = True  # Three-stage offloading: encoder -> transformer -> VAE
    encoder_device: str = "cuda"  # Device for text encoder (cuda recommended, encoder offloads after use)
    default_steps: int | None = None  # Default steps (None = model default: 4 distilled, 50 base)
    default_guidance: float | None = None  # Default CFG (None = model default: 1.0 distilled, 4.0 base)
    default_preset: str = ""  # Default preset to load (e.g., "quality")

    # Performance optimization
    compile: bool = False  # torch.compile the transformer
    compile_vae: bool = False  # torch.compile the VAE decoder
    compile_mode: str = "default"  # torch.compile mode
    compile_dynamic: bool = False  # dynamic shapes: avoid recompilation per resolution
    quantization: str = "none"  # none, fp8-dynamic, fp8-weight-only, int8, int4


@dataclass
class ZImageConfig:
    """Configuration for Z-Image text-to-image generation.

    Supports two variants:
    - turbo: Fast 8-9 step generation with CFG baked in (default)
    - base: Quality 28-50 step generation with full CFG control

    Key differences between variants:
    | Setting            | Base            | Turbo                 |
    |--------------------|-----------------|----------------------|
    | Scheduler shift    | 6.0             | 3.0                  |
    | Steps              | 35 (28-50 rec.) | 9 (8 actual forwards)|
    | CFG/guidance_scale | 4.0 (3.0-5.0)   | 0.0 (baked in)       |
    | Negative prompts   | Supported       | Not used             |
    | Model path         | models/Z-Image  | models/Z-Image-Turbo |

    Both variants use the same Qwen3-4B text encoder and DiT architecture.

    Generation parameters (steps, guidance_scale, shift, negative_prompt) are
    defined in presets/ and loaded via default_preset. This config only holds
    infrastructure settings (paths, variant selection).
    """

    model_path: str = ""  # Path to Z-Image model (turbo or base)
    text_encoder_path: str = ""  # Optional separate path for text encoder
    variant: str = "auto"  # auto, turbo, base (auto-detects from scheduler_config.json)

    # Preset configuration - defines steps, guidance_scale, shift, negative_prompt
    # Presets are loaded from presets/zimage/{preset_name}.md
    default_preset: str = ""  # Default preset to load (e.g., "photorealistic")


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
    dtype: str = "bfloat16"  # Model dtype
    cpu_offload: bool = True  # Enable sequential CPU offload for memory efficiency

    # Quantization (for VRAM-constrained GPUs like RTX 4090)
    # Valid options: none, fp8-dynamic, fp8-weight-only, int8, int4
    # - none: Full precision (BF16)
    # - fp8-dynamic: FP8 weights + activations (RTX 4090+, SM89+)
    # - fp8-weight-only: FP8 weights, BF16 compute (RTX 4090+, compile-safe)
    # - int8: TorchAO INT8 weight-only (any GPU)
    # - int4: TorchAO INT4 weight-only (max compression)
    quantize_text_encoder: str = "none"  # Qwen2.5-VL-7B: 14GB -> 7GB (fp8) or 4GB (int4)
    quantize_transformer: str = "none"  # DiT: 8GB -> 4GB (fp8) or 2GB (int4)
    quantize_vae: str = "none"  # VAE: 500MB -> 250MB (int8). Only int8 for Conv2d

    # Offloading strategy
    # - model: enable_model_cpu_offload() - swap entire components
    # - group: apply_group_offloading() - stream DiT blocks (4-6GB VRAM)
    # - sequential: enable_sequential_cpu_offload() - layer-by-layer (slowest)
    offload_type: str = "model"  # model, group, sequential
    num_blocks_per_group: int = 2  # For group offloading: blocks to keep on GPU

    # Generation settings
    num_inference_steps: int = 25  # Denoising steps for Edit-2511
    cfg_scale: float = 4.0  # Classifier-free guidance scale
    layer_num: int = 4  # Number of decomposition layers (outputs layer_num+1 images)

    # Resolution (only 640 or 1024 supported)
    resolution: int = 1024  # Base resolution (enforced to 640 or 1024)

    # Flow matching scheduler
    shift: float | None = None  # Dynamic shift computed from latent size if None

    def get_dtype(self) -> torch.dtype:
        """Convert string dtype to torch.dtype."""
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.dtype, torch.bfloat16)

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

    def validate_quantization(self) -> None:
        """Validate quantization settings and check hardware compatibility."""
        valid_quant = ("none", "fp8-dynamic", "fp8-weight-only", "int8", "int4")
        valid_vae_quant = ("none", "int8")  # Only int8 for Conv2d
        valid_offload = ("model", "group", "sequential")

        for field, value in [
            ("quantize_text_encoder", self.quantize_text_encoder),
            ("quantize_transformer", self.quantize_transformer),
        ]:
            if value not in valid_quant:
                raise ValueError(
                    f"Invalid {field}='{value}'. Valid options: {', '.join(valid_quant)}"
                )

        if self.quantize_vae not in valid_vae_quant:
            raise ValueError(
                f"Invalid quantize_vae='{self.quantize_vae}'. "
                f"VAE only supports: {', '.join(valid_vae_quant)} (Conv2d layers)"
            )

        if self.offload_type not in valid_offload:
            raise ValueError(
                f"Invalid offload_type='{self.offload_type}'. "
                f"Valid options: {', '.join(valid_offload)}"
            )

        # Check FP8 hardware compatibility
        fp8_options = ("fp8-dynamic", "fp8-weight-only")
        uses_fp8 = (
            self.quantize_text_encoder in fp8_options or self.quantize_transformer in fp8_options
        )
        if uses_fp8:
            try:
                from llm_dit.quantization import check_fp8_support

                if not check_fp8_support():
                    logger.warning(
                        "FP8 quantization requires RTX 4090+ (compute 8.9+) or AMD MI300. "
                        "FP8 may not work on this hardware. Consider 'int8' instead."
                    )
            except ImportError:
                pass  # quantization module not available

    def apply_preset(self, preset_name: str) -> None:
        """
        Apply an optimization preset to this config.

        Available presets:
        - "balanced": Good defaults for most systems (8-bit text encoder, model offload)
        - "rtx4090_fp8": Max performance on RTX 4090 (FP8, no offload, ~13GB VRAM)
        - "rtx4090_group": RTX 4090 with group offloading (~16-18GB VRAM)
        - "max_vram_savings": Minimum VRAM (~8-10GB), uses 4-bit quantization
        - "amd_mi300": AMD ROCm support with DiffSynth FP8

        Args:
            preset_name: Name of preset to apply

        Raises:
            ValueError: If preset_name is not valid
        """
        if preset_name not in QWEN_IMAGE_PRESETS:
            valid = list(QWEN_IMAGE_PRESETS.keys())
            raise ValueError(f"Unknown preset: {preset_name}. Valid presets: {valid}")

        preset = QWEN_IMAGE_PRESETS[preset_name]
        for key, value in preset.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Preset '{preset_name}' has unknown field: {key}")

        logger.info(f"Applied QwenImage preset: {preset_name}")

    @classmethod
    def from_preset(cls, preset_name: str, **overrides) -> "QwenImageConfig":
        """
        Create a QwenImageConfig from a preset with optional overrides.

        Args:
            preset_name: Preset to start from
            **overrides: Additional config values to override

        Returns:
            Configured QwenImageConfig instance

        Example:
            config = QwenImageConfig.from_preset(
                "rtx4090_fp8",
                model_path="/path/to/model",
                resolution=1024,
            )
        """
        config = cls()
        config.apply_preset(preset_name)

        # Apply overrides
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                logger.warning(f"Unknown config field: {key}")

        return config


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
        multipass: Generation mode (single, twopass, threepass)
        pass2_strength: img2img strength for second pass (0.0-1.0)
        pass3_strength: img2img strength for third pass (0.0-1.0)
        frequency_modulation: Enable timestep-based RoPE frequency scaling
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
    multipass: Literal["single", "twopass", "threepass"] = "single"
    pass2_strength: float = 0.5
    pass3_strength: float = 0.4
    frequency_modulation: bool = False

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
            "multipass": self.multipass,
            "pass2_strength": self.pass2_strength,
            "pass3_strength": self.pass3_strength,
            "frequency_modulation": self.frequency_modulation,
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
class EnhancementConfig:
    """Configuration for LTX-2 generation enhancement techniques.

    Collects all enhancement techniques ported from ComfyUI-LTXVideo and
    ComfyUI-KJNodes into a single config object for easy experimentation.

    These techniques are algorithm-level optimizations (pure PyTorch tensor
    operations) that can be enabled independently or in combination:

    1. Latent Normalization: Prevents CFG-induced drift ("overbaking")
    2. NAG: Normalized Attention Guidance for better CFG quality
    3. FETA: Feature Temporal Attention for temporal consistency
    4. TeaCache: Inference speedup via temporal caching
    5. FFN Chunking: Memory reduction via chunked feedforward
    6. Audio Normalization: Per-step audio latent normalization

    Example:
        config = EnhancementConfig(
            latent_norm_enabled=True,
            nag_enabled=True,
            tea_cache_enabled=True,
        )
        output = pipe(prompt="...", enhancement_config=config)
    """

    # Latent Normalization - prevents CFG overbaking
    latent_norm_enabled: bool = False
    latent_norm_factors: str = "0.9, 0.75, 0.5, 0.25, 0.0"
    latent_norm_target_mean: float = 0.0
    latent_norm_target_std: float = 1.0
    latent_norm_percentile: float = 95.0

    # NAG - Normalized Attention Guidance
    # Improves CFG quality by normalizing attention outputs to prevent divergence
    nag_enabled: bool = False
    nag_scale: float = 11.0  # Strength of negative guidance
    nag_alpha: float = 0.25  # Balance between guided and positive
    nag_tau: float = 2.5  # Clipping threshold

    # FETA - Feature Temporal Attention
    # Enhances temporal consistency by boosting cross-frame attention
    feta_enabled: bool = False
    feta_weight: float = 4.0  # Enhancement strength (2.0-8.0 typical)
    feta_start_step: int = 0  # First step to apply
    feta_end_step: int = -1  # Last step (-1 = all)

    # TeaCache - Temporal Efficient Attention Caching
    # Skips redundant transformer computations for 4-10x speedup
    tea_cache_enabled: bool = False
    tea_cache_threshold: float = 0.275  # L1 distance threshold for skip
    tea_cache_model_type: str = "14B"  # "14B", "1.3B", "i2v_480", "i2v_720"

    # FFN Chunking - Memory Efficiency
    # Chunks feedforward to reduce peak VRAM
    ffn_chunking_enabled: bool = False
    ffn_chunk_count: int = 4  # Number of chunks
    ffn_dim_threshold: int = 4096  # Only chunk if dim > threshold

    # Audio Latent Normalization (LTX-2 specific)
    # Per-step normalization of audio latents
    audio_norm_enabled: bool = False
    audio_norm_factors: str = "1,1,0.25,1,1,0.25"  # Per-step multipliers

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "latent_norm_enabled": self.latent_norm_enabled,
            "latent_norm_factors": self.latent_norm_factors,
            "latent_norm_target_mean": self.latent_norm_target_mean,
            "latent_norm_target_std": self.latent_norm_target_std,
            "latent_norm_percentile": self.latent_norm_percentile,
            "nag_enabled": self.nag_enabled,
            "nag_scale": self.nag_scale,
            "nag_alpha": self.nag_alpha,
            "nag_tau": self.nag_tau,
            "feta_enabled": self.feta_enabled,
            "feta_weight": self.feta_weight,
            "feta_start_step": self.feta_start_step,
            "feta_end_step": self.feta_end_step,
            "tea_cache_enabled": self.tea_cache_enabled,
            "tea_cache_threshold": self.tea_cache_threshold,
            "tea_cache_model_type": self.tea_cache_model_type,
            "ffn_chunking_enabled": self.ffn_chunking_enabled,
            "ffn_chunk_count": self.ffn_chunk_count,
            "ffn_dim_threshold": self.ffn_dim_threshold,
            "audio_norm_enabled": self.audio_norm_enabled,
            "audio_norm_factors": self.audio_norm_factors,
        }

    @classmethod
    def quality_preset(cls) -> "EnhancementConfig":
        """Quality-focused preset: latent norm + FETA."""
        return cls(
            latent_norm_enabled=True,
            feta_enabled=True,
            feta_weight=4.0,
        )

    @classmethod
    def speed_preset(cls) -> "EnhancementConfig":
        """Speed-focused preset: TeaCache + FFN chunking."""
        return cls(
            tea_cache_enabled=True,
            tea_cache_threshold=0.275,
            ffn_chunking_enabled=True,
        )

    @classmethod
    def all_preset(cls) -> "EnhancementConfig":
        """All enhancements enabled."""
        return cls(
            latent_norm_enabled=True,
            nag_enabled=True,
            feta_enabled=True,
            tea_cache_enabled=True,
            ffn_chunking_enabled=True,
        )


@dataclass
class FBCacheRuntimeConfig:
    """Configuration for Forward Block Cache (FBCache).

    FBCache accelerates DiT inference by skipping redundant transformer block
    computations when residual changes between steps are minimal. This is based
    on the observation that consecutive diffusion steps often produce similar
    intermediate representations.

    How it works:
        1. Always compute first transformer block
        2. Compare first-block residual to previous step
        3. If change is below threshold, skip remaining blocks and reuse cached output
        4. Always compute fully on first and last steps for quality

    Expected speedup: 30-50% with minimal quality degradation

    Adaptive thresholds by sigma phase:
        - Early (sigma > 0.7): Conservative (1%) - structure discovery phase
        - Middle (0.3 < sigma < 0.7): Aggressive (5%) - detail refinement, safe to skip
        - Late (sigma < 0.3): Conservative (1%) - fine details, be careful

    Attributes:
        enabled: Master toggle for FBCache (default: False)
        early_threshold: Threshold for high sigma phase (default: 0.01 = 1%)
        middle_threshold: Threshold for middle sigma phase (default: 0.05 = 5%)
        late_threshold: Threshold for low sigma phase (default: 0.01 = 1%)
        log_residuals: Log residual statistics for analysis (default: False)
        log_file: Optional file path for residual logs (default: None)

    Example config.toml:
        [rtx4090.fbcache]
        enabled = true
        early_threshold = 0.01
        middle_threshold = 0.05
        late_threshold = 0.01
        log_residuals = true

    Reference: DiffSynth-Engine FBCache implementation
    """

    enabled: bool = False
    early_threshold: float = 0.01
    middle_threshold: float = 0.05
    late_threshold: float = 0.01
    log_residuals: bool = False
    log_file: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "early_threshold": self.early_threshold,
            "middle_threshold": self.middle_threshold,
            "late_threshold": self.late_threshold,
            "log_residuals": self.log_residuals,
            "log_file": self.log_file,
        }


@dataclass
class RewriterConfig:
    """Configuration for prompt rewriting using LLM generation.

    The rewriter can use either the local model or a remote API backend
    for text generation. When use_api is True and api_url is set,
    the rewriter will use the API backend instead of the local model.

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
    max_tokens: int = 1024  # Maximum tokens to generate
    # API timeout settings
    timeout: float = 120.0  # API request timeout in seconds


@dataclass
class LoggingConfig:
    """Logging configuration for file and console output."""

    enabled: bool = False
    log_dir: str = ""
    log_level: str = "INFO"
    json_format: bool = True
    max_bytes: int = 10485760  # 10MB per log file
    backup_count: int = 5  # Rotated log files to keep
    log_requests: bool = True  # Log API request/response metadata
    log_generation_params: bool = True  # Log generation parameters


@dataclass
class WanConfig:
    """Configuration for Wan/HuMo video generation."""

    humo_path: str = ""  # Path to HuMo transformer
    base_path: str = ""  # Path to Wan2.1-T2V for VAE/text encoder
    whisper_path: str = ""  # Path to Whisper (optional, for audio)
    humo_variant: str = "17B"  # "17B" or "1.7B"
    num_frames: int = 97  # Number of frames (97 = ~3.9s at 25fps)
    fps: int = 25  # Output framerate
    height: int = 720  # Video height (multiple of 16)
    width: int = 1280  # Video width (multiple of 16)
    guidance_scale: float = 5.0  # Text guidance (scale_t)
    audio_scale: float = 0.0  # Audio guidance (scale_a), 0 = T2V mode
    num_inference_steps: int = 50  # Diffusion steps
    offload_mode: str = "model"  # none, model, sequential
    output_path: str = "wan_output.mp4"  # Output video path


@dataclass
class RuntimeConfig:
    """Unified runtime configuration composing all sub-configs.

    This replaces the old 158-field flat RuntimeConfig. Instead of flat fields
    like `flux2_model_path`, access via `flux2.model_path`.

    Sub-configs are the canonical config.py dataclasses -- same ones used by
    Config.from_toml(). Adding a new parameter only requires:
    1. Add field to the sub-config dataclass
    2. Add to config.toml
    That's it. No manual mapping needed.
    """

    # Top-level settings (not pipeline-specific)
    default_pipeline: str = "none"  # none, z-image, qwen-image, flux2, ltx2
    model_type: str = "zimage"  # zimage, qwenimage-layered, qwenimage-t2i, qwenimage-edit, ltx2, wan
    model_path: str = ""  # Z-Image model path (legacy, prefer zimage.model_path)
    text_encoder_path: str | None = None
    templates_dir: str | None = None

    # Server
    host: str = "127.0.0.1"
    port: int = 7860
    debug: bool = False

    # Composed sub-configs (reuse config.py dataclasses directly)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    rewriter: RewriterConfig = field(default_factory=RewriterConfig)
    quant: PipelineQuantConfig = field(default_factory=PipelineQuantConfig)
    api: APIConfig = field(default_factory=APIConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    pytorch: PyTorchConfig = field(default_factory=PyTorchConfig)

    # Pipeline-specific configs
    flux2: Flux2Config = field(default_factory=Flux2Config)
    ltx2: LTX2Config = field(default_factory=LTX2Config)
    zimage: ZImageConfig = field(default_factory=ZImageConfig)
    qwen_image: QwenImageConfig = field(default_factory=QwenImageConfig)
    wan: WanConfig = field(default_factory=WanConfig)

    # Feature configs
    dype: DyPEConfig = field(default_factory=DyPEConfig)
    slg: SLGConfig = field(default_factory=SLGConfig)
    fmtt: FMTTConfig = field(default_factory=FMTTConfig)
    fbcache: FBCacheRuntimeConfig = field(default_factory=FBCacheRuntimeConfig)

    # Mutable state set at runtime (not from config files)
    config_path: str | None = None  # Path to config file used
    current_profile: str | None = None  # Active config profile

    # -----------------------------------------------------------------------
    # Convenience properties for backward compatibility during migration.
    # These allow code to access config via the old flat-field pattern while
    # we migrate server.py to use sub-configs directly. Remove once Phase 3
    # (router decomposition) is complete.
    # -----------------------------------------------------------------------

    @property
    def encoder_device(self) -> str:
        return self.encoder.device

    @encoder_device.setter
    def encoder_device(self, value: str) -> None:
        self.encoder.device = value

    @property
    def dit_device(self) -> str:
        return self.optimization.dit_device

    @dit_device.setter
    def dit_device(self, value: str) -> None:
        self.optimization.dit_device = value

    @property
    def vae_device(self) -> str:
        return self.optimization.vae_device

    @vae_device.setter
    def vae_device(self, value: str) -> None:
        self.optimization.vae_device = value

    @property
    def dtype(self) -> str:
        return self.encoder.dtype

    @dtype.setter
    def dtype(self, value: str) -> None:
        self.encoder.dtype = value

    @property
    def hidden_layer(self) -> int:
        return self.encoder.hidden_layer

    @hidden_layer.setter
    def hidden_layer(self, value: int) -> None:
        self.encoder.hidden_layer = value

    @property
    def flash_attn(self) -> bool:
        return self.optimization.flash_attn

    @flash_attn.setter
    def flash_attn(self, value: bool) -> None:
        self.optimization.flash_attn = value

    @property
    def compile(self) -> bool:
        return self.optimization.compile

    @compile.setter
    def compile(self, value: bool) -> None:
        self.optimization.compile = value

    @property
    def compile_mode(self) -> str:
        return self.optimization.compile_mode

    @compile_mode.setter
    def compile_mode(self, value: str) -> None:
        self.optimization.compile_mode = value

    @property
    def cpu_offload(self) -> bool:
        return self.optimization.cpu_offload

    @cpu_offload.setter
    def cpu_offload(self, value: bool) -> None:
        self.optimization.cpu_offload = value

    @property
    def shift(self) -> float:
        return self.scheduler.shift

    @shift.setter
    def shift(self, value: float) -> None:
        self.scheduler.shift = value

    @property
    def shift_terminal(self) -> float | None:
        return self.scheduler.shift_terminal

    @shift_terminal.setter
    def shift_terminal(self, value: float | None) -> None:
        self.scheduler.shift_terminal = value

    @property
    def steps(self) -> int:
        return self.generation.num_inference_steps

    @steps.setter
    def steps(self, value: int) -> None:
        self.generation.num_inference_steps = value

    @property
    def guidance_scale(self) -> float:
        return self.generation.guidance_scale

    @guidance_scale.setter
    def guidance_scale(self, value: float) -> None:
        self.generation.guidance_scale = value

    @property
    def height(self) -> int:
        return self.generation.height

    @height.setter
    def height(self, value: int) -> None:
        self.generation.height = value

    @property
    def width(self) -> int:
        return self.generation.width

    @width.setter
    def width(self, value: int) -> None:
        self.generation.width = value

    @property
    def seed(self) -> int | None:
        return self.generation.seed

    @seed.setter
    def seed(self, value: int | None) -> None:
        self.generation.seed = value

    @property
    def negative_prompt(self) -> str | None:
        return self.generation.negative_prompt

    @negative_prompt.setter
    def negative_prompt(self, value: str | None) -> None:
        self.generation.negative_prompt = value

    @property
    def enable_thinking(self) -> bool:
        return self.generation.enable_thinking

    @property
    def default_template(self) -> str | None:
        return self.generation.default_template

    @property
    def cfg_normalization(self) -> float:
        return self.generation.cfg_normalization

    @property
    def cfg_truncation(self) -> float:
        return self.generation.cfg_truncation

    @property
    def attention_backend(self) -> str | None:
        return self.pytorch.attention_backend

    @property
    def long_prompt_mode(self) -> str:
        return self.pytorch.long_prompt_mode

    @property
    def tiled_vae(self) -> bool:
        return self.pytorch.tiled_vae

    @property
    def lora_paths(self) -> list[str]:
        return self.lora.paths

    @property
    def lora_scales(self) -> list[float]:
        return self.lora.scales

    @property
    def api_url(self) -> str | None:
        url = self.api.url
        return url if url else None

    @property
    def api_model(self) -> str:
        return self.api.model

    # Qwen-Image backward-compat properties
    @property
    def qwen_image_model_path(self) -> str:
        return self.qwen_image.model_path

    @qwen_image_model_path.setter
    def qwen_image_model_path(self, value: str) -> None:
        self.qwen_image.model_path = value

    @property
    def qwen_image_edit_model_path(self) -> str:
        return self.qwen_image.edit_model_path

    @property
    def qwen_image_cpu_offload(self) -> bool:
        return self.qwen_image.cpu_offload

    @property
    def qwen_image_layer_num(self) -> int:
        return self.qwen_image.layer_num

    @property
    def qwen_image_cfg_scale(self) -> float:
        return self.qwen_image.cfg_scale

    @property
    def qwen_image_resolution(self) -> int | None:
        return getattr(self.qwen_image, "resolution", None)

    @property
    def qwen_image_steps(self) -> int | None:
        return getattr(self.qwen_image, "num_inference_steps", None)

    @property
    def qwen_image_quantize_text_encoder(self) -> str:
        return getattr(self.qwen_image, "quantize_text_encoder", "none")

    @property
    def qwen_image_quantize_transformer(self) -> str | None:
        return getattr(self.qwen_image, "quantize_transformer", None)

    # FLUX.2 backward-compat properties
    @property
    def flux2_model_path(self) -> str | None:
        return self.flux2.model_path or None

    @property
    def flux2_vae_path(self) -> str | None:
        return self.flux2.vae_path

    @property
    def flux2_encoder_path(self) -> str | None:
        return self.flux2.encoder_path

    @property
    def flux2_model_name(self) -> str:
        return self.flux2.default_model

    @property
    def flux2_compile(self) -> bool:
        return getattr(self.flux2, "compile", False)

    @property
    def flux2_compile_vae(self) -> bool:
        return getattr(self.flux2, "compile_vae", False)

    @property
    def flux2_compile_mode(self) -> str:
        return getattr(self.flux2, "compile_mode", "default")

    @property
    def flux2_compile_dynamic(self) -> bool:
        return getattr(self.flux2, "compile_dynamic", False)

    @property
    def flux2_block_offload(self) -> bool:
        return self.flux2.block_offload

    @property
    def flux2_offload_between_stages(self) -> bool:
        return getattr(self.flux2, "offload_between_stages", True)

    @property
    def flux2_num_steps(self) -> int | None:
        return self.flux2.default_steps

    @property
    def flux2_guidance(self) -> float | None:
        return self.flux2.default_guidance

    # LTX-2 backward-compat properties
    @property
    def ltx2_model_path(self) -> str:
        return self.ltx2.model_path

    @ltx2_model_path.setter
    def ltx2_model_path(self, value: str) -> None:
        self.ltx2.model_path = value

    # Z-Image backward-compat properties
    @property
    def zimage_model_path(self) -> str | None:
        return self.zimage.model_path or None

    @zimage_model_path.setter
    def zimage_model_path(self, value: str) -> None:
        self.zimage.model_path = value

    @property
    def zimage_variant(self) -> str:
        return self.zimage.variant

    @zimage_variant.setter
    def zimage_variant(self, value: str) -> None:
        self.zimage.variant = value

    # Rewriter backward-compat properties
    @property
    def rewriter_use_api(self) -> bool:
        return self.rewriter.use_api

    @property
    def rewriter_api_url(self) -> str:
        return self.rewriter.api_url

    @property
    def rewriter_api_model(self) -> str:
        return self.rewriter.api_model

    @property
    def rewriter_temperature(self) -> float:
        return self.rewriter.temperature

    @property
    def rewriter_top_p(self) -> float:
        return self.rewriter.top_p

    @property
    def rewriter_top_k(self) -> int:
        return self.rewriter.top_k

    @property
    def rewriter_min_p(self) -> float:
        return self.rewriter.min_p

    @property
    def rewriter_presence_penalty(self) -> float:
        return self.rewriter.presence_penalty

    @property
    def rewriter_max_tokens(self) -> int:
        return self.rewriter.max_tokens

    # DyPE backward-compat properties
    @property
    def dype_enabled(self) -> bool:
        return self.dype.enabled

    @dype_enabled.setter
    def dype_enabled(self, value: bool) -> None:
        self.dype.enabled = value

    @property
    def dype_method(self) -> str:
        return self.dype.method

    @property
    def dype_scale(self) -> float:
        return self.dype.dype_scale

    # SLG backward-compat properties
    @property
    def slg_scale(self) -> float:
        return self.slg.scale

    @property
    def slg_layers(self) -> list[int] | None:
        return self.slg.layers

    # Additional scheduler backward-compat
    @property
    def dynamic_shift(self) -> bool:
        return self.scheduler.dynamic_shift

    @dynamic_shift.setter
    def dynamic_shift(self, value: bool) -> None:
        self.scheduler.dynamic_shift = value

    @property
    def d_noise(self) -> float:
        return self.scheduler.d_noise

    @d_noise.setter
    def d_noise(self, value: float) -> None:
        self.scheduler.d_noise = value

    # Additional generation backward-compat
    @property
    def cfg_norm_mode(self) -> str:
        return self.generation.cfg_norm_mode

    @cfg_norm_mode.setter
    def cfg_norm_mode(self, value: str) -> None:
        self.generation.cfg_norm_mode = value

    @property
    def system_prompt(self) -> str | None:
        return self.generation.system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str | None) -> None:
        self.generation.system_prompt = value

    @property
    def thinking_content(self) -> str | None:
        return self.generation.thinking_content

    @thinking_content.setter
    def thinking_content(self, value: str | None) -> None:
        self.generation.thinking_content = value

    @property
    def assistant_content(self) -> str | None:
        return self.generation.assistant_content

    @assistant_content.setter
    def assistant_content(self, value: str | None) -> None:
        self.generation.assistant_content = value

    # Additional encoder backward-compat
    @property
    def quantization(self) -> str:
        return self.encoder.quantization

    @quantization.setter
    def quantization(self, value: str) -> None:
        self.encoder.quantization = value

    @property
    def layer_weights(self) -> dict[int, float] | None:
        return self.encoder.layer_weights

    @layer_weights.setter
    def layer_weights(self, value: dict[int, float] | None) -> None:
        self.encoder.layer_weights = value

    # Additional pytorch backward-compat
    @property
    def embedding_cache(self) -> bool:
        return self.pytorch.embedding_cache

    @property
    def cache_size(self) -> int:
        return self.pytorch.cache_size

    @property
    def tile_size(self) -> int:
        return self.pytorch.tile_size

    @property
    def tile_overlap(self) -> int:
        return self.pytorch.tile_overlap

    @property
    def use_custom_scheduler(self) -> bool:
        return self.pytorch.use_custom_scheduler

    # Additional api backward-compat
    @property
    def local_encoder(self) -> bool:
        return self.api.local_encoder

    # Additional logging backward-compat
    @property
    def log_dir(self) -> str | None:
        d = self.logging.log_dir
        return d if d else None

    @log_dir.setter
    def log_dir(self, value: str | None) -> None:
        self.logging.log_dir = value or ""

    # Additional LTX-2 backward-compat
    @property
    def ltx2_num_frames(self) -> int:
        return self.ltx2.num_frames

    @property
    def ltx2_fps(self) -> int:
        return self.ltx2.fps

    @property
    def ltx2_guidance_scale(self) -> float:
        return self.ltx2.guidance_scale

    @property
    def ltx2_steps(self) -> int | None:
        return getattr(self.ltx2, "num_inference_steps", None)

    @property
    def ltx2_encoder_model_id(self) -> str:
        return getattr(self.ltx2, "encoder_model_id", "models/LTX-2/text_encoder")

    @property
    def ltx2_lora_path(self) -> str:
        return getattr(self.ltx2, "lora_path", "")

    @property
    def ltx2_lora_scale(self) -> float:
        return getattr(self.ltx2, "lora_scale", 1.0)

    @property
    def ltx2_audio(self) -> bool:
        return getattr(self.ltx2, "audio_enabled", False)

    @property
    def ltx2_save_embeddings(self) -> str | None:
        return self.ltx2.save_embeddings

    @ltx2_save_embeddings.setter
    def ltx2_save_embeddings(self, value: str | None) -> None:
        self.ltx2.save_embeddings = value

    @property
    def ltx2_load_embeddings(self) -> str | None:
        return self.ltx2.load_embeddings

    @ltx2_load_embeddings.setter
    def ltx2_load_embeddings(self, value: str | None) -> None:
        self.ltx2.load_embeddings = value

    @property
    def ltx2_text_encoder_device(self) -> str:
        return self.ltx2.text_encoder_device

    @property
    def ltx2_transformer_device(self) -> str:
        return self.ltx2.transformer_device

    @property
    def ltx2_vae_device(self) -> str:
        return self.ltx2.vae_device

    @property
    def ltx2_quantize(self) -> str:
        return self.ltx2.quantize

    @property
    def ltx2_skip_cleanup(self) -> bool:
        return self.ltx2.skip_cleanup

    @property
    def ltx2_gemma_variant(self) -> str:
        return self.ltx2.gemma_variant

    @property
    def ltx2_output_path(self) -> str:
        return self.ltx2.output_path

    # Additional FLUX.2 backward-compat
    @property
    def flux2_seed(self) -> int | None:
        return getattr(self.flux2, "seed", None)

    @property
    def flux2_output_path(self) -> str:
        return getattr(self.flux2, "output_path", "flux2_output.png")

    @property
    def flux2_input_images(self) -> list[str] | None:
        return getattr(self.flux2, "input_images", None)

    # Additional Qwen-Image backward-compat
    @property
    def qwen_image_edit_only(self) -> bool:
        return getattr(self.qwen_image, "edit_only", False)

    @property
    def qwen_image_offload_type(self) -> str:
        return getattr(self.qwen_image, "offload_type", "pipeline")

    # Additional DyPE backward-compat
    @property
    def dype_exponent(self) -> float:
        return self.dype.dype_exponent

    @property
    def dype_start_sigma(self) -> float:
        return self.dype.dype_start_sigma

    @property
    def dype_base_shift(self) -> float:
        return getattr(self.dype, "base_shift", 0.5)

    @property
    def dype_max_shift(self) -> float:
        return getattr(self.dype, "max_shift", 1.15)

    @property
    def dype_base_resolution(self) -> int:
        return getattr(self.dype, "base_resolution", 1024)

    @property
    def dype_anisotropic(self) -> bool:
        return getattr(self.dype, "anisotropic", False)

    @property
    def dype_multipass(self) -> str:
        return getattr(self.dype, "multipass", "single")

    @property
    def dype_pass2_strength(self) -> float:
        return getattr(self.dype, "pass2_strength", 0.5)

    @property
    def dype_pass3_strength(self) -> float:
        return getattr(self.dype, "pass3_strength", 0.4)

    @property
    def dype_frequency_modulation(self) -> bool:
        return getattr(self.dype, "frequency_modulation", False)

    # FBCache backward-compat (note: the old field was just `fbcache`)
    @property
    def fbcache_enabled(self) -> bool:
        return self.fbcache.enabled

    @property
    def fbcache_threshold(self) -> float | None:
        return getattr(self.fbcache, "middle_threshold", None)

    @property
    def fbcache_log(self) -> bool:
        return getattr(self.fbcache, "log_residuals", False)

    # Additional rewriter backward-compat
    @property
    def rewriter_timeout(self) -> float:
        return self.rewriter.timeout

    # -----------------------------------------------------------------------
    # Methods
    # -----------------------------------------------------------------------

    def get_dtype(self) -> "torch.dtype":
        import torch

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.dtype, torch.bfloat16)

    def resolve_device(self, device: str) -> str:
        """Resolve 'auto' to actual device."""
        import torch

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
        """Return variant-specific defaults for Qwen-Image models."""
        defaults = {
            "qwenimage-t2i": {
                "steps": 40,
                "resolution": 1024,
                "quantize_transformer": "fp8-weight-only",
                "guidance_scale": 4.0,
            },
            "qwenimage-edit": {
                "steps": 25,
                "resolution": 640,
                "quantize_transformer": "fp8-dynamic",
                "guidance_scale": 4.0,
            },
            "qwenimage-layered": {
                "steps": 50,
                "resolution": 640,
                "quantize_transformer": "fp8-dynamic",
                "guidance_scale": 4.0,
            },
        }
        return defaults.get(self.model_type, {})

    def get_qwen_image_steps(self) -> int:
        """Get effective steps, using variant default if not explicitly set."""
        if self.qwen_image_steps is not None:
            return self.qwen_image_steps
        return self.get_qwen_variant_defaults().get("steps", 40)

    def get_qwen_image_resolution(self) -> int:
        """Get effective resolution, using variant default if not explicitly set."""
        if self.qwen_image_resolution is not None:
            return self.qwen_image_resolution
        return self.get_qwen_variant_defaults().get("resolution", 1024)

    def get_qwen_image_quantize_transformer(self) -> str:
        """Get effective transformer quantization."""
        if self.qwen_image_quantize_transformer is not None:
            return self.qwen_image_quantize_transformer
        return self.get_qwen_variant_defaults().get("quantize_transformer", "none")

    def get_pipeline_quant_config(self, pipeline: str) -> PipelineQuantConfig:
        """Resolve effective quantization for a pipeline."""
        enc_default = self.quant.encoder.method
        tf_default = self.quant.transformer.method
        vae_default = self.quant.vae.method
        g = self.quant.encoder.granularity  # same for all components

        def _resolve(override: str | None, default: str) -> ComponentQuantConfig:
            method = override if override is not None else default
            return ComponentQuantConfig(method=method, granularity=g)

        if pipeline == "flux2":
            return PipelineQuantConfig(
                encoder=_resolve(getattr(self.flux2, "quant_encoder", None), enc_default),
                transformer=_resolve(getattr(self.flux2, "quant_transformer", None), tf_default),
                vae=_resolve(getattr(self.flux2, "quant_vae", None), vae_default),
            )
        elif pipeline == "ltx2":
            return PipelineQuantConfig(
                encoder=_resolve(getattr(self.ltx2, "quant_encoder", None), enc_default),
                transformer=_resolve(getattr(self.ltx2, "quant_transformer", None), tf_default),
                vae=_resolve(getattr(self.ltx2, "quant_vae", None), vae_default),
            )
        elif pipeline == "z_image":
            return PipelineQuantConfig(
                encoder=_resolve(getattr(self.zimage, "quant_encoder", None), enc_default),
                transformer=_resolve(getattr(self.zimage, "quant_transformer", None), "none"),
                vae=ComponentQuantConfig(method="none", granularity=g),
            )
        elif pipeline == "qwen_image":
            return PipelineQuantConfig(
                encoder=_resolve(getattr(self.qwen_image, "quant_encoder", None), enc_default),
                transformer=_resolve(getattr(self.qwen_image, "quant_transformer", None), tf_default),
                vae=_resolve(getattr(self.qwen_image, "quant_vae", None), vae_default),
            )
        else:
            return PipelineQuantConfig(
                encoder=ComponentQuantConfig(method=enc_default, granularity=g),
                transformer=ComponentQuantConfig(method=tf_default, granularity=g),
                vae=ComponentQuantConfig(method=vae_default, granularity=g),
            )

    def to_dict(self) -> dict:
        """Serialize to dictionary for API responses."""
        from dataclasses import fields as dc_fields, asdict

        result: dict[str, Any] = {}
        for f in dc_fields(self):
            value = getattr(self, f.name)
            if hasattr(value, "__dataclass_fields__"):
                # Nested dataclass - flatten with prefix for backward compat
                result[f.name] = asdict(value)
            else:
                result[f.name] = value
        return result

    @classmethod
    def from_toml_config(cls, toml_config: "Config") -> "RuntimeConfig":
        """Create RuntimeConfig directly from a parsed Config (TOML).

        This is the simplified replacement for the old 280-line manual mapping.
        Since RuntimeConfig now composes the same dataclasses as Config, we just
        assign them directly.
        """
        rc = cls()
        rc.default_pipeline = toml_config.default_pipeline or rc.default_pipeline
        rc.model_path = toml_config.model_path or rc.model_path
        rc.templates_dir = toml_config.templates_dir or rc.templates_dir

        # Direct sub-config assignment (this is the whole point of the refactor)
        rc.encoder = toml_config.encoder
        rc.generation = toml_config.generation
        rc.optimization = toml_config.optimization
        rc.scheduler = toml_config.scheduler
        rc.api = toml_config.api
        rc.lora = toml_config.lora
        rc.pytorch = toml_config.pytorch
        rc.rewriter = toml_config.rewriter
        rc.logging = toml_config.logging
        rc.zimage = toml_config.zimage
        rc.qwen_image = toml_config.qwen_image
        rc.ltx2 = toml_config.ltx2
        rc.flux2 = toml_config.flux2
        rc.dype = toml_config.dype
        rc.slg = toml_config.slg
        rc.fmtt = toml_config.fmtt
        rc.fbcache = toml_config.fbcache

        rc.wan = toml_config.wan

        return rc


@dataclass
class Config:
    """Complete configuration for Z-Image, Qwen-Image, and LTX-2 generation."""

    # Startup pipeline selection
    default_pipeline: str = "none"  # none, z-image, qwen-image, flux2, ltx2

    model_path: str = ""
    templates_dir: str | None = None
    presets_dir: str = "presets"  # Directory containing generation presets

    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    api: APIConfig = field(default_factory=APIConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    pytorch: PyTorchConfig = field(default_factory=PyTorchConfig)
    rewriter: RewriterConfig = field(default_factory=RewriterConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    zimage: ZImageConfig = field(default_factory=ZImageConfig)
    qwen_image: QwenImageConfig = field(default_factory=QwenImageConfig)
    ltx2: LTX2Config = field(default_factory=LTX2Config)
    flux2: Flux2Config = field(default_factory=Flux2Config)
    dype: DyPEConfig = field(default_factory=DyPEConfig)
    slg: SLGConfig = field(default_factory=SLGConfig)
    fmtt: FMTTConfig = field(default_factory=FMTTConfig)
    fbcache: FBCacheRuntimeConfig = field(default_factory=FBCacheRuntimeConfig)
    wan: WanConfig = field(default_factory=WanConfig)
    enhancement: EnhancementConfig = field(default_factory=EnhancementConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """Create config from dictionary."""
        encoder_data = data.pop("encoder", {})
        pipeline_data = data.pop("pipeline", {})
        generation_data = data.pop("generation", {})
        optimization_data = data.pop("optimization", {})
        scheduler_data = data.pop("scheduler", {})
        api_data = data.pop("api", {})
        lora_data = data.pop("lora", {})
        pytorch_data = data.pop("pytorch", {})
        rewriter_data = data.pop("rewriter", {})
        logging_data = data.pop("logging", {})
        zimage_data = data.pop("zimage", {})
        qwen_image_data = data.pop("qwen_image", {})
        ltx2_data = data.pop("ltx2", {})
        flux2_data = data.pop("flux2", {})
        dype_data = data.pop("dype", {})
        slg_data = data.pop("slg", {})
        fmtt_data = data.pop("fmtt", {})
        fbcache_data = data.pop("fbcache", {})
        wan_data = data.pop("wan", {})
        enhancement_data = data.pop("enhancement", {})

        return cls(
            default_pipeline=data.get("default_pipeline", "none"),
            model_path=data.get("model_path", ""),
            templates_dir=data.get("templates_dir"),
            presets_dir=data.get("presets_dir", "presets"),
            encoder=EncoderConfig(**encoder_data),
            pipeline=PipelineConfig(**pipeline_data),
            generation=GenerationConfig(**generation_data),
            optimization=OptimizationConfig(**optimization_data),
            scheduler=SchedulerConfig(**scheduler_data),
            api=APIConfig(**api_data),
            lora=LoRAConfig(**lora_data),
            pytorch=PyTorchConfig(**pytorch_data),
            rewriter=RewriterConfig(**rewriter_data),
            logging=LoggingConfig(**logging_data),
            zimage=ZImageConfig(**zimage_data),
            qwen_image=QwenImageConfig(**qwen_image_data),
            ltx2=LTX2Config(**ltx2_data),
            flux2=Flux2Config(**flux2_data),
            dype=DyPEConfig(**dype_data),
            slg=SLGConfig(**slg_data),
            fmtt=FMTTConfig(**fmtt_data),
            fbcache=FBCacheRuntimeConfig(**fbcache_data),
            wan=WanConfig(**wan_data),
            enhancement=EnhancementConfig(**enhancement_data),
        )

    @classmethod
    def from_toml(cls, path: str | Path, profile: str | None = None) -> "Config":
        """
        Load config from TOML file.

        Supports two config formats:
        1. Flat config (no profiles): [encoder], [pipeline], etc. at top level
        2. Profile-based config: [profile_name], [profile_name.encoder], etc.

        Args:
            path: Path to TOML config file
            profile: Profile name to load. If None, auto-detects:
                     - Uses flat config if [encoder] exists at top level
                     - Falls back to "default" profile for legacy configs

        Returns:
            Loaded Config

        Example flat TOML (recommended):
            model_path = "/path/to/model"

            [encoder]
            quantization = "int8"

            [pipeline]
            device = "cuda"
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

        # Auto-detect config format if no profile specified
        if profile is None:
            # Check for flat config: top-level sections like [encoder], [pipeline]
            flat_sections = {"encoder", "pipeline", "generation", "scheduler", "optimization"}
            if flat_sections & set(data.keys()):
                # Flat config detected - use top-level data directly
                logger.info("Loaded flat config (no profile)")
                return cls.from_dict(data)
            else:
                # Legacy profile-based config - default to "default" profile
                profile = "default"

        # Profile-based loading
        if profile not in data:
            available = list(data.keys())
            raise KeyError(f"Profile '{profile}' not found in config. Available: {available}")

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
                "dtype": self.encoder.dtype,
                "quantization": self.encoder.quantization,
                "cpu_offload": self.encoder.cpu_offload,
                "trust_remote_code": self.encoder.trust_remote_code,
                "max_length": self.encoder.max_length,
                "hidden_layer": self.encoder.hidden_layer,
            },
            "pipeline": {
                "device": self.pipeline.device,
                "dtype": self.pipeline.dtype,
                "enable_model_cpu_offload": self.pipeline.enable_model_cpu_offload,
                "enable_sequential_cpu_offload": self.pipeline.enable_sequential_cpu_offload,
            },
            "generation": {
                "height": self.generation.height,
                "width": self.generation.width,
                "num_inference_steps": self.generation.num_inference_steps,
                "guidance_scale": self.generation.guidance_scale,
                "cfg_normalization": self.generation.cfg_normalization,
                "cfg_truncation": self.generation.cfg_truncation,
                "cfg_norm_mode": self.generation.cfg_norm_mode,
                "enable_thinking": self.generation.enable_thinking,
                "default_template": self.generation.default_template,
            },
            "optimization": {
                "flash_attn": self.optimization.flash_attn,
                "compile": self.optimization.compile,
                "compile_mode": self.optimization.compile_mode,
                "cpu_offload": self.optimization.cpu_offload,
            },
            "scheduler": {
                "shift": self.scheduler.shift,
                "shift_terminal": self.scheduler.shift_terminal,
            },
            "api": {
                "url": self.api.url,
                "model": self.api.model,
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
                "timeout": self.rewriter.timeout,
            },
            "logging": {
                "enabled": self.logging.enabled,
                "log_dir": self.logging.log_dir,
                "log_level": self.logging.log_level,
                "json_format": self.logging.json_format,
                "max_bytes": self.logging.max_bytes,
                "backup_count": self.logging.backup_count,
                "log_requests": self.logging.log_requests,
                "log_generation_params": self.logging.log_generation_params,
            },
            "zimage": {
                "model_path": self.zimage.model_path,
                "text_encoder_path": self.zimage.text_encoder_path,
                "variant": self.zimage.variant,
                "default_preset": self.zimage.default_preset,
            },
            "qwen_image": {
                "model_path": self.qwen_image.model_path,
                "edit_model_path": self.qwen_image.edit_model_path,
                "device": self.qwen_image.device,
                "text_encoder_device": self.qwen_image.text_encoder_device,
                "dtype": self.qwen_image.dtype,
                "cpu_offload": self.qwen_image.cpu_offload,
                "quantize_text_encoder": self.qwen_image.quantize_text_encoder,
                "quantize_transformer": self.qwen_image.quantize_transformer,
                "quantize_vae": self.qwen_image.quantize_vae,
                "offload_type": self.qwen_image.offload_type,
                "num_blocks_per_group": self.qwen_image.num_blocks_per_group,
                "num_inference_steps": self.qwen_image.num_inference_steps,
                "cfg_scale": self.qwen_image.cfg_scale,
                "layer_num": self.qwen_image.layer_num,
                "resolution": self.qwen_image.resolution,
                "shift": self.qwen_image.shift,
            },
            "ltx2": self.ltx2.to_dict(),
            "dype": self.dype.to_dict(),
            "slg": self.slg.to_dict(),
            "fmtt": self.fmtt.to_dict(),
            "enhancement": self.enhancement.to_dict(),
        }


# Preset configurations
PRESETS = {
    "default": Config(
        encoder=EncoderConfig(device="auto", dtype="bfloat16"),
        pipeline=PipelineConfig(device="auto", dtype="bfloat16"),
    ),
    "low_vram": Config(
        encoder=EncoderConfig(
            device="cuda",
            dtype="bfloat16",
            quantization="int8",
            cpu_offload=True,
        ),
        pipeline=PipelineConfig(
            device="cuda",
            dtype="bfloat16",
            enable_model_cpu_offload=True,
        ),
    ),
    "cpu_only": Config(
        encoder=EncoderConfig(device="cpu", dtype="float32"),
        pipeline=PipelineConfig(device="cpu", dtype="float32"),
    ),
}


def get_preset(name: str) -> Config:
    """Get a preset configuration by name."""
    if name not in PRESETS:
        raise KeyError(f"Unknown preset: {name}. Available: {list(PRESETS.keys())}")
    return PRESETS[name]


def load_config(
    path: str | Path | None = None,
    profile: str | None = None,
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
        profile: Profile name within TOML file. If None, auto-detects:
                 flat config (recommended) or falls back to "default" profile
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
