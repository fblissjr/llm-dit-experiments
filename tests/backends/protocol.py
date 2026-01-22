"""
Backend protocol for portable LTX-2 tests.

Last Updated: 2026-01-19

Defines the interface that both llm_dit and ltx2 backends must implement.
This enables writing tests that work with either implementation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

"""
Backend protocol for portable LTX-2 tests.

Last Updated: 2026-01-20

Defines the interface that both llm_dit and ltx2 backends must implement.
This enables writing tests that work with either implementation.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


# --- NEW: JSON Serializer Helper ---
def json_serializer(obj: Any) -> Any:
    """JSON serializer for objects not serializable by default json code."""
    if isinstance(obj, torch.dtype):
        return str(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
        return list(obj.shape)
    if isinstance(
        obj,
        (
            np.int_,
            np.intc,
            np.intp,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ),
    ):
        return int(obj)
    if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    raise TypeError(f"Type {type(obj)} not serializable")


@dataclass
class GenerationConfig:
    """Configuration for video generation.

    These are the canonical parameters matching official LTX-2 defaults.
    Both backends must interpret these identically.

    Reference: coderef/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py
    """

    # Video dimensions
    num_frames: int = 121  # Must be 8k+1 (e.g., 9, 17, 25, 33, ..., 121)
    height: int = 512  # Divisible by 32
    width: int = 768  # Divisible by 32
    frame_rate: float = 24.0

    # Inference parameters
    num_inference_steps: int = 40
    guidance_scale: float = 4.0  # CFG scale
    seed: int = 10  # Default LTX-2 seed

    # Model configuration
    fp8: bool = False  # Use FP8 quantization for transformer
    dtype: torch.dtype = field(default=torch.bfloat16)

    # Conditioning (for I2V)
    conditioning_image: Optional[torch.Tensor] = None
    conditioning_frame_idx: int = 0
    conditioning_strength: float = 0.8

    # Debug options
    debug_trace: bool = False  # Enable detailed embedding/connector diagnostics

    def validate(self) -> None:
        """Validate configuration parameters."""
        # Frame count must be 8k+1
        if (self.num_frames - 1) % 8 != 0:
            raise ValueError(f"num_frames must be 8k+1 (got {self.num_frames})")
        # Dimensions must be divisible by 32
        if self.height % 32 != 0:
            raise ValueError(f"height must be divisible by 32 (got {self.height})")
        if self.width % 32 != 0:
            raise ValueError(f"width must be divisible by 32 (got {self.width})")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict, safe for JSON logging."""
        d = asdict(self)
        # Remove raw tensor from logs
        if "conditioning_image" in d:
            del d["conditioning_image"]
        return d


# =============================================================================
# Standard Configurations (Single Source of Truth)
# =============================================================================
# These are the canonical configs used by all tests and backends.
# DO NOT duplicate these elsewhere - import from here.

# Reference: coderef/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py

REFERENCE_CONFIG = GenerationConfig(
    # Official LTX-2 reference parameters for 1:1 comparison
    num_frames=121,  # 5 seconds at 24fps (15 latent frames)
    height=512,
    width=768,
    num_inference_steps=40,
    guidance_scale=4.0,
    seed=10,  # Official default seed
    fp8=False,
)

SHORT_CONFIG = GenerationConfig(
    # Reasonable quality, faster iteration (~2min on 24GB GPU)
    num_frames=33,  # ~1.4 seconds (4 latent frames)
    height=512,
    width=768,
    num_inference_steps=30,
    guidance_scale=3.0,
    seed=10,
    fp8=False,
)

SMOKE_CONFIG = GenerationConfig(
    # Reasonable quality, faster iteration (~2min on 24GB GPU)
    num_frames=33,  # ~1.4 seconds (4 latent frames)
    height=512,
    width=768,
    num_inference_steps=30,
    guidance_scale=3.0,
    seed=10,
    fp8=True,
)

# Memory estimates (RTX 4090 24GB with FP8 transformer)
CONFIG_MEMORY_ESTIMATES = {
    "smoke": {"vram_gb": 14, "time_estimate": "~30s"},
    "short": {"vram_gb": 16, "time_estimate": "~2min"},
    "reference": {"vram_gb": 20, "time_estimate": "~10min"},
}


@dataclass
class GenerationInputs:
    """Complete record of all generation inputs for reproducibility.

    Captures everything needed to reproduce a generation exactly.
    Logged at the start of each test run.
    """

    # Prompt
    prompt: str = ""
    negative_prompt: str = ""

    # Video config
    num_frames: int = 0
    height: int = 0
    width: int = 0
    frame_rate: float = 24.0

    # Inference config
    num_inference_steps: int = 0
    guidance_scale: float = 0.0
    seed: int = 0

    # Model paths
    transformer_path: str = ""
    text_encoder_path: str = ""
    vae_path: str = ""

    # Model config
    transformer_dtype: str = "bfloat16"
    transformer_quantization: str = "fp8-quanto"
    text_encoder_dtype: str = "bfloat16"
    text_encoder_quantization: str = "8bit"
    vae_dtype: str = "bfloat16"

    # Scheduler config
    base_shift: float = 0.95
    max_shift: float = 2.05
    terminal_sigma: float = 0.1

    # Conditioning (I2V)
    conditioning_image_path: Optional[str] = None
    conditioning_frame_idx: int = 0
    conditioning_strength: float = 0.0

    # Upsampling
    upsampling_enabled: bool = False
    upsampling_scale: float = 1.0

    # Tensor shapes (filled during generation)
    text_embedding_shape: Optional[tuple] = None  # e.g., (1, 256, 3840)
    latent_shape: Optional[tuple] = None  # e.g., (1, 128, 15, 12, 16)
    noise_shape: Optional[tuple] = None
    position_indices_shape: Optional[tuple] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "prompt": {
                "text": self.prompt,
                "negative": self.negative_prompt,
            },
            "video": {
                "num_frames": self.num_frames,
                "height": self.height,
                "width": self.width,
                "frame_rate": self.frame_rate,
            },
            "inference": {
                "steps": self.num_inference_steps,
                "guidance_scale": self.guidance_scale,
                "seed": self.seed,
            },
            "models": {
                "transformer": {
                    "path": self.transformer_path,
                    "dtype": self.transformer_dtype,
                    "quantization": self.transformer_quantization,
                },
                "text_encoder": {
                    "path": self.text_encoder_path,
                    "dtype": self.text_encoder_dtype,
                    "quantization": self.text_encoder_quantization,
                },
                "vae": {
                    "path": self.vae_path,
                    "dtype": self.vae_dtype,
                },
            },
            "scheduler": {
                "base_shift": self.base_shift,
                "max_shift": self.max_shift,
                "terminal_sigma": self.terminal_sigma,
            },
            "conditioning": {
                "image_path": self.conditioning_image_path,
                "frame_idx": self.conditioning_frame_idx,
                "strength": self.conditioning_strength,
            },
            "upsampling": {
                "enabled": self.upsampling_enabled,
                "scale": self.upsampling_scale,
            },
            "tensor_shapes": {
                "text_embedding": list(self.text_embedding_shape)
                if self.text_embedding_shape
                else None,
                "latent": list(self.latent_shape) if self.latent_shape else None,
                "noise": list(self.noise_shape) if self.noise_shape else None,
                "position_indices": list(self.position_indices_shape)
                if self.position_indices_shape
                else None,
            },
        }

    def log_summary(self, logger) -> None:
        """Log a formatted summary of all inputs."""
        logger.info("=" * 60)
        logger.info("GENERATION INPUTS")
        logger.info("=" * 60)
        logger.info(f"Prompt: {self.prompt}")
        if self.negative_prompt:
            logger.info(f"Negative: {self.negative_prompt}")
        logger.info("-" * 40)
        logger.info(
            f"Video: {self.num_frames} frames @ {self.height}x{self.width}, {self.frame_rate}fps"
        )
        logger.info(
            f"Inference: {self.num_inference_steps} steps, CFG {self.guidance_scale}, seed {self.seed}"
        )
        logger.info("-" * 40)
        logger.info(f"Transformer: {self.transformer_path}")
        logger.info(f"  dtype={self.transformer_dtype}, quant={self.transformer_quantization}")
        logger.info(f"Text Encoder: {self.text_encoder_path}")
        logger.info(f"  dtype={self.text_encoder_dtype}, quant={self.text_encoder_quantization}")
        logger.info(f"VAE: {self.vae_path}, dtype={self.vae_dtype}")
        logger.info("-" * 40)
        logger.info(
            f"Scheduler: base_shift={self.base_shift}, max_shift={self.max_shift}, terminal={self.terminal_sigma}"
        )
        if self.conditioning_image_path:
            logger.info(
                f"Conditioning: {self.conditioning_image_path} @ frame {self.conditioning_frame_idx}, strength={self.conditioning_strength}"
            )
        if self.upsampling_enabled:
            logger.info(f"Upsampling: {self.upsampling_scale}x")
        # Tensor shapes (populated during generation)
        if any(
            [
                self.text_embedding_shape,
                self.latent_shape,
                self.noise_shape,
                self.position_indices_shape,
            ]
        ):
            logger.info("-" * 40)
            logger.info("Tensor Shapes:")
            if self.text_embedding_shape:
                logger.info(f"  text_embedding: {self.text_embedding_shape}")
            if self.latent_shape:
                logger.info(f"  latent: {self.latent_shape}")
            if self.noise_shape:
                logger.info(f"  noise: {self.noise_shape}")
            if self.position_indices_shape:
                logger.info(f"  position_indices: {self.position_indices_shape}")
        logger.info("=" * 60)


@dataclass
class GenerationStats:
    """Statistics from video generation.

    Used for comparing performance and debugging between implementations.
    """

    # Timing (seconds)
    text_encoder_time: float = 0.0
    transformer_time: float = 0.0
    vae_time: float = 0.0
    total_time: float = 0.0

    # Memory (GB)
    text_encoder_peak_memory: float = 0.0
    transformer_peak_memory: float = 0.0
    vae_peak_memory: float = 0.0

    # Memory before/after
    vram_before_gb: float = 0.0
    vram_after_gb: float = 0.0
    ram_before_gb: float = 0.0
    ram_after_gb: float = 0.0

    # Generation metadata
    actual_num_frames: int = 0
    actual_height: int = 0
    actual_width: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "timing": {
                "text_encoder": self.text_encoder_time,
                "transformer": self.transformer_time,
                "vae": self.vae_time,
                "total": self.total_time,
            },
            "memory_gb": {
                "text_encoder_peak": self.text_encoder_peak_memory,
                "transformer_peak": self.transformer_peak_memory,
                "vae_peak": self.vae_peak_memory,
                "vram_before": self.vram_before_gb,
                "vram_after": self.vram_after_gb,
                "ram_before": self.ram_before_gb,
                "ram_after": self.ram_after_gb,
            },
            "output": {
                "num_frames": self.actual_num_frames,
                "height": self.actual_height,
                "width": self.actual_width,
            },
        }


@dataclass
class GenerationResult:
    """Result from video generation.

    Contains both the generated video and metadata for comparison.
    """

    # Video output
    video: torch.Tensor  # [F, H, W, C] uint8 or [B, C, F, H, W] float for latents
    latents: Optional[torch.Tensor] = None  # [B, C, F, H, W] raw latents before VAE

    # Generation info
    prompt: str = ""
    config: Optional[GenerationConfig] = None
    stats: Optional[GenerationStats] = None
    backend_name: str = ""

    # Intermediate outputs (for debugging/comparison)
    text_embeddings: Optional[torch.Tensor] = None  # [B, seq_len, dim]
    text_embedding_shape: Optional[tuple] = None

    def save_video(self, path: Path, fps: int = 24) -> None:
        """Save video to file using ffmpeg.

        Args:
            path: Output path (should end in .mp4)
            fps: Frame rate for output video
        """
        import subprocess
        import tempfile

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Video should be [F, H, W, C] uint8
        if self.video.dtype != torch.uint8:
            video = (self.video.clamp(0, 1) * 255).to(torch.uint8)
        else:
            video = self.video

        if video.dim() == 5:  # [B, C, F, H, W]
            video = video[0].permute(1, 2, 3, 0)  # [F, H, W, C]

        frames = video.cpu().numpy()
        _num_frames, height, width, _channels = frames.shape

        # Write to temp file then encode
        with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as f:
            frames.tofile(f.name)
            temp_path = f.name

        try:
            cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-s",
                f"{width}x{height}",
                "-pix_fmt",
                "rgb24",
                "-r",
                str(fps),
                "-i",
                temp_path,
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                str(path),
            ]
            subprocess.run(cmd, check=True, capture_output=True)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def save_metadata(self, path: Path) -> None:
        """Save generation metadata to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        metadata = {
            "prompt": self.prompt,
            "backend": self.backend_name,
            # --- UPDATED: Uses to_dict() to capture ALL fields automatically ---
            "config": self.config.to_dict() if self.config else None,
            "stats": self.stats.to_dict() if self.stats else None,
            "text_embedding_shape": list(self.text_embedding_shape)
            if self.text_embedding_shape
            else None,
        }

        # Uses the json_serializer helper defined at the top of protocol.py
        with open(path, "w") as f:
            json.dump(metadata, f, indent=2, default=json_serializer)


class Backend(ABC):
    """Abstract base class for video generation backends.

    Both llm_dit and ltx2 backends must implement this interface.
    This ensures tests work identically with either implementation.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return backend name ('llm_dit' or 'ltx2')."""
        ...

    @abstractmethod
    def generate_video(
        self,
        prompt: str,
        config: GenerationConfig,
        output_dir: Optional[Path] = None,
        save_video: bool = True,
        save_latents: bool = False,
    ) -> GenerationResult:
        """Generate video from text prompt.

        Args:
            prompt: Text prompt for video generation
            config: Generation configuration
            output_dir: Directory to save outputs (video, metadata)
            save_video: Whether to save video to file
            save_latents: Whether to include raw latents in result

        Returns:
            GenerationResult with video tensor and metadata
        """
        ...

    @abstractmethod
    def encode_text(
        self,
        prompt: str,
        output_dir: Optional[Path] = None,
        debug_trace: bool = False,
    ) -> torch.Tensor:
        """Encode text prompt to embeddings.

        Args:
            prompt: Text prompt to encode
            output_dir: Optional directory to save diagnostics
            debug_trace: If True, save detailed connector diagnostics

        Returns:
            Text embeddings tensor [1, seq_len, dim]
        """
        ...

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up GPU memory and resources."""
        ...

    def is_available(self) -> bool:
        """Check if this backend can be used."""
        return True
