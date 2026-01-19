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
    fp8: bool = True  # Use FP8 quantization for transformer
    dtype: torch.dtype = field(default=torch.bfloat16)

    # Conditioning (for I2V)
    conditioning_image: Optional[torch.Tensor] = None
    conditioning_frame_idx: int = 0
    conditioning_strength: float = 0.8

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
        num_frames, height, width, channels = frames.shape

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
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        metadata = {
            "prompt": self.prompt,
            "backend": self.backend_name,
            "config": {
                "num_frames": self.config.num_frames if self.config else 0,
                "height": self.config.height if self.config else 0,
                "width": self.config.width if self.config else 0,
                "num_inference_steps": self.config.num_inference_steps
                if self.config
                else 0,
                "guidance_scale": self.config.guidance_scale if self.config else 0,
                "seed": self.config.seed if self.config else 0,
                "fp8": self.config.fp8 if self.config else False,
            },
            "text_embedding_shape": list(self.text_embedding_shape)
            if self.text_embedding_shape
            else None,
        }

        if self.stats:
            metadata["stats"] = self.stats.to_dict()

        with open(path, "w") as f:
            json.dump(metadata, f, indent=2)


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
    def encode_text(self, prompt: str) -> torch.Tensor:
        """Encode text prompt to embeddings.

        Args:
            prompt: Text prompt to encode

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
