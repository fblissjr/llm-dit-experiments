"""
Output dataclasses for orchestration pipeline steps.

Last Updated: 2026-01-12

These typed outputs enable:
- Type checking between pipeline steps
- Automatic wiring of compatible outputs to inputs
- Preservation of intermediate results for debugging
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

__all__ = [
    "TextEmbeddings",
    "ImageOutput",
    "VideoOutput",
    "TranscriptionOutput",
    "ScenePrompt",
    "ScenePromptsOutput",
    "AudioFeatures",
]


@dataclass
class TextEmbeddings:
    """Text encoder output."""

    embeddings: torch.Tensor  # [B, seq_len, hidden_dim]
    attention_mask: torch.Tensor  # [B, seq_len]
    prompt: str
    token_count: int = 0

    def __post_init__(self):
        if self.token_count == 0:
            self.token_count = int(self.attention_mask.sum().item())


@dataclass
class ImageOutput:
    """Single image output from generation."""

    image: Image.Image
    latents: Optional[torch.Tensor] = None  # Preserved for img2img chaining
    seed: Optional[int] = None
    width: int = 0
    height: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.width == 0:
            self.width = self.image.width
        if self.height == 0:
            self.height = self.image.height

    def save(self, path: str) -> None:
        """Save image to file."""
        self.image.save(path)


@dataclass
class VideoOutput:
    """Video output from generation."""

    frames: np.ndarray  # [F, H, W, C] uint8 or [B, F, H, W, C]
    fps: float = 25.0
    audio: Optional[np.ndarray] = None  # [samples] float waveform
    audio_sr: int = 16000
    latents: Optional[torch.Tensor] = None  # For debugging
    seed: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        """Number of frames in video."""
        if self.frames.ndim == 4:
            return self.frames.shape[0]
        return self.frames.shape[1]  # Batched

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self.num_frames / self.fps

    def save(self, path: str, audio_path: Optional[str] = None) -> None:
        """Save video to file using imageio."""
        import imageio

        frames = self.frames
        if frames.ndim == 5:
            frames = frames[0]  # Remove batch dim

        writer = imageio.get_writer(path, fps=self.fps)
        for frame in frames:
            writer.append_data(frame)
        writer.close()

        # Mux audio if provided
        if audio_path or self.audio is not None:
            self._mux_audio(path, audio_path)

    def _mux_audio(self, video_path: str, audio_path: Optional[str] = None) -> None:
        """Mux audio into video using FFmpeg."""
        import os
        import subprocess
        import tempfile
        from pathlib import Path

        temp_audio_path = None
        try:
            if audio_path is None and self.audio is not None:
                # Write audio to temp file
                import soundfile as sf

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    sf.write(f.name, self.audio, self.audio_sr)
                    temp_audio_path = f.name
                    audio_path = temp_audio_path

            if audio_path:
                output = Path(video_path).with_suffix(".muxed.mp4")
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        video_path,
                        "-i",
                        audio_path,
                        "-c:v",
                        "copy",
                        "-c:a",
                        "aac",
                        "-shortest",
                        str(output),
                    ],
                    check=True,
                    capture_output=True,
                )
                # Replace original
                output.rename(video_path)
        finally:
            # Clean up temp audio file
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)


@dataclass
class TranscriptionOutput:
    """Audio transcription output with timestamps."""

    text: str
    segments: List[Dict[str, Any]]  # [{start, end, text}, ...]
    language: str = "en"

    @property
    def duration(self) -> float:
        """Total audio duration from segments."""
        if not self.segments:
            return 0.0
        return max(seg["end"] for seg in self.segments)

    def get_text_at(self, time: float) -> Optional[str]:
        """Get text at a specific timestamp."""
        for seg in self.segments:
            if seg["start"] <= time < seg["end"]:
                return seg["text"]
        return None


@dataclass
class ScenePrompt:
    """A single scene prompt with timing."""

    start: float
    end: float
    prompt: str
    style: str = ""
    camera: str = ""  # Camera movement description
    transition: str = ""  # Transition to next scene


@dataclass
class ScenePromptsOutput:
    """Collection of scene prompts for video generation."""

    scenes: List[ScenePrompt]
    style_prompt: str = ""
    character_description: str = ""

    @property
    def duration(self) -> float:
        """Total duration from scenes."""
        if not self.scenes:
            return 0.0
        return max(s.end for s in self.scenes)

    def __len__(self) -> int:
        return len(self.scenes)

    def __iter__(self):
        return iter(self.scenes)


@dataclass
class AudioFeatures:
    """Audio features for conditioning (e.g., Whisper features)."""

    features: torch.Tensor  # [B, T, D] or [T, D]
    sample_rate: int = 16000
    hop_length: int = 160  # Frames per feature
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Audio duration in seconds."""
        if self.features.ndim == 2:
            return self.features.shape[0] * self.hop_length / self.sample_rate
        return self.features.shape[1] * self.hop_length / self.sample_rate
