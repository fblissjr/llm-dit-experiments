"""
Consolidated run metadata for inspectable test runs.

Last Updated: 2026-02-10

RunManifest captures everything needed to understand a generation test run:
identity, inputs, config, outputs, statistics, and performance. Serialized
to JSON via orjson for human and Claude inspection.

Run with:
    uv run pytest tests/utils/ -v
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import orjson
import torch


@dataclass
class RunManifest:
    """Consolidated metadata for a complete test run.

    This is the single file Claude or a human reads to understand a run.
    All paths are stored as relative strings for portability.
    """

    # Identity
    test_name: str
    timestamp: str  # ISO format

    # Inputs
    prompt: str
    negative_prompt: str = ""
    seed: int = 0

    # Generation config
    num_frames: int = 0
    height: int = 0
    width: int = 0

    # Two-stage config (serialized dict, None for single-stage)
    two_stage: dict | None = None

    # Output files (relative paths for portability)
    video_path: str = ""
    frame_paths: list[str] = field(default_factory=list)
    frame_indices: list[int] = field(default_factory=list)

    # Video statistics
    video_shape: list[int] = field(default_factory=list)  # [F, H, W, C]
    pixel_mean: float = 0.0
    pixel_std: float = 0.0

    # Performance
    stage_timings: dict[str, float] = field(default_factory=dict)
    total_time: float = 0.0
    peak_vram_gb: float = 0.0

    # Environment (torch version, GPU, CUDA version)
    environment: dict = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        test_name: str,
        prompt: str,
        num_frames: int = 0,
        height: int = 0,
        width: int = 0,
        seed: int = 0,
        negative_prompt: str = "",
        two_stage_config: dict | None = None,
    ) -> RunManifest:
        """Factory that pre-fills identity, environment, and config fields.

        Args:
            test_name: Name of the test producing this run.
            prompt: Generation prompt.
            num_frames: Number of video frames.
            height: Output height in pixels.
            width: Output width in pixels.
            seed: Random seed.
            negative_prompt: Negative prompt (two-stage only).
            two_stage_config: Serialized TwoStageConfig dict, or None.

        Returns:
            RunManifest with identity and environment pre-filled.
        """
        env: dict = {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
        if torch.cuda.is_available():
            env["cuda_version"] = torch.version.cuda
            env["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            env["gpu_vram_gb"] = round(props.total_memory / 1024**3, 2)

        return cls(
            test_name=test_name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            num_frames=num_frames,
            height=height,
            width=width,
            two_stage=two_stage_config,
            environment=env,
        )

    def set_video_info(self, video: torch.Tensor | np.ndarray, video_path: Path) -> None:
        """Compute video statistics and set the video path.

        Args:
            video: Video tensor [F, H, W, C] (uint8 expected).
            video_path: Absolute path to the saved video file.
        """
        if isinstance(video, torch.Tensor):
            arr = video.cpu().float()
        else:
            arr = torch.from_numpy(video).float()

        self.video_shape = list(arr.shape)
        self.pixel_mean = float(arr.mean())
        self.pixel_std = float(arr.std())
        self.video_path = video_path.name  # relative to output_dir

    def set_frame_info(self, paths: list[Path], indices: list[int]) -> None:
        """Set frame paths (converted to relative filename strings).

        Args:
            paths: Absolute paths to saved frame PNGs.
            indices: Corresponding frame indices from the video.
        """
        self.frame_paths = [p.name for p in paths]
        self.frame_indices = list(indices)

    def save(self, path: Path) -> None:
        """Serialize to JSON via orjson."""
        data = _manifest_to_dict(self)
        path.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))

    @classmethod
    def load(cls, path: Path) -> RunManifest:
        """Deserialize from JSON."""
        data = orjson.loads(path.read_bytes())
        return cls(**data)


def _manifest_to_dict(m: RunManifest) -> dict:
    """Convert RunManifest to a plain dict for serialization."""
    return {
        "test_name": m.test_name,
        "timestamp": m.timestamp,
        "prompt": m.prompt,
        "negative_prompt": m.negative_prompt,
        "seed": m.seed,
        "num_frames": m.num_frames,
        "height": m.height,
        "width": m.width,
        "two_stage": m.two_stage,
        "video_path": m.video_path,
        "frame_paths": m.frame_paths,
        "frame_indices": m.frame_indices,
        "video_shape": m.video_shape,
        "pixel_mean": m.pixel_mean,
        "pixel_std": m.pixel_std,
        "stage_timings": m.stage_timings,
        "total_time": m.total_time,
        "peak_vram_gb": m.peak_vram_gb,
        "environment": m.environment,
    }
