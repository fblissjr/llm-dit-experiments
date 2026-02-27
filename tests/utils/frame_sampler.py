"""
Frame extraction utilities for video test inspection.

Last Updated: 2026-02-10

Samples evenly-spaced frames from [F, H, W, C] uint8 video tensors
and saves them as PNGs for human/Claude visual inspection.

Run with:
    uv run pytest tests/utils/test_frame_sampler.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def compute_sample_indices(total_frames: int, num_samples: int = 8) -> list[int]:
    """Evenly-spaced frame indices with first and last frame guaranteed.

    For 33 frames, num_samples=8: [0, 4, 9, 13, 18, 22, 27, 32]
    For 121 frames, num_samples=8: [0, 17, 34, 51, 68, 86, 103, 120]

    Args:
        total_frames: Total number of frames in the video.
        num_samples: Number of frames to sample.

    Returns:
        Sorted, deduplicated list of frame indices clamped to [0, total_frames-1].
    """
    if total_frames <= 0:
        return []
    if num_samples <= 0:
        return []
    if num_samples >= total_frames:
        return list(range(total_frames))

    # linspace gives us evenly-spaced floats including endpoints
    raw = np.linspace(0, total_frames - 1, num_samples)
    indices = sorted(set(int(round(x)) for x in raw))
    # Clamp just in case
    return [max(0, min(i, total_frames - 1)) for i in indices]


def sample_frames(
    video: torch.Tensor | np.ndarray,
    num_samples: int = 8,
) -> tuple[list[int], list[np.ndarray]]:
    """Extract evenly-spaced frames from a video tensor.

    Args:
        video: Video tensor of shape [F, H, W, C] (uint8 expected).
        num_samples: Number of frames to sample.

    Returns:
        Tuple of (frame_indices, frames_as_numpy) where each frame
        is an [H, W, C] uint8 numpy array.
    """
    if isinstance(video, torch.Tensor):
        video_np = video.cpu().numpy()
    else:
        video_np = video

    total_frames = video_np.shape[0]
    indices = compute_sample_indices(total_frames, num_samples)

    frames = [video_np[i] for i in indices]
    return indices, frames


def save_frames(
    frames: list[np.ndarray],
    indices: list[int],
    output_dir: Path,
    prefix: str = "frame",
) -> list[Path]:
    """Save frames as PNGs with frame index in the filename.

    Filenames use the actual video frame index (not sequential), so
    ``frame_0016.png`` means frame 16 of the original video.

    Args:
        frames: List of [H, W, C] uint8 numpy arrays.
        indices: Corresponding frame indices from the video.
        output_dir: Directory to save PNGs into.
        prefix: Filename prefix (default "frame").

    Returns:
        List of paths to saved PNG files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    for idx, frame in zip(indices, frames):
        path = output_dir / f"{prefix}_{idx:04d}.png"
        Image.fromarray(frame).save(path)
        saved.append(path)

    return saved
