"""Image difference calculation for experiment comparison."""

from pathlib import Path

import numpy as np
from PIL import Image


def compute_diff(
    image_a: Path | Image.Image,
    image_b: Path | Image.Image,
    mode: str = "highlight",
) -> Image.Image:
    """
    Compute difference between two images.

    Args:
        image_a: First image (path or PIL Image)
        image_b: Second image (path or PIL Image)
        mode: Difference visualization mode:
            - "absolute": Simple |A - B| grayscale
            - "highlight": Original A with differences highlighted in red
            - "heatmap": Color-coded difference magnitude (blue->green->red)

    Returns:
        PIL Image showing the difference
    """
    if isinstance(image_a, Path):
        image_a = Image.open(image_a)
    if isinstance(image_b, Path):
        image_b = Image.open(image_b)

    # Ensure same size
    if image_a.size != image_b.size:
        image_b = image_b.resize(image_a.size, Image.Resampling.LANCZOS)

    # Convert to numpy arrays
    arr_a = np.array(image_a.convert("RGB")).astype(np.float32)
    arr_b = np.array(image_b.convert("RGB")).astype(np.float32)

    if mode == "absolute":
        # Simple absolute difference, converted to grayscale
        diff = np.abs(arr_a - arr_b)
        diff_gray = np.mean(diff, axis=2)
        # Normalize and enhance contrast
        max_val = diff_gray.max()
        if max_val > 0:
            diff_norm = (diff_gray / max_val * 255).astype(np.uint8)
        else:
            diff_norm = diff_gray.astype(np.uint8)
        return Image.fromarray(diff_norm, mode="L")

    elif mode == "highlight":
        # Show original A with differences highlighted in red
        diff = np.abs(arr_a - arr_b)
        diff_mag = np.mean(diff, axis=2)
        threshold = 30  # Pixel difference threshold

        # Create output based on image_a
        output = arr_a.copy()
        mask = diff_mag > threshold

        # Blend: where different, show red tint
        output[mask, 0] = np.clip(output[mask, 0] * 0.5 + 255 * 0.5, 0, 255)  # Red
        output[mask, 1] = output[mask, 1] * 0.5  # Reduce green
        output[mask, 2] = output[mask, 2] * 0.5  # Reduce blue

        return Image.fromarray(output.astype(np.uint8))

    elif mode == "heatmap":
        # Color-coded difference magnitude (blue = same, red = different)
        diff = np.abs(arr_a - arr_b)
        diff_mag = np.mean(diff, axis=2)
        max_val = diff_mag.max()
        if max_val > 0:
            diff_norm = diff_mag / max_val
        else:
            diff_norm = diff_mag

        # Create RGB heatmap
        heatmap = np.zeros((*diff_norm.shape, 3), dtype=np.uint8)

        # Blue (low diff) -> Green (medium) -> Red (high diff)
        # Red channel: increases with difference
        heatmap[..., 0] = (diff_norm * 255).astype(np.uint8)
        # Green channel: peaks at 0.5 difference
        heatmap[..., 1] = ((1 - np.abs(diff_norm - 0.5) * 2) * 200).astype(np.uint8)
        # Blue channel: decreases with difference
        heatmap[..., 2] = ((1 - diff_norm) * 255).astype(np.uint8)

        return Image.fromarray(heatmap)

    else:
        raise ValueError(f"Unknown diff mode: {mode}. Use 'absolute', 'highlight', or 'heatmap'")


def compute_similarity_score(
    image_a: Path | Image.Image,
    image_b: Path | Image.Image,
) -> float:
    """
    Compute a simple similarity score between two images.

    Returns:
        Score from 0.0 (completely different) to 1.0 (identical)
    """
    if isinstance(image_a, Path):
        image_a = Image.open(image_a)
    if isinstance(image_b, Path):
        image_b = Image.open(image_b)

    # Ensure same size
    if image_a.size != image_b.size:
        image_b = image_b.resize(image_a.size, Image.Resampling.LANCZOS)

    arr_a = np.array(image_a.convert("RGB")).astype(np.float32)
    arr_b = np.array(image_b.convert("RGB")).astype(np.float32)

    # Mean absolute error normalized to [0, 1]
    mae = np.mean(np.abs(arr_a - arr_b)) / 255.0
    return 1.0 - mae
