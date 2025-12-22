"""
shared utilities for experiment scripts.

last updated: 2025-12-22

provides consistent helpers for:
- creating image grids with labels
- saving experiment metadata
- common experiment patterns
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont


def save_image_grid(
    images: list,
    path: Path | str,
    cols: Optional[int] = None,
    labels: Optional[list[str]] = None,
    cell_size: int = 256,
    label_height: int = 30,
) -> Path:
    """
    Save a grid of images with optional labels.

    Args:
        images: List of PIL Image objects or paths to images
        path: Output path for the grid
        cols: Number of columns (default: auto-calculate square grid)
        labels: Optional list of labels (same length as images)
        cell_size: Size of each cell in pixels
        label_height: Height of label area above each cell

    Returns:
        Path to saved grid

    Examples:
        # Simple grid
        save_image_grid([img1, img2, img3], "grid.png")

        # With labels and custom layout
        save_image_grid(
            [img1, img2, img3, img4],
            "grid.png",
            cols=2,
            labels=["baseline", "method_a", "method_b", "method_c"]
        )
    """
    if not images:
        raise ValueError("No images provided")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Auto-calculate columns for square grid
    if cols is None:
        import math
        cols = math.ceil(math.sqrt(len(images)))

    # Calculate grid dimensions
    rows = (len(images) + cols - 1) // cols
    grid_w = cols * cell_size
    grid_h = rows * (cell_size + label_height)

    # Create grid canvas
    grid = Image.new('RGB', (grid_w, grid_h), 'white')
    draw = ImageDraw.Draw(grid)

    # Load font
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    except:
        font = ImageFont.load_default()

    # Paste images
    for i, img_item in enumerate(images):
        row, col = i // cols, i % cols
        x = col * cell_size
        y = row * (cell_size + label_height)

        # Handle both PIL Images and paths
        if isinstance(img_item, Image.Image):
            img = img_item.convert('RGB').resize((cell_size, cell_size), Image.Resampling.LANCZOS)
            grid.paste(img, (x, y + label_height))
        else:
            img_path = Path(img_item)
            if img_path.exists():
                img = Image.open(img_path).convert('RGB').resize(
                    (cell_size, cell_size), Image.Resampling.LANCZOS
                )
                grid.paste(img, (x, y + label_height))
            else:
                # Draw placeholder for missing image
                draw.rectangle(
                    [x, y + label_height, x + cell_size, y + label_height + cell_size],
                    fill='gray'
                )
                draw.text(
                    (x + 10, y + label_height + cell_size // 2),
                    'missing',
                    fill='white',
                    font=font
                )

        # Add label if provided
        if labels and i < len(labels):
            # Center text
            text_bbox = draw.textbbox((0, 0), labels[i], font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_x = x + (cell_size - text_width) // 2
            text_y = y + 5

            draw.text((text_x, text_y), labels[i], fill='black', font=font)

    grid.save(path)
    return path


def save_metadata(path: Path | str, **kwargs) -> Path:
    """
    Save experiment metadata as JSON with timestamp.

    Args:
        path: Output path for metadata JSON file
        **kwargs: Metadata key-value pairs to save

    Returns:
        Path to saved metadata file

    Examples:
        save_metadata(
            "experiment.json",
            prompt="A cat sleeping",
            steps=9,
            seed=42,
            model="z-image-turbo",
        )
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Add timestamp
    metadata = {
        "timestamp": datetime.now().isoformat(),
        **kwargs,
    }

    # Convert Path objects to strings for JSON serialization
    for key, value in metadata.items():
        if isinstance(value, Path):
            metadata[key] = str(value)

    with open(path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return path


def create_comparison_grid(
    images: list[Image.Image],
    labels: list[str],
    cols: int = 3,
    label_height: int = 40,
) -> Image.Image:
    """
    Create a comparison grid with labels (returns PIL Image, doesn't save).

    This is useful when you want to manipulate the grid before saving or
    include it in a larger composite.

    Args:
        images: List of PIL Images
        labels: List of labels (same length as images)
        cols: Number of columns
        label_height: Height of label area below each image

    Returns:
        PIL Image of the grid

    Examples:
        grid = create_comparison_grid(
            [img1, img2, img3],
            ["baseline", "dype", "multipass"],
            cols=3
        )
        grid.save("comparison.png")
    """
    if not images:
        raise ValueError("No images to grid")

    # Calculate grid dimensions
    rows = (len(images) + cols - 1) // cols
    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)

    # Create grid canvas
    grid_width = max_width * cols
    grid_height = (max_height + label_height) * rows
    grid = Image.new("RGB", (grid_width, grid_height), (255, 255, 255))

    # Load font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(grid)

    # Paste images and add labels
    for idx, (img, label) in enumerate(zip(images, labels)):
        row = idx // cols
        col = idx % cols
        x = col * max_width
        y = row * (max_height + label_height)

        # Paste image
        grid.paste(img, (x, y))

        # Draw label (centered below image)
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = x + (max_width - text_width) // 2
        text_y = y + max_height + 10

        draw.text((text_x, text_y), label, fill=(0, 0, 0), font=font)

    return grid


def save_result_summary(path: Path | str, results: list[dict], title: str = "Experiment Results"):
    """
    Save a human-readable summary of experiment results.

    Args:
        path: Output path for summary text file
        results: List of result dictionaries
        title: Title for the summary

    Examples:
        results = [
            {"method": "baseline", "time": 12.3, "vram": 8.5},
            {"method": "dype", "time": 15.1, "vram": 9.2},
        ]
        save_result_summary("summary.txt", results, "DyPE Comparison")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        f.write(f"{title}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n\n")

        for result in results:
            for key, value in result.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

    return path
