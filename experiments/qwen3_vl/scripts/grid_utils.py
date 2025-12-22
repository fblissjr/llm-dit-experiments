"""
grid generation utilities for vl experiments.

last updated: 2025-12-22

note: this module now wraps the shared experiments.utils for consistency.
specialized functions like make_alpha_grid are kept here for vl-specific workflows.
"""

import sys
from pathlib import Path

# Add experiments to path for shared utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.utils import save_image_grid


def make_grid(
    images: list,
    labels: list[str],
    cols: int,
    output_path: Path | str,
    cell_size: int = 256,
    label_height: int = 25,
) -> Path:
    """
    create a labeled grid from a list of images.

    note: this is now a wrapper around experiments.utils.save_image_grid
    for backward compatibility with existing vl scripts.

    args:
        images: list of image paths or PIL Image objects
        labels: list of labels (same length as images)
        cols: number of columns
        output_path: where to save the grid
        cell_size: size of each cell in pixels
        label_height: height of label area above each cell

    returns:
        path to saved grid
    """
    return save_image_grid(
        images=images,
        path=output_path,
        cols=cols,
        labels=labels,
        cell_size=cell_size,
        label_height=label_height,
    )


def make_alpha_grid(
    output_dir: Path | str,
    prefix: str,
    alphas: list[float],
    output_name: str = 'grid.png',
    cell_size: int = 256,
) -> Path:
    """create grid for alpha sweep experiment.

    expects files named: {prefix}_a{alpha*10}.png
    """
    output_dir = Path(output_dir)
    images = [output_dir / f'{prefix}_a{int(a*10)}.png' for a in alphas]
    labels = [f'alpha={a}' for a in alphas]
    return make_grid(images, labels, len(alphas), output_dir / output_name, cell_size)


def make_alpha_strength_grid(
    output_dir: Path | str,
    prefix: str,
    alphas: list[float],
    strengths: list[float],
    output_name: str = 'grid.png',
    cell_size: int = 256,
) -> Path:
    """create grid for alpha x strength sweep.

    expects files named: {prefix}_a{alpha*10}_s{strength*10}.png
    rows = alphas, cols = strengths
    """
    output_dir = Path(output_dir)
    images = []
    labels = []

    for a in alphas:
        for s in strengths:
            images.append(output_dir / f'{prefix}_a{int(a*10)}_s{int(s*10)}.png')
            labels.append(f'a={a} s={s}')

    return make_grid(images, labels, len(strengths), output_dir / output_name, cell_size)
