#!/usr/bin/env python3
"""
LTX-2 Visual Comparison Grid Generator

Last Updated: 2026-01-16

Generate side-by-side comparison grids for layer blend experiments.
This tool creates visual outputs for human inspection - the only way
to actually understand what layers contribute to video quality.

Output format:
    ┌─────────────────────────────────────────────────────────────┐
    │  baseline    late_heavy    early_excluded   top_contributors │
    ├─────────────────────────────────────────────────────────────┤
    │  [frame]     [frame]       [frame]          [frame]         │ prompt 1
    │  0.3421      0.3512        0.3298           0.3189          │ (SigLIP)
    ├─────────────────────────────────────────────────────────────┤
    │  [frame]     [frame]       [frame]          [frame]         │ prompt 2
    │  0.3156      0.3289        0.3044           0.2987          │ (SigLIP)
    └─────────────────────────────────────────────────────────────┘

Usage:
    # From a layer_blend_sweep result directory
    uv run python experiments/ltx2/generate_comparison_grid.py \\
        --input experiments/results/ltx2_layer_blend_20260116_* \\
        --output comparison_grid.png

    # Specify which configs to compare (default: all)
    uv run python experiments/ltx2/generate_comparison_grid.py \\
        --input experiments/results/ltx2_layer_blend_* \\
        --configs baseline late_heavy top_contributors \\
        --output comparison_grid.png

    # Limit to specific prompts
    uv run python experiments/ltx2/generate_comparison_grid.py \\
        --input experiments/results/ltx2_layer_blend_* \\
        --prompts official_1 cat_street \\
        --output comparison_grid.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from PIL import Image


def load_experiment_results(input_dir: Path) -> dict:
    """Load results from a layer blend/profile experiment directory."""
    # Try to find the summary JSON
    summary_files = list(input_dir.glob("*_summary.json"))
    if not summary_files:
        raise FileNotFoundError(f"No summary file found in {input_dir}")

    summary_path = summary_files[0]
    with open(summary_path) as f:
        summary = json.load(f)

    # Load metadata for each sample
    metadata_dir = input_dir / "metadata"
    if metadata_dir.exists():
        for meta_file in metadata_dir.glob("*.json"):
            # Parse metadata
            pass

    return summary


def collect_images_and_scores(
    input_dir: Path,
    configs: list[str] | None = None,
    prompts: list[str] | None = None,
) -> dict:
    """
    Collect images and SigLIP scores organized by config and prompt.

    Returns:
        {
            "config_name": {
                "prompt_id": {
                    "image_path": Path,
                    "siglip_score": float | None,
                }
            }
        }
    """
    images_dir = input_dir / "images"
    metadata_dir = input_dir / "metadata"

    if not images_dir.exists():
        raise FileNotFoundError(f"No images directory found in {input_dir}")

    results = {}

    # Parse all metadata files
    for meta_file in metadata_dir.glob("*.json"):
        with open(meta_file) as f:
            meta = json.load(f)

        # Extract config and prompt info
        config_info = meta.get("config", {})
        config_name = config_info.get("blend_name")
        prompt_id = config_info.get("prompt_id")

        # For layer_profile experiments, use layer_idx as config
        if config_name is None:
            layer_idx = config_info.get("variable_value")
            if layer_idx is not None:
                config_name = f"layer_{layer_idx:02d}"

        if config_name is None or prompt_id is None:
            continue

        # Filter by requested configs/prompts
        if configs and config_name not in configs:
            continue
        if prompts and prompt_id not in prompts:
            continue

        # Find corresponding image
        image_path = images_dir / meta.get("output_path", "").split("/")[-1]
        if not image_path.exists():
            # Try alternate naming
            image_path = images_dir / f"{config_name}_{prompt_id}.png"

        if not image_path.exists():
            continue

        # Store result
        if config_name not in results:
            results[config_name] = {}

        results[config_name][prompt_id] = {
            "image_path": image_path,
            "siglip_score": meta.get("siglip_score"),
        }

    return results


def generate_comparison_grid(
    data: dict,
    output_path: Path,
    title: str = "LTX-2 Layer Configuration Comparison",
    figsize: tuple[int, int] | None = None,
):
    """
    Generate a visual comparison grid.

    Args:
        data: Output from collect_images_and_scores()
        output_path: Where to save the PNG
        title: Grid title
        figsize: Figure size (width, height) in inches
    """
    # Determine grid dimensions
    configs = sorted(data.keys())
    prompts = sorted(set(p for c in data.values() for p in c.keys()))

    n_cols = len(configs)
    n_rows = len(prompts)

    if n_cols == 0 or n_rows == 0:
        raise ValueError("No data to display")

    # Auto-calculate figsize
    if figsize is None:
        figsize = (max(4 * n_cols, 12), max(3 * n_rows, 8))

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(n_rows + 1, n_cols, figure=fig, height_ratios=[0.5] + [3] * n_rows)

    # Header row with config names
    for col_idx, config_name in enumerate(configs):
        ax_header = fig.add_subplot(gs[0, col_idx])
        ax_header.text(
            0.5, 0.5, config_name,
            ha='center', va='center',
            fontsize=10, fontweight='bold',
            transform=ax_header.transAxes
        )
        ax_header.axis('off')

    # Image cells
    for row_idx, prompt_id in enumerate(prompts):
        for col_idx, config_name in enumerate(configs):
            ax = fig.add_subplot(gs[row_idx + 1, col_idx])

            cell_data = data.get(config_name, {}).get(prompt_id)

            if cell_data and cell_data["image_path"].exists():
                # Load and display image
                img = Image.open(cell_data["image_path"])
                ax.imshow(img)

                # Add SigLIP score overlay
                siglip = cell_data.get("siglip_score")
                if siglip is not None:
                    score_text = f"SigLIP: {siglip:.4f}"
                    # Add semi-transparent background for readability
                    ax.text(
                        0.5, 0.02, score_text,
                        ha='center', va='bottom',
                        fontsize=9, color='white',
                        transform=ax.transAxes,
                        bbox=dict(boxstyle='round', facecolor='black', alpha=0.7)
                    )
            else:
                # Missing image placeholder
                ax.text(
                    0.5, 0.5, "Missing",
                    ha='center', va='center',
                    fontsize=12, color='gray',
                    transform=ax.transAxes
                )
                ax.set_facecolor('#f0f0f0')

            ax.axis('off')

            # Add prompt label on the left edge
            if col_idx == 0:
                ax.text(
                    -0.05, 0.5, prompt_id[:20],
                    ha='right', va='center',
                    fontsize=8, rotation=0,
                    transform=ax.transAxes
                )

    # Main title
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

    # Adjust layout
    plt.tight_layout(rect=[0.05, 0.02, 1, 0.96])

    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved comparison grid to: {output_path}")
    print(f"  Configs: {n_cols}")
    print(f"  Prompts: {n_rows}")


def find_latest_experiment(base_dir: Path, pattern: str = "ltx2_layer_blend_*") -> Path:
    """Find the most recent experiment directory matching the pattern."""
    candidates = sorted(base_dir.glob(pattern), reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No directories matching {pattern} in {base_dir}")
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(
        description="Generate visual comparison grid from layer experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Input experiment directory (or glob pattern)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="comparison_grid.png",
        help="Output PNG path (default: comparison_grid.png)",
    )
    parser.add_argument(
        "--configs", "-c",
        nargs="+",
        help="Specific configs to include (default: all)",
    )
    parser.add_argument(
        "--prompts", "-p",
        nargs="+",
        help="Specific prompts to include (default: all)",
    )
    parser.add_argument(
        "--title", "-t",
        type=str,
        default="LTX-2 Layer Configuration Comparison",
        help="Grid title",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Use the latest experiment in experiments/results/",
    )

    args = parser.parse_args()

    # Find input directory
    if args.latest:
        input_dir = find_latest_experiment(Path("experiments/results"))
    elif args.input:
        input_path = Path(args.input)
        if input_path.is_dir():
            input_dir = input_path
        else:
            # Treat as glob pattern
            matches = sorted(Path(".").glob(args.input), reverse=True)
            if not matches:
                raise FileNotFoundError(f"No directories matching: {args.input}")
            input_dir = matches[0]
    else:
        parser.error("Either --input or --latest is required")

    print(f"Loading experiment from: {input_dir}")

    # Collect data
    data = collect_images_and_scores(
        input_dir,
        configs=args.configs,
        prompts=args.prompts,
    )

    if not data:
        raise ValueError("No images found matching criteria")

    # Generate grid
    generate_comparison_grid(
        data,
        output_path=Path(args.output),
        title=args.title,
    )


if __name__ == "__main__":
    main()
