#!/usr/bin/env python3
"""
Experiment comparison tool for Z-Image ablation studies.

Generate image grids and comparisons from experiment results.

Usage:
    # List all experiments
    uv run experiments/compare.py --list

    # Generate grid (prompts x variable values)
    uv run experiments/compare.py -e think_block_20251209 --mode grid -o grid.png

    # Generate grid with custom thumbnail size
    uv run experiments/compare.py -e think_block_20251209 --mode grid --thumbnail-size 384

    # Side-by-side comparison
    uv run experiments/compare.py -e think_block_20251209 --mode side-by-side \\
        --values '","None' --prompt human_001 -o comparison.png

    # Difference overlay
    uv run experiments/compare.py -e think_block_20251209 --mode diff \\
        --values '","None' --prompt human_001 --diff-mode highlight -o diff.png

    # Show specific experiment details
    uv run experiments/compare.py -e think_block_20251209 --info
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.compare.discovery import discover_experiments, get_experiment_by_name
from experiments.compare.models import ComparisonSpec
from experiments.compare.grid import generate_grid, generate_side_by_side
from experiments.compare.diff import compute_diff


def list_experiments(args: argparse.Namespace) -> int:
    """List all available experiments."""
    experiments = discover_experiments()

    if not experiments:
        print("\nNo experiments found in experiments/results/")
        print("Run some experiments first with: uv run experiments/run_ablation.py")
        return 0

    print(f"\nFound {len(experiments)} experiment(s):\n")

    for exp in experiments:
        print(f"  {exp.name}")
        print(f"    Type: {exp.experiment_type}")
        print(f"    Variable: {exp.variable_name}")
        print(f"    Images: {len(exp.images)}")
        print(f"    Prompts: {len(exp.prompt_ids)} ({', '.join(exp.prompt_ids[:3])}{'...' if len(exp.prompt_ids) > 3 else ''})")
        print(f"    Values: {exp.variable_values}")
        print(f"    Seeds: {exp.seeds}")
        if exp.summary:
            total = exp.summary.get("total_runs", "?")
            success = exp.summary.get("successful_runs", "?")
            print(f"    Runs: {success}/{total} successful")
        print()

    return 0


def show_info(args: argparse.Namespace) -> int:
    """Show detailed info about an experiment."""
    exp = get_experiment_by_name(args.experiment)
    if not exp:
        print(f"Experiment not found: {args.experiment}")
        return 1

    print(f"\nExperiment: {exp.name}")
    print(f"  Type: {exp.experiment_type}")
    print(f"  Variable: {exp.variable_name}")
    print(f"  Base path: {exp.base_path}")
    print(f"\nPrompts ({len(exp.prompt_ids)}):")
    for pid in exp.prompt_ids:
        print(f"    {pid}")
    print(f"\nVariable values ({len(exp.variable_values)}):")
    for val in exp.variable_values:
        display = f'"{val}"' if val == "" else str(val)
        print(f"    {display}")
    print(f"\nSeeds ({len(exp.seeds)}): {exp.seeds}")
    print(f"\nTotal images: {len(exp.images)}")

    # Show sample images with metrics
    print("\nSample images (first 5):")
    for img in exp.images[:5]:
        val_display = f'"{img.variable_value}"' if img.variable_value == "" else str(img.variable_value)
        metric_str = ""
        if img.siglip_score is not None:
            metric_str = f" SL:{img.siglip_score:.3f}"
        print(f"    {img.prompt_id} | {img.variable_name}={val_display} | seed={img.seed}{metric_str}")

    return 0


def generate_grid_cmd(args: argparse.Namespace) -> int:
    """Generate a grid image."""
    exp = get_experiment_by_name(args.experiment)
    if not exp:
        print(f"Experiment not found: {args.experiment}")
        return 1

    print(f"Loaded experiment: {exp.name}")
    print(f"  {len(exp.images)} images, {len(exp.prompt_ids)} prompts, {len(exp.variable_values)} values")

    # Parse filters
    prompts = args.prompts.split(",") if args.prompts else None
    seeds = [args.seed] if args.seed else None

    spec = ComparisonSpec(
        experiment=exp,
        prompts=prompts,
        seeds=seeds,
    )

    output_path = args.output or Path(f"{exp.name}_grid.png")
    thumbnail_size = (args.thumbnail_size, args.thumbnail_size)

    print(f"Generating grid...")
    img = generate_grid(
        spec,
        output_path=output_path,
        thumbnail_size=thumbnail_size,
        show_metrics=not args.no_metrics,
    )
    print(f"Grid saved to: {output_path} ({img.width}x{img.height})")

    return 0


def generate_side_by_side_cmd(args: argparse.Namespace) -> int:
    """Generate side-by-side comparison."""
    exp = get_experiment_by_name(args.experiment)
    if not exp:
        print(f"Experiment not found: {args.experiment}")
        return 1

    if not args.values:
        print("Error: --values required (e.g., --values ',None' for empty string vs None)")
        return 1
    if not args.prompt:
        print("Error: --prompt required")
        return 1

    # Parse values (handle empty string specially)
    values = args.values.split(",")
    if len(values) != 2:
        print("Error: --values must have exactly 2 values separated by comma")
        return 1

    spec = ComparisonSpec(experiment=exp)
    seed = args.seed if args.seed else (exp.seeds[0] if exp.seeds else 42)
    output_path = args.output or Path(f"{exp.name}_compare.png")

    print(f"Comparing: '{values[0]}' vs '{values[1]}' for {args.prompt} (seed={seed})")
    try:
        img = generate_side_by_side(
            spec,
            value_a=values[0],
            value_b=values[1],
            prompt_id=args.prompt,
            seed=seed,
            output_path=output_path,
        )
        print(f"Comparison saved to: {output_path} ({img.width}x{img.height})")
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    return 0


def generate_diff_cmd(args: argparse.Namespace) -> int:
    """Generate difference overlay."""
    exp = get_experiment_by_name(args.experiment)
    if not exp:
        print(f"Experiment not found: {args.experiment}")
        return 1

    if not args.values:
        print("Error: --values required")
        return 1
    if not args.prompt:
        print("Error: --prompt required")
        return 1

    values = args.values.split(",")
    if len(values) != 2:
        print("Error: --values must have exactly 2 values")
        return 1

    seed = args.seed if args.seed else (exp.seeds[0] if exp.seeds else 42)

    # Find images
    img_a = None
    img_b = None
    for img in exp.images:
        if img.prompt_id == args.prompt and img.seed == seed:
            if str(img.variable_value) == values[0]:
                img_a = img
            elif str(img.variable_value) == values[1]:
                img_b = img

    if not img_a:
        print(f"Error: Could not find image for value '{values[0]}'")
        return 1
    if not img_b:
        print(f"Error: Could not find image for value '{values[1]}'")
        return 1

    output_path = args.output or Path(f"{exp.name}_diff.png")
    print(f"Computing {args.diff_mode} diff: '{values[0]}' vs '{values[1]}'")

    diff_img = compute_diff(img_a.path, img_b.path, mode=args.diff_mode)
    diff_img.save(output_path)
    print(f"Diff saved to: {output_path}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Experiment comparison tool for Z-Image ablation studies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all experiments
  uv run experiments/compare.py --list

  # Generate grid from an experiment
  uv run experiments/compare.py -e think_block_20251209 --mode grid

  # Compare two values side-by-side
  uv run experiments/compare.py -e think_block_20251209 --mode side-by-side \\
      --values ',None' --prompt human_001

  # Show pixel differences
  uv run experiments/compare.py -e think_block_20251209 --mode diff \\
      --values ',None' --prompt human_001 --diff-mode heatmap
""",
    )

    # Main options
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available experiments",
    )
    parser.add_argument(
        "--experiment", "-e",
        help="Experiment name (supports partial match)",
    )
    parser.add_argument(
        "--info", "-i",
        action="store_true",
        help="Show detailed info about an experiment",
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["grid", "side-by-side", "diff"],
        default="grid",
        help="Comparison mode (default: grid)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output path (default: auto-generated)",
    )

    # Filtering options
    parser.add_argument(
        "--prompts",
        help="Comma-separated prompt IDs to include (default: all)",
    )
    parser.add_argument(
        "--prompt",
        help="Single prompt ID (for side-by-side/diff modes)",
    )
    parser.add_argument(
        "--values",
        help="Comma-separated variable values for comparison (for side-by-side/diff)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed to use (default: first available)",
    )

    # Grid options
    parser.add_argument(
        "--thumbnail-size",
        type=int,
        default=256,
        help="Thumbnail size in pixels (default: 256)",
    )
    parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="Hide metric scores on grid",
    )

    # Diff options
    parser.add_argument(
        "--diff-mode",
        choices=["absolute", "highlight", "heatmap"],
        default="highlight",
        help="Diff visualization mode (default: highlight)",
    )

    args = parser.parse_args()

    # Route to appropriate command
    if args.list:
        return list_experiments(args)

    if not args.experiment:
        if args.info:
            parser.error("--experiment required with --info")
        parser.error("--experiment is required (use --list to see available)")

    if args.info:
        return show_info(args)

    if args.mode == "grid":
        return generate_grid_cmd(args)
    elif args.mode == "side-by-side":
        return generate_side_by_side_cmd(args)
    elif args.mode == "diff":
        return generate_diff_cmd(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
