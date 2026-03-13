"""Auto-discovery of experiments from results directory.

Last Updated: 2026-01-18

Supports both legacy and pipeline-organized directory structures:

Legacy structure (backward compatible):
    experiments/results/{experiment_type}_{timestamp}/

Pipeline-organized structure (new):
    experiments/results/{pipeline}/{experiment_type}_{timestamp}/

Where pipeline is one of: ltx2, z_image, wan, qwen3_vl
"""

import json
import re
from pathlib import Path

from .models import ExperimentImage, ExperimentRun

# Default results directory
RESULTS_DIR = Path(__file__).parent.parent / "results"

# Known pipelines for organized structure
KNOWN_PIPELINES = ["ltx2", "z_image", "qwen3_vl"]


def discover_experiments(results_dir: Path | None = None) -> list[ExperimentRun]:
    """
    Scan results directory and return all experiments.

    Supports both legacy and pipeline-organized directory structures:
    - Legacy: experiments/results/{experiment_type}_{timestamp}/
    - New: experiments/results/{pipeline}/{experiment_type}_{timestamp}/

    Args:
        results_dir: Path to results directory. Defaults to experiments/results/

    Returns:
        List of ExperimentRun objects sorted by timestamp (newest first)
    """
    if results_dir is None:
        results_dir = RESULTS_DIR

    if not results_dir.exists():
        return []

    experiments = []

    for item in results_dir.iterdir():
        if not item.is_dir():
            continue

        if item.name in KNOWN_PIPELINES:
            # Pipeline subdirectory - scan for experiments inside
            for exp_dir in item.iterdir():
                if not exp_dir.is_dir():
                    continue
                exp = _parse_experiment_dir(exp_dir, pipeline=item.name)
                if exp:
                    experiments.append(exp)
        elif item.name == "archive":
            # Skip archive directory
            continue
        elif item.name == "shared":
            # Skip shared directory (for cross-pipeline baselines)
            continue
        else:
            # Legacy: direct experiment directories (no pipeline prefix)
            exp = _parse_experiment_dir(item, pipeline=None)
            if exp:
                experiments.append(exp)

    # Sort by timestamp (newest first)
    experiments.sort(key=lambda x: x.timestamp, reverse=True)
    return experiments


def _parse_experiment_dir(exp_dir: Path, pipeline: str | None) -> ExperimentRun | None:
    """
    Parse a single experiment directory.

    Args:
        exp_dir: Path to experiment directory
        pipeline: Pipeline name (ltx2, z_image, etc.) or None for legacy

    Returns:
        ExperimentRun if valid, None otherwise
    """
    # Parse directory name: {experiment_type}_{YYYYMMDD_HHMMSS}
    match = re.match(r"^(.+?)_(\d{8}_\d{6})$", exp_dir.name)
    if not match:
        return None

    exp_type, timestamp = match.groups()

    # Find actual experiment subfolder
    # Structure: results/{exp_type}_{timestamp}/{exp_type}/images/
    inner_dir = exp_dir / exp_type
    if not inner_dir.exists():
        # Try direct structure (no nested folder)
        inner_dir = exp_dir

    run = ExperimentRun(
        name=exp_dir.name,
        experiment_type=exp_type,
        timestamp=timestamp,
        base_path=inner_dir,
        pipeline=pipeline,
    )

    # Load summary/results if exists
    # Try multiple summary filename patterns
    summary_paths = [
        inner_dir / f"{exp_type}_summary.json",  # Legacy pattern
        inner_dir / "results.json",               # New pattern from ExperimentRunnerBase
        exp_dir / "results.json",                 # Direct results.json
    ]
    for summary_path in summary_paths:
        if summary_path.exists():
            try:
                run.summary = json.loads(summary_path.read_text())
                break
            except json.JSONDecodeError:
                continue

    # Load images/videos from metadata
    metadata_dir = inner_dir / "metadata"
    if not metadata_dir.exists():
        metadata_dir = exp_dir / "metadata"  # Try direct path

    if metadata_dir.exists():
        for meta_file in sorted(metadata_dir.glob("*.json")):
            try:
                img = _load_image_from_metadata(meta_file, inner_dir)
                if img.path.exists():
                    run.images.append(img)
            except (json.JSONDecodeError, KeyError):
                continue

    if run.images:
        return run

    return None


def _load_image_from_metadata(meta_path: Path, base_dir: Path) -> ExperimentImage:
    """Load ExperimentImage from metadata JSON.

    Handles both image (.png) and video (.mp4) outputs.
    """
    data = json.loads(meta_path.read_text())
    config = data.get("config", {})

    # Output path - try multiple locations
    output_path = data.get("output_path", "")
    media_path = None

    if output_path:
        # Try as absolute path first
        candidate = Path(output_path)
        if candidate.is_absolute() and candidate.exists():
            media_path = candidate
        else:
            # Try relative to project root
            project_root = Path(__file__).parent.parent.parent
            candidate = project_root / output_path
            if candidate.exists():
                media_path = candidate

    # Fallback: construct from metadata filename, trying multiple formats
    if media_path is None or not media_path.exists():
        # Try images directory (Z-Image, image experiments)
        for ext in [".png", ".jpg", ".jpeg"]:
            candidate = base_dir / "images" / (meta_path.stem + ext)
            if candidate.exists():
                media_path = candidate
                break

        # Try videos directory (LTX-2, video experiments)
        if media_path is None or not media_path.exists():
            for ext in [".mp4", ".webm"]:
                candidate = base_dir / "videos" / (meta_path.stem + ext)
                if candidate.exists():
                    media_path = candidate
                    break

        # Final fallback to direct images path
        if media_path is None or not media_path.exists():
            media_path = base_dir / "images" / (meta_path.stem + ".png")

    return ExperimentImage(
        path=media_path,
        prompt_id=config.get("prompt_id", "unknown"),
        variable_name=config.get("variable_name", "unknown"),
        variable_value=config.get("variable_value", ""),
        seed=config.get("seed", 0),
        siglip_score=data.get("siglip_score"),
        image_reward=data.get("image_reward"),
        generation_time=data.get("generation_time_seconds"),
        config=config,
    )


def get_experiment_by_name(name: str, results_dir: Path | None = None) -> ExperimentRun | None:
    """
    Find experiment by exact name or partial match.

    Args:
        name: Experiment name or partial name to match
        results_dir: Path to results directory

    Returns:
        ExperimentRun if found, None otherwise
    """
    experiments = discover_experiments(results_dir)

    # Exact match first
    for exp in experiments:
        if exp.name == name:
            return exp

    # Partial match (prefix)
    for exp in experiments:
        if exp.name.startswith(name):
            return exp

    # Partial match (contains)
    for exp in experiments:
        if name in exp.name:
            return exp

    return None
