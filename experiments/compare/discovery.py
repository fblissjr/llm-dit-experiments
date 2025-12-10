"""Auto-discovery of experiments from results directory."""

import json
import re
from pathlib import Path

from .models import ExperimentImage, ExperimentRun

# Default results directory
RESULTS_DIR = Path(__file__).parent.parent / "results"


def discover_experiments(results_dir: Path | None = None) -> list[ExperimentRun]:
    """
    Scan results directory and return all experiments.

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

    for exp_dir in results_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        # Parse directory name: {experiment_type}_{YYYYMMDD_HHMMSS}
        match = re.match(r"^(.+?)_(\d{8}_\d{6})$", exp_dir.name)
        if not match:
            continue

        exp_type, timestamp = match.groups()

        # Find actual experiment subfolder
        # Structure: results/think_block_20251209_163804/think_block/images/
        inner_dir = exp_dir / exp_type
        if not inner_dir.exists():
            # Try direct structure (no nested folder)
            inner_dir = exp_dir

        run = ExperimentRun(
            name=exp_dir.name,
            experiment_type=exp_type,
            timestamp=timestamp,
            base_path=inner_dir,
        )

        # Load summary if exists
        summary_json = inner_dir / f"{exp_type}_summary.json"
        if summary_json.exists():
            try:
                run.summary = json.loads(summary_json.read_text())
            except json.JSONDecodeError:
                pass

        # Load images from metadata
        metadata_dir = inner_dir / "metadata"
        if metadata_dir.exists():
            for meta_file in sorted(metadata_dir.glob("*.json")):
                try:
                    img = _load_image_from_metadata(meta_file, inner_dir)
                    if img.path.exists():
                        run.images.append(img)
                except (json.JSONDecodeError, KeyError):
                    continue

        if run.images:
            experiments.append(run)

    # Sort by timestamp (newest first)
    experiments.sort(key=lambda x: x.timestamp, reverse=True)
    return experiments


def _load_image_from_metadata(meta_path: Path, base_dir: Path) -> ExperimentImage:
    """Load ExperimentImage from metadata JSON."""
    data = json.loads(meta_path.read_text())
    config = data.get("config", {})

    # Image path - try multiple locations
    output_path = data.get("output_path", "")
    image_path = None

    if output_path:
        # Try as absolute path first
        candidate = Path(output_path)
        if candidate.is_absolute() and candidate.exists():
            image_path = candidate
        else:
            # Try relative to project root
            project_root = Path(__file__).parent.parent.parent
            candidate = project_root / output_path
            if candidate.exists():
                image_path = candidate

    # Fallback: construct from metadata filename
    if image_path is None or not image_path.exists():
        image_path = base_dir / "images" / (meta_path.stem + ".png")

    return ExperimentImage(
        path=image_path,
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
