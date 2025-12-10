"""Experiment comparison tools for Z-Image ablation studies."""

from .models import ComparisonSpec, ExperimentImage, ExperimentRun
from .discovery import discover_experiments, get_experiment_by_name
from .grid import generate_grid, generate_side_by_side
from .diff import compute_diff

__all__ = [
    "ComparisonSpec",
    "ExperimentImage",
    "ExperimentRun",
    "discover_experiments",
    "get_experiment_by_name",
    "generate_grid",
    "generate_side_by_side",
    "compute_diff",
]
