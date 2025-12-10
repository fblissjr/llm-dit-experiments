"""Data models for experiment comparison."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ExperimentImage:
    """Single generated image with metadata."""

    path: Path
    prompt_id: str
    variable_name: str
    variable_value: Any
    seed: int
    # Metrics (may be None)
    siglip_score: float | None = None
    image_reward: float | None = None
    generation_time: float | None = None
    # Full config from metadata JSON
    config: dict = field(default_factory=dict)


@dataclass
class ExperimentRun:
    """Collection of images from a single experiment run."""

    name: str  # e.g., "think_block_20251209_163804"
    experiment_type: str  # e.g., "think_block"
    timestamp: str  # YYYYMMDD_HHMMSS format
    base_path: Path
    images: list[ExperimentImage] = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    @property
    def prompt_ids(self) -> list[str]:
        """Unique prompt IDs in this experiment, sorted."""
        return sorted(set(img.prompt_id for img in self.images))

    @property
    def variable_values(self) -> list[Any]:
        """Unique variable values tested, sorted by string representation."""
        # Use string representation for uniqueness (handles unhashable types like dict)
        seen = {}
        for img in self.images:
            key = str(img.variable_value)
            if key not in seen:
                seen[key] = img.variable_value
        return sorted(seen.values(), key=str)

    @property
    def seeds(self) -> list[int]:
        """Unique seeds used, sorted."""
        return sorted(set(img.seed for img in self.images))

    @property
    def variable_name(self) -> str | None:
        """The variable being tested (from first image)."""
        if self.images:
            return self.images[0].variable_name
        return None


@dataclass
class ComparisonSpec:
    """Specification for what to compare."""

    experiment: ExperimentRun
    prompts: list[str] | None = None  # None = all prompts
    variables: list[str] | None = None  # None = all variable values
    seeds: list[int] | None = None  # None = first seed only
