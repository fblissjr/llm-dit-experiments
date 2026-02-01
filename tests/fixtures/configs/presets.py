"""
Test preset helpers for loading generation presets in tests.

Last updated: 2026-02-01

This module provides thin wrappers around the production PresetRegistry
to make test preset loading convenient while ensuring test configs stay
1:1 in sync with production.

Example usage:
    from tests.fixtures.configs.presets import get_test_preset

    preset = get_test_preset("zimage_base_test")
    print(preset.guidance_scale)  # 4.0
    print(preset.metadata["prompt"])  # "A cat sleeping in sunlight"
"""

from pathlib import Path
from typing import Any

from llm_dit.presets import GenerationPreset, get_preset_registry, reset_preset_registry


# Default presets directory (relative to repo root)
DEFAULT_PRESETS_DIR = Path(__file__).parent.parent.parent.parent / "presets"


def get_test_preset(
    name: str,
    presets_dir: str | Path | None = None,
    **overrides: Any,
) -> GenerationPreset:
    """Get a test preset by name with optional overrides.

    Args:
        name: Preset name (e.g., "zimage_base_test")
        presets_dir: Optional presets directory path
        **overrides: Optional parameter overrides (model_path, etc.)

    Returns:
        GenerationPreset with any overrides applied

    Raises:
        ValueError: If preset not found

    Example:
        preset = get_test_preset("zimage_base_test", model_path="/custom/path")
    """
    if presets_dir is None:
        presets_dir = DEFAULT_PRESETS_DIR

    registry = get_preset_registry(presets_dir)
    preset = registry.get(name)

    if preset is None:
        available = registry.list_names()
        raise ValueError(
            f"Test preset '{name}' not found. Available: {available}"
        )

    # Apply overrides if provided
    if overrides:
        preset = _apply_overrides(preset, overrides)

    return preset


def get_test_presets_by_category(category: str = "testing") -> list[GenerationPreset]:
    """Get all test presets in a category.

    Args:
        category: Category name (default: "testing")

    Returns:
        List of GenerationPreset objects
    """
    registry = get_preset_registry(DEFAULT_PRESETS_DIR)
    return registry.list_by_category(category)


def get_test_presets_for_pipeline(
    pipeline_id: str,
    variant: str | None = None,
    category: str = "testing",
) -> list[GenerationPreset]:
    """Get test presets for a specific pipeline.

    Args:
        pipeline_id: Pipeline identifier (e.g., "zimage", "flux2")
        variant: Optional variant filter (e.g., "base", "turbo")
        category: Category filter (default: "testing")

    Returns:
        List of GenerationPreset objects matching the criteria
    """
    registry = get_preset_registry(DEFAULT_PRESETS_DIR)
    presets = registry.list_for_pipeline(pipeline_id, variant=variant)
    return [p for p in presets if p.category == category]


def _apply_overrides(preset: GenerationPreset, overrides: dict[str, Any]) -> GenerationPreset:
    """Apply overrides to a preset, returning a new instance.

    This creates a copy of the preset with the specified values overridden.
    Supports overriding both top-level attributes and metadata values.

    Args:
        preset: Original preset
        overrides: Dict of attribute names to new values

    Returns:
        New GenerationPreset with overrides applied
    """
    from dataclasses import fields, replace

    # Separate metadata overrides from regular overrides
    metadata_overrides = {}
    regular_overrides = {}

    field_names = {f.name for f in fields(preset)}

    for key, value in overrides.items():
        if key in field_names:
            regular_overrides[key] = value
        else:
            # Assume it's a metadata key
            metadata_overrides[key] = value

    # Apply regular overrides
    if regular_overrides:
        preset = replace(preset, **regular_overrides)

    # Apply metadata overrides
    if metadata_overrides:
        new_metadata = {**preset.metadata, **metadata_overrides}
        preset = replace(preset, metadata=new_metadata)

    return preset


def reset_test_registry() -> None:
    """Reset the preset registry (useful between tests).

    Call this in test fixtures if you need to reload presets from disk.
    """
    reset_preset_registry()
