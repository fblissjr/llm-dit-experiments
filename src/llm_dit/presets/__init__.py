"""
Presets package for cross-pipeline generation presets.

last updated: 2026-02-01

This package provides a file-based preset system for reusable generation
configurations (negative prompts, CFG, steps, prompt templates) that can
apply to one or multiple pipelines.

Example usage:

    from llm_dit.presets import get_preset_registry, GenerationPreset

    # Initialize registry (call once at startup)
    registry = get_preset_registry("presets/")

    # Get presets for a pipeline
    zimage_presets = registry.list_for_pipeline("zimage", variant="base")

    # Get a specific preset
    preset = registry.get("photorealistic")
    if preset:
        print(preset.negative_prompt)
        print(preset.guidance_scale)

    # Use preset parameters
    params = preset.get_params()  # Dict of non-None params
"""

from .schema import GenerationPreset
from .loader import load_preset, load_presets_from_dir
from .registry import PresetRegistry, get_preset_registry, reset_preset_registry

__all__ = [
    "GenerationPreset",
    "load_preset",
    "load_presets_from_dir",
    "PresetRegistry",
    "get_preset_registry",
    "reset_preset_registry",
]
