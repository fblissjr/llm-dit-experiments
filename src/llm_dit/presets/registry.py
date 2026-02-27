"""
Preset registry for caching and accessing presets.

last updated: 2026-02-01

The PresetRegistry provides a centralized way to access presets with:
- Lazy loading: Presets are loaded on first access
- Caching: Presets are cached after loading
- Filtering: Get presets by pipeline, category, or variant
"""

import logging
from pathlib import Path

from .loader import load_presets_from_dir
from .schema import GenerationPreset

logger = logging.getLogger(__name__)

# Global registry instance (singleton pattern)
_registry: "PresetRegistry | None" = None


class PresetRegistry:
    """Registry for managing generation presets.

    The registry lazily loads presets from a directory and caches them.
    It provides methods to query presets by pipeline, category, and variant.

    Example:
        registry = PresetRegistry("presets/")
        zimage_presets = registry.list_for_pipeline("zimage")
        photorealistic = registry.get("photorealistic")
    """

    def __init__(self, presets_dir: str | Path):
        """Initialize the registry with a presets directory.

        Args:
            presets_dir: Path to the presets directory
        """
        self.presets_dir = Path(presets_dir)
        self._presets: dict[str, GenerationPreset] | None = None
        self._loaded = False

    def _ensure_loaded(self) -> dict[str, GenerationPreset]:
        """Ensure presets are loaded (lazy loading).

        Returns:
            The loaded presets dict (never None after this call)
        """
        if self._loaded and self._presets is not None:
            return self._presets

        if self.presets_dir.exists():
            self._presets = load_presets_from_dir(self.presets_dir)
        else:
            logger.warning(f"Presets directory not found: {self.presets_dir}")
            self._presets = {}

        self._loaded = True
        return self._presets

    def reload(self) -> None:
        """Force reload all presets from disk."""
        self._loaded = False
        self._presets = None
        self._ensure_loaded()

    def get(self, name: str) -> GenerationPreset | None:
        """Get a preset by name.

        Args:
            name: Preset name

        Returns:
            The preset if found, None otherwise
        """
        presets = self._ensure_loaded()
        return presets.get(name)

    def get_all(self) -> dict[str, GenerationPreset]:
        """Get all presets.

        Returns:
            Dict mapping preset names to GenerationPreset objects
        """
        presets = self._ensure_loaded()
        return presets.copy()

    def list_for_pipeline(
        self,
        pipeline_id: str,
        variant: str | None = None,
    ) -> list[GenerationPreset]:
        """Get all presets that apply to a specific pipeline.

        Args:
            pipeline_id: Pipeline identifier (e.g., "zimage", "ltx2")
            variant: Optional variant filter (e.g., "base", "turbo")

        Returns:
            List of presets that apply to the pipeline
        """
        presets = self._ensure_loaded()
        results = []
        for preset in presets.values():
            if preset.applies_to_pipeline(pipeline_id):
                if variant is None or preset.applies_to_variant(variant):
                    results.append(preset)
        return results

    def list_by_category(self, category: str) -> list[GenerationPreset]:
        """Get all presets in a specific category.

        Args:
            category: Category name (e.g., "negative_prompt", "quality")

        Returns:
            List of presets in the category
        """
        presets = self._ensure_loaded()
        return [p for p in presets.values() if p.category == category]

    def list_names(self) -> list[str]:
        """Get list of all preset names.

        Returns:
            List of preset names
        """
        presets = self._ensure_loaded()
        return list(presets.keys())

    def __contains__(self, name: str) -> bool:
        """Check if a preset exists."""
        presets = self._ensure_loaded()
        return name in presets

    def __len__(self) -> int:
        """Get number of presets."""
        presets = self._ensure_loaded()
        return len(presets)


def get_preset_registry(presets_dir: str | Path | None = None) -> PresetRegistry:
    """Get or create the global preset registry.

    The first call must provide presets_dir. Subsequent calls can omit it.

    Args:
        presets_dir: Path to presets directory (required on first call)

    Returns:
        The global PresetRegistry instance

    Raises:
        ValueError: If called without presets_dir and no registry exists
    """
    global _registry

    if _registry is not None:
        if presets_dir is not None:
            # Re-initialize with new directory
            _registry = PresetRegistry(presets_dir)
        return _registry

    if presets_dir is None:
        raise ValueError(
            "presets_dir is required on first call to get_preset_registry()"
        )

    _registry = PresetRegistry(presets_dir)
    return _registry


def reset_preset_registry() -> None:
    """Reset the global registry (mainly for testing)."""
    global _registry
    _registry = None
