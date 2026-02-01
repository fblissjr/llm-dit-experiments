"""
Preset schema definition for cross-pipeline generation presets.

last updated: 2026-02-01

Presets allow users to define reusable generation configurations (negative prompts,
CFG, steps, prompt templates) that can apply to one or multiple pipelines.

Example preset file (YAML frontmatter + markdown body):

    ---
    name: photorealistic
    description: Universal photorealism - removes artistic/digital artifacts
    category: negative_prompt
    pipelines: [zimage]
    variant: base

    negative_prompt: |
      low quality, worst quality, blurry, pixelated
    guidance_scale: 4.0
    steps: 40
    ---

    Research-backed negative prompt for photorealistic generation.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenerationPreset:
    """A generation preset with parameters and metadata.

    Presets store generation configurations that can be applied to pipelines.
    They support both generation parameters (negative_prompt, guidance_scale, etc.)
    and prompt templates.

    Attributes:
        name: Unique identifier for the preset
        description: Human-readable description
        category: Preset category for grouping (e.g., "negative_prompt", "quality", "style")
        pipelines: List of pipeline IDs this preset applies to (["zimage", "ltx2", "all"])
        variant: Optional variant filter ("base", "turbo") - only apply for specific variant

        # Generation parameters (all optional)
        negative_prompt: Negative prompt text
        guidance_scale: CFG scale override
        steps: Number of inference steps override
        shift: Scheduler shift override
        d_noise: D-noise parameter override
        cfg_normalization: CFG normalization override

        # Prompt templates (optional)
        positive_template: Template prepended to user prompt
        system_prompt: System prompt for models that support it

        # Metadata
        metadata: Additional frontmatter fields not explicitly defined
        source_path: Path to the source preset file (set by loader)
    """

    name: str
    description: str = ""
    category: str = ""
    pipelines: list[str] = field(default_factory=list)
    variant: str | None = None

    # Generation parameters
    negative_prompt: str | None = None
    guidance_scale: float | None = None
    steps: int | None = None
    shift: float | None = None
    d_noise: float | None = None
    cfg_normalization: float | None = None

    # Prompt templates
    positive_template: str | None = None
    system_prompt: str | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    source_path: str | None = None

    def applies_to_pipeline(self, pipeline_id: str) -> bool:
        """Check if this preset applies to a given pipeline.

        Args:
            pipeline_id: Pipeline identifier (e.g., "zimage", "ltx2")

        Returns:
            True if preset applies to this pipeline
        """
        if not self.pipelines:
            return True  # No restriction = applies to all
        if "all" in self.pipelines:
            return True
        return pipeline_id in self.pipelines

    def applies_to_variant(self, variant: str | None) -> bool:
        """Check if this preset applies to a given variant.

        Args:
            variant: Variant identifier (e.g., "base", "turbo") or None

        Returns:
            True if preset applies to this variant
        """
        if self.variant is None:
            return True  # No restriction = applies to all variants
        if variant is None:
            return True  # No variant specified = use all presets
        return self.variant == variant

    def get_params(self) -> dict[str, Any]:
        """Get non-None generation parameters as a dict.

        Returns:
            Dict of parameter name to value, excluding None values
        """
        params = {}
        for key in [
            "negative_prompt",
            "guidance_scale",
            "steps",
            "shift",
            "d_noise",
            "cfg_normalization",
            "positive_template",
            "system_prompt",
        ]:
            value = getattr(self, key)
            if value is not None:
                params[key] = value
        return params

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization.

        Returns:
            Dict representation suitable for API responses
        """
        result = {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "pipelines": self.pipelines,
            "variant": self.variant,
            "params": self.get_params(),
        }
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    def to_api_response(self) -> dict[str, Any]:
        """Convert to API response format with flattened params.

        This format is used for /api/presets endpoints.
        """
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "pipelines": self.pipelines,
            "variant": self.variant,
            "params": self.get_params(),
        }
