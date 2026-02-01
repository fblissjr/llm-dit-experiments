"""
Pipeline Schema System for Dynamic UI Generation

last updated: 2026-01-25

This module provides dataclasses for defining pipeline schemas that the
frontend uses to generate forms dynamically. Each pipeline registers its
schema, and the web server exposes them via /api/pipelines.

Key concepts:
- ParamSchema: Describes a single UI control (slider, textarea, etc.)
- PipelineSchema: Complete schema for a pipeline's UI
- PIPELINE_REGISTRY: Auto-populated dict of all registered pipelines
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal, Any


ParamType = Literal["textarea", "slider", "number", "checkbox", "select", "image", "color"]
OutputType = Literal["image", "video", "layers"]
GroupType = Literal["basic", "advanced", "expert", "scheduler", "optimization", "enhancement"]


@dataclass
class ParamSchema:
    """Describes a UI control for a pipeline parameter.

    This maps directly to form controls in the frontend. The frontend
    reads these schemas and generates the appropriate input elements.

    Attributes:
        id: Maps to API field name (e.g., "guidance_scale")
        type: Control type - determines which Web Component renders it
        label: Human-readable label shown in UI
        default: Default value for the control
        min: Minimum value for slider/number inputs
        max: Maximum value for slider/number inputs
        step: Step increment for slider/number inputs
        options: List of valid options for select inputs
        group: Grouping for progressive disclosure (basic shown, advanced collapsed)
        tooltip: Help text shown on hover/focus
        conditional: Show only when another field matches condition
                     e.g., {"dype_enabled": True} - show only when dype_enabled is True
        placeholder: Placeholder text for textarea/text inputs
        rows: Number of rows for textarea inputs
        required: Whether the field is required for generation
        max_count: Maximum number of images allowed (for image type controls)
    """
    id: str
    type: ParamType
    label: str
    default: Any = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    options: list[str] | None = None
    group: GroupType = "basic"
    tooltip: str | None = None
    conditional: dict[str, Any] | None = None
    placeholder: str | None = None
    rows: int | None = None
    required: bool = False
    max_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict, excluding None values for cleaner JSON."""
        result = {}
        for key, value in asdict(self).items():
            if value is not None:
                result[key] = value
        return result


@dataclass
class PipelineSchema:
    """Complete schema for a pipeline's UI.

    This describes everything the frontend needs to render a pipeline's
    form and handle its output. The frontend fetches all schemas at startup
    and uses them to build the UI dynamically.

    Attributes:
        id: Unique identifier matching SUPPORTED_MODEL_TYPES (e.g., "zimage")
        name: Human-readable display name (e.g., "Z-Image")
        description: Brief description of the pipeline's purpose
        output_type: What the pipeline produces - affects result display
        color: Tailwind color class for theming (e.g., "blue", "purple")
        icon: Optional icon identifier or emoji
        params: List of ParamSchema defining all form controls
        supports_history: Whether to show generation history
        supports_img2img: Whether pipeline accepts input images
        supports_reference_images: Whether pipeline uses reference images (FLUX.2)
        supports_streaming: Whether generation uses SSE streaming (LTX-2)
        endpoint: API endpoint for generation (default: /api/{id}/generate)
        category: Grouping for pipeline tabs (e.g., "image", "video")
    """
    id: str
    name: str
    description: str
    output_type: OutputType
    color: str
    params: list[ParamSchema] = field(default_factory=list)
    icon: str | None = None
    supports_history: bool = True
    supports_img2img: bool = False
    supports_reference_images: bool = False
    supports_streaming: bool = False
    endpoint: str | None = None
    category: str = "image"

    def __post_init__(self):
        """Set default endpoint if not provided."""
        if self.endpoint is None:
            self.endpoint = f"/api/{self.id}/generate"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "output_type": self.output_type,
            "color": self.color,
            "icon": self.icon,
            "params": [p.to_dict() for p in self.params],
            "supports_history": self.supports_history,
            "supports_img2img": self.supports_img2img,
            "supports_reference_images": self.supports_reference_images,
            "supports_streaming": self.supports_streaming,
            "endpoint": self.endpoint,
            "category": self.category,
        }

    def get_params_by_group(self, group: GroupType) -> list[ParamSchema]:
        """Get all params belonging to a specific group."""
        return [p for p in self.params if p.group == group]

    def get_defaults(self) -> dict[str, Any]:
        """Get a dict of all param defaults."""
        return {p.id: p.default for p in self.params if p.default is not None}


# Global registry of all pipeline schemas
PIPELINE_REGISTRY: dict[str, PipelineSchema] = {}


def register_pipeline(schema: PipelineSchema) -> PipelineSchema:
    """Register a pipeline schema in the global registry.

    Can be used as a decorator or called directly.

    Example:
        @register_pipeline
        def create_zimage_schema() -> PipelineSchema:
            return PipelineSchema(...)

        # Or directly:
        register_pipeline(PipelineSchema(...))
    """
    PIPELINE_REGISTRY[schema.id] = schema
    return schema


def get_pipeline(pipeline_id: str) -> PipelineSchema | None:
    """Get a pipeline schema by ID."""
    return PIPELINE_REGISTRY.get(pipeline_id)


def get_all_pipelines() -> dict[str, PipelineSchema]:
    """Get all registered pipeline schemas."""
    return PIPELINE_REGISTRY.copy()


def get_pipelines_by_category(category: str) -> list[PipelineSchema]:
    """Get all pipelines in a category."""
    return [p for p in PIPELINE_REGISTRY.values() if p.category == category]


# Import all schema modules to trigger registration
# These imports must be at the bottom to avoid circular imports
from . import zimage
from . import ltx2
from . import flux2
from . import qwenimage
