"""
Preset loading from markdown files with YAML frontmatter.

last updated: 2026-02-01

Presets are stored as markdown files with YAML frontmatter containing
metadata and generation parameters. This mirrors the pattern used for
templates in src/llm_dit/templates/loader.py.

Format:
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
    shift: 6.0
    ---

    Research-backed negative prompt for photorealistic generation.
    Based on: internal/research/z-image/zimage_base_negative_prompts_research_by_gemini.md
"""

import logging
from pathlib import Path

import yaml

from .schema import GenerationPreset

logger = logging.getLogger(__name__)


def load_preset(path: str | Path) -> GenerationPreset:
    """Load a preset from a markdown file with YAML frontmatter.

    Args:
        path: Path to the markdown preset file

    Returns:
        Loaded GenerationPreset

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is invalid

    Example:
        preset = load_preset("presets/zimage/photorealistic.md")
        print(preset.negative_prompt)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Preset not found: {path}")

    content = path.read_text(encoding="utf-8")

    # Check for YAML frontmatter
    if not content.startswith("---"):
        # No frontmatter - invalid preset file
        raise ValueError(f"Preset file must have YAML frontmatter: {path}")

    # Parse frontmatter
    parts = content.split("---", 2)
    if len(parts) < 3:
        raise ValueError(f"Invalid frontmatter format in: {path}")

    # Parse YAML frontmatter
    try:
        frontmatter = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse frontmatter in {path}: {e}") from e

    # Extract known fields
    name = frontmatter.pop("name", path.stem)
    description = frontmatter.pop("description", "")
    category = frontmatter.pop("category", "")

    # Pipeline and variant filters
    pipelines = frontmatter.pop("pipelines", [])
    if isinstance(pipelines, str):
        pipelines = [pipelines]
    variant = frontmatter.pop("variant", None)

    # Generation parameters
    negative_prompt = frontmatter.pop("negative_prompt", None)
    guidance_scale = frontmatter.pop("guidance_scale", None)
    steps = frontmatter.pop("steps", None)
    shift = frontmatter.pop("shift", None)
    d_noise = frontmatter.pop("d_noise", None)
    cfg_normalization = frontmatter.pop("cfg_normalization", None)

    # Prompt templates
    positive_template = frontmatter.pop("positive_template", None)
    system_prompt = frontmatter.pop("system_prompt", None)

    # Clean up multiline strings
    if isinstance(negative_prompt, str):
        negative_prompt = negative_prompt.strip()
    if isinstance(positive_template, str):
        positive_template = positive_template.strip()
    if isinstance(system_prompt, str):
        system_prompt = system_prompt.strip()

    return GenerationPreset(
        name=name,
        description=description,
        category=category,
        pipelines=pipelines,
        variant=variant,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        steps=steps,
        shift=shift,
        d_noise=d_noise,
        cfg_normalization=cfg_normalization,
        positive_template=positive_template,
        system_prompt=system_prompt,
        metadata=frontmatter,  # Remaining fields
        source_path=str(path),
    )


def load_presets_from_dir(
    directory: str | Path,
    pattern: str = "**/*.md",
    recursive: bool = True,
) -> dict[str, GenerationPreset]:
    """Load all presets from a directory.

    Args:
        directory: Path to presets directory
        pattern: Glob pattern for preset files (default: **/*.md for recursive)
        recursive: Whether to search subdirectories (default: True)

    Returns:
        Dict mapping preset names to GenerationPreset objects

    Example:
        presets = load_presets_from_dir("presets/")
        photorealistic = presets["photorealistic"]
    """
    directory = Path(directory)
    if not directory.exists():
        logger.warning(f"Presets directory not found: {directory}")
        return {}

    # Use non-recursive pattern if requested
    if not recursive and "**" in pattern:
        pattern = pattern.replace("**/", "")

    presets = {}
    for path in sorted(directory.glob(pattern)):
        try:
            preset = load_preset(path)
            presets[preset.name] = preset
        except Exception as e:
            logger.warning(f"Failed to load preset {path}: {e}")

    logger.info(f"Loaded {len(presets)} presets from {directory}")
    return presets
