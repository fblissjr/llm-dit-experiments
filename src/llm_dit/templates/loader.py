"""
Template loading from markdown files with YAML frontmatter.

Templates are stored as markdown files with YAML frontmatter containing
metadata about the template (name, category, thinking content, etc.).

Based on: ComfyUI-QwenImageWanBridge/nodes/z_image_encoder.py load_z_image_templates()

Format:
    ---
    name: photorealistic
    description: Generate photorealistic images
    category: photography
    add_think_block: true
    thinking_content: |
      For photorealism, I need to consider...
    assistant_content: ""
    ---
    Generate a photorealistic image with accurate lighting...
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class Template:
    """
    A loaded template with metadata.

    Attributes:
        name: Template identifier (from frontmatter or filename)
        content: The system prompt content (markdown body)
        description: Human-readable description
        category: Template category for grouping
        add_think_block: Whether to enable thinking by default
        thinking_content: Pre-filled thinking content
        assistant_content: Pre-filled assistant content after thinking
        metadata: Any additional frontmatter fields
    """

    name: str
    content: str  # System prompt content
    description: str = ""
    category: str = ""
    add_think_block: bool = True
    thinking_content: str = ""
    assistant_content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def system_prompt(self) -> str:
        """Alias for content (backward compatibility)."""
        return self.content

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "name": self.name,
            "content": self.content,
            "description": self.description,
            "category": self.category,
            "add_think_block": self.add_think_block,
            "thinking_content": self.thinking_content,
            "assistant_content": self.assistant_content,
            "metadata": self.metadata,
        }


def load_template(path: str | Path) -> Template:
    """
    Load a template from a markdown file with YAML frontmatter.

    Args:
        path: Path to the markdown template file

    Returns:
        Loaded Template

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is invalid

    Example:
        template = load_template("templates/z_image/photorealistic.md")
        print(template.system_prompt)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Template not found: {path}")

    content = path.read_text(encoding="utf-8")

    # Check for YAML frontmatter
    if not content.startswith("---"):
        # No frontmatter, entire content is system prompt
        name = path.stem
        return Template(name=name, content=content.strip())

    # Parse frontmatter
    parts = content.split("---", 2)
    if len(parts) < 3:
        # Invalid format, treat as plain text
        name = path.stem
        return Template(name=name, content=content.strip())

    # Parse YAML frontmatter
    try:
        frontmatter = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError as e:
        logger.warning(f"Failed to parse frontmatter in {path}: {e}")
        frontmatter = {}

    # System prompt is everything after the second ---
    system_prompt = parts[2].strip()

    # Extract known fields
    name = frontmatter.pop("name", path.stem)
    description = frontmatter.pop("description", "")
    category = frontmatter.pop("category", "")
    add_think_block = frontmatter.pop("add_think_block", True)
    thinking_content = frontmatter.pop("thinking_content", "")
    assistant_content = frontmatter.pop("assistant_content", "")

    # Handle multiline strings from YAML
    if isinstance(thinking_content, str):
        thinking_content = thinking_content.strip()
    if isinstance(assistant_content, str):
        assistant_content = assistant_content.strip()

    return Template(
        name=name,
        content=system_prompt,  # Body is the system prompt content
        description=description,
        category=category,
        add_think_block=add_think_block,
        thinking_content=thinking_content,
        assistant_content=assistant_content,
        metadata=frontmatter,  # Remaining fields
    )


def load_templates_from_dir(
    directory: str | Path,
    pattern: str = "*.md",
) -> dict[str, Template]:
    """
    Load all templates from a directory.

    Args:
        directory: Path to templates directory
        pattern: Glob pattern for template files (default: *.md)

    Returns:
        Dict mapping template names to Template objects

    Example:
        templates = load_templates_from_dir("templates/z_image/")
        photo = templates["photorealistic"]
    """
    directory = Path(directory)
    if not directory.exists():
        logger.warning(f"Templates directory not found: {directory}")
        return {}

    templates = {}
    for path in sorted(directory.glob(pattern)):
        try:
            template = load_template(path)
            templates[template.name] = template
            logger.debug(f"Loaded template: {template.name}")
        except Exception as e:
            logger.warning(f"Failed to load template {path}: {e}")

    logger.info(f"Loaded {len(templates)} templates from {directory}")
    return templates
