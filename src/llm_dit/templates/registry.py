"""
Template registry with caching.

Provides a centralized registry for loading and accessing templates,
with lazy loading and caching for performance.
"""

import logging
from pathlib import Path
from typing import Iterator

from llm_dit.templates.loader import Template, load_template, load_templates_from_dir

logger = logging.getLogger(__name__)


class TemplateRegistry:
    """
    Registry for managing and accessing templates.

    Provides lazy loading and caching of templates from disk.

    Example:
        registry = TemplateRegistry.from_directory("templates/z_image/")
        template = registry.get("photorealistic")
        if template:
            print(template.system_prompt)
    """

    def __init__(self):
        self._templates: dict[str, Template] = {}
        self._loaded_dirs: set[Path] = set()

    def add(self, template: Template) -> None:
        """Add a template to the registry."""
        self._templates[template.name] = template

    def get(self, name: str) -> Template | None:
        """
        Get a template by name.

        Args:
            name: Template name (without .md extension)

        Returns:
            Template if found, None otherwise
        """
        return self._templates.get(name)

    def __getitem__(self, name: str) -> Template:
        """Get template by name, raises KeyError if not found."""
        template = self.get(name)
        if template is None:
            raise KeyError(f"Template not found: {name}")
        return template

    def __contains__(self, name: str) -> bool:
        """Check if template exists."""
        return name in self._templates

    def __iter__(self) -> Iterator[str]:
        """Iterate over template names."""
        return iter(self._templates)

    def __len__(self) -> int:
        """Return number of templates."""
        return len(self._templates)

    def list_names(self) -> list[str]:
        """Return list of all template names."""
        return list(self._templates.keys())

    def list_by_category(self, category: str) -> list[Template]:
        """Return templates in a specific category."""
        return [t for t in self._templates.values() if t.category == category]

    def categories(self) -> set[str]:
        """Return all unique categories."""
        return {t.category for t in self._templates.values() if t.category}

    def load_directory(self, directory: str | Path) -> int:
        """
        Load templates from a directory.

        Args:
            directory: Path to templates directory

        Returns:
            Number of templates loaded
        """
        directory = Path(directory).resolve()
        if directory in self._loaded_dirs:
            logger.debug(f"Directory already loaded: {directory}")
            return 0

        templates = load_templates_from_dir(directory)
        for name, template in templates.items():
            self._templates[name] = template

        self._loaded_dirs.add(directory)
        return len(templates)

    def load_file(self, path: str | Path) -> Template:
        """
        Load a single template file.

        Args:
            path: Path to template file

        Returns:
            Loaded template (also added to registry)
        """
        template = load_template(path)
        self._templates[template.name] = template
        return template

    @classmethod
    def from_directory(cls, directory: str | Path) -> "TemplateRegistry":
        """
        Create a registry from a directory.

        Args:
            directory: Path to templates directory

        Returns:
            Registry with all templates loaded
        """
        registry = cls()
        registry.load_directory(directory)
        return registry

    def clear(self) -> None:
        """Clear all loaded templates."""
        self._templates.clear()
        self._loaded_dirs.clear()


# Default registry instance (lazy loaded)
_default_registry: TemplateRegistry | None = None


def get_default_registry() -> TemplateRegistry:
    """Get or create the default template registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = TemplateRegistry()
    return _default_registry


def set_default_registry(registry: TemplateRegistry) -> None:
    """Set the default template registry."""
    global _default_registry
    _default_registry = registry
