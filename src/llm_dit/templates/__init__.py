"""
Template loading and management.

Provides:
- Template loading from markdown files with YAML frontmatter
- Template registry for caching and lookup
"""

from llm_dit.templates.loader import Template, load_template, load_templates_from_dir
from llm_dit.templates.registry import TemplateRegistry

__all__ = ["Template", "load_template", "load_templates_from_dir", "TemplateRegistry"]
