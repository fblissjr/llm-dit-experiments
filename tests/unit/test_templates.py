"""
Unit tests for template loading from markdown with YAML frontmatter.

These tests run on any platform without GPU or model files.
"""

import pytest

from llm_dit.templates import Template, load_template, load_templates_from_dir

pytestmark = pytest.mark.unit


class TestTemplate:
    """Test Template dataclass."""

    def test_template_defaults(self):
        tpl = Template(name="test", content="Generate images")
        assert tpl.name == "test"
        assert tpl.content == "Generate images"
        # Default values may be empty string or None depending on implementation
        assert tpl.description == "" or tpl.description is None
        assert tpl.category == "" or tpl.category is None
        # add_think_block defaults to True (matches DiffSynth reference)
        assert tpl.add_think_block is True
        assert tpl.thinking_content == ""
        assert tpl.assistant_content == ""


class TestLoadTemplate:
    """Test template loading from files."""

    def test_load_plain_markdown(self, tmp_path):
        """Template with no frontmatter."""
        tpl_path = tmp_path / "test.md"
        tpl_path.write_text("Generate realistic images.")

        tpl = load_template(tpl_path)
        assert tpl.name == "test"
        assert tpl.content == "Generate realistic images."

    def test_load_with_basic_frontmatter(self, tmp_path):
        """Template with name and description."""
        tpl_path = tmp_path / "photo.md"
        tpl_path.write_text(
            """---
name: photorealistic
description: Generate photorealistic images
category: photography
---
Generate photorealistic images with professional lighting."""
        )

        tpl = load_template(tpl_path)
        assert tpl.name == "photorealistic"
        assert tpl.description == "Generate photorealistic images"
        assert tpl.category == "photography"
        assert "Generate photorealistic" in tpl.content

    def test_load_with_extended_frontmatter(self, tmp_path):
        """Template with think block settings."""
        tpl_path = tmp_path / "structured.md"
        tpl_path.write_text(
            """---
name: json_structured
description: Parse JSON-structured prompts
category: structured
add_think_block: true
thinking_content: |
  Parsing the JSON structure to identify:
  - Subject and scene elements
  - Style and artistic direction
assistant_content: Processing your request...
---
Parse and interpret JSON-formatted image descriptions."""
        )

        tpl = load_template(tpl_path)
        assert tpl.name == "json_structured"
        assert tpl.add_think_block is True
        assert "Parsing the JSON" in tpl.thinking_content
        assert tpl.assistant_content == "Processing your request..."
        assert "Parse and interpret" in tpl.content

    def test_frontmatter_with_empty_body(self, tmp_path):
        """Template with only frontmatter, no body."""
        tpl_path = tmp_path / "empty.md"
        tpl_path.write_text(
            """---
name: empty_template
description: Template with no body
---
"""
        )

        tpl = load_template(tpl_path)
        assert tpl.name == "empty_template"
        assert tpl.content.strip() == ""

    def test_multiline_content(self, tmp_path):
        """Template body with multiple paragraphs."""
        tpl_path = tmp_path / "multi.md"
        tpl_path.write_text(
            """---
name: multiline
---
First paragraph.

Second paragraph.

Third paragraph."""
        )

        tpl = load_template(tpl_path)
        assert "First paragraph" in tpl.content
        assert "Second paragraph" in tpl.content
        assert "Third paragraph" in tpl.content


class TestLoadTemplatesFromDir:
    """Test loading multiple templates from directory."""

    def test_load_directory(self, tmp_path):
        # Create multiple templates
        (tmp_path / "photo.md").write_text(
            """---
name: photo
description: Photography
---
Photo template content"""
        )
        (tmp_path / "art.md").write_text(
            """---
name: art
description: Artwork
---
Art template content"""
        )
        (tmp_path / "anime.md").write_text(
            """---
name: anime
description: Anime style
---
Anime template content"""
        )

        templates = load_templates_from_dir(tmp_path)

        assert len(templates) == 3
        assert "photo" in templates
        assert "art" in templates
        assert "anime" in templates

    def test_ignores_non_markdown_files(self, tmp_path):
        (tmp_path / "valid.md").write_text("---\nname: valid\n---\nContent")
        (tmp_path / "readme.txt").write_text("Not a template")
        (tmp_path / "config.json").write_text('{"not": "template"}')

        templates = load_templates_from_dir(tmp_path)

        assert len(templates) == 1
        assert "valid" in templates

    def test_empty_directory(self, tmp_path):
        templates = load_templates_from_dir(tmp_path)
        assert len(templates) == 0

    def test_subdirectories_loaded_by_default(self, tmp_path):
        """Subdirectories are loaded by default (recursive=True)."""
        # Create template in root
        (tmp_path / "root.md").write_text("---\nname: root\n---\nRoot content")

        # Create subdirectory with template
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "nested.md").write_text("---\nname: nested\n---\nNested content")

        templates = load_templates_from_dir(tmp_path)

        # Should load both root and subdirectory templates (recursive=True by default)
        assert len(templates) == 2
        assert "root" in templates
        assert "nested" in templates

    def test_subdirectories_ignored_when_not_recursive(self, tmp_path):
        """Subdirectories are ignored when recursive=False."""
        # Create template in root
        (tmp_path / "root.md").write_text("---\nname: root\n---\nRoot content")

        # Create subdirectory with template
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "nested.md").write_text("---\nname: nested\n---\nNested content")

        templates = load_templates_from_dir(tmp_path, recursive=False)

        # Should only load root-level templates when recursive=False
        assert len(templates) == 1
        assert "root" in templates


class TestRealTemplatesDirectory:
    """Test loading from actual templates directory (if it exists)."""

    def test_load_z_image_templates(self, templates_dir):
        """Load templates from templates/z_image if it exists."""
        if not templates_dir.exists():
            pytest.skip("templates/z_image directory not found")

        templates = load_templates_from_dir(templates_dir)

        # Should have many templates (140+ according to docs)
        assert len(templates) > 0

        # Check for some expected templates
        template_names = list(templates.keys())
        # At minimum, there should be some templates
        assert len(template_names) > 0
