"""
Unit tests for the presets system.

last updated: 2026-02-01

Tests preset loading, schema validation, registry lookup, and pipeline filtering.
"""

import tempfile
from pathlib import Path

import pytest

from llm_dit.presets import (
    GenerationPreset,
    load_preset,
    load_presets_from_dir,
    PresetRegistry,
    get_preset_registry,
    reset_preset_registry,
)


class TestGenerationPreset:
    """Tests for the GenerationPreset dataclass."""

    def test_basic_creation(self):
        """Test creating a basic preset."""
        preset = GenerationPreset(
            name="test_preset",
            description="A test preset",
            category="test",
            pipelines=["zimage"],
        )
        assert preset.name == "test_preset"
        assert preset.description == "A test preset"
        assert preset.category == "test"
        assert preset.pipelines == ["zimage"]

    def test_applies_to_pipeline_specific(self):
        """Test pipeline filtering with specific pipelines."""
        preset = GenerationPreset(
            name="test",
            pipelines=["zimage", "ltx2"],
        )
        assert preset.applies_to_pipeline("zimage") is True
        assert preset.applies_to_pipeline("ltx2") is True
        assert preset.applies_to_pipeline("flux2") is False

    def test_applies_to_pipeline_all(self):
        """Test pipeline filtering with 'all' keyword."""
        preset = GenerationPreset(
            name="test",
            pipelines=["all"],
        )
        assert preset.applies_to_pipeline("zimage") is True
        assert preset.applies_to_pipeline("ltx2") is True
        assert preset.applies_to_pipeline("any_pipeline") is True

    def test_applies_to_pipeline_empty(self):
        """Test pipeline filtering with empty list (applies to all)."""
        preset = GenerationPreset(
            name="test",
            pipelines=[],
        )
        assert preset.applies_to_pipeline("zimage") is True
        assert preset.applies_to_pipeline("ltx2") is True

    def test_applies_to_variant(self):
        """Test variant filtering."""
        # Preset with specific variant
        preset = GenerationPreset(
            name="test",
            variant="base",
        )
        assert preset.applies_to_variant("base") is True
        assert preset.applies_to_variant("turbo") is False
        assert preset.applies_to_variant(None) is True

    def test_applies_to_variant_none(self):
        """Test preset without variant restriction."""
        preset = GenerationPreset(
            name="test",
            variant=None,
        )
        assert preset.applies_to_variant("base") is True
        assert preset.applies_to_variant("turbo") is True
        assert preset.applies_to_variant(None) is True

    def test_get_params(self):
        """Test extracting non-None parameters."""
        preset = GenerationPreset(
            name="test",
            negative_prompt="bad quality",
            guidance_scale=4.0,
            steps=40,
            shift=None,  # Should not appear in params
        )
        params = preset.get_params()
        assert params["negative_prompt"] == "bad quality"
        assert params["guidance_scale"] == 4.0
        assert params["steps"] == 40
        assert "shift" not in params

    def test_to_dict(self):
        """Test serialization to dict."""
        preset = GenerationPreset(
            name="test",
            description="Test preset",
            category="quality",
            pipelines=["zimage"],
            variant="base",
            negative_prompt="bad quality",
            guidance_scale=4.0,
        )
        d = preset.to_dict()
        assert d["name"] == "test"
        assert d["description"] == "Test preset"
        assert d["category"] == "quality"
        assert d["pipelines"] == ["zimage"]
        assert d["variant"] == "base"
        assert d["params"]["negative_prompt"] == "bad quality"
        assert d["params"]["guidance_scale"] == 4.0

    def test_to_api_response(self):
        """Test API response format."""
        preset = GenerationPreset(
            name="test",
            negative_prompt="bad quality",
        )
        response = preset.to_api_response()
        assert "name" in response
        assert "params" in response
        assert response["params"]["negative_prompt"] == "bad quality"


class TestLoadPreset:
    """Tests for loading presets from files."""

    def test_load_valid_preset(self, tmp_path):
        """Test loading a valid preset file."""
        preset_content = """---
name: test_preset
description: A test preset
category: quality
pipelines: [zimage]
variant: base

negative_prompt: |
  bad quality, low quality
guidance_scale: 4.0
steps: 40
---

This is the description body.
"""
        preset_file = tmp_path / "test.md"
        preset_file.write_text(preset_content)

        preset = load_preset(preset_file)
        assert preset.name == "test_preset"
        assert preset.description == "A test preset"
        assert preset.category == "quality"
        assert preset.pipelines == ["zimage"]
        assert preset.variant == "base"
        assert "bad quality" in preset.negative_prompt
        assert preset.guidance_scale == 4.0
        assert preset.steps == 40

    def test_load_preset_minimal(self, tmp_path):
        """Test loading a preset with minimal frontmatter."""
        preset_content = """---
name: minimal
---

Just a name.
"""
        preset_file = tmp_path / "minimal.md"
        preset_file.write_text(preset_content)

        preset = load_preset(preset_file)
        assert preset.name == "minimal"
        assert preset.pipelines == []
        assert preset.variant is None
        assert preset.negative_prompt is None

    def test_load_preset_name_from_filename(self, tmp_path):
        """Test that name defaults to filename if not in frontmatter."""
        preset_content = """---
description: No name field
---

Body text.
"""
        preset_file = tmp_path / "my_preset.md"
        preset_file.write_text(preset_content)

        preset = load_preset(preset_file)
        assert preset.name == "my_preset"

    def test_load_preset_no_frontmatter_raises(self, tmp_path):
        """Test that files without frontmatter raise an error."""
        preset_file = tmp_path / "no_frontmatter.md"
        preset_file.write_text("Just plain text without frontmatter.")

        with pytest.raises(ValueError, match="must have YAML frontmatter"):
            load_preset(preset_file)

    def test_load_preset_not_found(self):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_preset("/nonexistent/path.md")

    def test_load_preset_string_pipeline(self, tmp_path):
        """Test that string pipeline is converted to list."""
        preset_content = """---
name: single_pipeline
pipelines: zimage
---

Body.
"""
        preset_file = tmp_path / "single.md"
        preset_file.write_text(preset_content)

        preset = load_preset(preset_file)
        assert preset.pipelines == ["zimage"]


class TestLoadPresetsFromDir:
    """Tests for loading multiple presets from a directory."""

    def test_load_from_directory(self, tmp_path):
        """Test loading all presets from a directory."""
        # Create preset files
        (tmp_path / "preset1.md").write_text("""---
name: preset1
---
Body 1.
""")
        (tmp_path / "preset2.md").write_text("""---
name: preset2
---
Body 2.
""")

        presets = load_presets_from_dir(tmp_path)
        assert len(presets) == 2
        assert "preset1" in presets
        assert "preset2" in presets

    def test_load_from_subdirectories(self, tmp_path):
        """Test recursive loading from subdirectories."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        (tmp_path / "root.md").write_text("""---
name: root_preset
---
Root.
""")
        (subdir / "sub.md").write_text("""---
name: sub_preset
---
Sub.
""")

        presets = load_presets_from_dir(tmp_path, recursive=True)
        assert len(presets) == 2
        assert "root_preset" in presets
        assert "sub_preset" in presets

    def test_load_empty_directory(self, tmp_path):
        """Test loading from empty directory."""
        presets = load_presets_from_dir(tmp_path)
        assert presets == {}

    def test_load_nonexistent_directory(self, tmp_path):
        """Test loading from nonexistent directory returns empty dict."""
        presets = load_presets_from_dir(tmp_path / "nonexistent")
        assert presets == {}


class TestPresetRegistry:
    """Tests for the PresetRegistry class."""

    def test_registry_lazy_loading(self, tmp_path):
        """Test that registry loads presets lazily."""
        (tmp_path / "test.md").write_text("""---
name: test_preset
---
Body.
""")

        registry = PresetRegistry(tmp_path)
        # Presets should not be loaded yet
        assert registry._loaded is False

        # Access forces loading
        _ = registry.get("test_preset")
        assert registry._loaded is True

    def test_registry_get(self, tmp_path):
        """Test getting a preset by name."""
        (tmp_path / "test.md").write_text("""---
name: test_preset
---
Body.
""")

        registry = PresetRegistry(tmp_path)
        preset = registry.get("test_preset")
        assert preset is not None
        assert preset.name == "test_preset"

    def test_registry_get_missing(self, tmp_path):
        """Test getting a non-existent preset returns None."""
        registry = PresetRegistry(tmp_path)
        assert registry.get("nonexistent") is None

    def test_registry_list_for_pipeline(self, tmp_path):
        """Test listing presets for a pipeline."""
        (tmp_path / "zimage.md").write_text("""---
name: zimage_preset
pipelines: [zimage]
---
Z-Image.
""")
        (tmp_path / "ltx2.md").write_text("""---
name: ltx2_preset
pipelines: [ltx2]
---
LTX-2.
""")
        (tmp_path / "all.md").write_text("""---
name: all_preset
pipelines: [all]
---
All.
""")

        registry = PresetRegistry(tmp_path)
        zimage_presets = registry.list_for_pipeline("zimage")
        assert len(zimage_presets) == 2  # zimage_preset + all_preset
        names = [p.name for p in zimage_presets]
        assert "zimage_preset" in names
        assert "all_preset" in names
        assert "ltx2_preset" not in names

    def test_registry_list_for_pipeline_with_variant(self, tmp_path):
        """Test listing presets with variant filter."""
        (tmp_path / "base.md").write_text("""---
name: base_preset
pipelines: [zimage]
variant: base
---
Base.
""")
        (tmp_path / "turbo.md").write_text("""---
name: turbo_preset
pipelines: [zimage]
variant: turbo
---
Turbo.
""")
        (tmp_path / "any.md").write_text("""---
name: any_preset
pipelines: [zimage]
---
Any variant.
""")

        registry = PresetRegistry(tmp_path)
        base_presets = registry.list_for_pipeline("zimage", variant="base")
        assert len(base_presets) == 2  # base_preset + any_preset
        names = [p.name for p in base_presets]
        assert "base_preset" in names
        assert "any_preset" in names
        assert "turbo_preset" not in names

    def test_registry_list_by_category(self, tmp_path):
        """Test listing presets by category."""
        (tmp_path / "quality1.md").write_text("""---
name: quality1
category: quality
---
Quality 1.
""")
        (tmp_path / "style1.md").write_text("""---
name: style1
category: style
---
Style 1.
""")

        registry = PresetRegistry(tmp_path)
        quality_presets = registry.list_by_category("quality")
        assert len(quality_presets) == 1
        assert quality_presets[0].name == "quality1"

    def test_registry_reload(self, tmp_path):
        """Test reloading presets from disk."""
        (tmp_path / "initial.md").write_text("""---
name: initial
---
Initial.
""")

        registry = PresetRegistry(tmp_path)
        assert registry.get("initial") is not None
        assert registry.get("added") is None

        # Add a new preset file
        (tmp_path / "added.md").write_text("""---
name: added
---
Added.
""")

        # Still using cached version
        assert registry.get("added") is None

        # Reload from disk
        registry.reload()
        assert registry.get("added") is not None

    def test_registry_contains(self, tmp_path):
        """Test __contains__ method."""
        (tmp_path / "test.md").write_text("""---
name: test_preset
---
Body.
""")

        registry = PresetRegistry(tmp_path)
        assert "test_preset" in registry
        assert "nonexistent" not in registry

    def test_registry_len(self, tmp_path):
        """Test __len__ method."""
        (tmp_path / "a.md").write_text("---\nname: a\n---\nA.")
        (tmp_path / "b.md").write_text("---\nname: b\n---\nB.")

        registry = PresetRegistry(tmp_path)
        assert len(registry) == 2


class TestGlobalRegistry:
    """Tests for the global registry singleton."""

    def setup_method(self):
        """Reset global registry before each test."""
        reset_preset_registry()

    def test_get_preset_registry_first_call(self, tmp_path):
        """Test that first call requires presets_dir."""
        (tmp_path / "test.md").write_text("---\nname: test\n---\nBody.")

        registry = get_preset_registry(tmp_path)
        assert registry is not None
        assert "test" in registry

    def test_get_preset_registry_subsequent_calls(self, tmp_path):
        """Test that subsequent calls reuse the same registry."""
        (tmp_path / "test.md").write_text("---\nname: test\n---\nBody.")

        registry1 = get_preset_registry(tmp_path)
        registry2 = get_preset_registry()  # No presets_dir needed
        assert registry1 is registry2

    def test_get_preset_registry_no_dir_first_call(self):
        """Test that first call without presets_dir raises error."""
        with pytest.raises(ValueError, match="presets_dir is required"):
            get_preset_registry()


# =============================================================================
# Tests for Testing Presets (presets/testing/)
# =============================================================================


class TestTestingPresets:
    """Tests for the test-specific presets in presets/testing/."""

    # Presets directory relative to repo root
    PRESETS_DIR = Path(__file__).parent.parent.parent / "presets"

    def setup_method(self):
        """Reset global registry before each test."""
        reset_preset_registry()

    def test_zimage_base_test_preset_exists(self):
        """Verify the zimage_base_test preset loads correctly."""
        preset = load_preset(self.PRESETS_DIR / "testing" / "zimage_base.md")

        assert preset.name == "zimage_base_test"
        assert preset.category == "testing"
        assert preset.pipelines == ["zimage"]
        assert preset.variant == "base"

        # Generation parameters
        assert preset.guidance_scale == 4.0
        assert preset.steps == 30
        assert preset.shift == 6.0

        # Test metadata
        assert preset.metadata["prompt"] == "A cat sleeping in sunlight"
        assert preset.metadata["seed"] == 42
        assert preset.metadata["height"] == 512
        assert preset.metadata["width"] == 512
        assert preset.metadata["min_variance"] == 500
        assert preset.metadata["max_variance"] == 6000

    def test_zimage_turbo_test_preset_exists(self):
        """Verify the zimage_turbo_test preset loads correctly."""
        preset = load_preset(self.PRESETS_DIR / "testing" / "zimage_turbo.md")

        assert preset.name == "zimage_turbo_test"
        assert preset.category == "testing"
        assert preset.variant == "turbo"

        # Turbo-specific parameters
        assert preset.guidance_scale == 0.0  # CFG baked in
        assert preset.steps == 9
        assert preset.shift == 3.0

    def test_flux2_distilled_test_preset_exists(self):
        """Verify the flux2_distilled_test preset loads correctly."""
        preset = load_preset(self.PRESETS_DIR / "testing" / "flux2_distilled.md")

        assert preset.name == "flux2_distilled_test"
        assert preset.category == "testing"
        assert preset.pipelines == ["flux2"]
        assert preset.variant == "distilled"

        # Distilled parameters
        assert preset.guidance_scale == 1.0
        assert preset.steps == 4

    def test_flux2_base_test_preset_exists(self):
        """Verify the flux2_base_test preset loads correctly."""
        preset = load_preset(self.PRESETS_DIR / "testing" / "flux2_base.md")

        assert preset.name == "flux2_base_test"
        assert preset.category == "testing"
        assert preset.variant == "base"

        # Base parameters
        assert preset.guidance_scale == 3.5
        assert preset.steps == 28

    def test_registry_finds_all_testing_presets(self):
        """Verify registry can find all testing presets."""
        registry = get_preset_registry(self.PRESETS_DIR)

        testing_presets = registry.list_by_category("testing")
        assert len(testing_presets) >= 4

        names = [p.name for p in testing_presets]
        assert "zimage_base_test" in names
        assert "zimage_turbo_test" in names
        assert "flux2_distilled_test" in names
        assert "flux2_base_test" in names

    def test_registry_filters_testing_presets_by_pipeline(self):
        """Verify registry can filter testing presets by pipeline."""
        registry = get_preset_registry(self.PRESETS_DIR)

        zimage_testing = [
            p for p in registry.list_for_pipeline("zimage")
            if p.category == "testing"
        ]
        assert len(zimage_testing) >= 2

        names = [p.name for p in zimage_testing]
        assert "zimage_base_test" in names
        assert "zimage_turbo_test" in names
        assert "flux2_distilled_test" not in names


class TestTestFixtureHelpers:
    """Tests for the test fixture helper functions in tests/fixtures/configs/."""

    def setup_method(self):
        """Reset global registry before each test."""
        reset_preset_registry()

    def test_get_test_preset_basic(self):
        """Test getting a specific test preset."""
        from tests.fixtures.configs.presets import get_test_preset

        preset = get_test_preset("zimage_base_test")
        assert preset.name == "zimage_base_test"
        assert preset.guidance_scale == 4.0
        assert preset.steps == 30

    def test_get_test_preset_with_param_override(self):
        """Test getting a preset with parameter override."""
        from tests.fixtures.configs.presets import get_test_preset

        preset = get_test_preset("zimage_base_test", guidance_scale=5.0)
        assert preset.guidance_scale == 5.0  # Overridden
        assert preset.steps == 30  # Original

    def test_get_test_preset_with_metadata_override(self):
        """Test getting a preset with metadata override."""
        from tests.fixtures.configs.presets import get_test_preset

        preset = get_test_preset("zimage_base_test", seed=123)
        assert preset.metadata["seed"] == 123  # Overridden
        assert preset.metadata["prompt"] == "A cat sleeping in sunlight"  # Original

    def test_get_test_preset_not_found_error(self):
        """Test error when preset not found."""
        from tests.fixtures.configs.presets import get_test_preset

        with pytest.raises(ValueError, match="not found"):
            get_test_preset("nonexistent_preset_xyz")

    def test_get_test_presets_by_category(self):
        """Test getting all test presets by category."""
        from tests.fixtures.configs.presets import get_test_presets_by_category

        presets = get_test_presets_by_category("testing")
        assert len(presets) >= 4

        names = [p.name for p in presets]
        assert "zimage_base_test" in names
        assert "flux2_distilled_test" in names

    def test_get_test_presets_for_pipeline(self):
        """Test getting test presets for a specific pipeline."""
        from tests.fixtures.configs.presets import get_test_presets_for_pipeline

        zimage_presets = get_test_presets_for_pipeline("zimage")
        assert len(zimage_presets) >= 2

        names = [p.name for p in zimage_presets]
        assert "zimage_base_test" in names
        assert "zimage_turbo_test" in names

    def test_get_test_presets_for_pipeline_with_variant(self):
        """Test getting test presets filtered by variant."""
        from tests.fixtures.configs.presets import get_test_presets_for_pipeline

        base_presets = get_test_presets_for_pipeline("zimage", variant="base")
        assert len(base_presets) >= 1

        for preset in base_presets:
            assert preset.variant is None or preset.variant == "base"

    def test_reset_test_registry(self):
        """Test that reset_test_registry clears the global registry."""
        from tests.fixtures.configs.presets import get_test_preset, reset_test_registry

        # Load a preset (initializes registry)
        preset = get_test_preset("zimage_base_test")
        assert preset is not None

        # Reset
        reset_test_registry()

        # This should still work (re-initializes registry)
        preset2 = get_test_preset("zimage_base_test")
        assert preset2 is not None
