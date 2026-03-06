"""Cross-source config consistency validation.

last updated: 2026-02-12

Ensures that all test config sources (constants module, TOML overlays,
protocol configs) agree on parameter values. Catches drift between
independently maintained config systems.

Run with: uv run pytest tests/unit/test_config_consistency.py -v
"""

import sys
from pathlib import Path

import pytest

# Ensure project root is on path for test imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _coderef_available() -> bool:
    """Check if the LTX-2 reference repo is available for import."""
    coderef_path = _PROJECT_ROOT / "coderef" / "LTX-2" / "packages" / "ltx-pipelines" / "src"
    return coderef_path.exists()


class TestLTX2Constants:
    """Validate LTX-2 constants module values."""

    def test_reference_values_are_sane(self):
        """Reference values have valid ranges."""
        from tests.constants.ltx2 import (
            REFERENCE_CFG,
            REFERENCE_FPS,
            REFERENCE_FRAMES,
            REFERENCE_HEIGHT,
            REFERENCE_SEED,
            REFERENCE_STEPS,
            REFERENCE_WIDTH,
        )

        assert REFERENCE_HEIGHT % 32 == 0, "Height must be divisible by 32"
        assert REFERENCE_WIDTH % 32 == 0, "Width must be divisible by 32"
        assert (REFERENCE_FRAMES - 1) % 8 == 0, "Frames must be 8k+1"
        assert REFERENCE_STEPS > 0
        assert REFERENCE_CFG > 0
        assert REFERENCE_FPS > 0
        assert REFERENCE_SEED >= 0

    def test_smoke_tier_valid(self):
        """Smoke tier has valid dimensions and references correct constants."""
        from tests.constants.ltx2 import REFERENCE_CFG, REFERENCE_SEED, SMOKE

        assert SMOKE["height"] % 32 == 0
        assert SMOKE["width"] % 32 == 0
        assert (SMOKE["num_frames"] - 1) % 8 == 0
        assert SMOKE["guidance_scale"] == REFERENCE_CFG
        assert SMOKE["seed"] == REFERENCE_SEED

    def test_standard_tier_uses_reference_resolution(self):
        """Standard tier uses reference repo resolution."""
        from tests.constants.ltx2 import (
            REFERENCE_HEIGHT,
            REFERENCE_WIDTH,
            STANDARD,
        )

        assert STANDARD["height"] == REFERENCE_HEIGHT
        assert STANDARD["width"] == REFERENCE_WIDTH

    def test_full_smoke_matches_reference_resolution(self):
        """Full model smoke tier uses reference resolution."""
        from tests.constants.ltx2 import (
            FULL_SMOKE,
            REFERENCE_HEIGHT,
            REFERENCE_WIDTH,
        )

        assert FULL_SMOKE["height"] == REFERENCE_HEIGHT
        assert FULL_SMOKE["width"] == REFERENCE_WIDTH

    def test_full_reference_matches_all_reference_values(self):
        """Full reference tier matches all official reference values."""
        from tests.constants.ltx2 import (
            FULL_REFERENCE,
            REFERENCE_CFG,
            REFERENCE_FRAMES,
            REFERENCE_HEIGHT,
            REFERENCE_SEED,
            REFERENCE_STEPS,
            REFERENCE_WIDTH,
        )

        assert FULL_REFERENCE["height"] == REFERENCE_HEIGHT
        assert FULL_REFERENCE["width"] == REFERENCE_WIDTH
        assert FULL_REFERENCE["num_frames"] == REFERENCE_FRAMES
        assert FULL_REFERENCE["num_inference_steps"] == REFERENCE_STEPS
        assert FULL_REFERENCE["guidance_scale"] == REFERENCE_CFG
        assert FULL_REFERENCE["seed"] == REFERENCE_SEED

    def test_distilled_sigma_values_length(self):
        """Distilled sigma schedule has expected length."""
        from tests.constants.ltx2 import DISTILLED_SIGMA_VALUES

        assert len(DISTILLED_SIGMA_VALUES) == 9
        assert DISTILLED_SIGMA_VALUES[0] == 1.0
        assert DISTILLED_SIGMA_VALUES[-1] == 0.0


class TestLTX2TomlConsistency:
    """Ensure TOML overlays match constants module."""

    def _load_toml(self, name: str) -> dict:
        """Load a TOML overlay file."""
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib

        path = _PROJECT_ROOT / "tests" / "configs" / f"{name}.toml"
        with open(path, "rb") as f:
            return tomllib.load(f)

    def test_ltx2_smoke_toml_guidance(self):
        """Smoke TOML guidance_scale matches constants."""
        from tests.constants.ltx2 import SMOKE

        toml = self._load_toml("ltx2_smoke")
        assert toml["ltx2"]["guidance_scale"] == SMOKE["guidance_scale"]

    def test_ltx2_smoke_toml_dimensions(self):
        """Smoke TOML dimensions match constants."""
        from tests.constants.ltx2 import SMOKE

        toml = self._load_toml("ltx2_smoke")
        assert toml["ltx2"]["height"] == SMOKE["height"]
        assert toml["ltx2"]["width"] == SMOKE["width"]

    def test_ltx2_smoke_toml_frames(self):
        """Smoke TOML frame count matches constants."""
        from tests.constants.ltx2 import SMOKE

        toml = self._load_toml("ltx2_smoke")
        assert toml["ltx2"]["num_frames"] == SMOKE["num_frames"]

    def test_ltx2_standard_toml_dimensions(self):
        """Standard TOML uses reference landscape orientation."""
        from tests.constants.ltx2 import STANDARD

        toml = self._load_toml("ltx2_standard")
        assert toml["ltx2"]["height"] == STANDARD["height"]
        assert toml["ltx2"]["width"] == STANDARD["width"]

    def test_ltx2_standard_toml_guidance(self):
        """Standard TOML guidance_scale matches constants."""
        from tests.constants.ltx2 import STANDARD

        toml = self._load_toml("ltx2_standard")
        assert toml["ltx2"]["guidance_scale"] == STANDARD["guidance_scale"]

    def test_ltx2_reference_toml_matches_full_reference(self):
        """Reference TOML matches full reference constants."""
        from tests.constants.ltx2 import FULL_REFERENCE

        toml = self._load_toml("ltx2_reference")
        assert toml["ltx2"]["height"] == FULL_REFERENCE["height"]
        assert toml["ltx2"]["width"] == FULL_REFERENCE["width"]
        assert toml["ltx2"]["num_frames"] == FULL_REFERENCE["num_frames"]
        assert toml["ltx2"]["num_inference_steps"] == FULL_REFERENCE["num_inference_steps"]
        assert toml["ltx2"]["guidance_scale"] == FULL_REFERENCE["guidance_scale"]


class TestLTX2ProtocolConsistency:
    """Ensure protocol.py configs match constants module."""

    def test_smoke_config_matches_full_smoke(self):
        """Protocol SMOKE_CONFIG matches FULL_SMOKE constants."""
        from tests.backends.protocol import SMOKE_CONFIG
        from tests.constants.ltx2 import FULL_SMOKE

        assert SMOKE_CONFIG.height == FULL_SMOKE["height"]
        assert SMOKE_CONFIG.width == FULL_SMOKE["width"]
        assert SMOKE_CONFIG.num_frames == FULL_SMOKE["num_frames"]
        assert SMOKE_CONFIG.num_inference_steps == FULL_SMOKE["num_inference_steps"]
        assert SMOKE_CONFIG.guidance_scale == FULL_SMOKE["guidance_scale"]
        assert SMOKE_CONFIG.seed == FULL_SMOKE["seed"]
        assert SMOKE_CONFIG.fp8 == FULL_SMOKE["fp8"]

    def test_reference_config_matches_full_reference(self):
        """Protocol REFERENCE_CONFIG matches FULL_REFERENCE constants."""
        from tests.backends.protocol import REFERENCE_CONFIG
        from tests.constants.ltx2 import FULL_REFERENCE

        assert REFERENCE_CONFIG.height == FULL_REFERENCE["height"]
        assert REFERENCE_CONFIG.width == FULL_REFERENCE["width"]
        assert REFERENCE_CONFIG.num_frames == FULL_REFERENCE["num_frames"]
        assert REFERENCE_CONFIG.num_inference_steps == FULL_REFERENCE["num_inference_steps"]
        assert REFERENCE_CONFIG.guidance_scale == FULL_REFERENCE["guidance_scale"]
        assert REFERENCE_CONFIG.seed == FULL_REFERENCE["seed"]
        assert REFERENCE_CONFIG.fp8 == FULL_REFERENCE["fp8"]

    def test_short_config_is_smoke_alias(self):
        """SHORT_CONFIG is an alias for SMOKE_CONFIG."""
        from tests.backends.protocol import SHORT_CONFIG, SMOKE_CONFIG

        assert SHORT_CONFIG is SMOKE_CONFIG


class TestLTX2PromptConsistency:
    """Ensure prompt re-exports match constants."""

    def test_smoke_prompt_matches(self):
        """Fixture smoke prompt matches constants."""
        from tests.constants.ltx2 import SMOKE_PROMPT
        from tests.fixtures.prompts.ltx2 import SMOKE_TEST_PROMPT

        assert SMOKE_TEST_PROMPT == SMOKE_PROMPT

    def test_reference_prompts_superset(self):
        """Fixture reference prompts contain all constants prompts."""
        from tests.constants.ltx2 import REFERENCE_PROMPTS as CONST_PROMPTS
        from tests.fixtures.prompts.ltx2 import REFERENCE_PROMPTS as FIXTURE_PROMPTS

        for key, value in CONST_PROMPTS.items():
            assert key in FIXTURE_PROMPTS, f"Missing prompt key: {key}"
            assert FIXTURE_PROMPTS[key] == value, f"Prompt mismatch for '{key}'"


class TestLTX2ReferenceRepo:
    """Validate constants against the official LTX-2 reference repo."""

    @pytest.mark.skipif(not _coderef_available(), reason="coderef/LTX-2 not present")
    def test_constants_match_reference_repo(self):
        """Our constants match the official LTX-2 reference repo.

        Parses the constants file directly (via regex) instead of importing,
        because importing ltx_pipelines triggers ltx_core which may not be
        installed in this environment.

        The reference repo uses a PipelineParams dataclass with defaults,
        then overrides via `LTX_2_3_PARAMS = replace(...)`.
        """
        import re

        from tests.constants.ltx2 import (
            REFERENCE_CFG,
            REFERENCE_FPS,
            REFERENCE_FRAMES,
            REFERENCE_HEIGHT,
            REFERENCE_SEED,
            REFERENCE_STEPS,
            REFERENCE_WIDTH,
        )

        constants_path = (
            _PROJECT_ROOT
            / "coderef"
            / "LTX-2"
            / "packages"
            / "ltx-pipelines"
            / "src"
            / "ltx_pipelines"
            / "utils"
            / "constants.py"
        )

        source = constants_path.read_text()

        # Extract PipelineParams dataclass field defaults (e.g. "seed: int = 10")
        def _extract_field(name: str, cast=int):
            m = re.search(rf"{name}\s*:\s*\w+\s*=\s*([0-9.]+)", source)
            assert m, f"Could not find {name} in reference constants"
            return cast(m.group(1))

        assert REFERENCE_HEIGHT == _extract_field("stage_1_height")
        assert REFERENCE_WIDTH == _extract_field("stage_1_width")
        assert REFERENCE_FRAMES == _extract_field("num_frames")
        assert REFERENCE_SEED == _extract_field("seed")
        assert REFERENCE_FPS == _extract_field("frame_rate", cast=float)

        # num_inference_steps is overridden in LTX_2_3_PARAMS = replace(...)
        # Find the replace() call and extract the overridden value
        replace_match = re.search(
            r"LTX_2_3_PARAMS\s*=\s*replace\(.*?num_inference_steps\s*=\s*(\d+)",
            source,
            re.DOTALL,
        )
        assert replace_match, "Could not find num_inference_steps in LTX_2_3_PARAMS"
        assert REFERENCE_STEPS == int(replace_match.group(1))

        # CFG scale is inside MultiModalGuiderParams(...) in PipelineParams defaults
        cfg_match = re.search(
            r"video_guider_params.*?cfg_scale\s*=\s*([0-9.]+)",
            source,
            re.DOTALL,
        )
        assert cfg_match, "Could not find cfg_scale in video_guider_params"
        assert REFERENCE_CFG == float(cfg_match.group(1))


class TestConfigTomlExample:
    """Ensure config.toml.example matches constants."""

    def test_ltx2_defaults_match_reference(self):
        """config.toml.example LTX-2 section matches reference values."""
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib

        from tests.constants.ltx2 import REFERENCE_CFG, REFERENCE_HEIGHT, REFERENCE_WIDTH

        path = _PROJECT_ROOT / "config.toml.example"
        with open(path, "rb") as f:
            config = tomllib.load(f)

        ltx2 = config["ltx2"]
        assert ltx2["height"] == REFERENCE_HEIGHT
        assert ltx2["width"] == REFERENCE_WIDTH
        assert ltx2["guidance_scale"] == REFERENCE_CFG


class TestFLUX2Constants:
    """Basic validation of FLUX.2 constants."""

    def test_smoke_valid(self):
        from tests.constants.flux2 import SMOKE

        assert SMOKE["height"] > 0
        assert SMOKE["width"] > 0
        assert SMOKE["num_inference_steps"] > 0

    def test_tiers_increase_in_quality(self):
        from tests.constants.flux2 import REFERENCE, SMOKE, STANDARD

        assert SMOKE["height"] <= STANDARD["height"] <= REFERENCE["height"]
        assert SMOKE["num_inference_steps"] <= STANDARD["num_inference_steps"]


class TestZImageConstants:
    """Basic validation of Z-Image constants."""

    def test_smoke_valid(self):
        from tests.constants.zimage import SMOKE

        assert SMOKE["height"] > 0
        assert SMOKE["width"] > 0
        assert SMOKE["variant"] == "turbo"

    def test_turbo_no_cfg(self):
        """Turbo variant should have guidance_scale=0.0 (no CFG)."""
        from tests.constants.zimage import SMOKE, STANDARD_TURBO

        assert SMOKE["guidance_scale"] == 0.0
        assert STANDARD_TURBO["guidance_scale"] == 0.0

    def test_base_has_cfg(self):
        """Base variant should have positive CFG."""
        from tests.constants.zimage import REFERENCE_BASE, STANDARD_BASE

        assert STANDARD_BASE["guidance_scale"] > 0
        assert REFERENCE_BASE["guidance_scale"] > 0
