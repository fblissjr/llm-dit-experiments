"""
LTX-2 test prompts.

Last Updated: 2026-01-18

Provides test prompts for LTX-2 video generation testing.
Re-exports from experiments/ltx2/prompts.py to avoid duplication.

Prompt Categories:
- SMOKE_TEST: Single fast prompt for quick validation
- REFERENCE: Official prompts from LTX-2 repo for 1:1 comparison
- OFFICIAL: Full set of verbatim official prompts
- CATEGORY: Properly formatted category prompts
- STRUCTURED: Format ablation prompts (prose, json, yaml, etc.)

Usage:
    from tests.fixtures.prompts import ltx2

    # Quick smoke test (single short prompt)
    prompt = ltx2.get_smoke_test_prompt()

    # Reference comparison (official prompts with official params)
    prompts = ltx2.get_reference_prompts()

    # Full prompt suite
    all_prompts = ltx2.get_all_prompts()
"""

import sys
from pathlib import Path

# Add experiments to path if not already there
_experiments_path = Path(__file__).parent.parent.parent.parent / "experiments"
if str(_experiments_path) not in sys.path:
    sys.path.insert(0, str(_experiments_path))

# Re-export from experiments/ltx2/prompts.py
from ltx2.prompts import (
    # Prompt dictionaries
    OFFICIAL_PROMPTS,
    CATEGORY_PROMPTS,
    STRUCTURED_PROMPTS,
    LEGACY_SHORT_PROMPTS,
    # Quick subsets
    QUICK_OFFICIAL,
    QUICK_CATEGORY,
    QUICK_STRUCTURED,
    # Helper functions
    get_official_prompts,
    get_category_prompts,
    get_all_prompts,
    get_structured_prompts,
    word_count,
    validate_prompts,
)


# =============================================================================
# Test-Specific Prompts
# =============================================================================

# Single prompt for fastest possible test
SMOKE_TEST_PROMPT = "A cat walking"

# Reference prompts for 1:1 comparison with official LTX-2
# These are the prompts used in official demos/examples
REFERENCE_PROMPTS = {
    "cat_walking": "A cat walking",
    "cat_playing": "A cat playing with a ball",
    "sunset": "A beautiful sunset over the ocean",
}


# =============================================================================
# Test Helper Functions
# =============================================================================

def get_smoke_test_prompt() -> str:
    """Get single prompt for fastest possible validation."""
    return SMOKE_TEST_PROMPT


def get_reference_prompts() -> dict[str, str]:
    """Get prompts for 1:1 reference comparison with official LTX-2.

    These are simple prompts used in official LTX-2 examples.
    Should be run with official reference parameters for comparison.
    """
    return REFERENCE_PROMPTS.copy()


def get_quick_prompts() -> dict[str, str]:
    """Get small subset of prompts for quick testing.

    Returns ~5 diverse prompts for reasonable coverage without
    running the full suite.
    """
    prompts = {}
    # Mix of official and category
    for key in QUICK_OFFICIAL[:2]:
        prompts[f"official_{key}"] = OFFICIAL_PROMPTS[key]
    for key in QUICK_CATEGORY[:2]:
        prompts[f"category_{key}"] = CATEGORY_PROMPTS[key]
    return prompts


def get_e2e_test_prompts() -> dict[str, str]:
    """Get prompts for end-to-end testing.

    Returns a curated set of prompts that exercise different
    aspects of video generation (motion, dialogue, lighting, etc.)
    """
    return {
        # Motion and action
        "action": OFFICIAL_PROMPTS["action_cinematic"],
        # Dialogue and character
        "dialogue": OFFICIAL_PROMPTS["comedy_dialogue"],
        # Style and animation
        "animation": OFFICIAL_PROMPTS["animation_pixar"],
        # Documentary/realistic
        "documentary": OFFICIAL_PROMPTS["documentary"],
    }
