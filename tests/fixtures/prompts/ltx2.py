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
    CATEGORY_PROMPTS,
    LEGACY_SHORT_PROMPTS,
    # Prompt dictionaries
    OFFICIAL_PROMPTS,
    QUICK_CATEGORY,
    # Quick subsets
    QUICK_OFFICIAL,
    QUICK_STRUCTURED,
    STRUCTURED_PROMPTS,
    get_all_prompts,
    get_category_prompts,
    # Helper functions
    get_official_prompts,
    get_structured_prompts,
    validate_prompts,
    word_count,
)

# =============================================================================
# Test-Specific Prompts (canonical source: tests.constants.ltx2)
# =============================================================================

from tests.constants.ltx2 import REFERENCE_PROMPTS as _CONSTANTS_REFERENCE_PROMPTS
from tests.constants.ltx2 import SMOKE_PROMPT

# Re-export canonical smoke prompt under the legacy name
SMOKE_TEST_PROMPT = SMOKE_PROMPT

# Reference prompts: start with canonical set, add extended prompts here
REFERENCE_PROMPTS = {
    **_CONSTANTS_REFERENCE_PROMPTS,
    "expanded_cat_backrooms": "Style: cinematic-realistic. In a vast, empty room characterized by repetitive yellow wallpaper and a damp-looking beige carpet, a ginger tabby cat stands in the center of the frame. The ceiling consists of industrial grid tiles with recessed fluorescent panels that emit a steady, low-frequency electrical hum. The cat crouches low to the floor, its muscles tensing as it prepares for movement. It leaps upward, tucking its body into a tight rotation and performing a complete backflip in mid-air. The cat's fur ripples slightly during the turn, and its tail swishes to maintain balance. As it descends, the cat extends its paws and lands silently on the soft carpet. The constant buzzing of the overhead lights remains the only sound in the sterile environment as the cat stands still, looking down the endless, repeating hallway.",
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
