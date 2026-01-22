"""
Data module for LLM-DiT experiments.

Last Updated: 2026-01-17

Contains prompt datasets and evaluation data for video generation experiments.
"""

from llm_dit.data.prompts import (
    # Official prompts
    OFFICIAL_PROMPTS,
    # Category prompts
    CATEGORY_PROMPTS,
    # Structured format prompts
    STRUCTURED_PROMPTS,
    # Legacy prompts
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

__all__ = [
    # Prompt dictionaries
    "OFFICIAL_PROMPTS",
    "CATEGORY_PROMPTS",
    "STRUCTURED_PROMPTS",
    "LEGACY_SHORT_PROMPTS",
    # Quick subsets
    "QUICK_OFFICIAL",
    "QUICK_CATEGORY",
    "QUICK_STRUCTURED",
    # Helper functions
    "get_official_prompts",
    "get_category_prompts",
    "get_all_prompts",
    "get_structured_prompts",
    "word_count",
    "validate_prompts",
]
