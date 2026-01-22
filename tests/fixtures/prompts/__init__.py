"""
Test prompts for video generation models.

Last Updated: 2026-01-18

Provides test prompts organized by model/pipeline. Each model module
(ltx2.py, etc.) provides prompts appropriate for that model's format.

Usage:
    from tests.fixtures.prompts import ltx2

    # Get a single quick test prompt
    prompt = ltx2.get_smoke_test_prompt()

    # Get reference prompts for 1:1 comparison
    prompts = ltx2.get_reference_prompts()

    # Get all available prompts
    prompts = ltx2.get_all_prompts()
"""

from . import ltx2

__all__ = ["ltx2"]
