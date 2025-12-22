"""Standard prompt set for Z-Image ablation studies."""

from pathlib import Path
from typing import Any

import yaml


PROMPTS_DIR = Path(__file__).parent
STANDARD_PROMPTS_FILE = PROMPTS_DIR / "standard_prompts.yaml"


def load_standard_prompts(prompts_file: str | Path | None = None) -> dict[str, Any]:
    """Load the standard prompt set from YAML.

    Args:
        prompts_file: Path to prompts YAML file. If None, uses default.

    Returns:
        dict with 'version', 'prompts', and 'metadata' keys.
    """
    path = Path(prompts_file) if prompts_file else STANDARD_PROMPTS_FILE
    with open(path) as f:
        return yaml.safe_load(f)


def get_prompts_by_category(
    category: str, prompts_file: str | Path | None = None
) -> list[dict[str, Any]]:
    """Get all prompts in a specific category.

    Args:
        category: One of simple_objects, animals, humans, scenes,
                  landscapes, artistic_styles, lighting, abstract,
                  technical, text_rendering
        prompts_file: Path to prompts YAML file. If None, uses default.

    Returns:
        List of prompt dicts with id, category, prompt, test_elements, difficulty
    """
    data = load_standard_prompts(prompts_file)
    return [p for p in data["prompts"] if p["category"] == category]


def get_prompts_by_difficulty(difficulty: str) -> list[dict[str, Any]]:
    """Get all prompts of a specific difficulty.

    Args:
        difficulty: One of easy, medium, hard

    Returns:
        List of prompt dicts
    """
    data = load_standard_prompts()
    return [p for p in data["prompts"] if p["difficulty"] == difficulty]


def get_prompt_by_id(prompt_id: str) -> dict[str, Any] | None:
    """Get a specific prompt by its ID.

    Args:
        prompt_id: e.g., "animal_001", "scene_003"

    Returns:
        Prompt dict or None if not found
    """
    data = load_standard_prompts()
    for p in data["prompts"]:
        if p["id"] == prompt_id:
            return p
    return None


def get_all_prompt_texts() -> list[str]:
    """Get just the prompt text strings (for quick iteration).

    Returns:
        List of prompt strings
    """
    data = load_standard_prompts()
    return [p["prompt"] for p in data["prompts"]]


def get_prompt_ids() -> list[str]:
    """Get all prompt IDs.

    Returns:
        List of prompt IDs
    """
    data = load_standard_prompts()
    return [p["id"] for p in data["prompts"]]


def get_categories() -> list[str]:
    """Get all available categories.

    Returns:
        List of category names
    """
    return [
        "simple_objects",
        "animals",
        "humans",
        "scenes",
        "landscapes",
        "artistic_styles",
        "lighting",
        "abstract",
        "technical",
        "text_rendering",
    ]


# Convenience exports
__all__ = [
    "load_standard_prompts",
    "get_prompts_by_category",
    "get_prompts_by_difficulty",
    "get_prompt_by_id",
    "get_all_prompt_texts",
    "get_prompt_ids",
    "get_categories",
    "STANDARD_PROMPTS_FILE",
]
