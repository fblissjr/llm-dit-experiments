"""
Backend abstraction for portable LTX-2 tests.

Last Updated: 2026-01-19

This module provides a unified interface for video generation that works with
both our llm-dit implementation and the official LTX-2 repo. This enables
1:1 baseline comparison by running the same tests with either backend.

Auto-Detection:
    The module automatically detects which backend is available:
    1. If `llm_dit` package is available -> uses llm_dit backend
    2. If `ltx_pipelines` package is available -> uses LTX-2 official backend
    3. Environment variable LLM_DIT_TEST_BACKEND can override (values: "llm_dit", "ltx2")

Usage in Tests:
    from tests.backends import get_backend, GenerationConfig

    backend = get_backend()
    config = GenerationConfig(
        num_frames=121,
        height=512,
        width=768,
        num_inference_steps=30,
        guidance_scale=4.0,
        seed=10,
    )
    result = backend.generate_video(prompt="A cat walking", config=config)

Portability:
    To run tests in the official LTX-2 repo:
    1. Copy `tests/` directory to LTX-2 repo
    2. Tests will auto-detect and use ltx2 backend
    3. Same prompts, configs, assertions -> 1:1 comparison
"""

import os
from typing import TYPE_CHECKING

from .protocol import (
    Backend,
    GenerationConfig,
    GenerationInputs,
    GenerationResult,
    GenerationStats,
    # Standard configs (single source of truth)
    REFERENCE_CONFIG,
    SHORT_CONFIG,
    SMOKE_CONFIG,
    CONFIG_MEMORY_ESTIMATES,
)

if TYPE_CHECKING:
    from .llm_dit_backend import LLMDitBackend
    from .ltx2_backend import LTX2Backend

__all__ = [
    # Core types
    "Backend",
    "GenerationConfig",
    "GenerationInputs",
    "GenerationResult",
    "GenerationStats",
    # Standard configs (single source of truth - DO NOT duplicate)
    "REFERENCE_CONFIG",
    "SHORT_CONFIG",
    "SMOKE_CONFIG",
    "CONFIG_MEMORY_ESTIMATES",
    # Backend functions
    "get_backend",
    "get_backend_name",
    "is_llm_dit_available",
    "is_ltx2_available",
]


def is_llm_dit_available() -> bool:
    """Check if llm_dit package is available."""
    try:
        import llm_dit  # noqa: F401

        return True
    except ImportError:
        return False


def is_ltx2_available() -> bool:
    """Check if official LTX-2 packages are available."""
    try:
        import ltx_pipelines  # noqa: F401

        return True
    except ImportError:
        # Try adding coderef to path (for this repo)
        import sys
        from pathlib import Path

        # Find repo root (look for CLAUDE.md or .git)
        current = Path(__file__).parent
        for _ in range(10):
            if (current / "CLAUDE.md").exists() or (current / ".git").exists():
                coderef_core = current / "coderef/LTX-2/packages/ltx-core/src"
                coderef_pipelines = current / "coderef/LTX-2/packages/ltx-pipelines/src"
                if coderef_core.exists() and coderef_pipelines.exists():
                    if str(coderef_core) not in sys.path:
                        sys.path.insert(0, str(coderef_core))
                    if str(coderef_pipelines) not in sys.path:
                        sys.path.insert(0, str(coderef_pipelines))
                    try:
                        import ltx_pipelines  # noqa: F401

                        return True
                    except ImportError:
                        pass
                break
            current = current.parent
        return False


def get_backend_name() -> str:
    """Get the name of the backend that will be used.

    Returns:
        "llm_dit", "ltx2", or "none" if no backend available.
    """
    # Check environment override
    env_backend = os.environ.get("LLM_DIT_TEST_BACKEND", "").lower()
    if env_backend == "llm_dit" and is_llm_dit_available():
        return "llm_dit"
    if env_backend == "ltx2" and is_ltx2_available():
        return "ltx2"

    # Auto-detect (prefer llm_dit in this repo)
    if is_llm_dit_available():
        return "llm_dit"
    if is_ltx2_available():
        return "ltx2"
    return "none"


def get_backend() -> Backend:
    """Get the appropriate backend for the current environment.

    Returns:
        Backend instance (either LLMDitBackend or LTX2Backend).

    Raises:
        ImportError: If no backend is available.
    """
    backend_name = get_backend_name()

    if backend_name == "llm_dit":
        from .llm_dit_backend import LLMDitBackend

        return LLMDitBackend()
    elif backend_name == "ltx2":
        from .ltx2_backend import LTX2Backend

        return LTX2Backend()
    else:
        raise ImportError(
            "No video generation backend available. "
            "Install either llm_dit package or run from LTX-2 repo with ltx_pipelines."
        )
