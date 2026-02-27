"""Backward-compatibility shim for startup.py.

All functionality has been moved to llm_dit.model_manager.
This module re-exports the key symbols to avoid breaking existing imports.

Usage (new):
    from llm_dit.model_manager import ModelManager, LoadResult, build_dype_config

Usage (legacy, still works):
    from llm_dit.startup import PipelineLoader, LoadResult, build_dype_config
"""

from llm_dit.model_manager import (  # noqa: F401
    LoadResult,
    ModelManager,
    PipelineLoader,
    build_dype_config,
)

__all__ = [
    "LoadResult",
    "ModelManager",
    "PipelineLoader",
    "build_dype_config",
]
