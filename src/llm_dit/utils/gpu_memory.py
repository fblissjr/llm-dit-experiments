"""
GPU Memory Management Utilities for llm-dit.

Last Updated: 2026-01-23

IMPORTANT: This module is PURE PYTORCH only.
Do NOT import or use any diffusers components.
For diffusers-based pipelines, use diffusers' own utilities.

Provides:
- @free_gpu_memory_context: Decorator/context manager for automatic GPU cleanup
- verify_pure_pytorch: Runtime check to ensure model is not a diffusers wrapper

Ported from LTX-2 trainer patterns:
- Sequential loading pattern: load -> compute -> unload
- Block-by-block operations with memory cleanup
- Stage-based memory management for 24GB constraint

Usage:
    # As decorator
    @free_gpu_memory_context(after=True)
    def encode_prompts(encoder, prompts):
        return encoder.encode(prompts)

    # As context manager
    with free_gpu_memory_context():
        embeddings = encoder.encode(prompts)
    # Memory freed automatically on exit

    # Verify pure PyTorch
    verify_pure_pytorch(model)  # Raises if diffusers model
"""

import functools
import gc
import logging
from contextlib import contextmanager
from typing import Callable, Optional, TypeVar, Union

import torch

logger = logging.getLogger(__name__)


def cleanup_gpu_memory(synchronize: bool = True, log: bool = False) -> float:
    """
    Aggressively free GPU memory.

    This combines garbage collection with CUDA cache clearing and
    optional synchronization. More thorough than simple gc.collect().

    Args:
        synchronize: If True, call cuda.synchronize() before clearing cache.
            This ensures all pending operations complete first.
        log: If True, log memory before/after cleanup.

    Returns:
        Amount of memory freed in GB.

    Example:
        >>> del text_encoder
        >>> freed = cleanup_gpu_memory(log=True)
        # [Memory] Freed 11.5GB (was 19.2GB, now 7.7GB)
    """
    if not torch.cuda.is_available():
        gc.collect()
        return 0.0

    # Get memory before cleanup
    before = torch.cuda.memory_allocated() / 1e9

    # Python garbage collection first (releases Python references)
    gc.collect()

    # Synchronize to ensure all CUDA ops complete
    if synchronize:
        torch.cuda.synchronize()

    # Clear CUDA cache
    torch.cuda.empty_cache()

    # Get memory after cleanup
    after = torch.cuda.memory_allocated() / 1e9
    freed = before - after

    if log and freed > 0.1:  # Only log if > 100MB freed
        logger.info(f"[Memory] Freed {freed:.1f}GB (was {before:.1f}GB, now {after:.1f}GB)")

    return freed


class FreeGPUMemoryContext:
    """
    Context manager for automatic GPU memory cleanup.

    Cleans up memory on entry, exit, or both. Useful for:
    - Sequential model loading (encode -> unload -> load transformer)
    - Stage-based processing (clear between stages)
    - Memory-constrained operations

    Args:
        before: If True, cleanup before the block executes.
        after: If True, cleanup after the block completes (default: True).
        log: If True, log memory changes.

    Example:
        # Cleanup after encoding stage
        with FreeGPUMemoryContext(after=True, log=True):
            embeddings = encoder.encode(prompts)
        # Memory freed, ready to load transformer
    """

    def __init__(
        self,
        before: bool = False,
        after: bool = True,
        log: bool = False,
    ):
        self.before = before
        self.after = after
        self.log = log
        self._entry_memory: float = 0.0

    def __enter__(self) -> "FreeGPUMemoryContext":
        if torch.cuda.is_available():
            self._entry_memory = torch.cuda.memory_allocated() / 1e9
        if self.before:
            cleanup_gpu_memory(log=self.log)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self.after:
            cleanup_gpu_memory(log=self.log)
        return False  # Don't suppress exceptions


def free_gpu_memory_context(
    before: bool = False,
    after: bool = True,
    log: bool = False,
) -> Union[FreeGPUMemoryContext, Callable]:
    """
    Decorator/context manager factory for GPU memory cleanup.

    Can be used as:
    1. Context manager: with free_gpu_memory_context(after=True): ...
    2. Decorator: @free_gpu_memory_context(after=True)

    Args:
        before: Cleanup before execution.
        after: Cleanup after execution (default: True).
        log: Log memory changes.

    Returns:
        FreeGPUMemoryContext when called without arguments,
        or a decorator when called with arguments.

    Examples:
        # As context manager
        with free_gpu_memory_context(after=True):
            embeddings = encoder.encode(prompts)

        # As decorator
        @free_gpu_memory_context(after=True)
        def encode_and_cleanup(encoder, prompts):
            return encoder.encode(prompts)

        # Decorator with logging
        @free_gpu_memory_context(after=True, log=True)
        def load_and_encode():
            encoder = load_encoder()
            return encoder.encode(prompts)
    """
    ctx = FreeGPUMemoryContext(before=before, after=after, log=log)
    return ctx


F = TypeVar("F", bound=Callable)


def free_gpu_memory_after(func: F) -> F:
    """
    Simple decorator that frees GPU memory after function execution.

    This is the most common pattern - cleanup after an operation completes.

    Args:
        func: Function to wrap.

    Returns:
        Wrapped function that cleans up GPU memory after execution.

    Example:
        @free_gpu_memory_after
        def encode_prompts(encoder, prompts):
            return encoder.encode(prompts)
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        finally:
            cleanup_gpu_memory(log=False)

    return wrapper  # type: ignore[return-value]


def verify_pure_pytorch(model: torch.nn.Module, context: str = "") -> None:
    """
    Verify that a model is pure PyTorch, not a diffusers wrapper.

    This is a safety check to ensure we don't accidentally mix
    diffusers and pure PyTorch code paths.

    Args:
        model: Model to verify.
        context: Optional context string for error message.

    Raises:
        ValueError: If model is from diffusers.

    Example:
        # In pure PyTorch code path
        verify_pure_pytorch(transformer, "generate_video")
        # Proceeds if pure PyTorch, raises if diffusers
    """
    module_name = type(model).__module__
    if "diffusers" in module_name:
        ctx_msg = f" ({context})" if context else ""
        raise ValueError(
            f"Expected pure PyTorch model{ctx_msg}, got diffusers type: {type(model)}. "
            "Use the appropriate diffusers pipeline instead, or load pure PyTorch components."
        )


@contextmanager
def staged_gpu_loading(stages: list[str], log: bool = True):
    """
    Context manager for multi-stage GPU loading with automatic cleanup.

    Tracks memory usage across stages and logs transitions.
    Useful for sequential loading patterns.

    Args:
        stages: List of stage names for logging.
        log: Whether to log stage transitions.

    Yields:
        A callable to advance to the next stage.

    Example:
        with staged_gpu_loading(["text_encoder", "transformer", "vae"]) as advance:
            encoder = load_encoder()
            embeddings = encoder.encode(prompts)
            advance()  # Cleanup and log "text_encoder -> transformer"

            transformer = load_transformer()
            latents = denoise(transformer, embeddings)
            advance()  # Cleanup and log "transformer -> vae"

            vae = load_vae()
            video = decode(vae, latents)
    """
    current_idx = 0

    def advance():
        nonlocal current_idx
        if current_idx >= len(stages) - 1:
            return

        from_stage = stages[current_idx]
        to_stage = stages[current_idx + 1]

        freed = cleanup_gpu_memory(log=False)

        if log:
            logger.info(f"[Stage] {from_stage} -> {to_stage} (freed {freed:.1f}GB)")

        current_idx += 1

    try:
        if log and stages:
            logger.info(f"[Stage] Starting: {stages[0]}")
        yield advance
    finally:
        freed = cleanup_gpu_memory(log=False)
        if log:
            logger.info(f"[Stage] Complete (final cleanup freed {freed:.1f}GB)")


def get_peak_memory_gb() -> float:
    """
    Get peak GPU memory usage since last reset.

    Returns:
        Peak memory in GB, or 0.0 if no CUDA device.
    """
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    return 0.0


def reset_peak_memory() -> None:
    """Reset peak memory tracking."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
