"""
Shared infrastructure for Qwen3 encoders.

Last Updated: 2026-02-01

This module provides the common base functionality shared between
Qwen3Encoder (Z-Image) and Qwen3Flux2Encoder (FLUX.2 Klein).

Extracted shared code:
- Device management (to(), device property, dtype property)
- Offload logic (gc.collect(), cuda.empty_cache())
- Model loading patterns

This is part of the codebase unification effort - see
internal/docs/architecture/encoder_analysis.md for details.
"""

import gc
import logging
from abc import ABC, abstractmethod
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class Qwen3EncoderMixin:
    """
    Mixin providing shared Qwen3 encoder functionality.

    This mixin is designed to be inherited by both:
    - Qwen3Encoder (Z-Image) - wrapper around TransformersBackend
    - Qwen3Flux2Encoder (FLUX.2) - direct nn.Module implementation

    It provides:
    - Standard offload logic with garbage collection
    - Device property pattern
    - dtype property pattern
    """

    def _offload_with_cleanup(self, model: torch.nn.Module) -> None:
        """
        Offload model to CPU and perform memory cleanup.

        This is the shared offload logic used by both encoders:
        1. Move model to CPU
        2. Run garbage collection
        3. Clear CUDA cache

        Args:
            model: PyTorch model to offload
        """
        logger.info("Offloading encoder to CPU...")
        model.to("cpu")

        # Force garbage collection
        gc.collect()

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Encoder offloaded, VRAM freed")

    @staticmethod
    def _get_device(model: torch.nn.Module) -> torch.device:
        """
        Get device from model parameters.

        Handles both single-device and device_map="auto" cases.

        Args:
            model: PyTorch model

        Returns:
            Device the model is on
        """
        if model is None:
            return torch.device("cpu")

        # Try model.device first (set by some loaders)
        if hasattr(model, "device") and model.device is not None:
            return model.device

        # For device_map="auto" with accelerate
        if hasattr(model, "hf_device_map"):
            try:
                first_param = next(model.parameters())
                return first_param.device
            except StopIteration:
                pass

        # Fallback to first parameter
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @staticmethod
    def _get_dtype(model: torch.nn.Module) -> torch.dtype:
        """
        Get dtype from model parameters.

        Args:
            model: PyTorch model

        Returns:
            Data type of model parameters
        """
        if model is None:
            return torch.bfloat16  # Default for modern models

        try:
            return next(model.parameters()).dtype
        except StopIteration:
            return torch.bfloat16


class Qwen3EncoderProtocol(ABC):
    """
    Abstract protocol for Qwen3-based text encoders.

    Both Z-Image and FLUX.2 encoders should implement this protocol.
    This defines the common interface for text encoding.
    """

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Get model device."""
        ...

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype:
        """Get model dtype."""
        ...

    @abstractmethod
    def offload(self) -> None:
        """Offload model to CPU and free VRAM."""
        ...

    @abstractmethod
    def to(self, device: torch.device) -> "Qwen3EncoderProtocol":
        """Move model to device."""
        ...


# Common output dimensions for reference
QWEN3_4B_HIDDEN_DIM = 2560  # Qwen3-4B hidden dimension
QWEN3_8B_HIDDEN_DIM = 4096  # Qwen3-8B hidden dimension

# Default layer selections
ZIMAGE_DEFAULT_LAYER = -2  # Z-Image uses second-to-last layer
KLEIN_DEFAULT_LAYERS = [9, 18, 27]  # FLUX.2 Klein uses 3 intermediate layers
