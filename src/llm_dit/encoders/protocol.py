"""
Encoder Protocol definitions for model-agnostic text/vision encoding.

This module extends the existing TextEncoderBackend Protocol to support:
- Vision-language models (Qwen2.5-VL, Gemma3)
- Memory management (offloading, cleanup)
- Model identification (encoder type, capabilities)

Design goals:
1. Backwards compatible with existing TextEncoderBackend usage
2. Support for multiple encoder families (Qwen3, Qwen2.5-VL, Gemma3)
3. Unified interface for quantization, device placement, and offloading
4. Consistent hidden layer extraction across all encoder types
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import List, Optional, Protocol, Union, runtime_checkable

import torch
from PIL import Image

# Re-export EncodingOutput from backends.protocol for backwards compatibility
from llm_dit.backends.protocol import EncodingOutput


class EncoderType(str, Enum):
    """Supported encoder types."""

    QWEN3 = "qwen3"  # Qwen3-4B (text-only) for Z-Image
    QWEN3_VL = "qwen3_vl"  # Qwen3-VL (vision-language) future support
    GEMMA3 = "gemma3"  # Gemma 3-12B for LTX-2


class EncoderCapability(Enum):
    """Encoder capability flags."""

    TEXT_ENCODING = auto()  # Can encode text to embeddings
    VISION_ENCODING = auto()  # Can encode images (VL models)
    TEXT_GENERATION = auto()  # Can generate text (prompt rewriting)
    HIDDEN_LAYER_SELECTION = auto()  # Supports extracting from specific layers


@dataclass
class EncoderInfo:
    """Information about an encoder's configuration and capabilities."""

    encoder_type: EncoderType
    model_id: str
    hidden_dim: int
    max_sequence_length: int
    capabilities: set[EncoderCapability] = field(default_factory=set)
    quantization: Optional[str] = None
    device: Optional[torch.device] = None
    dtype: Optional[torch.dtype] = None

    @property
    def supports_vision(self) -> bool:
        """Whether this encoder supports image inputs."""
        return EncoderCapability.VISION_ENCODING in self.capabilities

    @property
    def supports_generation(self) -> bool:
        """Whether this encoder supports text generation."""
        return EncoderCapability.TEXT_GENERATION in self.capabilities


@runtime_checkable
class TextEncoderProtocol(Protocol):
    """
    Base protocol for text encoders.

    This extends the existing TextEncoderBackend pattern to support
    model identification and memory management.

    All encoders must implement:
    - encode(): Encode text (and optionally images) to embeddings
    - info: Encoder information and capabilities
    - offload(): Move model to CPU and free GPU memory
    - to(): Move model to specified device
    """

    @property
    def info(self) -> EncoderInfo:
        """Get encoder information and capabilities."""
        ...

    @property
    def embedding_dim(self) -> int:
        """
        Return the embedding dimension.

        Examples:
        - Qwen3-4B: 2560
        - Qwen2.5-VL 7B: 3584
        - Gemma3-12B: 4096 (video) / 2048 (audio)
        """
        ...

    @property
    def max_sequence_length(self) -> int:
        """Return maximum supported sequence length."""
        ...

    @property
    def device(self) -> torch.device:
        """Return the device the model is on."""
        ...

    @property
    def dtype(self) -> torch.dtype:
        """Return the model's dtype."""
        ...

    def encode(
        self,
        texts: List[str],
        images: Optional[List[Image.Image]] = None,
        return_padded: bool = False,
        layer_index: int = -2,
    ) -> EncodingOutput:
        """
        Encode text (and optionally images) to embeddings.

        Args:
            texts: List of pre-formatted text strings (with chat template).
            images: Optional list of PIL Images for vision-language models.
                   Pass None for text-only encoding.
            return_padded: If True, also return padded batch tensors.
            layer_index: Which hidden layer to extract (default: -2).

        Returns:
            EncodingOutput with variable-length embeddings per input.

        Raises:
            ValueError: If images provided but encoder doesn't support vision.
        """
        ...

    def offload(self) -> None:
        """
        Offload model to CPU and free GPU memory.

        Call this after encoding to release VRAM for other components.
        The model remains usable but will be slower until moved back to GPU.
        """
        ...

    def to(self, device: torch.device) -> "TextEncoderProtocol":
        """Move model to device."""
        ...


@runtime_checkable
class VisionLanguageEncoderProtocol(TextEncoderProtocol, Protocol):
    """
    Extended protocol for vision-language encoders.

    Adds vision-specific methods for processing images alongside text.
    Implemented by Qwen2.5-VL, Gemma3, and future VL models.
    """

    def encode_image(
        self,
        images: List[Image.Image],
        return_padded: bool = False,
    ) -> EncodingOutput:
        """
        Encode images only (no text).

        Useful for extracting visual features independently.

        Args:
            images: List of PIL Images to encode.
            return_padded: If True, return padded batch tensors.

        Returns:
            EncodingOutput with image embeddings.
        """
        ...

    def get_image_tokens(self, image: Image.Image) -> int:
        """
        Get the number of tokens an image will consume.

        Useful for planning sequence lengths and prompt truncation.

        Args:
            image: PIL Image to analyze.

        Returns:
            Estimated number of tokens the image will produce.
        """
        ...


@runtime_checkable
class GenerativeEncoderProtocol(TextEncoderProtocol, Protocol):
    """
    Extended protocol for encoders that support text generation.

    Enables using the same model for prompt rewriting, captioning, etc.
    """

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.6,
        top_p: float = 0.95,
        **kwargs,
    ) -> str:
        """
        Generate text using the loaded model.

        Args:
            prompt: User prompt/message.
            system_prompt: Optional system prompt.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            **kwargs: Additional generation parameters.

        Returns:
            Generated text.
        """
        ...


# Type alias for any encoder
AnyEncoder = Union[TextEncoderProtocol, VisionLanguageEncoderProtocol, GenerativeEncoderProtocol]
