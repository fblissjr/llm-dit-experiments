"""
Qwen3 Encoder implementation for Z-Image.

This encoder wraps the existing TransformersBackend to implement the new
TextEncoderProtocol while maintaining full backwards compatibility.

Qwen3-4B is a text-only model (no vision support), used as the text encoder
for Z-Image-Turbo. It provides:
- 2560-dimensional embeddings
- Text generation for prompt rewriting
- Hidden layer selection for experimentation
"""

import gc
import logging
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image

from llm_dit.backends.config import BackendConfig
from llm_dit.backends.transformers import TransformersBackend
from llm_dit.encoders.protocol import (
    EncoderCapability,
    EncoderInfo,
    EncoderType,
    EncodingOutput,
    GenerativeEncoderProtocol,
    TextEncoderProtocol,
)
from llm_dit.utils.embedding_cache import EmbeddingCache

logger = logging.getLogger(__name__)


class Qwen3Encoder:
    """
    Qwen3 text encoder for Z-Image.

    This is a thin wrapper around TransformersBackend that implements
    the new TextEncoderProtocol interface while maintaining full
    backwards compatibility.

    Implements:
    - TextEncoderProtocol: Basic text encoding
    - GenerativeEncoderProtocol: Text generation for prompt rewriting

    Example:
        encoder = Qwen3Encoder.from_pretrained("Tongyi-MAI/Z-Image-Turbo")
        output = encoder.encode(["A beautiful sunset"])
        embeddings = output.embeddings[0]  # [seq_len, 2560]

        # Prompt rewriting
        rewritten = encoder.generate("A cat", system_prompt="Enhance this...")
    """

    def __init__(
        self,
        backend: TransformersBackend,
        model_id: str,
    ):
        """
        Initialize Qwen3Encoder with a TransformersBackend.

        Args:
            backend: Initialized TransformersBackend instance.
            model_id: Model identifier (for info).
        """
        self._backend = backend
        self._model_id = model_id
        self._is_offloaded = False

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        model_subfolder: str = "text_encoder",
        tokenizer_subfolder: str = "tokenizer",
        quantization: Optional[str] = None,
        quantization_config: Optional[object] = None,
        device: str = "auto",
        dtype: str = "bfloat16",
        enable_cache: bool = False,
        cache_size: int = 100,
        **kwargs,
    ) -> "Qwen3Encoder":
        """
        Load Qwen3 encoder from pretrained model.

        Args:
            model_path: Path to model or HuggingFace ID.
            model_subfolder: Subfolder containing text encoder (default: "text_encoder").
            tokenizer_subfolder: Subfolder containing tokenizer (default: "tokenizer").
            quantization: Quantization mode (none, fp8-dynamic, fp8-weight-only, int8, int4).
            quantization_config: Optional quantization config object (legacy, usually None).
            device: Device to load model on (auto, cuda, cpu).
            dtype: Model dtype (bfloat16, float16).
            enable_cache: Enable embedding cache.
            cache_size: Cache size if enabled.
            **kwargs: Additional arguments for TransformersBackend.

        Returns:
            Initialized Qwen3Encoder.
        """
        # Create backend config
        config = BackendConfig.for_z_image(model_path, subfolder=model_subfolder)
        config.device = device
        config.dtype = dtype
        if quantization:
            config.quantization = quantization

        # Load backend
        backend = TransformersBackend.from_pretrained(
            model_path=model_path,
            model_subfolder=model_subfolder,
            tokenizer_subfolder=tokenizer_subfolder,
            config=config,
            quantization_config=quantization_config,
            enable_cache=enable_cache,
            cache_size=cache_size,
            **kwargs,
        )

        return cls(backend=backend, model_id=model_path)

    @property
    def info(self) -> EncoderInfo:
        """Get encoder information and capabilities."""
        capabilities = {
            EncoderCapability.TEXT_ENCODING,
            EncoderCapability.TEXT_GENERATION,
            EncoderCapability.HIDDEN_LAYER_SELECTION,
        }

        return EncoderInfo(
            encoder_type=EncoderType.QWEN3,
            model_id=self._model_id,
            hidden_dim=self._backend.embedding_dim,
            max_sequence_length=self._backend.max_sequence_length,
            capabilities=capabilities,
            quantization=self._backend.config.quantization,
            device=self._backend.device,
            dtype=self._backend.dtype,
        )

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension (2560 for Qwen3-4B)."""
        return self._backend.embedding_dim

    @property
    def max_sequence_length(self) -> int:
        """Return max sequence length."""
        return self._backend.max_sequence_length

    @property
    def device(self) -> torch.device:
        """Return model device."""
        return self._backend.device

    @property
    def dtype(self) -> torch.dtype:
        """Return model dtype."""
        return self._backend.dtype

    def encode(
        self,
        texts: List[str],
        images: Optional[List[Image.Image]] = None,
        return_padded: bool = False,
        layer_index: int = -2,
    ) -> EncodingOutput:
        """
        Encode text to embeddings.

        Args:
            texts: List of pre-formatted text strings.
            images: Must be None (Qwen3 doesn't support vision).
            return_padded: If True, return padded batch tensors.
            layer_index: Hidden layer to extract (default: -2).

        Returns:
            EncodingOutput with embeddings.

        Raises:
            ValueError: If images are provided (Qwen3 is text-only).
        """
        if images is not None:
            raise ValueError(
                "Qwen3Encoder does not support vision inputs. "
                "Use Gemma3Encoder for vision-language tasks."
            )

        return self._backend.encode(
            texts=texts,
            return_padded=return_padded,
            layer_index=layer_index,
        )

    def encode_blended(
        self,
        texts: List[str],
        layer_weights: dict[int, float],
        return_padded: bool = False,
    ) -> EncodingOutput:
        """
        Encode text using a weighted blend of multiple hidden layers.

        Args:
            texts: List of pre-formatted text strings.
            layer_weights: Dict mapping layer indices to weights.
            return_padded: If True, return padded batch tensors.

        Returns:
            EncodingOutput with blended embeddings.
        """
        return self._backend.encode_blended(
            texts=texts,
            layer_weights=layer_weights,
            return_padded=return_padded,
        )

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
        Generate text using Qwen3.

        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            **kwargs: Additional generation parameters.

        Returns:
            Generated text.
        """
        return self._backend.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )

    def offload(self) -> None:
        """
        Offload model to CPU and free GPU memory.

        Call this after encoding to release VRAM for other components.
        """
        if self._is_offloaded:
            logger.debug("Qwen3Encoder already offloaded")
            return

        logger.info("Offloading Qwen3 encoder to CPU...")
        self._backend.model.to("cpu")
        self._is_offloaded = True

        # Force garbage collection and clear CUDA cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Qwen3 encoder offloaded, VRAM freed")

    def to(self, device: torch.device) -> "Qwen3Encoder":
        """Move model to device."""
        self._backend = self._backend.to(device)
        self._is_offloaded = (device.type == "cpu")
        return self

    # Expose cache management from backend

    @property
    def cache_enabled(self) -> bool:
        """Check if caching is enabled."""
        return self._backend.cache_enabled

    def enable_cache(self, max_size: int = 100) -> EmbeddingCache:
        """Enable embedding caching."""
        return self._backend.enable_cache(max_size)

    def disable_cache(self) -> None:
        """Disable embedding caching."""
        self._backend.disable_cache()

    def clear_cache(self) -> None:
        """Clear cached embeddings."""
        self._backend.clear_cache()

    @property
    def backend(self) -> TransformersBackend:
        """Access underlying backend for advanced usage."""
        return self._backend


# Protocol compliance check
def _check_protocol_compliance():
    """Verify Qwen3Encoder implements required protocols."""
    # This runs at import time to catch protocol violations early
    assert isinstance(Qwen3Encoder, type)
    # Note: Can't check instance compliance without instantiation
    # Protocol compliance is checked via runtime_checkable decorator


_check_protocol_compliance()
