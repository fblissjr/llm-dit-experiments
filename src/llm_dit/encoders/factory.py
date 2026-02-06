"""
Encoder Factory for creating model-agnostic encoders.

This factory provides a unified interface for creating any encoder type
based on configuration. It handles:
- Automatic encoder type detection from pipeline type
- Quantization configuration
- Device placement
- Caching setup

Usage:
    from llm_dit.encoders import EncoderFactory, EncoderType

    # Create by type
    encoder = EncoderFactory.create(
        encoder_type=EncoderType.QWEN3,
        model_path="Tongyi-MAI/Z-Image-Turbo",
    )

    # Auto-detect from pipeline
    encoder = EncoderFactory.for_pipeline("z_image", model_path="...")
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional, Union

import torch

from llm_dit.encoders.protocol import (
    AnyEncoder,
    EncoderType,
)

if TYPE_CHECKING:
    from llm_dit.encoders.gemma3 import Gemma3Encoder

logger = logging.getLogger(__name__)


@dataclass
class EncoderConfig:
    """Configuration for encoder creation."""

    encoder_type: EncoderType
    model_path: str
    model_subfolder: Optional[str] = None
    tokenizer_subfolder: Optional[str] = None
    quantization: Optional[str] = None
    device: str = "auto"
    dtype: str = "bfloat16"
    enable_cache: bool = False
    cache_size: int = 100

    # Gemma3-specific
    output_mode: str = "video"  # "video" or "audio"


# Pipeline to encoder type mapping
PIPELINE_ENCODER_MAP = {
    "z_image": EncoderType.QWEN3,
    "ltx2": EncoderType.GEMMA3,
}


class EncoderFactory:
    """Factory for creating encoder instances."""

    @staticmethod
    def create(
        encoder_type: Union[EncoderType, str],
        model_path: str,
        model_subfolder: Optional[str] = None,
        tokenizer_subfolder: Optional[str] = None,
        quantization: Optional[str] = None,
        device: str = "auto",
        dtype: str = "bfloat16",
        enable_cache: bool = False,
        cache_size: int = 100,
        **kwargs,
    ) -> AnyEncoder:
        """
        Create an encoder instance.

        Args:
            encoder_type: Type of encoder to create.
            model_path: Path to model or HuggingFace ID.
            model_subfolder: Subfolder containing model weights.
            tokenizer_subfolder: Subfolder containing tokenizer.
            quantization: Quantization mode (torchao unified: none, fp8-dynamic, fp8-weight-only, int8, int4).
            device: Device to load on (auto, cuda, cpu).
            dtype: Model dtype (bfloat16, float16).
            enable_cache: Enable embedding cache.
            cache_size: Cache size if enabled.
            **kwargs: Additional encoder-specific arguments.

        Returns:
            Initialized encoder instance.

        Raises:
            ValueError: If encoder type is not supported.
        """
        # Convert string to enum if needed
        if isinstance(encoder_type, str):
            encoder_type = EncoderType(encoder_type)

        logger.info(f"Creating {encoder_type.value} encoder from {model_path}")

        if encoder_type == EncoderType.QWEN3:
            from llm_dit.encoders.qwen3 import Qwen3Encoder

            return Qwen3Encoder.from_pretrained(
                model_path=model_path,
                model_subfolder=model_subfolder or "text_encoder",
                tokenizer_subfolder=tokenizer_subfolder or "tokenizer",
                quantization=quantization,
                device=device,
                dtype=dtype,
                enable_cache=enable_cache,
                cache_size=cache_size,
                **kwargs,
            )

        elif encoder_type == EncoderType.GEMMA3:
            from llm_dit.encoders.gemma3 import Gemma3Encoder

            return Gemma3Encoder.from_pretrained(
                model_path=model_path,
                output_mode=kwargs.pop("output_mode", "video"),
                quantization=quantization,
                device=device,
                dtype=dtype,
                **kwargs,
            )

        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")

    @staticmethod
    def for_pipeline(
        pipeline_type: str,
        model_path: str,
        **kwargs,
    ) -> AnyEncoder:
        """
        Create an encoder appropriate for a pipeline type.

        Args:
            pipeline_type: Pipeline type (z_image, qwen_image, ltx2, etc.)
            model_path: Path to model.
            **kwargs: Additional arguments for create().

        Returns:
            Encoder instance appropriate for the pipeline.

        Raises:
            ValueError: If pipeline type is not recognized.
        """
        if pipeline_type not in PIPELINE_ENCODER_MAP:
            raise ValueError(
                f"Unknown pipeline type: {pipeline_type}. "
                f"Supported: {list(PIPELINE_ENCODER_MAP.keys())}"
            )

        encoder_type = PIPELINE_ENCODER_MAP[pipeline_type]
        logger.info(f"Pipeline '{pipeline_type}' uses encoder type: {encoder_type.value}")

        return EncoderFactory.create(
            encoder_type=encoder_type,
            model_path=model_path,
            **kwargs,
        )

    @staticmethod
    def from_config(config: EncoderConfig) -> AnyEncoder:
        """
        Create an encoder from a configuration object.

        Args:
            config: EncoderConfig with all settings.

        Returns:
            Initialized encoder instance.
        """
        return EncoderFactory.create(
            encoder_type=config.encoder_type,
            model_path=config.model_path,
            model_subfolder=config.model_subfolder,
            tokenizer_subfolder=config.tokenizer_subfolder,
            quantization=config.quantization,
            device=config.device,
            dtype=config.dtype,
            enable_cache=config.enable_cache,
            cache_size=config.cache_size,
            output_mode=config.output_mode,
        )

    @staticmethod
    def get_encoder_type_for_pipeline(pipeline_type: str) -> EncoderType:
        """
        Get the encoder type for a pipeline without creating an encoder.

        Args:
            pipeline_type: Pipeline type name.

        Returns:
            EncoderType for the pipeline.

        Raises:
            ValueError: If pipeline type is not recognized.
        """
        if pipeline_type not in PIPELINE_ENCODER_MAP:
            raise ValueError(
                f"Unknown pipeline type: {pipeline_type}. "
                f"Supported: {list(PIPELINE_ENCODER_MAP.keys())}"
            )

        return PIPELINE_ENCODER_MAP[pipeline_type]

    @staticmethod
    def create_gemma3(
        variant: Literal["bf16", "8bit", "q4-qat"] = "bf16",
        model_path: str = "models/LTX-2",
        text_encoder_path: Optional[str] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        max_sequence_length: int = 256,
        use_connector: bool = True,
    ) -> "Gemma3Encoder":
        """
        Create a Gemma3 encoder with a specific quantization variant.

        This is a convenience method for LTX-2 text encoding that wraps
        the gemma3_variants module. It handles:
        - Automatic torchao quantization for 8bit/q4-qat variants
        - LTX-2's custom tokenizer and connector weights
        - Staged CPU-to-GPU loading to avoid OOM

        Args:
            variant: Quantization variant:
                - "bf16": Full precision (~24GB VRAM)
                - "8bit": torchao int8 quantization (~12GB VRAM)
                - "q4-qat": torchao int4 quantization (~3GB VRAM)
            model_path: Path to LTX-2 model directory.
            text_encoder_path: Override path for Gemma model weights.
                For bf16/8bit: defaults to model_path/text_encoder/
                For q4-qat: specify path to Q4 QAT model
            device: Device to load on ("cuda", "cpu", "auto").
            dtype: Model dtype (bfloat16 recommended).
            max_sequence_length: Maximum sequence length (256 for LTX-2).
            use_connector: Whether to use Embeddings1DConnector.

        Returns:
            Initialized Gemma3Encoder.

        Example:
            # Memory-efficient 8-bit for RTX 4090
            encoder = EncoderFactory.create_gemma3(
                variant="8bit",
                model_path="models/LTX-2",
            )

            # Minimum memory for tight VRAM budget
            encoder = EncoderFactory.create_gemma3(
                variant="q4-qat",
                model_path="models/LTX-2",
                text_encoder_path="~/Storage/gemma-3-12b-it-qat-q4_0-unquantized",
            )
        """
        from llm_dit.encoders.gemma3_variants import create_gemma3_encoder

        logger.info(f"Creating Gemma3 encoder via factory (variant={variant})")

        return create_gemma3_encoder(
            variant=variant,
            model_path=model_path,
            text_encoder_path=text_encoder_path,
            device=device,
            dtype=dtype,
            max_sequence_length=max_sequence_length,
            use_connector=use_connector,
        )
