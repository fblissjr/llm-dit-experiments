"""
Model-agnostic encoder module for LLM-DiT experiments.

This module provides both high-level encoders with template support and
low-level protocol-based encoders for different model families.

High-Level Encoders (with template/conversation support):
- ZImageTextEncoder: Full-featured encoder for Z-Image with templates

Low-Level Protocol Encoders (model-agnostic):
- Qwen3Encoder: Qwen3-4B for Z-Image (text-only, 2560 dim)
- Gemma3Encoder: Gemma3-12B for LTX-2 (vision-language, 3840 dim output, DiT projects to 4096/2048)

Factory:
- EncoderFactory: Create encoders by type or pipeline

Quick Start (Low-Level):
    from llm_dit.encoders import EncoderFactory, EncoderType

    encoder = EncoderFactory.create(
        encoder_type=EncoderType.QWEN3,
        model_path="Tongyi-MAI/Z-Image-Turbo",
    )
    output = encoder.encode(["A beautiful sunset"])
    encoder.offload()  # Free VRAM

Quick Start (High-Level with templates):
    from llm_dit.encoders import ZImageTextEncoder

    encoder = ZImageTextEncoder.from_pretrained(
        "/path/to/z-image",
        templates_dir="templates/z_image",
    )
    output = encoder.encode("A cat", template="photorealistic")
"""

# High-level encoder with template support (existing)
from llm_dit.encoders.z_image import ZImageTextEncoder

# Protocol definitions
from llm_dit.encoders.protocol import (
    AnyEncoder,
    EncoderCapability,
    EncoderInfo,
    EncoderType,
    EncodingOutput,
    GenerativeEncoderProtocol,
    TextEncoderProtocol,
    VisionLanguageEncoderProtocol,
)

# Factory
from llm_dit.encoders.factory import (
    EncoderConfig,
    EncoderFactory,
    PIPELINE_ENCODER_MAP,
)

# Low-level protocol implementations
from llm_dit.encoders.qwen3 import Qwen3Encoder
from llm_dit.encoders.gemma3 import Gemma3Encoder, LTX2Encoder, SubLayerExtractor
from llm_dit.encoders.embeddings_connector import (
    Embeddings1DConnector,
    RopeType,
    load_connector_weights,
)

__all__ = [
    # High-level encoder (existing)
    "ZImageTextEncoder",
    # Protocols
    "TextEncoderProtocol",
    "VisionLanguageEncoderProtocol",
    "GenerativeEncoderProtocol",
    "AnyEncoder",
    # Data classes
    "EncodingOutput",
    "EncoderInfo",
    "EncoderConfig",
    # Enums
    "EncoderType",
    "EncoderCapability",
    # Factory
    "EncoderFactory",
    "PIPELINE_ENCODER_MAP",
    # Low-level implementations
    "Qwen3Encoder",
    "Gemma3Encoder",
    "LTX2Encoder",
    "SubLayerExtractor",
    # Embeddings connector
    "Embeddings1DConnector",
    "RopeType",
    "load_connector_weights",
]
