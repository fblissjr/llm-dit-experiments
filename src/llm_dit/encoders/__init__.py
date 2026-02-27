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
from llm_dit.encoders.embeddings_connector import (
    Embeddings1DConnector,
    RopeType,
    load_connector_weights,
)

# Factory
from llm_dit.encoders.factory import (
    PIPELINE_ENCODER_MAP,
    EncoderConfig,
    EncoderFactory,
)
from llm_dit.encoders.gemma3 import Gemma3Encoder, LTX2Encoder, SubLayerExtractor

# Gemma3 variant loaders
from llm_dit.encoders.gemma3_variants import (
    Gemma3Variant,
    create_gemma3_encoder,
    estimate_encoder_memory,
)

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

# Low-level protocol implementations
from llm_dit.encoders.qwen3 import Qwen3Encoder

# Qwen3 FLUX.2 encoder
from llm_dit.encoders.qwen3_flux2 import (
    Qwen3Flux2Encoder,
    load_qwen3_flux2_encoder,
)

# Unified Qwen3 encoder (new - supports both Z-Image and FLUX.2 modes)
from llm_dit.encoders.qwen3_unified import (
    KLEIN_4B_CONFIG,
    KLEIN_9B_CONFIG,
    ZIMAGE_CONFIG,
    Qwen3EncoderConfig,
    Qwen3UnifiedEncoder,
    get_unified_encoder,
)
from llm_dit.encoders.z_image import ZImageTextEncoder

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
    # Gemma3 variant loaders
    "create_gemma3_encoder",
    "Gemma3Variant",
    "estimate_encoder_memory",
    # Qwen3 FLUX.2 encoder
    "Qwen3Flux2Encoder",
    "load_qwen3_flux2_encoder",
    # Unified Qwen3 encoder
    "Qwen3UnifiedEncoder",
    "Qwen3EncoderConfig",
    "get_unified_encoder",
    "ZIMAGE_CONFIG",
    "KLEIN_4B_CONFIG",
    "KLEIN_9B_CONFIG",
]
