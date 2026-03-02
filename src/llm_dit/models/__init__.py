"""
LLM-DiT model components.

Last Updated: 2026-01-18

Provides pure PyTorch implementations of model components that can be used
standalone or integrated with diffusers pipelines.

LTX-2 Components (in llm_dit.models.ltx2):
    - LTX2Transformer: Pure PyTorch LTX-2 diffusion transformer (48 layers, 19B params)
    - LTXModelType: Model variant enum (VideoOnly, AudioVideo, AudioOnly)
    - load_ltx2_transformer: Load from official checkpoints
    - load_ltx2_transformer_quantized: Load with FP8 quantization for 24GB GPUs
    - Modality: Input container for video/audio latents

Other Models:
    - HuMoTransformer: Human motion generation
    - WanVAE: WAN video VAE
"""

from llm_dit.models.context_refiner import (
    ContextRefiner,
    ContextRefinerBlock,
    RMSNorm,
    RotaryEmbedding,
    GatedFeedForward,
)

__all__ = [
    # Context refiner
    "ContextRefiner",
    "ContextRefinerBlock",
    "RMSNorm",
    "RotaryEmbedding",
    "GatedFeedForward",
    # LTX-2 (lazy loaded from ltx2 subpackage)
    "LTX2Transformer",
    "LTXModelType",
    "load_ltx2_transformer",
    "load_ltx2_transformer_quantized",
    "Modality",
    # LTX-2 Connectors (lazy loaded)
    "LTX2TextConnectors",
    "LTX2ConnectorTransformer1d",
    "load_ltx2_connectors",
    # Other models (lazy loaded)
    "HuMoTransformer",
    "WanVAE",
]


# Lazy imports for heavy models (avoid loading torch unless needed)
def __getattr__(name: str):
    # LTX-2 components (from ltx2 subpackage)
    if name == "LTX2Transformer":
        from llm_dit.models.ltx2.transformer import LTX2Transformer
        return LTX2Transformer
    if name == "LTXModelType":
        from llm_dit.models.ltx2.transformer import LTXModelType
        return LTXModelType
    if name == "Modality":
        from llm_dit.models.ltx2.components import Modality
        return Modality
    if name == "load_ltx2_transformer":
        from llm_dit.models.ltx2.loader import load_ltx2_transformer
        return load_ltx2_transformer
    if name == "load_ltx2_transformer_quantized":
        from llm_dit.models.ltx2.loader import load_ltx2_transformer_quantized
        return load_ltx2_transformer_quantized

    # LTX-2 Connectors
    if name == "LTX2TextConnectors":
        from llm_dit.models.ltx2.connectors import LTX2TextConnectors
        return LTX2TextConnectors
    if name == "LTX2ConnectorTransformer1d":
        from llm_dit.models.ltx2.connectors import LTX2ConnectorTransformer1d
        return LTX2ConnectorTransformer1d
    if name == "load_ltx2_connectors":
        from llm_dit.models.ltx2.connectors import load_ltx2_connectors
        return load_ltx2_connectors

    # Other models
    if name == "HuMoTransformer":
        from llm_dit.models.humo_transformer import HuMoTransformer
        return HuMoTransformer
    if name == "WanVAE":
        from llm_dit.models.wan_vae import WanVAE
        return WanVAE

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
