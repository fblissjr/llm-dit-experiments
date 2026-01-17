"""
LLM-DiT model components.

Last Updated: 2026-01-17

Provides pure PyTorch implementations of model components that can be used
standalone or integrated with diffusers pipelines.

LTX-2 Components:
    - LTX2Transformer: Pure PyTorch LTX-2 diffusion transformer (48 layers, 19B params)
    - LTXModelType: Model variant enum (VideoOnly, AudioVideo, AudioOnly)
    - load_ltx2_transformer: Load from official checkpoints

Other Models:
    - QwenImageVAE, QwenImageDiT: Qwen image generation
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
    # LTX-2 (lazy loaded)
    "LTX2Transformer",
    "LTXModelType",
    "load_ltx2_transformer",
    "Modality",
    # Other models (lazy loaded)
    "QwenImageVAE",
    "QwenImageDiT",
    "HuMoTransformer",
    "WanVAE",
]


# Lazy imports for heavy models (avoid loading torch unless needed)
def __getattr__(name: str):
    # LTX-2 components
    if name == "LTX2Transformer":
        from llm_dit.models.ltx2_transformer import LTX2Transformer
        return LTX2Transformer
    if name == "LTXModelType":
        from llm_dit.models.ltx2_transformer import LTXModelType
        return LTXModelType
    if name == "Modality":
        from llm_dit.models.ltx2_components import Modality
        return Modality
    if name == "load_ltx2_transformer":
        from llm_dit.models.ltx2_loader import load_ltx2_transformer
        return load_ltx2_transformer

    # Other models
    if name == "QwenImageVAE":
        from llm_dit.models.qwen_image_vae import QwenImageVAE
        return QwenImageVAE
    if name == "QwenImageDiT":
        from llm_dit.models.qwen_image_dit import QwenImageDiT
        return QwenImageDiT
    if name == "HuMoTransformer":
        from llm_dit.models.humo_transformer import HuMoTransformer
        return HuMoTransformer
    if name == "WanVAE":
        from llm_dit.models.wan_vae import WanVAE
        return WanVAE

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
