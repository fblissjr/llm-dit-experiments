"""
LLM-DiT model components.

Provides pure PyTorch implementations of model components that can be used
standalone or integrated with diffusers pipelines.
"""

from llm_dit.models.context_refiner import (
    ContextRefiner,
    ContextRefinerBlock,
    RMSNorm,
    RotaryEmbedding,
    GatedFeedForward,
)

__all__ = [
    "ContextRefiner",
    "ContextRefinerBlock",
    "RMSNorm",
    "RotaryEmbedding",
    "GatedFeedForward",
]


# Lazy imports for heavy models (avoid loading torch unless needed)
def __getattr__(name: str):
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
