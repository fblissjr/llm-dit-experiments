"""
LLM-DiT pipelines for image generation.

Provides:
- ZImagePipeline: End-to-end text-to-image generation (Z-Image-Turbo)
- QwenImagePipeline: High-level API for image decomposition (Qwen-Image-Layered)
- QwenImageDiffusersPipeline: Low-level diffusers wrapper (Qwen-Image-Layered)
- setup_attention_backend: Configure attention backend (flash_attn_2, sdpa, etc.)
- MAX_TEXT_SEQ_LEN: Maximum text sequence length supported by Z-Image DiT (1504 tokens)
"""

from llm_dit.pipelines.z_image import ZImagePipeline, setup_attention_backend, MAX_TEXT_SEQ_LEN
from llm_dit.pipelines.qwen_image import QwenImagePipeline
from llm_dit.pipelines.qwen_image_diffusers import QwenImageDiffusersPipeline

__all__ = [
    "ZImagePipeline",
    "QwenImagePipeline",
    "QwenImageDiffusersPipeline",
    "setup_attention_backend",
    "MAX_TEXT_SEQ_LEN",
]
