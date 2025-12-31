"""
LLM-DiT pipelines for image generation.

Provides:
- ZImagePipeline: End-to-end text-to-image generation (Z-Image-Turbo)
- QwenImagePipeline: High-level API for image decomposition (Qwen-Image-Layered)
- QwenImageDiffusersPipeline: Low-level diffusers wrapper (Qwen-Image-Layered)
- QwenImage2512Pipeline: Text-to-image generation (Qwen-Image-2512)
- setup_attention_backend: Configure attention backend (flash_attn_2, sdpa, etc.)
- MAX_TEXT_SEQ_LEN: Maximum text sequence length supported by Z-Image DiT (1504 tokens)
"""

from llm_dit.pipelines.z_image import ZImagePipeline, setup_attention_backend, MAX_TEXT_SEQ_LEN
from llm_dit.pipelines.qwen_image import QwenImagePipeline
from llm_dit.pipelines.qwen_image_diffusers import QwenImageDiffusersPipeline
from llm_dit.pipelines.qwen_image_2512 import QwenImage2512Pipeline

__all__ = [
    "ZImagePipeline",
    "QwenImagePipeline",
    "QwenImageDiffusersPipeline",
    "QwenImage2512Pipeline",
    "setup_attention_backend",
    "MAX_TEXT_SEQ_LEN",
]
