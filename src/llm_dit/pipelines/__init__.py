"""
LLM-DiT pipelines for image generation.

Provides:
- ZImagePipeline: End-to-end text-to-image generation
- setup_attention_backend: Configure attention backend (flash_attn_2, sdpa, etc.)
"""

from llm_dit.pipelines.z_image import ZImagePipeline, setup_attention_backend

__all__ = ["ZImagePipeline", "setup_attention_backend"]
