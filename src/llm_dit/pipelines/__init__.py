"""
LLM-DiT pipelines for image generation.

Provides:
- ZImagePipeline: End-to-end text-to-image generation
- setup_attention_backend: Configure attention backend (flash_attn_2, sdpa, etc.)
- MAX_TEXT_SEQ_LEN: Maximum text sequence length supported by DiT (1504 tokens)
"""

from llm_dit.pipelines.z_image import ZImagePipeline, setup_attention_backend, MAX_TEXT_SEQ_LEN

__all__ = ["ZImagePipeline", "setup_attention_backend", "MAX_TEXT_SEQ_LEN"]
