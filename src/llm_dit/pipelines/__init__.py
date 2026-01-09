"""
LLM-DiT pipelines for image and video generation.

Provides:
- ZImagePipeline: End-to-end text-to-image generation (Z-Image-Turbo)
- QwenImagePipeline: High-level API for image decomposition (Qwen-Image-Layered)
- QwenImageDiffusersPipeline: Low-level diffusers wrapper (Qwen-Image-Layered)
- QwenImage2512Pipeline: Text-to-image generation (Qwen-Image-2512)
- LTX2Pipeline: Text-to-video generation (LTX-2, requires diffusers>=0.32.0)
- setup_attention_backend: Configure attention backend (flash_attn_2, sdpa, etc.)
- MAX_TEXT_SEQ_LEN: Maximum text sequence length supported by Z-Image DiT (1504 tokens)
"""

from llm_dit.pipelines.z_image import ZImagePipeline, setup_attention_backend, MAX_TEXT_SEQ_LEN
from llm_dit.pipelines.qwen_image import QwenImagePipeline
from llm_dit.pipelines.qwen_image_diffusers import QwenImageDiffusersPipeline
from llm_dit.pipelines.qwen_image_2512 import QwenImage2512Pipeline

# LTX2Pipeline requires diffusers with LTX2 support (>=0.32.0)
# Import lazily to avoid hard dependency
try:
    from llm_dit.pipelines.ltx2 import LTX2Pipeline, VideoOutput
except ImportError:
    LTX2Pipeline = None
    VideoOutput = None

__all__ = [
    "ZImagePipeline",
    "QwenImagePipeline",
    "QwenImageDiffusersPipeline",
    "QwenImage2512Pipeline",
    "LTX2Pipeline",
    "VideoOutput",
    "setup_attention_backend",
    "MAX_TEXT_SEQ_LEN",
]
