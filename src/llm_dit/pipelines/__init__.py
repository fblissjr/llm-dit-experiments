"""
LLM-DiT pipelines for image and video generation.

Provides:
- ZImagePipeline: End-to-end text-to-image generation (Z-Image-Turbo)
- QwenImagePipeline: High-level API for image generation (Qwen-Image T2I)
- QwenImageDiffusersPipeline: Low-level diffusers wrapper (Qwen-Image)
- QwenImage2512Pipeline: Text-to-image generation (Qwen-Image-2512)
- generate_video_with_offloading: LTX-2 text-to-video (pure PyTorch, recommended)
- WanVideoPipeline: Text/Image-to-video generation (Wan 2.1/2.2)
- setup_attention_backend: Configure attention backend (flash_attn_2, sdpa, etc.)
- MAX_TEXT_SEQ_LEN: Maximum text sequence length supported by Z-Image DiT (1504 tokens)

Note: LTX2Pipeline (diffusers wrapper) was removed in 2026-02-01.
      Use generate_video_with_offloading() instead for LTX-2 video generation.
"""

import logging

logger = logging.getLogger(__name__)

from llm_dit.pipelines.z_image import ZImagePipeline, setup_attention_backend, MAX_TEXT_SEQ_LEN
from llm_dit.pipelines.qwen_image import QwenImagePipeline
from llm_dit.pipelines.qwen_image_diffusers import QwenImageDiffusersPipeline
from llm_dit.pipelines.qwen_image_2512 import QwenImage2512Pipeline

# LTX2Pipeline (diffusers wrapper) removed 2026-02-01
# Use generate_video_with_offloading() from generate.py instead
LTX2Pipeline = None  # Deprecated - kept for API compatibility
VideoOutput = None  # Deprecated - use WanVideoOutput or generate directly

# Pure PyTorch generation utilities (no diffusers dependency)
from llm_dit.pipelines.generate import (
    GenerationConfig,
    generate_video,
    generate_video_with_offloading,
    create_position_indices,
    create_video_modality,
    cleanup_memory,
)
from llm_dit.pipelines.ltx2_config import LTX2OptimizationConfig

# WanVideoPipeline for Wan 2.1/2.2 video generation
try:
    from llm_dit.pipelines.wan_video import WanVideoPipeline
except ImportError as e:
    logger.debug(f"WanVideoPipeline not available: {e}")
    WanVideoPipeline = None

# WanVideoOutput placeholder (for future video output dataclass)
WanVideoOutput = None

__all__ = [
    "ZImagePipeline",
    "QwenImagePipeline",
    "QwenImageDiffusersPipeline",
    "QwenImage2512Pipeline",
    "LTX2Pipeline",
    "WanVideoPipeline",
    "VideoOutput",
    "WanVideoOutput",
    "setup_attention_backend",
    "MAX_TEXT_SEQ_LEN",
    # Pure PyTorch generation utilities
    "GenerationConfig",
    "generate_video",
    "generate_video_with_offloading",
    "create_position_indices",
    "create_video_modality",
    "cleanup_memory",
    "LTX2OptimizationConfig",
]
