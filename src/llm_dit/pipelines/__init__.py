"""
LLM-DiT pipelines for image and video generation.

Provides:
- ZImagePipeline: End-to-end text-to-image generation (Z-Image-Turbo)
- QwenImageDiffusersPipeline: Low-level diffusers wrapper (Qwen-Image-Edit)
- QwenImage2512Pipeline: Text-to-image generation (Qwen-Image-2512)
- generate_video_with_offloading: LTX-2 text-to-video (pure PyTorch, recommended)
- setup_attention_backend: Configure attention backend (flash_attn_2, sdpa, etc.)
- MAX_TEXT_SEQ_LEN: Maximum text sequence length supported by Z-Image DiT (1504 tokens)

Note: LTX2Pipeline (diffusers wrapper) was removed in 2026-02-01.
      Use generate_video_with_offloading() instead for LTX-2 video generation.
Note: QwenImagePipeline (layered decomposition) was removed in v0.8.6.
      Use QwenImageDiffusersPipeline (edit-only) instead.
"""

import logging

logger = logging.getLogger(__name__)

from llm_dit.pipelines.z_image import ZImagePipeline, setup_attention_backend, MAX_TEXT_SEQ_LEN
from llm_dit.pipelines.qwen_image_diffusers import QwenImageDiffusersPipeline
from llm_dit.pipelines.qwen_image_2512 import QwenImage2512Pipeline

# LTX2Pipeline (diffusers wrapper) removed 2026-02-01
# Use generate_video_with_offloading() from generate.py instead
LTX2Pipeline = None  # Deprecated - kept for API compatibility
VideoOutput = None  # Deprecated - use generate directly

# Pure PyTorch generation utilities (no diffusers dependency)
from llm_dit.pipelines.generate import (
    GenerationConfig,
    StepContext,
    StepSchedule,
    TwoStageConfig,
    constant_schedule,
    generate_video,
    generate_video_with_offloading,
    generate_video_two_stage,
    create_position_indices,
    create_video_modality,
    compute_audio_latent_frames,
    create_audio_position_indices,
    create_audio_modality,
    cleanup_memory,
)

__all__ = [
    "ZImagePipeline",
    "QwenImageDiffusersPipeline",
    "QwenImage2512Pipeline",
    "LTX2Pipeline",
    "VideoOutput",
    "setup_attention_backend",
    "MAX_TEXT_SEQ_LEN",
    # Pure PyTorch generation utilities
    "GenerationConfig",
    "StepContext",
    "StepSchedule",
    "TwoStageConfig",
    "constant_schedule",
    "generate_video",
    "generate_video_with_offloading",
    "generate_video_two_stage",
    "create_position_indices",
    "create_video_modality",
    "compute_audio_latent_frames",
    "create_audio_position_indices",
    "create_audio_modality",
    "cleanup_memory",
]
