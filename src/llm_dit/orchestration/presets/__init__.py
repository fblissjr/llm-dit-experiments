"""
Orchestration presets - Pre-configured pipelines for common workflows.

Last Updated: 2026-01-12

Presets are factory functions that create pre-configured orchestrators
for common use cases:
- music_video: Audio → Transcribe → Prompts → Frames → Video → Stitch
- image_to_video: Image → Wan I2V → Video
- style_transfer: Image → Describe → Z-Image regenerate
"""

# Presets will be added here as they're implemented
# from .music_video import create_music_video_pipeline

__all__ = []
