"""
Pipeline adapters - Wrap existing pipelines as orchestration steps.

Last Updated: 2026-01-12

Adapters enable existing pipelines to work in orchestration
without modification. They:
- Declare inputs/outputs matching the pipeline's interface
- Declare required models from the pool
- Execute the pipeline with pooled models

Note: Pipelines continue to work standalone - adapters are optional.
"""

from .wan_video import (
    WanVideoAdapter,
    WanTextEncoderStep,
    WanVAEDecodeStep,
)

__all__ = [
    "WanVideoAdapter",
    "WanTextEncoderStep",
    "WanVAEDecodeStep",
]
