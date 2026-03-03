"""
Schedulers for diffusion sampling.

Last Updated: 2026-03-03

Provides pure PyTorch implementations of diffusion schedulers,
independent of diffusers library.

Model-specific schedulers:
- FlowMatchScheduler: For Z-Image with shift-based sigma schedule
- LTX2Scheduler: For LTX-2 with token-count-dependent shifting
- flux2_scheduler: For FLUX.2 with SNR-based timestep shifting
"""

from llm_dit.schedulers.flow_match import FlowMatchScheduler, SchedulerOutput
from llm_dit.schedulers.ltx2_scheduler import (
    LTX2Scheduler,
    SchedulerProtocol,
    BASE_SHIFT_ANCHOR,
    MAX_SHIFT_ANCHOR,
)
from llm_dit.schedulers import flux2_scheduler

__all__ = [
    # Z-Image
    "FlowMatchScheduler",
    "SchedulerOutput",
    # LTX-2
    "LTX2Scheduler",
    "SchedulerProtocol",
    "BASE_SHIFT_ANCHOR",
    "MAX_SHIFT_ANCHOR",
    # FLUX.2
    "flux2_scheduler",
]
