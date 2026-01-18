"""
Schedulers for diffusion sampling.

Last Updated: 2026-01-18

Provides pure PyTorch implementations of diffusion schedulers,
independent of diffusers library.

Model-specific schedulers:
- FlowMatchScheduler: For Z-Image with shift-based sigma schedule
- LTX2Scheduler: For LTX-2 with token-count-dependent shifting
- LinearQuadraticScheduler: Alternative with linear + quadratic phases
- BetaScheduler: Alternative using beta distribution sampling

Sampling utilities:
- EulerDiffusionStep: Euler ODE solver for flow matching
- CFGGuider: Classifier-free guidance
"""

from llm_dit.schedulers.flow_match import FlowMatchScheduler, SchedulerOutput
from llm_dit.schedulers.ltx2_scheduler import (
    LTX2Scheduler,
    LinearQuadraticScheduler,
    BetaScheduler,
    EulerDiffusionStep,
    CFGGuider,
    SchedulerProtocol,
    BASE_SHIFT_ANCHOR,
    MAX_SHIFT_ANCHOR,
)

__all__ = [
    # Z-Image
    "FlowMatchScheduler",
    "SchedulerOutput",
    # LTX-2
    "LTX2Scheduler",
    "LinearQuadraticScheduler",
    "BetaScheduler",
    "EulerDiffusionStep",
    "CFGGuider",
    "SchedulerProtocol",
    "BASE_SHIFT_ANCHOR",
    "MAX_SHIFT_ANCHOR",
]
