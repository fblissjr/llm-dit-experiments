"""Reward functions for test-time optimization.

This module provides differentiable reward functions for guiding
diffusion sampling toward higher-quality outputs.

Available reward functions:
- DifferentiableSigLIP: SigLIP2-based image-text alignment reward
"""

from llm_dit.rewards.siglip import DifferentiableSigLIP

__all__ = ["DifferentiableSigLIP"]
