"""FMTT (Flow Map Trajectory Tilting) experiments for Z-Image.

This module implements test-time reward optimization using flow maps
to guide diffusion sampling toward higher-reward regions.

Reference: arXiv:2511.22688
"""

from experiments.fmtt.differentiable_siglip import DifferentiableSigLIP

__all__ = ["DifferentiableSigLIP"]
