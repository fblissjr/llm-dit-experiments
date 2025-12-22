"""
Flow Map Trajectory Tilting (FMTT) guidance for diffusion transformers.

Last Updated: 2025-12-22

This module implements test-time reward optimization using flow maps
to guide diffusion sampling toward higher-reward regions.

The key insight: At each denoising step, we can predict where the trajectory
will end up (via flow map), evaluate a reward on that prediction, and
backprop to nudge the current step toward higher-reward regions.

For Z-Image flow matching, the flow map is a single-step Euler approximation:
    x_clean = x_t + velocity * sigma

Reference: arXiv:2511.22688 (Test-Time Scaling of Diffusion Models with Flow Maps)
"""

import logging
from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def flow_map_direct(
    x_t: torch.Tensor,
    velocity: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Predict clean latents from noisy using single-step flow map.

    For Z-Image flow matching:
        x_clean = x_t + velocity * sigma

    This is the single-step Euler approximation. It's sufficient because:
    - Turbo model has only 8-9 steps (each covers significant trajectory)
    - Even imperfect gradients nudge in the right direction
    - Direct approximation is essentially free (reuses DiT velocity)

    Args:
        x_t: Current noisy latents, shape (B, C, H, W)
        velocity: Raw velocity from DiT (before pipeline negation), shape (B, C, H, W)
        sigma: Current sigma value (noise level), in [0, 1]

    Returns:
        Predicted clean latents, shape (B, C, H, W)
    """
    return x_t + velocity * sigma


class FMTTGuidance:
    """Flow Map Trajectory Tilting guidance for test-time reward optimization.

    Usage:
        fmtt = FMTTGuidance(
            vae=pipe.vae,
            reward_fn=DifferentiableSigLIP(device="cuda"),
            guidance_scale=1.0,
        )

        # In denoising loop:
        if fmtt.is_active(step, num_steps):
            grad, reward = fmtt.compute_gradient(latents, velocity, sigma, prompt)
            velocity_guided = velocity + fmtt.guidance_scale * grad
        else:
            velocity_guided = velocity

    Args:
        vae: VAE decoder for latent -> image
        reward_fn: Differentiable reward function with compute_reward(image, prompt)
        guidance_scale: Scale for reward gradients (0.5-2.0 typical)
        guidance_start: Start guidance at this fraction of steps (default: 0.0)
        guidance_stop: Stop guidance at this fraction of steps (default: 0.5)
        normalize_mode: Gradient normalization mode:
            - "unit": Normalize to unit norm (default, most stable)
            - "clip": Clip to max norm
            - "none": No normalization
        clip_value: Max gradient norm for "clip" mode
        decode_scale: Scale factor for intermediate VAE decode (0.5 = 512px for 1024px input)
    """

    def __init__(
        self,
        vae: Any,
        reward_fn: Any,
        guidance_scale: float = 1.0,
        guidance_start: float = 0.0,
        guidance_stop: float = 0.5,
        normalize_mode: str = "unit",
        clip_value: float = 1.0,
        decode_scale: float = 0.5,
    ):
        self.vae = vae
        self.reward_fn = reward_fn
        self.guidance_scale = guidance_scale
        self.guidance_start = guidance_start
        self.guidance_stop = guidance_stop
        self.normalize_mode = normalize_mode
        self.clip_value = clip_value
        self.decode_scale = decode_scale

        # Validate normalize mode
        if normalize_mode not in ("unit", "clip", "none"):
            raise ValueError(
                f"normalize_mode must be 'unit', 'clip', or 'none', got {normalize_mode}"
            )

    def is_active(self, step: int, num_steps: int) -> bool:
        """Check if guidance should be active at this step.

        Args:
            step: Current step index (0-indexed)
            num_steps: Total number of steps

        Returns:
            True if guidance should be applied
        """
        if num_steps <= 0:
            return True
        progress = step / num_steps
        return self.guidance_start <= progress < self.guidance_stop

    def compute_gradient(
        self,
        latents: torch.Tensor,
        velocity: torch.Tensor,
        sigma: float,
        prompt: str,
    ) -> Tuple[torch.Tensor, float]:
        """Compute gradient of reward w.r.t. latents through flow map.

        This is the core FMTT computation:
        1. Flow map: latents -> predicted clean latents
        2. Downscale latents (saves ~4x VRAM during VAE decode)
        3. VAE decode: latents -> image
        4. Reward: image -> scalar
        5. Backprop: scalar -> gradient w.r.t. latents

        Args:
            latents: Current noisy latents, shape (B, C, H, W)
            velocity: DiT velocity prediction (raw, before negation)
            sigma: Current noise level
            prompt: Text prompt for reward

        Returns:
            Tuple of (gradient tensor, reward value)
        """
        # Enable gradients on latents
        latents_grad = latents.detach().requires_grad_(True)

        # Flow map prediction (gradients flow through)
        predicted_clean = flow_map_direct(latents_grad, velocity.detach(), sigma)

        # Downscale latents before VAE decode to save VRAM
        # The gradient will still flow back to full-resolution latents via interpolate
        if self.decode_scale < 1.0:
            h, w = predicted_clean.shape[-2:]
            new_h, new_w = int(h * self.decode_scale), int(w * self.decode_scale)
            predicted_clean_small = F.interpolate(
                predicted_clean, size=(new_h, new_w), mode="bilinear", align_corners=False
            )
        else:
            predicted_clean_small = predicted_clean

        # VAE decode (with gradients for backprop to latents)
        scaled = (predicted_clean_small / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(scaled.to(self.vae.dtype)).sample

        # Reward computation
        reward = self.reward_fn.compute_reward(image, prompt)

        # Backprop to latents
        grad = torch.autograd.grad(
            reward.mean(),
            latents_grad,
            create_graph=False,
        )[0]

        # Check for numerical issues
        if grad.isnan().any() or grad.isinf().any():
            logger.warning("FMTT gradient instability detected, returning zero gradient")
            return torch.zeros_like(grad), reward.mean().item()

        # Normalize gradient
        grad = self._normalize_gradient(grad)

        return grad, reward.mean().item()

    def _normalize_gradient(self, grad: torch.Tensor) -> torch.Tensor:
        """Normalize gradient according to normalize_mode."""
        grad_norm = grad.norm()

        if self.normalize_mode == "unit":
            return grad / (grad_norm + 1e-8)
        elif self.normalize_mode == "clip" and grad_norm > self.clip_value:
            return grad * (self.clip_value / grad_norm)
        else:
            # "none" - return as-is
            return grad

    def guide_velocity(
        self,
        velocity: torch.Tensor,
        grad: torch.Tensor,
    ) -> torch.Tensor:
        """Apply FMTT guidance to velocity prediction.

        Args:
            velocity: Raw velocity from DiT
            grad: Gradient from compute_gradient

        Returns:
            Guided velocity
        """
        return velocity + self.guidance_scale * grad


def create_fmtt_guidance(
    vae: Any,
    reward_fn: Optional[Any] = None,
    guidance_scale: float = 1.0,
    guidance_start: float = 0.0,
    guidance_stop: float = 0.5,
    normalize_mode: str = "unit",
    decode_scale: float = 0.5,
    device: str = "cuda",
) -> FMTTGuidance:
    """Factory function to create FMTT guidance with optional reward function loading.

    Args:
        vae: VAE decoder
        reward_fn: Pre-loaded reward function, or None to load SigLIP
        guidance_scale: FMTT guidance scale
        guidance_start: When to start guidance
        guidance_stop: When to stop guidance
        normalize_mode: Gradient normalization mode
        decode_scale: Scale for intermediate VAE decode
        device: Device for reward function (if loading new)

    Returns:
        Configured FMTTGuidance instance
    """
    if reward_fn is None:
        from llm_dit.rewards.siglip import DifferentiableSigLIP
        reward_fn = DifferentiableSigLIP(device=device)

    return FMTTGuidance(
        vae=vae,
        reward_fn=reward_fn,
        guidance_scale=guidance_scale,
        guidance_start=guidance_start,
        guidance_stop=guidance_stop,
        normalize_mode=normalize_mode,
        decode_scale=decode_scale,
    )
