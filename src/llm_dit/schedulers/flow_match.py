"""
FlowMatch scheduler for Z-Image.

Pure PyTorch implementation matching the reference Z-Image behavior.
Based on DiffSynth-Studio implementation (Apache 2.0 license).

The FlowMatch scheduler uses a shifted sigma schedule that "bakes in"
CFG-like behavior from the Decoupled-DMD training process.

Usage:
    from llm_dit.schedulers import FlowMatchScheduler

    scheduler = FlowMatchScheduler(shift=3.0)
    scheduler.set_timesteps(9, device="cuda")

    for t in scheduler.timesteps:
        noise_pred = model(latents, t)
        latents = scheduler.step(noise_pred, t, latents)
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

logger = logging.getLogger(__name__)


@dataclass
class SchedulerOutput:
    """Output of scheduler step."""

    prev_sample: torch.Tensor


class FlowMatchScheduler:
    """
    Flow matching scheduler with Z-Image specific shift.

    This is a minimal, dependency-free implementation that exactly
    matches the reference Z-Image behavior.

    The shift parameter controls the sigma schedule transformation:
        sigma_shifted = shift * sigma / (1 + (shift - 1) * sigma)

    For Z-Image-Turbo, shift=3.0 is the default, which compresses
    the noise schedule to enable fewer inference steps (8-9).

    Attributes:
        num_train_timesteps: Number of timesteps used during training (1000)
        shift: Sigma schedule shift parameter (3.0 for Z-Image-Turbo)
        sigmas: Computed sigma values for inference
        timesteps: Computed timestep values for inference
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 3.0,
        sigma_min: float = 0.0,
        sigma_max: float = 1.0,
    ):
        """
        Initialize the scheduler.

        Args:
            num_train_timesteps: Number of training timesteps
            shift: Sigma schedule shift (3.0 for Z-Image-Turbo)
            sigma_min: Minimum sigma value
            sigma_max: Maximum sigma value
        """
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        # Will be set by set_timesteps()
        self.sigmas: Optional[torch.Tensor] = None
        self.timesteps: Optional[torch.Tensor] = None
        self._step_index: Optional[int] = None

    @property
    def config(self) -> dict:
        """Return scheduler config (for diffusers compatibility)."""
        return {
            "num_train_timesteps": self.num_train_timesteps,
            "shift": self.shift,
            "sigma_min": self.sigma_min,
            "sigma_max": self.sigma_max,
            # Compatibility with diffusers shift calculation
            "base_image_seq_len": 256,
            "max_image_seq_len": 4096,
            "base_shift": 0.5,
            "max_shift": 1.15,
        }

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Union[str, torch.device] = "cpu",
        mu: Optional[float] = None,
    ) -> None:
        """
        Set the discrete timesteps for inference.

        Args:
            num_inference_steps: Number of denoising steps
            device: Device for tensors
            mu: Override shift value (for diffusers compatibility, same as shift)
        """
        if mu is not None:
            self.shift = mu

        # Linear spacing in sigma space: 1.0 -> 0.0
        sigmas = torch.linspace(
            self.sigma_max,
            self.sigma_min,
            num_inference_steps + 1,
            device=device,
        )

        # Apply Z-Image shift transformation
        # This "bakes in" the CFG-like behavior from Decoupled-DMD training
        # Formula: sigma' = shift * sigma / (1 + (shift - 1) * sigma)
        sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)

        self.sigmas = sigmas

        # Timesteps are sigma * num_train_timesteps, excluding final sigma (0)
        # Z-Image uses timesteps in [0, 1000] range
        self.timesteps = sigmas[:-1] * self.num_train_timesteps

        self._step_index = None

        logger.debug(
            f"Scheduler: {num_inference_steps} steps, shift={self.shift:.2f}"
        )
        logger.debug(f"Sigmas: {sigmas.tolist()}")
        logger.debug(f"Timesteps: {self.timesteps.tolist()}")

    def _get_step_index(self, timestep: torch.Tensor) -> int:
        """Find the step index for a given timestep."""
        if self.timesteps is None:
            raise RuntimeError("Must call set_timesteps() before step()")

        # Try exact match first
        matches = (self.timesteps == timestep)
        if matches.any():
            return matches.nonzero()[0].item()

        # Fall back to closest match
        return (self.timesteps - timestep).abs().argmin().item()

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple[torch.Tensor]]:
        """
        Predict the sample at the previous timestep.

        Uses Euler method for flow matching:
            x_{t-1} = x_t + v * (sigma_{t-1} - sigma_t)

        where v is the velocity prediction from the model.

        Args:
            model_output: Velocity prediction from the model
            timestep: Current timestep value
            sample: Current noisy sample (x_t)
            return_dict: Whether to return SchedulerOutput or tuple

        Returns:
            SchedulerOutput with prev_sample, or tuple (prev_sample,)
        """
        if self.sigmas is None:
            raise RuntimeError("Must call set_timesteps() before step()")

        # Get step index
        step_index = self._get_step_index(timestep)

        # Get current and next sigma
        sigma = self.sigmas[step_index]
        sigma_next = self.sigmas[step_index + 1]

        # Euler step: x_{t-1} = x_t + v * (sigma_{t-1} - sigma_t)
        # Note: sigma_next < sigma (we're denoising), so this adds noise_pred * negative_value
        prev_sample = sample + model_output * (sigma_next - sigma)

        if return_dict:
            return SchedulerOutput(prev_sample=prev_sample)
        return (prev_sample,)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add noise to samples for img2img or inpainting.

        Uses linear interpolation:
            noisy = (1 - sigma) * original + sigma * noise

        Args:
            original_samples: Clean samples
            noise: Random noise
            timesteps: Timesteps specifying noise level

        Returns:
            Noisy samples
        """
        if self.sigmas is None:
            raise RuntimeError("Must call set_timesteps() before add_noise()")

        # Get sigma for the given timestep
        step_index = self._get_step_index(timesteps)
        sigma = self.sigmas[step_index]

        # Ensure sigma has right shape for broadcasting
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)

        # Linear interpolation between clean and noise
        noisy = (1 - sigma) * original_samples + sigma * noise
        return noisy

    def scale_model_input(
        self,
        sample: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Scale model input (no-op for flow matching).

        Flow matching doesn't require input scaling, but this method
        is provided for API compatibility with diffusers.
        """
        return sample

    def get_velocity(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute velocity for training.

        For flow matching, velocity is simply: v = noise - sample

        Args:
            sample: Clean samples (x_0)
            noise: Target noise (x_1)
            timesteps: Not used, included for API compatibility

        Returns:
            Velocity targets for training
        """
        return noise - sample


class FlowMatchSchedulerConfig:
    """
    Configuration for FlowMatchScheduler.

    Provides a diffusers-compatible config interface.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 3.0,
        base_image_seq_len: int = 256,
        max_image_seq_len: int = 4096,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.base_image_seq_len = base_image_seq_len
        self.max_image_seq_len = max_image_seq_len
        self.base_shift = base_shift
        self.max_shift = max_shift

    def get(self, key: str, default=None):
        """Get config value by key."""
        return getattr(self, key, default)
