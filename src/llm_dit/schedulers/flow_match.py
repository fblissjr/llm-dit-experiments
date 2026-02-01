"""
FlowMatch scheduler for Z-Image.

Pure PyTorch implementation matching the reference Z-Image behavior.
Based on DiffSynth-Studio implementation (Apache 2.0 license).

The FlowMatch scheduler uses a shifted sigma schedule. For Z-Image:
- Turbo models use linear shift (shift=3.0)
- Base models use FLUX-style exponential shift (use_dynamic_shifting=True)

Usage:
    from llm_dit.schedulers import FlowMatchScheduler

    # Z-Image-Turbo (distilled)
    scheduler = FlowMatchScheduler(shift=3.0)
    scheduler.set_timesteps(9, device="cuda")

    # Z-Image-Base (standard flow matching)
    scheduler = FlowMatchScheduler(use_dynamic_shifting=True)
    scheduler.set_timesteps(30, device="cuda", mu=3.0)

    for t in scheduler.timesteps:
        noise_pred = model(latents, t)
        latents = scheduler.step(noise_pred, t, latents)
"""

import logging
import math
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
        sigma_min: float | None = None,
        sigma_max: float = 1.0,
        shift_terminal: Optional[float] = None,
        use_dynamic_shifting: bool = False,
    ):
        """
        Initialize the scheduler.

        Args:
            num_train_timesteps: Number of training timesteps
            shift: Sigma schedule shift (3.0 for Z-Image-Turbo, used only if
                use_dynamic_shifting=False for linear shift mode)
            sigma_min: Minimum sigma value. For Z-Image-Base, should be 0.0.
                If None, defaults to 0.0 to match DiffSynth behavior.
            sigma_max: Maximum sigma value (default: 1.0)
            shift_terminal: If set, stretches the sigma schedule so the final
                sigma ends at this value instead of sigma_min. For example,
                shift_terminal=0.02 means the denoising stops at sigma=0.02.
                Used by Qwen-Image models. Default None (no stretching).
            use_dynamic_shifting: If True, uses FLUX-style exponential time shift
                for dynamic shifting (required for Z-Image-Base). If False, uses
                linear shift formula (for distilled models like Z-Image-Turbo).
        """
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        # Default sigma_min to 0.0 to match DiffSynth behavior
        self.sigma_min = sigma_min if sigma_min is not None else 0.0
        self.sigma_max = sigma_max
        self.shift_terminal = shift_terminal
        self.use_dynamic_shifting = use_dynamic_shifting

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
            "shift_terminal": self.shift_terminal,
            "use_dynamic_shifting": self.use_dynamic_shifting,
            # Compatibility with diffusers shift calculation
            "base_image_seq_len": 256,
            "max_image_seq_len": 4096,
            "base_shift": 0.5,
            "max_shift": 1.15,
        }

    def _time_shift(self, mu: float, t: torch.Tensor) -> torch.Tensor:
        """
        FLUX-style exponential time shift for dynamic shifting.

        This formula creates a steeper denoising curve at the beginning,
        which is essential for the model to establish structure early.

        Formula: exp(mu) / (exp(mu) + (1/t - 1))

        Args:
            mu: Shift parameter (dynamically calculated from image resolution)
            t: Sigma values in [0, 1] range

        Returns:
            Shifted sigma values
        """
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1))

    def _shift_sigma(self, sigma: torch.Tensor, shift: float) -> torch.Tensor:
        """
        Linear sigma shift formula (for distilled models like Z-Image-Turbo).

        Formula: shift * sigma / (1 + (shift - 1) * sigma)

        Args:
            sigma: Sigma values
            shift: Shift parameter

        Returns:
            Shifted sigma values
        """
        return shift * sigma / (1 + (shift - 1) * sigma)

    def _stretch_shift_to_terminal(self, sigmas: torch.Tensor) -> torch.Tensor:
        """
        Stretch the sigma schedule so the final sigma ends at shift_terminal.

        This adjusts the schedule so that instead of ending at sigma_min (typically 0),
        it ends at shift_terminal. The transformation preserves the relative spacing
        of the sigmas.

        Formula: sigma' = 1 - (1 - sigma) / scale_factor
        where scale_factor = (1 - sigma[-1]) / (1 - shift_terminal)

        Args:
            sigmas: Original sigma schedule tensor

        Returns:
            Stretched sigma schedule ending at shift_terminal
        """
        if self.shift_terminal is None:
            return sigmas

        one_minus_sigma = 1.0 - sigmas
        # Scale factor ensures final sigma becomes shift_terminal
        scale_factor = one_minus_sigma[-1] / (1.0 - self.shift_terminal)
        return 1.0 - (one_minus_sigma / scale_factor)

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Union[str, torch.device] = "cpu",
        mu: Optional[float] = None,
    ) -> None:
        """
        Set the discrete timesteps for inference.

        For Z-Image models, there are two shift modes:

        1. Dynamic shifting (use_dynamic_shifting=True) - for Z-Image-Base:
           - Uses FLUX-style exponential time shift
           - mu parameter controls the shift intensity (passed via set_timesteps)
           - Linspace from sigma_max to sigma_min (not including endpoint)
           - Apply exponential shift: exp(mu) / (exp(mu) + (1/t - 1))
           - Append 0 as final sigma target

        2. Linear shifting (use_dynamic_shifting=False) - for Z-Image-Turbo:
           - Uses simple linear shift formula
           - self.shift parameter controls the shift (set at init time)
           - Linspace from sigma_max to sigma_min
           - Apply linear shift: shift * sigma / (1 + (shift - 1) * sigma)
           - Append 0 as final sigma target

        Args:
            num_inference_steps: Number of denoising steps
            device: Device for tensors
            mu: Shift value for dynamic shifting mode. For Z-Image-Base, this is
                typically calculated from image resolution using calculate_shift().
        """
        # Store mu for dynamic shifting mode
        if mu is not None:
            self._mu = mu

        # Create sigma schedule: linspace from sigma_max to sigma_min
        # DiffSynth uses: torch.linspace(sigma_max, sigma_min, num_inference_steps)
        # This creates exactly num_inference_steps sigmas, NOT including endpoint as a
        # separate step. The final step goes from sigmas[-1] to 0 (appended below).
        sigmas = torch.linspace(
            self.sigma_max, self.sigma_min, num_inference_steps, device=device
        )

        # Apply shift transformation based on mode
        if self.use_dynamic_shifting:
            # FLUX-style exponential time shift (for Z-Image-Base and similar)
            # This creates a steeper denoising curve at the beginning
            # mu should be passed explicitly by the pipeline (calculated from resolution)
            # Default of 1.0 is reasonable for 1024x1024 images
            effective_mu = mu if mu is not None else 1.0
            sigmas = self._time_shift(effective_mu, sigmas)
            logger.debug(f"Scheduler: Using dynamic shifting with mu={effective_mu:.4f}")
        else:
            # Linear shift (for distilled models like Z-Image-Turbo)
            sigmas = self._shift_sigma(sigmas, self.shift)
            logger.debug(f"Scheduler: Using linear shift with shift={self.shift:.2f}")

        # Apply shift_terminal stretching if configured (for Qwen-Image models)
        if self.shift_terminal is not None:
            sigmas = self._stretch_shift_to_terminal(sigmas)

        # Compute timesteps from sigmas (before appending 0)
        timesteps = sigmas * self.num_train_timesteps

        # Append 0 as final sigma (target for last denoising step)
        # The step() function uses sigmas[i] and sigmas[i+1], so the final step
        # will go from sigmas[-1] to 0, providing actual denoising
        sigmas = torch.cat([sigmas, sigmas.new_zeros(1)])

        self.sigmas = sigmas
        self.timesteps = timesteps
        self._step_index = None

        logger.debug(f"Scheduler: {num_inference_steps} steps")
        logger.debug(f"Sigmas (first 5): {sigmas[:5].tolist()}")
        logger.debug(f"Sigmas (last 5): {sigmas[-5:].tolist()}")
        logger.debug(f"Timesteps (first 5): {self.timesteps[:5].tolist()}")

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
        use_fp32_accumulation: bool = True,
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
            use_fp32_accumulation: Whether to use float32 for accumulation to reduce
                numerical errors over many steps (default: True). Recommended for
                30+ step generations.

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
        if use_fp32_accumulation and sample.dtype != torch.float32:
            # Upcast to float32 for accumulation to reduce numerical errors
            orig_dtype = sample.dtype
            sample_fp32 = sample.float()
            model_output_fp32 = model_output.float()
            prev_sample_fp32 = sample_fp32 + model_output_fp32 * (sigma_next.float() - sigma.float())
            prev_sample = prev_sample_fp32.to(orig_dtype)
        else:
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

    def training_target(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get training target (velocity).

        For flow matching: v = noise - sample

        This is an alias for get_velocity() for API compatibility with
        DiffSynth-Studio training code.

        Args:
            sample: Clean samples (x_0)
            noise: Target noise (x_1)
            timestep: Not used, included for API compatibility

        Returns:
            Velocity targets for training
        """
        return self.get_velocity(sample, noise, timestep)

    def training_weight(
        self,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get training weight for timestep.

        Uses Gaussian distribution centered at t=500:
            w(t) = exp(-2 * ((t - 500) / 1000)^2)

        Normalized so weights sum to num_train_timesteps.

        Based on DiffSynth-Studio implementation.

        Args:
            timestep: Current timestep value (in [0, 1000] range)

        Returns:
            Training weight for the timestep
        """
        if not hasattr(self, '_timestep_weights') or self._timestep_weights is None:
            self._compute_timestep_weights(timestep.device)

        # Find closest timestep index
        step_index = self._get_step_index(timestep)
        return self._timestep_weights[step_index]

    def _compute_timestep_weights(self, device: torch.device) -> None:
        """
        Compute timestep weights based on Gaussian distribution.

        Creates a weight for each timestep that favors middle timesteps
        (around t=500) where the model learns most effectively.
        """
        if self.timesteps is None:
            # Create default timesteps if not set
            self.set_timesteps(self.num_train_timesteps, device=device)

        steps = len(self.timesteps)
        x = self.timesteps.cpu().float()

        # Gaussian centered at middle (500 for 1000 timesteps)
        center = self.num_train_timesteps / 2
        scale = self.num_train_timesteps
        y = torch.exp(-2 * ((x - center) / scale) ** 2)

        # Normalize so weights sum to num_timesteps
        y_shifted = y - y.min()
        weights = y_shifted * (steps / (y_shifted.sum() + 1e-8))

        # Empirical adjustment for non-1000 timesteps
        if steps != self.num_train_timesteps:
            weights = weights * (steps / self.num_train_timesteps)
            if len(weights) > 1:
                weights = weights + weights[1]

        self._timestep_weights = weights.to(device)


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
        shift_terminal: Optional[float] = None,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.base_image_seq_len = base_image_seq_len
        self.max_image_seq_len = max_image_seq_len
        self.base_shift = base_shift
        self.max_shift = max_shift
        self.shift_terminal = shift_terminal

    def get(self, key: str, default=None):
        """Get config value by key."""
        return getattr(self, key, default)
