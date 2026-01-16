"""
Latent Normalization Utilities for LTX-2 Pipeline.

Last Updated: 2026-01-16

Ported from ComfyUI-LTXVideo/latent_norm.py to address CFG-induced drift
("overbaking") during diffusion sampling. When CFG is applied, the combined
prediction can drift outside the expected latent distribution, causing:
- Oversaturation
- Color clipping
- Artifacts in high-detail regions

Two normalization approaches are implemented:

1. Statistical Normalization (LTXVStatNormLatent):
   - Computes robust statistics using percentile filtering
   - Rescales to target mean/std (typically 0, 1)
   - Applied per-step with tapering factor

2. AdaIN Normalization (LTXVAdainLatent):
   - Uses reference latent statistics (e.g., from initial noise)
   - Applies Adaptive Instance Normalization
   - Good for style consistency

Reference:
    ComfyUI-LTXVideo by Lightricks
    https://github.com/Lightricks/ComfyUI-LTXVideo

Example:
    from llm_dit.pipelines.utils import PerStepNormalizer

    normalizer = PerStepNormalizer(factors="0.9, 0.75, 0.5, 0.25, 0.0")

    # In diffusion loop after CFG:
    noise_pred = uncond + cfg_scale * (cond - uncond)
    noise_pred = normalizer(noise_pred, step=i)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union

import torch
import torch.nn.functional as F


@dataclass
class NormalizationConfig:
    """Configuration for latent normalization."""

    # Target statistics
    target_mean: float = 0.0
    target_std: float = 1.0

    # Percentile filtering (for statistical method)
    percentile: float = 95.0  # Use middle 95% of values

    # Factor schedule (per-step tapering)
    # Higher early = more aggressive correction, taper to 0 = no correction
    factors: str = "0.9, 0.75, 0.5, 0.25, 0.0"

    # Optional outlier clipping
    clip_outliers: bool = False
    clip_std: float = 3.0  # Clip values beyond N std devs

    # Per-frame vs per-batch normalization
    per_frame: bool = False

    def get_factors_list(self) -> List[float]:
        """Parse factors string into list."""
        return [float(f.strip()) for f in self.factors.split(",")]


def compute_robust_stats(
    x: torch.Tensor,
    percentile: float = 95.0,
    dim: Optional[tuple] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute mean and std using percentile filtering (robust to outliers).

    Uses the middle N% of values to compute statistics, which prevents
    extreme values from skewing the normalization.

    Args:
        x: Input tensor
        percentile: Percentage of values to use (e.g., 95 = middle 95%)
        dim: Dimensions to compute stats over. If None, uses all dims.

    Returns:
        (mean, std) tensors
    """
    if dim is None:
        flat = x.flatten()
    else:
        # Move specified dims to end and flatten
        other_dims = [i for i in range(x.ndim) if i not in dim]
        permuted = x.permute(*other_dims, *dim)
        flat = permuted.reshape(*permuted.shape[: len(other_dims)], -1)

    # Compute percentile bounds
    lower_pct = (100 - percentile) / 2 / 100
    upper_pct = 1 - lower_pct

    lower = torch.quantile(flat, lower_pct, dim=-1, keepdim=True)
    upper = torch.quantile(flat, upper_pct, dim=-1, keepdim=True)

    # Create mask for values within percentile range
    mask = (flat >= lower) & (flat <= upper)

    # Compute stats on filtered values
    # Use masked_fill to handle division properly
    masked = flat.masked_fill(~mask, 0)
    count = mask.sum(dim=-1, keepdim=True).clamp(min=1)

    mean = masked.sum(dim=-1, keepdim=True) / count
    var = ((masked - mean) ** 2).masked_fill(~mask, 0).sum(dim=-1, keepdim=True) / count
    std = var.sqrt().clamp(min=1e-6)

    return mean.squeeze(-1), std.squeeze(-1)


def statistical_normalize(
    latents: torch.Tensor,
    target_mean: float = 0.0,
    target_std: float = 1.0,
    percentile: float = 95.0,
    factor: float = 1.0,
    clip_outliers: bool = False,
    clip_std: float = 3.0,
    per_frame: bool = False,
) -> torch.Tensor:
    """
    Statistical normalization using percentile-filtered statistics.

    Port of LTXVStatNormLatent from ComfyUI-LTXVideo. Computes statistics
    on the middle N% of values (robust to outliers), then rescales to
    target mean/std with lerp factor.

    Args:
        latents: Input tensor, typically [B, C, F, H, W] for video
        target_mean: Target mean after normalization
        target_std: Target standard deviation after normalization
        percentile: Percentage of values to use for statistics (0-100)
        factor: Interpolation factor (0=no change, 1=full normalization)
        clip_outliers: Whether to clip extreme values
        clip_std: If clipping, clip beyond N standard deviations
        per_frame: Compute stats per-frame vs per-batch

    Returns:
        Normalized latents with same shape as input
    """
    if factor == 0.0:
        return latents

    original = latents

    # Determine dims for stats computation
    # For video latents [B, C, F, H, W]: compute over C, H, W per frame if per_frame
    # Otherwise compute over all dims except batch
    if per_frame and latents.ndim >= 4:
        # Assume [B, C, F, H, W] - compute per batch per frame
        B, C, *spatial = latents.shape
        # Reshape to [B, F, C*H*W] for per-frame stats
        latents_flat = latents.reshape(B, C, -1)
        if len(spatial) >= 2 and spatial[0] > 1:  # Has frames
            F_dim = spatial[0]
            latents_flat = latents.reshape(B, C, F_dim, -1).permute(0, 2, 1, 3)
            latents_flat = latents_flat.reshape(B * F_dim, -1)
    else:
        # Global stats per batch
        latents_flat = latents.reshape(latents.shape[0], -1)

    # Compute robust statistics
    mean, std = compute_robust_stats(latents_flat, percentile=percentile, dim=(-1,))

    # Reshape stats for broadcasting
    if per_frame and latents.ndim >= 4:
        B, C, *spatial = latents.shape
        if len(spatial) >= 2 and spatial[0] > 1:
            F_dim = spatial[0]
            mean = mean.reshape(B, F_dim, 1, 1, 1)
            std = std.reshape(B, F_dim, 1, 1, 1)
            # Permute back: [B, F, 1, 1, 1] -> [B, 1, F, 1, 1]
            mean = mean.transpose(1, 2)
            std = std.transpose(1, 2)
        else:
            mean = mean.reshape(B, 1, 1, 1, 1)[..., 0]
            std = std.reshape(B, 1, 1, 1, 1)[..., 0]
    else:
        # Reshape for broadcast: [B] -> [B, 1, 1, ...]
        shape = [latents.shape[0]] + [1] * (latents.ndim - 1)
        mean = mean.reshape(*shape)
        std = std.reshape(*shape)

    # Normalize to target distribution
    normalized = (latents - mean) / std * target_std + target_mean

    # Optional outlier clipping
    if clip_outliers:
        clip_range = clip_std * target_std
        normalized = normalized.clamp(
            target_mean - clip_range, target_mean + clip_range
        )

    # Lerp between original and normalized
    output = torch.lerp(original, normalized, factor)

    return output


def adain_normalize(
    latents: torch.Tensor,
    reference: torch.Tensor,
    factor: float = 1.0,
    per_frame: bool = False,
) -> torch.Tensor:
    """
    Adaptive Instance Normalization using reference statistics.

    Port of LTXVAdainLatent from ComfyUI-LTXVideo. Normalizes input latents
    to match the statistics of a reference tensor (typically the initial
    noise or a previous timestep's latents).

    Args:
        latents: Input tensor to normalize
        reference: Reference tensor to match statistics from
        factor: Interpolation factor (0=no change, 1=full normalization)
        per_frame: Compute stats per-frame vs globally

    Returns:
        Normalized latents matching reference statistics
    """
    if factor == 0.0:
        return latents

    original = latents

    # Compute stats for both tensors
    def compute_stats(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if per_frame and x.ndim == 5:
            # [B, C, F, H, W] - compute per batch per frame
            B, C, F, H, W = x.shape
            x_flat = x.permute(0, 2, 1, 3, 4).reshape(B, F, -1)
            mean = x_flat.mean(dim=-1, keepdim=True)
            std = x_flat.std(dim=-1, keepdim=True).clamp(min=1e-6)
            # Reshape for broadcast: [B, F, 1] -> [B, 1, F, 1, 1]
            mean = mean.unsqueeze(1).unsqueeze(-1)
            std = std.unsqueeze(1).unsqueeze(-1)
        else:
            # Global stats per batch
            x_flat = x.reshape(x.shape[0], -1)
            mean = x_flat.mean(dim=-1)
            std = x_flat.std(dim=-1).clamp(min=1e-6)
            # Reshape for broadcast
            shape = [x.shape[0]] + [1] * (x.ndim - 1)
            mean = mean.reshape(*shape)
            std = std.reshape(*shape)
        return mean, std

    input_mean, input_std = compute_stats(latents)
    ref_mean, ref_std = compute_stats(reference)

    # AdaIN: normalize input, then rescale to reference stats
    normalized = (latents - input_mean) / input_std * ref_std + ref_mean

    # Lerp between original and normalized
    output = torch.lerp(original, normalized, factor)

    return output


class PerStepNormalizer:
    """
    Per-step normalizer with tapering schedule.

    Applies normalization at each diffusion step with a tapering factor
    schedule. Typically starts aggressive (factor ~0.9) to correct early
    drift, then tapers off to avoid interfering with fine details.

    The factors string defines the interpolation strength at each step.
    If there are more steps than factors, the last factor is repeated.

    Example:
        # Aggressive early, none late
        normalizer = PerStepNormalizer(factors="0.9, 0.75, 0.5, 0.25, 0.0")

        # In diffusion loop:
        for i in range(num_steps):
            noise_pred = model(latents, t)
            noise_pred = normalizer(noise_pred, step=i)
            latents = scheduler.step(noise_pred, t, latents)
    """

    def __init__(
        self,
        factors: Union[str, List[float]] = "0.9, 0.75, 0.5, 0.25, 0.0",
        target_mean: float = 0.0,
        target_std: float = 1.0,
        percentile: float = 95.0,
        method: str = "statistical",
        reference: Optional[torch.Tensor] = None,
        per_frame: bool = False,
    ):
        """
        Initialize per-step normalizer.

        Args:
            factors: Comma-separated string or list of interpolation factors
            target_mean: Target mean for statistical method
            target_std: Target std for statistical method
            percentile: Percentile for robust stats
            method: "statistical" or "adain"
            reference: Reference tensor for adain method
            per_frame: Per-frame vs global normalization
        """
        if isinstance(factors, str):
            self.factors = [float(f.strip()) for f in factors.split(",")]
        else:
            self.factors = list(factors)

        self.target_mean = target_mean
        self.target_std = target_std
        self.percentile = percentile
        self.method = method
        self.reference = reference
        self.per_frame = per_frame

    def get_factor(self, step: int) -> float:
        """Get interpolation factor for given step."""
        if step >= len(self.factors):
            return self.factors[-1]
        return self.factors[step]

    def set_reference(self, reference: torch.Tensor) -> None:
        """Set reference tensor for AdaIN method."""
        self.reference = reference

    def __call__(
        self,
        latents: torch.Tensor,
        step: int,
        reference: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply normalization at given step.

        Args:
            latents: Input latents (typically noise prediction after CFG)
            step: Current diffusion step index (0-based)
            reference: Optional override for AdaIN reference

        Returns:
            Normalized latents
        """
        factor = self.get_factor(step)

        if factor == 0.0:
            return latents

        if self.method == "adain":
            ref = reference if reference is not None else self.reference
            if ref is None:
                raise ValueError("AdaIN method requires reference tensor")
            return adain_normalize(
                latents, ref, factor=factor, per_frame=self.per_frame
            )
        else:
            return statistical_normalize(
                latents,
                target_mean=self.target_mean,
                target_std=self.target_std,
                percentile=self.percentile,
                factor=factor,
                per_frame=self.per_frame,
            )

    def __repr__(self) -> str:
        return (
            f"PerStepNormalizer(method={self.method!r}, "
            f"factors={self.factors}, "
            f"target_mean={self.target_mean}, "
            f"target_std={self.target_std})"
        )


# =============================================================================
# Audio Latent Normalization (LTX-2 specific)
# =============================================================================


@dataclass
class AudioNormalizationConfig:
    """Configuration for audio latent normalization.

    LTX-2 generates joint video+audio latents. This config controls
    per-step normalization of the audio portion of the latents.

    The factors string specifies multipliers for each step. At certain
    steps, reducing audio latent magnitude (e.g., 0.25) can improve
    audio quality and prevent artifacts.

    Example factors: "1,1,0.25,1,1,0.25" - reduce at steps 2 and 5
    """

    enabled: bool = False
    factors: str = "1,1,0.25,1,1,0.25"

    def get_factors_list(self) -> List[float]:
        """Parse factors string into list."""
        return [float(f.strip()) for f in self.factors.split(",")]


def separate_audio_video_latents(
    latents: torch.Tensor,
    num_audio_channels: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Separate combined latents into video and audio portions.

    LTX-2 uses a combined latent space where audio and video latents
    are concatenated. This function splits them for independent processing.

    Args:
        latents: Combined latents [B, C, F, H, W] where C includes both
        num_audio_channels: Number of channels for audio (typically 16)

    Returns:
        (video_latents, audio_latents) tuple
    """
    total_channels = latents.shape[1]
    video_channels = total_channels - num_audio_channels

    video_latents = latents[:, :video_channels]
    audio_latents = latents[:, video_channels:]

    return video_latents, audio_latents


def recombine_audio_video_latents(
    video_latents: torch.Tensor,
    audio_latents: torch.Tensor,
) -> torch.Tensor:
    """
    Recombine video and audio latents after separate processing.

    Args:
        video_latents: Video portion of latents
        audio_latents: Audio portion of latents

    Returns:
        Combined latents [B, C_video + C_audio, F, H, W]
    """
    return torch.cat([video_latents, audio_latents], dim=1)


def normalize_audio_latents(
    latents: torch.Tensor,
    step: int,
    factors: Union[str, List[float]] = "1,1,0.25,1,1,0.25",
    num_audio_channels: int = 16,
) -> torch.Tensor:
    """
    Apply per-step normalization to audio latents.

    Ported from ComfyUI-KJNodes/nodes/ltxv_nodes.py (lines 765-867).

    LTX-2 can benefit from reducing audio latent magnitude at certain
    diffusion steps. This prevents audio artifacts and improves quality.

    Args:
        latents: Combined video+audio latents [B, C, F, H, W]
        step: Current diffusion step (0-indexed)
        factors: Per-step multipliers as string or list.
                 Values cycle if steps > len(factors).
        num_audio_channels: Number of audio channels (typically 16)

    Returns:
        Latents with audio portion normalized
    """
    # Parse factors if string
    if isinstance(factors, str):
        factor_list = [float(f.strip()) for f in factors.split(",")]
    else:
        factor_list = list(factors)

    # Get factor for this step (cycle if needed)
    factor = factor_list[step % len(factor_list)]

    # Early exit if no change needed
    if factor == 1.0:
        return latents

    # Separate latents
    video_latents, audio_latents = separate_audio_video_latents(
        latents, num_audio_channels
    )

    # Apply factor to audio
    audio_latents = audio_latents * factor

    # Recombine
    return recombine_audio_video_latents(video_latents, audio_latents)


class AudioLatentNormalizer:
    """
    Per-step audio latent normalizer.

    Provides a stateful interface for applying audio latent normalization
    across diffusion steps.

    Example:
        normalizer = AudioLatentNormalizer(factors="1,1,0.25,1,1,0.25")

        for step in range(num_steps):
            latents = scheduler.step(...)
            latents = normalizer(latents, step)
    """

    def __init__(
        self,
        factors: Union[str, List[float]] = "1,1,0.25,1,1,0.25",
        num_audio_channels: int = 16,
    ):
        """
        Initialize audio normalizer.

        Args:
            factors: Per-step multipliers
            num_audio_channels: Number of audio channels in latents
        """
        if isinstance(factors, str):
            self.factors = [float(f.strip()) for f in factors.split(",")]
        else:
            self.factors = list(factors)

        self.num_audio_channels = num_audio_channels

    def get_factor(self, step: int) -> float:
        """Get factor for given step (cycles if step >= len(factors))."""
        return self.factors[step % len(self.factors)]

    def __call__(
        self,
        latents: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        """
        Apply normalization at given step.

        Args:
            latents: Combined video+audio latents
            step: Current diffusion step

        Returns:
            Normalized latents
        """
        return normalize_audio_latents(
            latents,
            step=step,
            factors=self.factors,
            num_audio_channels=self.num_audio_channels,
        )

    def __repr__(self) -> str:
        return f"AudioLatentNormalizer(factors={self.factors})"
