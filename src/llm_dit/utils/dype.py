"""DyPE (Dynamic Position Extrapolation) for high-resolution generation.

DyPE is a training-free technique that enables generating images at
resolutions higher than the model's training resolution by dynamically
adjusting RoPE frequencies based on the diffusion timestep.

Key insight: Early diffusion steps establish low-frequency structure
while late steps add high-frequency details. DyPE matches position
encoding frequencies to this spectral progression.

Based on ComfyUI-DyPE implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Tuple, List

import torch
import torch.nn as nn

from .vision_yarn import (
    get_1d_vision_yarn_pos_embed,
    get_1d_yarn_pos_embed,
    get_1d_ntk_pos_embed,
)


@dataclass
class DyPEConfig:
    """Configuration for DyPE (Dynamic Position Extrapolation).

    Attributes:
        enabled: Whether DyPE is enabled
        method: RoPE extrapolation method (vision_yarn, yarn, ntk)
        dype_scale: Magnitude of DyPE effect (lambda_s)
        dype_exponent: Decay speed of DyPE (lambda_t, quadratic=2.0)
        dype_start_sigma: When to start DyPE decay (0-1, 1.0=from start)
        base_shift: Noise schedule shift at base resolution
        max_shift: Noise schedule shift at max resolution
        base_resolution: Training resolution (Z-Image: 1024, Qwen: 1328)
        anisotropic: Use per-axis scaling for extreme aspect ratios
    """

    enabled: bool = False
    method: Literal["vision_yarn", "yarn", "ntk"] = "vision_yarn"
    dype_scale: float = 2.0
    dype_exponent: float = 2.0
    dype_start_sigma: float = 1.0
    base_shift: float = 0.5
    max_shift: float = 1.15
    base_resolution: int = 1024
    anisotropic: bool = False

    def __post_init__(self):
        """Validate and clamp parameters."""
        self.dype_start_sigma = max(0.001, min(1.0, self.dype_start_sigma))

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "enabled": self.enabled,
            "method": self.method,
            "dype_scale": self.dype_scale,
            "dype_exponent": self.dype_exponent,
            "dype_start_sigma": self.dype_start_sigma,
            "base_shift": self.base_shift,
            "max_shift": self.max_shift,
            "base_resolution": self.base_resolution,
            "anisotropic": self.anisotropic,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DyPEConfig":
        """Create config from dictionary."""
        return cls(
            enabled=d.get("enabled", False),
            method=d.get("method", "vision_yarn"),
            dype_scale=d.get("dype_scale", 2.0),
            dype_exponent=d.get("dype_exponent", 2.0),
            dype_start_sigma=d.get("dype_start_sigma", 1.0),
            base_shift=d.get("base_shift", 0.5),
            max_shift=d.get("max_shift", 1.15),
            base_resolution=d.get("base_resolution", 1024),
            anisotropic=d.get("anisotropic", False),
        )


def compute_dype_shift(
    image_seq_len: int,
    base_seq_len: int,
    max_seq_len: int,
    config: DyPEConfig,
) -> float:
    """Compute dynamic noise schedule shift based on resolution.

    Higher resolutions get higher shift values, which pushes the noise
    schedule toward early denoising, focusing on fine details.

    Args:
        image_seq_len: Current image sequence length (patches)
        base_seq_len: Base resolution sequence length
        max_seq_len: Maximum expected sequence length
        config: DyPE configuration

    Returns:
        Noise schedule shift value
    """
    if max_seq_len <= base_seq_len:
        return config.base_shift

    slope = (config.max_shift - config.base_shift) / (max_seq_len - base_seq_len)
    intercept = config.base_shift - slope * base_seq_len
    return image_seq_len * slope + intercept


def compute_k_t(sigma: float, config: DyPEConfig) -> float:
    """Compute timestep-dependent scaling factor k_t.

    k_t controls how aggressively the RoPE frequencies are scaled
    based on the current diffusion timestep.

    Formula: k_t = dype_scale * (sigma ^ dype_exponent)

    Args:
        sigma: Current normalized sigma (0=clean, 1=noise)
        config: DyPE configuration

    Returns:
        Scaling factor k_t
    """
    return config.dype_scale * (sigma**config.dype_exponent)


def compute_mscale(scale_global: float, sigma: float, config: DyPEConfig) -> float:
    """Compute amplitude scaling factor (mscale).

    The mscale compensates for amplitude changes when frequencies are scaled.
    It transitions from aggressive scaling early to baseline late.

    Args:
        scale_global: Global resolution scale factor
        sigma: Current normalized sigma (0=clean, 1=noise)
        config: DyPE configuration

    Returns:
        Amplitude scaling factor
    """
    if scale_global <= 1.0:
        return 1.0

    mscale_start = 0.1 * math.log(scale_global) + 1.0
    mscale_end = 1.0

    t_effective = sigma
    t_norm = (
        1.0
        if t_effective > config.dype_start_sigma
        else (t_effective / config.dype_start_sigma)
    )

    return mscale_end + (mscale_start - mscale_end) * (t_norm**config.dype_exponent)


def axis_token_span(axis_pos: torch.Tensor) -> float:
    """Calculate the effective span of tokens along an axis.

    Args:
        axis_pos: Position indices along one axis

    Returns:
        Effective span (number of unique positions)
    """
    flat = axis_pos.float().reshape(-1)

    if flat.numel() <= 1:
        return 1.0

    min_val, max_val = flat.min(), flat.max()
    span = max_val - min_val

    if span <= 0:
        return 1.0

    unique_vals = torch.unique(flat)

    if unique_vals.numel() <= 1:
        return 1.0

    step = torch.diff(unique_vals).min().item()

    if step <= 1e-6:
        return float(flat.numel())
    return float((span / step) + 1.0)


class DyPEPosEmbed(nn.Module):
    """Dynamic Position Embedding base class.

    Computes position embeddings with Vision YaRN / YaRN / NTK methods
    and optional DyPE timestep modulation.

    This class handles the core computation; subclasses format output
    for specific model architectures (Z-Image, Qwen-Image, etc.).
    """

    def __init__(
        self,
        theta: float,
        axes_dim: List[int],
        config: DyPEConfig | None = None,
        base_patch_grid: Tuple[int, int] | None = None,
    ):
        """Initialize DyPE position embedder.

        Args:
            theta: RoPE base frequency
            axes_dim: Dimensions per axis [text, height, width]
            config: DyPE configuration (defaults to disabled)
            base_patch_grid: Base resolution patch grid (H, W)
        """
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.config = config or DyPEConfig()

        # Determine base patch grid
        if base_patch_grid is None:
            # Default: 1024px -> 128 latent -> 64 patches (patch_size=2)
            val = (self.config.base_resolution // 8) // 2
            self.base_patch_grid = (val, val)
        elif isinstance(base_patch_grid, int):
            self.base_patch_grid = (base_patch_grid, base_patch_grid)
        else:
            self.base_patch_grid = base_patch_grid

        self.base_patches = max(self.base_patch_grid)
        self.current_sigma = 1.0

    def set_timestep(self, sigma: float):
        """Set current diffusion timestep for DyPE modulation.

        Args:
            sigma: Normalized sigma value (0=clean, 1=noise)
        """
        self.current_sigma = sigma

    def get_components(
        self,
        pos: torch.Tensor,
        freqs_dtype: torch.dtype = torch.float32,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Compute (cos, sin) components for each axis.

        Args:
            pos: Position indices tensor of shape (..., seq_len, n_axes)
            freqs_dtype: Data type for frequency computation

        Returns:
            List of (cos, sin) tuples, one per axis
        """
        method = self.config.method if self.config.enabled else "ntk"

        if method == "vision_yarn":
            return self._calc_vision_yarn_components(pos, freqs_dtype)
        elif method == "yarn":
            return self._calc_yarn_components(pos, freqs_dtype)
        else:
            return self._calc_ntk_components(pos, freqs_dtype)

    def _calc_vision_yarn_components(
        self,
        pos: torch.Tensor,
        freqs_dtype: torch.dtype,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Compute Vision YaRN position embedding components."""
        n_axes = pos.shape[-1]
        components = []

        # Compute global scale from spatial dimensions
        if n_axes >= 3:
            h_span = axis_token_span(pos[..., 1])
            w_span = axis_token_span(pos[..., 2])
            scale_global = max(
                1.0,
                max(
                    h_span / self.base_patch_grid[0], w_span / self.base_patch_grid[1]
                ),
            )
        else:
            max_current_patches = axis_token_span(pos)
            scale_global = max(1.0, max_current_patches / self.base_patches)

        current_mscale = compute_mscale(scale_global, self.current_sigma, self.config)

        for i in range(n_axes):
            axis_pos = pos[..., i]
            axis_dim = self.axes_dim[i]
            current_patches = axis_token_span(axis_pos)

            if i > 0 and scale_global > 1.0:
                # Spatial axes with extrapolation needed
                base_axis_len = (
                    self.base_patch_grid[i - 1]
                    if (n_axes >= 3 and i - 1 < len(self.base_patch_grid))
                    else self.base_patches
                )

                scale_local = max(1.0, current_patches / base_axis_len)

                cos, sin = get_1d_vision_yarn_pos_embed(
                    dim=axis_dim,
                    pos=axis_pos,
                    theta=self.theta,
                    linear_scale=scale_local,
                    ntk_scale=scale_global,
                    ori_max_pe_len=base_axis_len,
                    dype=self.config.enabled,
                    current_timestep=self.current_sigma,
                    dype_scale=self.config.dype_scale,
                    dype_exponent=self.config.dype_exponent,
                    override_mscale=current_mscale,
                    freqs_dtype=freqs_dtype,
                )
            else:
                # Text axis or no extrapolation needed
                cos, sin = get_1d_ntk_pos_embed(
                    dim=axis_dim,
                    pos=axis_pos,
                    theta=self.theta,
                    ntk_factor=1.0,
                    freqs_dtype=freqs_dtype,
                )

            components.append((cos, sin))

        return components

    def _calc_yarn_components(
        self,
        pos: torch.Tensor,
        freqs_dtype: torch.dtype,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Compute YaRN position embedding components."""
        n_axes = pos.shape[-1]
        components = []

        # Compute if extrapolation is needed
        if n_axes >= 3:
            h_span = axis_token_span(pos[..., 1])
            w_span = axis_token_span(pos[..., 2])
            max_current_patches = max(h_span, w_span)
        else:
            max_current_patches = axis_token_span(pos)

        needs_extrapolation = max_current_patches > self.base_patches

        if needs_extrapolation and self.config.anisotropic:
            # Per-axis YaRN
            for i in range(n_axes):
                axis_pos = pos[..., i]
                axis_dim = self.axes_dim[i]
                current_patches = axis_token_span(axis_pos)
                base_axis_len = (
                    self.base_patch_grid[i - 1]
                    if (n_axes >= 3 and i > 0 and i - 1 < len(self.base_patch_grid))
                    else self.base_patches
                )

                if i > 0 and current_patches > base_axis_len:
                    max_pe_len = torch.tensor(
                        current_patches, dtype=freqs_dtype, device=pos.device
                    )
                    cos, sin = get_1d_yarn_pos_embed(
                        dim=axis_dim,
                        pos=axis_pos,
                        theta=self.theta,
                        max_pe_len=max_pe_len,
                        ori_max_pe_len=base_axis_len,
                        dype=self.config.enabled,
                        current_timestep=self.current_sigma,
                        dype_scale=self.config.dype_scale,
                        dype_exponent=self.config.dype_exponent,
                        use_aggressive_mscale=True,
                        freqs_dtype=freqs_dtype,
                    )
                else:
                    cos, sin = get_1d_ntk_pos_embed(
                        dim=axis_dim,
                        pos=axis_pos,
                        theta=self.theta,
                        ntk_factor=1.0,
                        freqs_dtype=freqs_dtype,
                    )

                components.append((cos, sin))
        else:
            # Isotropic YaRN
            cos_full_spatial, sin_full_spatial = None, None
            if needs_extrapolation:
                spatial_axis_dim = self.axes_dim[1]
                square_pos = torch.arange(
                    0, max_current_patches, device=pos.device
                ).float()
                max_pe_len = torch.tensor(
                    max_current_patches, dtype=freqs_dtype, device=pos.device
                )

                cos_full_spatial, sin_full_spatial = get_1d_yarn_pos_embed(
                    dim=spatial_axis_dim,
                    pos=square_pos,
                    theta=self.theta,
                    max_pe_len=max_pe_len,
                    ori_max_pe_len=self.base_patches,
                    dype=self.config.enabled,
                    current_timestep=self.current_sigma,
                    dype_scale=self.config.dype_scale,
                    dype_exponent=self.config.dype_exponent,
                    use_aggressive_mscale=False,
                    freqs_dtype=freqs_dtype,
                )

            for i in range(n_axes):
                axis_pos = pos[..., i]
                axis_dim = self.axes_dim[i]

                if i > 0 and needs_extrapolation:
                    offset_indices = axis_pos.long() - axis_pos.long().min()
                    pos_indices = offset_indices.view(-1)
                    pos_indices = torch.clamp(
                        pos_indices, max=cos_full_spatial.shape[0] - 1
                    )

                    cos = cos_full_spatial[pos_indices].view(*axis_pos.shape, -1)
                    sin = sin_full_spatial[pos_indices].view(*axis_pos.shape, -1)
                else:
                    cos, sin = get_1d_ntk_pos_embed(
                        dim=axis_dim,
                        pos=axis_pos,
                        theta=self.theta,
                        ntk_factor=1.0,
                        freqs_dtype=freqs_dtype,
                    )

                components.append((cos, sin))

        return components

    def _calc_ntk_components(
        self,
        pos: torch.Tensor,
        freqs_dtype: torch.dtype,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Compute NTK-scaled position embedding components."""
        n_axes = pos.shape[-1]
        components = []

        # Compute global scale
        if n_axes >= 3:
            h_span = axis_token_span(pos[..., 1])
            w_span = axis_token_span(pos[..., 2])
            scale_global = max(
                1.0,
                max(
                    h_span / self.base_patch_grid[0], w_span / self.base_patch_grid[1]
                ),
            )
        else:
            max_current_patches = axis_token_span(pos)
            scale_global = max(1.0, max_current_patches / self.base_patches)

        for i in range(n_axes):
            axis_pos = pos[..., i]
            axis_dim = self.axes_dim[i]

            ntk_factor = 1.0
            if i > 0 and scale_global > 1.0:
                base_ntk = scale_global ** (axis_dim / (axis_dim - 2))
                if self.config.enabled:
                    k_t = compute_k_t(self.current_sigma, self.config)
                    ntk_factor = base_ntk**k_t
                else:
                    ntk_factor = base_ntk
                ntk_factor = max(1.0, ntk_factor)

            cos, sin = get_1d_ntk_pos_embed(
                dim=axis_dim,
                pos=axis_pos,
                theta=self.theta,
                ntk_factor=ntk_factor,
                freqs_dtype=freqs_dtype,
            )
            components.append((cos, sin))

        return components

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        """Compute position embeddings.

        Subclasses should override this to format output for their
        specific model architecture.

        Args:
            pos: Position indices tensor

        Returns:
            Position embeddings in model-specific format
        """
        raise NotImplementedError(
            "Base class does not implement forward. Use a specific model subclass."
        )


class ZImageDyPERoPE(DyPEPosEmbed):
    """DyPE-enabled RoPE wrapper for Z-Image transformer.

    This class wraps the original Z-Image rope_embedder and can compute
    DyPE-modulated frequencies when enabled, or delegate to the original
    when disabled.

    The Z-Image transformer uses multi-axis RoPE with:
    - axes_lens = [1536, 512, 512] (text, height, width)
    - axes_dims = [32, 48, 48]
    - theta = 256
    """

    def __init__(
        self,
        original_embedder: nn.Module,
        config: DyPEConfig | None = None,
        scale_hint: float = 1.0,
    ):
        """Initialize Z-Image DyPE RoPE wrapper.

        Args:
            original_embedder: Original rope_embedder from transformer
            config: DyPE configuration
            scale_hint: Resolution scale hint from external source
        """
        # Get parameters from original embedder BEFORE super().__init__()
        # Must use local variables since we can't assign self.* before nn.Module init
        theta = getattr(original_embedder, "theta", 256)
        axes_dim = getattr(original_embedder, "axes_dim", [32, 48, 48])

        # Initialize base class FIRST (required by nn.Module)
        super().__init__(
            theta=theta,
            axes_dim=axes_dim,
            config=config,
            base_patch_grid=None,  # Will use config.base_resolution
        )

        # Now safe to assign module attributes
        self.original_embedder = original_embedder
        self.scale_hint = scale_hint

    def set_scale_hint(self, scale: float):
        """Set external resolution scale hint.

        Called by patched patchify_and_embed to inform the embedder
        of the actual resolution scale being used.

        Args:
            scale: Resolution scale factor (>1 means extrapolation needed)
        """
        self.scale_hint = max(1.0, scale)

    def _blend_to_full_scale(self) -> float:
        """Compute blend factor for coordinate rescaling.

        Returns a value from 0 to 1 indicating how much to blend from
        fractional (PI) coordinates to full integer scale.

        Returns:
            Blend factor (0 at start, 1 at end of denoising)
        """
        if not self.config.enabled:
            return 0.0

        t_effective = self.current_sigma
        if t_effective > self.config.dype_start_sigma:
            t_norm = 1.0
        else:
            t_norm = t_effective / self.config.dype_start_sigma

        t_factor = t_norm**self.config.dype_exponent
        return 1.0 - t_factor

    def _resize_rope_grid(self, pos: torch.Tensor) -> torch.Tensor:
        """Dynamically expand coordinates from PI (fractional) to integer.

        Z-Image uses fractional coordinates for PI (position interpolation).
        This method blends toward full integer coordinates based on timestep.

        Args:
            pos: Position indices tensor of shape (..., 3)

        Returns:
            Rescaled position indices
        """
        if not self.config.enabled:
            return pos

        # Check if we have image tokens (spatial dimensions non-zero)
        image_mask = (pos[..., 1] != 0) | (pos[..., 2] != 0)
        if not image_mask.any():
            return pos

        blend_val = self._blend_to_full_scale()
        if blend_val <= 0.001:
            return pos

        blend = torch.tensor(blend_val, device=pos.device, dtype=pos.dtype)
        pos_rescaled = pos.clone()

        for axis in (1, 2):  # Height and width axes
            coords = pos[..., axis]
            coords_image = coords[image_mask]
            if coords_image.numel() <= 1:
                continue

            unique_coords = torch.unique(coords_image)
            if unique_coords.numel() <= 1:
                continue

            unique_sorted, _ = torch.sort(unique_coords)
            deltas = torch.diff(unique_sorted)
            if deltas.numel() == 0:
                continue
            step = torch.median(deltas)

            # Skip if already integer-scaled
            if torch.isclose(
                step, torch.tensor(1.0, device=pos.device, dtype=pos.dtype), atol=1e-3
            ):
                continue
            if torch.isclose(
                step, torch.tensor(0.0, device=pos.device, dtype=pos.dtype)
            ):
                continue

            start = coords_image.min()
            full_scale_coords = (coords - start) / step + start
            pos_rescaled[..., axis] = coords + (full_scale_coords - coords) * blend

        return pos_rescaled

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Compute RoPE embeddings for Z-Image.

        Args:
            ids: Position indices tensor of shape (B, seq_len, 3)

        Returns:
            RoPE embeddings in Z-Image format (complex64)

        Note:
            The diffusers RopeEmbedder returns complex64 embeddings using torch.polar.
            Currently, we delegate to the original embedder for all cases because
            reimplementing the exact complex format with DyPE modulation requires
            matching diffusers' internal apply_rotary_emb function.

            TODO: Implement DyPE-modulated complex embeddings that match the
            diffusers format: torch.polar(ones, angles) -> complex64
        """
        # Always delegate to original embedder for now
        # The DyPE timestep modulation needs to be applied differently
        # (modifying how frequencies are computed in the original embedder)
        # rather than replacing the embedder output format
        return self.original_embedder(ids)


def patch_zimage_rope(
    transformer: nn.Module,
    config: DyPEConfig,
    width: int,
    height: int,
) -> nn.Module:
    """Patch Z-Image transformer with DyPE-enabled RoPE.

    This function replaces the transformer's rope_embedder with a
    DyPE-enabled wrapper that can dynamically adjust RoPE frequencies
    based on the diffusion timestep.

    Args:
        transformer: Z-Image transformer model (ZImageTransformer2DModel)
        config: DyPE configuration
        width: Target image width in pixels
        height: Target image height in pixels

    Returns:
        The patched transformer (same object, modified in place)

    Example:
        from llm_dit.utils.dype import patch_zimage_rope, DyPEConfig

        config = DyPEConfig(enabled=True, method="vision_yarn")
        pipe.transformer = patch_zimage_rope(pipe.transformer, config, 2048, 2048)
    """
    if not hasattr(transformer, "rope_embedder"):
        raise ValueError(
            "Transformer does not have rope_embedder attribute. "
            "Is this a ZImageTransformer2DModel?"
        )

    original_embedder = transformer.rope_embedder

    # Compute scale hint from resolution
    base_patches = (config.base_resolution // 8) // 2  # patch_size=2
    target_patches_h = (height // 8) // 2
    target_patches_w = (width // 8) // 2
    scale_hint = max(
        1.0,
        max(target_patches_h / base_patches, target_patches_w / base_patches),
    )

    # Create DyPE wrapper
    dype_embedder = ZImageDyPERoPE(
        original_embedder=original_embedder,
        config=config,
        scale_hint=scale_hint,
    )

    # Replace the rope_embedder
    transformer.rope_embedder = dype_embedder

    return transformer


def set_zimage_timestep(transformer: nn.Module, sigma: float) -> None:
    """Set timestep on Z-Image transformer's DyPE embedder.

    Call this at each denoising step to update the DyPE modulation.

    Args:
        transformer: Z-Image transformer with patched rope_embedder
        sigma: Normalized sigma value (0=clean, 1=noise)

    Example:
        for i, t in enumerate(timesteps):
            sigma = t.item()
            set_zimage_timestep(pipe.transformer, sigma)
            noise_pred = pipe.transformer(...)
    """
    if hasattr(transformer, "rope_embedder") and isinstance(
        transformer.rope_embedder, ZImageDyPERoPE
    ):
        transformer.rope_embedder.set_timestep(sigma)


def unpatch_zimage_rope(transformer: nn.Module) -> nn.Module:
    """Restore original rope_embedder on Z-Image transformer.

    If the transformer was patched with DyPE, this restores the original
    rope_embedder. Safe to call even if not patched (no-op).

    Args:
        transformer: Z-Image transformer (possibly with DyPE patch)

    Returns:
        The transformer with original rope_embedder restored
    """
    if hasattr(transformer, "rope_embedder") and isinstance(
        transformer.rope_embedder, ZImageDyPERoPE
    ):
        original = transformer.rope_embedder.original_embedder
        transformer.rope_embedder = original
    return transformer


def is_zimage_patched(transformer: nn.Module) -> bool:
    """Check if Z-Image transformer has DyPE patch applied.

    Args:
        transformer: Z-Image transformer

    Returns:
        True if DyPE patch is currently applied
    """
    return hasattr(transformer, "rope_embedder") and isinstance(
        transformer.rope_embedder, ZImageDyPERoPE
    )
