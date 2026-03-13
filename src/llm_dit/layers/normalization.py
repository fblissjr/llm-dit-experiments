"""
Normalization layers - canonical implementations for all models.

Last Updated: 2026-02-01

This module consolidates 6+ RMSNorm implementations scattered across the codebase
into a single, configurable canonical implementation.

Previous implementations:
- z_image/components.py: RMSNorm (eps=1e-5)
- flux2/transformer.py: RMSNorm (eps=1e-6 fixed, param named 'scale')
- DiffSynth-Studio: RMSNorm (DiffSynth-matched)
- ltx2/components.py: rms_norm function
- embeddings_connector.py: rms_norm function (duplicate)

All are mathematically equivalent: x * rsqrt(mean(x^2) + eps) * weight
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Canonical implementation supporting all model variants:
    - Z-Image (eps=1e-5)
    - FLUX.2 Klein (eps=1e-6)
    - DiffSynth-Studio (eps=1e-6)
    - LTX-2 (uses F.rms_norm internally)

    Args:
        dim: Dimension of the input features (last dimension).
        eps: Small constant for numerical stability. Default 1e-6.
            Note: Z-Image uses 1e-5, most others use 1e-6.
        elementwise_affine: If True, includes learnable weight parameter.
            Default True. Set False for weight-free normalization.

    Shape:
        - Input: (..., dim)
        - Output: (..., dim)

    Example:
        >>> norm = RMSNorm(768, eps=1e-6)
        >>> x = torch.randn(2, 128, 768)
        >>> out = norm(x)
        >>> assert out.shape == x.shape

    Weight Loading Compatibility:
        The weight parameter is named 'weight' to match most existing checkpoints.
        For FLUX.2 checkpoints that use 'scale', use a key mapping:

            state_dict['weight'] = state_dict.pop('scale')
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization.

        Uses PyTorch's native F.rms_norm when available (PyTorch 2.4+),
        falls back to manual implementation otherwise.
        """
        # Use native implementation if available (faster, more numerically stable)
        if hasattr(F, "rms_norm"):
            return F.rms_norm(x, (self.dim,), weight=self.weight, eps=self.eps)

        # Manual implementation for older PyTorch versions
        return _rms_norm_manual(x, self.weight, self.eps)

    def extra_repr(self) -> str:
        return f"{self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"


def rms_norm(
    x: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Functional RMS normalization over the last dimension.

    This is the functional equivalent of the RMSNorm class, useful when
    the weight is managed externally (e.g., in fused operations).

    Args:
        x: Input tensor of shape (..., dim).
        weight: Optional learnable scale of shape (dim,).
        eps: Small constant for numerical stability.

    Returns:
        Normalized tensor of same shape as input.

    Example:
        >>> x = torch.randn(2, 128, 768)
        >>> weight = torch.ones(768)
        >>> out = rms_norm(x, weight, eps=1e-6)
    """
    if hasattr(F, "rms_norm"):
        return F.rms_norm(x, (x.shape[-1],), weight=weight, eps=eps)

    return _rms_norm_manual(x, weight, eps)


def _rms_norm_manual(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    eps: float,
) -> torch.Tensor:
    """
    Manual RMS normalization implementation.

    Used as fallback when F.rms_norm is not available (PyTorch < 2.4).

    Implementation notes:
    - Casts to float32 for numerical precision during computation
    - Uses rsqrt for efficiency (single op vs sqrt + division)
    - Casts back to original dtype before applying weight
    """
    input_dtype = x.dtype

    # Cast to float32 for precision (matches DiffSynth, FLUX.2 implementations)
    x = x.float()

    # RMS = sqrt(mean(x^2))
    # Output = x / RMS = x * rsqrt(mean(x^2) + eps)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)

    # Cast back to original dtype, then apply weight
    x = x.to(input_dtype)

    if weight is not None:
        x = x * weight

    return x


# Compatibility aliases for migration
# These can be removed after all models migrate to the canonical implementation

class T5LayerNorm(RMSNorm):
    """
    T5-style RMS normalization.

    This is functionally identical to RMSNorm but provided for compatibility
    with T5 model code and checkpoints.

    Deprecated: Use RMSNorm directly.
    """
    pass
