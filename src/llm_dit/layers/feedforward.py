"""
FeedForward layers - canonical implementations for all models.

Last Updated: 2026-02-01

This module consolidates 5+ FeedForward implementations scattered across the codebase
into a single, configurable canonical implementation.

Previous implementations:
- ltx2/components.py: FeedForward (GELU tanh, standard MLP)
- z_image/components.py: FeedForward (SwiGLU, bias=False)
- DiffSynth-Studio: T5FeedForward (GeGLU, dropout)
- embeddings_connector.py: FeedForward (duplicate of ltx2)
- flux2/transformer.py: SiLUActivation + nn.Sequential (split tensor gating)

Architecture Variants:
- STANDARD: Linear -> Activation -> Linear (LTX-2, Connectors)
- SWIGLU: silu(w1(x)) * w3(x) -> w2 (Z-Image, Qwen-style)
- GEGLU: fc1(x) * gelu(gate(x)) -> fc2 (T5)

All variants use separate projection weights (not split tensors) for clarity
and checkpoint compatibility.
"""

from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FFNType(Enum):
    """FeedForward architecture variants."""

    STANDARD = "standard"  # Linear -> Act -> Linear (LTX-2)
    SWIGLU = "swiglu"      # SwiGLU: silu(w1) * w3 -> w2 (Z-Image)
    GEGLU = "geglu"        # GeGLU: fc1 * gelu(gate) -> fc2 (T5)


class FeedForward(nn.Module):
    """
    Configurable FeedForward layer supporting multiple architectures.

    This is the canonical implementation for all DiT models. Each model's
    original FFN behavior can be replicated through configuration.

    Args:
        dim: Input/output dimension.
        hidden_dim: Hidden dimension. If None, computed from mult.
        mult: Multiplier for hidden_dim (default 4). Ignored if hidden_dim set.
        ffn_type: Architecture variant (standard, swiglu, geglu).
        activation: Activation function ("gelu", "gelu_tanh", "silu").
            Only used by STANDARD type; SWIGLU always uses silu, GEGLU uses gelu.
        dropout: Dropout probability (default 0.0).
        bias: Whether to use bias in linear layers.

    Architecture Details:

    STANDARD (LTX-2, Connectors):
        hidden = activation(linear1(x))
        output = linear2(hidden)
        - Uses mult=4 typically
        - activation="gelu_tanh" for LTX-2

    SWIGLU (Z-Image, Qwen):
        gate = silu(w1(x))
        hidden = gate * w3(x)
        output = w2(hidden)
        - Uses mult=8/3 typically (Qwen-style)
        - bias=False typically

    GEGLU (T5):
        gate = gelu(gate_proj(x))
        hidden = fc1(x) * gate
        output = fc2(hidden)
        - Includes dropout after fc1*gate and fc2
        - bias=False typically

    Example:
        >>> # LTX-2 style
        >>> ff = FeedForward(4096, ffn_type=FFNType.STANDARD, activation="gelu_tanh")
        >>>
        >>> # Z-Image style
        >>> ff = FeedForward(3072, mult=8/3, ffn_type=FFNType.SWIGLU, bias=False)
        >>>
        >>> # T5 style
        >>> ff = FeedForward(4096, hidden_dim=10240, ffn_type=FFNType.GEGLU, dropout=0.1)

    Weight Loading Compatibility:
        The parameter names match the original implementations:
        - STANDARD: net.0.proj (GELUApprox) + net.2 (Linear)
        - SWIGLU: w1, w2, w3
        - GEGLU: gate (Sequential), fc1, fc2
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        mult: float = 4.0,
        ffn_type: FFNType = FFNType.STANDARD,
        activation: str = "gelu_tanh",
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()

        self.dim = dim
        self.ffn_type = ffn_type
        self.dropout_prob = dropout

        # Compute hidden dimension
        if hidden_dim is None:
            hidden_dim = int(dim * mult)
        self.hidden_dim = hidden_dim

        # Build architecture based on type
        if ffn_type == FFNType.STANDARD:
            self._build_standard(dim, hidden_dim, activation, bias)
        elif ffn_type == FFNType.SWIGLU:
            self._build_swiglu(dim, hidden_dim, bias)
        elif ffn_type == FFNType.GEGLU:
            self._build_geglu(dim, hidden_dim, dropout, bias)
        else:
            raise ValueError(f"Unknown FFN type: {ffn_type}")

    def _build_standard(
        self,
        dim: int,
        hidden_dim: int,
        activation: str,
        bias: bool,
    ) -> None:
        """
        Build standard FFN (LTX-2 style).

        Structure: Linear -> Activation -> Identity (placeholder) -> Linear
        """
        # Match ltx2/components.py structure: GELUApprox -> Identity -> Linear
        # Using nn.Sequential with named structure for checkpoint compatibility
        self.net = nn.Sequential(
            _GELUApprox(dim, hidden_dim, activation=activation, bias=bias),
            nn.Identity(),  # Placeholder for dropout if needed
            nn.Linear(hidden_dim, dim, bias=bias),
        )

    def _build_swiglu(
        self,
        dim: int,
        hidden_dim: int,
        bias: bool,
    ) -> None:
        """
        Build SwiGLU FFN (Z-Image style).

        Structure: silu(w1(x)) * w3(x) -> w2
        """
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)

    def _build_geglu(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float,
        bias: bool,
    ) -> None:
        """
        Build GeGLU FFN (T5 style).

        Structure: fc1(x) * gelu(gate(x)) -> dropout -> fc2 -> dropout
        """
        # Match T5 FeedForward structure (GeGLU gating)
        self.gate = nn.Sequential(nn.Linear(dim, hidden_dim, bias=bias), _GELU())
        self.fc1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply feed-forward transformation.

        Args:
            x: Input tensor of shape (..., dim)

        Returns:
            Output tensor of shape (..., dim)
        """
        if self.ffn_type == FFNType.STANDARD:
            return self.net(x)
        elif self.ffn_type == FFNType.SWIGLU:
            return self._forward_swiglu(x)
        elif self.ffn_type == FFNType.GEGLU:
            return self._forward_geglu(x)
        else:
            raise ValueError(f"Unknown FFN type: {self.ffn_type}")

    def _forward_swiglu(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU forward: silu(w1(x)) * w3(x) -> w2"""
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def _forward_geglu(self, x: torch.Tensor) -> torch.Tensor:
        """GeGLU forward: fc1(x) * gelu(gate(x)) -> dropout -> fc2 -> dropout"""
        x = self.fc1(x) * self.gate(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

    def extra_repr(self) -> str:
        return (
            f"{self.dim}, hidden_dim={self.hidden_dim}, "
            f"ffn_type={self.ffn_type.value}, dropout={self.dropout_prob}"
        )


class _GELUApprox(nn.Module):
    """
    GELU activation with linear projection (internal use).

    Matches the GELUApprox class from ltx2/components.py for checkpoint compatibility.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        activation: str = "gelu_tanh",
        bias: bool = True,
    ):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        if self.activation == "gelu_tanh":
            return F.gelu(x, approximate="tanh")
        elif self.activation == "gelu":
            return F.gelu(x)
        elif self.activation == "silu":
            return F.silu(x)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")


class _GELU(nn.Module):
    """
    GELU activation matching T5 implementation (tanh approximation).

    Internal use for GeGLU to match T5 implementation exactly.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # T5 uses the manual tanh approximation formula
        import math

        return 0.5 * x * (
            1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        )


# ============================================================================
# Preset Configurations
# ============================================================================

# These presets match the exact configuration of each original implementation.
# Use with: FeedForward(dim, **LTX2_FFN_PRESET)

LTX2_FFN_PRESET = dict(
    ffn_type=FFNType.STANDARD,
    activation="gelu_tanh",
    mult=4.0,
    bias=True,
)
"""LTX-2 FeedForward: Standard MLP with GELU(tanh), 4x hidden, bias=True."""

ZIMAGE_FFN_PRESET = dict(
    ffn_type=FFNType.SWIGLU,
    mult=8 / 3,  # Qwen-style hidden ratio
    bias=False,
)
"""Z-Image FeedForward: SwiGLU with 8/3x hidden, bias=False."""


CONNECTOR_FFN_PRESET = dict(
    ffn_type=FFNType.STANDARD,
    activation="gelu_tanh",
    mult=4.0,
    bias=True,
)
"""Connector FeedForward: Same as LTX-2 (standard MLP)."""
