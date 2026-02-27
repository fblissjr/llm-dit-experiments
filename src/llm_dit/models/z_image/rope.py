"""
Rotary Position Embeddings (RoPE) for Z-Image DiT.

Last updated: 2026-01-29

Implements 3-axis RoPE for joint position encoding of:
- Text sequence position (axis 0)
- Image height position (axis 1)
- Image width position (axis 2)

Based on DiffSynth-Studio implementation.
"""

from typing import List, Optional

import torch


class RopeEmbedder:
    """
    3-axis Rotary Position Embedder for Z-Image DiT.

    Creates position-dependent frequency tables for RoPE, supporting
    three independent axes for text+spatial position encoding.

    Args:
        theta: Base frequency for RoPE (default: 256.0)
        axes_dims: Dimensions allocated to each axis [text, height, width]
        axes_lens: Maximum positions for each axis

    Example:
        >>> embedder = RopeEmbedder(
        ...     theta=256.0,
        ...     axes_dims=[32, 48, 48],
        ...     axes_lens=[1024, 512, 512],
        ... )
        >>> # ids shape: (seq_len, 3) with [text_pos, h_pos, w_pos]
        >>> freqs_cis = embedder(position_ids)  # (seq_len, 64) complex
    """

    def __init__(
        self,
        theta: float = 256.0,
        axes_dims: List[int] = [32, 48, 48],
        axes_lens: List[int] = [1024, 512, 512],
    ):
        self.theta = theta
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        assert len(axes_dims) == len(axes_lens), "axes_dims and axes_lens must have the same length"
        self.freqs_cis: Optional[List[torch.Tensor]] = None

    @staticmethod
    def precompute_freqs_cis(
        dim: List[int],
        end: List[int],
        theta: float = 256.0,
    ) -> List[torch.Tensor]:
        """
        Precompute frequency tables for each axis.

        Args:
            dim: Dimensions for each axis
            end: Max positions for each axis
            theta: Base frequency

        Returns:
            List of complex frequency tensors, one per axis
        """
        with torch.device("cpu"):
            freqs_cis = []
            for i, (d, e) in enumerate(zip(dim, end)):
                # Compute frequencies: theta^(-2k/d) for k in [0, d/2)
                freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64, device="cpu") / d))
                # Create position indices
                timestep = torch.arange(e, device=freqs.device, dtype=torch.float64)
                # Outer product: (positions, freqs)
                freqs = torch.outer(timestep, freqs).float()
                # Convert to complex exponentials
                freqs_cis_i = torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64)
                freqs_cis.append(freqs_cis_i)

            return freqs_cis

    def __call__(self, ids: torch.Tensor) -> torch.Tensor:
        """
        Compute RoPE frequencies for given position IDs.

        Args:
            ids: Position IDs of shape (seq_len, num_axes)
                 Each row contains [text_pos, h_pos, w_pos]

        Returns:
            Complex frequency tensor of shape (seq_len, total_dim/2)
            where total_dim = sum(axes_dims)
        """
        assert ids.ndim == 2
        assert ids.shape[-1] == len(self.axes_dims)
        device = ids.device

        # Lazy initialization of frequency tables
        if self.freqs_cis is None:
            self.freqs_cis = self.precompute_freqs_cis(
                self.axes_dims,
                self.axes_lens,
                theta=self.theta,
            )
            self.freqs_cis = [freqs_cis.to(device) for freqs_cis in self.freqs_cis]
        elif self.freqs_cis[0].device != device:
            # Move to correct device if needed
            self.freqs_cis = [freqs_cis.to(device) for freqs_cis in self.freqs_cis]

        # Gather frequencies for each axis and concatenate
        result = []
        for i in range(len(self.axes_dims)):
            index = ids[:, i]
            result.append(self.freqs_cis[i][index])

        return torch.cat(result, dim=-1)


def apply_rotary_emb(
    x_in: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary position embeddings to input tensor.

    Args:
        x_in: Input tensor of shape (..., seq_len, heads, head_dim)
        freqs_cis: Complex frequencies of shape (seq_len, head_dim/2)

    Returns:
        Tensor with RoPE applied, same shape as input
    """
    # View as complex numbers: reshape last dim from d to (d/2, 2) then view as complex
    x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))

    # Add head dimension to freqs_cis: (seq_len, head_dim/2) -> (seq_len, 1, head_dim/2)
    freqs_cis = freqs_cis.unsqueeze(-2)

    # Multiply and convert back to real
    x_out = torch.view_as_real(x * freqs_cis).flatten(-2)

    return x_out.type_as(x_in)
