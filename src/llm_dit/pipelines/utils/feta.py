"""
FETA (Feature Temporal Attention) Enhancement for LTX-2 Pipeline.

Last Updated: 2026-01-16

Ported from ComfyUI-LTXVideo. FETA enhances temporal consistency in video
generation by boosting cross-frame attention based on measured temporal
coherence scores.

The key insight is that attention patterns between frames indicate temporal
consistency. High cross-frame attention = frames are relating well to each
other. FETA:
1. Computes attention scores Q @ K^T across the frame dimension
2. Masks out self-attention (diagonal elements)
3. Computes mean off-diagonal attention as "temporal coherence score"
4. Multiplicatively scales attention output by this score

This is a non-invasive, zero-parameter enhancement that works by amplifying
naturally coherent temporal relationships.

Reference:
    ComfyUI-LTXVideo by Lightricks
    https://github.com/Lightricks/ComfyUI-LTXVideo

Note:
    FETA requires access to attention internals (Q, K, V tensors) and is
    more invasive to integrate than latent normalization. It typically
    requires patching the transformer's attention mechanism.

Example:
    # Compute FETA score for attention scaling
    scale = compute_feta_score(q, k, num_heads=24, num_frames=33, weight=4.0)
    attention_output = attention_output * scale
"""

from dataclasses import dataclass
from typing import Optional, Callable

import torch
import torch.nn.functional as F


@dataclass
class FETAConfig:
    """Configuration for FETA enhancement."""

    # FETA weight - controls enhancement strength
    # Higher = more temporal consistency boost
    # Typical range: 2.0 - 8.0
    weight: float = 4.0

    # Whether to apply FETA
    enabled: bool = True

    # Minimum scale to prevent complete suppression
    min_scale: float = 1.0

    # Which transformer blocks to apply FETA to
    # None = all blocks, or list of block indices
    block_indices: Optional[list[int]] = None

    # Start step (0-indexed) - skip early noisy steps
    start_step: int = 0

    # End step - stop before final refinement steps
    # -1 = apply until end
    end_step: int = -1


def compute_feta_score(
    q: torch.Tensor,
    k: torch.Tensor,
    num_heads: int,
    num_frames: int,
    weight: float = 4.0,
    min_scale: float = 1.0,
) -> torch.Tensor:
    """
    Compute FETA temporal enhancement score from Q and K tensors.

    Computes attention patterns across frames, masks self-attention,
    and derives a scaling factor based on mean cross-frame attention.

    Args:
        q: Query tensor [B, num_heads, seq_len, head_dim]
        k: Key tensor [B, num_heads, seq_len, head_dim]
        num_heads: Number of attention heads
        num_frames: Number of frames in the video
        weight: FETA weight parameter (higher = stronger enhancement)
        min_scale: Minimum scale factor (prevents suppression)

    Returns:
        Scale factor tensor for multiplying attention output
    """
    # Get dimensions
    B, H, S, D = q.shape
    head_dim = D
    scale = head_dim ** -0.5

    # Compute attention scores
    # [B, H, S, S]
    attn_scores = torch.matmul(q * scale, k.transpose(-2, -1))

    # Apply softmax to get attention probabilities
    attn_probs = F.softmax(attn_scores, dim=-1)

    # We need to compute frame-to-frame attention
    # Reshape to separate frames: [B, H, F, tokens_per_frame, S]
    # For video latents, S typically = F * H_lat * W_lat
    tokens_per_frame = S // num_frames

    if S % num_frames != 0:
        # Cannot evenly divide - return neutral scale
        return torch.ones(1, device=q.device, dtype=q.dtype)

    # Reshape attention to frame-level view
    # Sum attention from each frame to all tokens in other frames
    attn_reshaped = attn_probs.reshape(B, H, num_frames, tokens_per_frame, num_frames, tokens_per_frame)

    # Sum over within-frame tokens to get frame-to-frame attention
    # [B, H, F, F]
    frame_attn = attn_reshaped.sum(dim=(3, 5)) / tokens_per_frame

    # Average over heads and batch
    # [F, F]
    frame_attn_avg = frame_attn.mean(dim=(0, 1))

    # Create mask for off-diagonal (cross-frame attention)
    # We want to measure how much frames attend to OTHER frames
    mask = torch.eye(num_frames, device=q.device, dtype=torch.bool)
    off_diag_attn = frame_attn_avg.masked_fill(mask, 0)

    # Compute mean cross-frame attention
    # Denominator: num_frames * (num_frames - 1) for off-diagonal elements
    num_off_diag = num_frames * (num_frames - 1)
    mean_cross_attn = off_diag_attn.sum() / num_off_diag

    # Compute enhancement factor
    # Higher cross-frame attention -> higher enhancement
    enhance_score = mean_cross_attn * (num_frames + weight)

    # Clamp to minimum scale
    scale_factor = enhance_score.clamp(min=min_scale)

    return scale_factor


def compute_feta_score_simple(
    attention_probs: torch.Tensor,
    num_frames: int,
    weight: float = 4.0,
    min_scale: float = 1.0,
) -> torch.Tensor:
    """
    Simplified FETA score from pre-computed attention probabilities.

    Use this when you already have attention probabilities computed
    and don't need to recompute from Q/K.

    Args:
        attention_probs: Attention probabilities [B, H, S, S] or [B*H, S, S]
        num_frames: Number of frames
        weight: FETA weight
        min_scale: Minimum scale

    Returns:
        Scale factor tensor
    """
    # Handle both 3D and 4D inputs
    if attention_probs.ndim == 3:
        attention_probs = attention_probs.unsqueeze(0)

    B, H, S, _ = attention_probs.shape
    tokens_per_frame = S // num_frames

    if S % num_frames != 0:
        return torch.ones(1, device=attention_probs.device, dtype=attention_probs.dtype)

    # Reshape to frame-level
    attn_reshaped = attention_probs.reshape(
        B, H, num_frames, tokens_per_frame, num_frames, tokens_per_frame
    )
    frame_attn = attn_reshaped.sum(dim=(3, 5)) / tokens_per_frame
    frame_attn_avg = frame_attn.mean(dim=(0, 1))

    # Off-diagonal mean
    mask = torch.eye(num_frames, device=attention_probs.device, dtype=torch.bool)
    off_diag_attn = frame_attn_avg.masked_fill(mask, 0)
    mean_cross_attn = off_diag_attn.sum() / (num_frames * (num_frames - 1))

    # Enhancement
    scale_factor = (mean_cross_attn * (num_frames + weight)).clamp(min=min_scale)

    return scale_factor


class FETAEnhancer:
    """
    FETA enhancement wrapper for transformer attention.

    This class provides a hook-style interface for applying FETA enhancement
    to transformer attention outputs. It can be used to wrap attention
    computation or as a post-attention modifier.

    Example with attention hook:
        feta = FETAEnhancer(num_frames=33, weight=4.0)

        def attention_hook(module, input, output):
            q, k, v = input  # or extract from module
            enhanced = feta.enhance(output, q, k)
            return enhanced

        transformer.register_forward_hook(attention_hook)
    """

    def __init__(
        self,
        num_frames: int,
        weight: float = 4.0,
        min_scale: float = 1.0,
        start_step: int = 0,
        end_step: int = -1,
    ):
        """
        Initialize FETA enhancer.

        Args:
            num_frames: Number of video frames
            weight: FETA weight (2.0-8.0 typical)
            min_scale: Minimum scale factor
            start_step: First step to apply FETA (0-indexed)
            end_step: Last step to apply FETA (-1 = all)
        """
        self.num_frames = num_frames
        self.weight = weight
        self.min_scale = min_scale
        self.start_step = start_step
        self.end_step = end_step
        self.current_step = 0

    def set_step(self, step: int) -> None:
        """Update current diffusion step."""
        self.current_step = step

    def should_apply(self, step: Optional[int] = None) -> bool:
        """Check if FETA should be applied at current/given step."""
        step = step if step is not None else self.current_step

        if step < self.start_step:
            return False
        if self.end_step >= 0 and step > self.end_step:
            return False
        return True

    def enhance(
        self,
        attention_output: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        num_heads: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Enhance attention output with FETA scaling.

        Args:
            attention_output: Output from attention computation
            q: Query tensor
            k: Key tensor
            num_heads: Number of attention heads (inferred if not provided)

        Returns:
            Enhanced attention output
        """
        if not self.should_apply():
            return attention_output

        if num_heads is None:
            # Infer from q shape
            if q.ndim == 4:
                num_heads = q.shape[1]
            else:
                num_heads = 1

        scale = compute_feta_score(
            q, k,
            num_heads=num_heads,
            num_frames=self.num_frames,
            weight=self.weight,
            min_scale=self.min_scale,
        )

        return attention_output * scale

    def enhance_with_probs(
        self,
        attention_output: torch.Tensor,
        attention_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Enhance using pre-computed attention probabilities.

        Args:
            attention_output: Output from attention computation
            attention_probs: Pre-computed attention probabilities

        Returns:
            Enhanced attention output
        """
        if not self.should_apply():
            return attention_output

        scale = compute_feta_score_simple(
            attention_probs,
            num_frames=self.num_frames,
            weight=self.weight,
            min_scale=self.min_scale,
        )

        return attention_output * scale


def create_feta_attention_patch(
    original_attention: Callable,
    num_frames: int,
    weight: float = 4.0,
    min_scale: float = 1.0,
) -> Callable:
    """
    Create a patched attention function with FETA enhancement.

    This creates a wrapper around an existing attention function that
    applies FETA enhancement to the output.

    Note: The exact signature depends on the attention implementation.
    This is a template - adjust for your specific transformer.

    Args:
        original_attention: Original attention function
        num_frames: Number of video frames
        weight: FETA weight
        min_scale: Minimum scale

    Returns:
        Patched attention function with FETA
    """

    def patched_attention(q, k, v, *args, **kwargs):
        # Call original attention
        attn_output, attn_probs = original_attention(q, k, v, *args, return_attn_probs=True, **kwargs)

        # Compute FETA scale
        scale = compute_feta_score_simple(
            attn_probs,
            num_frames=num_frames,
            weight=weight,
            min_scale=min_scale,
        )

        # Apply enhancement
        return attn_output * scale

    return patched_attention
