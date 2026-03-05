"""V2 Feature Extractor for LTX-2.3 (22B) Gemma3 encoder.

Per-token RMSNorm normalization (vs V1's per-batch masked mean/range).
Dual projections: video (188160 -> 4096) and audio (188160 -> 2048).
Rescaling: x * sqrt(target_dim / source_dim) before projection.

Reference: coderef/LTX-2/.../feature_extractor.py (FeatureExtractorV2)
"""

import math

import torch
from torch import nn


def norm_and_concat_per_token_rms(
    encoded_text: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-token RMSNorm normalization for V2 models.

    Args:
        encoded_text: [B, T, D, L] stacked hidden states.
        attention_mask: [B, T] binary mask.

    Returns:
        [B, T, D*L] normalized tensor with padding zeroed out.
    """
    B, T, D, L = encoded_text.shape  # noqa: N806
    variance = torch.mean(encoded_text**2, dim=2, keepdim=True)  # [B, T, 1, L]
    normed = encoded_text * torch.rsqrt(variance + 1e-6)
    normed = normed.reshape(B, T, D * L)
    mask_3d = attention_mask.bool().unsqueeze(-1)  # [B, T, 1]
    return torch.where(mask_3d, normed, torch.zeros_like(normed))


def _rescale_norm(x: torch.Tensor, target_dim: int, source_dim: int) -> torch.Tensor:
    """Rescale normalization: x * sqrt(target_dim / source_dim)."""
    return x * math.sqrt(target_dim / source_dim)


class FeatureExtractorV2(nn.Module):
    """22B: per-token RMS norm -> rescale -> dual aggregate embeds.

    Unlike V1 which uses a single Linear(188160, 3840) with per-batch normalization,
    V2 uses per-token RMSNorm and separate projections for video and audio with
    different output dimensions.

    Args:
        embedding_dim: Source embedding dim (typically 3840 for Gemma3).
        video_dim: Video projection output dim (4096).
        audio_dim: Audio projection output dim (2048), or None for video-only.
        feature_dim: Input feature dim (3840 * 49 = 188160).
        dtype: Parameter dtype.
    """

    def __init__(
        self,
        embedding_dim: int = 3840,
        video_dim: int = 4096,
        audio_dim: int | None = 2048,
        feature_dim: int = 188160,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.video_aggregate_embed = nn.Linear(feature_dim, video_dim, bias=True).to(dtype=dtype)
        self.audio_aggregate_embed: nn.Linear | None = None
        if audio_dim is not None:
            self.audio_aggregate_embed = nn.Linear(feature_dim, audio_dim, bias=True).to(dtype=dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Extract features with per-token RMSNorm and dual projections.

        Args:
            hidden_states: [B, T, D, L] stacked hidden states from all layers.
            attention_mask: [B, T] binary mask.

        Returns:
            Tuple of (video_features, audio_features).
            video_features: [B, T, video_dim]
            audio_features: [B, T, audio_dim] or None if audio_dim was None.
        """
        # Stack if list/tuple input
        if isinstance(hidden_states, (list, tuple)):
            hidden_states = torch.stack(hidden_states, dim=-1)

        normed = norm_and_concat_per_token_rms(hidden_states, attention_mask)
        normed = normed.to(hidden_states.dtype)

        v_dim = self.video_aggregate_embed.out_features
        video = self.video_aggregate_embed(_rescale_norm(normed, v_dim, self.embedding_dim))

        audio = None
        if self.audio_aggregate_embed is not None:
            a_dim = self.audio_aggregate_embed.out_features
            audio = self.audio_aggregate_embed(_rescale_norm(normed, a_dim, self.embedding_dim))

        return video, audio
