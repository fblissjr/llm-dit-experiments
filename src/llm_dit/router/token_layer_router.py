"""Per-token layer router for LTX-2 / Gemma conditioning.

Last Updated: 2026-01-16

This module implements a lightweight router that learns to select optimal
Gemma layers for each token in the prompt. Instead of uniform layer blending
(which wastes compute on low-contribution layers), the router predicts
per-token layer weights based on the token's hidden state.

Architecture:
    token_embed [B, T, 3840] → query_proj [B, T, 64] → dot product with
    layer_keys [49, 64] → softmax → layer_weights [B, T, 49]

Total parameters: ~250K (negligible vs Gemma's 2.6B)

Router Input Selection (empirically configurable):
    The router needs a [B, T, D] input to predict layer weights. Options:
    - "layer_0": Embedding layer (before any transformer blocks)
    - "layer_24": Middle layer
    - "layer_47": High-contribution layer (per projection analysis)
    - "layer_48": Final layer (LM head biased - may not transfer to DiT)
    - "mean": Average across all layers
    - "weighted": Weighted average (requires pre-computed weights)

    Per LTX-2 paper Section 3.2.1: "intermediate representations capture a
    hierarchy of linguistic meaning—from raw phonetics in early layers to
    complex semantics in later ones." Don't assume - test empirically.

Training strategy:
    1. Freeze Gemma + LTX-2 DiT
    2. Train router to maximize SigLIP score on (prompt, generated_image) pairs
    3. Optional: Add sparsity loss to encourage compute-efficient routing

The router can be trained with:
    - REINFORCE (score as reward)
    - Straight-through estimator (differentiable approximation)
    - Gumbel-softmax (differentiable sampling)
"""

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


# Valid router input extraction modes
RouterInputMode = Literal["layer_0", "layer_24", "layer_47", "layer_48", "mean", "weighted"]


def extract_router_input(
    hidden_states: torch.Tensor,
    mode: RouterInputMode = "mean",
    layer_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Extract router input from stacked Gemma hidden states.

    The router needs [B, T, D] but Gemma provides [B, T, D, L] with all 49 layers.
    This function extracts the appropriate representation based on mode.

    Args:
        hidden_states: Stacked hidden states [B, T, D, L] where L=49
        mode: Extraction mode:
            - "layer_0": Embedding layer (index 0, before transformer blocks)
            - "layer_24": Middle layer (index 24)
            - "layer_47": High-contribution layer (index 47)
            - "layer_48": Final layer (index 48, may be LM-biased)
            - "mean": Average across all layers
            - "weighted": Weighted average using layer_weights
        layer_weights: Required for "weighted" mode, shape [L] or [1, 1, 1, L]

    Returns:
        Router input tensor [B, T, D]

    Note:
        Per LTX-2 paper: early layers capture phonetics, late layers capture
        semantics. The optimal choice depends on your task - test empirically.
    """
    if mode == "layer_0":
        return hidden_states[:, :, :, 0]
    elif mode == "layer_24":
        return hidden_states[:, :, :, 24]
    elif mode == "layer_47":
        return hidden_states[:, :, :, 47]
    elif mode == "layer_48":
        return hidden_states[:, :, :, 48]
    elif mode == "mean":
        return hidden_states.mean(dim=-1)
    elif mode == "weighted":
        if layer_weights is None:
            raise ValueError("layer_weights required for 'weighted' mode")
        # Normalize weights
        w = layer_weights.view(1, 1, 1, -1)
        w = w / w.sum()
        return (hidden_states * w).sum(dim=-1)
    else:
        raise ValueError(f"Unknown router input mode: {mode}")


class TokenLayerRouter(nn.Module):
    """Per-token layer router for Gemma → LTX-2 conditioning.

    Maps each token's hidden state to a distribution over 49 Gemma layers,
    enabling dynamic layer selection per token.

    Args:
        hidden_dim: Gemma hidden dimension (default: 3840 for Gemma-2 9B)
        num_layers: Number of Gemma layers (default: 49 for Gemma-2 9B)
        bottleneck_dim: Internal dimension for query/key matching (default: 64)
        temperature: Softmax temperature (lower = sharper selection)
        routing_mode: How to convert scores to weights:
            - "soft": Softmax (default, differentiable)
            - "topk": Keep only top-k layers per token
            - "gumbel": Gumbel-softmax (differentiable sampling)
        top_k: Number of layers to keep in topk mode (default: 8)
        init_uniform: If True, initialize to produce uniform weights

    Example:
        >>> router = TokenLayerRouter()
        >>> # From Gemma hidden states
        >>> hidden_states = torch.randn(2, 128, 3840)  # [B, T, hidden_dim]
        >>> layer_weights = router(hidden_states)  # [B, T, 49]
        >>> # Apply to stacked layer outputs
        >>> layer_outputs = torch.randn(2, 128, 3840, 49)  # [B, T, D, L]
        >>> weighted = layer_outputs * layer_weights.unsqueeze(2)  # [B, T, D, L]
    """

    def __init__(
        self,
        hidden_dim: int = 3840,
        num_layers: int = 49,
        bottleneck_dim: int = 64,
        temperature: float = 1.0,
        routing_mode: Literal["soft", "topk", "gumbel"] = "soft",
        top_k: int = 8,
        init_uniform: bool = False,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bottleneck_dim = bottleneck_dim
        self.temperature = temperature
        self.routing_mode = routing_mode
        self.top_k = top_k

        # Query projection: token embed → query vector
        self.query_proj = nn.Linear(hidden_dim, bottleneck_dim, bias=False)

        # Layer keys: learnable per-layer vectors
        self.layer_keys = nn.Parameter(torch.randn(num_layers, bottleneck_dim))

        # Initialize
        self._init_weights(init_uniform)

    def _init_weights(self, init_uniform: bool = False):
        """Initialize weights.

        If init_uniform=True, initialize to produce approximately uniform
        layer weights (good starting point before training).
        """
        # Xavier init for query projection
        nn.init.xavier_uniform_(self.query_proj.weight)

        if init_uniform:
            # Initialize layer keys to be similar → uniform softmax
            nn.init.constant_(self.layer_keys, 0.0)
        else:
            # Random init, scaled for stability
            nn.init.normal_(self.layer_keys, std=1.0 / math.sqrt(self.bottleneck_dim))

    def forward(
        self,
        token_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute per-token layer weights.

        Args:
            token_embeds: Token hidden states from Gemma [B, T, hidden_dim]
            attention_mask: Optional mask for padding tokens [B, T]

        Returns:
            Layer weights [B, T, num_layers], sums to 1 over layers
        """
        # Project tokens to query space
        # [B, T, hidden_dim] → [B, T, bottleneck_dim]
        queries = self.query_proj(token_embeds)

        # Compute attention scores with layer keys
        # queries: [B, T, D]
        # layer_keys: [L, D]
        # scores: [B, T, L]
        scores = torch.einsum('btd,ld->btl', queries, self.layer_keys)

        # Scale by sqrt(dim) for stable softmax
        scores = scores / math.sqrt(self.bottleneck_dim)

        # Apply temperature
        scores = scores / self.temperature

        # Convert scores to weights based on routing mode
        if self.routing_mode == "soft":
            weights = F.softmax(scores, dim=-1)

        elif self.routing_mode == "topk":
            # Keep top-k, zero others
            topk_values, topk_indices = torch.topk(scores, self.top_k, dim=-1)
            weights = torch.zeros_like(scores)
            weights.scatter_(-1, topk_indices, F.softmax(topk_values, dim=-1))

        elif self.routing_mode == "gumbel":
            # Gumbel-softmax for differentiable sampling
            weights = F.gumbel_softmax(scores, tau=self.temperature, hard=False, dim=-1)

        else:
            raise ValueError(f"Unknown routing_mode: {self.routing_mode}")

        # Mask padding tokens (set to uniform weights so they don't affect output)
        if attention_mask is not None:
            # attention_mask: [B, T] where 1 = valid, 0 = padding
            pad_mask = (attention_mask == 0).unsqueeze(-1)  # [B, T, 1]
            uniform = torch.ones_like(weights) / self.num_layers
            weights = torch.where(pad_mask, uniform, weights)

        return weights

    def get_routing_stats(self, weights: torch.Tensor) -> dict:
        """Compute statistics about routing behavior.

        Useful for understanding what the router learned.

        Args:
            weights: Layer weights from forward() [B, T, L]

        Returns:
            Dict with routing statistics
        """
        with torch.no_grad():
            # Entropy of routing distribution (higher = more uniform)
            entropy = -(weights * (weights + 1e-8).log()).sum(dim=-1).mean()

            # Average weight per layer (which layers are used most)
            mean_per_layer = weights.mean(dim=(0, 1))  # [L]

            # Sparsity: how many layers get >1% weight on average
            sparsity = (weights > 0.01).float().sum(dim=-1).mean()

            # Top layer indices per token
            top_layer = weights.argmax(dim=-1)  # [B, T]
            layer_counts = torch.bincount(top_layer.flatten(), minlength=self.num_layers)

            return {
                "entropy": entropy.item(),
                "mean_per_layer": mean_per_layer.cpu().tolist(),
                "sparsity": sparsity.item(),
                "top_layer_distribution": (layer_counts.float() / layer_counts.sum()).cpu().tolist(),
            }

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, "
            f"num_layers={self.num_layers}, "
            f"bottleneck_dim={self.bottleneck_dim}, "
            f"temperature={self.temperature}, "
            f"routing_mode={self.routing_mode}"
        )


class SparsityLoss(nn.Module):
    """Sparsity loss to encourage compute-efficient routing.

    Encourages the router to use fewer layers per token, reducing compute
    while maintaining quality.

    Args:
        target_sparsity: Target average number of layers used per token
        loss_weight: Weight of sparsity loss in total loss
    """

    def __init__(self, target_sparsity: float = 8.0, loss_weight: float = 0.01):
        super().__init__()
        self.target_sparsity = target_sparsity
        self.loss_weight = loss_weight

    def forward(self, weights: torch.Tensor) -> torch.Tensor:
        """Compute sparsity loss.

        Args:
            weights: Layer weights [B, T, L]

        Returns:
            Sparsity loss (scalar)
        """
        # Effective number of layers (entropy-based)
        # Lower entropy = fewer effective layers
        entropy = -(weights * (weights + 1e-8).log()).sum(dim=-1)  # [B, T]
        effective_layers = entropy.exp()  # Convert to effective number

        # Loss: penalize deviation from target
        loss = (effective_layers.mean() - self.target_sparsity).abs()

        return self.loss_weight * loss
