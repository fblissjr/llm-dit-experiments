"""
Context Refiner module for Z-Image DiT.

The Context Refiner is a 2-layer transformer that processes text embeddings
WITHOUT timestep modulation before the main DiT blocks. This creates a stable
conditioning signal that doesn't change with denoising timestep.

Architecture:
- 2 transformer blocks
- Hidden dim: 3840
- 30 attention heads (128 dim per head)
- RMSNorm (eps=1e-5)
- Gated SiLU FFN
- Multi-axis RoPE position encoding
- No timestep/AdaLN modulation (key difference from noise refiner)

Reference: DiffSynth-Studio z_image_dit.py
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_dit.utils.attention import attention_forward, get_attention_backend


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    More efficient than LayerNorm as it doesn't compute mean.
    Used throughout the Z-Image DiT architecture.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE).

    Encodes position information directly into attention queries/keys
    through rotation in complex space.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        theta: float = 256.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Precompute frequencies
        self.register_buffer(
            "freqs_cis",
            self._precompute_freqs_cis(dim, max_seq_len, theta),
            persistent=False,
        )

    @staticmethod
    def _precompute_freqs_cis(
        dim: int,
        end: int,
        theta: float = 256.0,
    ) -> torch.Tensor:
        """Precompute the frequency tensor for complex exponentials (cos + i*sin).

        Args:
            dim: Dimension for RoPE (typically head_dim)
            end: Maximum sequence length
            theta: Base for frequency computation

        Returns:
            Complex tensor of shape (end, dim//2) containing cos + i*sin values
        """
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
        timestep = torch.arange(end, dtype=torch.float64)
        freqs = torch.outer(timestep, freqs).float()
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64)
        return freqs_cis

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to queries and keys.

        Args:
            q: Query tensor of shape (batch, seq_len, n_heads, head_dim)
            k: Key tensor of shape (batch, seq_len, n_kv_heads, head_dim)
            position_ids: Optional position indices

        Returns:
            Tuple of (rotated_q, rotated_k) with same shapes as inputs
        """
        seq_len = q.shape[1]

        if position_ids is None:
            freqs = self.freqs_cis[:seq_len]
        else:
            freqs = self.freqs_cis[position_ids]

        # Reshape for broadcasting: (seq_len, dim//2) -> (1, seq_len, 1, dim//2)
        freqs = freqs.unsqueeze(0).unsqueeze(2)

        # Apply rotation
        q_rotated = self._apply_rotary(q, freqs)
        k_rotated = self._apply_rotary(k, freqs)

        return q_rotated, k_rotated

    @staticmethod
    def _apply_rotary(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """Apply rotary embedding to a tensor.

        Converts to complex, multiplies by freqs_cis, converts back to real.
        """
        # x shape: (batch, seq_len, n_heads, head_dim)
        # freqs_cis shape: (1, seq_len, 1, head_dim//2)

        # Reshape x to complex: split last dim in half, treat as real+imag
        x_complex = torch.view_as_complex(
            x.float().reshape(*x.shape[:-1], -1, 2)
        )

        # Multiply by rotation frequencies
        x_rotated = x_complex * freqs_cis.to(x.device)

        # Convert back to real
        x_out = torch.view_as_real(x_rotated).flatten(-2)

        return x_out.type_as(x)


class GatedFeedForward(nn.Module):
    """Gated SiLU Feed-Forward Network.

    Uses the SwiGLU activation pattern:
        output = W2(SiLU(W1(x)) * W3(x))

    This is more expressive than standard MLP and commonly used in
    modern transformers like LLaMA and Qwen.
    """

    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        # Default hidden_dim follows Z-Image pattern: 8/3 * dim
        if hidden_dim is None:
            hidden_dim = int(dim / 3 * 8)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # Gate up
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # Down projection
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # Gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class ContextRefinerAttention(nn.Module):
    """Multi-head self-attention for Context Refiner.

    Features:
    - RMSNorm on queries and keys (qk_norm)
    - Rotary position embeddings
    - Uses the unified attention backend selector
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        qk_norm: bool = True,
        rope_theta: float = 256.0,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.head_dim = dim // n_heads
        self.qk_norm = qk_norm

        # Projections
        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        # QK normalization (applied before RoPE)
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        # Rotary embeddings
        self.rope = RotaryEmbedding(
            dim=self.head_dim,
            max_seq_len=max_seq_len,
            theta=rope_theta,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for self-attention.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            attention_mask: Optional attention mask
            position_ids: Optional position indices for RoPE

        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (batch, seq_len, n_heads, head_dim)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # Apply QK normalization
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Apply rotary embeddings
        q, k = self.rope(q, k, position_ids)

        # Transpose for attention: (batch, n_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Expand KV heads if using GQA (grouped query attention)
        if self.n_kv_heads < self.n_heads:
            repeat_factor = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        # Compute attention using the backend selector
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_output = attention_forward(q, k, v, mask=attention_mask, scale=scale)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        output = self.o_proj(attn_output)

        return output


class ContextRefinerBlock(nn.Module):
    """Single transformer block for Context Refiner.

    Uses the dual RMSNorm pattern:
    - Pre-norm before attention/FFN
    - Post-norm after attention/FFN (before residual add)

    NO timestep modulation (key difference from noise refiner blocks).
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        norm_eps: float = 1e-5,
        qk_norm: bool = True,
        ffn_hidden_dim: Optional[int] = None,
        rope_theta: float = 256.0,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.dim = dim

        # Self-attention
        self.attention = ContextRefinerAttention(
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            qk_norm=qk_norm,
            rope_theta=rope_theta,
            max_seq_len=max_seq_len,
        )

        # Gated FFN
        self.feed_forward = GatedFeedForward(dim, ffn_hidden_dim)

        # Dual normalization layers
        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)  # Pre-attention
        self.attention_norm2 = RMSNorm(dim, eps=norm_eps)  # Post-attention
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)        # Pre-FFN
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)        # Post-FFN

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through transformer block.

        Flow:
        1. Pre-norm -> Attention -> Post-norm -> Residual
        2. Pre-norm -> FFN -> Post-norm -> Residual

        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            attention_mask: Optional attention mask
            position_ids: Optional position indices

        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        # Self-attention with residual
        attn_out = self.attention(
            self.attention_norm1(x),
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        x = x + self.attention_norm2(attn_out)

        # Feed-forward with residual
        ffn_out = self.feed_forward(self.ffn_norm1(x))
        x = x + self.ffn_norm2(ffn_out)

        return x


class ContextRefiner(nn.Module):
    """Context Refiner module for Z-Image DiT.

    Processes text embeddings through 2 transformer layers WITHOUT timestep
    modulation, creating a stable conditioning signal for the main DiT.

    Default architecture (matches Z-Image):
    - 2 layers
    - 3840 hidden dim
    - 30 attention heads
    - RMSNorm with eps=1e-5
    - Gated SiLU FFN
    - RoPE with theta=256.0

    Example usage:

        # Create context refiner
        refiner = ContextRefiner(
            dim=3840,
            n_layers=2,
            n_heads=30,
        )

        # Process text embeddings
        # Input: (batch, seq_len, 3840) - already projected from 2560
        refined = refiner(text_embeddings)
        # Output: (batch, seq_len, 3840)

        # Load from diffusers checkpoint
        refiner = ContextRefiner.from_pretrained("path/to/z-image")
    """

    def __init__(
        self,
        dim: int = 3840,
        n_layers: int = 2,
        n_heads: int = 30,
        n_kv_heads: Optional[int] = None,
        norm_eps: float = 1e-5,
        qk_norm: bool = True,
        ffn_hidden_dim: Optional[int] = None,
        rope_theta: float = 256.0,
        max_seq_len: int = 2048,
    ):
        """Initialize Context Refiner.

        Args:
            dim: Hidden dimension (default: 3840 for Z-Image)
            n_layers: Number of transformer layers (default: 2)
            n_heads: Number of attention heads (default: 30)
            n_kv_heads: Number of KV heads for GQA (default: same as n_heads)
            norm_eps: Epsilon for RMSNorm (default: 1e-5)
            qk_norm: Whether to apply RMSNorm to Q and K (default: True)
            ffn_hidden_dim: Hidden dim for FFN (default: 8/3 * dim)
            rope_theta: Base theta for RoPE (default: 256.0)
            max_seq_len: Maximum sequence length for RoPE (default: 2048)
        """
        super().__init__()

        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads

        # Create transformer blocks
        self.layers = nn.ModuleList([
            ContextRefinerBlock(
                dim=dim,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                norm_eps=norm_eps,
                qk_norm=qk_norm,
                ffn_hidden_dim=ffn_hidden_dim,
                rope_theta=rope_theta,
                max_seq_len=max_seq_len,
            )
            for _ in range(n_layers)
        ])

        # Gradient checkpointing flag
        self._gradient_checkpointing = False

    def enable_gradient_checkpointing(self, enable: bool = True) -> None:
        """Enable or disable gradient checkpointing for memory efficiency."""
        self._gradient_checkpointing = enable

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through all context refiner layers.

        Args:
            x: Text embeddings of shape (batch, seq_len, dim)
            attention_mask: Optional attention mask
            position_ids: Optional position indices for RoPE

        Returns:
            Refined text embeddings of shape (batch, seq_len, dim)
        """
        for layer in self.layers:
            if self._gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer,
                    x,
                    attention_mask,
                    position_ids,
                    use_reentrant=False,
                )
            else:
                x = layer(x, attention_mask, position_ids)

        return x

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: str | torch.device = "cpu",
        torch_dtype: torch.dtype = torch.float32,
    ) -> "ContextRefiner":
        """Load Context Refiner weights from a Z-Image checkpoint.

        Args:
            model_path: Path to Z-Image model directory
            device: Device to load weights to
            torch_dtype: Data type for weights

        Returns:
            ContextRefiner with loaded weights
        """
        import json
        from pathlib import Path
        from safetensors.torch import load_file

        model_path = Path(model_path)

        # Load config to get architecture params
        config_path = model_path / "transformer" / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)

            # Extract relevant params (diffusers format)
            dim = config.get("inner_dim", 3840)
            n_heads = config.get("num_attention_heads", 30)
            n_layers = config.get("num_context_refiner_layers", 2)
        else:
            # Use Z-Image defaults
            dim = 3840
            n_heads = 30
            n_layers = 2

        # Create model
        model = cls(
            dim=dim,
            n_layers=n_layers,
            n_heads=n_heads,
        )

        # Load weights from safetensors
        weight_files = list(model_path.glob("transformer/*.safetensors"))
        if not weight_files:
            weight_files = list(model_path.glob("*.safetensors"))

        if not weight_files:
            raise FileNotFoundError(
                f"No safetensors files found in {model_path}"
            )

        # Load and filter context_refiner weights
        state_dict = {}
        for wf in weight_files:
            weights = load_file(wf, device="cpu")
            for key, value in weights.items():
                if "context_refiner" in key:
                    # Map diffusers weight names to our module structure
                    new_key = cls._map_weight_key(key)
                    if new_key:
                        state_dict[new_key] = value

        if not state_dict:
            raise ValueError(
                "No context_refiner weights found in checkpoint. "
                "Make sure this is a Z-Image model."
            )

        # Load weights
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device=device, dtype=torch_dtype)

        return model

    @staticmethod
    def _map_weight_key(diffusers_key: str) -> Optional[str]:
        """Map diffusers weight key to our module structure.

        Example mappings:
            transformer.context_refiner.0.attention.q_proj.weight
            -> layers.0.attention.q_proj.weight
        """
        if "context_refiner" not in diffusers_key:
            return None

        # Remove prefix up to context_refiner
        parts = diffusers_key.split("context_refiner.")
        if len(parts) < 2:
            return None

        suffix = parts[1]

        # Map to our structure
        return f"layers.{suffix}"

    def get_num_params(self, trainable_only: bool = False) -> int:
        """Get the total number of parameters.

        Args:
            trainable_only: If True, only count trainable parameters

        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        num_params = self.get_num_params() / 1e6
        return (
            f"ContextRefiner(\n"
            f"  dim={self.dim},\n"
            f"  n_layers={self.n_layers},\n"
            f"  n_heads={self.n_heads},\n"
            f"  params={num_params:.2f}M\n"
            f")"
        )
