"""
Wan DiT (Diffusion Transformer) for video generation.

Last Updated: 2026-01-13

Implements the Wan 2.1 DiT architecture for text-to-video generation.
Based on DiffSynth-Engine reference implementation.

Supports:
- Wan 2.1 T2V 1.3B (30 layers, 1536 dim, 12 heads)
- Wan 2.1 T2V 14B (40 layers, 5120 dim, 40 heads)
- Wan 2.1 I2V 14B (with CLIP image conditioning)

Weight key mapping (matched to official Wan weights):
- patch_embedding.weight, bias -> video patch embedding (Conv3d)
- time_embedding.{0,2}.* -> sinusoidal time MLP
- time_projection.1.* -> time projection to 6*dim
- text_embedding.{0,2}.* -> text context MLP
- blocks.N.modulation -> AdaLN modulation (6 params)
- blocks.N.norm3.* -> cross-attention pre-norm
- blocks.N.self_attn.{q,k,v,o}.*, norm_q.*, norm_k.* -> self-attention
- blocks.N.cross_attn.{q,k,v,o}.*, norm_q.*, norm_k.* -> text cross-attention
- blocks.N.ffn.{0,2}.* -> FFN (Linear, GELU, Linear)
- head.head.*, head.modulation -> output projection
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize using rsqrt (matches DiffSynth implementation)."""
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast to float32 for precision, normalize, cast back, THEN apply weight
        # Order matters: DiffSynth casts back BEFORE weight multiplication
        return self._norm(x.float()).to(x.dtype) * self.weight


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply AdaLN modulation: x * (1 + scale) + shift"""
    return x * (1 + scale) + shift


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    """Compute 1D sinusoidal positional embedding."""
    sinusoid = torch.outer(
        position.to(torch.float64),
        torch.pow(
            10000,
            -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(dim // 2)
        ),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0) -> torch.Tensor:
    """Precompute 1D RoPE frequencies as complex numbers."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def precompute_freqs_cis_3d(
    dim: int,
    end: int = 1024,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Precompute 3D RoPE frequencies for video (frame, height, width).

    Formula: t_dim = dim - 2*(dim//3), h_dim = w_dim = dim//3
    This ensures t_dim + h_dim + w_dim = dim exactly.
    """
    t_dim = dim - 2 * (dim // 3)  # Temporal gets slightly more dims
    h_dim = dim // 3
    w_dim = dim // 3
    f_freqs = precompute_freqs_cis(t_dim, end, theta)
    h_freqs = precompute_freqs_cis(h_dim, end, theta)
    w_freqs = precompute_freqs_cis(w_dim, end, theta)
    return f_freqs, h_freqs, w_freqs


def rope_apply(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding to query or key tensor."""
    b, s, n, d = x.shape
    # View as complex: [B, S, N, D] -> [B, S, N, D/2, 2] -> [B, S, N, D/2] complex
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(b, s, n, d // 2, 2))
    # Complex multiply with frequencies
    x_out = torch.view_as_real(x_out * freqs)
    return x_out.to(x.dtype).flatten(3)


class SelfAttention(nn.Module):
    """
    Self-attention with QK normalization and 3D RoPE.

    Weight keys: blocks.N.self_attn.{q,k,v,o}.*, norm_q.*, norm_k.*
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

        # QK normalization with RMSNorm (applied to full dim, then split to heads)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
    ) -> torch.Tensor:
        B, N, _ = x.shape

        # Project and normalize Q, K; project V
        q = self.norm_q(self.q(x))  # [B, N, D]
        k = self.norm_k(self.k(x))
        v = self.v(x)

        # Reshape to heads: [B, N, D] -> [B, N, H, D/H]
        q = rearrange(q, "b s (n d) -> b s n d", n=self.num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=self.num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=self.num_heads)

        # Apply RoPE to Q and K
        q = rope_apply(q, freqs)
        k = rope_apply(k, freqs)

        # Attention with scaled dot product
        # [B, N, H, D] -> [B, H, N, D] for SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).flatten(2)  # [B, H, N, D] -> [B, N, D]

        return self.o(out)


class CrossAttention(nn.Module):
    """
    Cross-attention for text conditioning.

    Weight keys: blocks.N.cross_attn.{q,k,v,o}.*, norm_q.*, norm_k.*

    Note: Cross-attention does NOT use RoPE (only self-attention uses it).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

        # QK normalization
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Cross-attention with optional context mask.

        Args:
            x: Query input [B, N, D]
            context: Key/value context [B, S, D]
            context_mask: Boolean mask [B, S] where True = attend, False = ignore
        """
        B, N, _ = x.shape
        _, S, _ = context.shape

        # Project and normalize
        q = self.norm_q(self.q(x))  # [B, N, D]
        k = self.norm_k(self.k(context))  # [B, S, D]
        v = self.v(context)

        # Reshape to heads
        q = rearrange(q, "b s (n d) -> b s n d", n=self.num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=self.num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=self.num_heads)

        # Attention
        q = q.transpose(1, 2)  # [B, H, N, D]
        k = k.transpose(1, 2)  # [B, H, S, D]
        v = v.transpose(1, 2)  # [B, H, S, D]

        # Prepare attention mask if provided
        # SDPA boolean masks: True = MASK OUT (don't attend), False = attend
        # Our input context_mask: True = attend (real token), False = ignore (padding)
        # So we need to INVERT: ~context_mask
        attn_mask = None
        if context_mask is not None:
            # Invert: True (attend) -> False (don't mask), False (padding) -> True (mask out)
            # Expand to [B, 1, 1, S] for broadcasting across heads and query positions
            attn_mask = (~context_mask).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S]

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).flatten(2)

        return self.o(out)


class DiTBlock(nn.Module):
    """
    Single DiT block with AdaLN modulation.

    Architecture:
    1. Self-attention with AdaLN modulation (shift, scale, gate)
    2. Cross-attention with LayerNorm (no modulation!)
    3. FFN with AdaLN modulation (shift, scale, gate)

    Weight keys:
    - blocks.N.modulation -> [1, 6, D] modulation parameters
    - blocks.N.norm3.* -> cross-attention pre-norm (has params)
    - blocks.N.self_attn.* -> self attention
    - blocks.N.cross_attn.* -> text cross attention
    - blocks.N.ffn.{0,2}.* -> FFN
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        # AdaLN modulation: 6 values (shift, scale, gate for self_attn and ffn)
        # Initialized with small values for stability
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / (dim ** 0.5))

        # Layer norms:
        # - norm1, norm2: no learnable params (elementwise_affine=False)
        # - norm3: has learnable params (for cross-attention)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps, elementwise_affine=True)

        # Attention layers
        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(dim, num_heads, eps)

        # FFN with GELU(tanh)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),  # [0]
            nn.GELU(approximate='tanh'),  # [1] - not saved
            nn.Linear(ffn_dim, dim),  # [2]
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        t_mod: torch.Tensor,
        freqs: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with AdaLN modulation.

        Args:
            x: Hidden states [B, N, D]
            context: Text embeddings [B, S, D] (already projected)
            t_mod: Time modulation [B, 6, D]
            freqs: RoPE frequencies [N, 1, head_dim/2] complex
            context_mask: Boolean mask [B, S] for cross-attention (True = attend)

        Returns:
            Updated hidden states [B, N, D]
        """
        # Compute modulation values from block params + time modulation
        # Cast modulation to match t_mod dtype/device (DiffSynth-Studio pattern)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=1)

        # Self-attention with modulation
        # norm1 -> modulate -> self_attn -> gated residual
        sa_input = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa * self.self_attn(sa_input, freqs)

        # Cross-attention WITHOUT modulation (DiffSynth-Studio does NOT pass mask)
        # norm3 -> cross_attn -> direct residual
        x = x + self.cross_attn(self.norm3(x), context)

        # FFN with modulation
        # norm2 -> modulate -> ffn -> gated residual
        ffn_input = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.ffn(ffn_input)

        return x


class OutputHead(nn.Module):
    """
    Output head with modulation.

    Weight keys: head.head.*, head.modulation

    Note: Receives time_embedding output (not time_projection).
    """

    def __init__(
        self,
        dim: int,
        out_dim: int,
        patch_size: Tuple[int, int, int],
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size

        # 2 modulation params (shift, scale)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / (dim ** 0.5))
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Hidden states [B, N, D]
            t_emb: Time embedding [B, D] from time_embedding (not projection!)

        Returns:
            Output [B, N, out_dim * prod(patch_size)]
        """
        # Combine modulation with time embedding
        # t_emb: [B, D] -> [B, 1, D]
        # modulation: [1, 2, D] -> cast to match t_emb dtype/device
        # Result: [B, 2, D] -> split to 2x [B, 1, D] (keep for broadcasting)
        shift, scale = (
            self.modulation.to(dtype=t_emb.dtype, device=t_emb.device) + t_emb.unsqueeze(1)
        ).chunk(2, dim=1)
        x = self.head(modulate(self.norm(x), shift, scale))
        return x


class WanDiT(nn.Module):
    """
    Wan DiT (Diffusion Transformer) for video generation.

    Base Wan architecture for text-to-video generation.
    For HuMo (audio-conditioned), use HuMoTransformer which extends this.

    Configs:
    - Wan 2.1 T2V 1.3B: num_layers=30, dim=1536, num_heads=12, ffn_dim=8960
    - Wan 2.1 T2V 14B: num_layers=40, dim=5120, num_heads=40, ffn_dim=13824
    - Wan 2.1 I2V 14B: Same as T2V 14B + has_clip_feature=True

    Args:
        dim: Hidden dimension
        num_layers: Number of transformer blocks
        num_heads: Number of attention heads
        ffn_dim: FFN intermediate dimension
        in_dim: Input channels (16 for VAE latents)
        out_dim: Output channels (16)
        text_dim: Text embedding dimension (4096 for UMT5-XXL)
        freq_dim: Sinusoidal embedding dimension (256)
        patch_size: 3D patch size (temporal, height, width)
        eps: LayerNorm epsilon
        has_clip_feature: Whether to use CLIP image conditioning (for I2V)
    """

    def __init__(
        self,
        dim: int = 1536,
        num_layers: int = 30,
        num_heads: int = 12,
        ffn_dim: int = 8960,
        in_dim: int = 16,
        out_dim: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        eps: float = 1e-6,
        has_clip_feature: bool = False,
    ):
        super().__init__()

        self.dim = dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.freq_dim = freq_dim
        self.has_clip_feature = has_clip_feature

        # Patch embedding: [B, C, T, H, W] -> [B, D, T', H', W']
        self.patch_embedding = nn.Conv3d(
            in_dim, dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # Text embedding: projects UMT5-XXL (4096) to model dim
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),  # [0]
            nn.GELU(approximate='tanh'),  # [1]
            nn.Linear(dim, dim),  # [2]
        )

        # Time embedding: sinusoidal -> MLP
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),  # [0]
            nn.SiLU(),  # [1]
            nn.Linear(dim, dim),  # [2]
        )

        # Time projection: projects to 6*dim for all block modulations
        # Note: Sequential with Identity at [0] to match weight keys (time_projection.1.*)
        self.time_projection = nn.Sequential(
            nn.SiLU(),  # [0] - activation, not saved but affects keys
            nn.Linear(dim, dim * 6),  # [1]
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(
                dim=dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                eps=eps,
            )
            for _ in range(num_layers)
        ])

        # Output head
        self.head = OutputHead(dim, out_dim, patch_size, eps)

        # Precompute 3D RoPE frequencies
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        # CLIP image embedding (for I2V)
        if has_clip_feature:
            self.img_emb = nn.Sequential(
                nn.LayerNorm(1280),
                nn.Linear(1280, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
            )

    def patchify(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        """Convert video to patches."""
        x = self.patch_embedding(x)  # [B, C, T, H, W] -> [B, D, T', H', W']
        grid_size = x.shape[2:]  # (T', H', W')
        x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
        return x, grid_size

    def unpatchify(
        self,
        x: torch.Tensor,
        grid_size: Tuple[int, int, int],
    ) -> torch.Tensor:
        """Convert patches back to video."""
        f, h, w = grid_size
        p1, p2, p3 = self.patch_size
        return rearrange(
            x,
            "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
            f=f, h=h, w=w,
            x=p1, y=p2, z=p3,
        )

    def _compute_freqs(
        self,
        f: int,
        h: int,
        w: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute 3D RoPE frequencies for given grid size."""
        # freqs is tuple of (f_freqs, h_freqs, w_freqs), each [max_seq, dim/2]
        freqs = torch.cat(
            [
                self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        )
        # Reshape to [f*h*w, 1, head_dim/2] for broadcasting in attention
        return freqs.reshape(f * h * w, 1, -1).to(device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        clip_feature: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            hidden_states: Video latents [B, C, T, H, W]
            timestep: Diffusion timesteps [B]
            encoder_hidden_states: Text embeddings [B, S, text_dim]
            encoder_attention_mask: Boolean mask [B, S] for cross-attention (True = attend)
            clip_feature: CLIP image features [B, S_clip, 1280] (for I2V)

        Returns:
            Noise prediction [B, C, T, H, W]
        """
        # Time embedding
        t_emb = sinusoidal_embedding_1d(self.freq_dim, timestep)  # [B, freq_dim]
        t_emb = self.time_embedding(t_emb)  # [B, D]
        t_mod = self.time_projection(t_emb).unflatten(1, (6, self.dim))  # [B, 6, D]

        # Text embedding
        context = self.text_embedding(encoder_hidden_states)  # [B, S, D]

        # Handle attention mask for CLIP features (if present)
        context_mask = encoder_attention_mask

        # Add CLIP features if present (for I2V)
        if self.has_clip_feature and clip_feature is not None:
            clip_emb = self.img_emb(clip_feature)  # [B, S_clip, D]
            context = torch.cat([clip_emb, context], dim=1)
            # Prepend True mask for CLIP tokens (always attend to CLIP)
            if context_mask is not None:
                clip_mask = torch.ones(
                    clip_emb.shape[0], clip_emb.shape[1],
                    dtype=torch.bool, device=clip_emb.device
                )
                context_mask = torch.cat([clip_mask, context_mask], dim=1)

        # Patchify video
        x, (f, h, w) = self.patchify(hidden_states)

        # Compute 3D RoPE frequencies
        freqs = self._compute_freqs(f, h, w, x.device)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, context, t_mod, freqs, context_mask)

        # Output head (receives t_emb, not t_mod)
        x = self.head(x, t_emb)

        # Unpatchify to video
        x = self.unpatchify(x, (f, h, w))

        return x

    @classmethod
    def from_config(cls, config_name: str) -> "WanDiT":
        """Create model from predefined config."""
        configs = {
            "wan2.1-t2v-1.3b": {
                "dim": 1536,
                "num_layers": 30,
                "num_heads": 12,
                "ffn_dim": 8960,
                "in_dim": 16,
                "out_dim": 16,
                "text_dim": 4096,
                "freq_dim": 256,
                "patch_size": (1, 2, 2),
                "eps": 1e-6,
                "has_clip_feature": False,
            },
            "wan2.1-t2v-14b": {
                "dim": 5120,
                "num_layers": 40,
                "num_heads": 40,
                "ffn_dim": 13824,
                "in_dim": 16,
                "out_dim": 16,
                "text_dim": 4096,
                "freq_dim": 256,
                "patch_size": (1, 2, 2),
                "eps": 1e-6,
                "has_clip_feature": False,
            },
            "wan2.1-i2v-14b": {
                "dim": 5120,
                "num_layers": 40,
                "num_heads": 40,
                "ffn_dim": 13824,
                "in_dim": 16,
                "out_dim": 16,
                "text_dim": 4096,
                "freq_dim": 256,
                "patch_size": (1, 2, 2),
                "eps": 1e-6,
                "has_clip_feature": True,
            },
        }
        if config_name not in configs:
            raise ValueError(f"Unknown config: {config_name}. Available: {list(configs.keys())}")
        return cls(**configs[config_name])

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        config_name: str = "wan2.1-t2v-1.3b",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "WanDiT":
        """Load model from safetensors checkpoint."""
        from safetensors import safe_open

        model = cls.from_config(config_name)

        # Load weights
        state_dict = {}
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

        # Load state dict
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Missing keys: {missing}")
        if unexpected:
            print(f"Unexpected keys: {unexpected}")

        return model.to(device=device, dtype=dtype)
