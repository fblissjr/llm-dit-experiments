"""
FLUX.2 Transformer with Double-Stream → Single-Stream Architecture.

Last Updated: 2026-01-23

Implements the FLUX.2 Klein diffusion transformer with:
- Double-stream blocks: Separate img/txt streams with joint attention
- Single-stream blocks: Merged sequence with unified attention
- Shared modulation: Computed once, used across all blocks

Key Differences from LTX-2:
- Double→Single stream instead of unified blocks throughout
- Modulation computed once at model level (not per-block)
- 4D RoPE (t, h, w, l) instead of 3D (t, h, w)
- Joint attention in double-stream (img and txt see each other)

Ported from: coderef/flux2/src/flux2/model.py

Usage:
    from llm_dit.models.flux2.transformer import Flux2Transformer
    from llm_dit.models.flux2.constants import Klein9BParams

    model = Flux2Transformer(Klein9BParams())
    output = model(
        x=latents,          # [B, seq_len, 128]
        x_ids=img_ids,      # [B, seq_len, 4]
        timesteps=t_vec,    # [B]
        ctx=txt_embeds,     # [B, txt_len, context_dim]
        ctx_ids=txt_ids,    # [B, txt_len, 4]
        guidance=None,      # [B] (optional for non-distilled)
    )
"""

import logging
import math
from contextlib import contextmanager

import torch
from einops import rearrange
from torch import Tensor, nn

from llm_dit.models.flux2.rope import EmbedND, apply_rope
from llm_dit.models.flux2.constants import Klein9BParams, Klein4BParams, Flux2Params
from llm_dit.utils.attention import attention_forward

logger = logging.getLogger(__name__)

try:
    import psutil
    _psutil = psutil  # Bind to a variable for type checker
    PSUTIL_AVAILABLE = True
except ImportError:
    _psutil = None
    PSUTIL_AVAILABLE = False


def _format_memory_gb(bytes_val: int | float) -> str:
    """Format memory value in GB with 2 decimal places."""
    return f"{bytes_val / 1e9:.2f}GB"


def _log_memory_state(prefix: str = "", device: torch.device | str | None = None) -> None:
    """Log current GPU and CPU memory state."""
    if not logger.isEnabledFor(logging.DEBUG):
        return

    msg_parts = [f"[FLUX2:Transformer:{prefix}]" if prefix else "[FLUX2:Transformer]"]

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        msg_parts.append(f"GPU allocated: {_format_memory_gb(allocated)}")
        msg_parts.append(f"reserved: {_format_memory_gb(reserved)}")

    if PSUTIL_AVAILABLE and _psutil is not None:
        process = _psutil.Process()
        mem_info = process.memory_info()
        msg_parts.append(f"CPU RSS: {_format_memory_gb(mem_info.rss)}")

    logger.debug(" → ".join(msg_parts))


def timestep_embedding(t: Tensor, dim: int, max_period: int = 10000, time_factor: float = 1000.0) -> Tensor:
    """
    Create sinusoidal timestep embeddings.

    Args:
        t: 1-D tensor of N timesteps (fractional values in [0, 1])
        dim: Dimension of the output embeddings
        max_period: Controls minimum frequency
        time_factor: Scaling factor for timesteps

    Returns:
        [N, dim] tensor of positional embeddings
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, device=t.device, dtype=torch.float32) / half
    )

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

    if torch.is_floating_point(t):
        embedding = embedding.to(t)

    return embedding


# =============================================================================
# KV-Cache Attention and Modulation Blending Helpers
# =============================================================================


def causal_attn_fn(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    num_txt_tokens: int,
    num_ref_tokens: int,
    kv_cache: dict | None = None,
) -> Tensor:
    """Causal attention where reference tokens only attend to themselves.

    Without cache: layout is [txt, ref, img]. txt+img attend to all, ref self-attends.
    With cache: layout is [txt, img]. Cached ref K/V injected into attention.

    Args:
        q: Query [B, H, L, D] (after RoPE)
        k: Key [B, H, L, D] (after RoPE)
        v: Value [B, H, L, D]
        num_txt_tokens: Number of text tokens
        num_ref_tokens: Number of reference tokens
        kv_cache: Optional dict with "k_ref" and "v_ref" from step 0

    Returns:
        Attention output [B, L, H*D]
    """
    if kv_cache is not None:
        # Cached path: input layout is [txt, img], inject cached ref K/V
        k_ref = kv_cache["k_ref"]
        v_ref = kv_cache["v_ref"]

        q_txt = q[:, :, :num_txt_tokens, :]
        q_img = q[:, :, num_txt_tokens:, :]
        k_txt = k[:, :, :num_txt_tokens, :]
        v_txt = v[:, :, :num_txt_tokens, :]
        k_img = k[:, :, num_txt_tokens:, :]
        v_img = v[:, :, num_txt_tokens:, :]

        # txt+img attend to all (txt + cached ref + img)
        q_txt_img = torch.cat([q_txt, q_img], dim=2)
        k_all = torch.cat([k_txt, k_ref, k_img], dim=2)
        v_all = torch.cat([v_txt, v_ref, v_img], dim=2)
        out = attention_forward(q_txt_img, k_all, v_all, is_causal=False)
    else:
        # Extract path: input layout is [txt, ref, img]
        ref_start = num_txt_tokens
        ref_end = num_txt_tokens + num_ref_tokens

        q_txt = q[:, :, :ref_start, :]
        q_ref = q[:, :, ref_start:ref_end, :]
        q_img = q[:, :, ref_end:, :]
        k_txt = k[:, :, :ref_start, :]
        v_txt = v[:, :, :ref_start, :]
        k_ref = k[:, :, ref_start:ref_end, :]
        v_ref = v[:, :, ref_start:ref_end, :]
        k_img = k[:, :, ref_end:, :]
        v_img = v[:, :, ref_end:, :]

        # txt+img attend to all keys
        q_txt_img = torch.cat([q_txt, q_img], dim=2)
        k_all = torch.cat([k_txt, k_ref, k_img], dim=2)
        v_all = torch.cat([v_txt, v_ref, v_img], dim=2)
        attn_txt_img = attention_forward(q_txt_img, k_all, v_all, is_causal=False)
        attn_txt = attn_txt_img[:, :, :ref_start, :]
        attn_img = attn_txt_img[:, :, ref_start:, :]

        # ref only attends to itself
        attn_ref = attention_forward(q_ref, k_ref, v_ref, is_causal=False)

        out = torch.cat([attn_txt, attn_ref, attn_img], dim=2)

    return rearrange(out, "b h n d -> b n (h d)")


def _blend_mod_triple(
    img_m: tuple[Tensor, ...],
    ref_m: tuple[Tensor, ...],
    num_ref: int,
    seq_len: int,
) -> tuple[Tensor, ...]:
    """Blend a (shift, scale, gate) triple: first num_ref positions get ref_m, rest get img_m."""
    blended = []
    for im, rm in zip(img_m, ref_m):
        if im.ndim == 2:
            im = im[:, None, :]
            rm = rm[:, None, :]
        B = im.shape[0]
        blended.append(
            torch.cat(
                [rm.expand(B, num_ref, -1), im.expand(B, seq_len, -1)[:, num_ref:, :]],
                dim=1,
            )
        )
    return tuple(blended)


def _blend_double_mods(
    img_mod: tuple[tuple[Tensor, ...], tuple[Tensor, ...]],
    ref_mod: tuple[tuple[Tensor, ...], tuple[Tensor, ...]],
    num_ref: int,
    seq_len: int,
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
    """Blend double block modulations (mod1, mod2) for [ref, img] layout."""
    img_mod1, img_mod2 = img_mod
    ref_mod1, ref_mod2 = ref_mod
    return (
        _blend_mod_triple(img_mod1, ref_mod1, num_ref, seq_len),
        _blend_mod_triple(img_mod2, ref_mod2, num_ref, seq_len),
    )


def _blend_single_mods(
    single_mod: tuple[Tensor, ...],
    ref_mod: tuple[Tensor, ...],
    num_txt: int,
    num_ref: int,
    seq_len: int,
) -> tuple[Tensor, ...]:
    """Blend single block modulations for [txt, ref, img] layout."""
    blended = []
    for im, rm in zip(single_mod, ref_mod):
        if im.ndim == 2:
            im = im[:, None, :]
            rm = rm[:, None, :]
        B = im.shape[0]
        im_expanded = im.expand(B, seq_len, -1)
        rm_expanded = rm.expand(B, num_ref, -1)
        blended.append(
            torch.cat(
                [im_expanded[:, :num_txt, :], rm_expanded, im_expanded[:, num_txt + num_ref:, :]],
                dim=1,
            )
        )
    return tuple(blended)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    More efficient than LayerNorm as it doesn't require mean computation.

    Note: This uses 'scale' instead of 'weight' to match FLUX.2 checkpoint format.
    Migration to llm_dit.layers.RMSNorm would require loader key mapping (scale->weight).
    See: llm_dit.layers.normalization.RMSNorm for the canonical implementation.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(nn.Module):
    """
    Query-Key normalization using RMSNorm.

    Normalizes Q and K before attention for improved stability.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class SiLUActivation(nn.Module):
    """
    Gated SiLU activation for MLP.

    Splits input in half and applies gated SiLU: SiLU(x1) * x2
    """

    def __init__(self):
        super().__init__()
        self.gate_fn = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return self.gate_fn(x1) * x2


class MLPEmbedder(nn.Module):
    """
    MLP for timestep/guidance embedding projection.

    Two-layer MLP with SiLU activation.
    """

    def __init__(self, in_dim: int, hidden_dim: int, disable_bias: bool = False):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=not disable_bias)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=not disable_bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class Modulation(nn.Module):
    """
    Adaptive Layer Normalization modulation.

    Computes scale/shift/gate parameters from timestep embedding.
    For double=True, outputs 6 values (2 sets of scale/shift/gate for attn + MLP).
    For double=False, outputs 3 values (single set).
    """

    def __init__(self, dim: int, double: bool, disable_bias: bool = False):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=not disable_bias)

    def forward(self, vec: Tensor) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...] | None]:
        """
        Compute modulation parameters.

        Args:
            vec: Timestep embedding [B, D]

        Returns:
            For double=True: ((shift1, scale1, gate1), (shift2, scale2, gate2))
            For double=False: ((shift, scale, gate), None)
        """
        out = self.lin(nn.functional.silu(vec))
        if out.ndim == 2:
            out = out[:, None, :]  # Add sequence dimension
        out = out.chunk(self.multiplier, dim=-1)
        return out[:3], out[3:] if self.is_double else None


class LastLayer(nn.Module):
    """
    Final output layer with adaptive normalization.

    Applies AdaLN then linear projection to output channels.
    """

    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=False),
        )

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        mod = self.adaLN_modulation(vec)
        shift, scale = mod.chunk(2, dim=-1)
        if shift.ndim == 2:
            shift = shift[:, None, :]
            scale = scale[:, None, :]
        x = (1 + scale) * self.norm_final(x) + shift
        x = self.linear(x)
        return x


class SelfAttention(nn.Module):
    """
    Self-attention module with QK-norm.

    Used in both double-stream and single-stream blocks.
    """

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim, bias=False)


class DoubleStreamBlock(nn.Module):
    """
    Double-stream transformer block with joint attention.

    Maintains separate image and text streams but performs joint attention
    where both modalities attend to the concatenated sequence.

    Architecture:
        1. Separate Q for img and txt
        2. Concatenate K and V from both streams
        3. Joint attention (both see everything)
        4. Separate MLP for img and txt
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0, f"{hidden_size=} must be divisible by {num_heads=}"

        self.hidden_size = hidden_size
        self.mlp_mult_factor = 2  # For gated SiLU

        # Image stream
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads)
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim * self.mlp_mult_factor, bias=False),
            SiLUActivation(),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=False),
        )

        # Text stream
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim * self.mlp_mult_factor, bias=False),
            SiLUActivation(),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=False),
        )

    def _prepare_qkv(
        self,
        img: Tensor,
        txt: Tensor,
        pe: Tensor,
        pe_ctx: Tensor,
        mod_img: tuple[tuple[Tensor, ...], tuple[Tensor, ...]],
        mod_txt: tuple[tuple[Tensor, ...], tuple[Tensor, ...]],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, int, tuple]:
        """Shared QKV preparation for all forward paths.

        Returns:
            (q, k, v, pe_full, num_txt_tokens, mods) where mods is a flat
            tuple of gate/shift/scale tensors for _apply_residuals.
        """
        img_mod1, img_mod2 = mod_img
        txt_mod1, txt_mod2 = mod_txt

        img_mod1_shift, img_mod1_scale, img_mod1_gate = img_mod1
        img_mod2_shift, img_mod2_scale, img_mod2_gate = img_mod2
        txt_mod1_shift, txt_mod1_scale, txt_mod1_gate = txt_mod1
        txt_mod2_shift, txt_mod2_scale, txt_mod2_gate = txt_mod2

        # Prepare image for attention (modulated normalization)
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1_scale) * img_modulated + img_mod1_shift

        # Get image Q, K, V
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # Prepare text for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1_scale) * txt_modulated + txt_mod1_shift

        # Get text Q, K, V
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # Joint attention: concatenate K, V from both streams
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        num_txt_tokens = txt_q.shape[2]
        pe_full = torch.cat((pe_ctx, pe), dim=2)

        mods = (
            img_mod1_gate, img_mod2_shift, img_mod2_scale, img_mod2_gate,
            txt_mod1_gate, txt_mod2_shift, txt_mod2_scale, txt_mod2_gate,
        )

        return q, k, v, pe_full, num_txt_tokens, mods

    def _apply_residuals(
        self,
        img: Tensor,
        txt: Tensor,
        img_attn: Tensor,
        txt_attn: Tensor,
        mods: tuple,
    ) -> tuple[Tensor, Tensor]:
        """Shared residual connections for all forward paths."""
        (
            img_mod1_gate, img_mod2_shift, img_mod2_scale, img_mod2_gate,
            txt_mod1_gate, txt_mod2_shift, txt_mod2_scale, txt_mod2_gate,
        ) = mods

        img = img + img_mod1_gate * self.img_attn.proj(img_attn)
        img = img + img_mod2_gate * self.img_mlp(
            (1 + img_mod2_scale) * (self.img_norm2(img)) + img_mod2_shift
        )

        txt = txt + txt_mod1_gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2_gate * self.txt_mlp(
            (1 + txt_mod2_scale) * (self.txt_norm2(txt)) + txt_mod2_shift
        )
        return img, txt

    def forward_kv_extract(
        self,
        img: Tensor,
        txt: Tensor,
        pe: Tensor,
        pe_ctx: Tensor,
        mod_img: tuple[tuple[Tensor, ...], tuple[Tensor, ...]],
        mod_txt: tuple[tuple[Tensor, ...], tuple[Tensor, ...]],
        num_ref_tokens: int,
    ) -> tuple[Tensor, Tensor, dict]:
        """Forward with causal attention. img has layout [ref, img]. Extracts ref KV cache."""
        q, k, v, pe_full, num_txt_tokens, mods = self._prepare_qkv(
            img, txt, pe, pe_ctx, mod_img, mod_txt,
        )
        q, k = apply_rope(q, k, pe_full)

        ref_start = num_txt_tokens
        ref_end = num_txt_tokens + num_ref_tokens
        cache = {
            "k_ref": k[:, :, ref_start:ref_end, :].clone(),
            "v_ref": v[:, :, ref_start:ref_end, :].clone(),
        }

        attn = causal_attn_fn(q, k, v, num_txt_tokens, num_ref_tokens)
        txt_attn, img_attn = attn[:, :num_txt_tokens], attn[:, num_txt_tokens:]
        img, txt = self._apply_residuals(img, txt, img_attn, txt_attn, mods)
        return img, txt, cache

    def forward_kv_cached(
        self,
        img: Tensor,
        txt: Tensor,
        pe: Tensor,
        pe_ctx: Tensor,
        mod_img: tuple[tuple[Tensor, ...], tuple[Tensor, ...]],
        mod_txt: tuple[tuple[Tensor, ...], tuple[Tensor, ...]],
        kv_cache: dict,
    ) -> tuple[Tensor, Tensor]:
        """Forward using cached ref KV. img has layout [img] only (no ref)."""
        q, k, v, pe_full, num_txt_tokens, mods = self._prepare_qkv(
            img, txt, pe, pe_ctx, mod_img, mod_txt,
        )
        q, k = apply_rope(q, k, pe_full)

        num_ref_tokens = kv_cache["k_ref"].shape[2]
        attn = causal_attn_fn(q, k, v, num_txt_tokens, num_ref_tokens, kv_cache)
        txt_attn, img_attn = attn[:, :num_txt_tokens], attn[:, num_txt_tokens:]
        return self._apply_residuals(img, txt, img_attn, txt_attn, mods)


class SingleStreamBlock(nn.Module):
    """
    Single-stream transformer block with unified attention.

    After double-stream processing, img and txt are concatenated and
    processed through unified self-attention blocks.
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()

        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = head_dim**-0.5
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp_mult_factor = 2  # For gated SiLU

        # Combined Q, K, V projection + MLP input
        self.linear1 = nn.Linear(
            hidden_size,
            hidden_size * 3 + self.mlp_hidden_dim * self.mlp_mult_factor,
            bias=False,
        )

        # Combined attention output + MLP output projection
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size, bias=False)

        self.norm = QKNorm(head_dim)
        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_act = SiLUActivation()

    def _qkv(self, x: Tensor, mod: tuple[Tensor, ...]) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Shared QKV + MLP split for all forward paths.

        Returns:
            (q, k, v, mlp, mod_gate)
        """
        mod_shift, mod_scale, mod_gate = mod

        x_mod = (1 + mod_scale) * self.pre_norm(x) + mod_shift

        qkv, mlp = torch.split(
            self.linear1(x_mod),
            [3 * self.hidden_size, self.mlp_hidden_dim * self.mlp_mult_factor],
            dim=-1,
        )

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        return q, k, v, mlp, mod_gate

    def _out(self, x: Tensor, attn: Tensor, mlp: Tensor, mod_gate: Tensor) -> Tensor:
        """Shared output projection for all forward paths."""
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + mod_gate * output

    def forward_kv_extract(
        self,
        x: Tensor,
        pe: Tensor,
        mod: tuple[Tensor, ...],
        num_txt_tokens: int,
        num_ref_tokens: int,
    ) -> tuple[Tensor, dict]:
        """Forward with causal attention. Extracts and returns ref KV cache."""
        q, k, v, mlp, mod_gate = self._qkv(x, mod)
        q, k = apply_rope(q, k, pe)

        ref_start = num_txt_tokens
        ref_end = num_txt_tokens + num_ref_tokens
        cache = {
            "k_ref": k[:, :, ref_start:ref_end, :].clone(),
            "v_ref": v[:, :, ref_start:ref_end, :].clone(),
        }

        attn = causal_attn_fn(q, k, v, num_txt_tokens, num_ref_tokens)
        return self._out(x, attn, mlp, mod_gate), cache

    def forward_kv_cached(
        self,
        x: Tensor,
        pe: Tensor,
        mod: tuple[Tensor, ...],
        num_txt_tokens: int,
        kv_cache: dict,
    ) -> Tensor:
        """Forward using cached ref KV. Input x has layout [txt, img] (no ref)."""
        q, k, v, mlp, mod_gate = self._qkv(x, mod)
        q, k = apply_rope(q, k, pe)
        num_ref_tokens = kv_cache["k_ref"].shape[2]
        attn = causal_attn_fn(q, k, v, num_txt_tokens, num_ref_tokens, kv_cache)
        return self._out(x, attn, mlp, mod_gate)


class Flux2Transformer(nn.Module):
    """
    FLUX.2 Diffusion Transformer.

    Implements the double-stream → single-stream architecture:
    1. Project inputs to hidden dimension
    2. Compute modulation once from timestep
    3. Process through double-stream blocks (joint img/txt attention)
    4. Concatenate img and txt
    5. Process through single-stream blocks (unified attention)
    6. Project back to output dimension

    Supports block-by-block offloading for memory-constrained GPUs:
        model.enable_block_offload(device="cuda", offload_device="cpu")

    Args:
        params: Model configuration (Klein9BParams, Klein4BParams, or Flux2Params)
    """

    def __init__(self, params: Klein9BParams | Klein4BParams | Flux2Params):
        super().__init__()

        # Block offloading state
        self._block_offload_enabled = False
        self._compute_device: torch.device | None = None
        self._offload_device: torch.device | None = None

        self.in_channels = params.in_channels
        self.out_channels = params.in_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")

        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads

        # Positional embedding (4D RoPE)
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)

        # Input projections
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=False)
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size, bias=False)

        # Timestep embedding
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size, disable_bias=True)

        # Optional guidance embedding (for non-distilled models)
        self.use_guidance_embed = params.use_guidance_embed
        if self.use_guidance_embed:
            self.guidance_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size, disable_bias=True)

        # Double-stream blocks
        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                )
                for _ in range(params.depth)
            ]
        )

        # Single-stream blocks
        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                )
                for _ in range(params.depth_single_blocks)
            ]
        )

        # Shared modulation modules (computed ONCE, shared across all blocks)
        self.double_stream_modulation_img = Modulation(
            self.hidden_size,
            double=True,
            disable_bias=True,
        )
        self.double_stream_modulation_txt = Modulation(
            self.hidden_size,
            double=True,
            disable_bias=True,
        )
        self.single_stream_modulation = Modulation(
            self.hidden_size,
            double=False,
            disable_bias=True,
        )

        # Final output layer
        self.final_layer = LastLayer(
            self.hidden_size,
            self.out_channels,
        )

        # Store config for inspection
        self._params = params

    def enable_block_offload(
        self,
        device: str | torch.device = "cuda",
        offload_device: str | torch.device = "cpu",
    ) -> "Flux2Transformer":
        """
        Enable block-by-block offloading for memory-constrained GPUs.

        Moves blocks to CPU and transfers them one at a time during forward pass.
        Keeps small layers (embeddings, modulation, final) on GPU.

        Args:
            device: GPU device for computation
            offload_device: Device to offload blocks to (usually "cpu")

        Returns:
            self for method chaining
        """
        logger.debug("[FLUX2:Transformer] Enabling block offload")
        _log_memory_state("Before enable_block_offload", device)

        self._block_offload_enabled = True
        self._compute_device = torch.device(device)
        self._offload_device = torch.device(offload_device)

        # Keep small layers on GPU (embeddings, modulation, final)
        logger.debug(f"[FLUX2:Transformer] Moving small layers to {self._compute_device}")
        small_layers = [
            ("img_in", self.img_in),
            ("txt_in", self.txt_in),
            ("time_in", self.time_in),
            ("pe_embedder", self.pe_embedder),
            ("double_stream_modulation_img", self.double_stream_modulation_img),
            ("double_stream_modulation_txt", self.double_stream_modulation_txt),
            ("single_stream_modulation", self.single_stream_modulation),
            ("final_layer", self.final_layer),
        ]

        small_layer_memory = 0
        for name, layer in small_layers:
            layer_params = sum(p.numel() * p.element_size() for p in layer.parameters())
            small_layer_memory += layer_params
            layer.to(self._compute_device)
            if name in ["img_in", "txt_in", "time_in"]:  # Log a few examples
                logger.debug(f"[FLUX2:Transformer]   {name}: {_format_memory_gb(layer_params)}")

        if self.use_guidance_embed:
            self.guidance_in.to(self._compute_device)
            guidance_params = sum(p.numel() * p.element_size() for p in self.guidance_in.parameters())
            small_layer_memory += guidance_params

        logger.debug(f"[FLUX2:Transformer] Total small layers on GPU: {_format_memory_gb(small_layer_memory)}")
        _log_memory_state("After moving small layers to GPU", self._compute_device)

        # Move all blocks to offload device (CPU)
        logger.debug(
            f"[FLUX2:Transformer] Moving {len(self.double_blocks)} double_blocks + "
            f"{len(self.single_blocks)} single_blocks to {self._offload_device}"
        )

        double_block_memory = 0
        for i, block in enumerate(self.double_blocks):
            block_params = sum(p.numel() * p.element_size() for p in block.parameters())
            double_block_memory += block_params
            block.to(self._offload_device)
            if i == 0:  # Log first block as example
                logger.debug(f"[FLUX2:Transformer]   double_block.0: {_format_memory_gb(block_params)}")

        single_block_memory = 0
        for i, block in enumerate(self.single_blocks):
            block_params = sum(p.numel() * p.element_size() for p in block.parameters())
            single_block_memory += block_params
            block.to(self._offload_device)
            if i == 0:  # Log first block as example
                logger.debug(f"[FLUX2:Transformer]   single_block.0: {_format_memory_gb(block_params)}")

        total_block_memory = double_block_memory + single_block_memory
        logger.debug(
            f"[FLUX2:Transformer] Total blocks on {self._offload_device}: "
            f"{_format_memory_gb(total_block_memory)} "
            f"(double: {_format_memory_gb(double_block_memory)}, "
            f"single: {_format_memory_gb(single_block_memory)})"
        )
        _log_memory_state("After enable_block_offload", self._compute_device)

        return self

    def disable_block_offload(self, device: str | torch.device = "cuda") -> "Flux2Transformer":
        """
        Disable block offloading and move entire model to device.

        Args:
            device: Target device for entire model

        Returns:
            self for method chaining
        """
        self._block_offload_enabled = False
        self._compute_device = None
        self._offload_device = None
        return self.to(device)

    def _move_block_to_device(self, block: nn.Module, device: torch.device, block_name: str = "") -> None:
        """Move a block to the specified device efficiently."""
        if logger.isEnabledFor(logging.DEBUG):
            block_params = sum(p.numel() * p.element_size() for p in block.parameters())
            logger.debug(
                f"[FLUX2:Transformer] Moving {block_name} to {device} "
                f"({_format_memory_gb(block_params)})"
            )
        block.to(device, non_blocking=True)

    @contextmanager
    def _offload_block(self, block: nn.Module, name: str):
        """Move block to compute device, yield, move back to offload device."""
        if self._block_offload_enabled:
            self._move_block_to_device(block, self._compute_device, name)
            if self._compute_device.type == "cuda":
                torch.cuda.synchronize()
        try:
            yield
        finally:
            if self._block_offload_enabled:
                self._move_block_to_device(block, self._offload_device, name)

    def forward(
        self,
        x: Tensor,
        x_ids: Tensor,
        timesteps: Tensor,
        ctx: Tensor,
        ctx_ids: Tensor,
        guidance: Tensor | None,
    ) -> Tensor:
        """
        Forward pass of the FLUX.2 transformer.

        Args:
            x: Image latents [B, img_len, in_channels]
            x_ids: Image position IDs [B, img_len, 4]
            timesteps: Diffusion timesteps [B]
            ctx: Text context embeddings [B, txt_len, context_in_dim]
            ctx_ids: Text position IDs [B, txt_len, 4]
            guidance: Optional guidance scale [B] (for non-distilled models)

        Returns:
            Velocity prediction [B, img_len, in_channels]
        """
        # Guard all logging behind is_compiling() to prevent graph breaks during torch.compile.
        # When torch.compile traces this function, is_compiling() returns True and all
        # logging is skipped, allowing clean graph capture with no breaks.
        _tracing = torch.compiler.is_compiling()
        if _tracing and self._block_offload_enabled:
            raise RuntimeError(
                "torch.compile is incompatible with block_offload. "
                "Set block_offload=false when using compile=true."
            )

        if not _tracing:
            logger.debug(
                f"[FLUX2:Transformer:forward] Starting forward pass - "
                f"x: {list(x.shape)}, ctx: {list(ctx.shape)}"
            )
            _log_memory_state("forward:start", self._compute_device if self._block_offload_enabled else x.device)

        num_txt_tokens = ctx.shape[1]

        # Compute timestep embedding
        timestep_emb = timestep_embedding(timesteps, 256)
        vec = self.time_in(timestep_emb)

        # Add guidance embedding if used
        if self.use_guidance_embed:
            if guidance is None:
                raise ValueError("Guidance embedding enabled but guidance not provided")
            guidance_emb = timestep_embedding(guidance, 256)
            vec = vec + self.guidance_in(guidance_emb)

        # Compute modulations ONCE (shared across all blocks)
        double_block_mod_img = self.double_stream_modulation_img(vec)
        double_block_mod_txt = self.double_stream_modulation_txt(vec)
        single_block_mod, _ = self.single_stream_modulation(vec)

        # Project inputs to hidden dimension
        img = self.img_in(x)
        txt = self.txt_in(ctx)

        if not _tracing:
            logger.debug(
                f"[FLUX2:Transformer:forward] After embeddings - "
                f"img: {list(img.shape)}, txt: {list(txt.shape)}"
            )
            _log_memory_state("forward:after_embeddings", img.device)

        # Compute positional embeddings
        pe_x = self.pe_embedder(x_ids)
        pe_ctx = self.pe_embedder(ctx_ids)

        # Double-stream blocks (joint attention)
        if not _tracing:
            logger.debug(f"[FLUX2:Transformer:forward] Processing {len(self.double_blocks)} double blocks")
        for i, block in enumerate(self.double_blocks):
            with self._offload_block(block, f"double_block.{i}"):
                img, txt, _ = block.forward_kv_extract(
                    img, txt, pe_x, pe_ctx,
                    double_block_mod_img, double_block_mod_txt,
                    num_ref_tokens=0,
                )
            if not _tracing and (i == 0 or i == len(self.double_blocks) - 1):
                _log_memory_state(f"forward:double_block.{i}:after", img.device)

        # Concatenate for single-stream processing
        if not _tracing:
            logger.debug("[FLUX2:Transformer:forward] Concatenating for single-stream processing")
        img = torch.cat((txt, img), dim=1)
        pe = torch.cat((pe_ctx, pe_x), dim=2)
        if not _tracing:
            logger.debug(f"[FLUX2:Transformer:forward] Concatenated sequence: {list(img.shape)}")
            _log_memory_state("forward:after_concat", img.device)

        # Single-stream blocks (unified attention)
        if not _tracing:
            logger.debug(f"[FLUX2:Transformer:forward] Processing {len(self.single_blocks)} single blocks")
        for i, block in enumerate(self.single_blocks):
            with self._offload_block(block, f"single_block.{i}"):
                img, _ = block.forward_kv_extract(
                    img, pe, single_block_mod, num_txt_tokens, num_ref_tokens=0,
                )
            if not _tracing and (i == 0 or i == len(self.single_blocks) - 1):
                _log_memory_state(f"forward:single_block.{i}:after", img.device)

        # Extract image tokens (remove prepended text tokens)
        if not _tracing:
            logger.debug("[FLUX2:Transformer:forward] Extracting image tokens")
        img = img[:, num_txt_tokens:, ...]

        # Final output projection
        if not _tracing:
            logger.debug("[FLUX2:Transformer:forward] Final output projection")
        img = self.final_layer(img, vec)
        if not _tracing:
            _log_memory_state("forward:end", img.device)

        return img

    def forward_kv_extract(
        self,
        x: Tensor,
        x_ids: Tensor,
        timesteps: Tensor,
        ctx: Tensor,
        ctx_ids: Tensor,
        guidance: Tensor | None,
        x_seq_concat: Tensor,
        x_seq_concat_ids: Tensor,
        ref_fixed_timestep: float = 0.0,
    ) -> tuple[Tensor, dict]:
        """First denoising step with reference tokens.

        Runs full forward pass with reference tokens concatenated and extracts
        KV cache for reference tokens to reuse on subsequent steps.

        Input x layout becomes [ref, img] after concatenation.

        Args:
            x: Image latents [B, img_len, in_channels]
            x_ids: Image position IDs [B, img_len, 4]
            timesteps: Diffusion timesteps [B]
            ctx: Text context [B, txt_len, context_in_dim]
            ctx_ids: Text position IDs [B, txt_len, 4]
            guidance: Optional guidance scale [B]
            x_seq_concat: Reference image tokens [B, ref_len, in_channels]
            x_seq_concat_ids: Reference position IDs [B, ref_len, 4]
            ref_fixed_timestep: Fixed timestep for reference modulation (default 0.0)

        Returns:
            (prediction [B, img_len, in_channels], kv_cache dict)
        """
        num_txt_tokens = ctx.shape[1]
        num_ref_tokens = x_seq_concat.shape[1]

        # Concatenate reference tokens with image tokens: [ref, img]
        x = torch.cat([x_seq_concat, x], dim=1)
        x_ids = torch.cat([x_seq_concat_ids, x_ids], dim=1)

        # Timestep embeddings -- separate for img and ref
        timestep_emb = timestep_embedding(timesteps, 256)
        vec = self.time_in(timestep_emb)
        ref_vec = self.time_in(
            timestep_embedding(torch.full_like(timesteps, ref_fixed_timestep), 256)
        )

        if self.use_guidance_embed:
            if guidance is None:
                raise ValueError("Guidance embedding enabled but guidance not provided")
            guidance_emb = timestep_embedding(guidance, 256)
            vec = vec + self.guidance_in(guidance_emb)
            ref_vec = ref_vec + self.guidance_in(guidance_emb)

        # Modulations
        double_block_mod_img = self.double_stream_modulation_img(vec)
        double_block_mod_txt = self.double_stream_modulation_txt(vec)
        single_block_mod, _ = self.single_stream_modulation(vec)

        ref_double_mod = self.double_stream_modulation_img(ref_vec)
        ref_single_mod, _ = self.single_stream_modulation(ref_vec)

        # Project inputs
        img = self.img_in(x)
        txt = self.txt_in(ctx)

        pe_x = self.pe_embedder(x_ids)
        pe_ctx = self.pe_embedder(ctx_ids)

        # Blend double block modulations: [ref_mod, img_mod]
        L_img = img.shape[1]
        double_block_mod_img = _blend_double_mods(
            double_block_mod_img, ref_double_mod, num_ref_tokens, L_img,
        )

        # Double blocks with KV extraction
        double_block_cache = []
        for i, block in enumerate(self.double_blocks):
            with self._offload_block(block, f"double_block.{i}"):
                img, txt, cache = block.forward_kv_extract(
                    img, txt, pe_x, pe_ctx,
                    double_block_mod_img, double_block_mod_txt,
                    num_ref_tokens,
                )
            double_block_cache.append(cache)

        # Concatenate for single-stream
        img = torch.cat((txt, img), dim=1)
        pe = torch.cat((pe_ctx, pe_x), dim=2)

        # Blend single block modulations: [txt_mod, ref_mod, img_mod]
        L = img.shape[1]
        single_block_mod = _blend_single_mods(
            single_block_mod, ref_single_mod, num_txt_tokens, num_ref_tokens, L,
        )

        # Single blocks with KV extraction
        single_block_cache = []
        for i, block in enumerate(self.single_blocks):
            with self._offload_block(block, f"single_block.{i}"):
                img, cache = block.forward_kv_extract(
                    img, pe, single_block_mod, num_txt_tokens, num_ref_tokens,
                )
            single_block_cache.append(cache)

        # Strip txt + ref tokens, keep only img tokens
        img = img[:, num_txt_tokens + num_ref_tokens:, ...]
        img = self.final_layer(img, vec)

        kv_cache = {
            "double_blocks": double_block_cache,
            "single_blocks": single_block_cache,
            "num_ref_tokens": num_ref_tokens,
        }
        return img, kv_cache

    def forward_kv_cached(
        self,
        x: Tensor,
        x_ids: Tensor,
        timesteps: Tensor,
        ctx: Tensor,
        ctx_ids: Tensor,
        guidance: Tensor | None,
        kv_cache: dict,
    ) -> Tensor:
        """Subsequent denoising steps using cached KV for reference tokens.

        Input x has layout [img] only (no ref tokens).

        Args:
            x: Image latents [B, img_len, in_channels]
            x_ids: Image position IDs [B, img_len, 4]
            timesteps: Diffusion timesteps [B]
            ctx: Text context [B, txt_len, context_in_dim]
            ctx_ids: Text position IDs [B, txt_len, 4]
            guidance: Optional guidance scale [B]
            kv_cache: Cache dict from forward_kv_extract

        Returns:
            Velocity prediction [B, img_len, in_channels]
        """
        num_txt_tokens = ctx.shape[1]

        timestep_emb = timestep_embedding(timesteps, 256)
        vec = self.time_in(timestep_emb)

        if self.use_guidance_embed:
            if guidance is None:
                raise ValueError("Guidance embedding enabled but guidance not provided")
            guidance_emb = timestep_embedding(guidance, 256)
            vec = vec + self.guidance_in(guidance_emb)

        double_block_mod_img = self.double_stream_modulation_img(vec)
        double_block_mod_txt = self.double_stream_modulation_txt(vec)
        single_block_mod, _ = self.single_stream_modulation(vec)

        img = self.img_in(x)
        txt = self.txt_in(ctx)

        pe_x = self.pe_embedder(x_ids)
        pe_ctx = self.pe_embedder(ctx_ids)

        # Double blocks with cached KV
        for i, block in enumerate(self.double_blocks):
            with self._offload_block(block, f"double_block.{i}"):
                img, txt = block.forward_kv_cached(
                    img, txt, pe_x, pe_ctx,
                    double_block_mod_img, double_block_mod_txt,
                    kv_cache["double_blocks"][i],
                )

        # Concatenate for single-stream
        img = torch.cat((txt, img), dim=1)
        pe = torch.cat((pe_ctx, pe_x), dim=2)

        # Single blocks with cached KV
        for i, block in enumerate(self.single_blocks):
            with self._offload_block(block, f"single_block.{i}"):
                img = block.forward_kv_cached(
                    img, pe, single_block_mod, num_txt_tokens,
                    kv_cache["single_blocks"][i],
                )

        # Strip txt tokens (no ref tokens in sequence)
        img = img[:, num_txt_tokens:, ...]
        img = self.final_layer(img, vec)
        return img

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


# Alias for backward compatibility
Flux2 = Flux2Transformer
