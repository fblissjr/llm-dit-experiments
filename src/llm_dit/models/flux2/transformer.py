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

import torch
from einops import rearrange
from torch import Tensor, nn

from llm_dit.models.flux2.rope import EmbedND, attention
from llm_dit.models.flux2.constants import Klein9BParams, Klein4BParams, Flux2Params

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

    def forward(
        self,
        img: Tensor,
        txt: Tensor,
        pe: Tensor,
        pe_ctx: Tensor,
        mod_img: tuple[tuple[Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor]],
        mod_txt: tuple[tuple[Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor]],
    ) -> tuple[Tensor, Tensor]:
        """
        Forward pass with joint attention.

        Args:
            img: Image tokens [B, img_len, D]
            txt: Text tokens [B, txt_len, D]
            pe: Image positional embeddings
            pe_ctx: Text positional embeddings
            mod_img: Image modulation ((shift1, scale1, gate1), (shift2, scale2, gate2))
            mod_txt: Text modulation ((shift1, scale1, gate1), (shift2, scale2, gate2))

        Returns:
            Updated (img, txt) tensors
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
        q = torch.cat((txt_q, img_q), dim=2)  # [B, H, txt_len + img_len, D]
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        # Concatenate positional embeddings
        pe = torch.cat((pe_ctx, pe), dim=2)

        # Compute attention
        attn = attention(q, k, v, pe)

        # Split attention output
        txt_attn, img_attn = attn[:, : txt_q.shape[2]], attn[:, txt_q.shape[2] :]

        # Update image with residual
        img = img + img_mod1_gate * self.img_attn.proj(img_attn)
        img = img + img_mod2_gate * self.img_mlp(
            (1 + img_mod2_scale) * (self.img_norm2(img)) + img_mod2_shift
        )

        # Update text with residual
        txt = txt + txt_mod1_gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2_gate * self.txt_mlp(
            (1 + txt_mod2_scale) * (self.txt_norm2(txt)) + txt_mod2_shift
        )

        return img, txt


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

    def forward(
        self,
        x: Tensor,
        pe: Tensor,
        mod: tuple[Tensor, Tensor, Tensor],
    ) -> Tensor:
        """
        Forward pass with unified attention.

        Args:
            x: Combined (txt, img) sequence [B, txt_len + img_len, D]
            pe: Combined positional embeddings
            mod: Modulation tuple (shift, scale, gate)

        Returns:
            Updated sequence tensor
        """
        mod_shift, mod_scale, mod_gate = mod

        # Apply modulated pre-norm
        x_mod = (1 + mod_scale) * self.pre_norm(x) + mod_shift

        # Project to Q, K, V + MLP input in one operation
        qkv, mlp = torch.split(
            self.linear1(x_mod),
            [3 * self.hidden_size, self.mlp_hidden_dim * self.mlp_mult_factor],
            dim=-1,
        )

        # Reshape Q, K, V
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)

        # Compute attention with RoPE
        attn = attention(q, k, v, pe)

        # Combine attention output with activated MLP
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))

        return x + mod_gate * output


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
        if self._block_offload_enabled and self._compute_device and self._offload_device:
            # Block-by-block offloading mode (incompatible with torch.compile)
            if not _tracing:
                logger.debug("[FLUX2:Transformer:forward] Using block offload mode")
            for i, block in enumerate(self.double_blocks):
                # Track peak memory during block processing
                peak_before = torch.cuda.max_memory_allocated(self._compute_device) if torch.cuda.is_available() else 0

                # Move block to GPU
                self._move_block_to_device(block, self._compute_device, f"double_block.{i}")
                if self._compute_device.type == "cuda":
                    torch.cuda.synchronize()
                if not _tracing:
                    _log_memory_state(f"forward:double_block.{i}:after_load", self._compute_device)

                img, txt = block(
                    img,
                    txt,
                    pe_x,
                    pe_ctx,
                    double_block_mod_img,
                    double_block_mod_txt,
                )

                # Log peak memory during forward pass
                if not _tracing:
                    peak_after = torch.cuda.max_memory_allocated(self._compute_device) if torch.cuda.is_available() else 0
                    peak_delta = peak_after - peak_before
                    if logger.isEnabledFor(logging.DEBUG) and torch.cuda.is_available():
                        logger.debug(
                            f"[FLUX2:Transformer:forward] double_block.{i} peak memory delta: "
                            f"{_format_memory_gb(peak_delta)}"
                        )

                # Move block back to CPU
                self._move_block_to_device(block, self._offload_device, f"double_block.{i}")
                if not _tracing:
                    _log_memory_state(f"forward:double_block.{i}:after_unload", self._compute_device)
        else:
            # Standard mode - all blocks on same device
            if not _tracing:
                logger.debug("[FLUX2:Transformer:forward] Using standard mode (all blocks on device)")
            for i, block in enumerate(self.double_blocks):
                img, txt = block(
                    img,
                    txt,
                    pe_x,
                    pe_ctx,
                    double_block_mod_img,
                    double_block_mod_txt,
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
        if self._block_offload_enabled and self._compute_device and self._offload_device:
            # Block-by-block offloading mode (incompatible with torch.compile)
            for i, block in enumerate(self.single_blocks):
                # Track peak memory during block processing
                peak_before = torch.cuda.max_memory_allocated(self._compute_device) if torch.cuda.is_available() else 0

                # Move block to GPU
                self._move_block_to_device(block, self._compute_device, f"single_block.{i}")
                if self._compute_device.type == "cuda":
                    torch.cuda.synchronize()
                if not _tracing:
                    _log_memory_state(f"forward:single_block.{i}:after_load", self._compute_device)

                img = block(
                    img,
                    pe,
                    single_block_mod,
                )

                # Log peak memory during forward pass
                if not _tracing:
                    peak_after = torch.cuda.max_memory_allocated(self._compute_device) if torch.cuda.is_available() else 0
                    peak_delta = peak_after - peak_before
                    if logger.isEnabledFor(logging.DEBUG) and torch.cuda.is_available():
                        logger.debug(
                            f"[FLUX2:Transformer:forward] single_block.{i} peak memory delta: "
                            f"{_format_memory_gb(peak_delta)}"
                        )

                # Move block back to CPU
                self._move_block_to_device(block, self._offload_device, f"single_block.{i}")
                if not _tracing:
                    _log_memory_state(f"forward:single_block.{i}:after_unload", self._compute_device)
        else:
            # Standard mode
            for i, block in enumerate(self.single_blocks):
                img = block(
                    img,
                    pe,
                    single_block_mod,
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

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


# Alias for backward compatibility
Flux2 = Flux2Transformer
