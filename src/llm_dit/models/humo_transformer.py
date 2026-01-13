"""
HuMo Transformer for video generation with optional audio conditioning.

Last Updated: 2026-01-11

This implements the HuMo DiT architecture from ByteDance:
https://github.com/Phantom-video/HuMo

The model supports both T2V (text-to-video) and audio-conditioned generation.
Audio conditioning is controlled via audio_scale parameter at runtime.

Weight key mapping (matched to official HuMo weights):
- patch_embedding.weight, bias -> video patch embedding (Conv3d flattened)
- time_embedding.{0,2}.* -> sinusoidal time MLP (index 0, 2 = Linear layers)
- time_projection.1.* -> time projection (single Linear at index 1)
- text_embedding.{0,2}.* -> text context MLP (index 0, 2 = Linear layers)
- audio_proj.audio_proj_glob_*.layer.* -> audio projection with .layer wrapper
- blocks.N.modulation -> AdaLN modulation weights (generates scale/shift)
- blocks.N.norm3.* -> only norm3 (other norms via modulation)
- blocks.N.self_attn.{q,k,v,o}.*, norm_q.*, norm_k.* -> self-attention
- blocks.N.cross_attn.{q,k,v,o}.*, norm_q.*, norm_k.* -> text cross-attention
- blocks.N.audio_cross_attn_wrapper.* -> audio cross-attention
- blocks.N.ffn.{0,2}.* -> FFN (index 0, 2 = Linear layers)
- head.head.*, head.modulation -> output projection with modulation
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x = x / rms
        return (self.weight * x).to(dtype)


class HuMo3DRoPE(nn.Module):
    """
    3D Rotary Position Embedding for HuMo video transformer.

    Computes separate frequency components for:
    - Frame axis (temporal): axes_dim[0] dims (default 16)
    - Height axis (spatial): axes_dim[1] dims (default 56)
    - Width axis (spatial): axes_dim[2] dims (default 56)
    Total: 128 dims (matches head_dim = 5120/40)

    Uses centered indices for height/width (negative + positive) for better
    extrapolation to different resolutions.
    """

    def __init__(
        self,
        theta: float = 10000.0,
        axes_dim: Tuple[int, int, int] = (16, 56, 56),
        max_seq: int = 4096,
    ):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.head_dim = sum(axes_dim)

        # Precompute positive and negative index frequencies
        pos_index = torch.arange(max_seq, dtype=torch.float32)
        neg_index = torch.arange(max_seq, dtype=torch.float32).flip(0) * -1 - 1

        self.register_buffer("pos_freqs", self._compute_freqs(pos_index))
        self.register_buffer("neg_freqs", self._compute_freqs(neg_index))

    def _compute_freqs(self, index: torch.Tensor) -> torch.Tensor:
        """Compute complex frequency tensor for all axes."""
        freqs = []
        for dim in self.axes_dim:
            # Compute frequencies: 1 / (theta^(2i/d)) for i in [0, dim/2)
            axis_freqs = torch.outer(
                index,
                1.0 / torch.pow(self.theta, torch.arange(0, dim, 2, dtype=torch.float32) / dim)
            )
            # Store as complex exponentials
            freqs.append(torch.polar(torch.ones_like(axis_freqs), axis_freqs))
        return torch.cat(freqs, dim=1)

    def forward(
        self,
        num_frames: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Compute 3D RoPE frequencies for given video shape.

        Args:
            num_frames: Number of frames (after temporal patchification)
            height: Height in patches
            width: Width in patches
            device: Output device
            dtype: Output dtype

        Returns:
            Complex tensor [T*H*W, head_dim//2] for RoPE application
        """
        seq_len = num_frames * height * width

        # Split precomputed frequencies by axis
        pos_split = self.pos_freqs.split([d // 2 for d in self.axes_dim], dim=1)
        neg_split = self.neg_freqs.split([d // 2 for d in self.axes_dim], dim=1)

        # Frame frequencies: sequential positive indices
        f_freqs = pos_split[0][:num_frames].view(num_frames, 1, 1, -1)
        f_freqs = f_freqs.expand(num_frames, height, width, -1)

        # Height frequencies: centered (negative first half, positive second half)
        h_half = height // 2
        h_freqs = torch.cat([
            neg_split[1][-(height - h_half):],  # Negative indices for first half
            pos_split[1][:h_half],              # Positive indices for second half
        ], dim=0).view(1, height, 1, -1)
        h_freqs = h_freqs.expand(num_frames, height, width, -1)

        # Width frequencies: centered
        w_half = width // 2
        w_freqs = torch.cat([
            neg_split[2][-(width - w_half):],
            pos_split[2][:w_half],
        ], dim=0).view(1, 1, width, -1)
        w_freqs = w_freqs.expand(num_frames, height, width, -1)

        # Concatenate and reshape to [seq_len, head_dim//2]
        freqs = torch.cat([f_freqs, h_freqs, w_freqs], dim=-1)
        freqs = freqs.reshape(seq_len, -1).to(device)

        return freqs


class LayerNormWithBias(nn.Module):
    """Layer normalization with bias, wrapped to match weight key pattern."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)


class LinearWrapper(nn.Module):
    """Wrapper that adds .layer attribute for weight key compatibility."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.layer = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class LayerNormWrapper(nn.Module):
    """Wrapper that adds .layer attribute for weight key compatibility."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.layer = nn.LayerNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class AudioProjection(nn.Module):
    """
    Audio projection from Whisper features to model dimension.

    Weight keys: audio_proj.audio_proj_glob_*.layer.*
    Uses LinearWrapper/LayerNormWrapper to add .layer attribute.
    """

    def __init__(
        self,
        seq_len: int = 8,
        blocks: int = 5,
        channels: int = 1280,
        intermediate_dim: int = 512,
        output_dim: int = 1536,
        context_tokens: int = 16,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.blocks = blocks
        self.channels = channels
        self.context_tokens = context_tokens
        self.output_dim = output_dim

        input_dim = seq_len * blocks * channels

        # Match weight keys: audio_proj.audio_proj_glob_*.layer.*
        self.audio_proj_glob_1 = LinearWrapper(input_dim, intermediate_dim)
        self.audio_proj_glob_2 = LinearWrapper(intermediate_dim, intermediate_dim)
        self.audio_proj_glob_3 = LinearWrapper(intermediate_dim, context_tokens * output_dim)
        self.audio_proj_glob_norm = LayerNormWrapper(output_dim)

    def forward(self, audio: torch.Tensor, num_frames: int) -> torch.Tensor:
        """
        Project audio features to context tokens.

        Args:
            audio: [B*F, seq_len, blocks, channels] from Whisper
            num_frames: Number of video frames

        Returns:
            [B, F, context_tokens, output_dim] audio context
        """
        batch_frames = audio.shape[0]
        batch_size = batch_frames // num_frames

        # Flatten: [B*F, seq_len * blocks * channels]
        x = audio.reshape(batch_frames, -1)

        # Project through layers
        x = F.relu(self.audio_proj_glob_1(x))
        x = F.relu(self.audio_proj_glob_2(x))
        x = self.audio_proj_glob_3(x)

        # Reshape to context tokens: [B*F, context_tokens, output_dim]
        x = x.reshape(batch_frames, self.context_tokens, self.output_dim)

        # Apply layer norm
        x = self.audio_proj_glob_norm(x)

        # Reshape to [B, F, context_tokens, output_dim]
        x = x.reshape(batch_size, num_frames, self.context_tokens, self.output_dim)

        return x


class SelfAttention(nn.Module):
    """
    Self-attention with QK norm.

    Weight keys: blocks.N.self_attn.{q,k,v,o}.*, norm_q.*, norm_k.*

    Note: QK norm is applied BEFORE splitting into heads (full hidden_size),
    not per-head as in some other implementations.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        # Projections with bias
        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)
        self.o = nn.Linear(hidden_size, hidden_size)

        # QK norm - applied before head split (full hidden_size)
        self.norm_q = RMSNorm(hidden_size, eps=eps)
        self.norm_k = RMSNorm(hidden_size, eps=eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, _ = x.shape

        # Project Q, K, V
        q = self.q(x)  # [B, N, D]
        k = self.k(x)
        v = self.v(x)

        # Apply QK norm BEFORE head split
        q = self.norm_q(q)
        k = self.norm_k(k)

        # Now reshape to heads
        q = q.reshape(B, N, self.num_heads, self.head_dim)
        k = k.reshape(B, N, self.num_heads, self.head_dim)
        v = v.reshape(B, N, self.num_heads, self.head_dim)

        # Apply RoPE if provided
        if freqs is not None:
            q = self._apply_rope(q, freqs)
            k = self._apply_rope(k, freqs)

        # Attention: [B, H, N, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(B, N, self.hidden_size)
        return self.o(out)

    def _apply_rope(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        """Apply rotary position embedding.

        Args:
            x: [B, N, num_heads, head_dim] query or key tensor
            freqs: [N, head_dim//2] complex frequency tensor from HuMo3DRoPE

        Returns:
            Rotated tensor with same shape as input
        """
        # Reshape x to pairs for complex view: [B, N, H, D] -> [B, N, H, D/2, 2]
        x_reshape = x.reshape(*x.shape[:-1], -1, 2)
        x_complex = torch.view_as_complex(x_reshape.float())  # [B, N, H, D/2]

        # freqs is already complex: [N, D/2] -> [1, N, 1, D/2] for broadcasting
        freqs_broadcast = freqs.unsqueeze(0).unsqueeze(2)

        # Complex multiplication for rotation
        x_rotated = x_complex * freqs_broadcast

        # Convert back to real: [B, N, H, D/2] -> [B, N, H, D]
        x_out = torch.view_as_real(x_rotated).flatten(-2)
        return x_out.to(x.dtype)


class CrossAttention(nn.Module):
    """
    Cross-attention for text conditioning.

    Weight keys: blocks.N.cross_attn.{q,k,v,o}.*, norm_q.*, norm_k.*

    Note: QK norm is applied BEFORE splitting into heads (full hidden_size).
    """

    def __init__(
        self,
        hidden_size: int,
        context_dim: int,
        num_heads: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(context_dim, hidden_size)
        self.v = nn.Linear(context_dim, hidden_size)
        self.o = nn.Linear(hidden_size, hidden_size)

        # QK norm - applied before head split (full hidden_size)
        self.norm_q = RMSNorm(hidden_size, eps=eps)
        self.norm_k = RMSNorm(hidden_size, eps=eps)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        _, S, _ = context.shape

        # Project
        q = self.q(x)  # [B, N, D]
        k = self.k(context)  # [B, S, D]
        v = self.v(context)

        # Apply QK norm BEFORE head split
        q = self.norm_q(q)
        k = self.norm_k(k)

        # Reshape to heads
        q = q.reshape(B, N, self.num_heads, self.head_dim)
        k = k.reshape(B, S, self.num_heads, self.head_dim)
        v = v.reshape(B, S, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(B, N, self.hidden_size)
        return self.o(out)


class AudioCrossAttention(nn.Module):
    """
    Cross-attention for audio conditioning.

    Weight keys: blocks.N.audio_cross_attn_wrapper.audio_cross_attn.{q,k,v,o}.*,
                 norm_q.*, norm_k.*

    Note: QK norm is applied BEFORE splitting into heads (full hidden_size).
    """

    def __init__(
        self,
        hidden_size: int,
        audio_dim: int = 1536,
        num_heads: int = 40,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(audio_dim, hidden_size)
        self.v = nn.Linear(audio_dim, hidden_size)
        self.o = nn.Linear(hidden_size, hidden_size)

        # QK norm - applied before head split (full hidden_size)
        self.norm_q = RMSNorm(hidden_size, eps=eps)
        self.norm_k = RMSNorm(hidden_size, eps=eps)

    def forward(
        self,
        x: torch.Tensor,
        audio: torch.Tensor,
        audio_scale: float = 1.0,
    ) -> torch.Tensor:
        B, N, _ = x.shape
        _, A, _ = audio.shape

        # Project
        q = self.q(x)  # [B, N, D]
        k = self.k(audio)  # [B, A, D]
        v = self.v(audio)

        # Apply QK norm BEFORE head split
        q = self.norm_q(q)
        k = self.norm_k(k)

        # Reshape to heads
        q = q.reshape(B, N, self.num_heads, self.head_dim)
        k = k.reshape(B, A, self.num_heads, self.head_dim)
        v = v.reshape(B, A, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(B, N, self.hidden_size)
        out = self.o(out) * audio_scale
        return out


class AudioCrossAttentionWrapper(nn.Module):
    """
    Wrapper for audio cross-attention with normalization.

    Weight keys: blocks.N.audio_cross_attn_wrapper.norm1_audio.*, audio_cross_attn.*
    """

    def __init__(
        self,
        hidden_size: int,
        audio_dim: int = 1536,
        num_heads: int = 40,
        eps: float = 1e-6,
    ):
        super().__init__()
        # norm1_audio has bias (LayerNorm style)
        self.norm1_audio = nn.LayerNorm(hidden_size, eps=eps)
        self.audio_cross_attn = AudioCrossAttention(
            hidden_size=hidden_size,
            audio_dim=audio_dim,
            num_heads=num_heads,
            eps=eps,
        )

    def forward(
        self,
        x: torch.Tensor,
        audio: torch.Tensor,
        audio_scale: float = 1.0,
    ) -> torch.Tensor:
        normed = self.norm1_audio(x)
        out = self.audio_cross_attn(normed, audio, audio_scale)
        return x + out


class TransformerBlock(nn.Module):
    """
    Single transformer block with AdaLN modulation and audio cross-attention.

    Weight keys:
    - blocks.N.modulation -> AdaLN modulation (generates 6 scale/shift values)
    - blocks.N.norm3.* -> only norm3 (pre-FFN norm with bias)
    - blocks.N.self_attn.* -> self attention
    - blocks.N.cross_attn.* -> text cross attention
    - blocks.N.audio_cross_attn_wrapper.* -> audio cross attention
    - blocks.N.ffn.{0,2}.* -> FFN (Linear, SiLU, Linear)

    AdaLN modulation produces 6 values: [shift1, scale1, shift2, scale2, shift3, scale3]
    Applied to self_attn output, cross_attn output, and ffn input.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ffn_dim: int,
        text_dim: int = 4096,
        audio_dim: int = 1536,
        eps: float = 1e-6,
        has_audio: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # AdaLN modulation: produces 6 values from time embedding
        # Shape [1, 6, D] - initialized with randn/sqrt(dim) like official HuMo
        self.modulation = nn.Parameter(torch.randn(1, 6, hidden_size) / (hidden_size ** 0.5))

        # Layer norms following official HuMo:
        # - norm1: pre-norm for self-attention (no parameters - elementwise_affine=False)
        # - norm2: pre-norm for FFN (no parameters)
        # - norm3: pre-norm for cross-attention (WITH parameters - elementwise_affine=True)
        self.norm1 = nn.LayerNorm(hidden_size, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_size, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(hidden_size, eps=eps, elementwise_affine=True)

        # Attention layers
        self.self_attn = SelfAttention(hidden_size, num_heads, eps)
        self.cross_attn = CrossAttention(hidden_size, text_dim, num_heads, eps)

        # FFN with GELU(tanh) like official HuMo (not SiLU)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_dim),  # [0]
            nn.GELU(approximate='tanh'),  # [1] - not saved
            nn.Linear(ffn_dim, hidden_size),  # [2]
        )

        # Audio cross-attention (optional)
        self.has_audio = has_audio
        if has_audio:
            self.audio_cross_attn_wrapper = AudioCrossAttentionWrapper(
                hidden_size, audio_dim, num_heads, eps
            )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        time_mod: torch.Tensor,
        audio: Optional[torch.Tensor] = None,
        audio_scale: float = 0.0,
        freqs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass following official HuMo implementation.

        Key: All modulation operations happen in float32 for numerical stability.
        Modulation order: shift_sa, scale_sa, gate_sa, shift_ffn, scale_ffn, gate_ffn
        """
        # CRITICAL: Modulation must be computed in float32
        # time_mod: [B, 6, D] from time_projection (should already be float32)
        # self.modulation: [1, 6, D] per-block learned modulation
        with torch.amp.autocast(device_type='cuda', enabled=False):
            # Ensure float32 for modulation computation
            mod = (self.modulation.float() + time_mod.float()).chunk(6, dim=1)
            # mod is now 6 tensors of shape [B, 1, D] in float32

        # Unpack modulation values (all [B, 1, D] in float32)
        shift_sa, scale_sa, gate_sa, shift_ffn, scale_ffn, gate_ffn = mod

        # Self-attention: norm1 -> modulate -> self_attn -> gated residual
        # Input to self-attn is modulated: norm(x) * (1 + scale) + shift
        y = self.self_attn(
            self.norm1(x).float() * (1 + scale_sa) + shift_sa,
            freqs
        )
        # Gated residual in float32
        with torch.amp.autocast(device_type='cuda', enabled=False):
            x = x.float() + y.float() * gate_sa
        x = x.to(context.dtype)  # Back to model dtype

        # Cross-attention: norm3 -> cross_attn -> direct residual (no modulation!)
        x = x + self.cross_attn(self.norm3(x), context)

        # Audio cross-attention (if audio provided and scale > 0)
        if self.has_audio and audio is not None and audio_scale > 0:
            x = self.audio_cross_attn_wrapper(x, audio, audio_scale)

        # FFN: norm2 -> modulate -> ffn -> gated residual
        y = self.ffn(self.norm2(x).float() * (1 + scale_ffn) + shift_ffn)
        # Gated residual in float32
        with torch.amp.autocast(device_type='cuda', enabled=False):
            x = x.float() + y.float() * gate_ffn
        x = x.to(context.dtype)  # Back to model dtype

        return x


class OutputHead(nn.Module):
    """
    Output head with modulation.

    Weight keys: head.head.*, head.modulation, head.norm.*

    Official HuMo: Receives time_embedding (not time_projection!) as [B, D],
    broadcasts with modulation [1, 2, D] to get shift and scale.
    """

    def __init__(self, hidden_size: int, out_features: int, eps: float = 1e-6):
        super().__init__()
        # Shape [1, 2, D] - initialized with randn/sqrt(dim) like official HuMo
        self.modulation = nn.Parameter(torch.randn(1, 2, hidden_size) / (hidden_size ** 0.5))
        self.norm = nn.LayerNorm(hidden_size, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(hidden_size, out_features)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token features [B, N, D]
            time_emb: Time embedding from time_embedding (NOT time_projection!) [B, D]

        Returns:
            Output predictions [B, N, out_features]
        """
        # CRITICAL: All modulation operations in float32
        # time_emb: [B, D] -> unsqueeze to [B, 1, D]
        # self.modulation: [1, 2, D]
        # Broadcasting: [B, 1, D] + [1, 2, D] = [B, 2, D]
        with torch.amp.autocast(device_type='cuda', enabled=False):
            mod = (self.modulation.float() + time_emb.float().unsqueeze(1)).chunk(2, dim=1)
            # mod is 2 tensors of shape [B, 1, D]
            shift, scale = mod
            x = self.head(self.norm(x).float() * (1 + scale) + shift)
        return x


class HuMoTransformer(nn.Module):
    """
    HuMo Transformer for video generation with audio conditioning.

    Supports:
    - T2V (text-to-video): audio_scale=0
    - TA (text+audio): audio_scale>0
    - TIA (text+image+audio): audio_scale>0 with image latents

    Weight keys matched to official HuMo checkpoint.

    Args:
        num_layers: Number of transformer blocks (40 for 17B, 30 for 1.7B)
        hidden_size: Model dimension (5120 for 17B, 2048 for 1.7B)
        num_heads: Number of attention heads (40 for both variants)
        ffn_dim: Feed-forward dimension (13824 for 17B)
        in_channels: Input latent channels (16)
        text_dim: Text embedding dimension (4096 for UMT5-XXL)
        freq_dim: Time embedding dimension (256)
        audio_dim: Audio context dimension (1536)
        patch_size: 3D patch size (1, 2, 2)
        eps: Layer norm epsilon
    """

    def __init__(
        self,
        num_layers: int = 40,
        hidden_size: int = 5120,
        num_heads: int = 40,
        ffn_dim: int = 13824,
        in_channels: int = 16,
        out_channels: int = 16,
        patch_in_channels: int = 36,  # 36 for HuMo-17B (I2V/audio), 16 for HuMo-1.7B (T2V only)
        text_dim: int = 4096,
        freq_dim: int = 256,
        audio_dim: int = 1536,
        audio_token_num: int = 16,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.audio_token_num = audio_token_num
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Patch embedding: Conv3d flattened to weight/bias
        # Weight keys: patch_embedding.weight, patch_embedding.bias
        # HuMo-17B uses 36 (noise 16 + image 16 + audio 4), HuMo-1.7B uses 16 (noise only)
        self.patch_embedding = nn.Conv3d(
            patch_in_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # Time embedding: Sequential with Linear at 0, 2
        # Weight keys: time_embedding.{0,2}.*
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, hidden_size),  # [0]
            nn.SiLU(),  # [1] - not saved
            nn.Linear(hidden_size, hidden_size),  # [2]
        )

        # Time projection: outputs 6 * hidden_size for all modulation values
        # Weight keys: time_projection.1.*
        # Output: 30720 = 6 * 5120 (shift/scale for self_attn, cross_attn, ffn)
        self.time_projection = nn.Sequential(
            nn.Identity(),  # [0] - placeholder
            nn.Linear(hidden_size, hidden_size * 6),  # [1] outputs all modulations
        )

        # Text embedding: Sequential with Linear at 0, 2
        # Weight keys: text_embedding.{0,2}.*
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, hidden_size),  # [0]
            nn.SiLU(),  # [1] - not saved
            nn.Linear(hidden_size, hidden_size),  # [2]
        )

        # Audio projection
        self.audio_proj = AudioProjection(
            seq_len=8,
            blocks=5,
            channels=1280,
            intermediate_dim=512,
            output_dim=audio_dim,
            context_tokens=audio_token_num,
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                text_dim=hidden_size,  # After text_embedding projection
                audio_dim=audio_dim,
                eps=eps,
                has_audio=True,
            )
            for _ in range(num_layers)
        ])

        # Output head
        out_features = out_channels * patch_size[0] * patch_size[1] * patch_size[2]
        self.head = OutputHead(hidden_size, out_features, eps)

        # 3D RoPE for positional encoding
        # axes_dim follows official HuMo formula: (d - 4*(d//6), 2*(d//6), 2*(d//6))
        # For head_dim=64: (24, 20, 20)
        # For head_dim=128: (44, 42, 42)
        head_dim = hidden_size // num_heads
        axes_dim = (
            head_dim - 4 * (head_dim // 6),  # temporal
            2 * (head_dim // 6),              # height
            2 * (head_dim // 6),              # width
        )
        self.rope = HuMo3DRoPE(
            theta=10000.0,
            axes_dim=axes_dim,
            max_seq=4096,
        )

        # Sinusoidal embedding for timesteps (not saved, computed)
        self._freq_dim = freq_dim

    def _get_sinusoidal_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Compute sinusoidal embedding for timesteps."""
        # Store original dtype for final conversion
        original_dtype = timesteps.dtype
        half_dim = self._freq_dim // 2
        # Use fp32 for numerical precision in sinusoidal computation
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, device=timesteps.device, dtype=torch.float32)
            / half_dim
        )
        args = timesteps[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        # Convert back to model dtype
        return embedding.to(original_dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        audio_hidden_states: Optional[torch.Tensor] = None,
        audio_scale: float = 0.0,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            hidden_states: Video latents [B, C, T, H, W]
            timestep: Diffusion timesteps [B]
            encoder_hidden_states: Text embeddings [B, S, text_dim]
            audio_hidden_states: Audio embeddings [B, T_audio, audio_dim] (optional)
            audio_scale: Audio guidance strength (0 = T2V mode)

        Returns:
            Noise prediction [B, C, T, H, W]
        """
        B, C, T, H, W = hidden_states.shape

        # Patch embed: [B, C, T, H, W] -> [B, D, T', H', W'] -> [B, N, D]
        x = self.patch_embedding(hidden_states)
        x = rearrange(x, "b d t h w -> b (t h w) d")

        # Time embedding - MUST be computed and stored in float32 for modulation
        with torch.amp.autocast(device_type='cuda', enabled=False):
            t_emb = self._get_sinusoidal_embedding(timestep.float())  # [B, freq_dim]
            t_emb = self.time_embedding(t_emb.float()).float()  # [B, D] - stays float32 for head
            t_mod = self.time_projection(t_emb).unflatten(1, (6, self.hidden_size)).float()  # [B, 6, D]

        # Project text context
        context = self.text_embedding(encoder_hidden_states)  # [B, S, D]

        # Process audio if provided
        audio = None
        if audio_hidden_states is not None and audio_scale > 0:
            num_frames = T // self.patch_size[0]
            audio = self._prepare_audio(audio_hidden_states, num_frames)

        # Compute 3D RoPE frequencies for video patches
        p1, p2, p3 = self.patch_size
        T_out = T // p1
        H_out = H // p2
        W_out = W // p3
        freqs = self.rope(T_out, H_out, W_out, hidden_states.device)

        # Transformer blocks - pass time modulation (float32)
        for block in self.blocks:
            x = block(x, context, t_mod, audio, audio_scale, freqs)

        # Output head - receives t_emb [B, D] (NOT t_mod!)
        # Official HuMo passes time_embedding output to head, not time_projection output
        x = self.head(x, t_emb)  # [B, N, C*p1*p2*p3]

        # Unpatchify to output channels (16), not input channels (36)
        C_out = self.out_channels

        x = x.reshape(B, T_out, H_out, W_out, C_out, p1, p2, p3)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7)
        x = x.reshape(B, C_out, T, H, W)

        return x

    def _prepare_audio(
        self,
        audio_hidden_states: torch.Tensor,
        num_frames: int,
    ) -> torch.Tensor:
        """
        Prepare audio embeddings for cross-attention.

        Audio projection expects input shaped as [B*F, seq_len=8, blocks=5, channels=1280]
        representing 40 Whisper timesteps per video frame.

        Args:
            audio_hidden_states: [B, T_audio, 1280] from Whisper encoder
            num_frames: Number of video frames (after temporal patchification)

        Returns:
            [B, num_frames * audio_token_num, audio_dim] audio context
        """
        B, T_audio, whisper_dim = audio_hidden_states.shape

        # Audio projection expects: seq_len * blocks = 8 * 5 = 40 timesteps per frame
        timesteps_per_frame = self.audio_proj.seq_len * self.audio_proj.blocks  # 40

        # Calculate how many Whisper timesteps we need total
        total_needed = num_frames * timesteps_per_frame

        # Interpolate or pad/truncate audio to match needed length
        if T_audio != total_needed:
            # Reshape for interpolation: [B, 1280, T_audio] -> interpolate -> [B, 1280, total_needed]
            audio_t = audio_hidden_states.transpose(1, 2)  # [B, 1280, T_audio]
            audio_t = F.interpolate(audio_t, size=total_needed, mode='linear', align_corners=False)
            audio_hidden_states = audio_t.transpose(1, 2)  # [B, total_needed, 1280]

        # Reshape to [B, num_frames, seq_len, blocks, channels]
        audio_reshaped = audio_hidden_states.view(
            B, num_frames,
            self.audio_proj.seq_len,  # 8
            self.audio_proj.blocks,   # 5
            whisper_dim               # 1280
        )

        # Flatten batch and frames for projection: [B*F, seq_len, blocks, channels]
        audio_flat = audio_reshaped.view(
            B * num_frames,
            self.audio_proj.seq_len,
            self.audio_proj.blocks,
            whisper_dim
        )

        # Project through audio_proj: [B*F, 8, 5, 1280] -> [B, F, 16, 1536]
        audio_tokens = self.audio_proj(audio_flat, num_frames)

        # Reshape for cross-attention: [B, F * context_tokens, output_dim]
        # audio_tokens is [B, F, context_tokens=16, output_dim=1536]
        audio_context = audio_tokens.view(B, -1, self.audio_proj.output_dim)

        return audio_context
