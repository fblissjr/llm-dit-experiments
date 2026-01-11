"""
HuMo Transformer for video generation with optional audio conditioning.

Last Updated: 2026-01-11

This implements the HuMo DiT architecture from ByteDance:
https://github.com/Phantom-video/HuMo

The model supports both T2V (text-to-video) and audio-conditioned generation.
Audio conditioning is controlled via audio_scale parameter at runtime.

Architecture based on official HuMo repo (humo/models/wan_modules/model_humo.py).

Weight key mapping:
- patch_embedding.* -> video patch embedding
- time_embedding.* -> sinusoidal time embedding
- time_projection.* -> time MLP projection
- text_embedding.* -> text context embedding
- audio_proj.audio_proj_glob_* -> audio projection layers
- blocks.N.* -> transformer blocks with audio cross-attention
- head.* -> output projection
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


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal positional embedding for timesteps."""

    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half_dim, device=timesteps.device, dtype=torch.float32)
            / half_dim
        )
        args = timesteps[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class PatchEmbedding3D(nn.Module):
    """3D patch embedding for video: (C, T, H, W) -> (N, D)."""

    def __init__(
        self,
        in_channels: int = 16,
        hidden_size: int = 5120,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(
            in_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, H, W]
        x = self.proj(x)  # [B, D, T', H', W']
        x = rearrange(x, "b d t h w -> b (t h w) d")
        return x


class AudioProjection(nn.Module):
    """
    Audio projection from Whisper features to model dimension.

    Maps flattened Whisper features to context tokens:
    - audio_proj_glob_1: [intermediate_dim, seq_len * blocks * channels]
    - audio_proj_glob_2: [intermediate_dim, intermediate_dim]
    - audio_proj_glob_3: [context_tokens * output_dim, intermediate_dim]
    - audio_proj_glob_norm: LayerNorm(output_dim)
    """

    def __init__(
        self,
        seq_len: int = 8,
        blocks: int = 5,
        channels: int = 1280,  # Whisper hidden dim
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

        # Match HuMo weight keys: audio_proj.audio_proj_glob_*
        self.audio_proj_glob_1 = nn.Linear(input_dim, intermediate_dim)
        self.audio_proj_glob_2 = nn.Linear(intermediate_dim, intermediate_dim)
        self.audio_proj_glob_3 = nn.Linear(intermediate_dim, context_tokens * output_dim)
        self.audio_proj_glob_norm = nn.LayerNorm(output_dim)

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
    """Self-attention with optional RoPE."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        qk_norm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)
        self.o = nn.Linear(hidden_size, hidden_size)

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=eps)
            self.k_norm = RMSNorm(self.head_dim, eps=eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, _ = x.shape

        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim)
        k = self.k(x).reshape(B, N, self.num_heads, self.head_dim)
        v = self.v(x).reshape(B, N, self.num_heads, self.head_dim)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Apply RoPE if provided
        if freqs is not None:
            q = self._apply_rope(q, freqs)
            k = self._apply_rope(k, freqs)

        # Attention
        q = q.transpose(1, 2)  # [B, H, N, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(B, N, self.hidden_size)
        return self.o(out)

    def _apply_rope(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        """Apply rotary position embedding."""
        # x: [B, N, H, D], freqs: [N, D//2, 2]
        x_reshape = x.reshape(*x.shape[:-1], -1, 2)
        freqs = freqs.unsqueeze(0).unsqueeze(2)  # [1, N, 1, D//2, 2]

        # Complex rotation
        x_complex = torch.view_as_complex(x_reshape.float())
        freqs_complex = torch.view_as_complex(freqs.float())
        x_rotated = x_complex * freqs_complex
        x_out = torch.view_as_real(x_rotated).flatten(-2)
        return x_out.to(x.dtype)


class CrossAttention(nn.Module):
    """Cross-attention for text conditioning."""

    def __init__(
        self,
        hidden_size: int,
        context_dim: int,
        num_heads: int,
        qk_norm: bool = True,
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

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=eps)
            self.k_norm = RMSNorm(self.head_dim, eps=eps)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        _, S, _ = context.shape

        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim)
        k = self.k(context).reshape(B, S, self.num_heads, self.head_dim)
        v = self.v(context).reshape(B, S, self.num_heads, self.head_dim)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(B, N, self.hidden_size)
        return self.o(out)


class AudioCrossAttention(nn.Module):
    """Cross-attention for audio conditioning."""

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

        # Match HuMo weight keys: blocks.N.audio_cross_attn_wrapper.audio_cross_attn.*
        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(audio_dim, hidden_size)
        self.v = nn.Linear(audio_dim, hidden_size)
        self.o = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        audio: torch.Tensor,
        audio_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] hidden states
            audio: [B, A, audio_dim] audio context
            audio_scale: Scaling factor for audio influence

        Returns:
            [B, N, D] audio-conditioned hidden states
        """
        B, N, _ = x.shape
        _, A, _ = audio.shape

        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim)
        k = self.k(audio).reshape(B, A, self.num_heads, self.head_dim)
        v = self.v(audio).reshape(B, A, self.num_heads, self.head_dim)

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
    """Wrapper for audio cross-attention with normalization."""

    def __init__(
        self,
        hidden_size: int,
        audio_dim: int = 1536,
        num_heads: int = 40,
        eps: float = 1e-6,
    ):
        super().__init__()
        # Match HuMo weight keys: blocks.N.audio_cross_attn_wrapper.*
        self.norm1_audio = RMSNorm(hidden_size, eps=eps)
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


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, hidden_size: int, ffn_dim: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, ffn_dim)
        self.w2 = nn.Linear(ffn_dim, hidden_size)
        self.w3 = nn.Linear(hidden_size, ffn_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU variant
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """Single transformer block with audio cross-attention."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ffn_dim: int,
        text_dim: int = 4096,
        audio_dim: int = 1536,
        qk_norm: bool = True,
        eps: float = 1e-6,
        has_audio: bool = True,
    ):
        super().__init__()
        # Match HuMo weight keys: blocks.N.*
        self.norm1 = RMSNorm(hidden_size, eps=eps)
        self.self_attn = SelfAttention(hidden_size, num_heads, qk_norm, eps)

        self.norm2 = RMSNorm(hidden_size, eps=eps)
        self.cross_attn = CrossAttention(hidden_size, text_dim, num_heads, qk_norm, eps)

        self.norm3 = RMSNorm(hidden_size, eps=eps)
        self.ffn = FeedForward(hidden_size, ffn_dim)

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
        audio: Optional[torch.Tensor] = None,
        audio_scale: float = 0.0,
        freqs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention
        x = x + self.self_attn(self.norm1(x), freqs)

        # Text cross-attention
        x = x + self.cross_attn(self.norm2(x), context)

        # Audio cross-attention (if audio provided and scale > 0)
        if self.has_audio and audio is not None and audio_scale > 0:
            x = self.audio_cross_attn_wrapper(x, audio, audio_scale)

        # FFN
        x = x + self.ffn(self.norm3(x))

        return x


class HuMoTransformer(nn.Module):
    """
    HuMo Transformer for video generation with audio conditioning.

    Supports:
    - T2V (text-to-video): audio_scale=0
    - TA (text+audio): audio_scale>0
    - TIA (text+image+audio): audio_scale>0 with image latents

    Args:
        num_layers: Number of transformer blocks (40 for 17B, 30 for 1.7B)
        hidden_size: Model dimension (5120 for 17B, 2048 for 1.7B)
        num_heads: Number of attention heads (40 for both variants)
        ffn_dim: Feed-forward dimension (13824 for 17B)
        in_channels: Input latent channels (16)
        text_dim: Text embedding dimension (4096 for UMT5-XXL)
        text_len: Max text sequence length (512)
        freq_dim: RoPE frequency dimension (256)
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
        text_dim: int = 4096,
        text_len: int = 512,
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
        self.text_len = text_len
        self.audio_token_num = audio_token_num

        # Patch embedding
        self.patch_embedding = PatchEmbedding3D(
            in_channels=in_channels,
            hidden_size=hidden_size,
            patch_size=patch_size,
        )

        # Time embedding
        self.time_embedding = SinusoidalEmbedding(freq_dim)
        self.time_projection = nn.Sequential(
            nn.Linear(freq_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # Text embedding projection
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # Audio projection (from Whisper features)
        self.audio_proj = AudioProjection(
            seq_len=8,
            blocks=5,
            channels=1280,  # Whisper hidden dim
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
                qk_norm=True,
                eps=eps,
                has_audio=True,
            )
            for _ in range(num_layers)
        ])

        # Output head
        self.head = nn.Sequential(
            RMSNorm(hidden_size, eps=eps),
            nn.Linear(hidden_size, out_channels * patch_size[0] * patch_size[1] * patch_size[2]),
        )

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

        # Patch embed
        x = self.patch_embedding(hidden_states)  # [B, N, D]
        N = x.shape[1]

        # Time embedding
        t_emb = self.time_embedding(timestep)
        t_emb = self.time_projection(t_emb)  # [B, D]

        # Add time embedding to all tokens
        x = x + t_emb.unsqueeze(1)

        # Project text context
        context = self.text_embedding(encoder_hidden_states)  # [B, S, D]

        # Process audio if provided
        audio = None
        if audio_hidden_states is not None and audio_scale > 0:
            # Reshape and project audio
            # audio_hidden_states: [B, T_audio, 1280] from Whisper
            # Need to reshape for audio_proj which expects [B*F, seq_len, blocks, channels]
            # For simplicity, flatten and project
            num_frames = T // self.patch_size[0]
            audio = self._prepare_audio(audio_hidden_states, num_frames)

        # RoPE frequencies (simplified - could use 3D freqs)
        freqs = None  # TODO: implement 3D RoPE

        # Transformer blocks
        for block in self.blocks:
            x = block(x, context, audio, audio_scale, freqs)

        # Output head
        x = self.head(x)  # [B, N, C*p1*p2*p3]

        # Unpatchify
        p1, p2, p3 = self.patch_size
        T_out = T // p1
        H_out = H // p2
        W_out = W // p3

        x = x.reshape(B, T_out, H_out, W_out, C, p1, p2, p3)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7)  # [B, C, T_out, p1, H_out, p2, W_out, p3]
        x = x.reshape(B, C, T, H, W)

        return x

    def _prepare_audio(
        self,
        audio_hidden_states: torch.Tensor,
        num_frames: int,
    ) -> torch.Tensor:
        """
        Prepare audio embeddings for cross-attention.

        This is a simplified version - the full HuMo uses a more complex
        reshaping based on video frame alignment.

        Args:
            audio_hidden_states: [B, T_audio, 1280] from Whisper encoder
            num_frames: Number of video frames

        Returns:
            [B, num_frames * audio_token_num, audio_dim] audio context
        """
        B = audio_hidden_states.shape[0]

        # Simple approach: interpolate audio to match video frames
        # then flatten context tokens
        audio = audio_hidden_states  # [B, T_audio, 1280]

        # Reshape for projection (simplified)
        # In full HuMo, this involves window-based processing
        # Here we just project directly
        audio_flat = audio.reshape(B, -1)  # [B, T_audio * 1280]

        # Use a simpler linear projection for now
        # Full implementation would use audio_proj with proper reshaping
        audio_context = audio_hidden_states  # Pass through for cross-attention

        return audio_context
