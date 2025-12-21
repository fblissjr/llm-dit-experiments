"""
Qwen-Image DiT components - transformer layers and attention.

This is a simplified port of the DiffSynth-Studio QwenImageDiT implementation,
containing the essential components for inference.

Based on: coderef/DiffSynth-Studio/diffsynth/models/qwen_image_dit.py
"""

import math
import functools
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange

# DyPE imports (lazy to avoid circular imports)
if TYPE_CHECKING:
    from llm_dit.utils.dype import DyPEConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class AdaLayerNorm(nn.Module):
    """Adaptive Layer Normalization with learned shift and scale."""

    def __init__(self, dim: int, single: bool = False):
        super().__init__()
        self.single = single
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        out_dim = 2 * dim if single else 6 * dim
        self.linear = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, out_dim),
        )

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        params = self.linear(conditioning)
        if self.single:
            shift, scale = params.chunk(2, dim=-1)
            return self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        else:
            shift, scale, gate = params.chunk(3, dim=-1)
            return self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1), gate.unsqueeze(1)


class TimestepEmbeddings(nn.Module):
    """Sinusoidal timestep embeddings with optional additional conditioning.

    Matches DiffSynth's TimestepEmbeddings from general_modules.py.

    Args:
        embedding_dim: Dimension for sinusoidal embedding
        out_dim: Output dimension after projection
        scale: Scale factor for timestep (default 1000.0)
        use_additional_t_cond: If True, add an embedding for conditioning type (0 or 1)
    """

    def __init__(
        self,
        embedding_dim: int,
        out_dim: int,
        scale: float = 1000.0,
        use_additional_t_cond: bool = False,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.scale = scale
        self.use_additional_t_cond = use_additional_t_cond
        self.linear_1 = nn.Linear(embedding_dim, out_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(out_dim, out_dim)

        # Optional additional conditioning embedding (matches DiffSynth)
        if use_additional_t_cond:
            self.addition_t_embedding = nn.Embedding(2, out_dim)

    def forward(
        self,
        timestep: torch.Tensor,
        dtype: torch.dtype,
        addition_t_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Sinusoidal embedding (compute in float for precision)
        half_dim = self.embedding_dim // 2
        exponent = -math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=timestep.device) / half_dim
        emb = timestep.float() * self.scale * torch.exp(exponent)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        # Cast to layer dtype before projection
        emb = emb.to(dtype=self.linear_1.weight.dtype)

        # Project
        emb = self.linear_1(emb)
        emb = self.act(emb)
        emb = self.linear_2(emb)

        # Add optional conditioning embedding
        if addition_t_cond is not None and self.use_additional_t_cond:
            addition_emb = self.addition_t_embedding(addition_t_cond)
            emb = emb + addition_emb.to(dtype=emb.dtype)

        return emb


class ApproximateGELU(nn.Module):
    """Approximate GELU activation."""

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x * torch.sigmoid(1.702 * x)


class QwenFeedForward(nn.Module):
    """Feed-forward network with approximate GELU."""

    def __init__(self, dim: int, dim_out: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        inner_dim = int(dim * 4)
        self.net = nn.ModuleList([
            ApproximateGELU(dim, inner_dim),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out or dim),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            x = module(x)
        return x


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding."""
    x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)
    return x_out.type_as(x)


class QwenEmbedRope(nn.Module):
    """3-axis RoPE for Qwen-Image (single image generation).

    Supports DyPE (Dynamic Position Extrapolation) for high-resolution generation.
    When DyPE is enabled, frequencies are computed dynamically based on the
    diffusion timestep (sigma) using Vision YaRN.
    """

    def __init__(self, theta: int = 10000, axes_dim: List[int] = [16, 56, 56], scale_rope: bool = True):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.scale_rope = scale_rope

        # Pre-compute frequency tables
        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1

        self.register_buffer("pos_freqs", self._compute_freqs(pos_index))
        self.register_buffer("neg_freqs", self._compute_freqs(neg_index))
        self.rope_cache = {}
        self.dype_cache = {}  # Separate cache for DyPE (timestep-dependent)

    def _compute_freqs(self, index: torch.Tensor) -> torch.Tensor:
        """Compute frequency tables for all axes."""
        freqs = []
        for dim in self.axes_dim:
            axis_freqs = torch.outer(
                index.float(),
                1.0 / torch.pow(self.theta, torch.arange(0, dim, 2).float() / dim)
            )
            freqs.append(torch.polar(torch.ones_like(axis_freqs), axis_freqs))
        return torch.cat(freqs, dim=1)

    def _compute_dype_freqs(
        self,
        height: int,
        width: int,
        frame: int,
        idx: int,
        sigma: float,
        dype_config: "DyPEConfig",
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Compute DyPE-modulated frequencies using Vision YaRN.

        Args:
            height: Image height in patches
            width: Image width in patches
            frame: Frame index
            idx: Image index (for frame offset)
            sigma: Current diffusion timestep (0-1, high=noise, low=detail)
            dype_config: DyPE configuration
            device: Output device
            dtype: Output dtype

        Returns:
            Complex frequency tensor (seq_len, total_dim)
        """
        from llm_dit.utils.vision_yarn import get_1d_vision_yarn_pos_embed, get_1d_ntk_pos_embed
        from llm_dit.utils.dype import compute_k_t, compute_mscale, axis_token_span

        seq_len = frame * height * width

        # Compute base patch grid from base resolution
        # Qwen-Image uses patch_size=2 on 8x downsampled latents
        base_patches = (dype_config.base_resolution // 8) // 2

        # Compute scaling factors
        h_span = float(height)
        w_span = float(width)
        scale_global = max(1.0, max(h_span / base_patches, w_span / base_patches))

        # Compute timestep-dependent parameters
        k_t = compute_k_t(sigma, dype_config)
        mscale = compute_mscale(scale_global, sigma, dype_config)

        components = []
        freqs_dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32

        for axis_idx, axis_dim in enumerate(self.axes_dim):
            if axis_idx == 0:
                # Frame axis: use simple NTK (no extrapolation needed for single frame)
                pos = torch.arange(idx, idx + frame, device=device, dtype=freqs_dtype)
                cos, sin = get_1d_ntk_pos_embed(
                    dim=axis_dim,
                    pos=pos,
                    theta=self.theta,
                    ntk_factor=1.0,
                    freqs_dtype=freqs_dtype,
                )
            else:
                # Height/Width axes: use Vision YaRN for extrapolation
                if axis_idx == 1:
                    axis_len = height
                else:
                    axis_len = width

                # Generate position indices with centering (matching scale_rope=True)
                if self.scale_rope:
                    # Center positions around 0
                    half = axis_len // 2
                    pos = torch.cat([
                        torch.arange(-half, 0, device=device, dtype=freqs_dtype),
                        torch.arange(0, axis_len - half, device=device, dtype=freqs_dtype),
                    ])
                else:
                    pos = torch.arange(axis_len, device=device, dtype=freqs_dtype)

                linear_scale = max(1.0, float(axis_len) / base_patches)

                if scale_global > 1.0:
                    # Use Vision YaRN with DyPE modulation
                    cos, sin = get_1d_vision_yarn_pos_embed(
                        dim=axis_dim,
                        pos=pos,
                        theta=self.theta,
                        ori_max_pe_len=base_patches,
                        linear_scale=linear_scale,
                        ntk_scale=scale_global,
                        dype=True,
                        dype_scale=dype_config.dype_scale,
                        dype_exponent=dype_config.dype_exponent,
                        current_timestep=sigma,
                        override_mscale=mscale,
                        freqs_dtype=freqs_dtype,
                    )
                else:
                    # No extrapolation needed
                    cos, sin = get_1d_ntk_pos_embed(
                        dim=axis_dim,
                        pos=pos,
                        theta=self.theta,
                        ntk_factor=1.0,
                        freqs_dtype=freqs_dtype,
                    )

            # Convert to complex form
            axis_freqs = torch.polar(cos, sin * math.pi)  # Note: sin already contains angles
            components.append(axis_freqs)

        # Expand and combine
        # Frame: (frame,) -> (frame, 1, 1, dim)
        freqs_frame = components[0].view(frame, 1, 1, -1).expand(frame, height, width, -1)
        # Height: (height,) -> (1, height, 1, dim)
        freqs_height = components[1].view(1, height, 1, -1).expand(frame, height, width, -1)
        # Width: (width,) -> (1, 1, width, dim)
        freqs_width = components[2].view(1, 1, width, -1).expand(frame, height, width, -1)

        freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1)
        return freqs.reshape(seq_len, -1).to(device)

    def forward(
        self,
        img_shapes: List[Tuple[int, int, int]],
        txt_seq_lens: List[int],
        device: torch.device,
        sigma: Optional[float] = None,
        dype_config: Optional["DyPEConfig"] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute RoPE embeddings for image and text.

        Args:
            img_shapes: List of (frame, height, width) for each image
            txt_seq_lens: List of text sequence lengths
            device: Device for output tensors
            sigma: Current diffusion timestep (0-1) for DyPE modulation
            dype_config: DyPE configuration (enables DyPE when provided and enabled)

        Returns:
            Tuple of (image_freqs, text_freqs)
        """
        use_dype = (
            dype_config is not None
            and dype_config.enabled
            and sigma is not None
        )

        if self.pos_freqs.device != device:
            self.pos_freqs = self.pos_freqs.to(device)
            self.neg_freqs = self.neg_freqs.to(device)

        vid_freqs = []
        max_vid_index = 0

        for idx, (frame, height, width) in enumerate(img_shapes):
            if use_dype:
                # DyPE: compute timestep-dependent frequencies
                # Note: we don't cache DyPE freqs since they vary per timestep
                freqs = self._compute_dype_freqs(
                    height, width, frame, idx,
                    sigma, dype_config, device,
                    self.pos_freqs.dtype,
                )
                vid_freqs.append(freqs)
            else:
                # Standard: use cached static frequencies
                rope_key = f"{idx}_{height}_{width}"

                if rope_key not in self.rope_cache:
                    seq_len = frame * height * width
                    freqs_pos = self.pos_freqs.split([d // 2 for d in self.axes_dim], dim=1)
                    freqs_neg = self.neg_freqs.split([d // 2 for d in self.axes_dim], dim=1)

                    # Frame dimension
                    freqs_frame = freqs_pos[0][idx:idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)

                    # Height and width with optional centering
                    if self.scale_rope:
                        freqs_height = torch.cat([
                            freqs_neg[1][-(height - height // 2):],
                            freqs_pos[1][:height // 2]
                        ], dim=0).view(1, height, 1, -1).expand(frame, height, width, -1)
                        freqs_width = torch.cat([
                            freqs_neg[2][-(width - width // 2):],
                            freqs_pos[2][:width // 2]
                        ], dim=0).view(1, 1, width, -1).expand(frame, height, width, -1)
                    else:
                        freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
                        freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

                    freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_len, -1)
                    self.rope_cache[rope_key] = freqs.clone().contiguous()

                vid_freqs.append(self.rope_cache[rope_key])

            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)

        max_len = max(txt_seq_lens)
        txt_freqs = self.pos_freqs[max_vid_index:max_vid_index + max_len]
        vid_freqs = torch.cat(vid_freqs, dim=0)

        return vid_freqs, txt_freqs


class QwenEmbedLayer3DRope(nn.Module):
    """Layer-aware 3D RoPE for multi-layer decomposition.

    Supports DyPE (Dynamic Position Extrapolation) for high-resolution generation.
    When DyPE is enabled, frequencies are computed dynamically based on the
    diffusion timestep (sigma) using Vision YaRN.
    """

    def __init__(self, theta: int = 10000, axes_dim: List[int] = [16, 56, 56], scale_rope: bool = True):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.scale_rope = scale_rope

        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1

        self.register_buffer("pos_freqs", self._compute_freqs(pos_index))
        self.register_buffer("neg_freqs", self._compute_freqs(neg_index))

    def _compute_freqs(self, index: torch.Tensor) -> torch.Tensor:
        freqs = []
        for dim in self.axes_dim:
            axis_freqs = torch.outer(
                index.float(),
                1.0 / torch.pow(self.theta, torch.arange(0, dim, 2).float() / dim)
            )
            freqs.append(torch.polar(torch.ones_like(axis_freqs), axis_freqs))
        return torch.cat(freqs, dim=1)

    @functools.lru_cache(maxsize=None)  # Unlimited cache (matches DiffSynth)
    def _compute_video_freqs(self, frame: int, height: int, width: int, idx: int = 0) -> torch.Tensor:
        seq_len = frame * height * width
        freqs_pos = self.pos_freqs.split([d // 2 for d in self.axes_dim], dim=1)
        freqs_neg = self.neg_freqs.split([d // 2 for d in self.axes_dim], dim=1)

        freqs_frame = freqs_pos[0][idx:idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)

        if self.scale_rope:
            freqs_height = torch.cat([
                freqs_neg[1][-(height - height // 2):],
                freqs_pos[1][:height // 2]
            ], dim=0).view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = torch.cat([
                freqs_neg[2][-(width - width // 2):],
                freqs_pos[2][:width // 2]
            ], dim=0).view(1, 1, width, -1).expand(frame, height, width, -1)
        else:
            freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

        return torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_len, -1).clone().contiguous()

    @functools.lru_cache(maxsize=None)  # Unlimited cache (matches DiffSynth)
    def _compute_condition_freqs(self, frame: int, height: int, width: int) -> torch.Tensor:
        """Compute freqs for condition image (uses negative index)."""
        seq_len = frame * height * width
        freqs_pos = self.pos_freqs.split([d // 2 for d in self.axes_dim], dim=1)
        freqs_neg = self.neg_freqs.split([d // 2 for d in self.axes_dim], dim=1)

        # Use negative index for condition image
        freqs_frame = freqs_neg[0][-1:].view(frame, 1, 1, -1).expand(frame, height, width, -1)

        if self.scale_rope:
            freqs_height = torch.cat([
                freqs_neg[1][-(height - height // 2):],
                freqs_pos[1][:height // 2]
            ], dim=0).view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = torch.cat([
                freqs_neg[2][-(width - width // 2):],
                freqs_pos[2][:width // 2]
            ], dim=0).view(1, 1, width, -1).expand(frame, height, width, -1)
        else:
            freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

        return torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_len, -1).clone().contiguous()

    def _compute_dype_freqs(
        self,
        height: int,
        width: int,
        frame: int,
        idx: int,
        is_condition: bool,
        sigma: float,
        dype_config: "DyPEConfig",
        device: torch.device,
    ) -> torch.Tensor:
        """Compute DyPE-modulated frequencies using Vision YaRN.

        Args:
            height: Image height in patches
            width: Image width in patches
            frame: Frame count
            idx: Image index (for frame offset, -1 for condition)
            is_condition: True if this is the condition image
            sigma: Current diffusion timestep (0-1)
            dype_config: DyPE configuration
            device: Output device

        Returns:
            Complex frequency tensor (seq_len, total_dim)
        """
        from llm_dit.utils.vision_yarn import get_1d_vision_yarn_pos_embed, get_1d_ntk_pos_embed
        from llm_dit.utils.dype import compute_k_t, compute_mscale

        seq_len = frame * height * width

        # Compute base patch grid from base resolution
        base_patches = (dype_config.base_resolution // 8) // 2

        # Compute scaling factors
        h_span = float(height)
        w_span = float(width)
        scale_global = max(1.0, max(h_span / base_patches, w_span / base_patches))

        # Compute timestep-dependent parameters
        k_t = compute_k_t(sigma, dype_config)
        mscale = compute_mscale(scale_global, sigma, dype_config)

        components = []
        freqs_dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32

        for axis_idx, axis_dim in enumerate(self.axes_dim):
            if axis_idx == 0:
                # Frame axis
                if is_condition:
                    # Condition uses negative index (-1)
                    pos = torch.tensor([-1.0], device=device, dtype=freqs_dtype)
                else:
                    pos = torch.arange(idx, idx + frame, device=device, dtype=freqs_dtype)
                cos, sin = get_1d_ntk_pos_embed(
                    dim=axis_dim,
                    pos=pos,
                    theta=self.theta,
                    ntk_factor=1.0,
                    freqs_dtype=freqs_dtype,
                )
            else:
                # Height/Width axes: use Vision YaRN for extrapolation
                if axis_idx == 1:
                    axis_len = height
                else:
                    axis_len = width

                # Generate position indices with centering
                if self.scale_rope:
                    half = axis_len // 2
                    pos = torch.cat([
                        torch.arange(-half, 0, device=device, dtype=freqs_dtype),
                        torch.arange(0, axis_len - half, device=device, dtype=freqs_dtype),
                    ])
                else:
                    pos = torch.arange(axis_len, device=device, dtype=freqs_dtype)

                linear_scale = max(1.0, float(axis_len) / base_patches)

                if scale_global > 1.0:
                    cos, sin = get_1d_vision_yarn_pos_embed(
                        dim=axis_dim,
                        pos=pos,
                        theta=self.theta,
                        ori_max_pe_len=base_patches,
                        linear_scale=linear_scale,
                        ntk_scale=scale_global,
                        dype=True,
                        dype_scale=dype_config.dype_scale,
                        dype_exponent=dype_config.dype_exponent,
                        current_timestep=sigma,
                        override_mscale=mscale,
                        freqs_dtype=freqs_dtype,
                    )
                else:
                    cos, sin = get_1d_ntk_pos_embed(
                        dim=axis_dim,
                        pos=pos,
                        theta=self.theta,
                        ntk_factor=1.0,
                        freqs_dtype=freqs_dtype,
                    )

            # Convert to complex form
            axis_freqs = torch.polar(cos, sin * math.pi)
            components.append(axis_freqs)

        # Expand and combine
        freqs_frame = components[0].view(frame, 1, 1, -1).expand(frame, height, width, -1)
        freqs_height = components[1].view(1, height, 1, -1).expand(frame, height, width, -1)
        freqs_width = components[2].view(1, 1, width, -1).expand(frame, height, width, -1)

        freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1)
        return freqs.reshape(seq_len, -1).to(device)

    def forward(
        self,
        img_shapes: List[Tuple[int, int, int]],
        txt_seq_lens: List[int],
        device: torch.device,
        sigma: Optional[float] = None,
        dype_config: Optional["DyPEConfig"] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute RoPE embeddings for image layers and text.

        Args:
            img_shapes: List of (frame, height, width) for each layer + condition
            txt_seq_lens: List of text sequence lengths
            device: Device for output tensors
            sigma: Current diffusion timestep (0-1) for DyPE modulation
            dype_config: DyPE configuration (enables DyPE when provided and enabled)

        Returns:
            Tuple of (image_freqs, text_freqs)
        """
        use_dype = (
            dype_config is not None
            and dype_config.enabled
            and sigma is not None
        )

        if self.pos_freqs.device != device:
            self.pos_freqs = self.pos_freqs.to(device)
            self.neg_freqs = self.neg_freqs.to(device)

        vid_freqs = []
        max_vid_index = 0
        layer_num = len(img_shapes) - 1

        for idx, (frame, height, width) in enumerate(img_shapes):
            is_condition = (idx == layer_num)

            if use_dype:
                # DyPE: compute timestep-dependent frequencies
                video_freq = self._compute_dype_freqs(
                    height, width, frame, idx,
                    is_condition, sigma, dype_config, device,
                )
            else:
                # Standard: use cached static frequencies
                if not is_condition:
                    video_freq = self._compute_video_freqs(frame, height, width, idx)
                else:
                    video_freq = self._compute_condition_freqs(frame, height, width)
                video_freq = video_freq.to(device)

            vid_freqs.append(video_freq)

            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)

        max_vid_index = max(max_vid_index, layer_num)
        max_len = max(txt_seq_lens)
        txt_freqs = self.pos_freqs[max_vid_index:max_vid_index + max_len]
        vid_freqs = torch.cat(vid_freqs, dim=0)

        return vid_freqs, txt_freqs


class QwenDoubleStreamAttention(nn.Module):
    """Dual-stream attention for joint image-text processing."""

    def __init__(self, dim_a: int, dim_b: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Image stream
        self.to_q = nn.Linear(dim_a, dim_a)
        self.to_k = nn.Linear(dim_a, dim_a)
        self.to_v = nn.Linear(dim_a, dim_a)
        self.norm_q = RMSNorm(head_dim, eps=1e-6)
        self.norm_k = RMSNorm(head_dim, eps=1e-6)

        # Text stream
        self.add_q_proj = nn.Linear(dim_b, dim_b)
        self.add_k_proj = nn.Linear(dim_b, dim_b)
        self.add_v_proj = nn.Linear(dim_b, dim_b)
        self.norm_added_q = RMSNorm(head_dim, eps=1e-6)
        self.norm_added_k = RMSNorm(head_dim, eps=1e-6)

        # Output projections
        self.to_out = nn.Sequential(nn.Linear(dim_a, dim_a))
        self.to_add_out = nn.Linear(dim_b, dim_b)

    def forward(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute QKV for both streams
        img_q, img_k, img_v = self.to_q(image), self.to_k(image), self.to_v(image)
        txt_q, txt_k, txt_v = self.add_q_proj(text), self.add_k_proj(text), self.add_v_proj(text)
        seq_txt = txt_q.shape[1]

        # Reshape to multi-head
        img_q = rearrange(img_q, "b s (h d) -> b h s d", h=self.num_heads)
        img_k = rearrange(img_k, "b s (h d) -> b h s d", h=self.num_heads)
        img_v = rearrange(img_v, "b s (h d) -> b h s d", h=self.num_heads)
        txt_q = rearrange(txt_q, "b s (h d) -> b h s d", h=self.num_heads)
        txt_k = rearrange(txt_k, "b s (h d) -> b h s d", h=self.num_heads)
        txt_v = rearrange(txt_v, "b s (h d) -> b h s d", h=self.num_heads)

        # Apply RMS norm
        img_q, img_k = self.norm_q(img_q), self.norm_k(img_k)
        txt_q, txt_k = self.norm_added_q(txt_q), self.norm_added_k(txt_k)

        # Apply RoPE
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_q = apply_rotary_emb(img_q, img_freqs)
            img_k = apply_rotary_emb(img_k, img_freqs)
            txt_q = apply_rotary_emb(txt_q, txt_freqs)
            txt_k = apply_rotary_emb(txt_k, txt_freqs)

        # Joint attention
        joint_q = torch.cat([txt_q, img_q], dim=2)
        joint_k = torch.cat([txt_k, img_k], dim=2)
        joint_v = torch.cat([txt_v, img_v], dim=2)

        # Scaled dot-product attention
        out = nn.functional.scaled_dot_product_attention(joint_q, joint_k, joint_v, attn_mask=attention_mask)
        out = rearrange(out, "b h s d -> b s (h d)")

        # Split outputs
        txt_out = out[:, :seq_txt]
        img_out = out[:, seq_txt:]

        # Project outputs
        img_out = self.to_out(img_out)
        txt_out = self.to_add_out(txt_out)

        return img_out, txt_out


class QwenImageTransformerBlock(nn.Module):
    """Single transformer block with dual-stream attention."""

    def __init__(self, dim: int, num_attention_heads: int, attention_head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim

        # Image modulation and norms
        self.img_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = QwenFeedForward(dim=dim, dim_out=dim)

        # Text modulation and norms
        self.txt_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        self.txt_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_mlp = QwenFeedForward(dim=dim, dim_out=dim)

        # Attention
        self.attn = QwenDoubleStreamAttention(
            dim_a=dim,
            dim_b=dim,
            num_heads=num_attention_heads,
            head_dim=attention_head_dim,
        )

    def _modulate(self, x: torch.Tensor, mod_params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shift, scale, gate = mod_params.chunk(3, dim=-1)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1), gate.unsqueeze(1)

    def forward(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute modulation parameters
        img_mod_attn, img_mod_mlp = self.img_mod(temb).chunk(2, dim=-1)
        txt_mod_attn, txt_mod_mlp = self.txt_mod(temb).chunk(2, dim=-1)

        # Attention block
        img_normed = self.img_norm1(image)
        img_modulated, img_gate = self._modulate(img_normed, img_mod_attn)

        txt_normed = self.txt_norm1(text)
        txt_modulated, txt_gate = self._modulate(txt_normed, txt_mod_attn)

        img_attn_out, txt_attn_out = self.attn(
            image=img_modulated,
            text=txt_modulated,
            image_rotary_emb=image_rotary_emb,
            attention_mask=attention_mask,
        )

        image = image + img_gate * img_attn_out
        text = text + txt_gate * txt_attn_out

        # MLP block
        img_normed_2 = self.img_norm2(image)
        img_modulated_2, img_gate_2 = self._modulate(img_normed_2, img_mod_mlp)

        txt_normed_2 = self.txt_norm2(text)
        txt_modulated_2, txt_gate_2 = self._modulate(txt_normed_2, txt_mod_mlp)

        img_mlp_out = self.img_mlp(img_modulated_2)
        txt_mlp_out = self.txt_mlp(txt_modulated_2)

        image = image + img_gate_2 * img_mlp_out
        text = text + txt_gate_2 * txt_mlp_out

        return text, image


class QwenImageDiTModel(nn.Module):
    """Full Qwen-Image DiT model."""

    def __init__(
        self,
        num_layers: int = 60,
        use_layer3d_rope: bool = False,
        use_additional_t_cond: bool = False,
    ):
        super().__init__()
        self._use_additional_t_cond = use_additional_t_cond

        # RoPE embeddings
        if use_layer3d_rope:
            self.pos_embed = QwenEmbedLayer3DRope(theta=10000, axes_dim=[16, 56, 56], scale_rope=True)
        else:
            self.pos_embed = QwenEmbedRope(theta=10000, axes_dim=[16, 56, 56], scale_rope=True)

        # Timestep embedding (with optional additional conditioning)
        self.time_text_embed = TimestepEmbeddings(
            256, 3072, scale=1000, use_additional_t_cond=use_additional_t_cond
        )

        # Input projections
        self.txt_norm = RMSNorm(3584, eps=1e-6)
        self.img_in = nn.Linear(64, 3072)
        self.txt_in = nn.Linear(3584, 3072)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            QwenImageTransformerBlock(
                dim=3072,
                num_attention_heads=24,
                attention_head_dim=128,
            )
            for _ in range(num_layers)
        ])

        # Output
        self.norm_out = AdaLayerNorm(3072, single=True)
        self.proj_out = nn.Linear(3072, 64)

    def forward(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_emb: torch.Tensor,
        prompt_emb_mask: torch.Tensor,
        height: int,
        width: int,
        img_shapes: Optional[List[Tuple[int, int, int]]] = None,
        addition_t_cond: Optional[torch.Tensor] = None,
        sigma: Optional[float] = None,
        dype_config: Optional["DyPEConfig"] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the DiT model.

        Args:
            latents: Input latent tensor
            timestep: Timestep for conditioning
            prompt_emb: Text embeddings
            prompt_emb_mask: Text attention mask
            height: Image height in pixels
            width: Image width in pixels
            img_shapes: Optional RoPE image shapes
            addition_t_cond: Optional additional timestep conditioning
            sigma: Current diffusion timestep (0-1) for DyPE modulation
            dype_config: DyPE configuration (enables DyPE when provided and enabled)

        Returns:
            Noise prediction tensor
        """
        # Compute img_shapes if not provided
        if img_shapes is None:
            img_shapes = [(latents.shape[0], height // 16, width // 16)]

        txt_seq_lens = prompt_emb_mask.sum(dim=1).tolist()

        # Pack latents if in spatial format
        if latents.dim() == 4:
            image = rearrange(
                latents,
                "B C (H P) (W Q) -> B (H W) (C P Q)",
                H=height // 16,
                W=width // 16,
                P=2,
                Q=2,
            )
        else:
            image = latents

        # Project inputs
        image = self.img_in(image)
        text = self.txt_in(self.txt_norm(prompt_emb))

        # Timestep conditioning (with optional additional conditioning)
        # If model has use_additional_t_cond but no value provided, default to 0 (generation mode)
        if self._use_additional_t_cond and addition_t_cond is None:
            addition_t_cond = torch.zeros(1, dtype=torch.long, device=latents.device)
        conditioning = self.time_text_embed(timestep, image.dtype, addition_t_cond=addition_t_cond)

        # Compute RoPE (with optional DyPE modulation)
        image_rotary_emb = self.pos_embed(
            img_shapes,
            txt_seq_lens,
            device=latents.device,
            sigma=sigma,
            dype_config=dype_config,
        )

        # Transformer blocks
        for block in self.transformer_blocks:
            text, image = block(
                image=image,
                text=text,
                temb=conditioning,
                image_rotary_emb=image_rotary_emb,
            )

        # Output projection
        image = self.norm_out(image, conditioning)
        image = self.proj_out(image)

        return image
