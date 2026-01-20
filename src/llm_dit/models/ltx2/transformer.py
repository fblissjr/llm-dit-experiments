"""
LTX-2 Transformer Model.

Last Updated: 2026-01-18

Pure PyTorch implementation of the LTX-2 diffusion transformer for video
and audio generation. This implementation gives full control over the
forward pass for research experiments (layer ablation, attention extraction,
routing, etc.) without depending on diffusers internals.

Architecture:
- 48 transformer blocks
- 32 attention heads with 128-dim head (4096 inner dim)
- 3840-dim text conditioning (Gemma3) projected to 4096
- RoPE position embeddings for 3D video (T, H, W)
- AdaLN-single for timestep conditioning

Ported from: coderef/LTX-2/packages/ltx-core/src/ltx_core/model/transformer/

Usage:
    from llm_dit.models.ltx2 import LTX2Transformer, LTXModelType, Modality

    # Create video-only model
    model = LTX2Transformer(model_type=LTXModelType.VideoOnly)

    # Forward pass
    video_output, audio_output = model(
        video=video_modality,
        audio=None,
    )
"""

from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn

from llm_dit.models.ltx2.attention import Attention, AttentionCallable, AttentionFunction
from llm_dit.models.ltx2.components import (
    AdaLayerNormSingle,
    FeedForward,
    Modality,
    PixArtAlphaTextProjection,
    rms_norm,
)
from llm_dit.models.ltx2.rope import (
    LTXRopeType,
    generate_freq_grid_np,
    generate_freq_grid_pytorch,
    precompute_freqs_cis,
)


class LTXModelType(Enum):
    """
    LTX model variant type.

    AudioVideo: Full audio+video model (requires more memory)
    VideoOnly: Video-only model (default for our research)
    AudioOnly: Audio-only model
    """
    AudioVideo = "ltx av model"
    VideoOnly = "ltx video only model"
    AudioOnly = "ltx audio only model"

    def is_video_enabled(self) -> bool:
        return self in (LTXModelType.AudioVideo, LTXModelType.VideoOnly)

    def is_audio_enabled(self) -> bool:
        return self in (LTXModelType.AudioVideo, LTXModelType.AudioOnly)


@dataclass
class TransformerConfig:
    """Configuration for a single modality in transformer blocks."""
    dim: int
    heads: int
    d_head: int
    context_dim: int


@dataclass(frozen=True)
class TransformerArgs:
    """
    Preprocessed arguments for transformer blocks.

    This bundles all the inputs needed for a single modality
    during the forward pass through transformer blocks.
    """
    x: torch.Tensor  # Hidden states [B, T, D]
    context: torch.Tensor  # Text conditioning [B, S, D]
    context_mask: Optional[torch.Tensor]  # Attention mask
    timesteps: torch.Tensor  # AdaLN timestep embeddings
    embedded_timestep: torch.Tensor  # Raw timestep embedding
    positional_embeddings: Tuple[torch.Tensor, torch.Tensor]  # RoPE (cos, sin)
    cross_positional_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]]
    cross_scale_shift_timestep: Optional[torch.Tensor]
    cross_gate_timestep: Optional[torch.Tensor]
    enabled: bool


class TransformerArgsPreprocessor:
    """
    Preprocesses modality inputs into TransformerArgs.

    Handles:
    - Latent patchification (projection to inner dim)
    - Timestep embedding and AdaLN computation
    - Text projection and masking
    - RoPE position embedding computation
    """

    def __init__(
        self,
        patchify_proj: nn.Linear,
        adaln: AdaLayerNormSingle,
        caption_projection: PixArtAlphaTextProjection,
        inner_dim: int,
        max_pos: list[int],
        num_attention_heads: int,
        use_middle_indices_grid: bool,
        timestep_scale_multiplier: int,
        double_precision_rope: bool,
        positional_embedding_theta: float,
        rope_type: LTXRopeType,
    ) -> None:
        self.patchify_proj = patchify_proj
        self.adaln = adaln
        self.caption_projection = caption_projection
        self.inner_dim = inner_dim
        self.max_pos = max_pos
        self.num_attention_heads = num_attention_heads
        self.use_middle_indices_grid = use_middle_indices_grid
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.double_precision_rope = double_precision_rope
        self.positional_embedding_theta = positional_embedding_theta
        self.rope_type = rope_type

    def _prepare_timestep(
        self,
        timestep: torch.Tensor,
        batch_size: int,
        hidden_dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare timestep embeddings."""
        timestep = timestep * self.timestep_scale_multiplier
        timestep, embedded_timestep = self.adaln(
            timestep.flatten(),
            hidden_dtype=hidden_dtype,
        )

        # Handle per-token timesteps
        timestep = timestep.view(batch_size, -1, timestep.shape[-1])
        embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.shape[-1])
        return timestep, embedded_timestep

    def _prepare_context(
        self,
        context: torch.Tensor,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Prepare context (text conditioning) for transformer blocks."""
        batch_size = x.shape[0]

        # FIX: Center embeddings per-dimension before caption_projection
        # Our Gemma encoder outputs embeddings with large per-dim mean offsets
        # (range [-8.7, 12.4]) which causes GELU to crush variance after linear_1
        # Centering prevents the negative mean shift that triggers GELU's dead zone
        context = context - context.mean(dim=1, keepdim=True)

        # DEBUG: Check context BEFORE caption_projection
        if hasattr(self, '_debug_context') and self._debug_context:
            print(f"[DEBUG CONTEXT] Before projection: mean={context.mean():.4f}, std={context.std():.4f}")
            per_dim_mean = context.mean(dim=(0, 1))
            print(f"[DEBUG CONTEXT] Per-dim mean range: [{per_dim_mean.min():.4f}, {per_dim_mean.max():.4f}]")

        context = self.caption_projection(context)

        # DEBUG: Check context AFTER caption_projection
        if hasattr(self, '_debug_context') and self._debug_context:
            print(f"[DEBUG CONTEXT] After projection: mean={context.mean():.4f}, std={context.std():.4f}")
            self._debug_context = False  # Only print once

        context = context.view(batch_size, -1, x.shape[-1])
        return context, attention_mask

    def _prepare_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        x_dtype: torch.dtype
    ) -> Optional[torch.Tensor]:
        """Convert boolean mask to float mask."""
        if attention_mask is None or torch.is_floating_point(attention_mask):
            return attention_mask

        # Convert boolean mask to additive attention mask: 0=attend, -10000=ignore
        # Using -10000.0 like official LTX-2 (not finfo.max which can cause precision issues)
        return (attention_mask - 1).to(x_dtype).reshape(
            (attention_mask.shape[0], 1, -1, attention_mask.shape[-1])
        ) * 10000.0  # Results in 0 for valid, -10000 for padding

    def _prepare_positional_embeddings(
        self,
        positions: torch.Tensor,
        inner_dim: int,
        max_pos: list[int],
        use_middle_indices_grid: bool,
        num_attention_heads: int,
        x_dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute RoPE position embeddings."""
        freq_grid_generator = generate_freq_grid_np if self.double_precision_rope else generate_freq_grid_pytorch
        pe = precompute_freqs_cis(
            positions,
            dim=inner_dim,
            out_dtype=x_dtype,
            theta=self.positional_embedding_theta,
            max_pos=max_pos,
            use_middle_indices_grid=use_middle_indices_grid,
            num_attention_heads=num_attention_heads,
            rope_type=self.rope_type,
            freq_grid_generator=freq_grid_generator,
        )
        return pe

    def prepare(self, modality: Modality) -> TransformerArgs:
        """Preprocess modality into transformer arguments."""
        x = self.patchify_proj(modality.latent)
        timestep, embedded_timestep = self._prepare_timestep(
            modality.timesteps,
            x.shape[0],
            modality.latent.dtype
        )
        context, attention_mask = self._prepare_context(
            modality.context,
            x,
            modality.context_mask
        )
        attention_mask = self._prepare_attention_mask(attention_mask, modality.latent.dtype)
        pe = self._prepare_positional_embeddings(
            positions=modality.positions,
            inner_dim=self.inner_dim,
            max_pos=self.max_pos,
            use_middle_indices_grid=self.use_middle_indices_grid,
            num_attention_heads=self.num_attention_heads,
            x_dtype=modality.latent.dtype,
        )

        return TransformerArgs(
            x=x,
            context=context,
            context_mask=attention_mask,
            timesteps=timestep,
            embedded_timestep=embedded_timestep,
            positional_embeddings=pe,
            cross_positional_embeddings=None,
            cross_scale_shift_timestep=None,
            cross_gate_timestep=None,
            enabled=modality.enabled,
        )


class BasicTransformerBlock(nn.Module):
    """
    Single transformer block for video-only LTX-2.

    Structure:
    1. Self-attention (attn1) with RoPE and AdaLN
    2. Cross-attention (attn2) with text conditioning
    3. Feed-forward (ff) with AdaLN

    Each component uses scale-shift modulation from timestep embeddings.
    """

    def __init__(
        self,
        idx: int,
        config: TransformerConfig,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        norm_eps: float = 1e-6,
        attention_function: Union[AttentionFunction, AttentionCallable] = AttentionFunction.DEFAULT,
    ):
        super().__init__()
        self.idx = idx
        self.norm_eps = norm_eps

        # Self-attention
        self.attn1 = Attention(
            query_dim=config.dim,
            heads=config.heads,
            dim_head=config.d_head,
            context_dim=None,  # Self-attention
            rope_type=rope_type,
            norm_eps=norm_eps,
            attention_function=attention_function,
        )

        # Cross-attention
        self.attn2 = Attention(
            query_dim=config.dim,
            context_dim=config.context_dim,
            heads=config.heads,
            dim_head=config.d_head,
            rope_type=rope_type,
            norm_eps=norm_eps,
            attention_function=attention_function,
        )

        # Feed-forward
        self.ff = FeedForward(config.dim, dim_out=config.dim)

        # Scale-shift table for AdaLN: 6 values (shift, scale, gate for attn and ff)
        self.scale_shift_table = nn.Parameter(torch.empty(6, config.dim))

    def get_ada_values(
        self,
        scale_shift_table: torch.Tensor,
        batch_size: int,
        timestep: torch.Tensor,
        indices: slice
    ) -> Tuple[torch.Tensor, ...]:
        """Extract adaptive normalization values from timestep embedding."""
        num_ada_params = scale_shift_table.shape[0]

        ada_values = (
            scale_shift_table[indices].unsqueeze(0).unsqueeze(0).to(
                device=timestep.device, dtype=timestep.dtype
            )
            + timestep.reshape(batch_size, timestep.shape[1], num_ada_params, -1)[:, :, indices, :]
        ).unbind(dim=2)

        return ada_values

    def forward(
        self,
        args: TransformerArgs,
    ) -> TransformerArgs:
        """
        Forward pass through transformer block.

        Args:
            args: TransformerArgs containing hidden states and conditioning

        Returns:
            Updated TransformerArgs with transformed hidden states
        """
        x = args.x
        batch_size = x.shape[0]

        # Get AdaLN values for attention
        shift_msa, scale_msa, gate_msa = self.get_ada_values(
            self.scale_shift_table, batch_size, args.timesteps, slice(0, 3)
        )

        # DEBUG: Track variance at each stage in block 0
        _debug_block0 = self.idx == 0 and hasattr(self, '_debug_step') and self._debug_step in [0, 20, 39]
        if _debug_block0:
            x_in_inter = x.std(dim=1).mean().item()
            print(f"[VARIANCE TRACE] Block 0, Step {self._debug_step}:")
            print(f"  1. x input inter-token std: {x_in_inter:.4f}")

        # Self-attention with RoPE
        norm_x = rms_norm(x, eps=self.norm_eps) * (1 + scale_msa) + shift_msa

        if _debug_block0:
            norm_x_inter = norm_x.std(dim=1).mean().item()
            print(f"  2. after RMSNorm+AdaLN inter-token std: {norm_x_inter:.4f}")

        self_attn_out = self.attn1(norm_x, pe=args.positional_embeddings) * gate_msa

        if _debug_block0:
            self_attn_inter = self_attn_out.std(dim=1).mean().item()
            gate_mean = gate_msa.mean().item()
            print(f"  3. self_attn_out*gate inter-token std: {self_attn_inter:.4f} (gate_mean={gate_mean:.4f})")

        x = x + self_attn_out

        if _debug_block0:
            x_post_self_inter = x.std(dim=1).mean().item()
            print(f"  4. x after self-attn residual inter-token std: {x_post_self_inter:.4f}")

        # Cross-attention with text conditioning
        norm_x_cross = rms_norm(x, eps=self.norm_eps)

        if _debug_block0:
            norm_x_cross_inter = norm_x_cross.std(dim=1).mean().item()
            print(f"  5. after 2nd RMSNorm (cross-attn input) inter-token std: {norm_x_cross_inter:.4f}")

        cross_attn_out = self.attn2(
            norm_x_cross,
            context=args.context,
            mask=args.context_mask
        )
        x = x + cross_attn_out

        # DEBUG: Log attention magnitudes for diagnostic blocks
        if self.idx in [0, 23, 47] and hasattr(self, '_debug_step'):
            if self._debug_step in [0, 20, 39]:
                x_inter_token = x.std(dim=1).mean()  # Variation across tokens
                self_attn_inter = self_attn_out.std(dim=1).mean()
                print(f"[ATTN DEBUG] Block {self.idx}, Step {self._debug_step}:")
                print(f"  x inter-token std: {x_inter_token:.6f}")
                print(f"  self_attn inter-token std: {self_attn_inter:.6f}")
                print(f"  self_attn: mean={self_attn_out.mean():.6f}, std={self_attn_out.std():.6f}")
                print(f"  cross_attn: mean={cross_attn_out.mean():.6f}, std={cross_attn_out.std():.6f}")

        # Get AdaLN values for FFN
        shift_mlp, scale_mlp, gate_mlp = self.get_ada_values(
            self.scale_shift_table, batch_size, args.timesteps, slice(3, None)
        )

        # Feed-forward
        x_scaled = rms_norm(x, eps=self.norm_eps) * (1 + scale_mlp) + shift_mlp
        x = x + self.ff(x_scaled) * gate_mlp

        return replace(args, x=x)


class LTX2Transformer(nn.Module):
    """
    LTX-2 Diffusion Transformer.

    Pure PyTorch implementation of the LTX-2 video generation model.
    Provides full control over the forward pass for research experiments.

    Default configuration (VideoOnly):
    - 48 transformer blocks
    - 32 attention heads x 128 head_dim = 4096 inner dim
    - 3840 caption channels (Gemma3) -> 4096 projected
    - 128 in/out channels (VAE latent dim)
    - RoPE with max positions [20, 2048, 2048] for T, H, W

    Args:
        model_type: Which variant to create (VideoOnly, AudioVideo, AudioOnly)
        num_attention_heads: Number of attention heads
        attention_head_dim: Dimension per attention head
        in_channels: Input latent channels (from VAE)
        out_channels: Output channels (velocity prediction)
        num_layers: Number of transformer blocks
        cross_attention_dim: Text conditioning dimension (after projection)
        norm_eps: Epsilon for normalization
        attention_type: Which attention backend to use
        caption_channels: Text encoder output dimension (before projection)
        positional_embedding_theta: RoPE base frequency
        positional_embedding_max_pos: Max positions for T, H, W
        timestep_scale_multiplier: Scaling for timestep input
        use_middle_indices_grid: Use middle of position range
        rope_type: RoPE variant (INTERLEAVED or SPLIT)
        double_precision_rope: Use numpy for RoPE computation
    """

    def __init__(
        self,
        *,
        model_type: LTXModelType = LTXModelType.VideoOnly,
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        in_channels: int = 128,
        out_channels: int = 128,
        num_layers: int = 48,
        cross_attention_dim: int = 4096,
        norm_eps: float = 1e-06,
        attention_type: Union[AttentionFunction, AttentionCallable] = AttentionFunction.DEFAULT,
        caption_channels: int = 3840,
        positional_embedding_theta: float = 10000.0,
        positional_embedding_max_pos: Optional[list[int]] = None,
        timestep_scale_multiplier: int = 1000,
        use_middle_indices_grid: bool = True,
        rope_type: LTXRopeType = LTXRopeType.SPLIT,  # Official LTX-2 config.json uses "split"
        double_precision_rope: bool = True,  # Official LTX-2 config.json uses true
    ):
        super().__init__()

        self._enable_gradient_checkpointing = False
        self.use_middle_indices_grid = use_middle_indices_grid
        self.rope_type = rope_type
        self.double_precision_rope = double_precision_rope
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.positional_embedding_theta = positional_embedding_theta
        self.model_type = model_type
        self.num_layers = num_layers

        if model_type.is_video_enabled():
            if positional_embedding_max_pos is None:
                positional_embedding_max_pos = [20, 2048, 2048]
            self.positional_embedding_max_pos = positional_embedding_max_pos
            self.num_attention_heads = num_attention_heads
            self.inner_dim = num_attention_heads * attention_head_dim

            self._init_video(
                in_channels=in_channels,
                out_channels=out_channels,
                caption_channels=caption_channels,
                norm_eps=norm_eps,
            )

            self._init_preprocessor()

            self._init_transformer_blocks(
                num_layers=num_layers,
                attention_head_dim=attention_head_dim,
                cross_attention_dim=cross_attention_dim,
                norm_eps=norm_eps,
                attention_type=attention_type,
            )

    def _init_video(
        self,
        in_channels: int,
        out_channels: int,
        caption_channels: int,
        norm_eps: float,
    ) -> None:
        """Initialize video-specific components."""
        # Input projection
        self.patchify_proj = nn.Linear(in_channels, self.inner_dim, bias=True)

        # Timestep conditioning
        self.adaln_single = AdaLayerNormSingle(self.inner_dim)

        # Text projection
        self.caption_projection = PixArtAlphaTextProjection(
            in_features=caption_channels,
            hidden_size=self.inner_dim,
        )

        # Output components
        self.scale_shift_table = nn.Parameter(torch.empty(2, self.inner_dim))
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=norm_eps)
        self.proj_out = nn.Linear(self.inner_dim, out_channels)

    def _init_preprocessor(self) -> None:
        """Initialize input preprocessor."""
        self.args_preprocessor = TransformerArgsPreprocessor(
            patchify_proj=self.patchify_proj,
            adaln=self.adaln_single,
            caption_projection=self.caption_projection,
            inner_dim=self.inner_dim,
            max_pos=self.positional_embedding_max_pos,
            num_attention_heads=self.num_attention_heads,
            use_middle_indices_grid=self.use_middle_indices_grid,
            timestep_scale_multiplier=self.timestep_scale_multiplier,
            double_precision_rope=self.double_precision_rope,
            positional_embedding_theta=self.positional_embedding_theta,
            rope_type=self.rope_type,
        )

    def _init_transformer_blocks(
        self,
        num_layers: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        norm_eps: float,
        attention_type: Union[AttentionFunction, AttentionCallable],
    ) -> None:
        """Initialize transformer blocks."""
        video_config = TransformerConfig(
            dim=self.inner_dim,
            heads=self.num_attention_heads,
            d_head=attention_head_dim,
            context_dim=cross_attention_dim,
        )

        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                idx=idx,
                config=video_config,
                rope_type=self.rope_type,
                norm_eps=norm_eps,
                attention_function=attention_type,
            )
            for idx in range(num_layers)
        ])

    def set_gradient_checkpointing(self, enable: bool) -> None:
        """Enable or disable gradient checkpointing."""
        self._enable_gradient_checkpointing = enable

    def _process_output(
        self,
        x: torch.Tensor,
        embedded_timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Apply final output projection with scale-shift modulation."""
        # Compute scale-shift from timestep
        scale_shift_values = (
            self.scale_shift_table[None, None].to(device=x.device, dtype=x.dtype)
            + embedded_timestep[:, :, None]
        )
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

        # Apply modulation and project
        x = self.norm_out(x)
        x = x * (1 + scale) + shift
        x = self.proj_out(x)

        return x

    def forward(
        self,
        video: Optional[Modality],
        audio: Optional[Modality] = None,
        layer_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass of LTX-2 transformer.

        Args:
            video: Video modality input (required for VideoOnly model)
            audio: Audio modality input (not used in VideoOnly)
            layer_mask: Optional mask for layer ablation [num_layers] with 0/1

        Returns:
            Tuple of (video_output, audio_output) velocity predictions
        """
        if not self.model_type.is_video_enabled() and video is not None:
            raise ValueError("Video is not enabled for this model")

        if video is None:
            return None, None

        # Preprocess inputs
        args = self.args_preprocessor.prepare(video)

        # Process through transformer blocks
        for idx, block in enumerate(self.transformer_blocks):
            # Optional layer masking for ablation
            if layer_mask is not None and not layer_mask[idx]:
                continue

            if self._enable_gradient_checkpointing and self.training:
                args = torch.utils.checkpoint.checkpoint(
                    block,
                    args,
                    use_reentrant=False,
                )
            else:
                args = block(args)

        # Final output projection
        video_out = self._process_output(args.x, args.embedded_timestep)

        return video_out, None

    def forward_with_layer_weights(
        self,
        video: Modality,
        layer_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with per-layer weighting (for routing research).

        This allows soft layer ablation by weighting each block's contribution.

        Args:
            video: Video modality input
            layer_weights: [num_layers] weights for each block output

        Returns:
            Video output velocity prediction
        """
        if layer_weights.shape[0] != self.num_layers:
            raise ValueError(
                f"layer_weights has {layer_weights.shape[0]} elements, "
                f"expected {self.num_layers}"
            )

        args = self.args_preprocessor.prepare(video)

        for idx, block in enumerate(self.transformer_blocks):
            weight = layer_weights[idx]

            if weight == 0:
                continue

            if self._enable_gradient_checkpointing and self.training:
                new_args = torch.utils.checkpoint.checkpoint(
                    block,
                    args,
                    use_reentrant=False,
                )
            else:
                new_args = block(args)

            # Weighted residual
            if weight != 1.0:
                weighted_x = args.x + weight * (new_args.x - args.x)
                args = replace(new_args, x=weighted_x)
            else:
                args = new_args

        return self._process_output(args.x, args.embedded_timestep)

    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def to_velocity(
    sample: torch.Tensor,
    sigma: float | torch.Tensor,
    denoised_sample: torch.Tensor,
    calc_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Convert sample and denoised to velocity.

    velocity = (sample - denoised) / sigma
    """
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.to(calc_dtype).item()
    if sigma == 0:
        raise ValueError("Sigma can't be 0.0")
    return ((sample.to(calc_dtype) - denoised_sample.to(calc_dtype)) / sigma).to(sample.dtype)


def to_denoised(
    sample: torch.Tensor,
    velocity: torch.Tensor,
    sigma: float | torch.Tensor,
    calc_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Convert sample and velocity to denoised sample.

    denoised = sample - velocity * sigma
    """
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.to(calc_dtype)
    return (sample.to(calc_dtype) - velocity.to(calc_dtype) * sigma).to(sample.dtype)
