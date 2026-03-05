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

import logging
from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from llm_dit.models.ltx2.attention import Attention, AttentionCallable, AttentionFunction

logger = logging.getLogger(__name__)
from llm_dit.layers import rms_norm
from llm_dit.models.ltx2.components import (
    AdaLayerNormSingle,
    FeedForward,
    Modality,
    PixArtAlphaTextProjection,
)
from llm_dit.models.ltx2.rope import (
    LTXRopeType,
    generate_freq_grid_np,
    generate_freq_grid_pytorch,
    precompute_freqs_cis,
)

# Additive attention mask fill value. Valid tokens get 0, padding tokens get
# this large negative value so they contribute ~0 after softmax.
_MASK_FILL_VALUE = -10000.0


class PerturbationType(Enum):
    """Types of attention perturbation for STG (Spatio-Temporal Guidance).

    Each type corresponds to a specific attention operation that can be
    selectively skipped during the perturbed forward pass.
    """
    SKIP_A2V_CROSS_ATTN = "skip_a2v_cross_attn"
    SKIP_V2A_CROSS_ATTN = "skip_v2a_cross_attn"
    SKIP_VIDEO_SELF_ATTN = "skip_video_self_attn"
    SKIP_AUDIO_SELF_ATTN = "skip_audio_self_attn"


@dataclass(frozen=True)
class Perturbation:
    """A single perturbation: skip a specific attention type in specific blocks."""
    type: PerturbationType
    blocks: list[int] | None  # None = all blocks

    def is_perturbed(self, perturbation_type: PerturbationType, block: int) -> bool:
        if self.type != perturbation_type:
            return False
        if self.blocks is None:
            return True
        return block in self.blocks


@dataclass(frozen=True)
class PerturbationConfig:
    """Perturbation configuration for a single sample in a batch."""
    perturbations: list[Perturbation] | None

    def is_perturbed(self, perturbation_type: PerturbationType, block: int) -> bool:
        if self.perturbations is None:
            return False
        return any(p.is_perturbed(perturbation_type, block) for p in self.perturbations)

    @staticmethod
    def empty() -> "PerturbationConfig":
        return PerturbationConfig([])


@dataclass(frozen=True)
class BatchedPerturbationConfig:
    """Perturbation configs for a batch, with mask generation utilities."""
    perturbations: list[PerturbationConfig]

    def mask(
        self, perturbation_type: PerturbationType, block: int,
        device: torch.device, dtype: torch.dtype,
    ) -> torch.Tensor:
        """Generate [B] mask: 1.0 where NOT perturbed, 0.0 where perturbed."""
        mask = torch.ones(len(self.perturbations), device=device, dtype=dtype)
        for idx, pc in enumerate(self.perturbations):
            if pc.is_perturbed(perturbation_type, block):
                mask[idx] = 0.0
        return mask

    def mask_like(
        self, perturbation_type: PerturbationType, block: int,
        values: torch.Tensor,
    ) -> torch.Tensor:
        """Generate broadcastable mask shaped [B, 1, 1, ...] matching values dims."""
        mask = self.mask(perturbation_type, block, values.device, values.dtype)
        return mask.view(mask.numel(), *([1] * len(values.shape[1:])))

    def any_in_batch(self, perturbation_type: PerturbationType, block: int) -> bool:
        return any(pc.is_perturbed(perturbation_type, block) for pc in self.perturbations)

    def all_in_batch(self, perturbation_type: PerturbationType, block: int) -> bool:
        return all(pc.is_perturbed(perturbation_type, block) for pc in self.perturbations)

    @staticmethod
    def empty(batch_size: int) -> "BatchedPerturbationConfig":
        return BatchedPerturbationConfig([PerturbationConfig.empty() for _ in range(batch_size)])


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
    apply_gated_attention: bool = False
    cross_attention_adaln: bool = False


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
    # V2 (LTX-2.3) additions
    prompt_timestep: Optional[torch.Tensor] = None  # Separate timestep for cross-attn AdaLN KV
    self_attention_mask: Optional[torch.Tensor] = None  # Additive log-space bias (B, 1, T, T)


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
        prompt_adaln_single: Optional[AdaLayerNormSingle] = None,
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
        self.prompt_adaln_single = prompt_adaln_single

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

        # NOTE: Do NOT center embeddings before caption_projection!
        # Testing (2026-01-20) showed centering DESTROYS the signal:
        #   - Without centering: std=1.0, blurry but has structure
        #   - With centering: std=0.3, complete noise (deep fried)
        # The per-dim mean offsets ARE the semantic signal, not noise.
        # Original hypothesis was wrong - centering removes 70% of variance.

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[TRANSFORMER] Before caption_projection: shape=%s, mean=%.4f, std=%.4f",
                list(context.shape), context.float().mean(), context.float().std(),
            )

        context = self.caption_projection(context)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[TRANSFORMER] After caption_projection: shape=%s, mean=%.4f, std=%.4f",
                list(context.shape), context.float().mean(), context.float().std(),
            )

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
        return (attention_mask - 1).to(x_dtype).reshape(
            (attention_mask.shape[0], 1, -1, attention_mask.shape[-1])
        ) * abs(_MASK_FILL_VALUE)

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
        batch_size = x.shape[0]
        timestep, embedded_timestep = self._prepare_timestep(
            modality.timesteps,
            batch_size,
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

        # V2: compute prompt_timestep for cross-attention KV-side AdaLN
        prompt_timestep = None
        if self.prompt_adaln_single is not None:
            scaled_ts = modality.timesteps * self.timestep_scale_multiplier
            prompt_timestep, _ = self.prompt_adaln_single(
                scaled_ts.flatten(),
                hidden_dtype=modality.latent.dtype,
            )
            prompt_timestep = prompt_timestep.view(batch_size, -1, prompt_timestep.shape[-1])

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
            prompt_timestep=prompt_timestep,
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
        self.cross_attention_adaln = config.cross_attention_adaln

        # Self-attention
        self.attn1 = Attention(
            query_dim=config.dim,
            heads=config.heads,
            dim_head=config.d_head,
            context_dim=None,  # Self-attention
            rope_type=rope_type,
            norm_eps=norm_eps,
            attention_function=attention_function,
            apply_gated_attention=config.apply_gated_attention,
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
            apply_gated_attention=config.apply_gated_attention,
        )

        # Feed-forward
        self.ff = FeedForward(config.dim, dim_out=config.dim)

        # Scale-shift table for AdaLN:
        # V1: 6 values (shift, scale, gate for self-attn and ff)
        # V2: 9 values (+3 for cross-attention Q-side modulation)
        adaln_params = 9 if config.cross_attention_adaln else 6
        self.scale_shift_table = nn.Parameter(torch.empty(adaln_params, config.dim))

        # V2: per-block KV-side modulation for cross-attention
        if config.cross_attention_adaln:
            self.prompt_scale_shift_table = nn.Parameter(torch.empty(2, config.dim))

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
        skip_self_attn: bool = False,
    ) -> TransformerArgs:
        """
        Forward pass through transformer block.

        Args:
            args: TransformerArgs containing hidden states and conditioning
            skip_self_attn: If True, skip self-attention (used for STG
                perturbed pass). Cross-attention and FFN still run.

        Returns:
            Updated TransformerArgs with transformed hidden states
        """
        x = args.x
        batch_size = x.shape[0]

        # Get AdaLN values for attention
        shift_msa, scale_msa, gate_msa = self.get_ada_values(
            self.scale_shift_table, batch_size, args.timesteps, slice(0, 3)
        )

        # Self-attention with RoPE (skipped during STG perturbed pass)
        if not skip_self_attn:
            norm_x = rms_norm(x, eps=self.norm_eps) * (1 + scale_msa) + shift_msa
            self_attn_out = self.attn1(norm_x, pe=args.positional_embeddings) * gate_msa
            x = x + self_attn_out

        # Cross-attention with text conditioning
        if self.cross_attention_adaln:
            # V2: AdaLN modulation on both Q and KV sides
            from llm_dit.models.ltx2.av_block import _apply_cross_attention_adaln
            shift_q, scale_q, gate_q = self.get_ada_values(
                self.scale_shift_table, batch_size, args.timesteps, slice(6, 9),
            )
            cross_attn_out = _apply_cross_attention_adaln(
                x, args.context, self.attn2,
                shift_q, scale_q, gate_q,
                self.prompt_scale_shift_table,
                args.prompt_timestep,
                args.context_mask, self.norm_eps,
            )
        else:
            # V1: simple cross-attention
            cross_attn_out = self.attn2(
                rms_norm(x, eps=self.norm_eps),
                context=args.context,
                mask=args.context_mask
            )
        x = x + cross_attn_out

        # Get AdaLN values for FFN
        shift_mlp, scale_mlp, gate_mlp = self.get_ada_values(
            self.scale_shift_table, batch_size, args.timesteps, slice(3, 6)
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
        rope_type: LTXRopeType = LTXRopeType.SPLIT,
        double_precision_rope: bool = True,
        # Audio parameters (only used when model_type includes audio)
        audio_num_attention_heads: int = 32,
        audio_attention_head_dim: int = 64,
        audio_in_channels: int = 128,
        audio_out_channels: int = 128,
        audio_cross_attention_dim: int = 2048,
        audio_positional_embedding_max_pos: Optional[list[int]] = None,
        # V2 (LTX-2.3) parameters
        apply_gated_attention: bool = False,
        cross_attention_adaln: bool = False,
        caption_projection_module: Optional[nn.Module] = None,
        audio_caption_projection_module: Optional[nn.Module] = None,
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
        self.cross_attention_adaln = cross_attention_adaln
        self.apply_gated_attention = apply_gated_attention

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
                caption_projection_module=caption_projection_module,
            )

            self._init_preprocessor()

        if model_type.is_audio_enabled():
            if audio_positional_embedding_max_pos is None:
                audio_positional_embedding_max_pos = [20]
            self.audio_positional_embedding_max_pos = audio_positional_embedding_max_pos
            self.audio_num_attention_heads = audio_num_attention_heads
            self.audio_inner_dim = audio_num_attention_heads * audio_attention_head_dim
            self.audio_cross_attention_dim = audio_cross_attention_dim

            self._init_audio(
                in_channels=audio_in_channels,
                out_channels=audio_out_channels,
                caption_channels=caption_channels,
                norm_eps=norm_eps,
                caption_projection_module=audio_caption_projection_module,
            )

            self._init_audio_preprocessor()

        if model_type.is_video_enabled() and model_type.is_audio_enabled():
            self._init_audio_video()
            # Cross-modal PE max position: max of video temporal and audio temporal
            self._cross_pe_max_pos = max(
                self.positional_embedding_max_pos[0],
                self.audio_positional_embedding_max_pos[0],
            )

        self._init_transformer_blocks(
            num_layers=num_layers,
            attention_head_dim=attention_head_dim,
            cross_attention_dim=cross_attention_dim,
            norm_eps=norm_eps,
            attention_type=attention_type,
            audio_attention_head_dim=audio_attention_head_dim if model_type.is_audio_enabled() else None,
            audio_cross_attention_dim=audio_cross_attention_dim if model_type.is_audio_enabled() else None,
        )

    def _adaln_embedding_coefficient(self) -> int:
        """Total AdaLN params per block: 6 base + 3 if cross_attention_adaln."""
        return 6 + (3 if self.cross_attention_adaln else 0)

    def _init_video(
        self,
        in_channels: int,
        out_channels: int,
        caption_channels: int,
        norm_eps: float,
        caption_projection_module: Optional[nn.Module] = None,
    ) -> None:
        """Initialize video-specific components."""
        # Input projection
        self.patchify_proj = nn.Linear(in_channels, self.inner_dim, bias=True)

        # Timestep conditioning
        self.adaln_single = AdaLayerNormSingle(
            self.inner_dim,
            embedding_coefficient=self._adaln_embedding_coefficient(),
        )

        # Text projection -- V2 moved projection to encoder (FeatureExtractorV2)
        if caption_projection_module is not None:
            self.caption_projection = caption_projection_module
        elif self.cross_attention_adaln:
            # V2: encoder already projects to cross_attention_dim, use identity
            self.caption_projection = nn.Identity()
        else:
            self.caption_projection = PixArtAlphaTextProjection(
                in_features=caption_channels,
                hidden_size=self.inner_dim,
            )

        # V2: prompt AdaLN for cross-attention KV modulation
        if self.cross_attention_adaln:
            self.prompt_adaln_single = AdaLayerNormSingle(
                self.inner_dim, embedding_coefficient=2,
            )
        else:
            self.prompt_adaln_single = None

        # Output components
        self.scale_shift_table = nn.Parameter(torch.empty(2, self.inner_dim))
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=norm_eps)
        self.proj_out = nn.Linear(self.inner_dim, out_channels)

    def _init_audio(
        self,
        in_channels: int,
        out_channels: int,
        caption_channels: int,
        norm_eps: float,
        caption_projection_module: Optional[nn.Module] = None,
    ) -> None:
        """Initialize audio-specific components (mirrors _init_video)."""
        self.audio_patchify_proj = nn.Linear(in_channels, self.audio_inner_dim, bias=True)
        self.audio_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=self._adaln_embedding_coefficient(),
        )
        if caption_projection_module is not None:
            self.audio_caption_projection = caption_projection_module
        elif self.cross_attention_adaln:
            self.audio_caption_projection = nn.Identity()
        else:
            self.audio_caption_projection = PixArtAlphaTextProjection(
                in_features=caption_channels,
                hidden_size=self.audio_inner_dim,
            )

        # V2: prompt AdaLN for audio cross-attention KV modulation
        if self.cross_attention_adaln:
            self.audio_prompt_adaln_single = AdaLayerNormSingle(
                self.audio_inner_dim, embedding_coefficient=2,
            )
        else:
            self.audio_prompt_adaln_single = None

        self.audio_scale_shift_table = nn.Parameter(torch.empty(2, self.audio_inner_dim))
        self.audio_norm_out = nn.LayerNorm(
            self.audio_inner_dim, elementwise_affine=False, eps=norm_eps,
        )
        self.audio_proj_out = nn.Linear(self.audio_inner_dim, out_channels)

    def _init_audio_video(self) -> None:
        """Initialize cross-modal AdaLN modules for audio-video interaction."""
        # Cross-attention scale/shift: 4 values (scale_a2v, shift_a2v, scale_v2a, shift_v2a)
        self.av_ca_video_scale_shift_adaln_single = AdaLayerNormSingle(
            self.inner_dim, embedding_coefficient=4,
        )
        self.av_ca_audio_scale_shift_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim, embedding_coefficient=4,
        )
        # Cross-attention gates: 1 value each
        self.av_ca_a2v_gate_adaln_single = AdaLayerNormSingle(
            self.inner_dim, embedding_coefficient=1,
        )
        self.av_ca_v2a_gate_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim, embedding_coefficient=1,
        )

    def _init_preprocessor(self) -> None:
        """Initialize video input preprocessor."""
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
            prompt_adaln_single=self.prompt_adaln_single,
        )

    def _init_audio_preprocessor(self) -> None:
        """Initialize audio input preprocessor (same class, different dims)."""
        self.audio_args_preprocessor = TransformerArgsPreprocessor(
            patchify_proj=self.audio_patchify_proj,
            adaln=self.audio_adaln_single,
            caption_projection=self.audio_caption_projection,
            inner_dim=self.audio_inner_dim,
            max_pos=self.audio_positional_embedding_max_pos,
            num_attention_heads=self.audio_num_attention_heads,
            use_middle_indices_grid=self.use_middle_indices_grid,
            timestep_scale_multiplier=self.timestep_scale_multiplier,
            double_precision_rope=self.double_precision_rope,
            positional_embedding_theta=self.positional_embedding_theta,
            rope_type=self.rope_type,
            prompt_adaln_single=self.audio_prompt_adaln_single,
        )

    def _init_transformer_blocks(
        self,
        num_layers: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        norm_eps: float,
        attention_type: Union[AttentionFunction, AttentionCallable],
        audio_attention_head_dim: Optional[int] = None,
        audio_cross_attention_dim: Optional[int] = None,
    ) -> None:
        """Initialize transformer blocks.

        For VideoOnly: creates BasicTransformerBlock instances (unchanged).
        For AudioVideo: creates BasicAVTransformerBlock with both configs.
        """
        video_config: Optional[TransformerConfig] = None
        if self.model_type.is_video_enabled():
            video_config = TransformerConfig(
                dim=self.inner_dim,
                heads=self.num_attention_heads,
                d_head=attention_head_dim,
                context_dim=cross_attention_dim,
                apply_gated_attention=self.apply_gated_attention,
                cross_attention_adaln=self.cross_attention_adaln,
            )

        audio_config: Optional[TransformerConfig] = None
        if self.model_type.is_audio_enabled():
            assert audio_attention_head_dim is not None
            assert audio_cross_attention_dim is not None
            audio_config = TransformerConfig(
                dim=self.audio_inner_dim,
                heads=self.audio_num_attention_heads,
                d_head=audio_attention_head_dim,
                context_dim=audio_cross_attention_dim,
                apply_gated_attention=self.apply_gated_attention,
                cross_attention_adaln=self.cross_attention_adaln,
            )

        if self.model_type.is_audio_enabled():
            # Deferred import to avoid circular dependency:
            # av_block.py imports TransformerArgs/TransformerConfig from this module
            from llm_dit.models.ltx2.av_block import BasicAVTransformerBlock

            # AV blocks handle both modalities
            self.transformer_blocks = nn.ModuleList([
                BasicAVTransformerBlock(
                    idx=idx,
                    video=video_config,
                    audio=audio_config,
                    rope_type=self.rope_type,
                    norm_eps=norm_eps,
                    attention_function=attention_type,
                )
                for idx in range(num_layers)
            ])
        else:
            # Video-only: use existing lightweight block
            assert video_config is not None
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
        scale_shift_table: nn.Parameter,
        norm_out: nn.LayerNorm,
        proj_out: nn.Linear,
    ) -> torch.Tensor:
        """Apply final output projection with scale-shift modulation.

        Args:
            x: Hidden states [B, T, D]
            embedded_timestep: Timestep embedding [B, T', D]
            scale_shift_table: Parameter [2, D] for output modulation
            norm_out: LayerNorm for output normalization
            proj_out: Linear projection to output channels
        """
        scale_shift_values = (
            scale_shift_table[None, None].to(device=x.device, dtype=x.dtype)
            + embedded_timestep[:, :, None]
        )
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

        x = norm_out(x)
        x = x * (1 + scale) + shift
        x = proj_out(x)

        return x

    def reset_fbcache(self) -> None:
        """Reset FBCache state. Call between generations."""
        self._fbcache_prev_residuals_video: dict[int, float] = {}
        self._fbcache_prev_residuals_audio: dict[int, float] = {}
        self._fbcache_skip_mask: list[bool] = [False] * self.num_layers

    def _prepare_cross_modal_args(
        self,
        video_args: TransformerArgs,
        audio_args: TransformerArgs,
        video: Modality,
        audio: Modality,
    ) -> Tuple[TransformerArgs, TransformerArgs]:
        """Compute cross-modal PE and timestep embeddings for AV attention.

        Cross-PE uses temporal dimension only ([:, 0:1, :] of each modality's
        positions), projected to audio cross-attention dim (2048 = 32 heads x 64).
        Both A2V and V2A attention use audio.heads and audio.d_head, so cross-PE
        must be at audio_cross_attention_dim.

        Cross-timestep embeddings come from the model-level AV AdaLN modules.

        Args:
            video_args: Preprocessed video TransformerArgs
            audio_args: Preprocessed audio TransformerArgs
            video: Video Modality (for positions and timesteps)
            audio: Audio Modality (for positions and timesteps)
        """
        # Cross-modal PE: 1D temporal positions projected to audio cross-attention dim.
        # Both A2V (Q=video, K=audio) and V2A (Q=audio, K=video) use audio heads,
        # so cross-PE must be at audio_cross_attention_dim for both.
        video_temporal_pos = video.positions[:, 0:1, :]   # [B, 1, T_video, 2]
        audio_temporal_pos = audio.positions[:, 0:1, :]   # [B, 1, T_audio, 2]

        video_cross_pe = self.audio_args_preprocessor._prepare_positional_embeddings(
            positions=video_temporal_pos,
            inner_dim=self.audio_cross_attention_dim,
            max_pos=[self._cross_pe_max_pos],
            use_middle_indices_grid=self.use_middle_indices_grid,
            num_attention_heads=self.audio_num_attention_heads,
            x_dtype=video_args.x.dtype,
        )
        audio_cross_pe = self.audio_args_preprocessor._prepare_positional_embeddings(
            positions=audio_temporal_pos,
            inner_dim=self.audio_cross_attention_dim,
            max_pos=[self._cross_pe_max_pos],
            use_middle_indices_grid=self.use_middle_indices_grid,
            num_attention_heads=self.audio_num_attention_heads,
            x_dtype=audio_args.x.dtype,
        )

        batch_size = video_args.x.shape[0]
        hidden_dtype = video_args.x.dtype
        timestep_mult = self.timestep_scale_multiplier

        # Scale raw timesteps (same scaling used by regular AdaLN in preprocessor)
        video_ts_scaled = (video.timesteps * timestep_mult).flatten()  # [B]
        audio_ts_scaled = (audio.timesteps * timestep_mult).flatten()  # [B]

        # Video cross-modal timestep embeddings
        # AdaLayerNormSingle expects raw scalars [B] and handles embedding internally
        v_cross_ss, _ = self.av_ca_video_scale_shift_adaln_single(
            video_ts_scaled, hidden_dtype=hidden_dtype,
        )
        v_cross_ss = v_cross_ss.view(batch_size, -1, v_cross_ss.shape[-1])
        v_cross_gate, _ = self.av_ca_a2v_gate_adaln_single(
            video_ts_scaled, hidden_dtype=hidden_dtype,
        )
        v_cross_gate = v_cross_gate.view(batch_size, -1, v_cross_gate.shape[-1])

        # Audio cross-modal timestep embeddings
        a_cross_ss, _ = self.av_ca_audio_scale_shift_adaln_single(
            audio_ts_scaled, hidden_dtype=hidden_dtype,
        )
        a_cross_ss = a_cross_ss.view(batch_size, -1, a_cross_ss.shape[-1])
        a_cross_gate, _ = self.av_ca_v2a_gate_adaln_single(
            audio_ts_scaled, hidden_dtype=hidden_dtype,
        )
        a_cross_gate = a_cross_gate.view(batch_size, -1, a_cross_gate.shape[-1])

        video_args = replace(
            video_args,
            cross_positional_embeddings=video_cross_pe,
            cross_scale_shift_timestep=v_cross_ss,
            cross_gate_timestep=v_cross_gate,
        )
        audio_args = replace(
            audio_args,
            cross_positional_embeddings=audio_cross_pe,
            cross_scale_shift_timestep=a_cross_ss,
            cross_gate_timestep=a_cross_gate,
        )
        return video_args, audio_args

    def forward(
        self,
        video: Optional[Modality],
        audio: Optional[Modality] = None,
        layer_mask: Optional[torch.Tensor] = None,
        stg_blocks: Optional[set[int]] = None,
        perturbation_config: Optional[BatchedPerturbationConfig] = None,
        fbcache_threshold: float = 0.0,
        step_index: int = 0,
        num_steps: int = 1,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass of LTX-2 transformer.

        Args:
            video: Video modality input (required for VideoOnly model)
            audio: Audio modality input (requires AudioVideo model)
            layer_mask: Optional mask for layer ablation [num_layers] with 0/1
            stg_blocks: Optional set of block indices where video self-attention
                is skipped (for STG perturbed pass, video-only backward compat)
            perturbation_config: Optional per-sample perturbation masks for STG
                (replaces stg_blocks for AV models). Takes precedence if both set.
            fbcache_threshold: L1 residual norm threshold for block skipping.
                0.0 = disabled. For AV models, blocks are only skipped when BOTH
                modalities are below threshold (conservative).
            step_index: Current denoising step index (0-based).
            num_steps: Total number of denoising steps.

        Returns:
            Tuple of (video_output, audio_output) velocity predictions
        """
        if not self.model_type.is_video_enabled() and video is not None:
            raise ValueError("Video is not enabled for this model")
        if audio is not None and not self.model_type.is_audio_enabled():
            raise ValueError("Audio passed to model without audio support")

        if video is None and audio is None:
            return None, None

        # Preprocess inputs
        video_args: Optional[TransformerArgs] = None
        audio_args: Optional[TransformerArgs] = None

        if video is not None:
            video_args = self.args_preprocessor.prepare(video)
        if audio is not None:
            audio_args = self.audio_args_preprocessor.prepare(audio)

        # Cross-modal PE and timestep embeddings for AV models
        if video_args is not None and audio_args is not None:
            assert video is not None and audio is not None
            video_args, audio_args = self._prepare_cross_modal_args(
                video_args, audio_args,
                video=video,
                audio=audio,
            )

        use_fbcache = fbcache_threshold > 0.0
        is_first_step = step_index == 0
        is_last_step = step_index == num_steps - 1

        if use_fbcache and not hasattr(self, '_fbcache_prev_residuals_video'):
            self.reset_fbcache()

        blocks_skipped = 0

        for idx, block in enumerate(self.transformer_blocks):
            if layer_mask is not None and not layer_mask[idx]:
                continue

            # FBCache: skip blocks with low residual change (not on first/last step)
            if (use_fbcache and not is_first_step and not is_last_step
                    and self._fbcache_skip_mask[idx]):
                blocks_skipped += 1
                continue

            if hasattr(block, 'has_audio'):  # BasicAVTransformerBlock
                # AV block: pass both modalities
                vx_before = video_args.x if video_args is not None else None
                ax_before = audio_args.x if audio_args is not None else None

                video_args, audio_args = block(
                    video=video_args,
                    audio=audio_args,
                    perturbation_config=perturbation_config,
                )

                # FBCache: track per-modality residuals, skip only when both below threshold
                if use_fbcache:
                    v_below = True
                    a_below = True
                    if video_args is not None and vx_before is not None:
                        v_norm = (video_args.x - vx_before).abs().mean().item()
                        v_prev = self._fbcache_prev_residuals_video.get(idx)
                        if v_prev is not None:
                            v_below = abs(v_norm - v_prev) < fbcache_threshold
                        else:
                            v_below = False
                        self._fbcache_prev_residuals_video[idx] = v_norm
                    if audio_args is not None and ax_before is not None:
                        a_norm = (audio_args.x - ax_before).abs().mean().item()
                        a_prev = self._fbcache_prev_residuals_audio.get(idx)
                        if a_prev is not None:
                            a_below = abs(a_norm - a_prev) < fbcache_threshold
                        else:
                            a_below = False
                        self._fbcache_prev_residuals_audio[idx] = a_norm
                    self._fbcache_skip_mask[idx] = v_below and a_below
            else:
                # Video-only BasicTransformerBlock
                assert video_args is not None
                skip_self_attn = stg_blocks is not None and idx in stg_blocks
                x_before = video_args.x

                if self._enable_gradient_checkpointing and self.training:
                    video_args = torch.utils.checkpoint.checkpoint(
                        block,
                        video_args,
                        skip_self_attn,
                        use_reentrant=False,
                    )
                else:
                    video_args = block(video_args, skip_self_attn=skip_self_attn)

                if use_fbcache:
                    residual_norm = (video_args.x - x_before).abs().mean().item()
                    prev_norm = self._fbcache_prev_residuals_video.get(idx)
                    if prev_norm is not None:
                        self._fbcache_skip_mask[idx] = abs(residual_norm - prev_norm) < fbcache_threshold
                    else:
                        self._fbcache_skip_mask[idx] = False
                    self._fbcache_prev_residuals_video[idx] = residual_norm

        if use_fbcache and blocks_skipped > 0:
            logger.debug(
                "[FBCache] Step %d: skipped %d/%d blocks", step_index, blocks_skipped, self.num_layers,
            )

        # Final output projections
        video_out: Optional[torch.Tensor] = None
        audio_out: Optional[torch.Tensor] = None

        if video_args is not None:
            video_out = self._process_output(
                video_args.x, video_args.embedded_timestep,
                self.scale_shift_table, self.norm_out, self.proj_out,
            )
        if audio_args is not None:
            audio_out = self._process_output(
                audio_args.x, audio_args.embedded_timestep,
                self.audio_scale_shift_table, self.audio_norm_out, self.audio_proj_out,
            )

        return video_out, audio_out

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

        return self._process_output(
            args.x, args.embedded_timestep,
            self.scale_shift_table, self.norm_out, self.proj_out,
        )

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
