"""
Audio-Video Transformer Block for LTX-2.

Last Updated: 2026-03-05

Implements the BasicAVTransformerBlock that handles video-only, audio-only,
or combined audio-video processing within LTX-2's dual-stream architecture.

Supports both V1 (19B, 6-param AdaLN) and V2 (22B, 9-param AdaLN with
cross_attention_adaln and apply_gated_attention).

When both video and audio configs are provided, the block creates cross-modal
attention modules (Audio-to-Video and Video-to-Audio) that enable the two
streams to exchange information during denoising.

Attribute names match the official LTX-2 implementation for checkpoint
weight compatibility:
  Video branch:  attn1, attn2, ff, scale_shift_table
  Audio branch:  audio_attn1, audio_attn2, audio_ff, audio_scale_shift_table
  Cross-modal:   audio_to_video_attn, video_to_audio_attn,
                 scale_shift_table_a2v_ca_audio, scale_shift_table_a2v_ca_video
  V2 additions:  prompt_scale_shift_table, audio_prompt_scale_shift_table

Ported from: coderef/LTX-2/packages/ltx-core/src/ltx_core/model/transformer/transformer.py
"""

from __future__ import annotations

from dataclasses import replace
from typing import Tuple, Union

import torch
import torch.nn as nn

from llm_dit.layers import rms_norm
from llm_dit.models.ltx2.attention import Attention, AttentionCallable, AttentionFunction
from llm_dit.models.ltx2.components import FeedForward
from llm_dit.models.ltx2.rope import LTXRopeType
from llm_dit.models.ltx2.transformer import (
    BatchedPerturbationConfig,
    PerturbationType,
    TransformerArgs,
    TransformerConfig,
)


def _adaln_size(cross_attention_adaln: bool) -> int:
    """Number of AdaLN params per block: 6 base + 3 for cross-attn AdaLN."""
    return 6 + (3 if cross_attention_adaln else 0)


def _apply_cross_attention_adaln(
    x: torch.Tensor,
    context: torch.Tensor,
    attn: Attention,
    q_shift: torch.Tensor,
    q_scale: torch.Tensor,
    q_gate: torch.Tensor,
    prompt_scale_shift_table: torch.Tensor,
    prompt_timestep: torch.Tensor,
    context_mask: torch.Tensor | None,
    norm_eps: float,
) -> torch.Tensor:
    """Apply cross-attention with AdaLN modulation on both Q and KV sides (V2)."""
    batch_size = x.shape[0]
    shift_kv, scale_kv = (
        prompt_scale_shift_table[None, None].to(device=x.device, dtype=x.dtype)
        + prompt_timestep.reshape(batch_size, prompt_timestep.shape[1], 2, -1)
    ).unbind(dim=2)
    attn_input = rms_norm(x, eps=norm_eps) * (1 + q_scale) + q_shift
    encoder_hidden_states = context * (1 + scale_kv) + shift_kv
    return attn(attn_input, context=encoder_hidden_states, mask=context_mask) * q_gate


class BasicAVTransformerBlock(nn.Module):
    """
    Audio-Video transformer block for LTX-2 (V1 and V2).

    Conditionally creates video, audio, and cross-modal attention branches
    based on which TransformerConfig objects are provided:

    - video only: equivalent to BasicTransformerBlock
    - audio only: same structure but for audio tokens
    - both: adds bidirectional cross-modal attention (A2V, V2A)

    V2 additions (LTX-2.3, 22B):
    - cross_attention_adaln: AdaLN on text cross-attention (scale_shift_table 6->9)
    - apply_gated_attention: per-head sigmoid gate on attention output
    - prompt_scale_shift_table: per-block KV modulation for cross-attention

    Forward order:
    1. Video self-attention + text cross-attention
    2. Audio self-attention + text cross-attention
    3. A2V cross-modal attention (Q=video, K/V=audio)
    4. V2A cross-modal attention (Q=audio, K/V=video)
    5. Video FFN
    6. Audio FFN

    Cross-modal attention uses audio.heads and audio.d_head for both
    directions, keeping the cross-attention in audio dimension space.
    """

    def __init__(
        self,
        idx: int,
        video: TransformerConfig | None = None,
        audio: TransformerConfig | None = None,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        norm_eps: float = 1e-6,
        attention_function: Union[AttentionFunction, AttentionCallable] = AttentionFunction.DEFAULT,
    ):
        super().__init__()
        self.idx = idx
        self.norm_eps = norm_eps
        self.has_video = video is not None
        self.has_audio = audio is not None
        self.has_cross_modal = self.has_video and self.has_audio

        # V2 detection from config
        self.cross_attention_adaln = (
            (video is not None and video.cross_attention_adaln)
            or (audio is not None and audio.cross_attention_adaln)
        )

        # --- Video branch ---
        if video is not None:
            self.attn1 = Attention(
                query_dim=video.dim,
                heads=video.heads,
                dim_head=video.d_head,
                context_dim=None,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
                apply_gated_attention=video.apply_gated_attention,
            )
            self.attn2 = Attention(
                query_dim=video.dim,
                context_dim=video.context_dim,
                heads=video.heads,
                dim_head=video.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
                apply_gated_attention=video.apply_gated_attention,
            )
            self.ff = FeedForward(video.dim, dim_out=video.dim)
            sst_size = _adaln_size(video.cross_attention_adaln)
            self.scale_shift_table = nn.Parameter(torch.empty(sst_size, video.dim))

            # V2: per-block cross-attention KV modulation
            if self.cross_attention_adaln:
                self.prompt_scale_shift_table = nn.Parameter(torch.empty(2, video.dim))

        # --- Audio branch ---
        if audio is not None:
            self.audio_attn1 = Attention(
                query_dim=audio.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                context_dim=None,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
                apply_gated_attention=audio.apply_gated_attention,
            )
            self.audio_attn2 = Attention(
                query_dim=audio.dim,
                context_dim=audio.context_dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
                apply_gated_attention=audio.apply_gated_attention,
            )
            self.audio_ff = FeedForward(audio.dim, dim_out=audio.dim)
            sst_size = _adaln_size(audio.cross_attention_adaln)
            self.audio_scale_shift_table = nn.Parameter(torch.empty(sst_size, audio.dim))

            # V2: per-block cross-attention KV modulation
            if self.cross_attention_adaln:
                self.audio_prompt_scale_shift_table = nn.Parameter(torch.empty(2, audio.dim))

        # --- Cross-modal attention (both modalities present) ---
        if self.has_cross_modal:
            assert video is not None and audio is not None
            # A2V: Q from video, K/V from audio
            self.audio_to_video_attn = Attention(
                query_dim=video.dim,
                context_dim=audio.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
                apply_gated_attention=video.apply_gated_attention,
            )
            # V2A: Q from audio, K/V from video
            self.video_to_audio_attn = Attention(
                query_dim=audio.dim,
                context_dim=video.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
                apply_gated_attention=audio.apply_gated_attention,
            )
            # Cross-modal AdaLN: 5 values each (4 scale/shift + 1 gate)
            self.scale_shift_table_a2v_ca_audio = nn.Parameter(torch.empty(5, audio.dim))
            self.scale_shift_table_a2v_ca_video = nn.Parameter(torch.empty(5, video.dim))

    def get_ada_values(
        self,
        scale_shift_table: torch.Tensor,
        batch_size: int,
        timestep: torch.Tensor,
        indices: slice,
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

    def get_av_ca_ada_values(
        self,
        scale_shift_table: torch.Tensor,
        batch_size: int,
        scale_shift_timestep: torch.Tensor,
        gate_timestep: torch.Tensor,
        num_scale_shift_values: int = 4,
    ) -> Tuple[torch.Tensor, ...]:
        """Extract cross-modal AdaLN values.

        Returns 5 tensors: (scale_a2v, shift_a2v, scale_v2a, shift_v2a, gate)
        """
        scale_shift_ada = self.get_ada_values(
            scale_shift_table[:num_scale_shift_values, :],
            batch_size, scale_shift_timestep, slice(None),
        )
        gate_ada = self.get_ada_values(
            scale_shift_table[num_scale_shift_values:, :],
            batch_size, gate_timestep, slice(None),
        )
        return (*scale_shift_ada, *gate_ada)

    def _apply_text_cross_attention(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        attn: Attention,
        scale_shift_table: torch.Tensor,
        prompt_scale_shift_table: torch.Tensor | None,
        timestep: torch.Tensor,
        prompt_timestep: torch.Tensor | None,
        context_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply text cross-attention, branching on V1 vs V2 AdaLN."""
        if self.cross_attention_adaln:
            assert prompt_scale_shift_table is not None
            assert prompt_timestep is not None
            shift_q, scale_q, gate = self.get_ada_values(
                scale_shift_table, x.shape[0], timestep, slice(6, 9),
            )
            return _apply_cross_attention_adaln(
                x, context, attn,
                shift_q, scale_q, gate,
                prompt_scale_shift_table,
                prompt_timestep,
                context_mask, self.norm_eps,
            )
        # V1: simple cross-attention without AdaLN modulation
        return attn(
            rms_norm(x, eps=self.norm_eps),
            context=context,
            mask=context_mask,
        )

    def forward(
        self,
        video: TransformerArgs | None = None,
        audio: TransformerArgs | None = None,
        perturbation_config: BatchedPerturbationConfig | None = None,
    ) -> Tuple[TransformerArgs | None, TransformerArgs | None]:
        """
        Forward pass through AV transformer block.

        Args:
            video: Video TransformerArgs (hidden states + conditioning)
            audio: Audio TransformerArgs (hidden states + conditioning)
            perturbation_config: Optional per-sample perturbation masks for STG

        Returns:
            Tuple of (updated_video, updated_audio) TransformerArgs
        """
        vx = video.x if video is not None else None
        ax = audio.x if audio is not None else None

        batch_size = (vx if vx is not None else ax).shape[0]  # type: ignore[union-attr]
        if perturbation_config is None:
            perturbation_config = BatchedPerturbationConfig.empty(batch_size)

        run_vx = video is not None and video.enabled and vx is not None and vx.numel() > 0
        run_ax = audio is not None and audio.enabled and ax is not None and ax.numel() > 0
        run_a2v = run_vx and self.has_cross_modal and (audio is not None and ax is not None and ax.numel() > 0)
        run_v2a = run_ax and self.has_cross_modal and (video is not None and vx is not None and vx.numel() > 0)

        # === Video self-attention + text cross-attention ===
        if run_vx:
            assert vx is not None and video is not None
            vshift_msa, vscale_msa, vgate_msa = self.get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps, slice(0, 3),
            )

            if not perturbation_config.all_in_batch(PerturbationType.SKIP_VIDEO_SELF_ATTN, self.idx):
                norm_vx = rms_norm(vx, eps=self.norm_eps) * (1 + vscale_msa) + vshift_msa
                v_mask = perturbation_config.mask_like(PerturbationType.SKIP_VIDEO_SELF_ATTN, self.idx, vx)
                vx = vx + self.attn1(
                    norm_vx,
                    pe=video.positional_embeddings,
                    mask=video.self_attention_mask,
                ) * vgate_msa * v_mask

            vx = vx + self._apply_text_cross_attention(
                vx, video.context, self.attn2,
                self.scale_shift_table,
                getattr(self, "prompt_scale_shift_table", None),
                video.timesteps, video.prompt_timestep,
                video.context_mask,
            )

        # === Audio self-attention + text cross-attention ===
        if run_ax:
            assert ax is not None and audio is not None
            ashift_msa, ascale_msa, agate_msa = self.get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(0, 3),
            )

            if not perturbation_config.all_in_batch(PerturbationType.SKIP_AUDIO_SELF_ATTN, self.idx):
                norm_ax = rms_norm(ax, eps=self.norm_eps) * (1 + ascale_msa) + ashift_msa
                a_mask = perturbation_config.mask_like(PerturbationType.SKIP_AUDIO_SELF_ATTN, self.idx, ax)
                ax = ax + self.audio_attn1(
                    norm_ax,
                    pe=audio.positional_embeddings,
                    mask=audio.self_attention_mask,
                ) * agate_msa * a_mask

            ax = ax + self._apply_text_cross_attention(
                ax, audio.context, self.audio_attn2,
                self.audio_scale_shift_table,
                getattr(self, "audio_prompt_scale_shift_table", None),
                audio.timesteps, audio.prompt_timestep,
                audio.context_mask,
            )

        # === Cross-modal attention (A2V and V2A) ===
        if run_a2v or run_v2a:
            assert vx is not None and ax is not None
            assert video is not None and audio is not None

            # Snapshot normalized states BEFORE cross-modal mutation
            vx_norm = rms_norm(vx, eps=self.norm_eps)
            ax_norm = rms_norm(ax, eps=self.norm_eps)

            # Cross-modal timestep embeddings are required for AV blocks
            assert audio.cross_scale_shift_timestep is not None
            assert audio.cross_gate_timestep is not None
            assert video.cross_scale_shift_timestep is not None
            assert video.cross_gate_timestep is not None

            # AdaLN values for audio side of cross-attention
            (
                scale_ca_audio_a2v, shift_ca_audio_a2v,
                scale_ca_audio_v2a, shift_ca_audio_v2a,
                gate_v2a,
            ) = self.get_av_ca_ada_values(
                self.scale_shift_table_a2v_ca_audio,
                ax.shape[0],
                audio.cross_scale_shift_timestep,
                audio.cross_gate_timestep,
            )

            # AdaLN values for video side of cross-attention
            (
                scale_ca_video_a2v, shift_ca_video_a2v,
                scale_ca_video_v2a, shift_ca_video_v2a,
                gate_a2v,
            ) = self.get_av_ca_ada_values(
                self.scale_shift_table_a2v_ca_video,
                vx.shape[0],
                video.cross_scale_shift_timestep,
                video.cross_gate_timestep,
            )

            # A2V: Q from video, K/V from audio
            if run_a2v:
                vx_scaled = vx_norm * (1 + scale_ca_video_a2v) + shift_ca_video_a2v
                ax_scaled = ax_norm * (1 + scale_ca_audio_a2v) + shift_ca_audio_a2v
                a2v_mask = perturbation_config.mask_like(PerturbationType.SKIP_A2V_CROSS_ATTN, self.idx, vx)
                vx = vx + (
                    self.audio_to_video_attn(
                        vx_scaled,
                        context=ax_scaled,
                        pe=video.cross_positional_embeddings,
                        k_pe=audio.cross_positional_embeddings,
                    )
                    * gate_a2v
                    * a2v_mask
                )

            # V2A: Q from audio, K/V from video
            if run_v2a:
                ax_scaled = ax_norm * (1 + scale_ca_audio_v2a) + shift_ca_audio_v2a
                vx_scaled = vx_norm * (1 + scale_ca_video_v2a) + shift_ca_video_v2a
                v2a_mask = perturbation_config.mask_like(PerturbationType.SKIP_V2A_CROSS_ATTN, self.idx, ax)
                ax = ax + (
                    self.video_to_audio_attn(
                        ax_scaled,
                        context=vx_scaled,
                        pe=audio.cross_positional_embeddings,
                        k_pe=video.cross_positional_embeddings,
                    )
                    * gate_v2a
                    * v2a_mask
                )

        # === Feed-forward networks ===
        if run_vx:
            assert vx is not None and video is not None
            vshift_mlp, vscale_mlp, vgate_mlp = self.get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps, slice(3, 6),
            )
            vx = vx + self.ff(rms_norm(vx, eps=self.norm_eps) * (1 + vscale_mlp) + vshift_mlp) * vgate_mlp

        if run_ax:
            assert ax is not None and audio is not None
            ashift_mlp, ascale_mlp, agate_mlp = self.get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(3, 6),
            )
            ax = ax + self.audio_ff(rms_norm(ax, eps=self.norm_eps) * (1 + ascale_mlp) + ashift_mlp) * agate_mlp

        video_out = replace(video, x=vx) if video is not None else None
        audio_out = replace(audio, x=ax) if audio is not None else None
        return video_out, audio_out
