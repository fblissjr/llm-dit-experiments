"""
Z-Image DiT Transformer - Pure PyTorch Implementation.

Last updated: 2026-02-01

Implements the S3-DiT (Scalable Sequence-to-Sequence DiT) architecture
used by Z-Image for text-to-image generation.

Key features:
- Per-token AdaLN modulation (supports both basic and Omni modes)
- Separate noise/context refiners before main transformer
- 3-axis RoPE for joint text+spatial position encoding
- SwiGLU feedforward networks

Based on DiffSynth-Studio implementation.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from llm_dit.layers import RMSNorm
from .attention import Attention
from .components import (
    FeedForward,
    FinalLayer,
    TimestepEmbedder,
    select_per_token,
)
from .constants import ADALN_EMBED_DIM, SEQ_MULTI_OF, X_PAD_DIM
from .rope import RopeEmbedder


class ZImageTransformerBlock(nn.Module):
    """
    Single transformer block with AdaLN modulation.

    Implements pre-norm transformer with adaptive layer normalization
    for timestep conditioning. Supports both global modulation (basic mode)
    and per-token modulation (Omni mode).

    Args:
        layer_id: Layer index (for logging/debugging)
        dim: Hidden dimension
        n_heads: Number of attention heads
        n_kv_heads: Number of KV heads (same as n_heads for MHA)
        norm_eps: Epsilon for layer normalization
        qk_norm: Whether to use QK normalization
        modulation: Whether to use AdaLN modulation
    """

    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
        qk_norm: bool,
        modulation: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads
        self.layer_id = layer_id

        # Self-attention
        self.attention = Attention(
            q_dim=dim,
            num_heads=n_heads,
            head_dim=dim // n_heads,
        )

        # Feedforward (SwiGLU with 8/3 expansion)
        self.feed_forward = FeedForward(dim=dim, hidden_dim=int(dim / 3 * 8))

        # Normalization layers (pre-norm style)
        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)
        self.attention_norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)

        # AdaLN modulation
        self.modulation = modulation
        if modulation:
            self.adaLN_modulation = nn.Sequential(
                nn.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True),
            )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: Optional[torch.Tensor] = None,
        noise_mask: Optional[torch.Tensor] = None,
        adaln_noisy: Optional[torch.Tensor] = None,
        adaln_clean: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through transformer block.

        Args:
            x: Input tensor (batch, seq_len, dim)
            attn_mask: Attention mask (batch, seq_len)
            freqs_cis: RoPE frequencies (seq_len, head_dim/2)
            adaln_input: Global AdaLN input (batch, adaln_dim) - basic mode
            noise_mask: Per-token noise indicator (batch, seq_len) - Omni mode
            adaln_noisy: Noisy token AdaLN (batch, adaln_dim) - Omni mode
            adaln_clean: Clean token AdaLN (batch, adaln_dim) - Omni mode

        Returns:
            Output tensor (batch, seq_len, dim)
        """
        if self.modulation:
            seq_len = x.shape[1]

            if noise_mask is not None:
                # Per-token modulation: different modulation for noisy/clean tokens
                mod_noisy = self.adaLN_modulation(adaln_noisy)
                mod_clean = self.adaLN_modulation(adaln_clean)

                scale_msa_noisy, gate_msa_noisy, scale_mlp_noisy, gate_mlp_noisy = mod_noisy.chunk(4, dim=1)
                scale_msa_clean, gate_msa_clean, scale_mlp_clean, gate_mlp_clean = mod_clean.chunk(4, dim=1)

                gate_msa_noisy, gate_mlp_noisy = gate_msa_noisy.tanh(), gate_mlp_noisy.tanh()
                gate_msa_clean, gate_mlp_clean = gate_msa_clean.tanh(), gate_mlp_clean.tanh()

                scale_msa_noisy, scale_mlp_noisy = 1.0 + scale_msa_noisy, 1.0 + scale_mlp_noisy
                scale_msa_clean, scale_mlp_clean = 1.0 + scale_msa_clean, 1.0 + scale_mlp_clean

                scale_msa = select_per_token(scale_msa_noisy, scale_msa_clean, noise_mask, seq_len)
                scale_mlp = select_per_token(scale_mlp_noisy, scale_mlp_clean, noise_mask, seq_len)
                gate_msa = select_per_token(gate_msa_noisy, gate_msa_clean, noise_mask, seq_len)
                gate_mlp = select_per_token(gate_mlp_noisy, gate_mlp_clean, noise_mask, seq_len)
            else:
                # Global modulation: same modulation for all tokens
                mod = self.adaLN_modulation(adaln_input)
                scale_msa, gate_msa, scale_mlp, gate_mlp = mod.unsqueeze(1).chunk(4, dim=2)
                gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
                scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

            # Attention block
            attn_out = self.attention(
                self.attention_norm1(x) * scale_msa,
                freqs_cis=freqs_cis,
                attention_mask=attn_mask,
            )
            x = x + gate_msa * self.attention_norm2(attn_out)

            # FFN block
            x = x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))
        else:
            # No modulation (for context refiner)
            attn_out = self.attention(
                self.attention_norm1(x),
                freqs_cis=freqs_cis,
                attention_mask=attn_mask,
            )
            x = x + self.attention_norm2(attn_out)

            # FFN block
            x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))

        return x


class ZImageDiT(nn.Module):
    """
    Z-Image DiT Transformer (S3-DiT Architecture).

    Pure PyTorch implementation of the Z-Image text-to-image transformer.

    Architecture:
    1. Patchify image to tokens
    2. Refine noise tokens (noise_refiner)
    3. Embed and refine caption tokens (context_refiner)
    4. Concatenate into unified sequence
    5. Process through main transformer layers
    6. Unpatchify back to image

    Args:
        all_patch_size: Supported patch sizes (default: (2,))
        all_f_patch_size: Frame patch sizes (default: (1,) for images)
        in_channels: VAE latent channels (default: 16)
        dim: Hidden dimension (default: 3840)
        n_layers: Number of main transformer layers (default: 30)
        n_refiner_layers: Number of refiner layers (default: 2)
        n_heads: Number of attention heads (default: 30)
        n_kv_heads: Number of KV heads (default: 30)
        norm_eps: Layer norm epsilon (default: 1e-5)
        qk_norm: Use QK normalization (default: True)
        cap_feat_dim: Caption feature dimension (default: 2560)
        rope_theta: RoPE base frequency (default: 256.0)
        t_scale: Timestep scaling (default: 1000.0)
        axes_dims: RoPE axis dimensions (default: [32, 48, 48])
        axes_lens: RoPE axis lengths (default: [1024, 512, 512])
        siglip_feat_dim: SigLIP feature dim for Omni mode (default: None)
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["ZImageTransformerBlock"]

    def __init__(
        self,
        all_patch_size: Tuple[int, ...] = (2,),
        all_f_patch_size: Tuple[int, ...] = (1,),
        in_channels: int = 16,
        dim: int = 3840,
        n_layers: int = 30,
        n_refiner_layers: int = 2,
        n_heads: int = 30,
        n_kv_heads: int = 30,
        norm_eps: float = 1e-5,
        qk_norm: bool = True,
        cap_feat_dim: int = 2560,
        rope_theta: float = 256.0,
        t_scale: float = 1000.0,
        axes_dims: List[int] = [32, 48, 48],
        axes_lens: List[int] = [1024, 512, 512],
        siglip_feat_dim: Optional[int] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.all_patch_size = all_patch_size
        self.all_f_patch_size = all_f_patch_size
        self.dim = dim
        self.n_heads = n_heads

        self.rope_theta = rope_theta
        self.t_scale = t_scale
        self.gradient_checkpointing = False

        assert len(all_patch_size) == len(all_f_patch_size)

        # Patchify embedders and final layers for each patch size
        all_x_embedder = {}
        all_final_layer = {}
        for patch_size, f_patch_size in zip(all_patch_size, all_f_patch_size):
            x_embedder = nn.Linear(
                f_patch_size * patch_size * patch_size * in_channels,
                dim,
                bias=True,
            )
            all_x_embedder[f"{patch_size}-{f_patch_size}"] = x_embedder

            final_layer = FinalLayer(
                dim,
                patch_size * patch_size * f_patch_size * self.out_channels,
            )
            all_final_layer[f"{patch_size}-{f_patch_size}"] = final_layer

        self.all_x_embedder = nn.ModuleDict(all_x_embedder)
        self.all_final_layer = nn.ModuleDict(all_final_layer)

        # Noise refiner (for image tokens, with modulation)
        self.noise_refiner = nn.ModuleList([
            ZImageTransformerBlock(
                1000 + layer_id,
                dim,
                n_heads,
                n_kv_heads,
                norm_eps,
                qk_norm,
                modulation=True,
            )
            for layer_id in range(n_refiner_layers)
        ])

        # Context refiner (for caption tokens, no modulation)
        self.context_refiner = nn.ModuleList([
            ZImageTransformerBlock(
                layer_id,
                dim,
                n_heads,
                n_kv_heads,
                norm_eps,
                qk_norm,
                modulation=False,
            )
            for layer_id in range(n_refiner_layers)
        ])

        # Timestep embedder
        self.t_embedder = TimestepEmbedder(min(dim, ADALN_EMBED_DIM), mid_size=1024)

        # Caption embedder
        self.cap_embedder = nn.Sequential(
            RMSNorm(cap_feat_dim, eps=norm_eps),
            nn.Linear(cap_feat_dim, dim, bias=True),
        )

        # Optional SigLIP components (Omni mode)
        self.siglip_feat_dim = siglip_feat_dim
        if siglip_feat_dim is not None:
            self.siglip_embedder = nn.Sequential(
                RMSNorm(siglip_feat_dim, eps=norm_eps),
                nn.Linear(siglip_feat_dim, dim, bias=True),
            )
            self.siglip_refiner = nn.ModuleList([
                ZImageTransformerBlock(
                    2000 + layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    norm_eps,
                    qk_norm,
                    modulation=False,
                )
                for layer_id in range(n_refiner_layers)
            ])
            self.siglip_pad_token = nn.Parameter(torch.empty((1, dim)))
        else:
            self.siglip_embedder = None
            self.siglip_refiner = None
            self.siglip_pad_token = None

        # Pad tokens
        self.x_pad_token = nn.Parameter(torch.empty((1, dim)))
        self.cap_pad_token = nn.Parameter(torch.empty((1, dim)))

        # Main transformer layers
        self.layers = nn.ModuleList([
            ZImageTransformerBlock(
                layer_id,
                dim,
                n_heads,
                n_kv_heads,
                norm_eps,
                qk_norm,
            )
            for layer_id in range(n_layers)
        ])

        # RoPE embedder
        head_dim = dim // n_heads
        assert head_dim == sum(axes_dims)
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        self.rope_embedder = RopeEmbedder(theta=rope_theta, axes_dims=axes_dims, axes_lens=axes_lens)

        # Store config for compatibility with diffusers
        self.config = _ConfigProxy(in_channels=in_channels)

    def get_num_params(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def create_coordinate_grid(
        size: Tuple[int, ...],
        start: Optional[Tuple[int, ...]] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Create coordinate grid for position IDs."""
        if start is None:
            start = tuple(0 for _ in size)

        axes = [
            torch.arange(x0, x0 + span, dtype=torch.int32, device=device)
            for x0, span in zip(start, size)
        ]
        grids = torch.meshgrid(axes, indexing="ij")
        return torch.stack(grids, dim=-1)

    def patchify_and_embed(
        self,
        all_image: List[torch.Tensor],
        all_cap_feats: List[torch.Tensor],
        patch_size: int = 2,
        f_patch_size: int = 1,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], Dict]:
        """
        Patchify images and prepare for transformer.

        Args:
            all_image: List of images, each (C, F, H, W) or (C, H, W)
            all_cap_feats: List of caption features, each (seq_len, cap_dim)
            patch_size: Spatial patch size
            f_patch_size: Frame patch size (1 for images)

        Returns:
            Tuple of (image_tokens, cap_tokens, metadata_dict)
        """
        pH = pW = patch_size
        pF = f_patch_size
        device = all_image[0].device

        all_image_out = []
        all_image_size = []
        all_image_pos_ids = []
        all_image_pad_mask = []
        all_cap_pos_ids = []
        all_cap_pad_mask = []
        all_cap_feats_out = []

        for i, (image, cap_feat) in enumerate(zip(all_image, all_cap_feats)):
            # Process Caption
            cap_ori_len = len(cap_feat)
            cap_padding_len = (-cap_ori_len) % SEQ_MULTI_OF

            # Padded position IDs for caption
            cap_padded_pos_ids = self.create_coordinate_grid(
                size=(cap_ori_len + cap_padding_len, 1, 1),
                start=(1, 0, 0),
                device=device,
            ).flatten(0, 2)
            all_cap_pos_ids.append(cap_padded_pos_ids)

            # Pad mask
            all_cap_pad_mask.append(
                torch.cat([
                    torch.zeros((cap_ori_len,), dtype=torch.bool, device=device),
                    torch.ones((cap_padding_len,), dtype=torch.bool, device=device),
                ], dim=0)
            )

            # Padded feature
            cap_padded_feat = torch.cat(
                [cap_feat, cap_feat[-1:].repeat(cap_padding_len, 1)],
                dim=0,
            )
            all_cap_feats_out.append(cap_padded_feat)

            # Process Image
            C, F, H, W = image.size()
            all_image_size.append((F, H, W))
            F_tokens, H_tokens, W_tokens = F // pF, H // pH, W // pW

            # Patchify: (C, F, H, W) -> (num_patches, patch_dim)
            image = image.view(C, F_tokens, pF, H_tokens, pH, W_tokens, pW)
            image = image.permute(1, 3, 5, 2, 4, 6, 0).reshape(
                F_tokens * H_tokens * W_tokens,
                pF * pH * pW * C,
            )

            image_ori_len = len(image)
            image_padding_len = (-image_ori_len) % SEQ_MULTI_OF

            # Position IDs for image
            image_ori_pos_ids = self.create_coordinate_grid(
                size=(F_tokens, H_tokens, W_tokens),
                start=(cap_ori_len + cap_padding_len + 1, 0, 0),
                device=device,
            ).flatten(0, 2)
            image_padding_pos_ids = (
                self.create_coordinate_grid(
                    size=(1, 1, 1),
                    start=(0, 0, 0),
                    device=device,
                )
                .flatten(0, 2)
                .repeat(image_padding_len, 1)
            )
            image_padded_pos_ids = torch.cat([image_ori_pos_ids, image_padding_pos_ids], dim=0)
            all_image_pos_ids.append(image_padded_pos_ids)

            # Pad mask
            all_image_pad_mask.append(
                torch.cat([
                    torch.zeros((image_ori_len,), dtype=torch.bool, device=device),
                    torch.ones((image_padding_len,), dtype=torch.bool, device=device),
                ], dim=0)
            )

            # Padded feature
            image_padded_feat = torch.cat([image, image[-1:].repeat(image_padding_len, 1)], dim=0)
            all_image_out.append(image_padded_feat)

        return all_image_out, all_cap_feats_out, {
            "x_size": all_image_size,
            "x_pos_ids": all_image_pos_ids,
            "cap_pos_ids": all_cap_pos_ids,
            "x_pad_mask": all_image_pad_mask,
            "cap_pad_mask": all_cap_pad_mask,
        }

    def _prepare_sequence(
        self,
        feats: List[torch.Tensor],
        pos_ids: List[torch.Tensor],
        inner_pad_mask: List[torch.Tensor],
        pad_token: nn.Parameter,
        noise_mask: Optional[List[List[int]]] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int], Optional[torch.Tensor]]:
        """Prepare sequence: apply pad token, RoPE embed, pad to batch, create attention mask."""
        item_seqlens = [len(f) for f in feats]
        max_seqlen = max(item_seqlens)
        bsz = len(feats)

        # Apply pad token
        feats_cat = torch.cat(feats, dim=0)
        feats_cat[torch.cat(inner_pad_mask)] = pad_token.to(dtype=feats_cat.dtype, device=feats_cat.device)
        feats = list(feats_cat.split(item_seqlens, dim=0))

        # RoPE
        freqs_cis = list(
            self.rope_embedder(torch.cat(pos_ids, dim=0)).split([len(p) for p in pos_ids], dim=0)
        )

        # Pad to batch
        feats = pad_sequence(feats, batch_first=True, padding_value=0.0)
        freqs_cis = pad_sequence(freqs_cis, batch_first=True, padding_value=0.0)[:, :feats.shape[1]]

        # Attention mask
        attn_mask = torch.zeros((bsz, max_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(item_seqlens):
            attn_mask[i, :seq_len] = 1

        # Noise mask
        noise_mask_tensor = None
        if noise_mask is not None:
            noise_mask_tensor = pad_sequence(
                [torch.tensor(m, dtype=torch.long, device=device) for m in noise_mask],
                batch_first=True,
                padding_value=0,
            )[:, :feats.shape[1]]

        return feats, freqs_cis, attn_mask, item_seqlens, noise_mask_tensor

    def _build_unified_sequence(
        self,
        x: torch.Tensor,
        x_freqs: torch.Tensor,
        x_seqlens: List[int],
        x_noise_mask: Optional[List[List[int]]],
        cap: torch.Tensor,
        cap_freqs: torch.Tensor,
        cap_seqlens: List[int],
        cap_noise_mask: Optional[List[List[int]]],
        siglip: Optional[torch.Tensor],
        siglip_freqs: Optional[torch.Tensor],
        siglip_seqlens: Optional[List[int]],
        siglip_noise_mask: Optional[List[List[int]]],
        omni_mode: bool,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Build unified sequence: x, cap, and optionally siglip."""
        bsz = len(x_seqlens)
        unified = []
        unified_freqs = []
        unified_noise_mask = []

        for i in range(bsz):
            x_len, cap_len = x_seqlens[i], cap_seqlens[i]

            if omni_mode:
                # Omni: [cap, x, siglip]
                if siglip is not None and siglip_seqlens is not None:
                    sig_len = siglip_seqlens[i]
                    unified.append(torch.cat([cap[i][:cap_len], x[i][:x_len], siglip[i][:sig_len]]))
                    unified_freqs.append(
                        torch.cat([cap_freqs[i][:cap_len], x_freqs[i][:x_len], siglip_freqs[i][:sig_len]])
                    )
                    unified_noise_mask.append(
                        torch.tensor(
                            cap_noise_mask[i] + x_noise_mask[i] + siglip_noise_mask[i],
                            dtype=torch.long,
                            device=device,
                        )
                    )
                else:
                    unified.append(torch.cat([cap[i][:cap_len], x[i][:x_len]]))
                    unified_freqs.append(torch.cat([cap_freqs[i][:cap_len], x_freqs[i][:x_len]]))
                    unified_noise_mask.append(
                        torch.tensor(cap_noise_mask[i] + x_noise_mask[i], dtype=torch.long, device=device)
                    )
            else:
                # Basic: [x, cap]
                unified.append(torch.cat([x[i][:x_len], cap[i][:cap_len]]))
                unified_freqs.append(torch.cat([x_freqs[i][:x_len], cap_freqs[i][:cap_len]]))

        # Compute unified seqlens
        if omni_mode:
            if siglip is not None and siglip_seqlens is not None:
                unified_seqlens = [a + b + c for a, b, c in zip(cap_seqlens, x_seqlens, siglip_seqlens)]
            else:
                unified_seqlens = [a + b for a, b in zip(cap_seqlens, x_seqlens)]
        else:
            unified_seqlens = [a + b for a, b in zip(x_seqlens, cap_seqlens)]

        max_seqlen = max(unified_seqlens)

        # Pad to batch
        unified = pad_sequence(unified, batch_first=True, padding_value=0.0)
        unified_freqs = pad_sequence(unified_freqs, batch_first=True, padding_value=0.0)

        # Attention mask
        attn_mask = torch.zeros((bsz, max_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(unified_seqlens):
            attn_mask[i, :seq_len] = 1

        # Noise mask
        noise_mask_tensor = None
        if omni_mode:
            noise_mask_tensor = pad_sequence(unified_noise_mask, batch_first=True, padding_value=0)[
                :, :unified.shape[1]
            ]

        return unified, unified_freqs, attn_mask, noise_mask_tensor

    def unpatchify(
        self,
        x: List[torch.Tensor],
        size: List[Tuple[int, int, int]],
        patch_size: int = 2,
        f_patch_size: int = 1,
        x_pos_offsets: Optional[List[Tuple[int, int]]] = None,
    ) -> List[torch.Tensor]:
        """Convert patch tokens back to images."""
        pH = pW = patch_size
        pF = f_patch_size
        bsz = len(x)
        assert len(size) == bsz

        if x_pos_offsets is not None:
            # Omni mode: extract target image from unified sequence
            result = []
            for i in range(bsz):
                unified_x = x[i][x_pos_offsets[i][0]:x_pos_offsets[i][1]]
                cu_len = 0
                x_item = None
                for j in range(len(size[i])):
                    if size[i][j] is None:
                        ori_len = 0
                        pad_len = SEQ_MULTI_OF
                        cu_len += pad_len + ori_len
                    else:
                        F, H, W = size[i][j]
                        ori_len = (F // pF) * (H // pH) * (W // pW)
                        pad_len = (-ori_len) % SEQ_MULTI_OF
                        x_item = (
                            unified_x[cu_len:cu_len + ori_len]
                            .view(F // pF, H // pH, W // pW, pF, pH, pW, self.out_channels)
                            .permute(6, 0, 3, 1, 4, 2, 5)
                            .reshape(self.out_channels, F, H, W)
                        )
                        cu_len += ori_len + pad_len
                result.append(x_item)
            return result
        else:
            # Basic mode: simple unpatchify
            for i in range(bsz):
                F, H, W = size[i]
                ori_len = (F // pF) * (H // pH) * (W // pW)
                x[i] = (
                    x[i][:ori_len]
                    .view(F // pF, H // pH, W // pW, pF, pH, pW, self.out_channels)
                    .permute(6, 0, 3, 1, 4, 2, 5)
                    .reshape(self.out_channels, F, H, W)
                )
            return x

    def forward(
        self,
        x: List[torch.Tensor],
        t: torch.Tensor,
        cap_feats: List[torch.Tensor],
        siglip_feats: Optional[List[torch.Tensor]] = None,
        image_noise_mask: Optional[List[List[int]]] = None,
        patch_size: int = 2,
        f_patch_size: int = 1,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ) -> Tuple[List[torch.Tensor], ...]:
        """
        Forward pass through Z-Image DiT.

        Args:
            x: List of latent images, each (C, F, H, W)
            t: Timestep tensor (batch,) in range [0, 1000]
            cap_feats: List of caption features, each (seq_len, cap_dim)
            siglip_feats: Optional SigLIP features for Omni mode
            image_noise_mask: Noise indicators for Omni mode
            patch_size: Spatial patch size
            f_patch_size: Frame patch size
            use_gradient_checkpointing: Enable gradient checkpointing
            use_gradient_checkpointing_offload: Enable CPU offload during checkpointing

        Returns:
            Tuple containing list of output images
        """
        assert patch_size in self.all_patch_size and f_patch_size in self.all_f_patch_size
        omni_mode = isinstance(x[0], list)
        device = x[0][-1].device if omni_mode else x[0].device

        # Timestep embeddings
        if omni_mode:
            # Dual embeddings: noisy (t) and clean (t=1)
            t_noisy = self.t_embedder(t * self.t_scale).type_as(x[0][-1])
            t_clean = self.t_embedder(torch.ones_like(t) * self.t_scale).type_as(x[0][-1])
            adaln_input = None
        else:
            # Single embedding for all tokens
            adaln_input = self.t_embedder(t * self.t_scale).type_as(x[0])
            t_noisy = t_clean = None

        # Patchify
        if omni_mode:
            # Omni mode patchify (not implemented in this basic version)
            raise NotImplementedError("Omni mode not yet implemented in pure PyTorch version")
        else:
            x, cap_feats, metadata = self.patchify_and_embed(
                x, cap_feats, patch_size, f_patch_size
            )
            x_size = metadata["x_size"]
            x_pos_ids = metadata["x_pos_ids"]
            cap_pos_ids = metadata["cap_pos_ids"]
            x_pad_mask = metadata["x_pad_mask"]
            cap_pad_mask = metadata["cap_pad_mask"]
            x_pos_offsets = x_noise_mask = cap_noise_mask = siglip_noise_mask = None

        # x embed & refine
        x_seqlens = [len(xi) for xi in x]
        x = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](torch.cat(x, dim=0))
        x, x_freqs, x_mask, _, x_noise_tensor = self._prepare_sequence(
            list(x.split(x_seqlens, dim=0)),
            x_pos_ids,
            x_pad_mask,
            self.x_pad_token,
            x_noise_mask,
            device,
        )

        for layer in self.noise_refiner:
            x = layer(
                x,
                attn_mask=x_mask,
                freqs_cis=x_freqs,
                adaln_input=adaln_input,
                noise_mask=x_noise_tensor,
                adaln_noisy=t_noisy,
                adaln_clean=t_clean,
            )

        # Cap embed & refine
        cap_seqlens = [len(ci) for ci in cap_feats]
        cap_feats = self.cap_embedder(torch.cat(cap_feats, dim=0))
        cap_feats, cap_freqs, cap_mask, _, _ = self._prepare_sequence(
            list(cap_feats.split(cap_seqlens, dim=0)),
            cap_pos_ids,
            cap_pad_mask,
            self.cap_pad_token,
            None,
            device,
        )

        for layer in self.context_refiner:
            cap_feats = layer(
                cap_feats,
                attn_mask=cap_mask,
                freqs_cis=cap_freqs,
            )

        # SigLIP embed & refine (Omni mode)
        siglip_seqlens = siglip_freqs = None

        # Unified sequence
        unified, unified_freqs, unified_mask, unified_noise_tensor = self._build_unified_sequence(
            x,
            x_freqs,
            x_seqlens,
            x_noise_mask,
            cap_feats,
            cap_freqs,
            cap_seqlens,
            cap_noise_mask,
            None,  # siglip_feats
            siglip_freqs,
            siglip_seqlens,
            siglip_noise_mask,
            omni_mode,
            device,
        )

        # Main transformer layers
        for layer in self.layers:
            unified = layer(
                unified,
                attn_mask=unified_mask,
                freqs_cis=unified_freqs,
                adaln_input=adaln_input,
                noise_mask=unified_noise_tensor,
                adaln_noisy=t_noisy,
                adaln_clean=t_clean,
            )

        # Final layer
        unified = (
            self.all_final_layer[f"{patch_size}-{f_patch_size}"](
                unified,
                noise_mask=unified_noise_tensor,
                c_noisy=t_noisy,
                c_clean=t_clean,
            )
            if omni_mode
            else self.all_final_layer[f"{patch_size}-{f_patch_size}"](unified, c=adaln_input)
        )

        # Unpatchify
        x = self.unpatchify(
            list(unified.unbind(dim=0)),
            x_size,
            patch_size,
            f_patch_size,
            x_pos_offsets,
        )

        return (x,)


class _ConfigProxy:
    """Simple config proxy to match diffusers interface."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get(self, key: str, default=None):
        return getattr(self, key, default)
