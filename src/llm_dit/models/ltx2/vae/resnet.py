"""
LTX-2 VAE ResNet Blocks.

Last Updated: 2026-01-18

ResNet blocks for the Video VAE encoder and decoder.

Ported from: ltx_core.model.video_vae.resnet
Original source: https://github.com/Lightricks/LTX-2
License: LTX-2 Community License
Copyright (c) 2025 Lightricks Ltd.
"""

from typing import Optional, Tuple, Union

import torch
from torch import nn

from .convolution import make_conv_nd, make_linear_nd
from .enums import NormLayerType, PaddingModeType
from .normalization import PixelNorm
from .timestep_embedding import PixArtAlphaCombinedTimestepSizeEmbeddings


class ResnetBlock3D(nn.Module):
    """
    A 3D ResNet block with optional noise injection and timestep conditioning.

    Parameters:
        in_channels: The number of channels in the input.
        out_channels: The number of output channels. If None, same as in_channels.
        dropout: The dropout probability.
        groups: The number of groups for group normalization.
        eps: The epsilon for normalization.
        norm_layer: Type of normalization (group_norm or pixel_norm).
        inject_noise: Whether to inject per-channel noise (StyleGAN-like).
        timestep_conditioning: Whether to condition on timestep.
    """

    def __init__(
        self,
        dims: Union[int, Tuple[int, int]],
        in_channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        groups: int = 32,
        eps: float = 1e-6,
        norm_layer: NormLayerType = NormLayerType.PIXEL_NORM,
        inject_noise: bool = False,
        timestep_conditioning: bool = False,
        spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.inject_noise = inject_noise

        # First normalization
        if norm_layer == NormLayerType.GROUP_NORM:
            self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        elif norm_layer == NormLayerType.PIXEL_NORM:
            self.norm1 = PixelNorm()

        self.non_linearity = nn.SiLU()

        self.conv1 = make_conv_nd(
            dims,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )

        if inject_noise:
            self.per_channel_scale1 = nn.Parameter(torch.zeros((in_channels, 1, 1)))

        # Second normalization
        if norm_layer == NormLayerType.GROUP_NORM:
            self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)
        elif norm_layer == NormLayerType.PIXEL_NORM:
            self.norm2 = PixelNorm()

        self.dropout = nn.Dropout(dropout)

        self.conv2 = make_conv_nd(
            dims,
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )

        if inject_noise:
            self.per_channel_scale2 = nn.Parameter(torch.zeros((in_channels, 1, 1)))

        # Shortcut connection
        self.conv_shortcut = (
            make_linear_nd(dims=dims, in_channels=in_channels, out_channels=out_channels)
            if in_channels != out_channels
            else nn.Identity()
        )

        # LayerNorm via GroupNorm with 1 group
        self.norm3 = (
            nn.GroupNorm(num_groups=1, num_channels=in_channels, eps=eps, affine=True)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.timestep_conditioning = timestep_conditioning
        if timestep_conditioning:
            self.scale_shift_table = nn.Parameter(torch.zeros(4, in_channels))

    def _feed_spatial_noise(
        self,
        hidden_states: torch.Tensor,
        per_channel_scale: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Inject per-channel spatial noise (StyleGAN-like)."""
        spatial_shape = hidden_states.shape[-2:]
        device = hidden_states.device
        dtype = hidden_states.dtype

        spatial_noise = torch.randn(spatial_shape, device=device, dtype=dtype, generator=generator)[None]
        scaled_noise = (spatial_noise * per_channel_scale)[None, :, None, ...]
        hidden_states = hidden_states + scaled_noise

        return hidden_states

    def forward(
        self,
        input_tensor: torch.Tensor,
        causal: bool = True,
        timestep: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_tensor: Input tensor (B, C, D, H, W).
            causal: Whether to use causal convolution.
            timestep: Timestep embedding for conditioning.
            generator: Random generator for noise injection.

        Returns:
            Output tensor.
        """
        hidden_states = input_tensor
        batch_size = hidden_states.shape[0]

        # DEBUG: Detailed trace for 128ch blocks
        debug_128 = hidden_states.shape[1] == 128 and not hasattr(self, '_traced')
        if debug_128:
            self._traced = True
            print(f"[TRACE] Input: mean={hidden_states.mean():.4f}, std={hidden_states.std():.4f}")

        hidden_states = self.norm1(hidden_states)
        if debug_128:
            print(f"[TRACE] After norm1 (PixelNorm): mean={hidden_states.mean():.4f}, std={hidden_states.std():.4f}")

        # Initialize scale/shift for timestep conditioning (will be overwritten if used)
        scale2: torch.Tensor | None = None
        shift2: torch.Tensor | None = None

        if self.timestep_conditioning:
            if timestep is None:
                raise ValueError("'timestep' parameter must be provided when 'timestep_conditioning' is True")
            ada_values = self.scale_shift_table[None, ..., None, None, None].to(
                device=hidden_states.device, dtype=hidden_states.dtype
            ) + timestep.reshape(
                batch_size,
                4,
                -1,
                timestep.shape[-3],
                timestep.shape[-2],
                timestep.shape[-1],
            )
            shift1, scale1, shift2, scale2 = ada_values.unbind(dim=1)
            hidden_states = hidden_states * (1 + scale1) + shift1

        hidden_states = self.non_linearity(hidden_states)
        if debug_128:
            print(f"[TRACE] After SiLU: mean={hidden_states.mean():.4f}, std={hidden_states.std():.4f}")
        hidden_states = self.conv1(hidden_states, causal=causal)
        if debug_128:
            print(f"[TRACE] After conv1: mean={hidden_states.mean():.4f}, std={hidden_states.std():.4f}")

        if self.inject_noise:
            hidden_states = self._feed_spatial_noise(
                hidden_states,
                self.per_channel_scale1.to(device=hidden_states.device, dtype=hidden_states.dtype),
                generator=generator,
            )

        hidden_states = self.norm2(hidden_states)
        if debug_128:
            print(f"[TRACE] After norm2 (PixelNorm): mean={hidden_states.mean():.4f}, std={hidden_states.std():.4f}")

        if self.timestep_conditioning:
            assert scale2 is not None and shift2 is not None  # Always set when timestep_conditioning=True
            hidden_states = hidden_states * (1 + scale2) + shift2

        hidden_states = self.non_linearity(hidden_states)
        if debug_128:
            print(f"[TRACE] After SiLU (2nd): mean={hidden_states.mean():.4f}, std={hidden_states.std():.4f}")
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states, causal=causal)
        if debug_128:
            print(f"[TRACE] After conv2 (final hidden): mean={hidden_states.mean():.4f}, std={hidden_states.std():.4f}")

        if self.inject_noise:
            hidden_states = self._feed_spatial_noise(
                hidden_states,
                self.per_channel_scale2.to(device=hidden_states.device, dtype=hidden_states.dtype),
                generator=generator,
            )

        input_tensor = self.norm3(input_tensor)
        input_tensor = self.conv_shortcut(input_tensor)

        # DEBUG: Trace residual addition
        if hidden_states.shape[1] == 128:  # Only trace final block (128 channels)
            print(f"[DEBUG ResnetBlock3D (128ch)] input_tensor: mean={input_tensor.mean():.4f}, range=[{input_tensor.min():.4f}, {input_tensor.max():.4f}]")
            print(f"[DEBUG ResnetBlock3D (128ch)] hidden_states: mean={hidden_states.mean():.4f}, range=[{hidden_states.min():.4f}, {hidden_states.max():.4f}]")

        output_tensor = input_tensor + hidden_states
        return output_tensor


class UNetMidBlock3D(nn.Module):
    """
    A 3D UNet mid-block with multiple residual blocks.

    Args:
        in_channels: The number of input channels.
        dropout: The dropout rate.
        num_layers: The number of residual blocks.
        resnet_eps: The epsilon for resnet blocks.
        resnet_groups: The number of groups for group normalization.
        norm_layer: The normalization layer type.
        inject_noise: Whether to inject noise into hidden states.
        timestep_conditioning: Whether to condition on timestep.
    """

    def __init__(
        self,
        dims: Union[int, Tuple[int, int]],
        in_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_groups: int = 32,
        norm_layer: NormLayerType = NormLayerType.GROUP_NORM,
        inject_noise: bool = False,
        timestep_conditioning: bool = False,
        spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
        attention_head_dim: Optional[int] = None,  # Not used, kept for API compatibility
    ):
        super().__init__()
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

        self.timestep_conditioning = timestep_conditioning

        if timestep_conditioning:
            self.time_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(
                embedding_dim=in_channels * 4, size_emb_dim=0
            )

        self.res_blocks = nn.ModuleList(
            [
                ResnetBlock3D(
                    dims=dims,
                    in_channels=in_channels,
                    out_channels=in_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    norm_layer=norm_layer,
                    inject_noise=inject_noise,
                    timestep_conditioning=timestep_conditioning,
                    spatial_padding_mode=spatial_padding_mode,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        causal: bool = True,
        timestep: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            hidden_states: Input tensor (B, C, D, H, W).
            causal: Whether to use causal convolution.
            timestep: Timestep for conditioning.
            generator: Random generator for noise injection.

        Returns:
            Output tensor.
        """
        timestep_embed = None
        if self.timestep_conditioning:
            if timestep is None:
                raise ValueError("'timestep' parameter must be provided when 'timestep_conditioning' is True")
            batch_size = hidden_states.shape[0]
            timestep_embed = self.time_embedder(
                timestep=timestep.flatten(),
                hidden_dtype=hidden_states.dtype,
            )
            timestep_embed = timestep_embed.view(batch_size, timestep_embed.shape[-1], 1, 1, 1)

        for resnet in self.res_blocks:
            hidden_states = resnet(
                hidden_states,
                causal=causal,
                timestep=timestep_embed,
                generator=generator,
            )

        return hidden_states
