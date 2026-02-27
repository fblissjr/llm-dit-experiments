"""
Flux VAE Decoder for Z-Image.

Last updated: 2026-01-29

Implements the 16-channel VAE decoder used by Z-Image.
Architecture based on DiffSynth-Studio flux_vae.py reference.

The decoder takes 16-channel latents and produces RGB images:
- Input: (B, 16, H/8, W/8) latents
- Output: (B, 3, H, W) images in [-1, 1] range
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..constants import FluxVAEConfig


class Attention(nn.Module):
    """Simple self-attention for VAE mid-block."""

    def __init__(
        self,
        q_dim: int,
        num_heads: int,
        head_dim: int,
        kv_dim: Optional[int] = None,
        bias_q: bool = False,
        bias_kv: bool = False,
        bias_out: bool = False,
    ):
        super().__init__()
        dim_inner = head_dim * num_heads
        kv_dim = kv_dim if kv_dim is not None else q_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.to_q = nn.Linear(q_dim, dim_inner, bias=bias_q)
        self.to_k = nn.Linear(kv_dim, dim_inner, bias=bias_kv)
        self.to_v = nn.Linear(kv_dim, dim_inner, bias=bias_kv)
        self.to_out = nn.Linear(dim_inner, q_dim, bias=bias_out)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        batch_size = encoder_hidden_states.shape[0]

        q = self.to_q(hidden_states)
        k = self.to_k(encoder_hidden_states)
        v = self.to_v(encoder_hidden_states)

        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_dim)
        hidden_states = hidden_states.to(q.dtype)

        hidden_states = self.to_out(hidden_states)

        return hidden_states


class VAEAttentionBlock(nn.Module):
    """Attention block for VAE mid-block."""

    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        in_channels: int,
        num_layers: int = 1,
        norm_num_groups: int = 32,
        eps: float = 1e-5,
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim

        self.norm = nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=eps, affine=True)

        self.transformer_blocks = nn.ModuleList([
            Attention(
                inner_dim,
                num_attention_heads,
                attention_head_dim,
                bias_q=True,
                bias_kv=True,
                bias_out=True,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        time_emb: Optional[torch.Tensor],
        text_emb: Optional[torch.Tensor],
        res_stack: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        batch, _, height, width = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states)

        hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        hidden_states = hidden_states + residual

        return hidden_states, time_emb, text_emb, res_stack


class ResnetBlock(nn.Module):
    """Residual block for VAE."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: Optional[int] = None,
        groups: int = 32,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels is not None:
            self.time_emb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.nonlinearity = nn.SiLU()
        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        time_emb: Optional[torch.Tensor],
        text_emb: Optional[torch.Tensor],
        res_stack: Optional[torch.Tensor],
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        x = hidden_states
        x = self.norm1(x)
        x = self.nonlinearity(x)
        x = self.conv1(x)
        if time_emb is not None and hasattr(self, "time_emb_proj"):
            emb = self.nonlinearity(time_emb)
            emb = self.time_emb_proj(emb)[:, :, None, None]
            x = x + emb
        x = self.norm2(x)
        x = self.nonlinearity(x)
        x = self.conv2(x)
        if self.conv_shortcut is not None:
            hidden_states = self.conv_shortcut(hidden_states)
        hidden_states = hidden_states + x
        return hidden_states, time_emb, text_emb, res_stack


class UpSampler(nn.Module):
    """2x upsampling layer."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        time_emb: Optional[torch.Tensor],
        text_emb: Optional[torch.Tensor],
        res_stack: Optional[torch.Tensor],
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        hidden_states = self.conv(hidden_states)
        return hidden_states, time_emb, text_emb, res_stack


class TileWorker:
    """Helper class for tiled processing to reduce VRAM usage."""

    def mask(self, height: int, width: int, border_width: int) -> torch.Tensor:
        """Create a blending mask for tile boundaries."""
        x = torch.arange(height).repeat(width, 1).T
        y = torch.arange(width).repeat(height, 1)
        mask = torch.stack([x + 1, height - x, y + 1, width - y]).min(dim=0).values
        mask = (mask / border_width).clip(0, 1)
        return mask

    def tile(
        self,
        model_input: torch.Tensor,
        tile_size: int,
        tile_stride: int,
        tile_device: torch.device,
        tile_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Split input into overlapping tiles."""
        batch_size, channel, _, _ = model_input.shape
        model_input = model_input.to(device=tile_device, dtype=tile_dtype)
        unfold_operator = nn.Unfold(kernel_size=(tile_size, tile_size), stride=(tile_stride, tile_stride))
        model_input = unfold_operator(model_input)
        model_input = model_input.view((batch_size, channel, tile_size, tile_size, -1))
        return model_input

    def untile(
        self,
        model_output: torch.Tensor,
        height: int,
        width: int,
        tile_size: int,
        tile_stride: int,
        border_width: int,
        tile_device: torch.device,
        tile_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Recombine tiles with blending."""
        mask = self.mask(tile_size, tile_size, border_width)
        mask = mask.to(device=tile_device, dtype=tile_dtype)
        mask = rearrange(mask, "h w -> 1 1 h w 1")
        model_output = model_output * mask

        fold_operator = nn.Fold(
            output_size=(height, width),
            kernel_size=(tile_size, tile_size),
            stride=(tile_stride, tile_stride),
        )
        mask_repeated = mask[0, 0, :, :, 0].reshape(1, -1, 1).expand(1, -1, model_output.shape[-1])
        model_output_flat = rearrange(model_output, "b c h w n -> b (c h w) n")
        model_output = fold_operator(model_output_flat) / fold_operator(mask_repeated)

        return model_output

    def tiled_forward(
        self,
        forward_fn,
        model_input: torch.Tensor,
        tile_size: int,
        tile_stride: int,
        tile_batch_size: int = 1,
        tile_device: Optional[torch.device] = None,
        tile_dtype: Optional[torch.dtype] = None,
        border_width: Optional[int] = None,
    ) -> torch.Tensor:
        """Run forward function with tiling."""
        if tile_device is None:
            tile_device = model_input.device
        if tile_dtype is None:
            tile_dtype = model_input.dtype

        inference_device, inference_dtype = model_input.device, model_input.dtype
        height, width = model_input.shape[2], model_input.shape[3]
        border_width = int(tile_stride * 0.5) if border_width is None else border_width

        # Tile
        model_input = self.tile(model_input, tile_size, tile_stride, tile_device, tile_dtype)

        # Inference
        tile_num = model_input.shape[-1]
        model_output_stack = []
        for tile_id in range(0, tile_num, tile_batch_size):
            tile_id_ = min(tile_id + tile_batch_size, tile_num)
            x = model_input[:, :, :, :, tile_id:tile_id_]
            x = x.to(device=inference_device, dtype=inference_dtype)
            x = rearrange(x, "b c h w n -> (n b) c h w")

            y = forward_fn(x)
            y = rearrange(y, "(n b) c h w -> b c h w n", n=tile_id_ - tile_id)
            y = y.to(device=tile_device, dtype=tile_dtype)
            model_output_stack.append(y)

        model_output = torch.concat(model_output_stack, dim=-1)

        # Calculate output size based on io_scale
        io_scale = model_output.shape[2] / tile_size
        out_height, out_width = int(height * io_scale), int(width * io_scale)
        out_tile_size, out_tile_stride = int(tile_size * io_scale), int(tile_stride * io_scale)
        out_border_width = int(border_width * io_scale)

        # Untile
        model_output = self.untile(
            model_output,
            out_height,
            out_width,
            out_tile_size,
            out_tile_stride,
            out_border_width,
            tile_device,
            tile_dtype,
        )

        return model_output.to(device=inference_device, dtype=inference_dtype)


class FluxVAEDecoder(nn.Module):
    """
    Flux VAE Decoder for Z-Image.

    Decodes 16-channel latents to RGB images.

    Architecture:
    - conv_in: 16 → 512
    - mid_block: ResNet + Attention + ResNet
    - up_blocks: 4x UpDecoderBlock2D with channel progression 512→512→256→128
    - conv_out: 128 → 3

    Args:
        use_conv_attention: Use Conv2d-based attention (default: False uses Linear)

    Properties:
        config: Returns VAE configuration with scaling_factor and shift_factor
    """

    def __init__(self, use_conv_attention: bool = False):
        super().__init__()
        self.scaling_factor = FluxVAEConfig.DECODER["scaling_factor"]
        self.shift_factor = FluxVAEConfig.DECODER["shift_factor"]

        # Input convolution (16 → 512)
        self.conv_in = nn.Conv2d(16, 512, kernel_size=3, padding=1)

        # Build blocks
        self.blocks = nn.ModuleList([
            # UNetMidBlock2D
            ResnetBlock(512, 512, eps=1e-6),
            VAEAttentionBlock(1, 512, 512, 1, eps=1e-6),
            ResnetBlock(512, 512, eps=1e-6),
            # UpDecoderBlock2D (512 → 512)
            ResnetBlock(512, 512, eps=1e-6),
            ResnetBlock(512, 512, eps=1e-6),
            ResnetBlock(512, 512, eps=1e-6),
            UpSampler(512),
            # UpDecoderBlock2D (512 → 512)
            ResnetBlock(512, 512, eps=1e-6),
            ResnetBlock(512, 512, eps=1e-6),
            ResnetBlock(512, 512, eps=1e-6),
            UpSampler(512),
            # UpDecoderBlock2D (512 → 256)
            ResnetBlock(512, 256, eps=1e-6),
            ResnetBlock(256, 256, eps=1e-6),
            ResnetBlock(256, 256, eps=1e-6),
            UpSampler(256),
            # UpDecoderBlock2D (256 → 128)
            ResnetBlock(256, 128, eps=1e-6),
            ResnetBlock(128, 128, eps=1e-6),
            ResnetBlock(128, 128, eps=1e-6),
        ])

        # Output layers
        self.conv_norm_out = nn.GroupNorm(num_channels=128, num_groups=32, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(128, 3, kernel_size=3, padding=1)

        # Config proxy for compatibility
        self._config = _VAEConfigProxy(
            scaling_factor=self.scaling_factor,
            shift_factor=self.shift_factor,
            block_out_channels=FluxVAEConfig.DECODER["block_out_channels"],
        )

    @property
    def config(self):
        """Return config for compatibility with diffusers interface."""
        return self._config

    @property
    def dtype(self) -> torch.dtype:
        """Return model dtype."""
        return next(self.parameters()).dtype

    def tiled_forward(
        self,
        sample: torch.Tensor,
        tile_size: int = 64,
        tile_stride: int = 32,
    ) -> torch.Tensor:
        """Run forward with tiling for large images."""
        return TileWorker().tiled_forward(
            lambda x: self._forward_core(x),
            sample,
            tile_size,
            tile_stride,
            tile_device=sample.device,
            tile_dtype=sample.dtype,
        )

    def _forward_core(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Core forward pass without scaling."""
        time_emb = None
        text_emb = None
        res_stack = None

        # Process through blocks
        for block in self.blocks:
            hidden_states, time_emb, text_emb, res_stack = block(
                hidden_states, time_emb, text_emb, res_stack
            )

        # Output projection
        hidden_states = self.conv_norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)

        return hidden_states

    def forward(
        self,
        sample: torch.Tensor,
        tiled: bool = False,
        tile_size: int = 64,
        tile_stride: int = 32,
        **kwargs,
    ) -> torch.Tensor:
        """
        Decode latents to image.

        Args:
            sample: Latent tensor (B, 16, H/8, W/8)
            tiled: Use tiled processing for large images
            tile_size: Tile size for tiled processing
            tile_stride: Tile stride for tiled processing

        Returns:
            Decoded image tensor (B, 3, H, W) in [-1, 1] range
        """
        if tiled:
            return self.tiled_forward(sample, tile_size=tile_size, tile_stride=tile_stride)

        # 1. Pre-process: un-scale latents
        hidden_states = sample / self.scaling_factor + self.shift_factor
        hidden_states = self.conv_in(hidden_states)

        # 2. Core forward
        hidden_states = self._forward_core(hidden_states)

        return hidden_states

    def decode(
        self,
        latents: torch.Tensor,
        return_dict: bool = True,
        **kwargs,
    ):
        """
        Decode latents to image (diffusers-compatible interface).

        Args:
            latents: Latent tensor (B, 16, H/8, W/8)
            return_dict: Whether to return a dict (ignored, always returns tuple)

        Returns:
            Tuple containing decoded image tensor
        """
        # Note: The pipeline handles scaling before calling decode
        # So we just run the forward pass without re-scaling
        hidden_states = self.conv_in(latents)

        # Core forward
        time_emb = None
        text_emb = None
        res_stack = None
        for block in self.blocks:
            hidden_states, time_emb, text_emb, res_stack = block(
                hidden_states, time_emb, text_emb, res_stack
            )

        hidden_states = self.conv_norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)

        return (hidden_states,)


class _VAEConfigProxy:
    """Simple config proxy to match diffusers interface."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get(self, key: str, default=None):
        return getattr(self, key, default)
