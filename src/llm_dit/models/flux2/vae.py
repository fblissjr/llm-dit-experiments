"""
FLUX.2 VAE (AutoEncoder) with patchify and BatchNorm latent normalization.

Last Updated: 2026-01-23

Implements the FLUX.2 image VAE with 2x2 patchify that increases channels
and reduces spatial dimensions. Uses BatchNorm for latent normalization.

Key Features:
- 8x spatial compression from encoder (standard VAE)
- 2x additional compression from patchify (total 16x)
- 32 latent channels -> 128 channels after patchify
- BatchNorm with running statistics for latent normalization
- GroupNorm (32 groups) for internal normalization

Ported from: coderef/flux2/src/flux2/autoencoder.py

Usage:
    from llm_dit.models.flux2.vae import AutoEncoder, AutoEncoderParams

    # Create VAE
    vae = AutoEncoder(AutoEncoderParams())

    # Encode image
    x = torch.randn(1, 3, 1024, 1024)  # RGB image
    z = vae.encode(x)  # [1, 128, 64, 64] - patchified latent

    # Decode back to image
    x_recon = vae.decode(z)  # [1, 3, 1024, 1024]
"""

import math
from dataclasses import dataclass, field

import torch
from einops import rearrange
from torch import Tensor, nn


def swish(x: Tensor) -> Tensor:
    """Swish/SiLU activation: x * sigmoid(x)."""
    return x * torch.sigmoid(x)


@dataclass
class AutoEncoderParams:
    """
    Parameters for the FLUX.2 AutoEncoder.

    These match the official FLUX.2 VAE configuration.
    """
    resolution: int = 256  # Reference resolution (not used in forward pass)
    in_channels: int = 3  # RGB input
    ch: int = 128  # Base channel count
    out_ch: int = 3  # RGB output
    ch_mult: list[int] = field(default_factory=lambda: [1, 2, 4, 4])  # Channel multipliers
    num_res_blocks: int = 2  # ResBlocks per level
    z_channels: int = 32  # Latent channels before patchify


class AttnBlock(nn.Module):
    """
    Self-attention block with GroupNorm.

    Used in the middle of the encoder and decoder for global context.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: Tensor) -> Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        # Reshape for scaled_dot_product_attention: [B, 1, H*W, C]
        q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()

        h_ = nn.functional.scaled_dot_product_attention(q, k, v)

        return rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))


class ResnetBlock(nn.Module):
    """
    Residual block with GroupNorm and Swish activation.

    Standard ResNet-style block with optional channel change via 1x1 conv.
    """

    def __init__(self, in_channels: int, out_channels: int | None = None):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class Downsample(nn.Module):
    """
    2x spatial downsampling with strided convolution.

    Uses asymmetric padding (0, 1, 0, 1) to match TF behavior.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        # Strided conv for 2x downsample
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        # Asymmetric padding for TF compatibility
        pad = (0, 1, 0, 1)
        x = nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    """
    2x spatial upsampling with nearest interpolation + convolution.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    """
    VAE Encoder: Image -> Latent.

    Downsamples spatially while increasing channels, ending with attention
    in the middle block. Outputs mean of VAE posterior (no sampling in inference).
    """

    def __init__(
        self,
        resolution: int,
        in_channels: int,
        ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
    ):
        super().__init__()
        self.quant_conv = nn.Conv2d(2 * z_channels, 2 * z_channels, 1)
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        # Downsampling blocks
        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        block_in = self.ch

        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]

            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out

            down = nn.Module()
            down.block = block
            down.attn = attn

            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
                curr_res = curr_res // 2

            self.down.append(down)

        # Middle block
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # Output
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        # Downsampling
        hs = [self.conv_in(x)]

        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)

            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # Middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # Output
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        h = self.quant_conv(h)

        return h


class Decoder(nn.Module):
    """
    VAE Decoder: Latent -> Image.

    Upsamples spatially while decreasing channels.
    """

    def __init__(
        self,
        ch: int,
        out_ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        in_channels: int,
        resolution: int,
        z_channels: int,
    ):
        super().__init__()
        self.post_quant_conv = nn.Conv2d(z_channels, z_channels, 1)
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.ffactor = 2 ** (self.num_resolutions - 1)

        # Compute initial block dimensions
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        # Initial convolution from latent
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # Middle block
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # Upsampling blocks
        self.up = nn.ModuleList()

        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]

            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out

            up = nn.Module()
            up.block = block
            up.attn = attn

            if i_level != 0:
                up.upsample = Upsample(block_in)
                curr_res = curr_res * 2

            self.up.insert(0, up)  # Prepend for correct order

        # Output
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        z = self.post_quant_conv(z)

        # Get dtype for proper tracing (from upsampling weights)
        upscale_dtype = next(self.up.parameters()).dtype

        # Initial projection
        h = self.conv_in(z)

        # Middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # Cast to proper dtype
        h = h.to(upscale_dtype)

        # Upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)

            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # Output
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)

        return h


class AutoEncoder(nn.Module):
    """
    FLUX.2 AutoEncoder with patchify and BatchNorm latent normalization.

    The patchify operation groups 2x2 spatial patches into channels:
    - Encode: [B, 32, H, W] -> patchify -> [B, 128, H/2, W/2]
    - Decode: [B, 128, H/2, W/2] -> unpatchify -> [B, 32, H, W]

    BatchNorm is used to normalize latents before the transformer and
    denormalize before decoding. Running statistics are tracked during
    training and used in inference mode.

    Args:
        params: AutoEncoderParams configuration
    """

    def __init__(self, params: AutoEncoderParams):
        super().__init__()
        self.params = params

        self.encoder = Encoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )

        self.decoder = Decoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            out_ch=params.out_ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )

        # Patchify parameters
        self.ps = [2, 2]  # 2x2 patch size

        # BatchNorm for latent normalization
        # After patchify: 32 * 2 * 2 = 128 channels
        self.bn_eps = 1e-4
        self.bn_momentum = 0.1
        self.bn = nn.BatchNorm2d(
            math.prod(self.ps) * params.z_channels,  # 128 channels
            eps=self.bn_eps,
            momentum=self.bn_momentum,
            affine=False,  # No learnable scale/shift
            track_running_stats=True,
        )

    def patchify(self, z: Tensor) -> Tensor:
        """
        Group 2x2 spatial patches into channels.

        [B, C, H, W] -> [B, C*4, H/2, W/2]
        """
        return rearrange(
            z,
            "... c (i pi) (j pj) -> ... (c pi pj) i j",
            pi=self.ps[0],
            pj=self.ps[1],
        )

    def unpatchify(self, z: Tensor) -> Tensor:
        """
        Ungroup channels back to 2x2 spatial patches.

        [B, C*4, H/2, W/2] -> [B, C, H, W]
        """
        return rearrange(
            z,
            "... (c pi pj) i j -> ... c (i pi) (j pj)",
            pi=self.ps[0],
            pj=self.ps[1],
        )

    def normalize(self, z: Tensor) -> Tensor:
        """
        Normalize latents using BatchNorm with running statistics.

        Always uses inference mode (running stats) for deterministic behavior.
        """
        # Set to inference mode to use running statistics
        was_training = self.bn.training
        self.bn.train(False)
        result = self.bn(z)
        self.bn.train(was_training)
        return result

    def inv_normalize(self, z: Tensor) -> Tensor:
        """
        Inverse normalize latents (denormalize).

        Reverses the BatchNorm normalization using running statistics.
        """
        was_training = self.bn.training
        self.bn.train(False)
        s = torch.sqrt(self.bn.running_var.view(1, -1, 1, 1) + self.bn_eps)
        m = self.bn.running_mean.view(1, -1, 1, 1)
        result = z * s + m
        self.bn.train(was_training)
        return result

    def encode(self, x: Tensor) -> Tensor:
        """
        Encode image to patchified, normalized latent.

        Args:
            x: Input image [B, 3, H, W] in range [-1, 1]

        Returns:
            Patchified latent [B, 128, H/16, W/16]
        """
        # Encode to VAE latent
        moments = self.encoder(x)
        # Take mean (no sampling in inference)
        mean = torch.chunk(moments, 2, dim=1)[0]

        # Patchify: [B, 32, H/8, W/8] -> [B, 128, H/16, W/16]
        z = self.patchify(mean)

        # Normalize with BatchNorm
        z = self.normalize(z)

        return z

    def decode(self, z: Tensor) -> Tensor:
        """
        Decode patchified latent to image.

        Args:
            z: Patchified latent [B, 128, H/16, W/16]

        Returns:
            Reconstructed image [B, 3, H, W] in range [-1, 1]
        """
        # Inverse normalize
        z = self.inv_normalize(z)

        # Unpatchify: [B, 128, H/16, W/16] -> [B, 32, H/8, W/8]
        z = self.unpatchify(z)

        # Decode to image
        dec = self.decoder(z)

        return dec

    def get_latent_shape(self, image_height: int, image_width: int) -> tuple[int, int, int]:
        """
        Calculate latent dimensions for a given image size.

        Args:
            image_height: Input image height
            image_width: Input image width

        Returns:
            (channels, latent_height, latent_width)
        """
        # Total compression: 16x (8x VAE + 2x patchify)
        latent_h = image_height // 16
        latent_w = image_width // 16
        channels = self.params.z_channels * math.prod(self.ps)  # 32 * 4 = 128
        return (channels, latent_h, latent_w)


# Alias for consistency with reference code
Flux2VAE = AutoEncoder
