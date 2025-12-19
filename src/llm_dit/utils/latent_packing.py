"""Latent packing utilities for Qwen-Image models.

The Qwen-Image DiT expects latents packed in a specific 2x2 spatial pattern:
- Input: (B, 16, H, W) raw VAE latents
- Packed: (B, H*W//4, 64) sequence for transformer

This spatial packing groups 2x2 patches of latents into single sequence tokens,
effectively treating 4 adjacent latent pixels as one token.
"""

import torch
from typing import Tuple


def pack_latents_2x2(
    latents: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Pack VAE latents into 2x2 spatial tokens for the DiT.

    Converts (B, C, H, W) latents to (B, seq_len, C*4) packed format.
    Each 2x2 spatial patch becomes one sequence token with 4x channels.

    Args:
        latents: Tensor of shape (B, 16, H, W) from VAE encoder.
            For multi-layer generation, shape is (B*layers, 16, H, W).
        height: Original image height in pixels (for computing latent dims).
        width: Original image width in pixels.

    Returns:
        Packed tensor of shape (B, H*W//4, 64) where:
        - B is batch size
        - H*W//4 is sequence length (one token per 2x2 patch)
        - 64 = 16 channels * 4 (2x2 patch)

    Example:
        >>> latents = torch.randn(1, 16, 64, 64)  # 512x512 image
        >>> packed = pack_latents_2x2(latents, 512, 512)
        >>> packed.shape
        torch.Size([1, 1024, 64])  # 1024 = 64*64/4
    """
    batch_size = latents.shape[0]
    channels = latents.shape[1]
    latent_h = latents.shape[2]
    latent_w = latents.shape[3]

    # Reshape to expose 2x2 patches
    # (B, C, H, W) -> (B, C, H/2, 2, W/2, 2)
    latents = latents.view(
        batch_size, channels, latent_h // 2, 2, latent_w // 2, 2
    )

    # Permute to group spatial patches together
    # (B, C, H/2, 2, W/2, 2) -> (B, H/2, W/2, C, 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)

    # Flatten patches into sequence
    # (B, H/2, W/2, C, 2, 2) -> (B, H*W/4, C*4)
    packed = latents.reshape(
        batch_size, (latent_h // 2) * (latent_w // 2), channels * 4
    )

    return packed


def unpack_latents_2x2(
    latents: torch.Tensor,
    height: int,
    width: int,
    vae_scale_factor: int = 8,
) -> torch.Tensor:
    """Unpack DiT output latents back to VAE format.

    Converts (B, seq_len, C*4) packed format back to (B, C, H, W) latents.

    Args:
        latents: Packed tensor of shape (B, H*W//4, 64).
        height: Original image height in pixels.
        width: Original image width in pixels.
        vae_scale_factor: VAE spatial compression factor (default 8).

    Returns:
        Unpacked tensor of shape (B, 16, H, W) for VAE decoder.

    Example:
        >>> packed = torch.randn(1, 1024, 64)  # 512x512 image
        >>> latents = unpack_latents_2x2(packed, 512, 512)
        >>> latents.shape
        torch.Size([1, 16, 64, 64])
    """
    batch_size = latents.shape[0]

    # Compute latent dimensions
    latent_h = height // vae_scale_factor
    latent_w = width // vae_scale_factor

    # Infer channel count from packed dimension (64 = 16 * 4)
    packed_channels = latents.shape[-1]
    channels = packed_channels // 4

    # Reshape sequence back to spatial grid
    # (B, H*W/4, C*4) -> (B, H/2, W/2, C, 2, 2)
    latents = latents.view(
        batch_size, latent_h // 2, latent_w // 2, channels, 2, 2
    )

    # Permute back to channel-first format
    # (B, H/2, W/2, C, 2, 2) -> (B, C, H/2, 2, W/2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)

    # Reshape to combine spatial dimensions
    # (B, C, H/2, 2, W/2, 2) -> (B, C, H, W)
    unpacked = latents.reshape(batch_size, channels, latent_h, latent_w)

    return unpacked


def pack_multi_layer_latents(
    latents: torch.Tensor,
    height: int,
    width: int,
    layer_num: int,
) -> torch.Tensor:
    """Pack multi-layer latents for layer decomposition.

    For layer decomposition, we generate multiple layers simultaneously.
    Each layer is processed as a separate "frame" in the sequence.

    Args:
        latents: Tensor of shape (layer_num+1, 16, H, W).
            The +1 is for the composite/base layer.
        height: Original image height in pixels.
        width: Original image width in pixels.
        layer_num: Number of decomposition layers (not including composite).

    Returns:
        Packed tensor of shape (1, (layer_num+1)*H*W//4, 64).

    Example:
        >>> latents = torch.randn(4, 16, 64, 64)  # 3 layers + composite
        >>> packed = pack_multi_layer_latents(latents, 512, 512, 3)
        >>> packed.shape
        torch.Size([1, 4096, 64])  # 4 * 1024 = 4096 tokens
    """
    total_layers = layer_num + 1  # Including composite layer
    assert latents.shape[0] == total_layers, (
        f"Expected {total_layers} layers, got {latents.shape[0]}"
    )

    # Pack each layer
    latent_h = latents.shape[2]
    latent_w = latents.shape[3]
    channels = latents.shape[1]

    # Reshape all layers at once
    # (layers, C, H, W) -> (layers, C, H/2, 2, W/2, 2)
    latents = latents.view(
        total_layers, channels, latent_h // 2, 2, latent_w // 2, 2
    )

    # Permute: (layers, C, H/2, 2, W/2, 2) -> (layers, H/2, W/2, C, 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)

    # Flatten to sequence, keep layers separate first
    # (layers, H/2, W/2, C, 2, 2) -> (layers, H*W/4, C*4)
    packed = latents.reshape(
        total_layers, (latent_h // 2) * (latent_w // 2), channels * 4
    )

    # Combine all layers into single sequence
    # (layers, seq_per_layer, C*4) -> (1, layers*seq_per_layer, C*4)
    packed = packed.view(1, -1, channels * 4)

    return packed


def unpack_multi_layer_latents(
    latents: torch.Tensor,
    height: int,
    width: int,
    layer_num: int,
    vae_scale_factor: int = 8,
) -> torch.Tensor:
    """Unpack multi-layer latents from DiT output.

    Args:
        latents: Packed tensor of shape (1, (layer_num+1)*H*W//4, 64).
        height: Original image height in pixels.
        width: Original image width in pixels.
        layer_num: Number of decomposition layers.
        vae_scale_factor: VAE spatial compression factor.

    Returns:
        Unpacked tensor of shape (layer_num+1, 16, H, W).

    Example:
        >>> packed = torch.randn(1, 4096, 64)
        >>> latents = unpack_multi_layer_latents(packed, 512, 512, 3)
        >>> latents.shape
        torch.Size([4, 16, 64, 64])
    """
    total_layers = layer_num + 1
    latent_h = height // vae_scale_factor
    latent_w = width // vae_scale_factor
    seq_per_layer = (latent_h // 2) * (latent_w // 2)
    packed_channels = latents.shape[-1]
    channels = packed_channels // 4

    # Split back into layers
    # (1, layers*seq, C*4) -> (layers, seq, C*4)
    latents = latents.view(total_layers, seq_per_layer, packed_channels)

    # Reshape to spatial grid
    # (layers, seq, C*4) -> (layers, H/2, W/2, C, 2, 2)
    latents = latents.view(
        total_layers, latent_h // 2, latent_w // 2, channels, 2, 2
    )

    # Permute back
    # (layers, H/2, W/2, C, 2, 2) -> (layers, C, H/2, 2, W/2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)

    # Reshape to final format
    # (layers, C, H/2, 2, W/2, 2) -> (layers, C, H, W)
    unpacked = latents.reshape(total_layers, channels, latent_h, latent_w)

    return unpacked


def compute_packed_sequence_length(
    height: int,
    width: int,
    vae_scale_factor: int = 8,
) -> int:
    """Compute the sequence length after 2x2 packing.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        vae_scale_factor: VAE spatial compression factor.

    Returns:
        Number of tokens after packing.

    Example:
        >>> compute_packed_sequence_length(512, 512)
        1024
        >>> compute_packed_sequence_length(1024, 1024)
        4096
    """
    latent_h = height // vae_scale_factor
    latent_w = width // vae_scale_factor
    return (latent_h // 2) * (latent_w // 2)


def get_img_shapes_for_rope(
    height: int,
    width: int,
    layer_num: int,
    vae_scale_factor: int = 8,
    include_condition: bool = False,
    condition_height: int = None,
    condition_width: int = None,
) -> list[Tuple[int, int, int]]:
    """Generate img_shapes list for RoPE position encoding.

    The Qwen-Image DiT uses multi-axis RoPE where each axis encodes different
    spatial information. The img_shapes list tells the RoPE module how to
    compute position embeddings for each "frame" (layer).

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        layer_num: Number of decomposition layers.
        vae_scale_factor: VAE spatial compression factor.
        include_condition: Whether to include condition image shape.
        condition_height: Height of condition image (defaults to height).
        condition_width: Width of condition image (defaults to width).

    Returns:
        List of (frame, height, width) tuples for RoPE computation.
        - For generation layers: (1, latent_h//2, latent_w//2) per layer
        - For condition image: (1, cond_h//2, cond_w//2) at the end
    """
    latent_h = height // vae_scale_factor // 2  # Half due to 2x2 packing
    latent_w = width // vae_scale_factor // 2

    # Each decomposition layer gets its own shape entry
    shapes = [(1, latent_h, latent_w) for _ in range(layer_num + 1)]

    if include_condition:
        cond_h = (condition_height or height) // vae_scale_factor // 2
        cond_w = (condition_width or width) // vae_scale_factor // 2
        shapes.append((1, cond_h, cond_w))

    return shapes
