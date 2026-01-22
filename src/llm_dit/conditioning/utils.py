"""
Conditioning system utility functions.

Last Updated: 2026-01-18

Provides helper functions for the conditioning system:
- timesteps_from_mask: Compute per-token timesteps from denoise mask
- post_process_latent: Blend denoised output with clean latent
- patchify_latent: Convert 5D latent to token sequence
- create_keyframe_positions: Create positions for appended keyframes
"""

from __future__ import annotations

import torch


def timesteps_from_mask(
    denoise_mask: torch.Tensor,
    sigma: float | torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-token timesteps from denoise mask and sigma.

    LTX-2 uses timesteps in [0, 1000] range. The denoise_mask scales
    the timestep per-token, allowing conditioned regions to have lower
    timesteps (less denoising).

    Args:
        denoise_mask: [B, T, 1] mask where 1.0 = full denoising, 0.0 = no denoising
        sigma: Current noise level (scalar or tensor)

    Returns:
        Timesteps tensor [B, T, 1] in [0, 1000] range
    """
    return denoise_mask * sigma * 1000


def post_process_latent(
    denoised: torch.Tensor,
    denoise_mask: torch.Tensor,
    clean: torch.Tensor,
) -> torch.Tensor:
    """
    Blend denoised output with clean latent based on mask.

    Formula: output = denoised * mask + clean * (1 - mask)

    Where mask=1.0 means full denoising (output=denoised),
    and mask=0.0 means no denoising (output=clean).

    Args:
        denoised: [B, T, D] denoised latent from model
        denoise_mask: [B, T, 1] blending mask (broadcasts to D)
        clean: [B, T, D] clean reference latent

    Returns:
        Blended latent [B, T, D]
    """
    return (denoised * denoise_mask + clean.float() * (1 - denoise_mask)).to(denoised.dtype)


def patchify_latent(latent: torch.Tensor) -> torch.Tensor:
    """
    Convert 5D latent tensor to 3D token sequence.

    Args:
        latent: [B, C, F, H, W] latent tensor in VAE space

    Returns:
        Tokens [B, T, D] where T = F * H * W and D = C
    """
    batch, channels, frames, height, width = latent.shape
    # Reshape: [B, C, F, H, W] -> [B, F*H*W, C]
    tokens = latent.permute(0, 2, 3, 4, 1)  # [B, F, H, W, C]
    tokens = tokens.reshape(batch, frames * height * width, channels)  # [B, T, C]
    return tokens


def unpatchify_latent(
    tokens: torch.Tensor,
    frames: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """
    Convert 3D token sequence back to 5D latent tensor.

    Args:
        tokens: [B, T, D] token sequence
        frames: Number of latent frames
        height: Latent height
        width: Latent width

    Returns:
        Latent [B, C, F, H, W] where C = D
    """
    batch, _, channels = tokens.shape
    # Reshape: [B, T, C] -> [B, F, H, W, C] -> [B, C, F, H, W]
    latent = tokens.reshape(batch, frames, height, width, channels)
    latent = latent.permute(0, 4, 1, 2, 3)  # [B, C, F, H, W]
    return latent


def create_position_bounds(
    frames: int,
    height: int,
    width: int,
    device: torch.device,
    fps: float = 24.0,
    frame_offset: int = 0,
) -> torch.Tensor:
    """
    Create position bounds for latent tokens.

    LTX-2 uses position bounds [start, end) for each patch, enabling
    temporal interpolation with RoPE. The model computes the middle
    point of each patch's bounds.

    Args:
        frames: Number of latent frames
        height: Latent height
        width: Latent width
        device: Target device
        fps: Frames per second for temporal scaling
        frame_offset: Temporal offset for appended keyframes

    Returns:
        Position bounds [1, 3, T, 2] where:
        - T = frames * height * width
        - Last dim is [start, end] bounds
        - First dim (positions[:, 0]) is temporal, scaled to seconds
    """
    # Create meshgrid indices
    t_indices = torch.arange(frames, device=device, dtype=torch.float32)
    h_indices = torch.arange(height, device=device, dtype=torch.float32)
    w_indices = torch.arange(width, device=device, dtype=torch.float32)

    # Create 3D grid: (t, h, w)
    grid_t, grid_h, grid_w = torch.meshgrid(t_indices, h_indices, w_indices, indexing='ij')

    # Apply frame offset to temporal dimension
    grid_t = grid_t + frame_offset

    # Stack dimensions and create bounds
    patch_starts = torch.stack([grid_t, grid_h, grid_w], dim=0)  # [3, F, H, W]
    patch_ends = patch_starts + 1.0

    # Stack start/end into bounds: [3, F, H, W, 2]
    positions = torch.stack([patch_starts, patch_ends], dim=-1)

    # Flatten spatial dims: [3, T, 2]
    num_tokens = frames * height * width
    positions = positions.view(3, num_tokens, 2)

    # Scale temporal positions to seconds
    positions[0] = positions[0] / fps

    # Add batch dimension: [1, 3, T, 2]
    return positions.unsqueeze(0)


def get_tokens_per_frame(height: int, width: int) -> int:
    """
    Calculate number of tokens per latent frame.

    Args:
        height: Latent height
        width: Latent width

    Returns:
        Tokens per frame (height * width)
    """
    return height * width


def validate_spatial_match(
    cond_shape: tuple[int, ...],
    target_height: int,
    target_width: int,
) -> bool:
    """
    Validate that conditioning latent spatial dimensions match target.

    Args:
        cond_shape: Shape of conditioning latent [B, C, F, H, W]
        target_height: Expected latent height
        target_width: Expected latent width

    Returns:
        True if shapes match, False otherwise
    """
    if len(cond_shape) != 5:
        return False
    _, _, _, cond_height, cond_width = cond_shape
    return cond_height == target_height and cond_width == target_width
