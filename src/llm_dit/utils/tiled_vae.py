"""
Tiled VAE decode for large images.

Enables 2K+ generation without OOM by processing the latent in overlapping
tiles and blending the results smoothly.

Based on DiffSynth-Studio implementation (Apache 2.0 license).

Usage:
    from llm_dit.utils.tiled_vae import TiledVAEDecoder

    # Wrap existing VAE
    tiled_vae = TiledVAEDecoder(vae, tile_size=512, tile_overlap=64)

    # Decode - automatically tiles if needed
    image = tiled_vae.decode(latents)

    # Or use the wrapper function
    from llm_dit.utils.tiled_vae import decode_latents
    image = decode_latents(vae, latents, tile_size=512)
"""

import logging
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class TiledVAEDecoder:
    """
    Wrapper that adds tiled decode capability to any VAE.

    For images larger than tile_size, processes in overlapping tiles
    and blends the results smoothly using a gradient mask.

    Attributes:
        vae: The underlying VAE model
        tile_size: Tile size in pixel space (default: 512)
        tile_overlap: Overlap between tiles in pixels (default: 64)
        scale_factor: VAE downsampling factor (typically 8)
    """

    def __init__(
        self,
        vae,
        tile_size: int = 512,
        tile_overlap: int = 64,
        scale_factor: Optional[int] = None,
    ):
        """
        Initialize the tiled VAE decoder.

        Args:
            vae: VAE model with decode() method
            tile_size: Tile size in pixel space
            tile_overlap: Overlap between tiles in pixels
            scale_factor: VAE downsampling factor (auto-detected if None)
        """
        self.vae = vae
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap

        # Auto-detect scale factor from VAE config
        if scale_factor is not None:
            self.scale_factor = scale_factor
        elif hasattr(vae, "config") and hasattr(vae.config, "block_out_channels"):
            self.scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        else:
            # Default for most VAEs
            self.scale_factor = 8

        # Latent space tile parameters
        self.latent_tile_size = tile_size // self.scale_factor
        self.latent_overlap = tile_overlap // self.scale_factor
        self.latent_stride = self.latent_tile_size - self.latent_overlap

        logger.debug(
            f"TiledVAEDecoder: tile_size={tile_size}, overlap={tile_overlap}, "
            f"scale_factor={self.scale_factor}"
        )

    @property
    def config(self):
        """Pass through to underlying VAE config."""
        return self.vae.config

    @property
    def dtype(self):
        """Pass through to underlying VAE dtype."""
        return next(self.vae.parameters()).dtype

    @property
    def device(self):
        """Pass through to underlying VAE device."""
        return next(self.vae.parameters()).device

    def decode(
        self,
        latents: torch.Tensor,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        """
        Decode latents to image, using tiled decode if necessary.

        Args:
            latents: Latent tensor, shape (B, C, H, W)
            return_dict: If True, return dict with 'sample' key

        Returns:
            Decoded image tensor, shape (B, 3, H*scale, W*scale)
        """
        _, _, h, w = latents.shape

        # Check if tiling needed
        if h <= self.latent_tile_size and w <= self.latent_tile_size:
            logger.debug(f"Latents {h}x{w} fit in tile, using direct decode")
            result = self.vae.decode(latents, return_dict=False)
            if isinstance(result, tuple):
                result = result[0]
            if return_dict:
                return {"sample": result}
            return result

        logger.info(
            f"Latents {h}x{w} exceed tile size {self.latent_tile_size}, "
            f"using tiled decode"
        )
        result = self._tiled_decode(latents)

        if return_dict:
            return {"sample": result}
        return result

    def _tiled_decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latents using overlapping tiles with gradient blending.

        Args:
            latents: Latent tensor, shape (B, C, H, W)

        Returns:
            Decoded image tensor
        """
        batch, channels, height, width = latents.shape
        device = latents.device
        dtype = latents.dtype

        # Output dimensions
        out_height = height * self.scale_factor
        out_width = width * self.scale_factor
        out_tile = self.latent_tile_size * self.scale_factor
        out_overlap = self.latent_overlap * self.scale_factor

        # Initialize output and weight accumulator
        # Use float32 for accumulation to avoid precision issues
        output = torch.zeros(
            batch, 3, out_height, out_width, device=device, dtype=torch.float32
        )
        weights = torch.zeros(
            1, 1, out_height, out_width, device=device, dtype=torch.float32
        )

        # Create blending mask
        blend_mask = self._create_blend_mask(out_tile, out_overlap, device)

        # Count tiles for logging
        n_tiles_h = max(1, (height - self.latent_overlap) // self.latent_stride)
        n_tiles_w = max(1, (width - self.latent_overlap) // self.latent_stride)
        total_tiles = n_tiles_h * n_tiles_w
        tile_idx = 0

        # Process tiles
        for y in range(0, height - self.latent_overlap + 1, self.latent_stride):
            for x in range(0, width - self.latent_overlap + 1, self.latent_stride):
                # Handle edge tiles - adjust to include full tile
                y_end = min(y + self.latent_tile_size, height)
                x_end = min(x + self.latent_tile_size, width)
                y_start = max(0, y_end - self.latent_tile_size)
                x_start = max(0, x_end - self.latent_tile_size)

                # Actual tile dimensions (may be smaller at edges)
                tile_h = y_end - y_start
                tile_w = x_end - x_start

                # Extract tile
                tile = latents[:, :, y_start:y_end, x_start:x_end]

                # Pad if smaller than expected (edge case)
                if tile_h < self.latent_tile_size or tile_w < self.latent_tile_size:
                    pad_h = self.latent_tile_size - tile_h
                    pad_w = self.latent_tile_size - tile_w
                    tile = F.pad(tile, (0, pad_w, 0, pad_h), mode="reflect")

                # Decode tile
                with torch.no_grad():
                    decoded = self.vae.decode(tile, return_dict=False)
                    if isinstance(decoded, tuple):
                        decoded = decoded[0]

                # Crop if we padded
                if tile_h < self.latent_tile_size or tile_w < self.latent_tile_size:
                    decoded = decoded[
                        :, :, : tile_h * self.scale_factor, : tile_w * self.scale_factor
                    ]

                # Output coordinates
                oy = y_start * self.scale_factor
                ox = x_start * self.scale_factor
                oh = decoded.shape[2]
                ow = decoded.shape[3]

                # Get appropriate mask slice
                tile_mask = blend_mask[:oh, :ow]

                # Accumulate with blending
                output[:, :, oy : oy + oh, ox : ox + ow] += decoded.float() * tile_mask
                weights[:, :, oy : oy + oh, ox : ox + ow] += tile_mask

                tile_idx += 1
                if tile_idx % 4 == 0 or tile_idx == total_tiles:
                    logger.debug(f"Decoded tile {tile_idx}/{total_tiles}")

        # Normalize by weights
        output = output / weights.clamp(min=1e-8)

        # Convert back to original dtype
        return output.to(dtype)

    def _create_blend_mask(
        self, size: int, overlap: int, device: torch.device
    ) -> torch.Tensor:
        """
        Create a gradient blending mask for tile overlap regions.

        The mask has value 1.0 in the center and linearly ramps down
        to 0.0 at the edges within the overlap region.

        Args:
            size: Tile size in pixels
            overlap: Overlap size in pixels
            device: Device for the mask tensor

        Returns:
            Mask tensor of shape (size, size)
        """
        mask = torch.ones(size, size, device=device, dtype=torch.float32)

        if overlap <= 0:
            return mask

        # Create linear ramp from 0 to 1
        ramp = torch.linspace(0, 1, overlap, device=device)

        # Apply to all four edges
        # Top edge
        mask[:overlap, :] *= ramp.view(-1, 1)
        # Bottom edge
        mask[-overlap:, :] *= ramp.flip(0).view(-1, 1)
        # Left edge
        mask[:, :overlap] *= ramp.view(1, -1)
        # Right edge
        mask[:, -overlap:] *= ramp.flip(0).view(1, -1)

        return mask


def decode_latents(
    vae,
    latents: torch.Tensor,
    tile_size: int = 512,
    tile_overlap: int = 64,
    scaling_factor: Optional[float] = None,
    shift_factor: Optional[float] = None,
) -> torch.Tensor:
    """
    Convenience function to decode latents with optional tiling.

    Args:
        vae: VAE model
        latents: Latent tensor
        tile_size: Tile size for large images
        tile_overlap: Overlap between tiles
        scaling_factor: VAE scaling factor (auto-detected if None)
        shift_factor: VAE shift factor (auto-detected if None)

    Returns:
        Decoded image tensor, normalized to [0, 1]
    """
    # Apply VAE scaling/shift
    if scaling_factor is None and hasattr(vae, "config"):
        scaling_factor = getattr(vae.config, "scaling_factor", 0.18215)
    if shift_factor is None and hasattr(vae, "config"):
        shift_factor = getattr(vae.config, "shift_factor", 0.0)

    if scaling_factor:
        latents = latents / scaling_factor
    if shift_factor:
        latents = latents + shift_factor

    # Check if tiling needed
    _, _, h, w = latents.shape
    scale_factor = 8
    if hasattr(vae, "config") and hasattr(vae.config, "block_out_channels"):
        scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

    latent_tile_size = tile_size // scale_factor

    if h > latent_tile_size or w > latent_tile_size:
        # Use tiled decode
        tiled_vae = TiledVAEDecoder(vae, tile_size, tile_overlap)
        image = tiled_vae.decode(latents)
    else:
        # Direct decode
        with torch.no_grad():
            image = vae.decode(latents, return_dict=False)
            if isinstance(image, tuple):
                image = image[0]

    # Normalize to [0, 1]
    image = (image / 2 + 0.5).clamp(0, 1)
    return image


def estimate_vae_memory(
    height: int,
    width: int,
    dtype: torch.dtype = torch.float16,
    scale_factor: int = 8,
) -> dict:
    """
    Estimate VRAM usage for VAE decode.

    Args:
        height: Output image height
        width: Output image width
        dtype: Model dtype
        scale_factor: VAE downsampling factor

    Returns:
        Dict with memory estimates in GB
    """
    bytes_per_element = 2 if dtype in (torch.float16, torch.bfloat16) else 4

    latent_h = height // scale_factor
    latent_w = width // scale_factor

    # Latent size
    latent_bytes = 1 * 16 * latent_h * latent_w * bytes_per_element

    # Output size
    output_bytes = 1 * 3 * height * width * bytes_per_element

    # Activations (rough estimate - VAE is relatively small)
    activation_bytes = latent_bytes * 4

    return {
        "latent_gb": latent_bytes / 1e9,
        "output_gb": output_bytes / 1e9,
        "activation_gb": activation_bytes / 1e9,
        "total_gb": (latent_bytes + output_bytes + activation_bytes) / 1e9,
    }
