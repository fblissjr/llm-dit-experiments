"""
Qwen-Image VAE wrapper for encoding and decoding latents.

The Qwen-Image-Layered model uses a 3D causal VAE derived from Wan-family.
Key characteristics:
- 16 latent channels (same as Z-Image)
- 3D causal convolutions for temporal consistency (video-ready)
- Per-channel mean/std normalization
- For single images: adds/removes temporal dimension automatically

Architecture:
    Input Image (B, 3, H, W)
        -> unsqueeze(2) -> (B, 3, 1, H, W)
        -> Encoder3D -> (B, 32, 1, H/8, W/8)
        -> quant_conv -> (B, 32, 1, H/8, W/8)
        -> [:, :16] -> (B, 16, 1, H/8, W/8)
        -> normalize (per-channel mean/std)
        -> squeeze(2) -> (B, 16, H/8, W/8)
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Latent normalization values (per-channel)
# These are critical for correct encoding/decoding
LATENT_MEAN = [
    -0.7571, -0.7089, -0.9113, 0.1075,
    -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632,
    -0.1922, -0.9497, 0.2503, -0.2921,
]

LATENT_STD = [
    2.8184, 1.4541, 2.3275, 2.6558,
    1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579,
    1.6382, 1.1253, 2.8251, 1.9160,
]


class QwenImageVAE(nn.Module):
    """
    Wrapper for the Qwen-Image 3D Causal VAE.

    This wrapper provides a simple interface for encoding images to latents
    and decoding latents back to images, handling the 3D temporal dimension
    automatically for single images.

    Attributes:
        z_dim: Latent channel dimension (16)
        scale_factor: Spatial compression factor (8x downscaling)

    Example:
        vae = QwenImageVAE.from_pretrained("/path/to/Qwen_Qwen-Image-Layered")
        latents = vae.encode(images)  # (B, 3, H, W) -> (B, 16, H/8, W/8)
        decoded = vae.decode(latents)  # (B, 16, H/8, W/8) -> (B, 3, H, W)
    """

    # Architecture constants
    BASE_DIM = 96
    Z_DIM = 16
    DIM_MULT = [1, 2, 4, 4]
    NUM_RES_BLOCKS = 2
    SCALE_FACTOR = 8  # 8x spatial downscaling

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        quant_conv: nn.Module,
        post_quant_conv: nn.Module,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """
        Initialize the VAE wrapper.

        Args:
            encoder: 3D encoder module
            decoder: 3D decoder module
            quant_conv: Quantization convolution
            post_quant_conv: Post-quantization convolution
            device: Device for computation
            dtype: Data type
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quant_conv = quant_conv
        self.post_quant_conv = post_quant_conv

        self._device = device
        self._dtype = dtype

        # Register normalization buffers
        mean = torch.tensor(LATENT_MEAN).view(1, 16, 1, 1, 1)
        std_inv = 1.0 / torch.tensor(LATENT_STD).view(1, 16, 1, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std_inv", std_inv)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        vae_subfolder: str = "vae",
        device: str | torch.device = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> "QwenImageVAE":
        """
        Load VAE from pretrained Qwen-Image-Layered model.

        Args:
            model_path: Path to Qwen-Image-Layered model directory
            vae_subfolder: Subfolder containing VAE weights
            device: Device to load model on
            torch_dtype: Model dtype

        Returns:
            Initialized QwenImageVAE
        """
        model_path = Path(model_path)
        device = torch.device(device)

        # Import the model components
        # We use our own simplified implementation based on DiffSynth
        from llm_dit.models._qwen_image_vae_components import (
            QwenImageEncoder3d,
            QwenImageDecoder3d,
            QwenImageCausalConv3d,
        )

        # Create model components with default Qwen-Image architecture
        encoder = QwenImageEncoder3d(
            dim=cls.BASE_DIM,
            z_dim=cls.Z_DIM * 2,  # Outputs 32 channels, we take first 16
            dim_mult=cls.DIM_MULT,
            num_res_blocks=cls.NUM_RES_BLOCKS,
            attn_scales=[],
            temperal_downsample=[False, True, True],
            dropout=0.0,
            image_channels=4,  # RGBA for layer decomposition
        )

        decoder = QwenImageDecoder3d(
            dim=cls.BASE_DIM,
            z_dim=cls.Z_DIM,
            dim_mult=cls.DIM_MULT,
            num_res_blocks=cls.NUM_RES_BLOCKS,
            attn_scales=[],
            temperal_upsample=[True, True, False],
            dropout=0.0,
            image_channels=4,  # RGBA for layer decomposition
        )

        quant_conv = QwenImageCausalConv3d(cls.Z_DIM * 2, cls.Z_DIM * 2, 1)
        post_quant_conv = QwenImageCausalConv3d(cls.Z_DIM, cls.Z_DIM, 1)

        # Load weights
        vae_path = model_path / vae_subfolder
        weight_files = list(vae_path.glob("*.safetensors"))
        if not weight_files:
            raise ValueError(f"No safetensors files found in {vae_path}")

        logger.info(f"Loading VAE weights from {vae_path}")
        from safetensors.torch import load_file

        state_dict = {}
        for weight_file in sorted(weight_files):
            logger.debug(f"Loading {weight_file.name}")
            file_state_dict = load_file(weight_file, device="cpu")
            state_dict.update(file_state_dict)

        # Create temporary full model to load state dict
        temp_model = _QwenImageVAEFull(encoder, decoder, quant_conv, post_quant_conv)

        # Load weights - handle different key prefixes
        missing, unexpected = temp_model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"Missing keys: {missing[:5]}... ({len(missing)} total)")
        if unexpected:
            logger.debug(f"Unexpected keys: {unexpected[:5]}...")

        # Move to device and dtype
        encoder = encoder.to(device=device, dtype=torch_dtype)
        decoder = decoder.to(device=device, dtype=torch_dtype)
        quant_conv = quant_conv.to(device=device, dtype=torch_dtype)
        post_quant_conv = post_quant_conv.to(device=device, dtype=torch_dtype)

        encoder.eval()
        decoder.eval()
        quant_conv.eval()
        post_quant_conv.eval()

        logger.info(
            f"Loaded Qwen-Image VAE: z_dim={cls.Z_DIM}, "
            f"scale_factor={cls.SCALE_FACTOR}, device={device}, dtype={torch_dtype}"
        )

        return cls(encoder, decoder, quant_conv, post_quant_conv, device, torch_dtype)

    @property
    def device(self) -> torch.device:
        """Return model device."""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Return model dtype."""
        return self._dtype

    @property
    def scale_factor(self) -> int:
        """Return spatial scale factor (8x)."""
        return self.SCALE_FACTOR

    @property
    def latent_channels(self) -> int:
        """Return number of latent channels (16)."""
        return self.Z_DIM

    def encode(
        self,
        images: torch.Tensor,
        return_distribution: bool = False,
    ) -> torch.Tensor:
        """
        Encode images to normalized latents.

        Args:
            images: Input images of shape (B, 3, H, W) in range [-1, 1]
            return_distribution: If True, return (mean, logvar) instead of just mean

        Returns:
            Latent tensor of shape (B, 16, H/8, W/8), normalized
        """
        # Add temporal dimension for 3D VAE
        # (B, C, H, W) -> (B, C, 1, H, W)
        x = images.unsqueeze(2)

        # Encode
        with torch.no_grad():
            x = self.encoder(x)
            x = self.quant_conv(x)

        # Take first 16 channels (mean of distribution)
        x = x[:, :16]

        # Apply normalization: (x - mean) * (1/std)
        x = (x - self.mean.to(x)) * self.std_inv.to(x)

        # Remove temporal dimension
        x = x.squeeze(2)

        return x

    def decode(
        self,
        latents: torch.Tensor,
        num_layers: int | None = None,
    ) -> torch.Tensor | List[torch.Tensor]:
        """
        Decode latents to images.

        Args:
            latents: Latent tensor of shape (B, 16, H, W) or (layers+1, 16, H, W)
            num_layers: If provided, split output into list of layer images

        Returns:
            Decoded images of shape (B, 3, H*8, W*8) in range [-1, 1],
            or list of images if num_layers is provided
        """
        # Add temporal dimension
        x = latents.unsqueeze(2)

        # Denormalize: x / (1/std) + mean = x * std + mean
        std = 1.0 / self.std_inv.to(x)
        x = x * std + self.mean.to(x)

        # Decode
        with torch.no_grad():
            x = self.post_quant_conv(x)
            x = self.decoder(x)

        # Remove temporal dimension
        x = x.squeeze(2)

        # If multi-layer output, split into list
        if num_layers is not None:
            total = num_layers + 1  # layers + composite
            if x.shape[0] == total:
                return [x[i] for i in range(total)]

        return x

    def to(self, device: torch.device) -> "QwenImageVAE":
        """Move model to device."""
        super().to(device)
        self._device = device
        return self


class _QwenImageVAEFull(nn.Module):
    """Temporary container for loading state dict."""

    def __init__(self, encoder, decoder, quant_conv, post_quant_conv):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quant_conv = quant_conv
        self.post_quant_conv = post_quant_conv
