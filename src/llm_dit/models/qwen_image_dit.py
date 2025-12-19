"""
Qwen-Image DiT transformer wrapper.

The Qwen-Image-Layered model uses a 60-layer dual-stream DiT with:
- Inner dimension: 3072
- Attention heads: 24 @ 128 dim/head
- Text input: 3584 -> 3072 projection
- Image input: 64 -> 3072 projection (after 2x2 packing)
- 3-axis RoPE with dims [16, 56, 56] (layer/frame, height, width)

The DiT uses dual-stream attention where text and image tokens are
processed jointly but with separate projections.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from einops import rearrange

logger = logging.getLogger(__name__)


class QwenImageDiT(nn.Module):
    """
    Wrapper for the Qwen-Image DiT transformer.

    This wrapper provides a simplified interface for running inference
    with the Qwen-Image-Layered DiT model.

    Attributes:
        num_layers: Number of transformer blocks (60)
        inner_dim: Hidden dimension (3072)
        num_heads: Number of attention heads (24)
        head_dim: Dimension per head (128)

    Example:
        dit = QwenImageDiT.from_pretrained("/path/to/Qwen_Qwen-Image-Layered")
        noise_pred = dit(
            latents=packed_latents,
            timestep=t,
            prompt_embeds=text_embeddings,
            prompt_mask=attention_mask,
            height=1024,
            width=1024,
        )
    """

    # Architecture constants
    NUM_LAYERS = 60
    INNER_DIM = 3072
    NUM_HEADS = 24
    HEAD_DIM = 128
    TEXT_DIM = 3584
    LATENT_DIM = 64  # After 2x2 packing (16 * 4)

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """
        Initialize the DiT wrapper.

        Args:
            model: The underlying QwenImageDiT model
            device: Device for computation
            dtype: Data type
        """
        super().__init__()
        self.model = model
        self._device = device
        self._dtype = dtype

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        transformer_subfolder: str = "transformer",
        device: str | torch.device = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        use_layer3d_rope: bool = True,
        **kwargs,
    ) -> "QwenImageDiT":
        """
        Load DiT from pretrained Qwen-Image-Layered model.

        Args:
            model_path: Path to Qwen-Image-Layered model directory
            transformer_subfolder: Subfolder containing transformer weights
            device: Device to load model on
            torch_dtype: Model dtype
            use_layer3d_rope: Use layer-aware 3D RoPE for multi-layer decomposition

        Returns:
            Initialized QwenImageDiT
        """
        model_path = Path(model_path)
        device = torch.device(device)

        # Import model components
        from llm_dit.models._qwen_image_dit_components import QwenImageDiTModel

        # Create model
        logger.info(f"Creating QwenImageDiT (num_layers={cls.NUM_LAYERS}, use_layer3d_rope={use_layer3d_rope})")
        model = QwenImageDiTModel(
            num_layers=cls.NUM_LAYERS,
            use_layer3d_rope=use_layer3d_rope,
        )

        # Load weights
        transformer_path = model_path / transformer_subfolder
        weight_files = list(transformer_path.glob("*.safetensors"))
        if not weight_files:
            raise ValueError(f"No safetensors files found in {transformer_path}")

        logger.info(f"Loading DiT weights from {len(weight_files)} safetensors files")
        from safetensors.torch import load_file

        state_dict = {}
        for weight_file in sorted(weight_files):
            logger.debug(f"Loading {weight_file.name}")
            file_state_dict = load_file(weight_file, device="cpu")
            state_dict.update(file_state_dict)

        # Load state dict
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"Missing keys: {missing[:5]}... ({len(missing)} total)")
        if unexpected:
            logger.debug(f"Unexpected keys: {unexpected[:5]}...")

        # Move to device and dtype
        model = model.to(device=device, dtype=torch_dtype)
        model.eval()

        logger.info(
            f"Loaded QwenImageDiT: {cls.NUM_LAYERS} layers, "
            f"dim={cls.INNER_DIM}, heads={cls.NUM_HEADS}, "
            f"device={device}, dtype={torch_dtype}"
        )

        return cls(model, device, torch_dtype)

    @property
    def device(self) -> torch.device:
        """Return model device."""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Return model dtype."""
        return self._dtype

    def forward(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor,
        height: int,
        width: int,
        img_shapes: Optional[List[Tuple[int, int, int]]] = None,
        layer_num: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Run DiT forward pass.

        Args:
            latents: Packed latent tensor of shape (B, seq_len, 64)
            timestep: Timestep tensor of shape (B,)
            prompt_embeds: Text embeddings of shape (B, seq_len, 3584)
            prompt_mask: Attention mask of shape (B, seq_len)
            height: Image height in pixels
            width: Image width in pixels
            img_shapes: Optional list of (frame, height, width) for RoPE.
                       If None, computed from height/width.
            layer_num: Number of decomposition layers (for multi-layer output)

        Returns:
            Noise prediction of shape (B, seq_len, 64)
        """
        # If latents are in spatial format, pack them
        if latents.dim() == 4:
            # (B, C, H, W) -> (B, seq, C*4)
            latents = rearrange(
                latents,
                "B C (H P) (W Q) -> B (H W) (C P Q)",
                H=height // 16,
                W=width // 16,
                P=2,
                Q=2,
            )

        # Compute img_shapes if not provided
        if img_shapes is None:
            latent_h = height // 16
            latent_w = width // 16
            if layer_num is not None:
                # Multi-layer: one shape per layer plus composite
                img_shapes = [(1, latent_h, latent_w) for _ in range(layer_num + 1)]
            else:
                img_shapes = [(latents.shape[0], latent_h, latent_w)]

        # Forward through model
        output = self.model(
            latents=latents,
            timestep=timestep,
            prompt_emb=prompt_embeds,
            prompt_emb_mask=prompt_mask,
            height=height,
            width=width,
            img_shapes=img_shapes,
        )

        return output

    def to(self, device: torch.device) -> "QwenImageDiT":
        """Move model to device."""
        self.model = self.model.to(device)
        self._device = device
        return self


def _unpack_latents(latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """
    Unpack latents from sequence format back to spatial format.

    Args:
        latents: Packed tensor of shape (B, H*W/4, C*4)
        height: Image height in pixels
        width: Image width in pixels

    Returns:
        Unpacked tensor of shape (B, C, H, W)
    """
    return rearrange(
        latents,
        "B (H W) (C P Q) -> B C (H P) (W Q)",
        H=height // 16,
        W=width // 16,
        P=2,
        Q=2,
    )
