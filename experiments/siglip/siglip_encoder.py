"""
last updated: 2025-12-19

SigLIP2 Vision Encoder for Z-Image Omni.
Based on diffusers PR 12857 prepare_siglip_embeds().
"""

from pathlib import Path
from typing import List, Optional, Union

import torch
from PIL import Image


class SigLIPEncoder:
    """
    SigLIP2 vision encoder for extracting image embeddings.

    Based on diffusers PR 12857 pipeline_z_image_omni.py
    """

    def __init__(
        self,
        model_path: str = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        from transformers import AutoModel, AutoProcessor, AutoConfig

        if model_path is None:
            model_path = str(Path.home() / "Storage/google_siglip2-so400m-patch14-384")

        self.model_path = model_path
        self.device = device
        self.dtype = dtype

        # Check model type
        config = AutoConfig.from_pretrained(model_path)
        model_type = config.model_type
        print(f"Loading SigLIP ({model_type}) from {model_path}")

        # Load the full model and extract vision model
        full_model = AutoModel.from_pretrained(model_path)
        self.model = full_model.vision_model.to(device)
        self.processor = AutoProcessor.from_pretrained(model_path)

        # Get hidden dim from config
        self.hidden_size = self.model.config.hidden_size
        print(f"SigLIP2 hidden size: {self.hidden_size}")

    def encode(
        self,
        images: Union[Image.Image, List[Image.Image]],
    ) -> List[torch.Tensor]:
        """
        Encode images to SigLIP embeddings.

        Follows diffusers PR 12857 prepare_siglip_embeds() exactly.

        Args:
            images: Single image or list of images

        Returns:
            List of tensors, each (H, W, hidden_size) spatial grid
        """
        if isinstance(images, Image.Image):
            images = [images]

        embeddings = []

        for image in images:
            # Process image
            siglip_inputs = self.processor(images=[image], return_tensors="pt")

            # Move to device and extract only pixel_values for vision model
            pixel_values = siglip_inputs["pixel_values"].to(self.device)

            # Check if spatial_shapes available (Siglip2 only)
            spatial_shapes = siglip_inputs.get("spatial_shapes", None)

            # Forward pass
            with torch.no_grad():
                hidden_state = self.model(pixel_values=pixel_values).last_hidden_state
                # Shape: (1, num_patches, hidden_size)

            B, N, C = hidden_state.shape

            if spatial_shapes is not None:
                # Siglip2: use spatial_shapes
                shape = spatial_shapes[0]
                hidden_state = hidden_state[:, :shape[0] * shape[1]]
                hidden_state = hidden_state.view(shape[0], shape[1], C)
            else:
                # Standard SigLIP: infer spatial shape from num_patches
                # For 224x224 with 16x16 patches: 14x14 = 196 patches
                H = W = int(N ** 0.5)
                hidden_state = hidden_state.squeeze(0)  # Remove batch
                hidden_state = hidden_state.view(H, W, C)

            embeddings.append(hidden_state.to(self.dtype))

        return embeddings

    def encode_batch(
        self,
        images: List[Image.Image],
        batch_size: int = 1,
    ) -> List[List[torch.Tensor]]:
        """
        Encode images and replicate for batch.

        Returns:
            List of lists: [[emb1, emb2, ...], [emb1, emb2, ...], ...] for batch
        """
        embeddings = self.encode(images)
        return [embeddings.copy() for _ in range(batch_size)]

    def unload(self):
        """Unload model to free memory."""
        del self.model
        del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    # Quick test
    print("Testing SigLIP encoder...")
    encoder = SigLIPEncoder()

    # Test with a simple image
    img = Image.new("RGB", (512, 512), (128, 64, 192))
    embeddings = encoder.encode(img)

    print(f"\nResults:")
    print(f"  Shape: {embeddings[0].shape}")
    print(f"  Mean: {embeddings[0].mean():.4f}")
    print(f"  Std: {embeddings[0].std():.4f}")
    print(f"  Expected: (H_patches, W_patches, {encoder.hidden_size})")
