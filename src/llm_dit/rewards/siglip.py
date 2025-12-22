"""Differentiable SigLIP2 wrapper for reward computation.

The image processing path is gradient-enabled, allowing backprop
through the reward to the input latents for FMTT.

Usage:
    from llm_dit.rewards import DifferentiableSigLIP

    reward_fn = DifferentiableSigLIP(device="cuda")

    # image: tensor in [-1, 1], shape (B, 3, H, W), requires_grad=True
    reward = reward_fn.compute_reward(image, "A cat sleeping")
    reward.backward()  # Gradients flow to image
"""

import logging
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Default model - 2B params, best quality
DEFAULT_MODEL = "google/siglip2-giant-opt-patch16-384"


class DifferentiableSigLIP:
    """SigLIP2 reward function with gradient-enabled image path.

    Architecture:
        google/siglip2-giant-opt-patch16-384 (2B params)
        - Vision encoder: ViT-Giant, patch16, 384px input
        - Text encoder: Shared with vision, 1024 dim
        - Output: Cosine similarity in [-1, 1]

    Memory estimate (bf16):
        - Model weights: ~4GB
        - Activations for 1024x1024 -> 384x384: ~500MB
        - Gradient storage: ~2GB (for image path only)
        Total: ~6.5GB

    Limitations:
        - Text input limited to 64 tokens (SigLIP constraint)
        - Longer prompts are automatically truncated
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        """Initialize differentiable SigLIP2 reward function.

        Args:
            model_name: HuggingFace model ID
            device: Device for computation
            dtype: Model dtype (bf16 recommended)
        """
        from transformers import AutoModel, AutoProcessor

        logger.info(f"Loading SigLIP2 model: {model_name}")

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device, dtype)
        self.model.eval()

        self.device = device
        self.dtype = dtype

        # Get image size from processor config
        self.image_size = self.processor.image_processor.size.get("height", 384)

        # Cache for text embeddings (constant per prompt)
        self._text_cache: dict[str, torch.Tensor] = {}

        logger.info(f"SigLIP2 loaded: {self.image_size}px input, {dtype}")

    def _get_text_embedding(self, prompt: str) -> torch.Tensor:
        """Get cached or compute text embedding (no gradients needed).

        Text embeddings are constant for a given prompt, so we cache them
        and compute without gradients to save memory.

        Note: SigLIP has a 64 token limit. Longer prompts are truncated.
        """
        if prompt not in self._text_cache:
            text_inputs = self.processor(
                text=[prompt],
                return_tensors="pt",
                padding=True,
                truncation=True,  # SigLIP max is 64 tokens
                max_length=64,
            ).to(self.device)

            with torch.no_grad():
                text_embeds = self.model.get_text_features(**text_inputs)
                text_embeds = F.normalize(text_embeds, p=2, dim=-1)

            self._text_cache[prompt] = text_embeds

        return self._text_cache[prompt]

    def _differentiable_preprocess(self, image: torch.Tensor) -> torch.Tensor:
        """Differentiable image preprocessing.

        Input: image in [-1, 1] range, shape (B, C, H, W)
        Output: normalized image ready for SigLIP, shape (B, 3, 384, 384)

        This replaces the processor's non-differentiable preprocessing
        with pure PyTorch operations that allow gradient flow.
        """
        # Map from [-1, 1] to [0, 1]
        image = (image + 1) / 2

        # Clamp to valid range (numerical stability)
        image = image.clamp(0, 1)

        # Resize to SigLIP input size with bilinear interpolation
        # (differentiable, unlike PIL resize)
        if image.shape[-2:] != (self.image_size, self.image_size):
            image = F.interpolate(
                image,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )

        # Normalize with ImageNet stats (SigLIP uses these)
        mean = torch.tensor(
            [0.48145466, 0.4578275, 0.40821073],
            device=self.device,
            dtype=self.dtype,
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            [0.26862954, 0.26130258, 0.27577711],
            device=self.device,
            dtype=self.dtype,
        ).view(1, 3, 1, 1)

        image = (image - mean) / std

        return image

    def compute_reward(
        self,
        image: torch.Tensor,
        prompt: str,
        return_similarity: bool = True,
    ) -> torch.Tensor:
        """Compute differentiable reward for image-prompt alignment.

        Args:
            image: Generated image in [-1, 1], shape (B, 3, H, W)
            prompt: Text prompt (truncated to 64 tokens if longer)
            return_similarity: If True, return raw cosine similarity

        Returns:
            Reward tensor, shape (B,), with gradients attached

        Note:
            The text path uses no_grad (cached), only image path has gradients.
            This saves ~50% memory vs full backprop through both branches.
        """
        # Ensure image is on correct device and dtype
        image = image.to(device=self.device, dtype=self.dtype)

        # Get cached text embedding (no gradients)
        text_embeds = self._get_text_embedding(prompt)  # (1, embed_dim)

        # Process image with gradients
        processed_image = self._differentiable_preprocess(image)

        # Get image embedding WITH gradients
        image_embeds = self.model.get_image_features(pixel_values=processed_image)
        image_embeds = F.normalize(image_embeds, p=2, dim=-1)

        # Cosine similarity as reward
        # For batch size > 1, we want pairwise similarity (same text for all images)
        similarity = (image_embeds * text_embeds).sum(dim=-1)  # (B,)

        return similarity

    def clear_cache(self):
        """Clear text embedding cache (call between prompts if memory-constrained)."""
        self._text_cache.clear()

    def __repr__(self) -> str:
        return f"DifferentiableSigLIP(model={DEFAULT_MODEL}, device={self.device})"


def test_gradient_flow():
    """Test that gradients flow correctly through the reward function."""
    print("Testing DifferentiableSigLIP gradient flow...")

    # Create reward function
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reward_fn = DifferentiableSigLIP(device=device)

    # Create test image with gradients
    image = torch.randn(1, 3, 512, 512, device=device, dtype=torch.float32)
    image = image.requires_grad_(True)

    # Compute reward
    prompt = "A beautiful sunset over the ocean"
    reward = reward_fn.compute_reward(image, prompt)

    print(f"  Reward value: {reward.item():.4f}")
    print(f"  Reward requires_grad: {reward.requires_grad}")

    # Backprop
    reward.backward()

    print(f"  Image grad shape: {image.grad.shape}")
    print(f"  Image grad norm: {image.grad.norm().item():.6f}")
    print(f"  Image grad has NaN: {image.grad.isnan().any().item()}")
    print(f"  Image grad has Inf: {image.grad.isinf().any().item()}")

    if (
        image.grad is not None
        and not image.grad.isnan().any()
        and not image.grad.isinf().any()
    ):
        print("\nGradient flow test PASSED")
        return True
    else:
        print("\nGradient flow test FAILED")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = test_gradient_flow()
    exit(0 if success else 1)
