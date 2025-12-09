"""ImageReward scoring for human-preference-aligned evaluation.

ImageReward is trained on human preference data specifically for text-to-image
evaluation, making it more aligned with human judgment than CLIP Score.

Usage:
    from experiments.metrics import ImageRewardScorer, compute_image_reward

    # Quick single-image scoring
    score = compute_image_reward("A cat sleeping in sunlight", "image.png")
    print(f"ImageReward: {score:.4f}")

    # Batch scoring
    scorer = ImageRewardScorer()
    scores = scorer.score_batch(prompts, images)

References:
    - Paper: https://arxiv.org/abs/2304.05977
    - GitHub: https://github.com/THUDM/ImageReward
"""

import logging
from pathlib import Path
from typing import Union

import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Type aliases
ImageInput = Union[str, Path, Image.Image]


class ImageRewardScorer:
    """ImageReward scorer for human-preference-aligned evaluation.

    ImageReward is trained on 137k human preference comparisons and
    correlates better with human judgment than CLIP Score.

    Attributes:
        model_name: Model name (default: ImageReward-v1.0)
        device: Device for computation
    """

    def __init__(
        self,
        model_name: str = "ImageReward-v1.0",
        device: str | None = None,
    ):
        """Initialize ImageReward scorer.

        Args:
            model_name: Model name from HuggingFace
            device: Device for computation (auto-detected if None)
        """
        self.model_name = model_name

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self._model = None

    def _load_model(self):
        """Lazy load ImageReward model."""
        if self._model is not None:
            return

        try:
            import ImageReward as RM
        except ImportError:
            raise ImportError(
                "ImageReward is required for human-preference scoring. "
                "Install with: uv add image-reward"
            )

        logger.info(
            "Loading ImageReward model: %s on %s",
            self.model_name,
            self.device,
        )

        self._model = RM.load(self.model_name, device=self.device)
        logger.info("ImageReward model loaded successfully")

    def _load_image(self, image: ImageInput) -> Image.Image:
        """Load image from path or return if already PIL Image."""
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        return image.convert("RGB")

    def score(self, prompt: str, image: ImageInput) -> float:
        """Compute ImageReward score for a single image-text pair.

        Args:
            prompt: Text prompt
            image: Image path or PIL Image

        Returns:
            ImageReward score (higher is better, typically -2 to 2)
        """
        self._load_model()

        pil_image = self._load_image(image)
        score = self._model.score(prompt, pil_image)

        return float(score)

    def score_batch(
        self,
        prompts: list[str],
        images: list[ImageInput],
    ) -> list[float]:
        """Compute ImageReward scores for a batch of image-text pairs.

        Args:
            prompts: List of text prompts
            images: List of images (paths or PIL Images)

        Returns:
            List of ImageReward scores
        """
        if len(prompts) != len(images):
            raise ValueError("prompts and images must have same length")

        self._load_model()

        scores = []
        for prompt, image in zip(prompts, images):
            pil_image = self._load_image(image)
            score = self._model.score(prompt, pil_image)
            scores.append(float(score))

        return scores

    def rank(
        self,
        prompt: str,
        images: list[ImageInput],
    ) -> list[int]:
        """Rank multiple images for the same prompt by preference.

        Args:
            prompt: Text prompt
            images: List of candidate images

        Returns:
            List of indices sorted by preference (best first)
        """
        self._load_model()

        pil_images = [self._load_image(img) for img in images]
        scores = [self._model.score(prompt, img) for img in pil_images]

        # Return indices sorted by score (descending)
        return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)


# Convenience functions
_default_scorer: ImageRewardScorer | None = None


def _get_default_scorer() -> ImageRewardScorer:
    """Get or create default scorer."""
    global _default_scorer
    if _default_scorer is None:
        _default_scorer = ImageRewardScorer()
    return _default_scorer


def compute_image_reward(prompt: str, image: ImageInput) -> float:
    """Compute ImageReward score for a single image-text pair.

    Uses a cached default scorer for efficiency.

    Args:
        prompt: Text prompt
        image: Image path or PIL Image

    Returns:
        ImageReward score (higher is better)

    Example:
        score = compute_image_reward("A sunset over mountains", "output.png")
        print(f"ImageReward: {score:.4f}")
    """
    return _get_default_scorer().score(prompt, image)


def compute_image_rewards_batch(
    prompts: list[str],
    images: list[ImageInput],
) -> list[float]:
    """Compute ImageReward scores for a batch of image-text pairs.

    Uses a cached default scorer for efficiency.

    Args:
        prompts: List of text prompts
        images: List of images (paths or PIL Images)

    Returns:
        List of ImageReward scores

    Example:
        scores = compute_image_rewards_batch(
            ["A cat", "A dog"],
            ["cat.png", "dog.png"]
        )
    """
    return _get_default_scorer().score_batch(prompts, images)


# Score interpretation guidelines
IMAGE_REWARD_GUIDELINES = """
ImageReward Score Interpretation:

Range: Unbounded, typically -2 to 2
Trained on: 137k human preference comparisons

Guidelines:
- 1.0+: Excellent - likely preferred by humans
- 0.5-1.0: Good - above average quality
- 0.0-0.5: Average - acceptable generation
- -0.5-0.0: Below average - some issues
- <-0.5: Poor - significant misalignment

Advantages over CLIP Score:
- Trained specifically on text-to-image preferences
- Better correlation with human judgment
- Considers aesthetic quality, not just semantic match
- Penalizes artifacts and unrealistic elements

Notes:
- Compare scores relative to baseline for same prompt
- More reliable for ranking images than absolute scoring
- Works best with natural language prompts
"""
