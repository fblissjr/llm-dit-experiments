"""SigLIP2 scoring for image-text alignment evaluation.

SigLIP2 (Sigmoid Loss for Language-Image Pre-training v2) outperforms CLIP
on zero-shot classification and image-text retrieval tasks. It uses sigmoid
loss instead of softmax, eliminating the need for large batch sizes.

Usage:
    from experiments.metrics import SigLIPScorer, compute_siglip_score

    # Quick single-image scoring
    score = compute_siglip_score("A cat sleeping in sunlight", "image.png")
    print(f"SigLIP Score: {score:.4f}")

    # Batch scoring
    scorer = SigLIPScorer()
    scores = scorer.score_batch(prompts, images)

References:
    - Paper: https://arxiv.org/abs/2502.14786
    - Models: https://huggingface.co/google/siglip2-giant-opt-patch16-384
"""

import logging
from pathlib import Path
from typing import Union

import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Type aliases
ImageInput = Union[str, Path, Image.Image]

# Available SigLIP2 models (in order of quality/speed tradeoff)
SIGLIP2_MODELS = {
    # Best quality (2B params)
    "giant-384": "google/siglip2-giant-opt-patch16-384",
    # Good quality (400M params)
    "so400m-512": "google/siglip2-so400m-patch16-512",
    "so400m-384": "google/siglip2-so400m-patch14-384",
    # Faster, smaller
    "base-384": "google/siglip2-base-patch16-384",
    "base-256": "google/siglip2-base-patch16-256",
}

DEFAULT_MODEL = "giant-384"  # Best quality, 2B params


class SigLIPScorer:
    """SigLIP2 scorer for image-text alignment.

    SigLIP2 improves on CLIP with:
    - Sigmoid loss (no large batch requirement)
    - Better zero-shot and retrieval performance
    - Multilingual support
    - Dense feature extraction

    Attributes:
        model_name: Model variant (default: so400m-512)
        device: Device for computation
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
    ):
        """Initialize SigLIP2 scorer.

        Args:
            model_name: Model variant key or full HuggingFace path
            device: Device for computation (auto-detected if None)
        """
        # Resolve model name
        if model_name in SIGLIP2_MODELS:
            self.model_id = SIGLIP2_MODELS[model_name]
        else:
            self.model_id = model_name

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self._model = None
        self._processor = None

    def _load_model(self):
        """Lazy load SigLIP2 model."""
        if self._model is not None:
            return

        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError:
            raise ImportError(
                "transformers is required for SigLIP2 scoring. "
                "Install with: uv add transformers"
            )

        logger.info(
            "Loading SigLIP2 model: %s on %s",
            self.model_id,
            self.device,
        )

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModel.from_pretrained(self.model_id).to(self.device)
        self._model.eval()

        logger.info("SigLIP2 model loaded successfully")

    def _load_image(self, image: ImageInput) -> Image.Image:
        """Load image from path or return if already PIL Image."""
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        return image.convert("RGB")

    @torch.no_grad()
    def score(self, prompt: str, image: ImageInput) -> float:
        """Compute SigLIP2 score for a single image-text pair.

        Args:
            prompt: Text prompt
            image: Image path or PIL Image

        Returns:
            SigLIP2 similarity score (higher is better, typically 0.1-0.4)
        """
        self._load_model()

        pil_image = self._load_image(image)

        # Process inputs
        inputs = self._processor(
            text=[prompt],
            images=[pil_image],
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        # Get embeddings
        outputs = self._model(**inputs)

        # Compute cosine similarity
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds

        # Normalize
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # Cosine similarity
        similarity = (image_embeds @ text_embeds.T).item()

        return similarity

    @torch.no_grad()
    def score_batch(
        self,
        prompts: list[str],
        images: list[ImageInput],
    ) -> list[float]:
        """Compute SigLIP2 scores for a batch of image-text pairs.

        Args:
            prompts: List of text prompts
            images: List of images (paths or PIL Images)

        Returns:
            List of SigLIP2 scores
        """
        if len(prompts) != len(images):
            raise ValueError("prompts and images must have same length")

        self._load_model()

        pil_images = [self._load_image(img) for img in images]

        # Process inputs
        inputs = self._processor(
            text=prompts,
            images=pil_images,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        # Get embeddings
        outputs = self._model(**inputs)

        # Compute pairwise cosine similarity
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds

        # Normalize
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # Pairwise similarity (diagonal)
        similarities = (image_embeds * text_embeds).sum(dim=-1)

        return similarities.cpu().tolist()

    @torch.no_grad()
    def score_multi_prompt(
        self,
        image: ImageInput,
        prompts: dict[str, str],
    ) -> dict[str, float]:
        """Score a single image against multiple prompts.

        Useful for comparing alignment with full vs compressed prompts.

        Args:
            image: Image path or PIL Image
            prompts: Dict mapping prompt names to prompt text

        Returns:
            Dict mapping prompt names to SigLIP2 scores
        """
        self._load_model()

        pil_image = self._load_image(image)
        prompt_list = list(prompts.values())

        # Process inputs
        inputs = self._processor(
            text=prompt_list,
            images=[pil_image] * len(prompt_list),
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        # Get embeddings
        outputs = self._model(**inputs)

        # Get image embedding (same for all)
        image_embed = outputs.image_embeds[0:1]
        image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True)

        # Get text embeddings
        text_embeds = outputs.text_embeds
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # Compute similarities
        similarities = (image_embed @ text_embeds.T).squeeze(0)

        return {
            name: similarities[i].item()
            for i, name in enumerate(prompts.keys())
        }


# Convenience functions
_default_scorer: SigLIPScorer | None = None


def _get_default_scorer() -> SigLIPScorer:
    """Get or create default scorer."""
    global _default_scorer
    if _default_scorer is None:
        _default_scorer = SigLIPScorer()
    return _default_scorer


def compute_siglip_score(prompt: str, image: ImageInput) -> float:
    """Compute SigLIP2 score for a single image-text pair.

    Uses a cached default scorer for efficiency.

    Args:
        prompt: Text prompt
        image: Image path or PIL Image

    Returns:
        SigLIP2 similarity score (higher is better)

    Example:
        score = compute_siglip_score("A sunset over mountains", "output.png")
        print(f"SigLIP Score: {score:.4f}")
    """
    return _get_default_scorer().score(prompt, image)


def compute_siglip_scores_batch(
    prompts: list[str],
    images: list[ImageInput],
) -> list[float]:
    """Compute SigLIP2 scores for a batch of image-text pairs.

    Uses a cached default scorer for efficiency.

    Args:
        prompts: List of text prompts
        images: List of images (paths or PIL Images)

    Returns:
        List of SigLIP2 scores

    Example:
        scores = compute_siglip_scores_batch(
            ["A cat", "A dog"],
            ["cat.png", "dog.png"]
        )
    """
    return _get_default_scorer().score_batch(prompts, images)


# Score interpretation guidelines
SIGLIP_SCORE_GUIDELINES = """
SigLIP2 Score Interpretation (giant-opt-patch16-384):

Range: [-1, 1] (cosine similarity)
Typical generation scores: 0.15 - 0.40

Guidelines:
- 0.35+: Excellent alignment, prompt clearly reflected
- 0.28-0.35: Good alignment, main concepts present
- 0.20-0.28: Moderate alignment, some concepts missing
- 0.15-0.20: Weak alignment, significant mismatch
- <0.15: Poor alignment, likely unrelated

Advantages over CLIP:
- Sigmoid loss (no large batch requirement)
- Better zero-shot classification (+2-3% on ImageNet)
- Better image-text retrieval (56% vs ~45% R@1 on COCO)
- Multilingual support
- More robust to noise
- 2B parameter model for higher quality embeddings

Notes:
- Compare scores relative to baseline for same prompt
- Works well with natural language prompts
- More stable training and inference than CLIP
"""
