"""
Metrics utilities for evaluating generated content.

Last Updated: 2026-01-17

Provides scoring functions for text-image/video alignment using
vision-language models like SigLIP and CLIP.

For gradient-enabled reward computation (FMTT training), use
`llm_dit.rewards.siglip.DifferentiableSigLIP` instead.

Usage:
    from llm_dit.utils.metrics import SigLIPScorer, compute_siglip_score

    # Quick one-off scoring
    score = compute_siglip_score(image, "A cat sleeping")

    # Batch scoring with caching
    scorer = SigLIPScorer()
    scores = scorer.score_batch(images, prompts)
"""

import logging
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)

# Default model for scoring
DEFAULT_SIGLIP_MODEL = "google/siglip2-giant-opt-patch16-384"


class SigLIPScorer:
    """
    SigLIP-based text-image alignment scorer.

    This is a lightweight scorer for evaluation that does not compute
    gradients. For training with backprop through the score, use
    `llm_dit.rewards.siglip.DifferentiableSigLIP` instead.

    Memory usage: ~4GB for SigLIP2-giant

    Example:
        scorer = SigLIPScorer()

        # Score single image
        score = scorer.score(image, "A beautiful sunset")

        # Score batch
        scores = scorer.score_batch(images, prompts)

        # Score video frames (returns per-frame and mean score)
        frame_scores, mean_score = scorer.score_video(frames, prompt)
    """

    def __init__(
        self,
        model_name: str = DEFAULT_SIGLIP_MODEL,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        max_cache_size: int = 100,
    ):
        """
        Initialize SigLIP scorer.

        Args:
            model_name: HuggingFace model ID
            device: Device for computation
            dtype: Model dtype (bf16 recommended for memory)
            max_cache_size: Maximum text embeddings to cache
        """
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self._max_cache_size = max_cache_size

        self._model = None
        self._processor = None
        self._text_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._image_size: int = 384

    def _ensure_loaded(self) -> None:
        """Lazy load model on first use."""
        if self._model is not None:
            return

        from transformers import AutoModel, AutoProcessor

        logger.info(f"Loading SigLIP scorer: {self.model_name}")

        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model = self._model.to(self.device, self.dtype)
        self._model.requires_grad_(False)

        # Get image size from processor
        self._image_size = self._processor.image_processor.size.get("height", 384)

        logger.info(f"SigLIP loaded: {self._image_size}px, {self.dtype}")

    def _get_text_embedding(self, text: str) -> torch.Tensor:
        """Get cached or compute text embedding."""
        if text in self._text_cache:
            self._text_cache.move_to_end(text)
            return self._text_cache[text]

        self._ensure_loaded()

        inputs = self._processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,  # SigLIP limit
        ).to(self.device)

        with torch.no_grad():
            embeds = self._model.get_text_features(**inputs)
            embeds = F.normalize(embeds, p=2, dim=-1)

        # LRU eviction
        if len(self._text_cache) >= self._max_cache_size:
            self._text_cache.popitem(last=False)

        self._text_cache[text] = embeds
        return embeds

    def _preprocess_image(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
    ) -> torch.Tensor:
        """Preprocess image for SigLIP."""
        self._ensure_loaded()

        # Convert to PIL if needed
        if isinstance(image, torch.Tensor):
            # Assume [C, H, W] or [H, W, C] in [0, 1] or [-1, 1]
            if image.dim() == 3:
                if image.shape[0] in (1, 3, 4):  # C, H, W
                    image = image.permute(1, 2, 0)
                # Now H, W, C
                image = image.cpu().numpy()
                if image.min() < 0:
                    image = (image + 1) / 2
                image = (image * 255).clip(0, 255).astype(np.uint8)
                image = Image.fromarray(image)
        elif isinstance(image, np.ndarray):
            if image.dtype == np.float32 or image.dtype == np.float64:
                if image.min() < 0:
                    image = (image + 1) / 2
                image = (image * 255).clip(0, 255).astype(np.uint8)
            image = Image.fromarray(image)

        # Now image is PIL
        inputs = self._processor(
            images=image,
            return_tensors="pt",
        )

        return inputs["pixel_values"].to(self.device, self.dtype)

    def score(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        text: str,
    ) -> float:
        """
        Compute text-image alignment score.

        Args:
            image: Input image (PIL, numpy, or tensor)
            text: Text prompt

        Returns:
            Cosine similarity score in [-1, 1]
        """
        self._ensure_loaded()

        # Get embeddings
        text_embed = self._get_text_embedding(text)
        pixel_values = self._preprocess_image(image)

        with torch.no_grad():
            image_embed = self._model.get_image_features(pixel_values=pixel_values)
            image_embed = F.normalize(image_embed, p=2, dim=-1)

        # Cosine similarity
        similarity = (image_embed * text_embed).sum(dim=-1)

        return similarity.item()

    def score_batch(
        self,
        images: List[Union[Image.Image, np.ndarray, torch.Tensor]],
        texts: Union[str, List[str]],
    ) -> List[float]:
        """
        Score batch of images against text(s).

        Args:
            images: List of images
            texts: Single text (used for all images) or list of texts

        Returns:
            List of similarity scores
        """
        if isinstance(texts, str):
            texts = [texts] * len(images)

        if len(texts) != len(images):
            raise ValueError(f"Got {len(images)} images but {len(texts)} texts")

        return [self.score(img, txt) for img, txt in zip(images, texts)]

    def score_video(
        self,
        frames: Union[np.ndarray, List[Image.Image], torch.Tensor],
        text: str,
        sample_rate: int = 1,
    ) -> Tuple[List[float], float]:
        """
        Score video frames against text prompt.

        Args:
            frames: Video frames [F, H, W, C] or list of images
            text: Text prompt
            sample_rate: Score every Nth frame (1 = all frames)

        Returns:
            Tuple of (per-frame scores, mean score)
        """
        # Handle different input formats
        if isinstance(frames, torch.Tensor):
            if frames.dim() == 4:  # [F, H, W, C] or [F, C, H, W]
                frames = [frames[i] for i in range(0, len(frames), sample_rate)]
        elif isinstance(frames, np.ndarray):
            if frames.ndim == 4:  # [F, H, W, C]
                frames = [frames[i] for i in range(0, len(frames), sample_rate)]
        elif isinstance(frames, list):
            frames = frames[::sample_rate]

        scores = self.score_batch(frames, text)
        mean_score = sum(scores) / len(scores) if scores else 0.0

        return scores, mean_score

    def clear_cache(self) -> None:
        """Clear text embedding cache."""
        self._text_cache.clear()

    def offload(self) -> None:
        """Offload model to CPU."""
        if self._model is not None:
            self._model.to("cpu")
            self._text_cache.clear()
            torch.cuda.empty_cache()
            logger.info("SigLIP scorer offloaded to CPU")


# Convenience function for one-off scoring
def compute_siglip_score(
    image: Union[Image.Image, np.ndarray, torch.Tensor, str, Path],
    text: str,
    device: str = "cuda",
) -> float:
    """
    Compute SigLIP text-image alignment score.

    Convenience function that creates a scorer, computes score, and returns.
    For batch scoring, create a SigLIPScorer instance directly.

    Args:
        image: Input image (PIL, numpy, tensor, or path)
        text: Text prompt
        device: Compute device

    Returns:
        Cosine similarity score in [-1, 1]

    Example:
        >>> score = compute_siglip_score("output.png", "A cat sleeping")
        >>> print(f"Score: {score:.3f}")
    """
    # Load from path if needed
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")

    scorer = SigLIPScorer(device=device)
    return scorer.score(image, text)


def compute_video_siglip_score(
    video_path: Union[str, Path],
    text: str,
    device: str = "cuda",
    sample_frames: int = 8,
) -> Tuple[float, List[float]]:
    """
    Compute SigLIP score for video.

    Args:
        video_path: Path to video file
        text: Text prompt
        device: Compute device
        sample_frames: Number of frames to sample

    Returns:
        Tuple of (mean score, per-frame scores)
    """
    try:
        import av
    except ImportError:
        raise ImportError("av required for video scoring. Install: pip install av")

    # Extract frames
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    total_frames = stream.frames or 100

    # Calculate sample indices
    if total_frames <= sample_frames:
        indices = list(range(total_frames))
    else:
        indices = [int(i * total_frames / sample_frames) for i in range(sample_frames)]

    frames = []
    for i, frame in enumerate(container.decode(stream)):
        if i in indices:
            frames.append(frame.to_image())
        if len(frames) >= sample_frames:
            break

    container.close()

    # Score
    scorer = SigLIPScorer(device=device)
    per_frame, mean_score = scorer.score_video(frames, text)

    return mean_score, per_frame
