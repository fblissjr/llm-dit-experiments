"""Metrics for evaluating Z-Image generation quality.

Primary metrics:
- ImageReward: Human-preference aligned (best for overall quality judgment)
- SigLIP2: Image-text alignment (better than CLIP for retrieval/classification)

ImageReward is preferred for human preference because it:
- Is trained on 137k human preference comparisons
- Better correlates with human judgment
- Considers aesthetic quality, not just semantic match

SigLIP2 is preferred over CLIP for similarity because it:
- Uses sigmoid loss (no large batch requirement)
- Better zero-shot and retrieval performance
- More robust and stable
- 56% R@1 on COCO vs ~45% for CLIP
"""

from experiments.metrics.image_reward import (
    IMAGE_REWARD_GUIDELINES,
    ImageRewardScorer,
    compute_image_reward,
    compute_image_rewards_batch,
)
from experiments.metrics.siglip_score import (
    SIGLIP_SCORE_GUIDELINES,
    SigLIPScorer,
    compute_siglip_score,
    compute_siglip_scores_batch,
)

__all__ = [
    # Primary: ImageReward (human-preference aligned)
    "ImageRewardScorer",
    "compute_image_reward",
    "compute_image_rewards_batch",
    "IMAGE_REWARD_GUIDELINES",
    # Primary: SigLIP2 (image-text alignment)
    "SigLIPScorer",
    "compute_siglip_score",
    "compute_siglip_scores_batch",
    "SIGLIP_SCORE_GUIDELINES",
]
