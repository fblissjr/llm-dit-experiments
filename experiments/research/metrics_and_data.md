# Metrics and Data Collection

What to measure, how to collect it, and how to interpret results.

---

## 1. Image Quality Metrics

### FID (Frechet Inception Distance)

Measures distribution similarity between generated and reference images.

```python
# Using clean-fid library
from cleanfid import fid

score = fid.compute_fid(
    "path/to/generated/",
    "path/to/reference/",
    mode="clean",
    num_workers=4
)
# Lower is better. Good: <50, Excellent: <20
```

**Sub-questions:**
- Q: What reference dataset to use? COCO, custom curated, or model-specific?
- Q: How many images needed for stable FID? (Typically 5K-50K)
- Q: Does FID correlate with human preference for this model?

### CLIP Score

Text-image alignment using CLIP embeddings.

```python
import clip
import torch
from PIL import Image

model, preprocess = clip.load("ViT-B/32", device="cuda")

def clip_score(image_path: str, text: str) -> float:
    image = preprocess(Image.open(image_path)).unsqueeze(0).to("cuda")
    text_tokens = clip.tokenize([text]).to("cuda")

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tokens)

    similarity = (image_features @ text_features.T).item()
    return similarity * 100  # Scale to 0-100

# Higher is better. Good: >25, Excellent: >30
```

**Sub-questions:**
- Q: Which CLIP model variant? (ViT-B/32, ViT-L/14, OpenCLIP)
- Q: Should we use the full prompt or key elements only?
- Q: How does CLIP score correlate with human judgment for Z-Image?

### LPIPS (Learned Perceptual Image Patch Similarity)

Perceptual diversity/similarity between images.

```python
import lpips

loss_fn = lpips.LPIPS(net='alex')

def lpips_distance(img1_path: str, img2_path: str) -> float:
    img1 = lpips.im2tensor(lpips.load_image(img1_path))
    img2 = lpips.im2tensor(lpips.load_image(img2_path))
    return loss_fn(img1, img2).item()

# For diversity: Higher variance in LPIPS across seeds = more diverse
# For consistency: Lower LPIPS between same prompt = more consistent
```

**Sub-questions:**
- Q: Is high diversity good (creative) or bad (unstable)?
- Q: What is the expected LPIPS range for same-prompt different-seed?

### Detail/Sharpness Metrics

```python
import cv2
import numpy as np

def laplacian_variance(image_path: str) -> float:
    """Higher = sharper image."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return cv2.Laplacian(img, cv2.CV_64F).var()

def edge_density(image_path: str) -> float:
    """Ratio of edge pixels to total pixels."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, 100, 200)
    return edges.sum() / edges.size
```

---

## 2. Embedding Space Analysis

### Embedding Statistics

```python
from dataclasses import dataclass
from typing import List
import torch
import numpy as np

@dataclass
class EmbeddingStats:
    mean_norm: float
    std_norm: float
    mean_variance: float  # Variance across hidden dim
    sparsity: float       # % of near-zero values
    effective_rank: float # PCA-based dimensionality

def analyze_embedding(embedding: torch.Tensor) -> EmbeddingStats:
    """Analyze a single embedding [seq_len, hidden_dim]."""
    norms = embedding.norm(dim=-1)  # Per-token norms
    variances = embedding.var(dim=-1)  # Per-token variance

    # Sparsity: fraction of values < 0.01
    sparsity = (embedding.abs() < 0.01).float().mean().item()

    # Effective rank via singular values
    U, S, V = torch.svd(embedding)
    normalized_S = S / S.sum()
    entropy = -(normalized_S * torch.log(normalized_S + 1e-10)).sum()
    effective_rank = torch.exp(entropy).item()

    return EmbeddingStats(
        mean_norm=norms.mean().item(),
        std_norm=norms.std().item(),
        mean_variance=variances.mean().item(),
        sparsity=sparsity,
        effective_rank=effective_rank,
    )
```

**Sub-questions:**
- Q: Do "better" prompts have higher embedding norms?
- Q: Does effective rank correlate with prompt complexity?
- Q: Are sparse embeddings worse for generation?

### Cosine Similarity Analysis

```python
def pairwise_cosine_similarity(embeddings: List[torch.Tensor]) -> np.ndarray:
    """Compute pairwise cosine similarity matrix."""
    n = len(embeddings)
    sim_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            # Mean-pool to single vector
            vec_i = embeddings[i].mean(dim=0)
            vec_j = embeddings[j].mean(dim=0)
            sim = torch.cosine_similarity(vec_i.unsqueeze(0), vec_j.unsqueeze(0))
            sim_matrix[i, j] = sim.item()

    return sim_matrix

# Use for:
# - Comparing embeddings across conditions
# - Measuring semantic drift from modifications
# - Clustering prompts by embedding similarity
```

### Token-Level Analysis

```python
def token_importance_by_norm(embedding: torch.Tensor) -> torch.Tensor:
    """Rank tokens by L2 norm (proxy for importance)."""
    norms = embedding.norm(dim=-1)  # [seq_len]
    return norms / norms.sum()  # Normalized importance

def token_attention_weights(
    embedding: torch.Tensor,
    query: torch.Tensor = None
) -> torch.Tensor:
    """Compute attention weights if we had a query."""
    if query is None:
        # Self-attention: mean embedding as query
        query = embedding.mean(dim=0, keepdim=True)  # [1, hidden_dim]

    # Scaled dot-product attention weights
    scores = torch.matmul(query, embedding.T)  # [1, seq_len]
    weights = torch.softmax(scores / np.sqrt(embedding.shape[-1]), dim=-1)
    return weights.squeeze()
```

---

## 3. Token Attribution Mapping

Trace which tokens contribute to which image regions.

### Leave-One-Out Attribution

```python
def token_attribution_loo(
    prompt: str,
    encoder,
    pipeline,
    seed: int = 42
) -> dict:
    """Leave-one-out token importance."""
    tokens = encoder.tokenizer.encode(prompt)
    baseline_image = generate(prompt, seed=seed)

    attributions = {}
    for i in range(len(tokens)):
        # Remove token i
        modified_tokens = tokens[:i] + tokens[i+1:]
        modified_prompt = encoder.tokenizer.decode(modified_tokens)

        modified_image = generate(modified_prompt, seed=seed)

        # Measure difference
        diff = lpips_distance(baseline_image, modified_image)
        token_text = encoder.tokenizer.decode([tokens[i]])
        attributions[f"{i}:{token_text}"] = diff

    return attributions  # Higher diff = more important token
```

### Regional Attribution

```python
def regional_attribution(
    prompt: str,
    token_index: int,
    encoder,
    pipeline,
    grid_size: int = 4,
    seed: int = 42
) -> np.ndarray:
    """Which image regions does token_index affect?"""
    baseline = generate(prompt, seed=seed)

    # Remove target token
    tokens = encoder.tokenizer.encode(prompt)
    modified_tokens = tokens[:token_index] + tokens[token_index+1:]
    modified_prompt = encoder.tokenizer.decode(modified_tokens)
    modified = generate(modified_prompt, seed=seed)

    # Compute per-region difference
    baseline_arr = np.array(baseline)
    modified_arr = np.array(modified)

    h, w = baseline_arr.shape[:2]
    region_h, region_w = h // grid_size, w // grid_size

    attribution_map = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            region_baseline = baseline_arr[
                i*region_h:(i+1)*region_h,
                j*region_w:(j+1)*region_w
            ]
            region_modified = modified_arr[
                i*region_h:(i+1)*region_h,
                j*region_w:(j+1)*region_w
            ]
            attribution_map[i, j] = np.abs(
                region_baseline.astype(float) - region_modified.astype(float)
            ).mean()

    return attribution_map
```

**Sub-questions:**
- Q: Do subject nouns map to central regions?
- Q: Do style adjectives affect the entire image uniformly?
- Q: Is there positional bias (early tokens -> left side)?

---

## 4. Prompt Format Comparison

### Systematic Format Testing

```python
@dataclass
class PromptFormat:
    name: str
    system_prompt: str | None
    force_think_block: bool
    thinking_content: str | None
    template: str | None

FORMATS_TO_TEST = [
    PromptFormat("bare", None, False, None, None),
    PromptFormat("system_only", "You are helpful.", False, None, None),
    PromptFormat("think_empty", None, True, None, None),
    PromptFormat("think_quality", None, True, "Sharp, detailed", None),
    PromptFormat("full_default", "default", True, "Thinking...", None),
    PromptFormat("template_photo", None, False, None, "photorealistic"),
]

def compare_formats(base_prompt: str, formats: List[PromptFormat]) -> pd.DataFrame:
    results = []
    for fmt in formats:
        # Count tokens
        full_prompt = build_full_prompt(base_prompt, fmt)
        token_count = len(encoder.tokenizer.encode(full_prompt))

        # Generate
        image = generate(base_prompt, **fmt.__dict__)

        # Metrics
        clip = clip_score(image, base_prompt)
        detail = laplacian_variance(image)

        results.append({
            'format': fmt.name,
            'tokens': token_count,
            'clip_score': clip,
            'detail': detail,
            'tokens_per_clip': token_count / clip,  # Efficiency
        })

    return pd.DataFrame(results)
```

### Token Efficiency Metric

```
Token Efficiency = Quality Metric / Token Count

Goal: Maximize quality per token spent
```

---

## 5. Rewriter Evaluation

### Rewriter Quality Assessment

```python
def evaluate_rewriter(
    original_prompts: List[str],
    rewriter_template: str,
    encoder,
    pipeline
) -> pd.DataFrame:
    results = []

    for original in original_prompts:
        # Rewrite
        rewritten = rewrite_prompt(original, template=rewriter_template)

        # Token counts
        orig_tokens = len(encoder.tokenizer.encode(original))
        rewrite_tokens = len(encoder.tokenizer.encode(rewritten))

        # Semantic preservation
        orig_embed = encoder.encode(original).embeddings.mean(dim=0)
        rewrite_embed = encoder.encode(rewritten).embeddings.mean(dim=0)
        semantic_sim = torch.cosine_similarity(
            orig_embed.unsqueeze(0), rewrite_embed.unsqueeze(0)
        ).item()

        # Generation quality
        orig_image = generate(original, seed=42)
        rewrite_image = generate(rewritten, seed=42)

        orig_clip = clip_score(orig_image, original)
        rewrite_clip = clip_score(rewrite_image, rewritten)
        rewrite_clip_orig = clip_score(rewrite_image, original)  # Does rewrite still match original intent?

        results.append({
            'original': original[:50],
            'rewritten': rewritten[:50],
            'orig_tokens': orig_tokens,
            'rewrite_tokens': rewrite_tokens,
            'token_expansion': rewrite_tokens / orig_tokens,
            'semantic_similarity': semantic_sim,
            'orig_clip': orig_clip,
            'rewrite_clip': rewrite_clip,
            'rewrite_clip_to_orig': rewrite_clip_orig,
            'quality_gain': rewrite_clip - orig_clip,
        })

    return pd.DataFrame(results)
```

**Sub-questions:**
- Q: Does rewriting improve CLIP score?
- Q: What is the optimal token expansion ratio?
- Q: Does semantic similarity predict quality preservation?
- Q: Which rewriter templates perform best?

---

## 6. User Study Protocol

### A/B Preference Test

```python
import random

def create_ab_test(
    conditions: List[str],
    prompts: List[str],
    seeds: List[int],
    output_dir: str
) -> List[dict]:
    """Generate image pairs for A/B testing."""
    comparisons = []

    for prompt in prompts:
        for seed in seeds:
            # Generate all conditions
            images = {}
            for cond in conditions:
                img = generate(prompt, condition=cond, seed=seed)
                images[cond] = img

            # Create pairwise comparisons
            for i, cond_a in enumerate(conditions):
                for cond_b in conditions[i+1:]:
                    # Randomize left/right
                    if random.random() > 0.5:
                        left, right = cond_a, cond_b
                    else:
                        left, right = cond_b, cond_a

                    comparison_id = f"{prompt[:20]}_{seed}_{left}_vs_{right}"
                    comparisons.append({
                        'id': comparison_id,
                        'prompt': prompt,
                        'left_image': images[left],
                        'right_image': images[right],
                        'left_condition': left,
                        'right_condition': right,
                    })

    return comparisons

# Then present to users:
# "Which image better matches the prompt? [Left] [Right] [Tie]"
```

### Study Design Considerations

1. **Sample size**: Minimum 30 comparisons per condition pair
2. **Rater count**: 3-5 raters per comparison for reliability
3. **Prompt diversity**: Cover different subjects, styles, complexities
4. **Seed control**: Same seed across conditions for fair comparison
5. **Presentation**: Randomize order, don't reveal conditions

---

## 7. Data Collection Infrastructure

### Experiment Logger

```python
import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict

@dataclass
class ExperimentRun:
    experiment_name: str
    timestamp: str
    config: dict
    prompt: str
    seed: int
    metrics: dict
    image_path: str
    embedding_stats: dict | None = None

class ExperimentLogger:
    def __init__(self, output_dir: str = "experiments/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.runs = []

    def log_run(self, run: ExperimentRun):
        self.runs.append(run)

        # Save incrementally
        run_path = self.output_dir / f"{run.experiment_name}_{run.timestamp}.json"
        with open(run_path, 'w') as f:
            json.dump(asdict(run), f, indent=2)

    def export_csv(self, filename: str):
        df = pd.DataFrame([asdict(r) for r in self.runs])
        df.to_csv(self.output_dir / filename, index=False)

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(r) for r in self.runs])
```

### Automated Metrics Pipeline

```bash
#!/bin/bash
# experiments/run_metrics.sh

EXPERIMENT_DIR=$1
REFERENCE_DIR="data/reference_images"

echo "Computing FID..."
python -m cleanfid --source $EXPERIMENT_DIR --reference $REFERENCE_DIR

echo "Computing CLIP scores..."
python experiments/compute_clip.py --images $EXPERIMENT_DIR --prompts prompts.txt

echo "Computing LPIPS diversity..."
python experiments/compute_lpips.py --images $EXPERIMENT_DIR

echo "Aggregating results..."
python experiments/aggregate_metrics.py --dir $EXPERIMENT_DIR
```

---

## 8. Baseline Measurements

Before running ablations, establish baselines:

### Required Baselines

| Measurement | Command | Expected |
|-------------|---------|----------|
| Default FID | `compute_fid(generated, coco)` | ~30-50 |
| Default CLIP | `clip_score(image, prompt)` | ~25-30 |
| Encoding time | `profiler.py --tests encode_short` | ~50-100ms |
| Generation time | `profiler.py --tests full_generation` | ~2-5s |
| Peak VRAM | `nvidia-smi` during generation | ~12-18GB |

### Prompt Test Set

Create a standardized test set:

```
experiments/prompts/
    standard_50.txt      # 50 diverse prompts
    portraits_20.txt     # 20 portrait prompts
    landscapes_20.txt    # 20 landscape prompts
    abstract_10.txt      # 10 abstract prompts
    long_prompts_10.txt  # 10 prompts >500 tokens
```

---

## Summary: Key Metrics by Research Question

| Research Question | Primary Metrics | Secondary Metrics |
|-------------------|-----------------|-------------------|
| Hidden layer comparison | CLIP, FID | Embedding cosine sim |
| Think block impact | CLIP, detail score | Token count, semantic drift |
| Shift parameter | FID, inference time | VRAM, LPIPS diversity |
| Long prompt modes | CLIP (full prompt), content preservation | Compression ratio |
| Token importance | LOO attribution | Regional attribution |
| Rewriter quality | CLIP gain, semantic similarity | Token expansion ratio |
