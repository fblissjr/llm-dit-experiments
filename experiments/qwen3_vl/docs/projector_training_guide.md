# Projector Training Guide: Qwen3-VL to Z-Image Alignment

> **Last Updated:** 2025-12-12

## Overview

This guide documents how to train a lightweight projection layer to align Qwen3-VL embeddings with Qwen3-4B's embedding space for optimal Z-Image generation.

## Why Train a Projector?

### The Problem

Our experiments revealed that while Qwen3-VL and Qwen3-4B share the same architecture (2560 hidden dimensions), their embedding distributions differ significantly:

| Token Type | Per-Dimension Correlation | Std Ratio Range | Issue |
|------------|---------------------------|-----------------|-------|
| Text tokens | 0.999 | ~1.0x | Near-perfect match |
| Image tokens | 0.737 | 0.1x - 617x | Extreme outliers |

**Key outlier dimensions:**
- Dimension 396: 617x std ratio (most extreme)
- Dimension 4: 42x std ratio
- Multiple dimensions with 10-50x ratios

These mismatches cause visual artifacts even with normalization techniques.

### The Solution

Train a lightweight projection layer that learns to map Qwen3-VL embeddings to Qwen3-4B's distribution. This is:
- **Small**: Only 6.5M-26M parameters (vs 3.8B model)
- **Fast**: Adds <1ms inference overhead
- **Effective**: Learns complex non-linear mappings that normalization cannot

## Architecture Options

### Option 1: Linear Projection (Simplest)

```
VL_emb (2560) --> Linear(2560, 2560) --> projected_emb (2560)
```

**Parameters:** 6.5M (2560 * 2560 + 2560 bias)

**Pros:**
- Simplest, fastest training
- No hyperparameter tuning
- 500-1K pairs may suffice

**Cons:**
- Cannot learn non-linear relationships
- May not fully correct outlier dimensions

### Option 2: MLP Projection (Recommended)

```
VL_emb (2560) --> Linear(2560, 4096) --> GELU --> Linear(4096, 2560) --> projected_emb
```

**Parameters:** 20.9M

**Pros:**
- Can learn non-linear mappings
- Better outlier handling
- Still fast at inference

**Cons:**
- More data needed (2K-5K pairs)
- Requires hyperparameter tuning

### Option 3: Attention-Based Projection

```
VL_emb (seq, 2560) --> CrossAttention(query=learnable, kv=VL) --> Linear --> projected
```

**Parameters:** ~26M

**Pros:**
- Can attend to relevant VL features
- Handles variable sequence lengths naturally
- Potentially highest quality

**Cons:**
- Most complex to train
- Needs more data (5K-10K pairs)
- Slower inference

## Dataset Requirements

### Minimum Dataset Size

| Architecture | Minimum Pairs | Recommended Pairs |
|--------------|---------------|-------------------|
| Linear | 500 | 1,000-2,000 |
| MLP | 2,000 | 5,000-10,000 |
| Attention | 5,000 | 10,000-20,000 |

### Data Diversity Requirements

For best results, the dataset should cover:

**1. Image Content Types (aim for equal distribution):**
- Photographs (portraits, landscapes, objects)
- Illustrations (cartoon, anime, vector art)
- Abstract art (patterns, textures, shapes)
- Text-heavy images (signs, documents, screenshots)
- Complex scenes (multiple objects, detailed backgrounds)

**2. Style Variations:**
- Photorealistic
- Artistic/painterly
- Flat/graphic design
- Vintage/retro
- Modern/minimalist

**3. Subject Matter:**
- People (faces, full body, groups)
- Animals
- Objects/products
- Architecture/interiors
- Nature/landscapes
- Abstract concepts

### Dataset Construction

**Step 1: Collect Image-Text Pairs**

Use existing captioned datasets or generate captions from images or synthetic generated images from z-image itself.

**Step 2: Extract Paired Embeddings**

For each (image, caption) pair:

```python
# Extract Qwen3-VL embedding (target we want to project FROM)
vl_result = vl_extractor.extract(
    image,
    text=caption,
    hidden_layer=-8,  # Our optimal layer
    text_tokens_only=False,
    scale_to_text=False,  # Raw embeddings for training
)
vl_emb = vl_result.embeddings  # (seq_len, 2560)

# Extract Qwen3-4B embedding (target we want to project TO)
text_emb = text_encoder.encode(caption).embeddings[0]  # (seq_len, 2560)
```

**Step 3: Handle Sequence Length Mismatch**

Options:
1. **Pool to fixed length**: Average pool both to fixed length (e.g., 64 tokens)
2. **Per-token with padding**: Pad shorter sequences, use loss masking
3. **Global statistics**: Compute mean/std per dimension, align those

**Recommended approach for MLP:** Pool to fixed length (simplest, works well):

```python
def pool_to_length(emb, target_len=64):
    """Average pool embedding to fixed length."""
    # emb: (seq, dim)
    seq_len = emb.shape[0]
    if seq_len <= target_len:
        # Pad with zeros
        padded = torch.zeros(target_len, emb.shape[1], dtype=emb.dtype, device=emb.device)
        padded[:seq_len] = emb
        return padded
    else:
        # Interpolate down
        emb_t = emb.T.unsqueeze(0)  # (1, dim, seq)
        pooled = F.interpolate(emb_t, size=target_len, mode='linear', align_corners=False)
        return pooled.squeeze(0).T  # (target_len, dim)
```

### Example Dataset Structure

```
projector_dataset/
  train/
    pairs.jsonl      # {"image_path": "...", "caption": "...", "vl_emb_path": "...", "text_emb_path": "..."}
    embeddings/
      vl/
        00000.pt     # (64, 2560) pooled VL embedding
        00001.pt
        ...
      text/
        00000.pt     # (64, 2560) pooled text embedding
        00001.pt
        ...
  val/
    pairs.jsonl
    embeddings/
      ...
```

## Training Approach

### Loss Function

**Primary Loss: MSE on embeddings**
```python
loss = F.mse_loss(projected_vl, target_text)
```

**Optional: Add per-dimension weighting for outlier dimensions**
```python
# Weight outlier dimensions higher to ensure they're corrected
dim_weights = torch.ones(2560)
dim_weights[396] = 10.0  # Weight the worst outlier dimension
dim_weights[4] = 5.0
loss = (dim_weights * (projected_vl - target_text) ** 2).mean()
```

### Training Hyperparameters

**For MLP Projector (recommended):**

```python
config = {
    "learning_rate": 1e-4,
    "batch_size": 32,
    "epochs": 50,
    "optimizer": "AdamW",
    "weight_decay": 0.01,
    "scheduler": "cosine",
    "warmup_steps": 500,
}
```

### Training Script Template

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class VLProjector(nn.Module):
    """MLP projector from Qwen3-VL to Qwen3-4B embedding space."""

    def __init__(self, dim=2560, hidden_dim=4096):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.proj(x)


class ProjectorDataset(Dataset):
    def __init__(self, vl_dir, text_dir):
        self.vl_files = sorted(Path(vl_dir).glob("*.pt"))
        self.text_dir = Path(text_dir)

    def __len__(self):
        return len(self.vl_files)

    def __getitem__(self, idx):
        vl_emb = torch.load(self.vl_files[idx])
        text_emb = torch.load(self.text_dir / self.vl_files[idx].name)
        return vl_emb, text_emb


def train_projector(
    train_dir: str,
    val_dir: str,
    output_path: str,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-4,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data
    train_dataset = ProjectorDataset(
        f"{train_dir}/embeddings/vl",
        f"{train_dir}/embeddings/text",
    )
    val_dataset = ProjectorDataset(
        f"{val_dir}/embeddings/vl",
        f"{val_dir}/embeddings/text",
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Model
    projector = VLProjector().to(device)
    optimizer = torch.optim.AdamW(projector.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Train
        projector.train()
        train_loss = 0
        for vl_emb, text_emb in train_loader:
            vl_emb = vl_emb.to(device)
            text_emb = text_emb.to(device)

            projected = projector(vl_emb)
            loss = F.mse_loss(projected, text_emb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validate
        projector.eval()
        val_loss = 0
        with torch.no_grad():
            for vl_emb, text_emb in val_loader:
                vl_emb = vl_emb.to(device)
                text_emb = text_emb.to(device)
                projected = projector(vl_emb)
                val_loss += F.mse_loss(projected, text_emb).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(projector.state_dict(), output_path)
            print(f"  Saved best model (val_loss={val_loss:.6f})")

        scheduler.step()

    print(f"Training complete. Best val_loss: {best_val_loss:.6f}")
```

## Evaluation Metrics

### 1. Embedding Space Metrics

```python
def evaluate_projector(projector, val_loader, device):
    projector.eval()
    all_projected = []
    all_target = []

    with torch.no_grad():
        for vl_emb, text_emb in val_loader:
            projected = projector(vl_emb.to(device))
            all_projected.append(projected.cpu())
            all_target.append(text_emb)

    projected = torch.cat(all_projected, dim=0)  # (N, seq, dim)
    target = torch.cat(all_target, dim=0)

    # Flatten to (N*seq, dim) for per-dimension analysis
    proj_flat = projected.view(-1, projected.shape[-1])
    tgt_flat = target.view(-1, target.shape[-1])

    # Per-dimension correlation
    proj_std = proj_flat.std(dim=0)
    tgt_std = tgt_flat.std(dim=0)
    std_ratio = proj_std / tgt_std.clamp(min=1e-6)

    print(f"Std ratio range: [{std_ratio.min():.3f}, {std_ratio.max():.3f}]")
    print(f"Outlier dims (>2x): {(std_ratio > 2).sum().item()}")

    # Cosine similarity
    cos_sim = F.cosine_similarity(proj_flat, tgt_flat, dim=-1).mean()
    print(f"Mean cosine similarity: {cos_sim:.4f}")

    # MSE
    mse = F.mse_loss(projected, target)
    print(f"MSE: {mse:.6f}")

    return {
        "std_ratio_range": (std_ratio.min().item(), std_ratio.max().item()),
        "outlier_dims": (std_ratio > 2).sum().item(),
        "cosine_similarity": cos_sim.item(),
        "mse": mse.item(),
    }
```

### 2. Generation Quality (Visual Evaluation)

Generate images with and without projector:

```python
# Without projector (baseline)
blended_baseline = blend_interpolate(vl_emb, text_emb, alpha=0.5)
image_baseline = pipeline(prompt_embeds=blended_baseline, ...)

# With projector
projected_vl = projector(vl_emb)
blended_projected = blend_interpolate(projected_vl, text_emb, alpha=0.5)
image_projected = pipeline(prompt_embeds=blended_projected, ...)
```

**What to look for:**
- Reduced grid/blocky artifacts
- Better color accuracy
- Improved style transfer fidelity
- Preserved prompt adherence

### 3. Automated Quality Metrics

Use SigLIP or ImageReward to compare:

```python
# Image-text alignment (higher = better prompt following)
clip_score_baseline = compute_clip_score(image_baseline, prompt)
clip_score_projected = compute_clip_score(image_projected, prompt)

# Style similarity to reference
style_sim_baseline = compute_style_similarity(image_baseline, ref_image)
style_sim_projected = compute_style_similarity(image_projected, ref_image)
```

## Integration After Training

### Using the Trained Projector

```python
# Load trained projector
projector = VLProjector()
projector.load_state_dict(torch.load("projector.pt"))
projector.eval()

# In pipeline
vl_result = vl_extractor.extract(image, text=prompt, ...)
vl_emb = vl_result.embeddings

# Project before blending
with torch.no_grad():
    projected_vl = projector(vl_emb.unsqueeze(0)).squeeze(0)

# Now blend as usual
blended = blend_interpolate(projected_vl, text_emb, alpha=0.5)
```

### Adding to VLEmbeddingExtractor

We could add a `projector_path` parameter to `VLEmbeddingExtractor`:

```python
vl_extractor = VLEmbeddingExtractor.from_pretrained(
    model_path,
    projector_path="path/to/projector.pt",  # NEW
    ...
)
# Projection happens automatically during extract()
```

## Recommended Next Steps

1. **Start simple**: Train a linear projector on 1K pairs
2. **Evaluate**: Check if outlier dimensions are corrected
3. **Scale up**: If linear works, try MLP with 5K pairs
4. **Iterate**: Adjust dataset composition based on failure modes

## Recommended Dataset: Z-Image Generated Images

**This is the optimal approach** - use images generated by Z-Image itself.

### Why Z-Image Images Are Ideal

| Aspect | External Dataset | Z-Image Generated |
|--------|------------------|-------------------|
| Caption accuracy | Variable quality | **Perfect** - you wrote the prompt |
| DiT alignment | Unknown | **Exact** - DiT created from these embeddings |
| Ground truth | Must extract | **Exact** - save embeddings during generation |
| Domain match | Web images | **Native** - DiT's visual vocabulary |

**Key insight:** We're not training the DiT, just aligning embedding spaces. The DiT already "knows" what Qwen3-4B embeddings produce. Using images the DiT generated means we're teaching the projector to output embeddings the DiT has literally seen before.

### Generation Script

```python
"""Generate projector training dataset from Z-Image."""
import json
import torch
from pathlib import Path
from PIL import Image

# Diverse prompts covering many styles/subjects
PROMPTS = [
    "A cat sleeping in sunlight",
    "Portrait of an elderly man, oil painting style",
    "Cyberpunk cityscape at night with neon lights",
    "Watercolor painting of a forest in autumn",
    "Anime girl with blue hair and red eyes",
    "Photorealistic mountain landscape at sunset",
    "Abstract geometric shapes in vibrant colors",
    "A robot holding a flower, digital art",
    # ... add 2000+ diverse prompts
]

def generate_dataset(output_dir: str, num_samples: int = 2000):
    from llm_dit.startup import PipelineLoader
    from llm_dit.cli import load_runtime_config

    output_dir = Path(output_dir)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "embeddings").mkdir(parents=True, exist_ok=True)

    # Load pipeline
    class Args:
        config = "config.toml"
        profile = "default"
    config = load_runtime_config(Args())
    pipe = PipelineLoader(config).load_pipeline().pipeline

    for i, prompt in enumerate(PROMPTS[:num_samples]):
        # Get text embedding (this is our ground truth target!)
        text_emb = pipe.encoder.encode(prompt).embeddings[0]

        # Generate image
        image = pipe(prompt=prompt, num_inference_steps=9)

        # Save both
        image.save(output_dir / "images" / f"{i:05d}.png")
        torch.save(text_emb.cpu(), output_dir / "embeddings" / f"{i:05d}.pt")

        # Save metadata
        with open(output_dir / "metadata.jsonl", "a") as f:
            f.write(json.dumps({"id": i, "prompt": prompt}) + "\n")

    print(f"Generated {num_samples} image-embedding pairs")
```

### Then Extract VL Embeddings

```python
"""Extract VL embeddings for each generated image."""
import json
import torch
from pathlib import Path
from PIL import Image
from llm_dit.vl import VLEmbeddingExtractor

def extract_vl_embeddings(dataset_dir: str):
    dataset_dir = Path(dataset_dir)
    (dataset_dir / "vl_embeddings").mkdir(exist_ok=True)

    vl_extractor = VLEmbeddingExtractor.from_pretrained(
        "/path/to/Qwen3-VL-4B-Instruct",
        device="cuda",
        torch_dtype=torch.bfloat16,
    )

    # Load metadata for prompts
    with open(dataset_dir / "metadata.jsonl") as f:
        metadata = [json.loads(line) for line in f]

    for item in metadata:
        i = item["id"]
        prompt = item["prompt"]

        image = Image.open(dataset_dir / "images" / f"{i:05d}.png")

        vl_result = vl_extractor.extract(
            image,
            text=prompt,
            hidden_layer=-6,
            text_tokens_only=False,
            scale_to_text=False,  # Raw embeddings for training
        )

        torch.save(vl_result.embeddings.cpu(),
                   dataset_dir / "vl_embeddings" / f"{i:05d}.pt")

    print(f"Extracted VL embeddings for all images")
```

### Dataset Structure

```
projector_dataset/
  images/           # Z-Image generated images
    00000.png
    00001.png
    ...
  embeddings/       # Qwen3-4B embeddings (ground truth targets)
    00000.pt
    00001.pt
    ...
  vl_embeddings/    # Qwen3-VL embeddings (projector inputs)
    00000.pt
    00001.pt
    ...
  metadata.jsonl    # {"id": 0, "prompt": "..."}
```

### Alternative: External Datasets

If you need more data, these can supplement Z-Image generated data:

1. **LAION-Aesthetics** - High quality captioned images
2. **COCO Captions** - Object-focused descriptions

But Z-Image generated images should be the primary source for perfect ground truth.

## Files Referenced

| File | Purpose |
|------|---------|
| `src/llm_dit/vl/qwen3_vl.py` | VLEmbeddingExtractor for extraction |
| `src/llm_dit/vl/blending.py` | Blending utilities |
| `experiments/qwen3_vl/scripts/run_comparison.py` | Experiment runner |
