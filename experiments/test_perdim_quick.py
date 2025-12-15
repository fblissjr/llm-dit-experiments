#!/usr/bin/env python3
"""Quick test of per-dimension analysis logic."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch
import torch.nn.functional as F
import numpy as np
import json

QWEN3_4B_PATH = "/home/fbliss/Storage/Qwen3-4B"
QWEN3_EMBEDDING_PATH = "/home/fbliss/Storage/Qwen3-Embedding-4B"

print("Loading models...")

from llm_dit.embedding import EmbeddingExtractor
from llm_dit.backends.transformers import TransformersBackend

# Test prompt
prompt = "A cat sleeping in sunlight"

# Load Qwen3-4B
print("Loading Qwen3-4B...")
qwen3_backend = TransformersBackend.from_pretrained(
    QWEN3_4B_PATH,
    device_map="cuda",
    torch_dtype=torch.bfloat16,
    model_subfolder="",
    tokenizer_subfolder="",
)

output = qwen3_backend.encode([prompt])
qwen3_emb = output.embeddings[0].cpu()
print(f"Qwen3-4B: {qwen3_emb.shape}")

del qwen3_backend
torch.cuda.empty_cache()

# Load Qwen3-Embedding-4B
print("Loading Qwen3-Embedding-4B...")
embedding_extractor = EmbeddingExtractor.from_pretrained(
    QWEN3_EMBEDDING_PATH,
    device="cuda",
    torch_dtype=torch.bfloat16,
)

emb_emb = embedding_extractor.encode_for_zimage(prompt, hidden_layer=-2, scale_factor=1.0)
print(f"Qwen3-Embedding: {emb_emb.shape}")

embedding_extractor.unload()
del embedding_extractor
torch.cuda.empty_cache()

# Truncate to same length
min_len = min(len(qwen3_emb), len(emb_emb))
qwen3_emb = qwen3_emb[:min_len]
emb_emb = emb_emb[:min_len]

print(f"\nAnalyzing {min_len} tokens, 2560 dimensions...")

# Flatten
qwen3_flat = qwen3_emb.view(-1, 2560)
emb_flat = emb_emb.view(-1, 2560)

# Per-dimension stats
qwen3_means = qwen3_flat.mean(dim=0)
qwen3_stds = qwen3_flat.std(dim=0)
emb_means = emb_flat.mean(dim=0)
emb_stds = emb_flat.std(dim=0)

print(f"\nQwen3-4B stats:")
print(f"  Mean std: {qwen3_stds.mean():.2f}")
print(f"  Median std: {qwen3_stds.median():.2f}")
print(f"  Std range: {qwen3_stds.min():.2f} - {qwen3_stds.max():.2f}")

print(f"\nQwen3-Embedding stats:")
print(f"  Mean std: {emb_stds.mean():.2f}")
print(f"  Median std: {emb_stds.median():.2f}")
print(f"  Std range: {emb_stds.min():.2f} - {emb_stds.max():.2f}")

# Std ratios
safe_qwen3_stds = torch.where(qwen3_stds < 1e-6, torch.ones_like(qwen3_stds), qwen3_stds)
std_ratios = emb_stds / safe_qwen3_stds

print(f"\nStd ratios:")
print(f"  Mean: {std_ratios.mean():.3f}")
print(f"  Median: {std_ratios.median():.3f}")
print(f"  Range: {std_ratios.min():.3f} - {std_ratios.max():.3f}")

# Top outliers
top_high = torch.argsort(std_ratios, descending=True)[:10]
top_low = torch.argsort(std_ratios, descending=False)[:10]

print(f"\nTop 10 high std ratio dimensions:")
for dim in top_high:
    print(f"  Dim {dim}: {std_ratios[dim]:.3f}x (emb={emb_stds[dim]:.2f}, qwen3={qwen3_stds[dim]:.2f})")

print(f"\nTop 10 low std ratio dimensions:")
for dim in top_low:
    print(f"  Dim {dim}: {std_ratios[dim]:.3f}x (emb={emb_stds[dim]:.2f}, qwen3={qwen3_stds[dim]:.2f})")

# Dead dimensions
qwen3_dead = (qwen3_stds < 0.01).sum().item()
emb_dead = (emb_stds < 0.01).sum().item()

print(f"\nDead dimensions (std < 0.01):")
print(f"  Qwen3-4B: {qwen3_dead}")
print(f"  Qwen3-Embedding: {emb_dead}")

# Hyperactive dimensions (> 5x median)
qwen3_hyper = (qwen3_stds > qwen3_stds.median() * 5).sum().item()
emb_hyper = (emb_stds > emb_stds.median() * 5).sum().item()

print(f"\nHyperactive dimensions (std > 5x median):")
print(f"  Qwen3-4B: {qwen3_hyper}")
print(f"  Qwen3-Embedding: {emb_hyper}")

# Correlations
mean_corr = F.cosine_similarity(emb_means.unsqueeze(0), qwen3_means.unsqueeze(0)).item()
std_corr = F.cosine_similarity(emb_stds.unsqueeze(0), qwen3_stds.unsqueeze(0)).item()
global_corr = F.cosine_similarity(emb_flat.flatten().unsqueeze(0), qwen3_flat.flatten().unsqueeze(0)).item()

print(f"\nCorrelations:")
print(f"  Global cosine similarity: {global_corr:.4f}")
print(f"  Mean correlation: {mean_corr:.4f}")
print(f"  Std correlation: {std_corr:.4f}")

# Save key results
output_file = Path("experiments/results/embedding_perdim_quick.json")
output_file.parent.mkdir(parents=True, exist_ok=True)

results = {
    "prompt": prompt,
    "num_tokens": min_len,
    "global_cosine": global_corr,
    "mean_correlation": mean_corr,
    "std_correlation": std_corr,
    "std_ratios": {
        "mean": std_ratios.mean().item(),
        "median": std_ratios.median().item(),
        "min": std_ratios.min().item(),
        "max": std_ratios.max().item(),
    },
    "top_high_dims": [
        {"dim": int(d), "ratio": float(std_ratios[d]), "emb_std": float(emb_stds[d]), "qwen3_std": float(qwen3_stds[d])}
        for d in top_high
    ],
    "top_low_dims": [
        {"dim": int(d), "ratio": float(std_ratios[d]), "emb_std": float(emb_stds[d]), "qwen3_std": float(qwen3_stds[d])}
        for d in top_low
    ],
    "dead_dimensions": {"qwen3": qwen3_dead, "embedding": emb_dead},
    "hyperactive_dimensions": {"qwen3": qwen3_hyper, "embedding": emb_hyper},
}

with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {output_file}")
print("\nDone!")
