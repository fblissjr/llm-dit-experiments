# Outlier Dimension 396: The 617x Problem

*Last updated: 2025-12-18*

## Plain English Summary

When we extract text embeddings from Qwen3-4B to feed into Z-Image, we get a 2560-dimensional vector for each token. Think of this as 2560 different "channels" of information about the text.

**The problem:** One of these channels (dimension 396) is broken. It's screaming 617 times louder than it should be compared to other channels.

Imagine an orchestra where one violinist is playing 617 times louder than everyone else. Even if they're playing the right notes, the sound is ruined. That's what dimension 396 does to our embeddings.

**Why it matters:** This outlier causes visible glitches and artifacts in generated images, especially when using vision-language (VL) conditioning.

**The fix:** Either use layer -6 instead of -2 (which doesn't have this problem), or explicitly scale down the loud dimension.

---

## Technical Deep Dive

### What We're Measuring

When comparing embeddings between models or layers, we look at **per-dimension statistics**:

```python
# For each of 2560 dimensions, compute std across tokens/samples
per_dim_std = embeddings.std(dim=0)  # Shape: (2560,)

# Compare to reference (Qwen3-4B text embeddings at layer -2)
std_ratio = target_per_dim_std / reference_per_dim_std
```

A "healthy" std_ratio is close to 1.0 (matching distributions). Outliers are dimensions where this ratio is extreme (>10x or <0.1x).

### The Numbers

**Qwen3-4B Text Embeddings at Layer -2** (reference stats from `qwen3_4b_stats.npz`):

| Dimension | Std Value | Rank | Notes |
|-----------|-----------|------|-------|
| **4** | 2752.57 | 1st | Highest variance |
| **396** | 948.70 | 2nd | Second highest |
| 0 | 193.41 | 3rd | |
| 100 | 77.22 | 4th | |
| 9 | 36.02 | 5th | |
| Average dim | ~3.5-4.0 | - | Normal range |

**Qwen3-VL Image Tokens at Layer -2** (compared to Qwen3-4B reference):

| Dimension | Std Ratio | Interpretation |
|-----------|-----------|----------------|
| **396** | **617x** | Catastrophically high |
| **4** | 42x | Very high |
| Multiple dims | 10-50x | High |
| Normal dims | ~1.0x | Expected |

**Layer -6** (both text and VL):
- NO dimensions above 10x threshold
- The 617x outlier in dim 396 is **specific to layer -2**

### Why Dimension 396 Is Extreme

The likely causes:

1. **SFT/RLHF Training Artifacts**
   - Late layers (-1, -2) are heavily modified during instruction tuning
   - Certain dimensions may have been "hijacked" for task-specific signals
   - Dimension 396 may encode something like "helpfulness score" or "safety signal"

2. **Attention Sink Pattern**
   - LLMs often develop "attention sinks" - dimensions that accumulate signal
   - These create statistical outliers that don't carry semantic information
   - Layer -2 is right before the LM head, where these patterns are strongest

3. **VL Fine-tuning Amplification**
   - Qwen3-VL adds vision processing to Qwen3
   - This fine-tuning may have further amplified existing outliers
   - Image tokens pass through different processing than text tokens

### Visual Impact

**Layer -2 with VL image tokens:**
```
Heavy glitch artifacts, grid patterns, corrupted regions
The 617x outlier in dim 396 causes catastrophic interference
```

**Layer -6 with VL image tokens:**
```
Clean photorealistic output
No outliers = no interference
```

### The Distribution Mismatch Problem

Z-Image's DiT was trained on layer -2 embeddings. This means:

1. **DiT expects the outliers** - It was trained with dims 4 and 396 being extreme
2. **DiT may have learned to ignore them** - Or use them as noise
3. **Layer -6 is technically OOD** - But often produces better results

This creates a paradox: the "correct" layer (-2) has artifacts, while the "wrong" layer (-6) works better. Possible explanations:

- The outliers are noise that the DiT learned to tolerate, not use
- Layer -6 has more useful semantic content that outweighs the distribution shift
- The distillation process (Decoupled-DMD) made the DiT robust to nearby layers

### Code Reference

**Detecting outliers:**
```python
from llm_dit.vl import get_outlier_dimensions

# Get dimensions with std ratio > threshold
outliers = get_outlier_dimensions(embeddings, threshold=10.0)
for dim, ratio in outliers[:5]:
    print(f"Dimension {dim}: {ratio:.1f}x std ratio")
```

**Masking outliers:**
```python
from llm_dit.vl import mask_outlier_dimensions

# Zero out extreme dimensions
masked_emb, info = mask_outlier_dimensions(
    embeddings,
    threshold=10.0,
    mode="zero",  # or "clamp", "scale"
)
print(f"Masked {len(info['masked_dimensions'])} dimensions")
```

**Per-dimension reference stats:**
```python
import numpy as np
stats = np.load('src/llm_dit/vl/qwen3_4b_stats.npz')
per_dim_std = stats['per_dim_std']  # Shape: (2560,)
print(f"Dim 396 std: {per_dim_std[396]:.2f}")  # ~948.7
print(f"Dim 4 std: {per_dim_std[4]:.2f}")      # ~2752.6
```

### Recommendations

**For VL conditioning:**
- Use layer -6 (no outliers, clean results)
- If using layer -2, apply outlier masking to image tokens before blending

**For pure text encoding:**
- Layer -2 works (DiT was trained on it)
- Layer -6 may produce better prompt adherence for some use cases
- Experiment with `--hidden-layer -6` flag

**For research:**
- The outlier pattern may reveal what Qwen3's SFT optimized for
- Comparing outlier profiles across layers could identify the "SFT boundary"
- Training a DiT on layer -6 embeddings might produce better results

---

## Open Questions

1. **What does dimension 396 encode?** Is it a task-specific signal or random noise?
2. **Why does layer -6 work despite being OOD?** Is the DiT robust, or is -6 genuinely better?
3. **Would training on layer -6 improve Z-Image?** Or would it just shift the outliers?
4. **Do other instruct LLMs (Llama, Mistral) have similar outlier patterns?**

---

## Related Files

- `src/llm_dit/vl/qwen3_4b_stats.npz` - Reference statistics for Qwen3-4B layer -2
- `src/llm_dit/vl/blending.py` - Outlier masking functions
- `internal/research/hidden_layer_selection.md` - Layer selection research
- `experiments/qwen3_vl/AGENTS.md` - VL experiment findings

---

## References

- Internal finding from VL conditioning experiments (2025-12-12)
- Qwen3-4B model card: https://huggingface.co/Qwen/Qwen3-4B
- Z-Image paper: arXiv:2511.22699v3
