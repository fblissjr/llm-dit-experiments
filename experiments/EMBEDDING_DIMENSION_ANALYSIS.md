# Qwen3-Embedding-4B Dimension Analysis Summary

Last updated: 2025-12-14

## Executive Summary

This document summarizes the investigation into why Qwen3-Embedding-4B produces severe visual artifacts when used with Z-Image DiT, despite achieving 98% cosine similarity with Qwen3-4B embeddings.

## Problem Statement

**Observable Symptoms:**
- Industrial/data-center backgrounds in all generated images
- Complete loss of prompt semantic content
- Consistent artifacts across different prompts and seeds
- Images are unusable for any practical purpose

**Known Constraints:**
- Global cosine similarity: 98% at layer -2
- Std ratio after 1.15x scaling: Close match (~1.15x)
- Per-dimension normalization (dividing each dim by its std): No improvement
- Z-Image DiT was trained exclusively on Qwen3-4B embeddings

## Root Cause Hypothesis

**Core Insight:** Cosine similarity only measures vector **angle**, not distribution characteristics that the DiT learned during training.

The DiT's attention mechanism and learned weights are sensitive to:
1. **Per-dimension variance patterns** - Specific dimensions may be weighted heavily
2. **Distribution shapes** - Kurtosis and skewness of dimension activations
3. **Outlier dimensions** - Extreme variance ratios that get over-attended
4. **Dead dimensions** - Near-zero variance creating semantic gaps

Contrastive training (used for Qwen3-Embedding-4B) optimizes for **discriminative power** in retrieval tasks, which fundamentally changes the embedding space structure:
- Amplifies certain dimensions for better separation
- Suppresses others that don't aid retrieval
- Changes distribution shapes to maximize similarity margins
- Creates "hyperactive" dimensions encoding discriminative (not generative) features

## Analysis Approach

We perform **per-dimension statistical analysis** to identify problematic dimensions:

### Phase 1: Dimension Statistics
For each of 2560 dimensions:
- Mean, std, min, max, range
- Kurtosis (tail heaviness)
- Skewness (asymmetry)
- Classification: normal, dead, hyperactive, outlier

### Phase 2: Cross-Model Comparison
Compare Qwen3-Embedding vs Qwen3-4B:
- Std ratio per dimension (emb_std / qwen3_std)
- Mean correlation
- Distribution shape differences
- Identify dimensions unique to each model's special categories

### Phase 3: Attention Sensitivity
Simulate DiT attention:
- Compute token-level attention weights (simplified)
- Identify dimensions with largest attention-weighted differences
- Find which dimensions likely cause over-attention

### Phase 4: Aggregation
Combine across multiple diverse prompts:
- Consistently problematic dimensions (global issue)
- Prompt-specific outliers (content-dependent)

## Tools Provided

### 1. Analysis Scripts

**`test_perdim_quick.py`** - Fast single-prompt test
- Runtime: ~30 seconds
- Output: JSON summary with top outliers
- Use: Initial hypothesis validation

**`analyze_embedding_perdim.py`** - Comprehensive analysis
- Runtime: ~5 minutes (5 prompts)
- Output: Full JSON with all statistics
- Use: Root cause identification

**`visualize_perdim_results.py`** - Generate plots and report
- Input: Analysis JSON
- Output: PNG plots + markdown report
- Use: Understanding and sharing findings

### 2. Fix Implementation

**`fix_embedding_dimensions.py`** - Test dimension corrections
- Implements 4 fix strategies
- Generates comparison images
- Validates which fix works best

**Fix Modes:**
1. **rescale** - Rescale outlier dimensions to match Qwen3-4B variance
2. **mask** - Rescale + zero out embedding-only dead dimensions
3. **clamp** - Rescale + mask + clamp hyperactive dimensions to ±3σ
4. **full** - Full per-dimension distribution matching (Z-score normalization)

## Expected Results

Based on prior VL embedding analysis (similar problem), we expect:

### Primary Findings
- **50-200 outlier dimensions** with std ratios >3x or <0.33x
- **20-100 hyperactive dimensions** unique to embedding model
- **Std correlation ~0.7-0.8** (significantly lower than global cosine ~0.98)
- **Specific dimension clusters** encoding "industrial" semantic features

### Secondary Findings
- More dead dimensions in embedding model (gap-filling with defaults)
- Distribution shape mismatches (kurtosis/skewness)
- Attention sensitivity to first/last token positions

## Success Criteria

After applying the optimal fix:

### Qualitative
- ✅ Visual artifacts completely eliminated
- ✅ Image quality matches Qwen3-4B baseline
- ✅ Prompt semantic content fully preserved
- ✅ Diverse prompts work correctly

### Quantitative
- ✅ Global cosine similarity remains >95%
- ✅ Std correlation improves to >0.95
- ✅ Mean correlation improves to >0.9
- ✅ Outlier dimension count <10
- ✅ Dead/hyperactive dimension counts match Qwen3-4B

## Integration Path

Once the optimal fix is validated:

### Step 1: Add to EmbeddingExtractor
```python
# src/llm_dit/embedding/qwen3_embedding.py
class EmbeddingExtractor:
    def __init__(self, ...):
        self._dimension_fixes = self._load_dimension_fixes()

    def encode_for_zimage(self, text, fix_dimensions=True):
        emb = self.extract(text, mode="full_sequence", hidden_layer=-2)
        if fix_dimensions:
            emb = self._apply_dimension_fixes(emb)
        return emb

    def _apply_dimension_fixes(self, emb):
        # Apply learned dimension corrections
        for dim, scale in self._dimension_fixes["outliers"].items():
            emb[:, dim] *= scale
        # ... mask dead, clamp hyperactive
        return emb
```

### Step 2: Add CLI Flag
```bash
uv run scripts/generate.py \
    --use-embedding-encoder \
    --fix-embedding-dimensions \
    "A cat sleeping"
```

### Step 3: Add Config Option
```toml
[embedding]
model_path = "/path/to/Qwen3-Embedding-4B"
fix_dimensions = true
fix_mode = "rescale"  # or "mask", "clamp", "full"
```

### Step 4: Document in CLAUDE.md
Add section on dimension fixing with:
- When to use it (always with Qwen3-Embedding-4B)
- Which fix mode is optimal
- Performance impact (minimal - just rescaling)

## Known Limitations

### What This Fixes
- ✅ Per-dimension distribution mismatches
- ✅ Outlier dimensions causing artifacts
- ✅ Dead/hyperactive dimension issues

### What This Doesn't Fix
- ❌ Fundamental embedding space differences (needs adapter/LoRA)
- ❌ Semantic drift from contrastive training (inherent to model)
- ❌ Training distribution mismatch (DiT expects Qwen3-4B)

### When It May Not Work
- If artifacts are caused by **token-level** rather than dimension-level issues
- If Qwen3-Embedding has fundamentally incompatible semantic encoding
- If DiT has learned non-linear dependencies on specific dimension patterns

## Alternative Approaches

If dimension fixing doesn't fully resolve artifacts:

### 1. Minimal Adapter
Train a small linear adapter (2560→2560) to map embedding space:
```python
adapter = nn.Linear(2560, 2560, bias=False)
# Train on paired (Qwen3-Embedding, Qwen3-4B) embeddings
emb_fixed = adapter(emb_embedding)
```

### 2. LoRA-Style Correction
Add low-rank correction matrices:
```python
emb_fixed = emb + (U @ V^T) @ emb
# Where U, V are learned low-rank matrices (e.g., rank 64)
```

### 3. Ensemble Approach
Blend Qwen3-Embedding with Qwen3-4B:
```python
emb_final = alpha * emb_qwen3 + (1 - alpha) * emb_embedding
# Find optimal alpha (expect 0.7-0.9 for Qwen3-4B)
```

### 4. Different Embedding Model
Test other 2560-dim embedding models:
- GTE-large variants
- E5-large variants
- Custom fine-tuned Qwen3-4B for retrieval

## Research Value

This analysis contributes to understanding:

1. **Embedding space compatibility** - When can you substitute embedding models?
2. **Contrastive vs generative training** - How objectives shape embedding structure
3. **DiT attention sensitivity** - Which embedding characteristics matter most
4. **Zero-shot adaptation** - Can statistical fixes bridge embedding spaces?

## Files and Locations

```
experiments/
├── test_perdim_quick.py              # Quick test script
├── analyze_embedding_perdim.py       # Full analysis script
├── visualize_perdim_results.py       # Visualization script
├── fix_embedding_dimensions.py       # Fix implementation
└── results/
    ├── README.md                     # Usage guide
    ├── perdim_analysis_hypothesis.md # Detailed hypotheses
    ├── embedding_perdim_analysis.json # Analysis data
    └── perdim_visualizations/        # Plots and report
        ├── std_ratio_summary.png
        ├── top_outliers.png
        ├── dimension_categories.png
        ├── correlation_summary.png
        └── analysis_report.md
```

## Next Steps

1. **Run quick test** to validate hypothesis:
   ```bash
   uv run experiments/test_perdim_quick.py
   ```

2. **Review results** and identify top outlier dimensions

3. **Run fix test** with different modes:
   ```bash
   uv run experiments/fix_embedding_dimensions.py --prompt "A cat sleeping"
   ```

4. **Compare images** to see which fix eliminates artifacts

5. **Validate on diverse prompts** before integrating

6. **Document findings** and update CLAUDE.md

## Timeline Estimate

- Analysis (Steps 1-2): 30 minutes
- Testing fixes (Step 3): 1 hour
- Validation (Step 4): 2 hours
- Integration (Steps 5-6): 1 hour

**Total:** ~4-5 hours to complete full investigation and integration

## Success Probability

Based on similar VL embedding analysis where we found:
- Layer -2 had 617x outlier in dimension 396
- Fixing that one dimension significantly improved results
- Full dimension rescaling eliminated artifacts

**Estimated probability of success:** 80-90%

The dimension-level approach should work because:
- ✅ We know the problem is distribution-based (not semantic)
- ✅ We have reference statistics (Qwen3-4B)
- ✅ The fix is non-destructive (preserves cosine similarity)
- ✅ Prior success with similar problem (VL embeddings)

## Conclusion

This investigation provides a systematic approach to diagnosing and fixing embedding compatibility issues. The tools and methodology are reusable for:
- Testing other embedding models with Z-Image
- Adapting embeddings for other DiT models
- Understanding embedding space structure
- Bridging contrastive and generative training objectives

The analysis scripts, fix implementations, and documentation form a complete toolkit for embedding adaptation.
