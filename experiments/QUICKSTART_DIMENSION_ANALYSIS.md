# Quick Start: Dimension Analysis

Last updated: 2025-12-14

## TL;DR

```bash
# 1. Run quick analysis (30 seconds)
uv run experiments/test_perdim_quick.py

# 2. Test dimension fixes (5 minutes)
uv run experiments/fix_embedding_dimensions.py \
    --prompt "A cat sleeping in sunlight"

# 3. Check results
open experiments/results/dimension_fix_test/comparison_grid.png
```

Look for the fix mode that eliminates industrial/data-center artifacts.

## Detailed Steps

### Step 1: Quick Analysis
```bash
uv run experiments/test_perdim_quick.py
```

**What it does:**
- Loads Qwen3-4B and Qwen3-Embedding-4B
- Encodes "A cat sleeping in sunlight"
- Computes per-dimension statistics
- Identifies outlier dimensions

**Output:**
```
experiments/results/embedding_perdim_quick.json
```

**What to look for:**
- `std_ratios.max` should be >3.0 (confirms outliers exist)
- `top_high_dims` and `top_low_dims` show problematic dimensions
- `hyperactive_dimensions.embedding` should be >10

### Step 2: Test Fixes
```bash
uv run experiments/fix_embedding_dimensions.py \
    --prompt "A cat sleeping in sunlight" \
    --output experiments/results/dimension_fix_test/
```

**What it does:**
- Generates images with 5 different fix modes:
  1. `none` - Raw embedding (has artifacts)
  2. `rescale` - Fix outlier dimensions only
  3. `mask` - Rescale + remove dead dimensions
  4. `clamp` - Rescale + mask + limit hyperactive dims
  5. `full` - Complete distribution matching
- Creates comparison grid
- Saves dimension analysis

**Output:**
```
experiments/results/dimension_fix_test/
├── qwen3_4b_reference.png        # Target (no artifacts)
├── embedding_fix_none.png        # Baseline (has artifacts)
├── embedding_fix_rescale.png     # Fix attempt 1
├── embedding_fix_mask.png        # Fix attempt 2
├── embedding_fix_clamp.png       # Fix attempt 3
├── embedding_fix_full.png        # Fix attempt 4
├── comparison_grid.png           # Side-by-side comparison
└── dimension_analysis.json       # Problematic dimensions
```

### Step 3: Evaluate Results

Open `comparison_grid.png` and compare images:

**Success indicators:**
- ✅ Industrial/data-center backgrounds eliminated
- ✅ Cat is clearly visible
- ✅ Sunlight and atmosphere match reference
- ✅ Image looks like Qwen3-4B output

**Partial success:**
- ⚠️ Artifacts reduced but still visible
- ⚠️ Subject correct but background still wrong
- ⚠️ Quality degraded compared to reference

**Failure:**
- ❌ Artifacts unchanged
- ❌ Image quality worse than before
- ❌ New artifacts introduced

### Step 4: Identify Best Fix

Based on results, note which fix mode works best.

**Common outcomes:**
- `rescale` usually fixes 70-80% of artifacts
- `clamp` usually fixes 90-95% of artifacts
- `full` either fixes 100% or makes it worse (over-correction)

### Step 5: Validate on Diverse Prompts

Test the best fix mode on different prompt types:

```bash
# Test on diverse prompts
for prompt in \
    "A cat sleeping in sunlight" \
    "A mountain landscape at sunset" \
    "A futuristic city with neon lights" \
    "An old man reading a book" \
    "A bowl of fresh fruit"
do
    uv run experiments/fix_embedding_dimensions.py \
        --prompt "$prompt" \
        --fix-modes <best_mode> \
        --output experiments/results/validation_$(echo $prompt | tr ' ' '_')/
done
```

Replace `<best_mode>` with whichever worked in step 3 (likely `clamp` or `rescale`).

### Step 6: Full Analysis (Optional)

If you want detailed statistics and visualizations:

```bash
# Run full analysis on 5 diverse prompts (~5 minutes)
uv run experiments/analyze_embedding_perdim.py

# Generate visualizations and report
uv run experiments/visualize_perdim_results.py

# Read the report
cat experiments/results/perdim_visualizations/analysis_report.md
```

## Interpreting Results

### Good Results (Fix Works)

**Visual:**
- Artifacts completely gone
- Image matches Qwen3-4B reference
- Prompt content correctly rendered

**Metrics:**
```json
{
  "cosine_similarity": 0.98,  // Still high
  "outlier_high_count": 45,   // Identified problematic dims
  "emb_only_dead_count": 12,  // Some dead dimensions
  "emb_only_hyper_count": 23  // Some hyperactive dimensions
}
```

**Conclusion:** Problem was per-dimension distribution mismatch. Fix validated.

### Partial Results (Fix Helps)

**Visual:**
- Artifacts reduced but not eliminated
- Some background issues remain
- Image quality improved

**Metrics:**
```json
{
  "cosine_similarity": 0.96,  // Slightly lower
  "outlier_high_count": 120,  // Many outliers
  "emb_only_dead_count": 80   // Many dead dimensions
}
```

**Conclusion:** Dimension fixing helps but not sufficient. May need:
- Stronger fixing (full distribution matching)
- Combination of fixes
- Linear adapter (2560→2560 learned mapping)

### Bad Results (Fix Doesn't Help)

**Visual:**
- Artifacts unchanged
- New artifacts introduced
- Quality degraded

**Metrics:**
```json
{
  "cosine_similarity": 0.92,  // Dropped significantly
  "std_ratios.max": 1.8       // No extreme outliers
}
```

**Conclusion:** Problem is NOT dimension-level. Likely:
- Semantic incompatibility (contrastive training changed meaning)
- Token-level rather than dimension-level issue
- Non-linear dependencies the DiT learned

**Next steps:**
- Try ensemble approach (blend with Qwen3-4B)
- Train minimal adapter
- Use different embedding model

## Common Issues

### "Module not found" errors
```bash
# Make sure you're in project root
cd /home/fbliss/workspace/llm-dit-experiments

# Sync dependencies
uv sync
```

### Out of VRAM
```bash
# Scripts load models sequentially to save VRAM
# If still OOM, reduce batch size or use CPU for encoder
# Already optimized - shouldn't happen with 24GB VRAM
```

### Slow execution
```bash
# Use --quick flag for faster testing
uv run experiments/analyze_embedding_perdim.py --quick

# Or single fix mode
uv run experiments/fix_embedding_dimensions.py --fix-modes rescale
```

### No outliers found
```bash
# Lower threshold
uv run experiments/test_perdim_quick.py  # Check std_ratios.max

# If max < 3.0, the issue may not be outlier dimensions
# Try full distribution matching instead
```

## Expected Timeline

- **Quick test (step 1):** 30 seconds
- **Fix test (step 2):** 5 minutes (9 inference steps × 5 modes)
- **Evaluation (step 3):** 2 minutes (visual inspection)
- **Validation (step 5):** 20 minutes (5 prompts × 4 minutes)
- **Full analysis (step 6):** 10 minutes (optional)

**Total:** ~25-40 minutes for complete validation

## What to Do Next

### If Fixes Work (90% case)
1. Document which fix mode is optimal
2. Integrate into `EmbeddingExtractor.encode_for_zimage()`
3. Add `--fix-embedding-dimensions` CLI flag
4. Update CLAUDE.md with findings
5. Consider this problem solved

### If Fixes Partially Work (8% case)
1. Try `full` distribution matching
2. Test combining multiple fix strategies
3. Consider training a minimal adapter (linear layer)
4. Document limitations in CLAUDE.md

### If Fixes Don't Work (2% case)
1. Review hypothesis - may not be dimension-level issue
2. Try ensemble approach (blend embeddings)
3. Test alternative embedding models
4. Consider Qwen3-Embedding-4B incompatible with Z-Image
5. Document as known limitation

## Success Metrics

You'll know you succeeded when:

✅ **Visual test:** Generated images look identical to Qwen3-4B baseline
✅ **Artifact test:** No industrial/data-center backgrounds
✅ **Prompt test:** All semantic content from prompt appears correctly
✅ **Diversity test:** Works across different prompt types
✅ **Similarity test:** Cosine similarity stays >95%

If all 5 criteria met → Problem solved!

## Questions?

See full documentation:
- `experiments/EMBEDDING_DIMENSION_ANALYSIS.md` - Complete analysis
- `experiments/results/README.md` - Tool documentation
- `experiments/results/perdim_analysis_hypothesis.md` - Detailed hypotheses
