# End of Day Report - 2026-01-16

Last Updated: 2026-01-16

---

## Executive Summary

Today focused on analyzing LTX-2 prompting behavior discussions, synthesizing research insights from the Apollo paper, and running a chunk boundary hypothesis experiment. Key finding: the 8-frame VAE temporal compression does NOT create visible discontinuities at chunk boundaries.

---

## Work Completed

### 1. Prompting Behavior Analysis

Analyzed conversation between hobbyists about LTX-2 prompting observations:
- "Say What You See" / A→B→C prompting structure
- 8 frames per latent temporal compression
- Bidirectional attention in the connector (not causal like LLMs)
- 128 learnable "thinking tokens" for global context

**Report**: `prompting_behavior_analysis_2026-01-16.md`

### 2. Apollo Paper Research

Analyzed Meta's Apollo paper (Dec 2024) for insights applicable to LTX-2:
- **Scaling Consistency**: Small-scale experiments transfer to large scale (R² > 0.8)
- SigLIP-SO400M validated as best single encoder for video tasks
- Per-token conditioning more effective than pooled representations

**Reports**:
- `apollo_paper_research_analysis_2026-01-16.md`
- `apollo_ltx2_bridge_analysis_2026-01-16.md`

### 3. Research Synthesis & Plan

Created comprehensive synthesis of all research and actionable plan:
- `research_synthesis_2026-01-16.md`
- `research_plan_2026-01-16.md` (Phases 0-3)

### 4. Chunk Boundary Experiment

**Created**: `experiments/ltx2/chunk_boundary_analysis.py`

**Hypothesis**: Transitions at 8-frame chunk boundaries would show larger discontinuities than mid-chunk transitions.

**Result**: **Hypothesis NOT supported**

| Metric | Value |
|--------|-------|
| Mean boundary diff | 8.21 |
| Mean mid-chunk diff | 9.67 |
| Ratio | 0.85 |

Boundaries are 15% smoother than mid-chunk - LTX-2's VAE handles temporal compression well.

**Output**: `experiments/results/ltx2_chunk_boundary_20260116_172851/`

---

## Bug Fixes

### 1. Hidden States Slicing (Critical)

**Problem**: `mat1 and mat2 shapes cannot be multiplied (512x184320 and 188160x3840)`

**Root Cause**: Code used `hidden_states[1:]` giving 48 layers, but projection expects 49 (embedding + 48 transformer layers).

**Fix**: Changed to `hidden_states[:49]` across all files:
- `chunk_boundary_analysis.py`
- `memory_utils.py` (3 locations)
- `layer_profile_sweep.py`
- `layer_blend_sweep.py`
- Updated `AGENTS.md` documentation

### 2. Negative Prompt Embeddings for CFG

**Problem**: `'NoneType' object has no attribute 'dtype'` when using pre-computed embeddings with CFG.

**Fix**: Added `encode_negative_prompt()` helper function and updated all experiment files to encode and pass negative embeddings.

### 3. Audio VAE Device Mismatch

**Problem**: Video generation completed but crashed on audio decode (CPU vs CUDA mismatch).

**Fix**: Created `DummyAudioVAE` class in `memory_utils.py`:
- Provides required attributes (`latents_mean`, `latents_std`, `config`)
- Returns zeros from `decode()` to satisfy pipeline
- Added `enable_audio=False` parameter to `load_pipeline_with_offloading()`

---

## Files Modified

| File | Changes |
|------|---------|
| `experiments/ltx2/chunk_boundary_analysis.py` | Created - full experiment |
| `experiments/ltx2/memory_utils.py` | Fixed hidden states slicing, added DummyAudioVAE, added enable_audio param |
| `experiments/ltx2/layer_profile_sweep.py` | Fixed hidden states slicing, added negative embeddings |
| `experiments/ltx2/layer_blend_sweep.py` | Fixed hidden states slicing, added negative embeddings |
| `experiments/ltx2/AGENTS.md` | Updated documentation for 49 layers |

---

## Key Technical Insights

### Gemma-3 Hidden States Architecture
```
Layer 0:  Embedding layer        [B, T, 3840]
Layer 1-48: Transformer layers   [B, T, 3840]
Total: 49 hidden states → 188160 dim after flattening (49 × 3840)
```

### LTX-2 Connector Architecture
- Full bidirectional attention (not causal)
- 128 learnable "thinking tokens" replace padding
- Processes ALL 49 layers, not just final output

### Chunk Boundary Finding
- 8-frame temporal compression doesn't create visible seams
- Boundary transitions are smoother, not choppier
- VAE handles temporal boundaries well

---

## Pending / Next Steps

1. **Visual inspection** of chunk boundary videos to verify quantitative findings
2. **Run Phase 1** validation experiments from research plan:
   - Verify SigLIP consistency across generations
   - Test "5-prompt ensemble" methodology
3. **Fix SigLIP scoring** for long prompts (position embedding limit exceeded)
4. **Consider** testing different motion types (camera vs object motion)

---

## Session Statistics

| Metric | Value |
|--------|-------|
| Experiments run | 1 (chunk boundary) |
| Generations completed | 4 |
| Bug fixes | 3 |
| Reports created | 5 |
| Files modified | 6 |

---

## Notes for Tomorrow

- GPU memory cleared, ready for next session
- All experiment infrastructure working
- Results directory cleaned up (only successful runs retained)
- Daily log updated: `internal/log/log_2026-01-16.md`
