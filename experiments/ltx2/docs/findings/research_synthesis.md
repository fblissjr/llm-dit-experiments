# ltx-2 research synthesis

*last updated: 2026-01-17*

---

## research status legend

- ✅ **Validated** - Confirmed through experiments or architecture analysis
- 🔬 **Open** - Hypothesis needs testing or re-testing
- ⚠️ **Needs Verification** - Previous results may have bugs
- 🚫 **Dead-End** - Tested, doesn't work

---

## executive summary

This document synthesizes findings from multiple sources: the LTX-2 paper, Meta's Apollo paper, community observations, and our experimental work. Key research directions are tracked with explicit status markers.

**Core finding**: Multiple sources converge on the importance of **8-frame temporal units** and **layer-specific information**, but per-token semantic routing remains unvalidated due to implementation bugs in early experiments.

---

## part 1: validated architecture findings ✅

### 1.1 text conditioning pipeline ✅

```
Gemma-3 (2B/12B, frozen)
    │
    ├── 49 hidden layers extracted
    │   └── Each: [B, T, 3840]
    │
    ▼
Stack: [B, T, 3840, 49]
    │
    ▼
_pack_text_embeds() - Mean-center, scale to [-8, +8]
    │
    ▼
Flatten: [B, T, 188160]
    │
    ▼
text_proj_in: Linear(188160 → 3840) - LEARNED
    │
    ▼
LTX2ConnectorTransformer1d
    ├── 128 learnable "thinking tokens"
    └── FULL BIDIRECTIONAL ATTENTION (2 layers)
    │
    ▼
DiT Cross-Attention (48 blocks)
```

**Status**: ✅ Validated through architecture inspection

### 1.2 key architectural insights ✅

| Finding | Status | Evidence |
|---------|--------|----------|
| Token dimension preserved throughout | ✅ | Projection operates on layer dim, not token dim |
| Bidirectional connector (not causal) | ✅ | Architecture inspection |
| 128 thinking tokens as global registers | ✅ | Paper Section 3.2.1 |
| 8-frame VAE temporal compression | ✅ | Architecture, valid frame counts: 9, 17, 25... |

### 1.3 vae temporal structure ✅

```
Pixel frames:  [0, 1, 2, 3, 4, 5, 6, 7 | 8, 9, 10, 11, 12, 13, 14, 15 | ...]
                        ↓                           ↓
Latent frames:         [0]                         [1]
```

**Status**: ✅ Validated - 8:1 temporal compression confirmed

---

## part 2: converging evidence (multiple sources)

### 2.1 the 8-frame temporal unit ✅

Three independent sources point to 8 frames as fundamental:

| Source | Finding | Status |
|--------|---------|--------|
| LTX-2 VAE | 8 pixel frames → 1 latent frame | ✅ |
| Apollo Paper | 8 frames often sufficient for understanding | ✅ |
| Community | Transitions may align to 8-frame boundaries | 🔬 |

**Synthesis**: Video AI systems (both understanding and generation) appear to exploit temporal structure at ~0.33 second granularity (8 frames @ 24fps ≈ 333ms).

**Possible explanation**: Human perception groups motion into ~250ms chunks. Both systems may have learned to align with this perceptual unit.

### 2.2 scaling consistency ✅

**Apollo validation**: Design decisions at small scale transfer to large scale (R² > 0.8).

**Implications for our research**:
- RTX 4090 experiments are legitimate architecture exploration
- 500-1000 prompt evaluations sufficient for routing decisions
- Small-scale findings should transfer to larger scale

### 2.3 siglip for evaluation ✅

**Apollo finding**: SigLIP-SO400M captures video semantics effectively.

**Our usage**: SigLIP scores for generated video quality evaluation.

**Status**: ✅ Validated - SigLIP appropriate for spatial quality, but need separate temporal metrics.

---

## part 3: findings needing verification ⚠️

### 3.1 layer contribution patterns ⚠️

**Prior analysis (Jan 15)**:
- Late layers (43-47) contribute ~25% of signal
- Early layers (0-4) contribute <1%
- Layer 48 anomalously low (near-zero norm)

**Status**: ⚠️ Needs verification - may be affected by extraction bugs

**Issue**: Jan 15 experiments had implementation bugs. These layer contribution patterns should be re-verified with corrected layer extraction code.

### 3.2 projection weights uniform ⚠️

**Prior analysis**: Projection W has nearly uniform Frobenius norms across layer blocks (~2% variation).

**Status**: ⚠️ Needs verification - depends on correct layer extraction

### 3.3 token-type layer preferences ⚠️

**Prior claim**: All POS types (nouns, verbs, adjectives) peak uniformly at Layer 46 with similar ratios.

**Status**: ⚠️ Needs verification - analysis may have been affected by bugs

---

## part 4: open research directions 🔬

### 4.1 per-token layer routing 🔬

**Hypothesis**: Different tokens may benefit from different layer mixtures. Routing could improve generation quality.

**Status**: 🔬 Open - needs testing with corrected implementation

**Apollo parallel**: Dual encoder synergy (~7% gain) suggests complementary information in different representations.

**Expected improvement**: 3-10% on complex prompts (calibrated from Apollo findings).

### 4.2 chunk-aligned prompting 🔬

**Hypothesis**: Describing one "event" per 8-frame chunk may align with model's learned temporal structure.

**Status**: 🔬 Open - not yet tested

**Test design**:
```python
# Chunk-aligned: events at 8-frame boundaries
aligned = "Ball drops [0-7]. Ball bounces [8-15]. Ball rolls [16-23]."

# Misaligned: events at arbitrary points
misaligned = "Ball drops [0-10]. Ball bounces [10-20]."
```

### 4.3 temporal token routing 🔬

**Hypothesis**: Route temporal tokens (verbs, sequence markers) through different layers than spatial tokens (nouns, adjectives).

**Status**: 🔬 Open - derived from Apollo's "LLMs struggle with temporal" finding

### 4.4 activation steering 🔬

**Hypothesis**: Find quality/detail directions in embedding space without training.

```python
detailed_acts = mean([encoder(p) for p in detailed_prompts])
vague_acts = mean([encoder(p) for p in vague_prompts])
detail_direction = detailed_acts - vague_acts

steered = hidden + alpha * detail_direction
```

**Status**: 🔬 Open - zero-training approach worth testing

### 4.5 thinking token analysis 🔬

**Hypothesis**: The 128 learnable registers may encode prompt-specific information more valuable than layer selection.

**Status**: 🔬 Open - connector internals unexplored

---

## part 5: what we know about prompting ✅

### 5.1 prompting behavior ✅

| Observation | Status | Explanation |
|-------------|--------|-------------|
| A→B→C structure helps | ✅ | Training data correlations (prompt position ↔ frame content) |
| "Say what you see" works | ✅ | Visual state descriptions match training caption style |
| i2v needs same-family images | ✅ | VAE trained on specific distribution; random images are OOD |
| Compression trick for i2v | ✅ | Introduces artifacts matching training distribution |

### 5.2 what doesn't work 🚫

| Approach | Status | Why |
|----------|--------|-----|
| Implicit temporal ("then", "after") | 🚫 | LLMs encode these weakly (Apollo finding) |
| Narrative continuity without explicit description | 🚫 | No explicit camera/state tracking |
| Random internet images for i2v | 🚫 | Out-of-distribution for VAE |

---

## part 6: llm→dit transfer assessment

### 6.1 techniques that transfer ✅

| Technique | Status | Notes |
|-----------|--------|-------|
| Layer probing | ✅ | Directly applicable |
| Attention visualization | ✅ | Cross-attention analysis |
| Feature extraction | ✅ | Already using all 49 layers |
| KV cache reuse | ✅ | LTX-2 already caches across denoising steps |
| Position interpolation (RoPE) | ✅ | Shared mechanism |

### 6.2 techniques that don't transfer 🚫

| Technique | Status | Why |
|-----------|--------|-----|
| Causal masking effects | 🚫 | System is bidirectional |
| Token probability analysis | 🚫 | DiT doesn't output token probabilities |
| Chain-of-thought reasoning | 🚫 | Different generation paradigm |
| Sequential processing assumptions | 🚫 | Bidirectional attention |

### 6.3 techniques needing modification 🔬

| Technique | Modification | Status |
|-----------|--------------|--------|
| Activation steering | Needs embedding space adaptation | 🔬 |
| Token routing (MoE) | Novel for layer routing | 🔬 |
| Prompt engineering | Visual state descriptions, not task instructions | ✅ |

---

## part 7: research priorities

### tier 0: validation (do first)

| Task | Status | Purpose |
|------|--------|---------|
| Re-verify layer contribution patterns | ⚠️ | Validate with corrected extraction |
| Projection W analysis | ⚠️ | Verify uniform weights finding |
| Layer 48 anomaly verification | ⚠️ | Confirm near-zero norm |

### tier 1: high-value experiments

| Task | Status | Expected Impact |
|------|--------|-----------------|
| Layer routing with corrected code | 🔬 | 3-10% on complex prompts |
| Chunk-aligned prompting | 🔬 | Improved transitions |
| Activation steering | 🔬 | Zero-training quality boost |

### tier 2: exploratory

| Task | Status | Notes |
|------|--------|-------|
| Thinking token analysis | 🔬 | Unexplored connector internals |
| Sub-layer routing (attention vs MLP) | 🔬 | Finer-grained control |
| Timestep-conditional routing | 🔬 | Different layers at different noise levels |

---

## key numbers reference

| Metric | Value | Source |
|--------|-------|--------|
| Gemma-3 layers | 49 | Architecture |
| Hidden dimension | 3840 | Architecture |
| Packed dimension | 188,160 | 49 × 3840 |
| Thinking tokens | 128 | Architecture |
| DiT blocks | 48 | Architecture |
| VAE temporal compression | 8:1 | Architecture |
| Apollo scaling R² | >0.8 | Apollo paper |
| Apollo dual encoder gain | ~7% | Apollo paper |
| Expected routing gain | 3-10% | Calibrated estimate |
| Max context tokens | 1504 | Architecture |

---

## references

- **LTX-2 Paper**: arXiv:2601.03233
- **Apollo Paper**: arXiv:2412.10360, Meta AI
- **Project docs**: `experiments/ltx2/docs/`
- **Apollo analysis**: `experiments/ltx2/docs/findings/apollo_analysis.md`

---

*This document consolidates the Jan 15-16 research synthesis with explicit status tracking. Findings marked ⚠️ require re-verification due to implementation bugs in early experiments.*
