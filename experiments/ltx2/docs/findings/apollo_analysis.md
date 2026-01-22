# apollo analysis: what transfers from video understanding to generation

*last updated: 2026-01-17*

---

## research status legend

- ✅ **Validated** - Confirmed through experiments or architecture analysis
- 🔬 **Open** - Hypothesis needs testing or re-testing
- ⚠️ **Needs Verification** - Previous results may have bugs
- 🚫 **Dead-End** - Tested, doesn't work

---

## executive summary

Meta's Apollo paper (arXiv:2412.10360) systematically studied Video-LMM design for **video understanding** (video-to-text). LTX-2 performs **video generation** (text-to-video). Despite the reversed information flow, several findings have meaningful implications for our layer routing research.

**Key transfer assessment:**

| Finding | Status | Transfer? | Notes |
|---------|--------|-----------|-------|
| Scaling consistency (R² > 0.8) | ✅ | Yes | Validates small-scale experiments |
| SigLIP best for spatial understanding | ✅ | Yes (for eval) | Validates our evaluation metrics |
| Dual encoder synergy (~7% gain) | 🔬 | Conceptual | Multi-layer routing is our analog |
| 8-frame temporal unit | ✅ | Parallel | Both systems converge on this |
| LLMs struggle with temporal | ✅ | Warning | Applies to text encoding too |
| FPS vs uniform sampling | N/A | No | Generation doesn't sample input |
| Token resampling architecture | N/A | No | Different operation (compress vs project) |

---

## part 1: fundamental architecture difference

### information flow comparison

```
APOLLO (Video Understanding)           LTX-2 (Video Generation)
========================               ========================
Video Frames                           Text Prompt
     |                                      |
[Vision Encoder] SigLIP/InternVideo2   [Gemma-3 12B] 49 layers (frozen)
     |                                      |
[Token Resampling] 256 → 8-32 tokens   [Feature Extractor] 188K → 3840
     |                                      |
[LLM Decoder] Generate text            [Bidirectional Connector] + 128 thinking tokens
     |                                      |
Text Output                            [DiT Cross-Attention] 48 blocks
                                            |
                                       Video Output
```

### why this matters

| Aspect | Apollo | LTX-2 | Implication |
|--------|--------|-------|-------------|
| Primary encoder | Vision (SigLIP) | Text (Gemma-3) | Encoder findings don't transfer directly |
| Operation | Compress visual tokens | Project text conditioning | Opposite optimization pressures |
| LLM role | Generate output | Provide features | Different layer usage patterns |
| Attention | Causal (for generation) | Cross-attention (for conditioning) | Different attention requirements |

---

## part 2: validated findings that transfer

### 2.1 scaling consistency ✅

**Apollo finding**: Design decisions at 7B scale transfer to 72B with R² > 0.9. ~3000 samples sufficient for reliable architecture decisions.

**Status**: ✅ Validated (methodological principle, architecture-agnostic)

**Implications for our research**:
- RTX 4090 experiments on Gemma-3 12B are legitimate architecture exploration
- 500-1000 prompt evaluations valid for routing decisions
- Small-scale findings should transfer to larger scale

**Caveat**: Apollo tested scaling within same model family. Cross-family transfer (Gemma → Qwen) less validated.

### 2.2 SigLIP for evaluation ✅

**Apollo finding**: SigLIP-SO400M is the best single encoder for video tasks, outperforming dedicated video encoders on most benchmarks.

**Status**: ✅ Validated (for evaluation purposes)

**Implications**:
- Frame-level SigLIP scores are valid quality metric
- SigLIP captures video-relevant semantics effectively
- Need separate temporal metrics (SigLIP is spatial-focused)

### 2.3 LLMs struggle with temporal ✅

**Apollo finding**: LLMs are the bottleneck for temporal reasoning. Video encoders only outperform image encoders when tasks specifically require temporal understanding.

**Status**: ✅ Validated (applies as warning)

**Critical insight**: If LLMs struggle to **extract** temporal information from video, they likely also struggle to **inject** temporal information into conditioning. Gemma-3 may not encode "then", "after", "while" distinctly.

**How LTX-2 partially addresses this**:
1. Bidirectional connector allows tokens to see future context
2. 128 thinking tokens provide global aggregation space
3. Multi-layer extraction may capture temporal info at different depths

### 2.4 eight-frame temporal unit ✅

**Apollo finding**: 8 frames often sufficient for video understanding. 8-32 tokens per frame optimal.

**LTX-2 architecture**: VAE uses 8-frame temporal compression.

**Status**: ✅ Validated (intriguing parallel)

| System | Unit | What it represents |
|--------|------|-------------------|
| Apollo | 8-32 tokens/frame | Information density for understanding |
| LTX-2 | 8 frames/latent | Temporal compression for generation |

**Possible explanation**: Human perception groups motion into ~250ms chunks (8 frames at 30fps). Both systems may have learned to align with this perceptual unit.

---

## part 3: findings that need testing 🔬

### 3.1 dual encoder synergy → multi-layer routing 🔬

**Apollo finding**: Combining SigLIP (spatial) + InternVideo2 (temporal) with interpolation gives ~7% improvement.

**Status**: 🔬 Open - conceptual parallel to our routing hypothesis

**The analogy**:

| Apollo Dual Encoder | LTX-2 Multi-Layer Extraction |
|---------------------|------------------------------|
| SigLIP captures spatial semantics | Late Gemma layers capture semantic content |
| InternVideo2 captures temporal dynamics | Early/middle layers capture syntactic/structural |
| ~7% improvement from combination | 🔬 Unknown improvement from routing |

**Prediction**: If layers truly specialize (like Apollo's encoders), per-token routing should show 3-10% improvement on complex prompts. If layers are redundant, routing will collapse to uniform weighting.

**Note**: Jan 15 routing experiments had bugs. This hypothesis needs re-testing with corrected implementation.

### 3.2 temporal token routing 🔬

**Status**: 🔬 Open - hypothesis derived from Apollo temporal finding

**Hypothesis**: Route temporal tokens (verbs, sequence markers) through different layers than spatial tokens (nouns, adjectives). If temporal information IS distributed across layers differently than spatial information, per-token routing could learn to extract temporal signal from the "right" layers.

### 3.3 progressive router training 🔬

**Status**: 🔬 Open - Apollo's staged training principle applied to routing

**Proposed approach**:
1. Phase 1: Reconstruction pretraining (no DiT)
2. Phase 2: Attention mimicry (single DiT forward)
3. Phase 3: Generation fine-tuning (full pipeline)

---

## part 4: findings that need verification ⚠️

### 4.1 layer contribution patterns ⚠️

**Prior analysis**: Late layers (43-47) contribute ~25%, early layers <1%, layer 48 anomalously low (0.02%).

**Status**: ⚠️ Needs verification - may be affected by extraction bugs

These findings were from the same session that had routing implementation bugs. Should re-verify with corrected layer extraction code.

### 4.2 projection weights uniform ⚠️

**Prior analysis**: Projection W has nearly uniform Frobenius norms across layer blocks (~2% variation).

**Status**: ⚠️ Needs verification - depends on correct layer extraction

---

## part 5: findings that don't transfer

### 5.1 encoder selection (does not transfer)

**Apollo finding**: SigLIP-SO400M >> other vision encoders.

**Why not applicable**: We don't have a vision encoder choice. Our "encoder" is Gemma-3 (frozen, non-negotiable for LTX-2 compatibility).

### 5.2 token resampling architecture (does not transfer)

**Apollo finding**: 2-layer MLP + adaptive pooling outperforms Perceiver for compression.

**Why not applicable**: Apollo compresses visual tokens (reduce count). LTX-2 projects text features (maintain count, change dimensionality). Different operations with different optimization targets.

### 5.3 FPS vs uniform sampling (does not transfer)

**Apollo finding**: Consistent FPS sampling vastly outperforms uniform frame sampling.

**Why not applicable**: LTX-2 generates frames; it doesn't sample input frames.

### 5.4 text data retention (does not transfer)

**Apollo finding**: Retain 10% text-only data during video fine-tuning.

**Why not applicable**: We don't fine-tune Gemma-3. It remains frozen.

---

## part 6: research recommendations

### high priority (direct Apollo support)

| Direction | Apollo Justification | Status |
|-----------|---------------------|--------|
| Validate routing at small scale (500-1K prompts) | Scaling consistency | 🔬 Open |
| Use SigLIP + temporal metrics | SigLIP for spatial, gaps for temporal | ✅ Implemented |
| Simple routing before complex | MLP > Perceiver for video | 🔬 Open |

### medium priority (indirect Apollo support)

| Direction | Apollo Parallel | Status |
|-----------|-----------------|--------|
| Per-token temporal routing | LLMs struggle with temporal | 🔬 Open |
| Layer specialization analysis | Dual encoder specialization | ⚠️ Needs verification |
| Chunk-aligned prompting | 8-frame finding | 🔬 Open |

### exploratory (beyond Apollo)

| Direction | Status | Notes |
|-----------|--------|-------|
| Sub-layer routing (attention vs MLP) | 🔬 Open | Apollo uses encoder as black box |
| Timestep-conditional routing | 🔬 Open | Denoising dynamics unique to generation |
| DiT block-specific routing | 🔬 Open | Different blocks may need different layers |

---

## key takeaways

### what Apollo tells us

1. **Our methodology is valid**: Small-scale experiments (1K prompts) on 12B models reliably predict larger-scale behavior
2. **SigLIP metrics are appropriate**: Apollo confirms SigLIP captures video-relevant semantics
3. **Expect modest improvements**: Apollo's ~7% dual encoder gain provides calibration; routing improvements likely in 3-10% range
4. **Temporal is the hard problem**: LLMs struggle with temporal integration; validates focus on temporal token handling
5. **Simpler often wins**: MLP > Perceiver; start with simple linear routing

### what Apollo doesn't tell us

1. **Generation is fundamentally different**: Understanding and generation have reversed information flows
2. **Text encoder internals unexplored**: Apollo treats encoders as black boxes; our layer analysis is novel
3. **Denoising dynamics matter**: Timestep-conditional effects are unstudied
4. **Prompt engineering theory missing**: Why certain structures work remains empirical

---

## appendix: key numbers reference

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

---

## references

- **Apollo Paper**: arXiv:2412.10360, Meta AI (December 2024)
- **LTX-2 Architecture**: Lightricks (2024)
- **Project documentation**: `experiments/ltx2/docs/`

---

*This document consolidates insights from 4 previous Apollo analysis reports (2026-01-16). Routing predictions have been updated to "🔬 Open" status pending re-testing with corrected implementations.*
