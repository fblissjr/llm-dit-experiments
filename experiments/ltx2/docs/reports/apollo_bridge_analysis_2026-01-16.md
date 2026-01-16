# Apollo Video-LMM Bridge Analysis: What Transfers to LTX-2

**Last updated: 2026-01-16**

**Purpose**: Analyze Meta's Apollo paper findings (video understanding) for transfer potential to LTX-2 video generation research.

---

## Executive Summary

Apollo is a systematic exploration of video understanding architectures for Video-LMMs. While video understanding and video generation are fundamentally different tasks (one reads video, one writes it), several Apollo findings have surprising relevance to our LTX-2 layer routing research.

**Key transferable insights:**
1. Scaling consistency - small model decisions transfer to large (validates our RTX 4090 experiments)
2. Dual encoder synergy - analogous to multi-layer extraction
3. Temporal compression sweet spots - may inform latent space design intuitions
4. Image encoders beat video encoders on spatial tasks - relevant for frame-level quality

**Key non-transferable aspects:**
1. FPS sampling strategies - generation doesn't "sample" input frames
2. Perceiver Resampler - designed for compression, not expansion
3. Token-per-frame budgets - different information flow direction

---

## 1. Architecture Comparison

### Apollo (Video Understanding)

```
Video Frames
     |
[Visual Encoder] - SigLIP (image) or InternVideo2 (video)
     |
[Perceiver Resampler] - Compress to 8-32 tokens/frame
     |
[LLM Decoder] - Gemma/Qwen/etc for reasoning
     |
Text Output (caption, QA answer)
```

**Information flow**: Video --> Compressed Tokens --> Language

### LTX-2 (Video Generation)

```
Text Prompt
     |
[Gemma-3 12B] - Extract ALL 49 hidden states
     |
[Feature Extractor] - Linear projection (188160 -> 3840)
     |
[Text Connector] - Bidirectional transformer + 128 thinking tokens
     |
[DiT Cross-Attention] - 48 transformer blocks
     |
[VAE Decoder]
     |
Video Output
```

**Information flow**: Language --> Expanded Conditioning --> Video

### Fundamental Difference

| Aspect | Apollo (Understanding) | LTX-2 (Generation) |
|--------|------------------------|-------------------|
| Task | Video --> Text | Text --> Video |
| Encoder role | Compress video | Expand text |
| Token count | Reduce (hundreds --> 8-32) | Preserve/expand (prompt --> latents) |
| LLM role | Decoder (generate text) | Encoder (provide features) |
| Attention type | Causal (for generation) | Cross-attention (for conditioning) |

---

## 2. What Transfers: Finding-by-Finding Analysis

### 2.1 FPS Sampling vs Uniform Sampling

**Apollo Finding**: FPS (Furthest Point Sampling) significantly outperforms uniform sampling for video understanding. Uniform sampling creates inconsistent "video speed" perception.

| Transfer Status | Applicability |
|-----------------|--------------|
| Does NOT transfer | Generation has no input frames to sample |

**Why it doesn't transfer**: LTX-2 generates frames, it doesn't select which input frames to process. The VAE latent space handles temporal compression at generation time.

**Indirect relevance**: If training LTX-2 on video data, FPS-based frame selection for training clips might improve the training distribution. But this affects training data curation, not the model architecture we're experimenting with.

**Verdict**: Skip for our routing experiments.

---

### 2.2 Tokens Per Frame (8-32 Optimal)

**Apollo Finding**: 8-32 tokens per frame is the sweet spot for temporal compression. Fewer loses information, more creates redundancy.

| Transfer Status | Relevance |
|-----------------|-----------|
| Partial transfer | Different direction, but informs temporal reasoning |

**Analysis**:

Apollo compresses: `N frames --> N * (8-32) tokens`

LTX-2 expands: `T text tokens --> (F/8) latent frames --> F pixel frames`

The 8-frame VAE chunk in LTX-2 is doing temporal compression in the **output** space, not input. But there's a curious parallel:

| System | Temporal Unit | Compression Ratio |
|--------|--------------|-------------------|
| Apollo | 8-32 tokens/frame | ~8-32x spatial compression |
| LTX-2 VAE | 8 pixel frames/latent | 8x temporal compression |

**Hypothesis worth testing**: Does the 8-frame chunk boundary in LTX-2 create similar "information quantization" effects that Apollo found with token budgets? Our `chunk_boundary_analysis.py` experiment touches on this.

**Verdict**: Conceptually interesting, but doesn't directly inform layer routing.

---

### 2.3 SigLIP (Image) Beats Video Encoders for Spatial Tasks

**Apollo Finding**: SigLIP (trained on images) outperforms dedicated video encoders (InternVideo2, VideoMAE, V-JEPA) on spatial understanding tasks. Video encoders only win on temporal-specific tasks.

| Transfer Status | Relevance |
|-----------------|-----------|
| Transfers conceptually | Informs our evaluation strategy |

**Why this matters for LTX-2**:

1. **Our SigLIP-based evaluation is appropriate**: We use SigLIP scores to evaluate generated frames. Apollo validates that SigLIP excels at spatial/semantic alignment even for video content.

2. **Per-frame quality matters**: If image encoders beat video encoders at spatial understanding, then per-frame quality in generation is probably the primary quality driver. Temporal coherence is secondary.

3. **Layer routing implications**: Early Gemma layers (syntactic) might matter less than late layers (semantic) for spatial composition. This aligns with our finding that layers 43-47 contribute ~25% of signal.

**Action item**: Continue using SigLIP as primary metric. Consider adding temporal coherence metrics separately (optical flow consistency, CLIP on video, etc.) only if SigLIP plateaus.

---

### 2.4 Video Encoders Help Only on Temporal Tasks

**Apollo Finding**: Dedicated video encoders (InternVideo2, V-JEPA) only provide benefit when the task requires fine-grained temporal understanding. "LLMs struggle with fine-grained temporal integration."

| Transfer Status | Relevance |
|-----------------|-----------|
| Transfers as warning | LLMs (including Gemma) may not provide strong temporal conditioning |

**Critical insight for LTX-2**:

Apollo's text encoder is the LLM itself (Gemma, Qwen). LTX-2's text encoder is also Gemma-3 12B. If Apollo found that LLMs "struggle with fine-grained temporal integration" for *understanding*, the same limitation likely applies to *conditioning*.

**Implications**:

1. **Gemma may be weak for temporal prompts**: Prompts like "ball bounces then rolls" might not produce distinct early/late layer representations for the temporal sequence.

2. **Layer routing may help**: If temporal information is distributed differently across layers than spatial information, per-token routing could learn to route temporal words (verbs, sequence markers) differently than spatial words (nouns, adjectives).

3. **The bidirectional connector compensates**: LTX-2's bidirectional transformer connector allows tokens to see future context, partially compensating for Gemma's causal limitations.

**Testable hypothesis**: Does layer routing improve adherence to temporal prompts more than spatial prompts?

---

### 2.5 Dual Encoders (SigLIP + InternVideo2) Give ~7% Improvement

**Apollo Finding**: Combining SigLIP (spatial) + InternVideo2 (temporal) with embedding interpolation gives ~7% improvement over either alone.

| Transfer Status | Relevance |
|-----------------|-----------|
| Strong parallel | Multi-layer extraction is conceptually similar |

**The analogy**:

| Apollo Dual Encoder | LTX-2 Multi-Layer |
|---------------------|-------------------|
| SigLIP (spatial) | Late Gemma layers (semantic) |
| InternVideo2 (temporal) | Early/middle Gemma layers (syntactic?) |
| Interpolate embeddings | Learned linear projection |

**How Apollo combines**:
```python
# Interpolation at embedding level
combined = alpha * siglip_embed + (1-alpha) * internvideo_embed
```

**How LTX-2 combines (current)**:
```python
# Concatenate all layers, project down
combined = Linear(concat([layer_0, layer_1, ..., layer_48]))
# Shape: [B, T, 188160] -> [B, T, 3840]
```

**How we propose to improve (routing)**:
```python
# Per-token weighted blend
weights = router(token_hidden)  # [B, T, 49]
combined = einsum('btl,btld->btd', weights, layer_stack)
```

**Key insight**: Apollo's 7% improvement from dual encoders suggests that LTX-2's multi-layer extraction is doing something similar. Our routing hypothesis extends this: instead of fixed blending, learn content-dependent blending.

**Prediction**: If Apollo gets 7% from two complementary encoders, we might see 3-7% improvement from routing across 49 layers, assuming the layers truly specialize.

---

### 2.6 Scaling Consistency

**Apollo Finding**: Design decisions that work at small scale (7B LLM) transfer to large scale (72B LLM). "We could significantly reduce the computational cost of architecture exploration by using smaller models."

| Transfer Status | Relevance |
|-----------------|-----------|
| Directly transfers | Validates our RTX 4090 approach |

**This is critical for our research**:

Apollo explicitly validates that:
1. 7B model experiments predict 72B model behavior
2. Hyperparameter tuning transfers across scales
3. Architecture decisions transfer across scales

**For LTX-2 routing**:
- We can validate routing on Gemma-3 12B (our current setup)
- Findings should transfer to larger encoders (Gemma-3 27B, future models)
- Our RTX 4090 experiments are not just "hobbyist approximations" - they're legitimate architecture exploration

**Caveat**: Apollo tested scaling within the *same* model family. Cross-family transfer (e.g., Gemma --> Qwen) is less validated.

---

### 2.7 Text Data Retention (10%) During Video Fine-tuning

**Apollo Finding**: Retaining 10% text-only data during video fine-tuning prevents catastrophic forgetting and maintains LLM reasoning capability.

| Transfer Status | Relevance |
|-----------------|-----------|
| Does not transfer | We're not fine-tuning the LLM |

**Why it doesn't apply**: LTX-2 keeps Gemma-3 frozen. We only train the feature extractor, connector, and DiT. There's no risk of forgetting because we don't modify Gemma's weights.

**Indirect consideration**: If we ever fine-tuned Gemma for better video conditioning (e.g., LoRA), this finding suggests preserving text capability is important.

---

## 3. What Does NOT Transfer

### 3.1 Encoder-Only vs Cross-Attention Paradigm

| Apollo | LTX-2 |
|--------|-------|
| Video encoder produces tokens | Text encoder produces conditioning |
| LLM processes tokens autoregressively | DiT uses cross-attention |
| Single forward pass | Iterative denoising loop |

**Why the difference matters**:

Apollo's video tokens are *input* to the LLM. The LLM sees them once and generates text.

LTX-2's text embeddings are *conditioning* for the DiT. The DiT cross-attends to them at every layer, every denoising step.

**Implication**: Layer routing in LTX-2 affects what the DiT "sees" at all 48 blocks and all ~30 denoising steps. This is much higher leverage than Apollo's single-pass processing.

### 3.2 Perceiver Resampler Irrelevance

Apollo uses Perceiver Resampler to compress video tokens. LTX-2 has no equivalent compression step between text encoder and DiT. The 128 thinking tokens serve a different purpose (global aggregation, not compression).

### 3.3 Different Training Objectives

| Apollo | LTX-2 |
|--------|-------|
| Next-token prediction | Diffusion loss |
| Cross-entropy on text | MSE on latent noise |
| Language modeling | Denoising score matching |

The training signals are completely different, so learned representations optimize for different properties.

---

## 4. Surprising Parallels

### 4.1 "LLMs Struggle with Fine-Grained Temporal Integration"

**Apollo's observation** about video understanding applies to video generation:

- Gemma (LTX-2's encoder) is an LLM
- LLMs process text sequentially with causal attention
- Temporal concepts in text (before/after, while, then) may not produce representations optimized for temporal generation

**LTX-2's architectural response**:
1. Bidirectional connector (fixes causal limitation for text)
2. 128 thinking tokens (global information aggregation)
3. Multi-layer extraction (different temporal information at different depths?)

**Our routing opportunity**: If temporal information IS distributed across layers differently than spatial information, per-token routing could learn to extract temporal signal from the "right" layers for verbs/temporal markers.

### 4.2 Thinking Tokens as Register Tokens

Apollo doesn't use thinking tokens, but the LLM interpretability literature (attention sinks, register tokens) suggests they serve similar purposes:

| Concept | Purpose |
|---------|---------|
| Attention sinks (LLM) | Dump probability mass when nothing useful to attend to |
| Register tokens (ViT) | Learnable scratch space for global computation |
| Thinking tokens (LTX-2) | Bidirectional aggregation + computation space |

**Apollo connection**: Apollo's Perceiver Resampler uses learnable queries. LTX-2's thinking tokens serve a similar role as learnable aggregation targets.

---

## 5. The Temporal Compression Connection

### Apollo's Finding
8-32 tokens per frame is optimal for representing video content to an LLM.

### LTX-2's Architecture
8 pixel frames per latent chunk is the VAE's temporal compression ratio.

### Parallel Analysis

Both systems discovered "8" as a meaningful temporal unit:

| System | Unit | What it represents |
|--------|------|-------------------|
| Apollo | 8-32 tokens/frame | Information density for understanding |
| LTX-2 | 8 frames/latent | Temporal compression for generation |

**Possible explanation**: Human perception groups motion into ~250ms chunks (8 frames at 30fps). Both systems may be learning to align with this perceptual unit.

**Testable hypothesis**: Do transitions at LTX-2 chunk boundaries (frame 8, 16, 24...) exhibit different properties than mid-chunk transitions? (This is what `chunk_boundary_analysis.py` tests.)

---

## 6. Dual Encoder --> Multi-Layer Analogy

### Apollo's Approach

```
SigLIP (spatial expert)    InternVideo2 (temporal expert)
         \                        /
          \                      /
           [Interpolate/Concat]
                   |
              Combined Embedding
```

**Key insight**: Specialized encoders capture different aspects. Combination outperforms either alone.

### LTX-2's Current Approach

```
Gemma Layer 0 (syntax)
Gemma Layer 10 (?)
Gemma Layer 20 (?)        --> [Learned Linear Projection] --> Combined
Gemma Layer 30 (?)
Gemma Layer 40 (semantic)
Gemma Layer 48 (LM head tuned)
```

**Key insight**: Different layers capture different linguistic aspects. Linear combination learns fixed blend.

### Our Routing Proposal

```
Gemma Layer 0   ----\
Gemma Layer 10  -----\
Gemma Layer 20  ------[Per-Token Router] --> Per-Token Weighted Blend
Gemma Layer 30  -----/
Gemma Layer 40  ----/
Gemma Layer 48  ---/
```

**Key insight**: Different tokens may benefit from different layer combinations. Dynamic routing could outperform fixed projection.

### Transfer Assessment

| Apollo Element | LTX-2 Analog | Transfer? |
|----------------|--------------|-----------|
| SigLIP (spatial) | Late Gemma layers | Conceptually |
| InternVideo2 (temporal) | Early/middle layers (?) | Hypothesis |
| Embedding interpolation | Linear projection | Directly |
| Complementary specialization | Layer specialization | Testable |

**Prediction**: If layers truly specialize (like Apollo's dual encoders), routing should improve. If layers are redundant, routing will collapse to uniform weighting.

---

## 7. Actionable Insights for Our Research

### Prioritize

| Action | Rationale | Apollo Support |
|--------|-----------|---------------|
| Continue using SigLIP for evaluation | "SigLIP beats video encoders on spatial" | Strong |
| Test routing across layers | "Dual encoders give 7% improvement" | Moderate |
| Focus on late layers initially | "Semantic understanding in late layers" | Moderate |
| Validate on small scale first | "Scaling consistency" | Strong |

### Deprioritize

| Action | Rationale |
|--------|-----------|
| FPS-like frame selection | Doesn't apply to generation |
| Perceiver-style compression | Different information flow |
| Token budget experiments | Not relevant to text-to-video |

### Test Empirically

| Hypothesis | Test Method | Expected Outcome |
|------------|-------------|-----------------|
| Late layers dominate for spatial prompts | Layer ablation sweep | Layers 40-48 contribute most |
| Different layers for temporal vs spatial | Compare verb-heavy vs noun-heavy prompts | Different routing patterns |
| Routing improves complex prompts more | Complexity-stratified evaluation | >7% on complex, <3% on simple |
| 8-frame boundaries affect transitions | Chunk boundary analysis | Different motion at boundaries |

---

## 8. Summary Table

| Apollo Finding | Transfers to LTX-2? | How to Apply |
|----------------|---------------------|--------------|
| FPS >> uniform sampling | No | N/A (generation, not understanding) |
| 8-32 tokens/frame optimal | Partial | Informs intuitions about temporal compression |
| SigLIP beats video encoders | Yes | Validates SigLIP-based evaluation |
| Video encoders help temporal only | Yes (warning) | LLM conditioning may be weak for temporal |
| Dual encoders give ~7% boost | Yes (analog) | Multi-layer routing is the analog |
| Scaling consistency | Yes | Our 12B experiments predict larger scale |
| 10% text retention | No | We don't fine-tune the LLM |

---

## 9. Recommended Next Steps

### Immediate (This Week)

1. **Run layer ablation sweep** to verify late layers dominate (aligns with Apollo's spatial finding)
2. **Compare routing patterns** for spatial vs temporal prompts
3. **Document layer contribution patterns** to establish baseline

### Short-Term (Next 2 Weeks)

1. **Implement per-token routing** with attention-based mechanism
2. **Compare against uniform blend baseline** with SigLIP scores
3. **Stratify evaluation** by prompt complexity (Apollo suggests benefits scale with complexity)

### Medium-Term (Month)

1. **Analyze routing interpretability** - do tokens route to expected layers?
2. **Test temporal prompts** specifically - Apollo suggests LLMs struggle here
3. **Consider sub-layer routing** (attention vs MLP) for finer granularity

---

## Appendix: Apollo Architecture Details

For reference, Apollo tested these encoders:

| Encoder | Type | Specialty | LTX-2 Analog |
|---------|------|-----------|--------------|
| SigLIP | Image | Spatial semantics | Late Gemma layers |
| DINOv2 | Image | Self-supervised visual | - |
| InternVideo2 | Video | Temporal understanding | Early/middle layers (?) |
| VideoMAE | Video | Self-supervised video | - |
| LanguageBind-Video | Video | Cross-modal alignment | Thinking tokens (?) |
| V-JEPA | Video | Predictive visual | - |

Apollo's best configuration: SigLIP + InternVideo2 with Perceiver Resampler (8-32 tokens/frame).

---

## References

- Apollo Paper: Meta AI (2024) - "Apollo: An Exploration of Video Understanding in Large Multimodal Models"
- LTX-2 Architecture: Lightricks (2024)
- Our layer routing analysis: `experiments/ltx2/docs/text_conditioning_architecture.md`
- Scaling laws validation: Apollo Appendix A (7B vs 72B experiments)
