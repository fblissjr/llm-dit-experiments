# Apollo to LTX-2 Bridge Analysis: What Transfers Between Video Understanding and Generation

Last updated: 2026-01-16

---

## Executive Summary

Meta's Apollo paper systematically studies Video-LMM design for **video understanding** (video-to-text). LTX-2 performs **video generation** (text-to-video). Despite the reversed information flow, several Apollo findings have meaningful implications for our layer routing research.

**Key Transfer Summary:**

| Category | Transfer Status | Confidence |
|----------|-----------------|------------|
| Scaling consistency for experiments | Transfers | High |
| SigLIP for evaluation metrics | Transfers | High |
| Dual encoder synergy principle | Conceptually transfers | Medium |
| Token resampling architecture | Does NOT transfer | High |
| FPS sampling strategy | Does NOT transfer | High |
| 8-frame temporal finding | Intriguing parallel | Medium |

**Bottom line**: Apollo validates our experimental methodology (small-scale experiments, SigLIP metrics) while providing limited direct architectural guidance. The generation/understanding split creates fundamentally different optimization targets.

---

## Part 1: Fundamental Difference - Understanding vs Generation

### Information Flow Comparison

```
APOLLO (Video Understanding)
==========================
Video Frames
     |
     v
[Vision Encoder] - SigLIP/InternVideo2
     |
     v
[Token Resampling] - Perceiver/Pooling (COMPRESS: 256 tokens -> 8-32 per frame)
     |
     v
[LLM Decoder] - Generate text response
     |
     v
Text Output


LTX-2 (Video Generation)
========================
Text Prompt
     |
     v
[Gemma-3 12B] - 49 decoder layers (frozen)
     |
     v
[Feature Extractor] - Linear projection (EXPAND: 3840*49 -> 3840)
     |
     v
[Bidirectional Connector] - 2 transformer layers + 128 thinking tokens
     |
     v
[DiT Cross-Attention] - 48 blocks attend to text embeddings
     |
     v
[VAE Decoder] - 8 frames per latent chunk
     |
     v
Video Output
```

### Why This Matters

| Aspect | Apollo | LTX-2 | Implication |
|--------|--------|-------|-------------|
| **Primary encoder** | Vision (SigLIP) | Text (Gemma-3) | Apollo's encoder findings don't transfer |
| **Compression vs expansion** | Compress visual tokens | Expand text conditioning | Opposite optimization pressures |
| **LLM role** | Generate output | Provide features | Different layer usage patterns |
| **Attention direction** | Causal (for generation) | Cross-attention (for conditioning) | Different attention requirements |
| **Training objective** | Next-token prediction | Diffusion denoising | Different what gets optimized |

---

## Part 2: What Transfers

### 2.1 Scaling Consistency (Strong Transfer)

**Apollo Finding**: Design decisions at 7B scale transfer to 72B with R^2 > 0.9 correlation. ~3000 samples sufficient for reliable architecture decisions.

**Transfer Assessment**: **TRANSFERS**

**Why it transfers**: This is a methodological finding about neural architecture search, not specific to video understanding.

**Implications for our research**:

| Apollo Validation | Our Analog | Confidence |
|-------------------|------------|------------|
| 7B -> 72B decisions transfer | Gemma-3 12B experiments predict larger scale | High |
| ~3000 samples sufficient | 500-1000 prompt evaluations valid for routing decisions | High |
| Hyperparameters transfer across scale | Router learning rates, temperatures should generalize | Medium |

**Practical application**:
- Our RTX 4090 experiments on Gemma-3 12B are legitimate architecture exploration, not just hobbyist approximations
- We can validate routing strategies with 500-1000 prompts before scaling
- If routing improves quality at small scale, likely maintains at larger scale

**Caveat**: Apollo tested scaling within same model family. Cross-family transfer (Gemma -> Qwen) less validated.

---

### 2.2 SigLIP for Evaluation (Strong Transfer)

**Apollo Finding**: SigLIP-SO400M is the best single encoder for video tasks, outperforming dedicated video encoders on most benchmarks.

**Transfer Assessment**: **TRANSFERS** (for evaluation purposes)

**Why it transfers**: We use SigLIP to evaluate generated frames. Apollo validates that SigLIP captures video-relevant semantics effectively.

**Implications**:

| Apollo Insight | Our Application |
|----------------|-----------------|
| SigLIP best for spatial understanding | Frame-level SigLIP scores valid quality metric |
| Video encoders only win on temporal-specific tasks | Need separate temporal coherence metrics |
| Language-supervised > self-supervised | Our SigLIP-based metrics well-calibrated |

**Practical application**:
- Continue using SigLIP as primary generation quality metric
- Add complementary temporal metrics (optical flow consistency, frame interpolation) separately
- Don't expect SigLIP alone to capture temporal coherence improvements

---

### 2.3 Dual Encoder Synergy (Conceptual Transfer)

**Apollo Finding**: Combining SigLIP (spatial) + InternVideo2 (temporal) with embedding interpolation gives ~7% improvement over either alone.

**Transfer Assessment**: **CONCEPTUALLY TRANSFERS**

**The analogy**:

| Apollo Dual Encoder | LTX-2 Multi-Layer Extraction |
|---------------------|------------------------------|
| SigLIP captures spatial semantics | Late Gemma layers capture semantic content |
| InternVideo2 captures temporal dynamics | Early/middle layers capture syntactic/structural content |
| Channel-wise concatenation + interpolation | Concatenate all 49 layers + learned projection |
| ~7% improvement from combination | Unknown improvement from routing (our research target) |

**What transfers**:
- The principle that specialized representations complement each other
- Combining different "views" of the same input improves downstream tasks
- Learned interpolation/combination outperforms simple averaging

**What doesn't transfer**:
- Specific encoder choices (vision vs text)
- Compression ratios (Apollo compresses, we project)
- The 7% improvement magnitude (different domains)

**Prediction for our research**: If Gemma layers truly specialize (like Apollo's encoders), per-token routing should show 3-10% improvement on complex prompts. If layers are redundant, routing will collapse to uniform weighting.

---

### 2.4 "LLMs Struggle with Fine-Grained Temporal Integration" (Warning Transfer)

**Apollo Finding**: LLMs are the bottleneck for temporal reasoning in video understanding. Video encoders only outperform image encoders when tasks specifically require temporal understanding.

**Transfer Assessment**: **TRANSFERS AS WARNING**

**Why this is critical**:
- LTX-2 uses Gemma-3 (an LLM) for text encoding
- If LLMs struggle to **extract** temporal information from video, they likely struggle to **inject** temporal information into conditioning
- This limitation exists at the encoder level, before any generation happens

**Implications for text-to-video**:

| Apollo Observation | LTX-2 Analog |
|--------------------|--------------|
| LLMs can't extract fine-grained temporal relationships | Gemma may not encode "then", "after", "while" distinctly |
| Video encoders help only for temporal tasks | No "temporal encoder" available for text conditioning |
| ~7% improvement from video encoder | Unknown if routing can compensate |

**Why LTX-2's architecture partially addresses this**:

1. **Bidirectional connector**: Allows tokens to see future context, partially compensating for Gemma's causal limitations
2. **128 thinking tokens**: Provide global aggregation space for temporal relationships
3. **Multi-layer extraction**: Different layers may encode temporal information differently

**Research opportunity**: Per-token routing could route temporal words (verbs, sequence markers) through different layers than spatial words (nouns, adjectives). This is the core hypothesis of our routing research.

---

## Part 3: What Does NOT Transfer

### 3.1 Encoder Selection (Does Not Transfer)

**Apollo Finding**: SigLIP-SO400M >> other vision encoders for video understanding.

**Transfer Assessment**: **DOES NOT TRANSFER**

**Why not**: We don't have a vision encoder choice to make. Our "encoder" is Gemma-3 (frozen, non-negotiable for LTX-2 compatibility).

**What to do instead**: Focus on how to best extract features FROM Gemma-3, not which encoder to use.

---

### 3.2 Token Resampling Architecture (Does Not Transfer)

**Apollo Finding**: 2-layer MLP + adaptive average pooling outperforms Perceiver Resampler for video token compression.

**Transfer Assessment**: **DOES NOT TRANSFER**

**Why not**: Apollo compresses visual tokens (reduce count). LTX-2 projects text features (maintain count, change dimensionality).

| Apollo Token Resampling | LTX-2 Feature Extraction |
|-------------------------|--------------------------|
| Goal: Compress 256+ visual tokens to 8-32 | Goal: Project 49 layer features to single representation |
| Operation: Reduce sequence length | Operation: Reduce feature depth |
| MLP + pooling best | Linear projection used (different optimization target) |

**Why the architectures differ**:

```python
# Apollo: Compress sequence length
# Input: [B, 256, D] (256 visual tokens)
# Output: [B, 32, D] (32 resampled tokens)
# Method: Perceiver cross-attention or MLP + pooling

# LTX-2: Project feature depth
# Input: [B, T, 3840, 49] (49 layers stacked)
# Output: [B, T, 3840] (single projection)
# Method: Normalize + linear projection
```

**What is relevant**: Apollo found that simpler architectures (MLP) outperform complex ones (Perceiver) for video. This MAY suggest that LTX-2's simple linear projection is appropriate, but the optimization target is too different to draw strong conclusions.

---

### 3.3 FPS vs Uniform Sampling (Does Not Transfer)

**Apollo Finding**: Consistent FPS sampling vastly outperforms uniform frame sampling for video understanding.

**Transfer Assessment**: **DOES NOT TRANSFER** to inference

**Why not**: LTX-2 generates frames; it doesn't sample input frames. There's no "sampling strategy" choice at inference time.

**Weak indirect relevance**: If LTX-2 were trained on video data with poor temporal sampling, the model might have learned inconsistent temporal representations. But we can't change this post-training.

---

### 3.4 Text Data Retention (Does Not Transfer)

**Apollo Finding**: Retain 10% text-only data during video fine-tuning to prevent catastrophic forgetting.

**Transfer Assessment**: **DOES NOT TRANSFER**

**Why not**: We don't fine-tune Gemma-3. It remains frozen. No forgetting risk.

---

## Part 4: Specific Bridge Insights

### 4.1 SigLIP for Metrics - Justified

**Apollo validation**: SigLIP is the best single encoder for capturing video semantics.

**Our usage**: SigLIP scores to evaluate generated video quality.

**Assessment**: **Well-justified**

| Aspect | Apollo Evidence | Our Metric Design |
|--------|-----------------|-------------------|
| Spatial understanding | SigLIP best for spatial | Frame-level SigLIP captures composition |
| Semantic alignment | Language-supervised encoders win | SigLIP trained with text alignment |
| Temporal limitation | Video encoders only help for temporal | Need separate temporal metrics |

**Recommendation**: Continue SigLIP as primary metric. Add optical flow consistency or frame-to-frame LPIPS for temporal coherence evaluation.

---

### 4.2 Token Resampling vs Linear Projection - Different Problems

**Apollo architecture**:
```python
# Apollo: Perceiver Resampler (found suboptimal)
learned_queries = nn.Parameter([32, D])  # 32 query tokens
resampled = cross_attention(Q=learned_queries, K=visual_tokens, V=visual_tokens)
# Output: [B, 32, D]

# Apollo: MLP + Pooling (found optimal)
projected = MLP(visual_tokens)  # [B, 256, D] -> [B, 256, D]
pooled = adaptive_avg_pool(projected)  # [B, 256, D] -> [B, 32, D]
```

**LTX-2 architecture**:
```python
# LTX-2: Multi-layer projection
stacked = torch.stack(hidden_states, dim=-1)  # [B, T, 3840, 49]
normalized = per_layer_normalize(stacked)  # Per-layer normalization
flattened = rearrange(normalized, 'b t d l -> b t (d l)')  # [B, T, 188160]
projected = Linear(188160, 3840)(flattened)  # [B, T, 3840]
```

**Comparison**:

| Aspect | Apollo Resampling | LTX-2 Projection |
|--------|-------------------|------------------|
| Operation | Reduce sequence length | Reduce feature depth |
| Compression ratio | 8x (256->32) | 49x (188160->3840) |
| Cross-attention | Yes (Perceiver) / No (MLP+Pool) | No (linear) |
| Learned queries | 32 tokens | None |
| Information flow | Visual -> fewer visual | Multi-layer -> single layer |

**Key insight**: LTX-2's linear projection is conceptually closer to Apollo's MLP + pooling (simple, no cross-attention) than to Perceiver Resampler. Apollo finding that simpler is better may indirectly support LTX-2's design.

---

### 4.3 Temporal Compression - The 8-Frame Coincidence

**Apollo finding**: 8 frames often sufficient for video understanding. 8-32 tokens per frame optimal.

**LTX-2 architecture**: VAE uses 8-frame temporal compression (8 pixel frames per latent).

**Is this coincidence or fundamental?**

| Evidence for coincidence | Evidence for fundamental |
|-------------------------|-------------------------|
| Different optimization targets | Human perception chunks at ~250ms (8 frames @ 30fps) |
| Apollo measures understanding, not generation | Both systems learned similar temporal unit |
| VAE designed independently of Apollo | Cognitive research suggests 200-300ms event boundaries |

**Hypothesis**: 8 frames may be a perceptually meaningful temporal unit that both systems independently converged on.

**Implications for our research**:

1. **Chunk boundary hypothesis strengthened**: If 8 frames is fundamental to temporal perception, LTX-2's chunk boundaries may create real "event boundaries" in generated video

2. **Prompt alignment**: Describing one "event" per 8-frame chunk may align with model's learned temporal structure

3. **Router temporal awareness**: Could route differently at chunk boundaries vs mid-chunk

**Testable prediction**: Transitions described at chunk boundaries (frames 8, 16, 24...) will be sharper than mid-chunk transitions.

---

### 4.4 Layer Aggregation - Similar Principle

**Apollo approach**: Channel-wise concatenation of SigLIP + InternVideo2 embeddings, then projection.

```python
# Apollo dual encoder combination
siglip_embed = siglip_encoder(frames)  # [B, T, D_siglip]
intern_embed = internvideo_encoder(frames)  # [B, T, D_intern]
combined = concat([siglip_embed, intern_embed], dim=-1)  # [B, T, D_siglip+D_intern]
projected = Linear(D_siglip+D_intern, D_output)(combined)
```

**LTX-2 approach**: Stack 49 Gemma layers, normalize, project.

```python
# LTX-2 multi-layer combination
layer_outputs = gemma(text, output_hidden_states=True).hidden_states[1:]  # 49 layers
stacked = torch.stack(layer_outputs, dim=-1)  # [B, T, 3840, 49]
normalized = per_layer_normalize(stacked)
flattened = rearrange(normalized, 'b t d l -> b t (d l)')  # [B, T, 188160]
projected = Linear(188160, 3840)(flattened)
```

**Principle alignment**: Both systems:
1. Combine multiple feature sources (encoders / layers)
2. Use concatenation-based combination
3. Learn a projection to unify

**Key difference**: Apollo combines 2 specialized encoders. LTX-2 combines 49 layers from single encoder.

**What this suggests for routing**: Apollo's ~7% improvement from dual encoders provides upper bound intuition. Our 49 layers may have less specialization than two purpose-built encoders, suggesting more modest improvements from routing.

---

### 4.5 Scaling Consistency - Validates Our Experiments

**Apollo methodology**:
- Test at 7B parameter scale
- Validate decisions transfer to 72B
- Use ~3000 samples for reliable design decisions

**Our methodology**:
- Test on Gemma-3 12B (frozen) + small routers (~500K params)
- 500-1000 prompt evaluations
- RTX 4090 compute budget

**Apollo justification for our approach**:

| Our limitation | Apollo evidence it's okay |
|----------------|---------------------------|
| Can't test 70B+ models | 7B decisions transfer to 72B |
| Limited prompt budget | ~3000 samples sufficient for design decisions |
| Single GPU experiments | Architecture decisions don't require massive compute |
| Small routing networks | Design choices matter more than scale at exploration phase |

**Confidence boost**: Our experiments are not "toy" - they're valid architecture exploration per Apollo's methodology findings.

---

## Part 5: Implications for Our Research

### 5.1 Does Apollo Inform Gemma Layer Routing?

**Short answer**: Indirectly yes.

**Direct guidance**: None. Apollo doesn't study text encoder layer contributions.

**Indirect guidance**:

| Apollo Principle | Routing Implication |
|------------------|---------------------|
| Specialized encoders complement each other | Specialized layers may complement each other |
| LLMs struggle with temporal | Route temporal tokens differently |
| Simpler aggregation often wins | Start with simple routing before complex |
| ~7% dual encoder improvement | Expect modest (3-10%) routing improvements |

**Research priority**: Validate layer specialization hypothesis before investing in complex routing architectures.

---

### 5.2 Should We Consider Dual Encoding (Spatial + Temporal)?

**Apollo finding**: SigLIP (spatial) + InternVideo2 (temporal) is optimal.

**Translation to text conditioning**: Would require:
- Spatial text encoder (emphasis on objects, composition)
- Temporal text encoder (emphasis on actions, sequences)
- Combination mechanism

**Assessment**: **NOT RECOMMENDED** at this stage

**Why not**:
1. We can't swap encoders (LTX-2 trained on Gemma-3)
2. Adding second encoder would require retraining DiT cross-attention
3. Layer routing is a cheaper way to get encoder diversity
4. Temporal encoding may be better addressed through prompt structure

**Alternative approach**: Our layer routing already provides "encoder diversity" through different depth features. Validate this before adding external encoders.

---

### 5.3 Does the 8-Frame Finding Relate to Chunk Boundaries?

**Assessment**: **PLAUSIBLE CONNECTION**

**Argument for connection**:
- Both systems arrived at 8 frames as meaningful temporal unit
- May reflect human perceptual event boundaries (~250ms)
- Cognitive science supports ~200-300ms event segmentation

**Argument against connection**:
- Apollo's 8 frames is about **sampling** for understanding
- LTX-2's 8 frames is about **compression** for generation
- Different optimization pressures could yield same number coincidentally

**Testable hypotheses**:

| Hypothesis | Test Method |
|------------|-------------|
| Chunk boundaries are perceptual event boundaries | Analyze human ratings of generated transitions by frame position |
| Prompts aligned to chunks generate better | Compare chunk-aligned vs misaligned prompt structures |
| Routing should differ at boundaries | Analyze learned router attention at boundary positions |

**Research priority**: Medium. Interesting theoretical question, but routing experiments are more actionable.

---

### 5.4 How Does Apollo's Progressive Training Relate to Router Training?

**Apollo finding**: Progressive training across stages (freeze LLM -> unfreeze -> full training) works well.

**Our context**: We train only the router. Gemma-3 and DiT remain frozen.

**Possible adaptation**:

```
Phase 1: Reconstruction pretraining (no DiT)
         - Train router to reconstruct final layer from earlier layers
         - Cheap, establishes baseline routing patterns

Phase 2: Attention mimicry (single DiT forward)
         - Extract DiT cross-attention patterns
         - Train router to pre-emphasize tokens DiT naturally attends to

Phase 3: Generation fine-tuning (full DiT inference)
         - Optimize routing for generation quality metrics
         - Most expensive, but directly optimizes target
```

**Apollo parallel**:
- Phase 1 = Apollo's "freeze LLM" stage (establish representations)
- Phase 2 = Apollo's "unfreeze" stage (integrate with downstream)
- Phase 3 = Apollo's "full training" stage (end-to-end optimization)

**Recommendation**: Follow this progressive schedule for router training. Apollo validates staged approach.

---

## Part 6: Research Directions Suggested by Apollo

### 6.1 High Priority (Direct Apollo Support)

| Direction | Apollo Justification | Implementation |
|-----------|---------------------|----------------|
| Validate routing at small scale first | Scaling consistency | 500-1000 prompt experiments |
| Use SigLIP + temporal metrics | SigLIP for spatial, gaps for temporal | Add optical flow metrics |
| Progressive router training | Staged training works | Reconstruction -> mimicry -> generation |
| Simple routing before complex | MLP > Perceiver for video | Start with linear router, not attention-based |

### 6.2 Medium Priority (Indirect Apollo Support)

| Direction | Apollo Parallel | Uncertainty |
|-----------|-----------------|-------------|
| Per-token temporal routing | LLMs struggle with temporal | Untested whether routing helps |
| Layer specialization analysis | Dual encoder specialization | Don't know if Gemma layers specialize like encoders |
| Chunk-aligned prompting | 8-frame finding | Unclear if understanding -> generation |

### 6.3 Exploratory (Novel Directions Beyond Apollo)

| Direction | Why Apollo Doesn't Cover | Research Question |
|-----------|--------------------------|-------------------|
| Sub-layer routing (attention vs MLP) | Apollo uses encoder as black box | Do attention and MLP layers specialize differently? |
| Timestep-conditional routing | Apollo doesn't have denoising | Should early vs late diffusion steps use different layers? |
| DiT block-specific routing | Apollo has single LLM consumption point | Should different DiT blocks use different layer mixtures? |

---

## Part 7: Specific Experiment Recommendations

### 7.1 Experiments Apollo Validates

**1. Small-Scale Routing Validation**

Apollo's scaling consistency says this will transfer to larger scale.

```python
# Experiment design
n_prompts = 1000
routing_strategies = ['uniform', 'late_layers', 'learned_router']
metrics = ['siglip_score', 'temporal_consistency']

for strategy in routing_strategies:
    scores = evaluate_strategy(strategy, prompts[:n_prompts])
    # If differences are clear at 1000 prompts, they're real
```

**2. SigLIP + Temporal Metric Evaluation**

Apollo shows SigLIP is spatial-focused. Add temporal metric.

```python
# Comprehensive evaluation suite
def evaluate_video(video, prompt):
    spatial = siglip_frame_scores(video, prompt).mean()
    temporal = optical_flow_consistency(video)
    semantic = llm_judge_prompt_adherence(video, prompt)
    return {
        'spatial': spatial,      # Apollo-validated
        'temporal': temporal,    # Apollo gap
        'semantic': semantic,    # Beyond Apollo
    }
```

**3. Progressive Router Training**

Apollo's staged training principle applied to routing.

```python
# Phase 1: Reconstruction (cheap)
router.train_reconstruction(encoder, prompts, steps=1000)

# Phase 2: Attention mimicry (medium)
router.train_mimicry(encoder, dit, prompts, steps=2000)

# Phase 3: Generation quality (expensive)
router.train_generation(full_pipeline, prompts, steps=1000)
```

### 7.2 Experiments to Deprioritize Based on Apollo

| Experiment | Reason to Deprioritize |
|------------|------------------------|
| Vision encoder swapping | Apollo findings don't transfer to text conditioning |
| Perceiver-style routing | Apollo found simpler (MLP) better |
| Massive prompt datasets | Apollo shows ~3000 sufficient for design decisions |
| End-to-end DiT fine-tuning | Out of scope, not validated by Apollo |

### 7.3 Novel Experiments Beyond Apollo

**1. Temporal Token Routing**

Apollo's "LLMs struggle with temporal" + our routing hypothesis.

```python
# Hypothesis: Route temporal tokens through different layers than spatial tokens
temporal_tokens = identify_temporal_tokens(prompt)  # verbs, sequence markers
spatial_tokens = identify_spatial_tokens(prompt)    # nouns, adjectives

# Learn different routing for each type
routing_weights = router(hidden_states, token_types)
```

**2. Sub-Layer Component Analysis**

Apollo treats encoders as black boxes. We can look inside Gemma.

```python
# Extract attention-only and MLP-only contributions
with SubLayerExtractor(gemma, layers=[0, 20, 40, 48]) as extractor:
    outputs = gemma(text)
    attention_only = extractor.attention_outputs
    mlp_only = extractor.mlp_outputs

# Compare routing based on attention vs MLP features
```

**3. Chunk Boundary Analysis**

Test if 8-frame Apollo finding relates to generation.

```python
# Generate video with transition at different frame positions
transition_frames = [4, 8, 12, 16, 20, 24]  # Some aligned to chunks, some not
for frame in transition_frames:
    prompt = f"Ball is red for frames 0-{frame-1}, then turns blue"
    video = generate(prompt)
    transition_quality[frame] = measure_transition_sharpness(video, frame)

# Hypothesis: frame=8, 16, 24 have sharper transitions
```

---

## Part 8: Summary Tables

### 8.1 Transfer Assessment Summary

| Apollo Finding | Transfer Status | Confidence | Action |
|----------------|-----------------|------------|--------|
| Scaling consistency | TRANSFERS | High | Validate routing at 1K prompts |
| SigLIP best encoder | TRANSFERS (eval) | High | Continue SigLIP metrics |
| Dual encoder synergy | CONCEPTUAL | Medium | Treat layers as "encoders" |
| 8 tokens/frame optimal | Does not transfer | High | Different problem |
| MLP > Perceiver | WEAK PARALLEL | Low | Consider simple routing first |
| FPS > uniform sampling | Does not transfer | High | Not applicable |
| LLMs struggle temporal | TRANSFERS (warning) | High | Route temporal tokens differently |
| Progressive training | TRANSFERS (method) | Medium | Staged router training |
| Text data retention | Does not transfer | High | We don't fine-tune Gemma |

### 8.2 Architecture Comparison Summary

| Component | Apollo | LTX-2 | Transfer? |
|-----------|--------|-------|-----------|
| Primary encoder | Vision (SigLIP) | Text (Gemma-3) | No |
| Encoder combinations | 2 encoders | 49 layers | Conceptual |
| Token resampling | Compress (256->32) | Project (188K->3.8K) | No |
| Downstream model | LLM decoder | DiT cross-attention | No |
| Training objective | Next-token | Denoising | No |
| Temporal handling | Video encoder | Bidirectional connector | Different approaches |

### 8.3 Research Priority Matrix

| Priority | Research Direction | Apollo Support | Effort | Expected Impact |
|----------|-------------------|----------------|--------|-----------------|
| 1 | Per-token layer routing | Indirect (temporal) | Medium | 3-10% on complex prompts |
| 2 | Validate scaling consistency | Direct | Low | Methodology confidence |
| 3 | Progressive router training | Direct | Medium | Better convergence |
| 4 | Sub-layer (attn/MLP) routing | None (novel) | Medium | Unknown |
| 5 | Chunk boundary analysis | Indirect (8-frame) | Low | Unknown |
| 6 | Timestep-conditional routing | None (novel) | High | Unknown |

---

## Part 9: Conclusions

### What Apollo Tells Us

1. **Our methodology is valid**: Small-scale experiments (1K prompts) on smaller models (12B) reliably predict larger-scale behavior. Our RTX 4090 experiments are legitimate architecture exploration.

2. **SigLIP metrics are appropriate**: Apollo confirms SigLIP captures video-relevant semantics. Our frame-level SigLIP evaluation is well-justified.

3. **Expect modest improvements**: Apollo's ~7% dual encoder gain provides calibration. Our routing improvements likely in 3-10% range on complex prompts.

4. **Temporal is the hard problem**: Apollo's finding that "LLMs struggle with fine-grained temporal integration" validates our focus on how text encoding handles temporal concepts.

5. **Simpler architectures often win**: Apollo's MLP > Perceiver finding suggests starting with simple linear routing before attention-based routing.

### What Apollo Doesn't Tell Us

1. **Generation is fundamentally different**: Understanding (video->text) and generation (text->video) have reversed information flows. Direct architecture transfer is limited.

2. **Text encoder internals unexplored**: Apollo treats encoders as black boxes. Our layer-level analysis is novel territory.

3. **Denoising dynamics matter**: Apollo's single-pass LLM is very different from iterative DiT denoising. Timestep-conditional effects are unstudied.

4. **Prompt engineering theory missing**: Why certain prompt structures work for generation remains empirical, not principled.

### Final Recommendations

**Do**:
- Use Apollo's scaling consistency to validate experiments at small scale
- Trust SigLIP metrics for spatial quality; add temporal metrics separately
- Apply progressive training strategy to router development
- Start with simple routing mechanisms before complex ones
- Route temporal tokens differently (Apollo supports this direction)

**Don't**:
- Copy Apollo's encoder selection findings (different domain)
- Expect >10% improvements (Apollo's gains were modest)
- Skip small-scale validation (Apollo says it predicts large-scale)
- Use Apollo's token compression ratios (different operation)

---

## References

### Apollo Paper
- Title: "Apollo: An Exploration of Video Understanding in Large Multimodal Models"
- Source: Meta AI (arXiv:2412.10360)
- Key findings: 84 model configurations tested, scaling consistency validated

### LTX-2 Architecture
- Text encoder: Gemma-3 12B (49 decoder layers, frozen)
- Feature extraction: Linear projection (188160 -> 3840)
- Connector: Bidirectional transformer + 128 thinking tokens
- Generator: Dual-stream DiT (14B video, 5B audio)

### Project Documentation
- `experiments/ltx2/docs/text_conditioning_architecture.md` - Full architecture details
- `experiments/ltx2/docs/gemma3_sublayer_architecture.md` - Sub-layer extraction
- `experiments/ltx2/ltx2_research_guide.md` - Research context
- `experiments/ltx2/docs/reports/apollo_research_analysis_2026-01-16.md` - Detailed Apollo analysis

---

## Appendix A: Detailed Architecture Comparison

### Apollo Vision Encoder Pipeline

```python
# Apollo's vision encoding (simplified)
class ApolloVisionEncoder:
    def __init__(self):
        self.siglip = SigLIP_SO400M()  # Spatial encoder
        self.internvideo = InternVideo2()  # Temporal encoder
        self.resampler = MLPResampler(
            input_tokens=256,
            output_tokens=32,
            hidden_dim=4096
        )

    def forward(self, frames):
        # frames: [B, T, H, W, C]
        siglip_features = self.siglip(frames)  # [B, T, 256, D_siglip]
        intern_features = self.internvideo(frames)  # [B, T, 256, D_intern]

        # Concatenate encoders
        combined = concat([siglip_features, intern_features], dim=-1)

        # Resample to fewer tokens
        resampled = self.resampler(combined)  # [B, T, 32, D_output]

        return resampled
```

### LTX-2 Text Encoder Pipeline

```python
# LTX-2's text encoding (simplified)
class LTX2TextEncoder:
    def __init__(self):
        self.gemma = Gemma3_12B(frozen=True)
        self.projector = Linear(3840 * 49, 3840)
        self.connector = BidirectionalTransformer(
            num_layers=2,
            hidden_dim=3840,
            num_thinking_tokens=128
        )

    def forward(self, text):
        # text: str or List[str]

        # Get all layer hidden states
        outputs = self.gemma(text, output_hidden_states=True)
        hidden_states = outputs.hidden_states[1:]  # Skip embedding, get 49 layers

        # Stack and normalize
        stacked = torch.stack(hidden_states, dim=-1)  # [B, T, 3840, 49]
        normalized = per_layer_normalize(stacked)

        # Project layers to single representation
        flattened = rearrange(normalized, 'b t d l -> b t (d l)')
        projected = self.projector(flattened)  # [B, T, 3840]

        # Bidirectional processing with thinking tokens
        with_thinking = self.connector(projected)  # [B, T+128, 3840]

        return with_thinking
```

### Key Differences Table

| Aspect | Apollo | LTX-2 |
|--------|--------|-------|
| Input modality | Video frames | Text tokens |
| Encoder count | 2 (SigLIP + InternVideo2) | 1 (Gemma-3) |
| Feature sources | 2 specialized encoders | 49 generalist layers |
| Combination method | Concatenate encoders | Concatenate layers |
| Token count change | Reduce (256 -> 32) | Maintain (T -> T+128) |
| Cross-attention | None in encoder | Bidirectional in connector |
| Downstream model | LLM decoder | DiT cross-attention |

---

## Appendix B: Experimental Setup Recommendations

### Minimum Viable Experiment Design

Based on Apollo's scaling consistency findings:

```python
# Recommended experimental setup
EXPERIMENT_CONFIG = {
    'n_prompts': 1000,  # Apollo: ~3000 sufficient, we can start smaller
    'n_seeds_per_prompt': 3,  # For variance estimation
    'metrics': ['siglip', 'temporal_consistency', 'human_preference'],

    # Routing strategies to compare
    'strategies': [
        'uniform',           # Baseline: equal layer weights
        'late_emphasis',     # Heuristic: emphasize layers 40-48
        'learned_global',    # Learned but same for all tokens
        'learned_per_token', # Full routing (our hypothesis)
    ],

    # Training budget
    'router_training_steps': 3000,
    'training_prompts': 500,
    'validation_prompts': 500,
}

# Expected runtime on RTX 4090
# - Encoding: ~0.5s per prompt
# - Generation: ~30s per video
# - Evaluation: ~2s per video
# - Total: ~50 hours for full experiment
```

### Evaluation Protocol

```python
def comprehensive_evaluation(video, prompt, baseline_video=None):
    """
    Evaluation protocol inspired by Apollo's rigorous methodology.
    """
    results = {}

    # 1. Spatial quality (Apollo-validated)
    frame_scores = [siglip_score(frame, prompt) for frame in video]
    results['siglip_mean'] = np.mean(frame_scores)
    results['siglip_std'] = np.std(frame_scores)  # Consistency across frames

    # 2. Temporal coherence (Apollo gap - add our own)
    results['flow_consistency'] = optical_flow_smoothness(video)
    results['frame_lpips'] = mean_adjacent_lpips(video)

    # 3. Semantic accuracy (beyond Apollo)
    results['prompt_adherence'] = llm_judge_prompt_match(video, prompt)

    # 4. Relative improvement (if baseline provided)
    if baseline_video is not None:
        baseline_siglip = np.mean([siglip_score(f, prompt) for f in baseline_video])
        results['siglip_improvement'] = results['siglip_mean'] - baseline_siglip

    return results
```

This document serves as the definitive bridge analysis between Apollo findings and LTX-2 research, providing clear guidance on what transfers, what doesn't, and how to prioritize experiments accordingly.
