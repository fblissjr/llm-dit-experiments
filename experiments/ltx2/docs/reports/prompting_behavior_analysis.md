# LTX-2 Prompting Behavior Analysis: Bridging LLM Intuitions to DiT Conditioning

*last updated: 2026-01-17*

---

## Executive Summary

This report analyzes observations from hobbyist discussions about LTX-2 prompting behavior, evaluating claims against the actual architecture. The analysis bridges LLM intuitions (familiar to the target audience) with DiT-specific mechanisms (where those intuitions may or may not transfer).

**Key finding**: Many observations are accurate but misattributed. The "projector can't do token-level attending" claim is **partially incorrect** - the architecture does support per-token processing through the bidirectional connector, but the uniform layer projection creates a bottleneck that explains the observed behaviors.

---

## Part 1: Analysis of Questions Raised

### 1.1 What They Are Really Asking

| Observation | Underlying Question |
|-------------|---------------------|
| "Say what you see" / A->B->C | Why does temporal description ordering matter for a bidirectional system? |
| Laser blast camera failure | Why doesn't the model understand implied narrative continuity? |
| "Projector turns it into T5" | Does the projection layer destroy per-token information? |
| i2v static output issues | Why does describing the initial frame fix issues? |
| VAE 8-frame chunks | Do chunk boundaries create natural transition points? |

### 1.2 The Core Tension

The hobbyists are experiencing a contradiction:

1. **Expectation**: Bidirectional attention should let the model "see the whole prompt" and infer relationships
2. **Reality**: The model behaves like it processes sequentially, failing on implied transitions

This tension exists because **bidirectionality operates at the connector level, but the conditioning signal has already been compressed** through the uniform layer projection.

---

## Part 2: Answers to Their Questions

### 2.1 Does "Projector Can't Do Token-Level Attending" Hold?

**Verdict**: Partially correct, but the mechanism is misidentified.

#### The Actual Architecture

```
Prompt Text
    |
[Gemma-3 12B: 49 layers, output_hidden_states=True]
    |
hidden_states: tuple[49] of [B, T, 3840]
    |
[Stack to [B, T, 3840, 49]]
    |
[_pack_text_embeds: normalize + flatten to [B, T, 188160]]
    |
[text_proj_in: Linear(188160 -> 3840)]  <-- THE PROJECTOR
    |
[LTX2ConnectorTransformer1d]
    - 128 learnable "thinking tokens" prepended
    - 2 layers of FULL BIDIRECTIONAL attention
    - RoPE positional embeddings
    |
[Caption Projection: 3840 -> 4096]
    |
[DiT Cross-Attention in 48 blocks]
```

#### What Actually Happens

| Component | Token-Level? | Evidence |
|-----------|--------------|----------|
| Gemma-3 encoding | Yes | Each position has unique hidden states |
| Layer stacking | Yes | Shape is [B, T, 3840, 49] - T is preserved |
| `text_proj_in` | Yes | Linear operates on last dim, T preserved |
| Connector | Yes | 2 transformer layers with bidirectional attention |
| DiT cross-attention | Yes | Q from video, K/V from text per-position |

**The projector (text_proj_in) IS token-level.** It's a Linear layer applied to the layer dimension [188160 -> 3840], not across tokens.

#### What They're Actually Observing

The bottleneck is not "token-level attending" but **layer blending uniformity**:

```python
# Current: Uniform projection across all 49 layers
text_proj_in: Linear(188160, 3840)  # Learned but FIXED per-layer weights

# Effect: Every token gets the same layer mixture
# Early layers (syntax) mixed with late layers (semantics)
# No per-token specialization for "what kind of information this token needs"
```

From prior analysis, the projection matrix W has nearly uniform Frobenius norms across layer blocks (~2% variation). The differentiation comes from hidden state magnitudes:

| Layer Range | Contribution |
|-------------|--------------|
| 43-47 (late) | ~25% of signal |
| 0-4 (early) | <1% of signal |
| 48 (final) | 0.02% (anomalously low) |

**Conclusion**: Token-level information IS preserved, but layer-level information is uniformly mixed. The "T5-like" behavior comes from the bidirectional connector, which IS intentional and NOT a limitation.

### 2.2 Why Does A->B->C Structure Help Given Bidirectional Attention?

**Answer**: Temporal anchoring for cross-attention, not connector attention.

#### The Mechanism

1. **Bidirectional attention in connector**: All text tokens see each other
2. **Cross-attention in DiT**: Video patches query text tokens
3. **Video patches have temporal position**: Early patches = early frames
4. **Text-video alignment is learned**: Model learned that prompt structure correlates with video temporal structure

#### Why Sequential Description Helps

```
Prompt: "A walks. B happens. C results."
        |       |         |
        v       v         v
Video:  [Frame 0-30] [Frame 30-60] [Frame 60-90]
```

The model learned during training that:
- First sentence correlates with early frames
- Last sentence correlates with late frames
- Transitions correlate with temporal boundaries

This is NOT because the connector is causal - it's because **training data had this correlation**.

#### Transfer Assessment

| LLM Intuition | Applicability |
|---------------|---------------|
| "Position matters for meaning" | Transfers - RoPE encodes position |
| "Earlier = processed first" | Does NOT transfer - bidirectional |
| "Sequence = narrative" | Transfers - but via learned correlation |

### 2.3 Why Does the Compression Trick Work for i2v?

**Observation**: Upscale -> Video compress -> Downscale makes i2v work better.

**Answer**: Distribution matching to training data.

#### The VAE Expectation

LTX-2's VAE was trained on:
1. Video frames from curated datasets
2. Specific compression artifacts
3. Particular color/contrast distributions

Random internet images have:
1. Different compression (JPEG vs video codecs)
2. Different color grading
3. Different noise characteristics

#### Why the Trick Works

```
Original Image -> VAE Encode -> Latent
                                  |
                                  v
                    [Out-of-distribution latent]

Video-Compressed Image -> VAE Encode -> Latent
                                          |
                                          v
                          [In-distribution latent]
```

The video compression step applies:
- Temporal smoothing (even on single frame)
- Block-based artifacts similar to training data
- Color quantization matching training distribution

#### Transfer Assessment

| LLM Intuition | Applicability |
|---------------|---------------|
| "Tokenization affects output" | Transfers - VAE = visual tokenizer |
| "Distribution shift matters" | Transfers - directly applicable |
| "Preprocessing can fix OOD" | Transfers - common technique |

### 2.4 Why 8 Frames Per Latent Matters for Prompting

**Architecture**: LTX-2 VAE uses 8x temporal compression (8 frames -> 1 latent temporal position).

#### Implications

```
Latent Timeline:  [L0]    [L1]    [L2]    [L3]    ...
                   |       |       |       |
Frame Timeline:  [0-7]  [8-15] [16-23] [24-31]  ...
                   ^       ^       ^       ^
                 Chunk   Chunk   Chunk   Chunk
                   0       1       2       3
```

**Each latent position encodes 8 frames simultaneously.** This means:
- Within a chunk (8 frames): Smooth interpolation
- Across chunks: Discrete state changes possible

#### The Chunk Boundary Hypothesis

**Hypothesis from discussion**: "Chunk boundaries are transition points where state changes can happen cleanly."

**Assessment**: Plausible but needs testing.

Reasoning:
1. VAE decoder must interpolate within chunks
2. Adjacent chunks have weaker coupling
3. DiT can attend to different text for different chunks

**Counter-evidence**: The VAE uses overlapping temporal convolutions, so boundaries aren't truly discrete.

---

## Part 3: Questions They Should Be Asking

### 3.1 Missing Questions

| Unasked Question | Why It Matters |
|------------------|----------------|
| "Which Gemma layers encode what?" | Enables selective layer routing |
| "How do thinking tokens distribute attention?" | Understanding global context mechanism |
| "What does the projection W actually learn?" | Layer importance, concept-layer associations |
| "Does DiT cross-attention vary by block?" | Early blocks vs late blocks may use text differently |
| "How does CFG interact with layer conditioning?" | Negative prompt layer effects |

### 3.2 Testable Hypotheses

#### Hypothesis 1: Layer-Semantic Correspondence

**Claim**: Different Gemma layers encode different semantic aspects.

**Test**:
```python
# For each layer i in [0, 24, 47]:
#   Generate video using only layer i
#   Measure: motion accuracy, object identity, style adherence
```

**Prediction**:
- Early layers: Better text rendering (surface features)
- Middle layers: Better object identity
- Late layers: Better style/composition

#### Hypothesis 2: Thinking Token Function

**Claim**: 128 thinking tokens aggregate global context.

**Test**:
```python
# Hook into connector attention
# Measure: How much do thinking tokens attend to content tokens?
# Measure: How much do content tokens attend to thinking tokens?
```

**Prediction**: Thinking tokens will show high entropy attention (attending broadly).

#### Hypothesis 3: Temporal Prompt-Video Correlation

**Claim**: Prompt position correlates with video temporal position.

**Test**:
```python
# Prompt: "Red ball. Blue ball. Green ball."
# Measure: Frame-by-frame color dominance
# Expected: Red->Blue->Green temporal sequence
```

### 3.3 Priority Research Areas

| Area | Difficulty | Impact | Recommended? |
|------|------------|--------|--------------|
| Per-token layer routing | Medium | High | Yes - primary focus |
| Thinking token ablation | Low | Medium | Yes - quick win |
| DiT block text usage | High | High | Future work |
| VAE chunk boundary analysis | Medium | Medium | Optional |

---

## Part 4: Orthogonal Research Areas

### 4.1 LLM Techniques That Should Transfer

| Technique | DiT Analog | Confidence | Notes |
|-----------|------------|------------|-------|
| **Activation steering** | Embedding space manipulation | Medium | Add "detail direction" to text embeddings |
| **Layer probing** | Layer contribution analysis | High | Already done for LTX-2 |
| **Attention pattern analysis** | Cross-attention visualization | High | Standard interpretability |
| **Token importance** | Gradient-based saliency | Medium | Which text tokens matter for which video regions? |
| **KV cache reuse** | Cached text embeddings | High | LTX-2 already caches across denoising steps |

### 4.2 LLM Techniques That Likely Won't Transfer

| Technique | Why It Won't Transfer |
|-----------|----------------------|
| **Chain-of-thought reasoning** | DiT doesn't "think" during denoising |
| **Prompt position manipulation** | Bidirectional attention removes positional causality |
| **Token probability analysis** | DiT doesn't output token probabilities |
| **Beam search decoding** | Denoising is continuous, not discrete |

### 4.3 DiT-Specific Techniques to Explore

| Technique | Description | Relevance |
|-----------|-------------|-----------|
| **Timestep-conditional routing** | Different layer blends at different noise levels | High |
| **CFG on embeddings** | Apply guidance in text embedding space | High |
| **Cross-attention steering** | Modify K/V in DiT blocks | Medium |
| **Latent guidance** | Steer generation toward target images | Medium |

---

## Part 5: The Chunk Boundary Hypothesis

### 5.1 Technical Background

**VAE Configuration** (from LTX-2):
- Temporal compression: 8x (8 frames per latent)
- Spatial compression: 8x (8x8 pixel patches)
- Latent channels: 16

**At 24 fps**:
- 8 frames = 333ms per latent chunk
- 1 second = ~3 chunks
- Typical generation (97 frames) = ~12 chunks

### 5.2 Hypothesis Analysis

**Claim**: State changes happen "cleanly" at chunk boundaries.

**Evidence For**:
1. VAE encoder processes 8-frame blocks
2. DiT attention operates on latent positions
3. Cross-attention can weight text differently per latent position

**Evidence Against**:
1. VAE uses causal temporal convolutions with overlap
2. DiT has full attention across all latent positions
3. No explicit boundary mechanism in architecture

**Assessment**: Partially true. Boundaries are softer than claimed, but chunk-level planning may still be beneficial.

### 5.3 Position Embedding Considerations

**Question**: Would position embeddings go out-of-distribution near boundaries?

**Answer**: Unlikely. LTX-2 uses RoPE (Rotary Position Embeddings), which:
- Gracefully interpolate to unseen positions
- Don't have hard boundaries
- Are applied independently per position

**However**: Training data statistics may create implicit boundaries if training videos had scene cuts at predictable intervals.

### 5.4 Proposed Experiment

**Design**: Test whether chunk-aligned transitions improve quality.

```python
# Experiment: 16 frames at 2fps (8 seconds, 2 chunks)
# Condition A: Transition at frame 8 (chunk boundary)
# Condition B: Transition at frame 6 (within chunk)

prompt_A = "A red ball sits still [frames 0-7]. The ball begins rolling [frames 8-15]."
prompt_B = "A red ball sits still [frames 0-5]. The ball begins rolling [frames 6-15]."

# Measure: Transition smoothness, temporal coherence, prompt adherence
```

**Predictions**:
- Condition A: Sharper, cleaner transition
- Condition B: More gradual transition (VAE interpolation within chunk)

**Implementation Notes**:
- Use minimal frames to isolate effect
- Control for prompt length
- Multiple seeds for statistical significance

---

## Part 6: Bridge Analysis - LLM to DiT Transfer

### 6.1 Valid Intuitions

| LLM Intuition | Why It's Valid | How to Apply |
|---------------|----------------|--------------|
| **"Different layers encode different things"** | Universal transformer property | Use layer ablation/routing |
| **"Attention patterns reveal what matters"** | Attention is attention | Visualize cross-attention |
| **"Tokenization affects everything"** | VAE = visual tokenizer | Consider VAE compression |
| **"Context window has limits"** | DiT has max sequence length | Respect RoPE limits (1504 tokens) |
| **"Distribution shift breaks things"** | Training data distribution matters | Match prompt style to training |

### 6.2 Intuitions Requiring Modification

| LLM Intuition | Modification Needed | DiT Reality |
|---------------|---------------------|-------------|
| **"Later in sequence = later processed"** | Remove sequential assumption | Bidirectional attention |
| **"Add more tokens = more information"** | Quality over quantity | 128 thinking tokens compress |
| **"Reasoning helps"** | Not during generation | Reasoning affects embeddings only |
| **"Repeat important things"** | Repetition may not help | Attention will find it once |

### 6.3 Intuitions That Don't Apply

| LLM Intuition | Why It Fails |
|---------------|--------------|
| **"The model is reasoning"** | DiT is denoising, not reasoning |
| **"Token probabilities indicate confidence"** | No token outputs |
| **"Prompt engineering = persuasion"** | Model can't be "convinced" |
| **"Context in, context out"** | Output is pixels, not text |

### 6.4 Transfer Confidence Matrix

| Concept | Transfer Confidence | Notes |
|---------|---------------------|-------|
| Attention sinks | High | Thinking tokens serve similar role |
| Layer hierarchy | High | Directly applicable |
| Activation steering | Medium | Needs embedding space adaptation |
| KV caching | High | Already implemented |
| Causal masking effects | Does Not Apply | System is bidirectional |
| Token routing (MoE) | Medium | Novel for layer routing |
| Position interpolation | High | RoPE is shared |

---

## Part 7: Recommended Research Directions

### 7.1 Immediate (Zero Training)

1. **Layer ablation sweep**: Test each layer in isolation, measure SigLIP score
2. **Thinking token analysis**: Extract and visualize their learned embeddings
3. **Prompt structure ablation**: Test A->B->C vs B->A->C ordering

### 7.2 Lightweight Training ($10-100)

1. **Per-token layer router**: ~249K parameters, train to optimize SigLIP
2. **Embedding CFG**: Apply classifier-free guidance in text embedding space
3. **Attention steering**: Learn directions for "more detail", "more motion", etc.

### 7.3 Advanced (Future)

1. **Timestep-conditional routing**: Different layer blends at different noise levels
2. **Per-DiT-block text features**: Early blocks get early layers, late blocks get late layers
3. **Thinking token specialization**: Route different tokens through different subsets

---

## Conclusions

### Key Takeaways

1. **The "projector can't attend" claim is wrong** - per-token processing is preserved throughout the pipeline

2. **The real bottleneck is uniform layer blending** - all tokens get the same 49-layer mixture, regardless of their semantic role

3. **A->B->C structure helps due to learned correlations** - training data associated prompt position with video temporal position

4. **Chunk boundaries are soft, not hard** - VAE overlap smooths transitions, but chunk-level planning may still help

5. **Most LLM interpretability intuitions transfer** - attention analysis, layer probing, activation steering all apply

### Actionable Recommendations

| Priority | Action | Expected Impact |
|----------|--------|-----------------|
| High | Implement per-token layer routing | 5-15% quality improvement |
| High | Run layer ablation to validate routing targets | Foundation for routing |
| Medium | Test chunk-aligned prompting | May improve transitions |
| Medium | Analyze thinking token attention patterns | Understand global context |
| Low | Explore timestep-conditional routing | Future optimization |

---

## References

### Architecture Documentation
- `experiments/ltx2/docs/text_conditioning_architecture.md`
- `experiments/ltx2/docs/gemma3_sublayer_architecture.md`

### Prior Analysis
- `experiments/ltx2/docs/reports/sublayer_extraction_and_router_infrastructure_2026-01-16.md`
- `internal/log/log_2026-01-16.md`

### LTX-2 Prompting Guides
- `experiments/ltx2/prompting_guide.md`
- `experiments/ltx2/ltx2_official_prompting_guide.md`

### Research Context
- `experiments/ltx2/ltx2_research_guide.md`
- `experiments/ltx2/CLAUDE.md`
