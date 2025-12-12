# Related Work: Analogous Techniques Across Domains

> **Last Updated:** 2025-12-11

This document analyzes techniques from various domains that share conceptual similarities with our VL hidden state conditioning approach.

## Summary Table

| Domain | Technique | Similarity | Key Insight | What Worked | What Didn't |
|--------|-----------|------------|-------------|-------------|-------------|
| Image Gen | IP-Adapter | HIGH | Vision -> cross-attention | Decoupled attention | Direct embedding injection |
| LLM | Activation Steering | HIGH | Direction vectors in hidden space | Addition/subtraction of directions | Layer choice critical |
| Embeddings | Platonic Rep. (Morris) | HIGH | Universal geometry across models | Zero-shot translation | Still requires some alignment |
| Embeddings | CLIP/ALIGN | MEDIUM | Multi-modal alignment | Contrastive training | Alignment is for retrieval, not generation |
| Style Transfer | Neural Style Transfer | MEDIUM | Separate style/content | Gram matrices | Only texture, no semantics |
| Style Transfer | AdaIN | MEDIUM | Statistical matching | Mean/std transfer | Low-level only |
| Diffusion | Prompt-to-Prompt | MEDIUM | Cross-attention manipulation | Attention map editing | Complex workflow |
| Retrieval | Dense Retrieval | LOW | Token averaging | Importance weighting | Uniform averaging loses signal |

---

## 1. IP-Adapter (Most Relevant)

**Paper**: [IP-Adapter: Text Compatible Image Prompt Adapter](https://arxiv.org/abs/2308.06721)

### Architecture
```
Reference Image -> CLIP/SigLIP Encoder -> [Trained Projection] -> Decoupled Cross-Attention
                                                                          |
Text Prompt -> Text Encoder -> Standard Cross-Attention                   |
                                    |                                      |
                                    v                                      v
                              [Diffusion Model with dual cross-attention]
```

### What They Learned
1. **Direct injection fails**: Early experiments tried adding image embeddings directly to text embeddings. This caused "content override" - exactly what we observe.
2. **Decoupled attention works**: Separate K/V projections for image features prevent vision from hijacking text conditioning.
3. **Projection layer is crucial**: A trained projection from CLIP space to diffusion-compatible space.
4. **Scale parameter works**: Like our alpha, they have a scale to control image influence.

### Implications for Our Approach
- We're essentially doing what their "early experiments" showed doesn't work well
- But we have an advantage: our embeddings are already in a more compatible space (same architecture)
- Consider: Could we add a small trainable projection layer?
- Consider: Could we inject VL features via attention K/V rather than embedding addition?

---

## 2. Activation Steering / Representation Engineering

**Papers**:
- [Steering Llama 2 with Contrastive Activation Addition](https://arxiv.org/abs/2312.06681)
- [Representation Engineering](https://arxiv.org/abs/2310.01405)

### Core Concept
Find direction vectors in activation space that encode specific concepts (honesty, emotion, style). Add/subtract these vectors to modify model behavior.

### What They Learned
1. **Directions exist**: LLMs encode concepts as directions in activation space
2. **Layer choice matters**: Different layers encode different aspects (early=syntax, middle=semantics, late=task)
3. **Addition is additive**: Multiple steering vectors can be combined
4. **Scaling is important**: Too much steering causes incoherence

### Implications for Our Approach
- Our VL embeddings may contain multiple "directions" (style, content, composition)
- Could we decompose VL embeddings into separate steering vectors?
- The layer-dependent behavior we observe (-2 vs -15 vs -25) aligns with their findings
- Consider: Could we extract a "style direction" by contrasting VL(image) - VL(neutral)?

---

## 3. Neural Style Transfer (Classic)

**Paper**: [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)

### Architecture
```
Content Image -> VGG -> Content Activations (deeper layers)
Style Image -> VGG -> Style Statistics (Gram matrices, multiple layers)
                           |
                           v
                  [Optimization to match both]
```

### What They Learned
1. **Content/style separation**: Activations encode content, correlations encode style
2. **Layer hierarchy**: Early layers = low-level style (textures), deep layers = high-level content
3. **Gram matrices**: Second-order statistics (correlations) capture style effectively

### Implications for Our Approach
- Could we compute Gram matrices of VL embeddings for "style-only" transfer?
- Our `style_only` blend mode (last 1/3 of dimensions) is a crude approximation - Gram matrices might be better
- Consider: Per-layer style statistics matching?

---

## 4. AdaIN (Adaptive Instance Normalization)

**Paper**: [Arbitrary Style Transfer in Real-time](https://arxiv.org/abs/1703.06868)

### Core Concept
Transfer style by matching feature statistics (mean, std) from style image to content image.

### Formula
```
AdaIN(x, y) = sigma(y) * ((x - mu(x)) / sigma(x)) + mu(y)
```

### What They Learned
1. **Statistics encode style**: Mean and std of activations carry style information
2. **Real-time possible**: No optimization needed, just forward pass
3. **Limitations**: Only transfers low-level style (colors, textures), not semantic patterns

### Implications for Our Approach
- Our `scale_embeddings()` is similar (matching std)
- We could also match mean, skewness, kurtosis
- This explains why scaling alone doesn't transfer high-level semantics

---

## 5. Cross-Attention Manipulation in Diffusion

**Papers**:
- [Prompt-to-Prompt](https://arxiv.org/abs/2208.01626)
- [Plug-and-Play](https://arxiv.org/abs/2211.12572)

### Core Concept
Modify attention maps or features during diffusion to control generation.

### What They Learned
1. **Attention maps = spatial control**: Where the model attends determines spatial layout
2. **Feature injection = semantic control**: Injecting features at specific timesteps/layers controls content
3. **Early timesteps matter more**: Structure is determined early, details late

### Implications for Our Approach
- We inject VL at the START (embedding level) - maybe we should inject at specific layers/timesteps?
- Consider: VL injection only at early diffusion steps (structure) vs late steps (style)?
- Consider: Attention-based injection rather than embedding addition?

---

## 6. Dense Retrieval and Embedding Pooling

**Papers**: Various on dense passage retrieval, sentence embeddings

### Core Concept
Represent documents/passages as single vectors for retrieval.

### What They Learned
1. **Importance weighting beats mean**: TF-IDF, attention weights, etc.
2. **CLS tokens work**: Using a special token's representation captures whole-sequence semantics
3. **Late interaction**: ColBERT-style late interaction preserves token-level info

### Implications for Our Approach
- Our current approach is essentially "mean pooling" of VL tokens into text length
- `attention_weighted` blend mode is on the right track
- Consider: Could we use VL's attention patterns to identify "important" image regions?

---

## 7. Platonic Representation Hypothesis

### The Original Paper (Huh et al., 2024)

**Paper**: [The Platonic Representation Hypothesis](https://arxiv.org/abs/2405.07987)

**Authors**: Minyoung Huh, Brian Cheung, Tongzhou Wang, Phillip Isola (MIT/Google)

### Core Argument
Neural network representations across different models and domains are increasingly **converging toward a unified statistical model of reality**. The authors invoke Plato's concept of ideal forms - the hypothetical shared representation space is a "platonic representation."

### Key Claims
1. **Representation Convergence**: Different neural networks represent data in increasingly similar ways across time and domains
2. **Cross-Modal Alignment**: As vision and language models scale up, they measure distances between datapoints in progressively more comparable ways
3. **Selective Pressures**: Multiple mechanisms drive networks toward this convergence (scale, diverse data, architectural advances)

### Why This Matters for Our Work
If all sufficiently capable models converge toward the same underlying representation:
- Qwen3-VL and Qwen3-4B should share similar semantic spaces (same family = even more similar)
- The OOD gap we observe may be fine-tuning drift, not fundamentally different spaces
- Zero-shot transfer should work better between related models than unrelated ones
- Training an adapter should be easier than bridging truly different architectures

---

### Related Work: Jack Morris's Embedding Research

**Papers**:
- [Text Embeddings Reveal (Almost) As Much As Text](https://arxiv.org/abs/2310.06816) (EMNLP 2023, Outstanding Paper)
- [vec2vec: Harnessing the Universal Geometry of Embeddings](https://vec2vec.github.io/)

Morris's work provides empirical evidence supporting the Platonic hypothesis:

### Key Findings
1. **Embedding inversion works**: Text can be recovered from embeddings with 92-94% accuracy - embeddings preserve nearly all information
2. **Universal geometry**: Embeddings from different models occupy similar geometric structures
3. **Zero-shot translation**: Model A's embeddings can be mapped to Model B's space without paired training data

### Implications for Our Approach
- The "Strong Platonic Representation Hypothesis" in vec2vec: all encoders learn nearly the same representations
- This explains why our zero-shot VL-to-text-embedding transfer works at all
- Future work: Could vec2vec-style translation outperform linear interpolation?

---

## 8. Spherical Interpolation (Slerp)

**Usage**: GAN latent space interpolation, embedding interpolation

### Formula
```
slerp(a, b, t) = sin((1-t)*omega)/sin(omega) * a + sin(t*omega)/sin(omega) * b
where omega = arccos(a . b / (|a| * |b|))
```

### Why It Matters
- Linear interpolation can "cut through" the embedding manifold
- Slerp follows the curved surface of the hypersphere
- May preserve meaningful structure better than linear blend

### Implications for Our Approach
- We currently use linear interpolation
- Slerp might produce more coherent blends
- Easy to implement and test

---

## Key Questions to Investigate

### From IP-Adapter
1. Would decoupled cross-attention (separate K/V for VL) work better?
2. Is a trained projection layer necessary for quality?

### From Activation Steering
3. Can we decompose VL into style/content direction vectors?
4. Would "VL(style_image) - VL(neutral_image) + text" work better?

### From Style Transfer
5. Would Gram matrix matching transfer style without content?
6. Can we identify which embedding dimensions encode style vs content?

### From Diffusion Manipulation
7. Should we inject VL at specific DiT layers rather than at embedding level?
8. Should VL influence vary by diffusion timestep?

### From Retrieval
9. Can we use attention patterns to weight VL token importance?
10. Is there a CLS-like "summary" token in VL embeddings we should prioritize?

---

## Recommended Experiments

### Quick Tests (No Training)
1. Slerp vs linear interpolation
2. Latent arithmetic: VL(style) - VL(neutral) + text
3. Per-dimension analysis: correlation of each dim with visual style

### Medium Effort
4. Gram matrix style matching
5. VL injection at specific DiT layers
6. Timestep-dependent VL strength

### Requires Training
7. Small projection layer (freeze DiT, train projection only)
8. LoRA adapter for VL integration

---

## See Also

- [techniques.md](techniques.md) - Deep dive into specific research questions
- [findings.md](findings.md) - Our experimental findings
