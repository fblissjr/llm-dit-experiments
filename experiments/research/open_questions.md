# Open Questions

Unanswered questions requiring investigation, organized by category.

---

## Architecture Questions

### Q-A1: Why Penultimate Layer?

**Status**: READY TO INVESTIGATE - `--hidden-layer` parameter is now implemented (default: -2)

**Question**: Why does Z-Image extract `hidden_states[-2]` instead of other layers?

**Context**: This is a common pattern but rarely validated empirically. Different layers encode different information:
- Earlier layers: Syntactic, positional
- Middle layers: Semantic composition
- Later layers: Task-specific (next-token prediction)

**Investigation approach**:
1. Extract embeddings from layers -1 through -6 using `--hidden-layer`
2. Generate images with each layer
3. Compare quality metrics
4. Analyze embedding statistics (norm, variance, PCA)

**Hypothesis**: Layer -2 may not be optimal for all prompt types.

**How to test**:
```bash
# Test different layers
for layer in -1 -2 -3 -4 -5 -6; do
  uv run scripts/generate.py \
    --model-path /path/to/z-image \
    --hidden-layer $layer \
    --seed 42 \
    --output "output_layer${layer}.png" \
    "A detailed portrait of a woman"
done
```

---

### Q-A2: What Does the Context Refiner Actually Do?

**Question**: What transformation does the 2-layer context refiner perform?

**Context**: The context refiner processes projected embeddings (3840 dim) through 2 transformer layers WITHOUT timestep modulation.

**Investigation approach**:
1. Compare embeddings before and after context refiner
2. Measure cosine similarity, norm change, variance change
3. Visualize the transformation via PCA/t-SNE
4. Ablate: skip refiner entirely

**Hypothesis**: The refiner may be performing semantic refinement, attention redistribution, or noise reduction.

---

### Q-A3: Why No Timestep Modulation in Context Refiner?

**Question**: Why was timestep information excluded from the context refiner?

**Context**: Main DiT blocks use AdaLN (Adaptive Layer Norm) with timestep injection. Context refiner explicitly omits this.

**Possible explanations**:
1. Text conditioning should be stable across timesteps
2. Training simplification
3. Decoupled-DMD requirement
4. Design oversight

**Investigation approach**:
1. Research original Z-Image paper for justification
2. Test if adding timestep hurts or helps (requires fine-tuning)
3. Compare with other LLM-DiT architectures

---

### Q-A4: Why RoPE Theta=256 in Context Refiner?

**Question**: Why does the context refiner use `theta=256` while Qwen3 uses `theta=1000000`?

**Context**: RoPE theta controls the frequency of position encodings. Different theta values affect how the model perceives position.

**Investigation approach**:
1. Test different theta values in context refiner
2. Measure if position information is preserved/lost
3. Check if position matters at all in refined embeddings

**Hypothesis**: Lower theta may provide finer position discrimination for short sequences.

---

## Training Questions (Reverse Engineering)

### Q-T1: What Prompt Format Was Used During Training?

**Question**: Did training use think blocks? System prompts? What format?

**Context**: The official HF Space uses no think block by default. But the model may have been trained with thinking content.

**Investigation approach**:
1. Test quality with various formats (see ablation studies)
2. Look for training artifacts (does think block improve quality?)
3. Check if Alibaba/Tongyi published training details

**Impact**: Knowing the training format would inform optimal inference format.

---

### Q-T2: How Much CFG Was Baked In?

**Question**: What CFG scale was the Decoupled-DMD trained to replace?

**Context**: Z-Image claims CFG=0.0 because guidance is baked in. But we don't know the equivalent CFG scale.

**Investigation approach**:
1. Compare Z-Image outputs with other models at various CFG
2. Try small runtime CFG (0.5-2.0) to see if it helps or hurts
3. Research Decoupled-DMD paper for training CFG values

**Hypothesis**: The baked-in CFG may be equivalent to CFG=3-7 range.

---

### Q-T3: What Was the Training Resolution Range?

**Question**: What image resolutions was Z-Image trained on?

**Context**: Model supports various aspect ratios, but quality may vary.

**Investigation approach**:
1. Generate at various resolutions and measure quality
2. Look for resolution-specific artifacts
3. Test extreme aspect ratios

**Known**: DiT supports up to 512x512 latent (4096x4096 pixel) based on RoPE axes.

---

### Q-T4: What Dataset Was Used?

**Question**: What images and captions were used for training?

**Context**: Unknown. Affects what the model "knows" and style biases.

**Investigation approach**:
1. Probe for dataset-specific knowledge (famous artworks, photographers)
2. Test style coverage (anime, photorealism, abstract, etc.)
3. Check for biases (geography, culture, time period)

---

## Practical Questions

### Q-P1: What Is the Optimal Batch Size?

**Question**: What batch size maximizes throughput on RTX 4090?

**Context**: Current implementation focuses on single-image generation.

**Investigation approach**:
1. Benchmark batch sizes 1, 2, 4, 8
2. Measure images/second and VRAM usage
3. Check if quality degrades with batching

**Constraint**: 24GB VRAM on RTX 4090.

---

### Q-P2: Where Are the Bottlenecks?

**Question**: What is the time breakdown for text encoding, DiT inference, and VAE decoding?

**Context**: Profiler exists but detailed breakdown may not be complete.

**Investigation approach**:
```python
# Profile each component
with torch.cuda.amp.autocast():
    t0 = time.time()
    embeddings = encoder.encode(prompt)
    t1 = time.time()
    latents = dit.generate(embeddings, steps=9)
    t2 = time.time()
    image = vae.decode(latents)
    t3 = time.time()

print(f"Encode: {t1-t0:.2f}s, DiT: {t2-t1:.2f}s, VAE: {t3-t2:.2f}s")
```

**Expected**: DiT dominates, but verification needed.

---

### Q-P3: Does torch.compile Help?

**Question**: Does compiling the model with `torch.compile` improve inference speed?

**Context**: PyTorch 2.0+ supports JIT compilation for speedup.

**Investigation approach**:
1. Benchmark with and without `torch.compile`
2. Test different compile modes (default, reduce-overhead, max-autotune)
3. Measure warmup time vs sustained throughput

**Caveat**: Compile may not help for short runs due to compilation overhead.

---

### Q-P4: What Quantization Is Safe?

**Question**: How much quality is lost with 8-bit or 4-bit quantization?

**Context**: Quantization saves memory but may reduce precision.

**Investigation approach**:
1. Compare full precision vs 8-bit vs 4-bit embeddings
2. Measure cosine similarity of embeddings
3. Compare generated image quality
4. Measure VRAM savings

**Components to test**: Text encoder (Qwen3-4B), DiT, VAE.

---

## Prompt Engineering Questions

### Q-E1: Does Token Order Matter?

**Question**: Does the position of descriptors in a prompt affect the image?

**Context**: Some models show position bias (early tokens = more influence).

**Investigation approach**:
1. Test "red cat on blue couch" vs "blue couch with red cat"
2. Shuffle token order and measure output variance
3. Test front-loading vs back-loading key descriptors

---

### Q-E2: What Makes a "Good" Prompt?

**Question**: What prompt characteristics correlate with high-quality outputs?

**Potential factors**:
- Token count
- Specificity
- Structure (lists, descriptions)
- Style keywords
- Think block content

**Investigation approach**:
1. Collect 100+ prompts with quality ratings
2. Extract features (length, word types, structure)
3. Correlate features with quality metrics
4. Build a "prompt quality predictor"

---

### Q-E3: Why Do Some Prompts Fail?

**Question**: What causes generation failures (artifacts, wrong content, low quality)?

**Investigation approach**:
1. Collect failing prompts
2. Analyze patterns (length, complexity, ambiguity)
3. Compare embeddings of failing vs successful prompts
4. Test if modifications (rewording, simplifying) fix failures

**Hypothesis**: Failures may correlate with embedding anomalies (low norm, high variance).

---

### Q-E4: Do Templates Actually Help?

**Question**: Do the 140+ templates improve quality or just add tokens?

**Investigation approach**:
1. Compare quality: bare prompt vs templated prompt
2. Measure token efficiency (quality per token)
3. Identify which template categories help most
4. Test if template benefits are prompt-dependent

---

## Embedding Space Questions

### Q-S1: What Do Embeddings Represent?

**Question**: What information is encoded in the 2560-dimensional embeddings?

**Investigation approach**:
1. Cluster embeddings by prompt type
2. Find interpretable directions (style, subject, quality)
3. Test if linear probes can predict prompt properties
4. Visualize embedding space with t-SNE/UMAP

---

### Q-S2: Are There "Style Vectors"?

**Question**: Can we identify directions in embedding space that correspond to styles?

**Investigation approach**:
1. Compute mean embeddings for style categories
2. Compute difference vectors: `style_a - style_b`
3. Test if adding vectors transfers style
4. Check if vectors generalize across prompts

---

### Q-S3: Do Embeddings Have Pathologies?

**Question**: Are there embedding patterns that predict generation failure?

**Potential pathologies**:
- Very low norm (empty/meaningless)
- Very high variance (conflicting concepts)
- Outlier tokens (tokenization issues)
- Collapsed dimensions (information loss)

**Investigation approach**:
1. Collect embedding statistics from many prompts
2. Correlate statistics with generation quality
3. Build anomaly detector for problematic embeddings

---

### Q-S4: How Does the Embedding Cache Affect Results?

**Question**: Does caching embeddings have any side effects?

**Context**: Embedding cache stores encoded prompts for reuse.

**Investigation approach**:
1. Generate same prompt with and without cache
2. Verify embeddings are identical
3. Check for any numerical precision issues
4. Test cache across different dtypes (fp32, bf16, fp16)

---

## Comparison Questions

### Q-C1: How Does Z-Image Compare to FLUX/SD3?

**Question**: What are Z-Image's strengths and weaknesses vs other models?

**Dimensions to compare**:
- Text alignment (CLIP score)
- Image quality (FID)
- Speed (images/second)
- Style coverage
- Prompt following accuracy
- Anatomical correctness

---

### Q-C2: Does Z-Image Have Unique Capabilities?

**Question**: What can Z-Image do that other models cannot?

**Potential unique features**:
- LLM-based understanding (reasoning about prompts?)
- Think block for explicit planning
- Turbo speed (8-9 steps)
- Text rendering (LLM text knowledge?)

**Investigation approach**:
1. Test complex reasoning prompts
2. Test text rendering prompts
3. Compare with FLUX/SD3 on same prompts

---

## Priority Investigation Order

### Tier 1: Quick Wins (< 1 day each)

1. **Q-P2**: Where are the bottlenecks? (Profiling)
2. **Q-E1**: Does token order matter? (Simple test)
3. **Q-A1**: Why penultimate layer? (Layer comparison)
4. **Q-E4**: Do templates actually help? (A/B test)

### Tier 2: Medium Effort (1-3 days each)

5. **Q-A2**: What does context refiner do? (Analysis)
6. **Q-S1**: What do embeddings represent? (Visualization)
7. **Q-E3**: Why do some prompts fail? (Failure analysis)
8. **Q-P4**: What quantization is safe? (Quality comparison)

### Tier 3: Deep Dives (1+ weeks each)

9. **Q-S2**: Are there style vectors? (Steering research)
10. **Q-T1**: What prompt format was used during training? (Reverse engineering)
11. **Q-A3**: Why no timestep in refiner? (Architecture research)
12. **Q-C1**: How does Z-Image compare? (Benchmark suite)

---

## Question Template

When investigating a question, document:

```markdown
## Q-XX: [Question Title]

### Question
[Clear statement of the question]

### Hypothesis
[What you expect to find]

### Method
[How you'll investigate]

### Results
[What you found]

### Conclusions
[What this means]

### Follow-up Questions
[New questions raised by this investigation]
```
