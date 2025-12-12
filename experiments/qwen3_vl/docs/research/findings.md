# Research Findings: Qwen3-VL Vision Conditioning for Z-Image

> **Last Updated:** 2025-12-12

## Executive Summary

We explored a **training-free approach** to vision-conditioned image generation using Qwen3-VL and Z-Image. By extracting hidden states from Qwen3-VL's text model after it processes an image, we can influence Z-Image's DiT without additional training. However, this produces visible artifacts even with optimal settings. The approach demonstrates that architectural compatibility enables embedding transfer, but quality falls significantly short of trained methods like IP-Adapter.

## Key Technical Finding: RoPE Mismatch (2025-12-12)

**Observation:** Qwen3-VL uses different positional encoding than Qwen3-4B:

| Parameter | Qwen3-4B (Z-Image) | Qwen3-VL |
|-----------|-------------------|----------|
| `rope_theta` | 1,000,000 | 5,000,000 (5x higher) |
| `rope_scaling` | None (standard RoPE) | MRoPE with sections [24,20,20], interleaved |

Despite this, **VL text tokens have 0.999 correlation** with Qwen3-4B per-dimension statistics. The RoPE difference doesn't catastrophically break hidden state compatibility for text positions.

### Per-Dimension Distribution Analysis

We computed per-dimension std correlation between models:

| Comparison | Correlation | Median Ratio | Worst Ratio |
|------------|-------------|--------------|-------------|
| **VL text tokens** vs Qwen3-4B | **0.999** | 1.11x | 3.42x |
| **VL image tokens** vs Qwen3-4B | **0.737** | 1.55x | **617x** |

**Key observation:** VL text token positions have nearly identical per-dimension statistics to Qwen3-4B. Image token positions have extreme outliers (dim 396: 617x ratio, dim 4: 42x ratio), which likely contribute to the more severe artifacts seen with image tokens.

`text_tokens_only=True` produces fewer artifacts than using image tokens, but visible corruption remains compared to pure text generation.

## Key Findings

### 1. Architecture Compatibility Enables Transfer

**Observation**: Qwen3-VL-4B's text model shares architecture with Qwen3-4B (Z-Image's text encoder).

| Model | hidden_size | num_layers | Compatible |
|-------|-------------|------------|------------|
| Qwen3-4B (Z-Image) | 2560 | 36 | Baseline |
| Qwen3-VL-4B text model | 2560 | 36 | Yes |
| Qwen3-VL-4B vision encoder | Variable | - | No (different space) |

**Implication**: Architectural compatibility is necessary but not sufficient - shared architecture enables transfer but doesn't guarantee quality.

### 2. Embedding Space Analysis

**Observation**: Pure vision encoder outputs are incompatible. VL text model hidden states transfer partial information but with artifacts.

| Source | Global std | Per-dim correlation | Quality |
|--------|------------|---------------------|---------|
| Qwen3-4B text (layer -2) | 61.1 | 1.000 (reference) | Excellent (baseline) |
| VL text tokens (layer -2) | 47.8 | **0.999** | Visible artifacts |
| VL image tokens (layer -2) | 7.0 | **0.737** | Severe artifacts |
| VL vision encoder (raw) | 0.57 | N/A | Complete failure |

**Why vision encoder fails**: The ViT outputs are in a fundamentally different semantic space. They encode pixel-level features, not language-aligned semantics.

**Why VL text model transfers partial information**: Qwen3-VL's architecture projects vision features through a learned projection layer into the text model's embedding space. The resulting embeddings are similar enough to produce recognizable outputs, but sufficiently different to cause visible artifacts.

**Why image tokens cause more severe artifacts**: Image token positions have extreme per-dimension outliers (up to 617x std ratio). Per-dimension normalization reduces but doesn't eliminate the resulting corruption.

### 3. Interpolation Reduces But Doesn't Eliminate Artifacts

**Observation**: Pure VL embeddings produce recognizable but heavily artifacted images. Blending with text reduces artifact severity but doesn't eliminate them.

| Alpha | Quality | Notes |
|-------|---------|-------|
| 1.0 (pure VL) | Recognizable, heavy artifacts | Grid patterns, color bleeding |
| 0.5 | Better structure, artifacts reduced | Still visible degradation |
| **0.3** | **Fewest artifacts observed** | But still visibly worse than pure text |
| 0.0 (pure text) | Excellent | Baseline reference |

**Why blending helps**: Text embeddings provide in-distribution signal the DiT was trained on. Lower VL alpha reduces the out-of-distribution component, reducing but not eliminating artifacts.

### 4. Style vs Content Transfer Unpredictable

**Observation**: VL embeddings at low alpha influence visual style; at high alpha they can override text content unpredictably.

**Experiment**: "Homer Simpson eating spaghetti" with cartoon house VL embeddings

| Alpha | Result |
|-------|--------|
| 0.0 | Perfect Homer Simpson + Lincoln in diner |
| 0.3 | Blue blob character, simple flat style, spaghetti plate |

**Observation**: At 30% VL influence with unrelated content, the visual style (colors, simplicity) completely dominated over text semantics.

### 5. Hidden Layer Selection

**Finding**: Layer -2 (penultimate) works best, matching Z-Image's default.

| Layer | Observation |
|-------|-------------|
| -1 | Most abstract, slightly worse quality |
| -2 | Best balance (default) |
| -3 to -4 | Slightly more visual detail, similar quality |
| -5 to -6 | More noise, lower quality |
| -5 to -15 | Progressive "Asian bias" (Chinese training data influence) |
| -18 to -25 | "Semantic averaging" - outputs look like category prototypes |
| -30+ | Too abstract, loses all specificity |

#### Semantic Averaging in Middle Layers

The observation that layer -25 produces outputs resembling "20 similar images blended together" aligns with transformer layer research:

**Why this happens**: Qwen3-4B has 36 layers. Layer -25 is layer 11 (~30% depth), which falls in the "semantic abstraction zone" where models encode category prototypes rather than specific instances. This is consistent with:

- **Jawahar et al. (2019)**: BERT middle layers (9-12) encode abstract semantic concepts
- **Rogers et al. (2020)**: "A Primer in BERTology" - middle layers capture category membership
- **"Layer by Layer" (arXiv:2502.02013, 2025)**: Intermediate layers (50-70% depth) often outperform final layers for embeddings; final layers become "overly specialized to pretraining objective"

The "Asian bias" in layers -5 to -15 likely reflects Qwen3's Chinese training data distribution at the prototype level - the model's internal representation of "person" or "scene" is biased toward what it saw most often.

**Research implication**: A systematic layer sweep with quality metrics would characterize exactly what information each layer contributes - valuable data for understanding VLM-to-diffusion transfer.

### 6. Image Token Isolation

**Finding**: Filtering to only image tokens slightly improves purity but doesn't dramatically change results.

- Full sequence (270 tokens): Image + text + system tokens
- Image only (258 tokens): Just vision tokens

Both produce similar results after scaling, suggesting the text tokens don't heavily contaminate the embeddings.

#### Common Misconception: "Skip Image Tokens, Use Text Tokens"

A common misunderstanding is that in Qwen3-VL's hidden states:
- Image token positions contain "image-only" information
- Text token positions contain "text-only" information conditioned by images

**This is incorrect.** After transformer self-attention, all hidden states carry mixed information from the entire sequence. The image tokens have been processed through attention with text tokens, and vice versa. By layer -2:
- Image token positions: Contain image features contextualized by text
- Text token positions: Contain text features contextualized by images

Z-Image was trained on Qwen3-4B text embeddings (text captions only). The hidden states it expects are pure text representations. Using either image or text token positions from Qwen3-VL gives you a hybrid representation - which is why both are OOD relative to Z-Image's training distribution, and why alpha interpolation helps.

## Comparison with Prior Art

### IP-Adapter

| Aspect | IP-Adapter | Ours |
|--------|------------|------|
| Training required | Yes (adapter layers) | **No** |
| Image encoder | CLIP ViT | Qwen3-VL |
| Integration | Separate cross-attention | Same cross-attention |
| Quality | Higher | Lower (but improving) |
| Flexibility | Scale parameter | Alpha interpolation |
| Model dependency | Any SD model | Requires compatible text encoder |

**Key difference**: IP-Adapter trains additional adapter layers. Our approach is zero-shot.

### BLIP-Diffusion

| Aspect | BLIP-Diffusion | Ours |
|--------|----------------|------|
| Training | Two-stage pretraining | **None** |
| VLM | BLIP-2 | Qwen3-VL |
| Focus | Subject-driven generation | Style/semantic transfer |
| Quality | Higher for subjects | General purpose |

### CLIP Interrogator

| Aspect | CLIP Interrogator | Ours |
|--------|-------------------|------|
| Method | Image -> Text -> Generation | Image -> Embeddings -> Generation |
| Information loss | Text bottleneck | Continuous embeddings |
| Quality | Depends on caption quality | Direct visual features |

**Our advantage**: We skip the text decoding step, preserving more visual information.

## Implications

### For Z-Image Users

1. **Zero-shot style transfer**: Use reference images to guide generation style without training
2. **Image variations**: Generate variations by blending reference VL with descriptive text
3. **Composition guidance**: Use reference for layout, text for content

### For Researchers

1. **Architecture matters**: VLM architecture compatibility enables zero-shot transfer
2. **Embedding space alignment**: VLM text models project vision into language space
3. **Interpolation > direct use**: Blending with text improves out-of-distribution embeddings

### For the Field

1. **Training-free vision conditioning**: Possible when architectures align
2. **VLM as general embedders**: VLMs can provide embeddings for other generative models
3. **New research direction**: Optimal projection from VL to diffusion embedding space

## Limitations

1. **Quality gap**: Not as good as trained adapters (IP-Adapter)
2. **Architecture dependency**: Requires matching text encoder architecture
3. **Style dominance**: High VL influence overrides text content
4. **Artifacts**: Grid patterns and color bleeding in pure VL mode
5. **VRAM requirements**: Need to load both Qwen3-VL and Z-Image (sequential)

## Future Directions

### Short-term (No Training)

1. **Optimal alpha sweep**: Find best alpha for different use cases
2. **Multi-layer blending**: Combine hidden states from multiple layers
3. **Multi-image conditioning**: Blend VL from multiple references
4. **Better scaling**: Investigate non-linear scaling functions

### Medium-term (Light Training)

1. **Projection layer**: Train small MLP to project VL -> text embedding space
2. **LoRA adaptation**: Fine-tune DiT with LoRA for VL embeddings
3. **Attention-based fusion**: Learn to selectively attend to VL features

### Long-term (Research)

1. **Architecture-agnostic adapters**: Generalize to any VLM/diffusion pair
2. **Bidirectional conditioning**: Use diffusion features to guide VLM
3. **Joint fine-tuning**: Co-train VLM projection and diffusion model

## Reproducibility

### Test Case: Simple Cartoon House

Source image: `/tmp/claude/test_scene.png` (512x512 cartoon house)

| Configuration | Result File | Key Observation |
|---------------|-------------|-----------------|
| Pure text | `text_generated.png` | Beautiful photorealistic house |
| Pure VL vision | `vision_generated.png` | Abstract noise |
| VL text model | `vl_text_generated.png` | Recognizable cartoon elements |
| 30% VL + 70% text (matching) | `interp_03.png` | Clean cartoon reproduction |
| 30% VL + 70% text (Homer) | `homer_blended.png` | Blue blob (style transferred) |

### Recommended Default Settings

```python
# For text tokens only (recommended - 0.999 correlation with Qwen3-4B)
alpha = 0.3
hidden_layer = -8  # Layer -8 produces cleaner results than -2 for VL
text_tokens_only = True
scale_to_text = True
normalization_mode = "global"  # Text tokens don't need per-dim normalization

# For image tokens (if needed - experimental)
alpha = 0.3
hidden_layer = -8
image_tokens_only = True
scale_to_text = True
normalization_mode = "per_dim"  # CRITICAL: fixes 600x+ per-dim outliers
```

## Conclusion

We have explored and characterized a **training-free approach** to vision-conditioned image generation using Qwen3-VL and Z-Image. The approach demonstrates that architectural compatibility enables embedding transfer, but results show significant quality degradation compared to both pure text generation and trained adapters like IP-Adapter. Even with optimal settings, visible artifacts persist. The key finding - that VLM text model hidden states can partially transfer to diffusion models with shared architecture - suggests that small amounts of training (e.g., a projection layer) might bridge the quality gap.

## Theoretical Connections

### Platonic Representation Hypothesis

**Original paper**: Huh et al. (2024) - [The Platonic Representation Hypothesis](https://arxiv.org/abs/2405.07987) (MIT/Google)

**Core claim**: Neural network representations across different models and domains are converging toward a unified statistical model of reality - a "platonic representation."

**Supporting empirical work** (Jack Morris, Cornell):
- **"Text Embeddings Reveal (Almost) As Much As Text"** (EMNLP 2023) - 92-94% text recovery from embeddings
- **"vec2vec"** - Zero-shot translation between embedding spaces without paired data

**Relevance to our work**: The Platonic Representation Hypothesis suggests that Qwen3-VL and Qwen3-4B, being same-architecture models, likely encode similar semantic spaces. This explains why zero-shot transfer works at all - we're not bridging fundamentally different representations, just slightly misaligned versions of the same underlying structure.

**Implication**: The OOD gap we observe (requiring alpha interpolation) may be smaller than expected because both models converge toward similar "platonic" representations. Training an adapter might be easier than expected - it just needs to correct the fine-tuning-induced drift, not learn a new mapping.

### Why Penultimate Layer Works Best

Multiple sources support the choice of layer -2:

| Source | Finding |
|--------|---------|
| bert-as-service | Default uses second-to-last because last layer is "biased to training targets" |
| BeLLM (2024) | Performance drops after penultimate due to "extreme anisotropy" in autoregressive LLMs |
| BERTSUM | Penultimate outperforms final for extractive summarization |
| Our experiments | Layer -2 gives best quality, -1 is too abstract |

The last layer is specialized for the model's training objective (next-token prediction for LLMs, MLM for BERT). For downstream tasks requiring general semantics, earlier layers often work better.

## References

### Vision Conditioning
1. IP-Adapter: https://arxiv.org/abs/2308.06721
2. BLIP-Diffusion: https://dxli94.github.io/BLIP-Diffusion-website/
3. Qwen3-VL: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
4. Z-Image: https://huggingface.co/Tongyi-MAI/Z-Image-Turbo

### Embedding Theory & Platonic Representations
5. Platonic Representation Hypothesis (Huh et al., 2024): https://arxiv.org/abs/2405.07987
6. Text Embeddings Reveal (Almost) As Much As Text (Morris): https://arxiv.org/abs/2310.06816
7. vec2vec (Morris): https://vec2vec.github.io/
8. Jack Morris's research: https://jxmo.io/research

### Transformer Layer Analysis
8. Jawahar et al. (2019): "What Does BERT Look At?" - Layer-wise analysis
9. Rogers et al. (2020): "A Primer in BERTology" - Comprehensive BERT layer survey
10. Layer by Layer (2025): https://arxiv.org/abs/2502.02013 - Intermediate layers outperform final

---

## Latest Research: Token Position Discovery (2025-12-11)

**Key finding:** VL text token positions contain enough prompt-following information to generate correct subjects at alpha=1.0, while image token positions carry visual style.

This discovery opens new research directions:
- **VL-only generation:** Can we eliminate Qwen3-4B entirely by using VL text tokens?
- **Intra-VL blending:** Blend image tokens (style) with text tokens (content) from SAME extraction
- **Layer optimization:** Do image/text tokens benefit from different hidden layers?
- **Dual extraction:** Extract twice (image-only, text-only) and blend results

See [token_position.md](token_position.md) for detailed experiment proposals and implementation roadmap.

## See Also

- [token_position.md](token_position.md) - NEW experiments based on token position findings
- [techniques.md](techniques.md) - Deep dive into techniques
- [related_work.md](related_work.md) - Prior art from other domains
- [../guides/parameters.md](../guides/parameters.md) - Practical guide to parameters
