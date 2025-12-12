# Token Position Experiments: VL Intra-Blending Research

> **Created:** 2025-12-11
> **Status:** Experimental - Based on new findings about VL token positions

## Key Discovery

When extracting embeddings from Qwen3-VL (which processes both an image AND text prompt), we discovered that:

- **TEXT TOKEN POSITIONS** contain enough prompt-following information to generate the correct subject (e.g., "Homer Simpson eating spaghetti") even at alpha=1.0 (pure VL, no Qwen3-4B blending)
- **IMAGE TOKEN POSITIONS** reproduce the reference image's visual style/content

This challenges the original assumption that we need to blend VL with Qwen3-4B text embeddings. It suggests we might achieve better results by intelligently combining different token positions from the SAME VL extraction.

## Research Questions

### Primary Questions

1. **Can we eliminate Qwen3-4B entirely?**
   - If VL text tokens already follow prompts at alpha=1.0, why blend with Qwen3-4B?
   - Would pure VL (with prompt in text tokens) match or exceed VL+Qwen3-4B blends?

2. **Can we blend WITHIN VL for best of both worlds?**
   - Image tokens: Visual style from reference
   - Text tokens: Prompt adherence from VL's text processing
   - Hypothesis: 30% image tokens + 70% text tokens might give style transfer + prompt following

3. **Do different hidden layers work better for different token types?**
   - Image tokens might benefit from deeper layers (more visual abstraction)
   - Text tokens might benefit from middle layers (better semantic understanding)
   - Current default (-2) might not be optimal for both

4. **What VL text input produces the best prompt adherence?**
   - Exact prompt text (current approach)
   - Generic description prompt
   - Instruction to generate
   - No text (pure visual conditioning)

### Secondary Questions

5. **Can we use DUAL VL extractions?**
   - Extract once with image tokens only, text=None → pure style
   - Extract again with text tokens only, text=prompt → pure prompt
   - Blend the two extractions at various ratios

6. **Are there optimal layer combinations?**
   - Extract image tokens from layer -5 (more visual detail)
   - Extract text tokens from layer -15 (better semantics)
   - Blend the two layer extractions

## New Experiment Types

### 1. `vl_intra_token_blend`
**Blend image tokens with text tokens from SAME VL extraction**

Configurations:
- Baseline: Full sequence (alpha=1.0, text=prompt)
- Baseline: Image tokens only (alpha=1.0, text=prompt)
- Baseline: Text tokens only (alpha=1.0, text=prompt)
- TODO: Intra-blend ratios (requires new infrastructure)

**Expected outcome:** Text-only tokens should show best prompt adherence, image-only should show best style transfer, full sequence is compromise.

**Implementation need:** Requires extracting image and text token subsets separately, then blending them before passing to Z-Image.

### 2. `vl_only_vs_qwen3`
**Direct comparison: Can VL replace Qwen3-4B?**

Configurations:
- Pure Qwen3-4B (alpha=0.0) - baseline text
- Pure VL text tokens (alpha=1.0, text=prompt, text_tokens_only=True) - our finding
- Pure VL full sequence (alpha=1.0, text=prompt) - includes both image+text
- Traditional blend (alpha=0.3) - current best practice

**Expected outcome:** If VL text tokens match Qwen3-4B quality, we can simplify the pipeline.

**Impact:** Could eliminate need for Qwen3-4B entirely, reducing VRAM and complexity.

### 3. `vl_layer_by_token`
**Test different hidden layers for different token types**

Configurations:
- Image tokens at layers: -1, -2, -5, -10, -15
- Text tokens at layers: -1, -2, -5, -10, -15
- All at alpha=1.0 with text=prompt

**Expected outcome:**
- Image tokens: Deeper layers (-5 to -10) may preserve more visual detail
- Text tokens: Middle layers (-10 to -15) may have better semantic understanding
- Currently we use -2 for both, which may not be optimal

**Research value:** Would inform optimal layer selection for each use case (style vs content).

### 4. `vl_double_conditioning`
**Use TWO separate VL extractions, blend them**

Configurations:
- Single extraction: Image tokens only, text=None (pure style)
- Single extraction: Text tokens only, text=prompt (pure prompt)
- TODO: Dual blend at various ratios

**Expected outcome:** Cleanest separation of style and content, potentially better quality than single extraction.

**Implementation need:** Requires infrastructure to extract twice and blend the results.

### 5. `vl_prompt_variations`
**Test different VL text inputs for prompt adherence**

Configurations:
- No text in VL, image tokens only
- No text in VL, text tokens only (tests if text tokens work without VL text input)
- Prompt in VL, image tokens
- Prompt in VL, text tokens (our finding - best prompt adherence)
- Generic description in VL, text tokens
- Instruction in VL, text tokens

**Expected outcome:** Different VL text inputs affect how well text tokens follow prompts.

**Research value:** Optimizes the VL input for best prompt adherence.

## Implementation Roadmap

### Phase 1: Experiments with Existing Infrastructure (DONE)
- [x] `vl_only_vs_qwen3` - Uses existing alpha and token selection
- [x] `vl_layer_by_token` - Uses existing hidden_layer parameter
- [x] `vl_prompt_variations` - Uses existing text parameter
- [x] `vl_intra_token_blend` (baselines only) - Single extractions for comparison

### Phase 2: Infrastructure for Intra-VL Blending
Requires new code to:
1. Extract image token subset from VL result
2. Extract text token subset from VL result
3. Blend them at specified ratio BEFORE Z-Image

**New parameter needed:**
```python
vl_intra_blend_ratio: float  # 0.0 = all text tokens, 1.0 = all image tokens
```

**Implementation approach:**
```python
# In VLEmbeddingExtractor.extract():
if vl_intra_blend_ratio is not None:
    # Extract both subsets
    image_emb = extract_image_tokens(...)
    text_emb = extract_text_tokens(...)
    # Blend them
    blended = vl_intra_blend_ratio * image_emb + (1 - vl_intra_blend_ratio) * text_emb
    return blended
```

### Phase 3: Infrastructure for Dual Extraction
Requires:
1. Extract twice with different parameters
2. Blend the two extraction results
3. Handle different sequence lengths

**Usage pattern:**
```python
# Extract style (image tokens, no text)
style_result = extractor.extract(image=img, text=None, image_tokens_no_markers=True)

# Extract content (text tokens, with prompt)
content_result = extractor.extract(image=img, text=prompt, text_tokens_only=True)

# Blend the two
final_emb = blend_embeddings(style_result.embeddings, content_result.embeddings, alpha=0.3)
```

### Phase 4: Multi-Layer Extraction
Most advanced: Extract from different layers for different token types.

```python
# Extract image tokens from layer -5 (more visual detail)
img_result = extractor.extract(..., hidden_layer=-5, image_tokens_no_markers=True)

# Extract text tokens from layer -15 (better semantics)
text_result = extractor.extract(..., hidden_layer=-15, text_tokens_only=True)

# Blend
final = blend_embeddings(img_result.embeddings, text_result.embeddings, alpha=0.3)
```

## Expected Research Outcomes

### If VL text tokens match Qwen3-4B quality:
- **Simplification:** Remove Qwen3-4B from pipeline entirely
- **VRAM savings:** Only load one model (Qwen3-VL) instead of two
- **Speed improvement:** One extraction instead of two + blend
- **User benefit:** Simpler setup, single model download

### If intra-VL blending works well:
- **Better quality:** Clean separation of style (image tokens) and content (text tokens)
- **More control:** Direct ratio control over style transfer strength
- **New feature:** "Style transfer slider" - 0% to 100% image token influence

### If different layers work better for different tokens:
- **Quality improvement:** Optimal layer per token type
- **Research insight:** Better understanding of VLM layer specialization
- **Future optimization:** Layer-specific extraction strategies

## Running the Experiments

### Quick tests (recommended to start):
```bash
# Test if VL can replace Qwen3-4B
uv run experiments/qwen3_vl/scripts/run_comparison.py \
  --image experiments/inputs/test_scene.png \
  --prompt "Homer Simpson eating spaghetti" \
  --experiment vl_only_vs_qwen3 \
  --output-dir experiments/results/vl_vs_qwen3

# Test different layers for image vs text tokens
uv run experiments/qwen3_vl/scripts/run_comparison.py \
  --image experiments/inputs/test_scene.png \
  --prompt "Homer Simpson eating spaghetti" \
  --experiment vl_layer_by_token \
  --quick \
  --output-dir experiments/results/layer_by_token

# Test VL prompt variations
uv run experiments/qwen3_vl/scripts/run_comparison.py \
  --image experiments/inputs/test_scene.png \
  --prompt "Homer Simpson eating spaghetti" \
  --experiment vl_prompt_variations \
  --output-dir experiments/results/prompt_variations
```

### Full experiments:
```bash
# All layer combinations (10 configs)
uv run experiments/qwen3_vl/scripts/run_comparison.py \
  --image experiments/inputs/test_scene.png \
  --prompt "Homer Simpson eating spaghetti" \
  --experiment vl_layer_by_token \
  --output-dir experiments/results/layer_by_token_full
```

## Success Metrics

### Quality Assessment
- Visual inspection of generated images
- Prompt adherence (does it generate the requested subject?)
- Style transfer (does it match reference visual style?)
- Artifact presence (color bleeding, grid patterns)

### Quantitative Metrics (if compute available)
- SigLIP score (image-text alignment)
- ImageReward score (human preference)
- CLIP similarity to reference (for style transfer)

### Efficiency Metrics
- GPU memory usage
- Generation time
- Model loading time

## Future Directions

### If results are promising:

1. **Integrate into main pipeline**
   - Add `vl_intra_blend_ratio` to config.toml
   - Add CLI flags for dual extraction
   - Update web UI with new controls

2. **Optimize extraction**
   - Cache VL extractions for different token selections
   - Parallel extraction for dual mode
   - Optimize layer switching

3. **Research extensions**
   - Attention-weighted blending (important tokens get different ratios)
   - Per-token blend ratios (graduated alpha across sequence)
   - Multi-image conditioning (blend multiple VL extractions)

4. **Documentation**
   - Update findings.md with new insights
   - Create user guide for intra-VL blending
   - Write paper on zero-shot VL-to-diffusion transfer

## References

- [findings.md](findings.md) - Original VL conditioning findings
- [techniques.md](techniques.md) - Deep dive into techniques
- Parent issue: Token position discovery (2025-12-11)

---

**Status:** Experiments 1-3 implemented and ready to run. Experiments 4-5 require Phase 2 infrastructure (intra-VL blending and dual extraction).
