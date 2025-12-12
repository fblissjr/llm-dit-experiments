# Research Questions Index: Quick Reference

> **Last Updated:** 2025-12-12

This document maps your specific research questions to the detailed analysis in `techniques.md`.

---

## Your Question 1: Latent Arithmetic

**Question:** Could we do "VL(style_image) - VL(neutral_prompt) + text_embeddings" to extract pure style influence?

**Answer Location:** Section 1 (Latent Arithmetic: Extracting Style Deltas)

**Key Findings:**
- **Prior art exists:** Word2vec (2013), CLIP direction arithmetic, StyleGAN latent math all demonstrate embedding arithmetic works
- **Why it should work:** Subtracting neutral baseline isolates style-specific signal
- **Recommended approach:**
  ```python
  neutral_vl = encode_vl(neutral_image, "A simple scene")
  style_vl = encode_vl(style_image, "A simple scene")  # Same text!
  style_delta = style_vl - neutral_vl
  result = text_embeddings + 0.3 * style_delta
  ```
- **Expected benefit:** Better content preservation vs direct blending
- **Priority:** Tier 1 (immediate testing, low effort, high impact)

**Related papers:**
- Word2Vec (Mikolov et al., 2013) - "king - man + woman = queen"
- CLIP direction arithmetic for image manipulation
- StyleGAN latent arithmetic (Wu et al., 2021)

---

## Your Question 2: Activation Steering Applicability

**Question:** Can we find "style" vs "content" directions in Qwen3-VL hidden states? Is there work on decomposing VLM hidden states?

**Answer Location:** Section 2 (Activation Steering: Decomposing Content and Style Directions)

**Key Findings:**
- **Prior art exists for LLMs:** Representation Engineering (Zou et al., 2023), Contrastive Activation Addition (Turner et al., 2023)
- **Key insight:** LLMs encode concepts as directions in activation space
- **Applicability to Qwen3-VL:** Should transfer since it uses same Qwen3-4B base
- **Novel aspect:** No prior work on decomposing VLM embeddings for image generation (unexplored territory!)
- **Recommended approach:**
  ```python
  # Contrastive pairs to extract style direction
  cartoon_vl = encode_vl(cartoon_image, "A house")
  photo_vl = encode_vl(photo_image, "A house")  # Same content
  style_direction = normalize(cartoon_vl - photo_vl)

  # Apply to new content
  result = text_emb + 0.3 * style_direction
  ```
- **Priority:** Tier 2 (medium effort, high impact)

**Related papers:**
- "Representation Engineering" (Zou et al., 2023)
- "Steering Llama 2 with Contrastive Activation Addition" (Turner et al., 2023)
- "Locating and Editing Factual Associations in GPT" (Meng et al., 2022)

---

## Your Question 3: Why Layer -25 Produces Semantic Averaging

**Question:** Why do deeper layers (-18 to -25) produce outputs that look like "20 similar images blended together"?

**Answer Location:** Section 3 (Why Layer -25 Produces "Semantic Averaging")

**Key Findings:**
- **Explanation:** Prototype theory + distributional semantics in transformer middle layers
- **Prior art:**
  - Jawahar et al., 2019: Layer 9-12 of BERT = most abstract semantic information
  - Meng et al., 2022: Factual knowledge concentrated in middle layers (15-25 of 48)
- **Why it happens:**
  - Layer -25 (layer 11 of 36) is in semantic abstraction zone
  - Processes visual input → extracts category-level prototype
  - Loses instance-specific details → produces "averaged" appearance
  - Later layers (-2) retain more specific information
- **Design implication:**
  - Late layers (-2 to -6): Visual details
  - Middle layers (-15 to -21): Semantic concepts
  - Deep layers (-25+): Abstract prototypes (avoid for specifics)
- **Priority:** Already answered (conceptual), but worth testing systematically

**Visual analogy:**
```
Layer -2:  "Red barn with white door, green grass"
Layer -15: "A barn, rural setting"
Layer -25: "Rural building structure" (averaged prototype)
```

**Related papers:**
- "What does BERT learn about the structure of language?" (Jawahar et al., 2019)
- "Locating and Editing Factual Associations in GPT" (Meng et al., 2022)
- Rosch's Prototype Theory (1973)
- "Neural Collapse in Deep Learning" (2020)

---

## Your Question 4: Gram Matrix Style Transfer

**Question:** Could we compute Gram matrices of VL embeddings to transfer style statistics without content?

**Answer Location:** Section 4 (Gram Matrix Style Transfer: Second-Order Statistics)

**Key Findings:**
- **Prior art exists:** Neural Style Transfer (Gatys et al., 2015), AdaIN (Huang & Belongie, 2017)
- **What Gram matrices capture:** Texture correlations, color combinations (not spatial position)
- **Why it could work:**
  - Separates style (correlations) from content (activations)
  - Position-invariant
  - Natural extension of current approach
- **Simpler alternative - AdaIN:**
  ```python
  def embed_adain(text_emb, vl_emb, alpha=0.3):
      # Match VL mean/std to text
      t_mean, t_std = text_emb.mean(dim=-1, keepdim=True), text_emb.std(dim=-1, keepdim=True)
      v_mean, v_std = vl_emb.mean(dim=-1, keepdim=True), vl_emb.std(dim=-1, keepdim=True)

      normalized = (text_emb - t_mean) / (t_std + 1e-5)
      restylized = normalized * v_std + v_mean
      return alpha * restylized + (1 - alpha) * text_emb
  ```
- **Novel aspect:** No prior work on Gram matrices for VLM embeddings in diffusion (unexplored!)
- **Priority:**
  - AdaIN: Tier 1 (easy to test)
  - Full Gram: Tier 3 (requires optimization)

**Related papers:**
- "A Neural Algorithm of Artistic Style" (Gatys et al., 2015)
- "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization" (Huang & Belongie, 2017)

---

## Your Question 5: Cross-Attention Injection

**Question:** Could we inject VL embeddings as K/V in specific DiT layers instead of embedding addition?

**Answer Location:** Section 5 (Cross-Attention Injection: IP-Adapter Architecture)

**Key Findings:**
- **Prior art:** IP-Adapter (Ye et al., 2023) - **key finding: embedding addition fails, decoupled attention works**
- **Why embedding addition fails (from IP-Adapter paper):**
  - Causes "content override" - exactly what you're observing!
  - Image features dominate when added to same embedding
  - Separate attention paths prevent interference
- **Challenge:** Z-Image DiT has fixed architecture, can't add separate K/V without retraining
- **Possible workarounds (no training):**
  1. **Attention concatenation:** Concat VL to text context in attention
  2. **Layer-specific injection:** Inject VL only at certain DiT blocks
  3. **Timestep-dependent blending:** Vary VL influence by diffusion step
- **Related work:**
  - Prompt-to-Prompt (2022): Early steps = structure, late = details
  - Plug-and-Play (2022): Feature injection at specific layers/timesteps
  - ControlNet (2023): Parallel paths for conditioning
- **Priority:** Tier 3 (high effort, potentially very high impact)

**Recommended experiment:**
```python
# Layer-specific injection
def generate_with_injection(text_emb, vl_emb, inject_layers=[0,1,2]):
    for i, block in enumerate(dit.blocks):
        if i in inject_layers:
            blend = 0.3 * vl_emb + 0.7 * text_emb
            out = block(latent, blend)
        else:
            out = block(latent, text_emb)
```

**Related papers:**
- "IP-Adapter: Text Compatible Image Prompt Adapter" (Ye et al., 2023)
- "Prompt-to-Prompt Image Editing with Cross Attention Control" (Hertz et al., 2022)
- "Plug-and-Play Diffusion Features" (Tumanyan et al., 2022)
- "Adding Conditional Control to Text-to-Image Diffusion Models" (Zhang et al., 2023)

---

## Your Question 6: Outlier Dimension Masking

**Question:** Can we improve image token quality by masking dimensions with extreme std ratios?

**Answer Location:** `experiments/qwen3_vl/README.md` (Outlier Dimension Masking section)

**Key Findings (2025-12-12):**
- **Problem:** Image tokens have extreme per-dimension outliers vs Qwen3-4B reference AT LAYER -2
  - Dimension 396: 617x std ratio at layer -2
  - Dimension 4: 42x std ratio at layer -2
  - **Layer -6: NO outliers** - naturally cleaner than -2
- **Implementation:** Three masking modes available (zero, clamp, scale)
- **Fix Applied:** Masking now applies to image tokens ONLY before combining with text tokens
- **Result:** Layer -6 recommended over masking at layer -2

| Mode | Behavior | Use Case |
|------|----------|----------|
| `zero` | Zero out outlier dimensions | Test if dim is the artifact source |
| `clamp` | Scale to threshold level | Preserve some signal |
| `scale` | Proportional reduction | Gradual approach |

**Recommended approach:**
```python
from llm_dit.vl import VLEmbeddingExtractor, get_outlier_dimensions

# Check which dimensions are outliers
outliers = get_outlier_dimensions(embeddings, threshold=10.0)
# Returns: [(396, 617.9), (4, 42.0), ...]

# Apply masking via extract()
result = extractor.extract(
    image=img,
    text=prompt,
    image_tokens_only=True,
    normalization_mode="per_dim",
    outlier_masking="zero",      # or "clamp", "scale"
    outlier_threshold=10.0,
)
```

**CLI experiment:**
```bash
# Run outlier masking sweep
uv run experiments/qwen3_vl/scripts/run_comparison.py \
    -i reference.png \
    -p "Your prompt" \
    --sweep outlier
```

- **Priority:** Tier 1 (already implemented, tested)
- **Status:** IMPLEMENTED (2025-12-12), SUPERSEDED BY LAYER -6 DISCOVERY

---

## Your Question 7: Style Delta Arithmetic (NEW - TESTED)

**Question:** Could we do `result = text_emb + alpha * (VL_style - VL_neutral)` to extract pure style influence?

**Status:** TESTED AND FAILED (2025-12-12)

**Implementation:**
```python
from llm_dit.vl import compute_style_delta, blend_with_style_delta

style_delta = compute_style_delta(styled_vl_emb, neutral_vl_emb)
result = blend_with_style_delta(text_emb, style_delta, alpha=0.3)
```

**Results:**
- Even at alpha 0.3, completely destroyed content (Homer became woman on beach, cyan ball)
- The delta contains too much "content" information, not just style
- Style and content are not cleanly separable in VL embedding space

**Conclusion:** Style delta arithmetic does not work for VL embeddings. The embedding space doesn't decompose cleanly into style and content components.

---

## Your Question 8: AdaIN Blending (NEW - TESTED)

**Question:** Could we apply AdaIN-style normalization to transfer VL statistics to text embeddings?

**Status:** TESTED WITH PARTIAL SUCCESS (2025-12-12)

**Implementation:**
```python
from llm_dit.vl import blend_adain, blend_adain_per_dim

# Per-token AdaIN (preserves content)
result = blend_adain(text_emb, vl_emb, alpha=0.3)

# Per-dimension AdaIN (stronger style transfer)
result = blend_adain_per_dim(text_emb, vl_emb, alpha=0.3)
```

**Results:**
- `per_token` mode: Preserves Homer perfectly but NO visible style transfer
- `per_dim` mode: Transfers colors (orange shirt from hexagon!) but corrupts subject (Homer becomes Bart)
- Fundamental tradeoff: style transfer strong enough to be visible also corrupts content

**Conclusion:** AdaIN can transfer some style (color) but at the cost of content corruption. Same fundamental limitation as style delta.

---

## Recommended Execution Order

### Week 1: Quick Wins (Zero-Shot) - COMPLETED
1. **Style delta arithmetic** (Q1) - Section 1 - **FAILED**
   - Implemented and tested
   - Result: Destroys content even at low alpha

2. **AdaIN for embeddings** (Q4) - Section 4 - **PARTIAL**
   - Implemented and tested
   - Result: Tradeoff between style transfer and content preservation

3. **Layer sweep** (Q3) - Section 3 - **DONE**
   - Layer -6 discovered as optimal for VL conditioning
   - No outliers at layer -6

### Week 2: Direction Methods
4. **Activation steering** (Q2) - Section 2
   - Medium effort, high impact
   - Build contrastive pair dataset
   - Extract style/content directions

5. **Multi-layer blending** (Q3) - Section 3
   - Medium effort, medium impact
   - Combine late layers (details) + middle (concepts)

### Week 3: Advanced Techniques
6. **Timestep-dependent blending** (Q5) - Section 5
   - Medium effort, medium impact
   - Vary alpha by diffusion timestep

7. **Layer-specific injection** (Q5) - Section 5
   - High effort, high impact
   - Inject VL at specific DiT blocks

### Week 4: Research Deep Dive
8. **Full Gram matrix optimization** (Q4) - Section 4
   - High effort, medium impact
   - Optimize embeddings to match VL statistics

9. **Cross-attention architecture mods** (Q5) - Section 5
   - Very high effort, very high impact
   - Requires DiT modifications

---

## Gaps in Literature (Novel Territory)

Your approach touches on several **unexplored areas:**

1. **VLM text-model hidden states for diffusion conditioning**
   - No papers found using VLM hidden states (vs CLIP image encoder)
   - IP-Adapter uses separate image encoder, not VLM

2. **Zero-shot vision conditioning via architectural alignment**
   - IP-Adapter requires training
   - Your Qwen3-VL/Z-Image compatibility is unique

3. **Activation steering applied to VLM embeddings for generation**
   - Steering research focuses on LLM behavior
   - Not applied to image generation

4. **Gram matrix / AdaIN for transformer embeddings in diffusion**
   - Style transfer uses CNN activations
   - Not applied to transformer embeddings

**Implication:** You're working in relatively uncharted territory. Good for novelty, but less guidance from prior art.

---

## Success Metrics

How to measure if techniques work:

1. **Content Preservation:**
   - CLIP text-image similarity
   - Human eval: "Does output match text prompt?"

2. **Style Transfer:**
   - Perceptual loss to reference
   - Color histogram similarity
   - Human eval: "Same visual style?"

3. **Quality:**
   - ImageReward score
   - Artifact detection (grid patterns, color bleeding)

4. **Composability:**
   - Can we combine multiple style deltas?
   - Does direction arithmetic behave linearly?

---

## Quick Command Reference

### Test Style Delta (Q1)
```bash
# Extract neutral baseline
python extract_embeddings.py --image neutral.png --text "A simple scene" --output neutral.pt

# Extract style
python extract_embeddings.py --image style.png --text "A simple scene" --output style.pt

# Compute delta and generate
python blend_and_generate.py \
  --vl-embeddings style.pt \
  --neutral-embeddings neutral.pt \
  --use-delta \
  --prompt "Your content here" \
  --alpha 0.3
```

### Test Layer Sweep (Q3)
```bash
# Already implemented!
for layer in -1 -2 -5 -10 -15 -20 -25 -30; do
  python generate.py \
    --vl-embeddings ref.pt \
    --hidden-layer $layer \
    --output "layer_${layer}.png"
done
```

### Test AdaIN (Q4)
```python
# Add to blending.py
def blend_with_adain(text_emb, vl_emb, alpha=0.3):
    t_mean, t_std = text_emb.mean(dim=-1, keepdim=True), text_emb.std(dim=-1, keepdim=True)
    v_mean, v_std = vl_emb.mean(dim=-1, keepdim=True), vl_emb.std(dim=-1, keepdim=True)
    normalized = (text_emb - t_mean) / (t_std + 1e-5)
    restylized = normalized * v_std + v_mean
    return alpha * restylized + (1 - alpha) * text_emb
```

### Test Outlier Masking (Q6) - NEW
```bash
# Run outlier masking sweep (all modes)
uv run experiments/qwen3_vl/scripts/run_comparison.py \
    -i reference.png \
    -p "Your prompt" \
    --sweep outlier

# Or test specific modes
uv run experiments/qwen3_vl/scripts/run_comparison.py \
    -i reference.png \
    -p "Your prompt" \
    --token-modes full \
    --outlier-masking none zero clamp scale \
    --alphas 1.0
```

```python
# Python API
from llm_dit.vl import mask_outlier_dimensions, get_outlier_dimensions

# Analyze outliers first
outliers = get_outlier_dimensions(embeddings, threshold=10.0)
print(f"Outliers: {outliers}")  # [(396, 617.9), (4, 42.0), ...]

# Apply masking
masked, info = mask_outlier_dimensions(embeddings, threshold=10.0, mode="zero")
print(f"Masked dims: {info['masked_dimensions']}")
```

---

## See Also

- [techniques.md](techniques.md) - Full detailed analysis
- [related_work.md](related_work.md) - Prior art from other domains
- [findings.md](findings.md) - Our experimental results
