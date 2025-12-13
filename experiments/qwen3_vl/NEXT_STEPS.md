# VL Conditioning Research: Next Steps

last updated: 2025-12-12

## Today's Key Findings Summary

### 1. Profile Mismatch Bug (FIXED)

`run_comparison.py` was using `rtx4090` profile while `test_all_blend_modes.py` used `default` profile. The critical difference:

| Profile | `long_prompt_mode` | Effect |
|---------|-------------------|--------|
| `default` | `interpolate` | Simple linear resampling |
| `rtx4090` | `attention_pool` | Cosine similarity weighting |

The `attention_pool` mode was processing embeddings differently, causing inconsistent results between scripts. Fixed by changing `run_comparison.py` to use `default` profile.

### 2. Hidden Layer Configuration

Both scripts now use:
- **VL extraction**: Layer -6 (from sweep presets)
- **Text encoding**: Layer -6 (from updated config.toml `[default.encoder].hidden_layer = -6`)

Layer -6 produces cleaner results than -2 for VL conditioning. This is ~83% depth in Qwen3's 36-layer architecture.

### 3. Style Transfer Best Settings

From empirical testing:
- **Steps**: 4 (better style fusion than 9)
- **Blend mode**: `adain_per_dim` (preserves text content, transfers VL style)
- **Alpha**: 0.3-0.5
- **Strength**: 0.9 (for img2img)
- **Token mode**: `full` (text_tokens_only=False - CRITICAL)
- **Profile**: `default` (uses `interpolate` long_prompt_mode)

### 4. New Sweep: `vl_only_vs_qwen3`

Implemented the missing experiment that compares:
1. Pure Qwen3-4B (alpha=0.0) - baseline
2. VL text tokens only (alpha=1.0, text_tokens_only=True)
3. VL full sequence (alpha=1.0, includes image+text)
4. Traditional blend (alpha=0.3)

---

## Gemini's Research Feedback Analysis

Gemini provided a comprehensive research prompt based on our earlier findings. Here's how it aligns with our empirical work:

### Confirmed by Our Experiments

#### "Image Token Artifacts"
Gemini notes: "Image tokens possess extremely low standard deviation (~7.0 compared to expected ~58.0) and cause severe high-frequency grid artifacts"

**Our findings match:**
- Image tokens have std ~13.25 vs text std ~61-70
- Dimension 396 has 617x std ratio (extreme outlier)
- Dimension 4 has 42x std ratio
- These cause visible grid/blocky artifacts at layer -2

#### "Layer -8 Sweet Spot" (75% depth)
Gemini suggests: "Optimal extraction point is deep in the model (Layer -8, approx 75% depth)"

**Our findings:**
- Layer -6 works best for VL (~83% depth in 36-layer model)
- Layer -2 (penultimate, ~94% depth) causes heavy artifacts
- This is consistent with "middle-late" layers being optimal

Gemini's hypothesis: "Final layers suffer from SFT Drift - becoming over-specialized to next-token prediction while middle-late layers preserve Platonic/universal semantic concepts"

**This explains our observation:** VL fine-tuning overwrote later layers for vision tasks, making them worse for cross-model transfer.

### Contradicts or Nuances Our Findings

#### "Text Token" Theory
Gemini suggests: "Text tokens after the image absorb visual context via self-attention (acting as In-Context LoRA)"

**Our findings CONTRADICT this:**
- `text_tokens_only=True` removes ALL image information
- VL does NOTHING with text_tokens_only=True (we tested this extensively)
- Image information is in IMAGE TOKEN POSITIONS (~1026 tokens for 1024x1024)
- Text tokens do NOT absorb visual context - they remain pure text embeddings

**Why the confusion:** The text tokens ARE conditioned by image tokens via attention during VL forward pass, but when we EXTRACT just the text token positions, we're discarding all the image token embeddings that carry the visual information. The attention-based conditioning isn't "stored" in the text token hidden states - it's a runtime computation.

#### Style Vector Approach
Gemini suggests: "Condense the 256 toxic image tokens into a single clean Style Vector"

**Our findings show this is hard:**
- AdaIN transfers style but corrupts content (Homer becomes Bart)
- Style delta arithmetic also corrupts content
- Simple pooling loses spatial information
- The "style" and "content" aren't cleanly separable in embedding space

### Research Directions Worth Exploring

#### 1. Modality Gap / Geometric Anisotropy
Gemini asks: "Why do vision tokens occupy a different cone/sub-manifold than text tokens?"

**Relevant to us:**
- Our per-dimension normalization addresses this partially
- Whitening transforms could help more
- The 617x outlier in dim 396 suggests specific dimensions are problematic

**Implementation idea:** Full whitening transform instead of just scaling:
```python
def whiten_embeddings(emb, reference_cov):
    """Apply whitening transform to match reference covariance."""
    # Compute ZCA whitening
    U, S, Vt = torch.linalg.svd(emb, full_matrices=False)
    whitened = U @ torch.diag(1.0 / S) @ Vt
    # Apply reference covariance
    return whitened @ reference_cov
```

#### 2. Layer-Wise Representation Evolution
Gemini cites: "Platonic Representation Hypothesis (Huh et al., 2024)"

**This could explain:**
- Why layer -6 works better than -2
- Why both VL and text encoders can share this layer
- Potential for finding even better layers via probing

**Research question:** Is there a principled way to find the optimal layer rather than grid search?

#### 3. Model Stitching / Zero-Shot Transfer
Gemini asks: "Has anyone attempted to stitch VLM hidden states into diffusion text encoder without training?"

**This is exactly what we're doing.** Literature search could reveal:
- Best practices for frozen model connection
- Known failure modes
- Normalization techniques that work

#### 4. Style Vector Extraction
Gemini suggests: "Attention-Weighted Pooling, GeM Pooling, SVD/PCA-based extraction"

**We haven't tried:**
- GeM (Generalized Mean) Pooling
- SVD-based style extraction
- Attention-weighted pooling with learned/heuristic weights

**Implementation idea:**
```python
def gem_pool(embeddings, p=3.0):
    """Generalized Mean Pooling - emphasizes larger activations."""
    return (embeddings.abs() ** p).mean(dim=0) ** (1.0 / p)

def svd_style_vector(embeddings, k=10):
    """Extract top-k singular vectors as style representation."""
    U, S, Vt = torch.linalg.svd(embeddings, full_matrices=False)
    # Top-k components capture global structure, not spatial noise
    return (U[:, :k] @ torch.diag(S[:k]) @ Vt[:k, :]).mean(dim=0)
```

---

## Open Questions

### Technical Questions

1. **Why does layer -6 work better than -8?**
   - Gemini suggests -8 (75% depth), we found -6 (~83% depth) optimal
   - Is this model-specific (Qwen3-VL vs generic VLM)?
   - Need systematic layer sweep with proper metrics

2. **Why does AdaIN corrupt content?**
   - Per-dim statistics transfer should preserve structure
   - But "Homer becomes Bart" suggests semantic drift
   - Is the corruption in specific dimensions? Can we mask them?

3. **Is the 617x outlier in dim 396 fundamental or fixable?**
   - Only appears at layer -2, not -6
   - Related to VL fine-tuning?
   - Could we surgically fix this dimension?

4. **Why do 4 steps work better than 9 for style transfer?**
   - More aggressive denoising = more style influence?
   - Or is noise schedule different at low steps?

### Architectural Questions

1. **Can we use VL to REPLACE Qwen3-4B entirely?**
   - If VL text tokens (with image context) work, we could simplify pipeline
   - But our tests show text_tokens_only discards all image info
   - Need to re-test with correct understanding

2. **Is there a hybrid approach?**
   - Use Qwen3-4B for text encoding
   - Use VL just for style vector extraction
   - Add style vector to text embeddings
   - Current AdaIN approach is close but corrupts content

3. **Would a minimal adapter help?**
   - Single linear layer between VL and DiT
   - Train on small dataset of (image, VL_emb, good_output) triples
   - Could fix the distribution mismatch

---

## Concrete Next Steps

### Immediate (Can Do Now)

1. **Run `vl_only_vs_qwen3` sweep** - finally test the VL vs Qwen3 comparison properly

2. **Document profile fix in CLAUDE.md** - Done, but verify it's complete

3. **Test whitening transform** - implement full whitening instead of just scaling

### Short-Term Research

1. **Use arxiv researcher agent** to find papers on:
   - "Platonic Representation Hypothesis" (Huh et al., 2024)
   - "Model Stitching" / "Representation Stitching"
   - "Activation Steering" for visual style
   - AdaIN variants for transformers

2. **Implement GeM pooling and SVD-based style extraction**
   - New blend modes: `gem_pool`, `svd_style`
   - Test if they preserve content better than AdaIN

3. **Systematic layer sweep with metrics**
   - Not just visual comparison
   - Use CLIP similarity, ImageReward, or SigLIP
   - Find principled optimal layer

### Medium-Term

1. **Investigate whitening/ZCA transforms**
   - Full distribution matching, not just scaling
   - Could fix the modality gap more fundamentally

2. **Train minimal adapter**
   - If zero-shot approaches plateau
   - Single linear layer should be enough
   - Small dataset of reference images

3. **Explore "Representation Engineering"**
   - Zou et al.'s work on activation steering
   - Could enable surgical style injection

---

## Code Changes Made Today

### `run_comparison.py`
1. Changed default profile from `rtx4090` to `default`
2. Added `vl_only_vs_qwen3` sweep preset
3. Added `special_sweeps` handling for custom config generation
4. Fixed sweep choices to include new presets

### `config.toml`
1. Changed `[default.encoder].hidden_layer` from -2 to -6
2. Changed `[default.encoder].max_length` from 512 to 1400
3. Changed `[default.vl].device` from "auto" to "cpu"

### `CLAUDE.md`
1. Added "Profile Configuration (CRITICAL)" section
2. Updated Experiment Runner examples with current sweeps
3. Added recommended style transfer settings

---

## Session Notes

The key insight from today: **consistency matters**. The profile mismatch between scripts was causing different `long_prompt_mode` behavior, which affected embedding processing in subtle ways. Always verify both scripts use the same configuration when comparing results.

Gemini's research prompt is valuable but contains one significant error: the claim that text tokens "absorb" visual context. Our empirical testing shows the opposite - text token positions contain no image information when extracted. The visual information is in the image token embeddings themselves, not "absorbed" into text tokens via attention.

This suggests the "In-Context LoRA" metaphor is misleading for our use case. The VLM's attention mechanism CONDITIONS the outputs on image tokens at runtime, but the hidden states at text positions don't STORE that conditioning - they're computed fresh each forward pass.

For style transfer to work, we need to explicitly include image token information in our blending, which is why `text_tokens_only=False` (token_mode="full") is required.

---

## Gemini's Deep Research Prompt (For Future Use)

This prompt can be used with deep research tools (Gemini Deep Research, Perplexity Pro, etc.) to find academic papers supporting our work:

```
**Research Goal: Theoretical Basis and Engineering Techniques for Zero-Shot Vision Conditioning via VLM Hidden States**

**Context:**
I am researching a novel, training-free method to generate images by "transplanting" hidden states from a Vision-Language Model (specifically **Qwen3-VL-4B**) into a Text-to-Image Diffusion Transformer (specifically **Z-Image**, which uses the Qwen3-4B text encoder). Because both models share the exact same base transformer architecture (hidden size 2560, same tokenizer), I have achieved successful zero-shot transfer. However, I have encountered specific "out-of-distribution" (OOD) artifacts that I need to solve using inference-time activation manipulation.

**My Empirical Discoveries:**
1.  **The "Text Token" Anomaly:** When the VLM processes an image + text prompt, the hidden states at the *text token positions* (after the image) act as a perfect conditioning signal. They appear to "absorb" visual context via self-attention (acting as an "In-Context LoRA") and drive the diffusion model correctly without artifacts.
2.  **The "Image Token" Artifacts:** The hidden states at the *image token positions* contain the strongest visual style information, but they are "toxic" to the diffusion model. They possess extremely low standard deviation (~7.0 compared to the expected ~58.0) and cause severe high-frequency grid artifacts and color bleeding when injected.
3.  **The "Layer -8" Sweet Spot:** Extracting hidden states from the penultimate layer (Layer -2) often fails or degrades quality. The optimal extraction point is deep in the model (Layer -8, approx 75% depth), which produces the cleanest semantic alignment.

**Research Request:**
Please conduct a deep literature review (prioritizing papers from 2023-2025) to answer the following four questions. I am looking for theoretical explanations for my findings and concrete mathematical techniques for my next phase ("Style Vector Injection").

**1. The "Modality Gap" and Geometric Anisotropy**
Search for research analyzing the geometric differences between vision-encoder projections and native text embeddings within Multimodal LLMs (like LLaVA, Qwen-VL, or Flamingo).
*   Why do vision tokens often occupy a different "cone" or sub-manifold than text tokens, even after the projection layer?
*   Search for "Representation Collapse" or "Anisotropy" in vision tokens. Does the literature confirm that vision tokens typically exhibit lower variance or different "outlier dimension" behavior than text tokens?
*   Are there known **inference-time normalization techniques** (e.g., Whitening, Centering, or "Moment Matching") used to align these OOD manifolds zero-shot?

**2. Layer-Wise Representation Evolution (The "Layer -8" Theory)**
I hypothesize that the final layers of Instruct-tuned models suffer from "SFT Drift"--becoming over-specialized to next-token prediction (syntax/formatting)--while the middle-late layers (like Layer -8) preserve the "Platonic" or universal semantic concepts.
*   Search for **"BERTology"** or **"Layer-wise information probing"** applied to modern Large Language Models.
*   Is there evidence that "Semantic Averaging" or "Prototype" representations peak at 75% depth?
*   Does the **"Platonic Representation Hypothesis"** (Huh et al., 2024) suggest that intermediate layers of different models are more aligned than their final output layers?

**3. "Model Stitching" and Zero-Shot Transfer**
Has anyone else attempted to stitch a VLM's hidden states directly into a Diffusion model's text encoder without training an adapter?
*   Search for **"Model Stitching"**, **"Representation Stitching"**, or **"Zero-Shot Cross-Model Transfer"**.
*   Look for papers similar to "Z-Image" or "GILL" but that focus on *training-free* mechanisms. Are there established best practices for connecting two frozen models that share a base architecture?

**4. Techniques for "Style Vector" Extraction**
My goal is to condense the 256 "toxic" image tokens into a single clean "Style Vector" to add to my text embeddings (`Text_Emb + alpha * Style_Vec`).
*   What are the state-of-the-art methods for **pooling** transformer tokens to capture global style while filtering out spatial grid noise? (e.g., Attention-Weighted Pooling, GeM Pooling, or SVD/PCA-based extraction).
*   Search for **"Representation Engineering"** (Zou et al.) or **"Activation Steering"** specifically applied to visual style in LLMs.
*   Are there **AdaIN (Adaptive Instance Normalization)** variants designed for *Transformer* embeddings (not CNNs)?

**Output Format:**
Please provide a report that:
1.  Validates or corrects my "Layer -8" and "Modality Gap" hypotheses with citations.
2.  Suggests specific mathematical formulas (e.g., "Match the second moment," "Apply a whitening transform") that I can implement in Python to fix the Image Token variance mismatch.
3.  Recommends the best pooling strategy for creating a "Style Vector" from VLM hidden states.
```

### Critical Correction to Gemini's Prompt

**Discovery #1 ("Text Token Anomaly") is INCORRECT based on our testing.**

Gemini states: "Text tokens absorb visual context via self-attention (acting as In-Context LoRA)"

**Our empirical finding:** `text_tokens_only=True` removes ALL image information. VL does literally nothing when you only extract text token positions. The visual information is stored in the IMAGE TOKEN POSITIONS (~1026 tokens for 1024x1024), not "absorbed" into text tokens.

The attention mechanism CONDITIONS outputs at runtime but doesn't STORE conditioning in hidden states. When we extract just text token positions, we get pure text embeddings with no visual content.

**Corrected understanding:**
- Image tokens: Contain visual information but have distribution mismatch (low std, outliers)
- Text tokens: Contain NO visual information when extracted - just text embeddings
- Full sequence: Required to get any VL effect

This fundamentally changes the "Style Vector" approach - we can't just use text tokens and expect them to carry style. We need to work with the "toxic" image tokens directly and fix their distribution issues.

---

## Gemini's Roadmap Analysis (With Corrections)

Gemini provided this analysis based on earlier findings. **Several conclusions are incorrect based on our later testing.**

### Gemini's Conclusions

**Q1: Can we eliminate the separate Qwen3-4B text encoder?**
Gemini says: "YES - VL Text Token positions contain sufficient prompt adherence"

**INCORRECT.** Our testing shows `text_tokens_only=True` makes VL do literally nothing. Text token positions contain NO visual information - they're just text embeddings. We CANNOT eliminate Qwen3-4B using text tokens alone.

**Q2: Where does Style vs Content live?**
Gemini says: "Style in image tokens, Content in text tokens - blend VL(Image) + VL(Text)"

**PARTIALLY CORRECT but misleading.** Image tokens DO contain style, but text tokens contain NO visual content at all. The "blend VL tokens" approach won't work because:
- Image tokens: Have visual info but distribution mismatch
- Text tokens: Have NO visual info, just text embeddings
- You can't get "content from text tokens" because there's no visual content there

**Q3: Which hidden layer is best?**
Gemini says: "Layer -8 is the new standard"

**CLOSE.** We found Layer -6 works best (~83% depth). Layer -8 (~78% depth) is similar range. Both are better than -2.

### Gemini's Suggested Roadmap

#### Step 1: "Architecture Killer" Experiment
```bash
uv run experiments/qwen3_vl/scripts/run_comparison.py \
  --image experiments/inputs/test_scene.png \
  --prompt "Homer Simpson eating spaghetti" \
  --sweep vl_only_vs_qwen3 \
  -o experiments/results/vl_vs_qwen3
```

**Status:** Implemented. This sweep now exists.

**Expected result:** Based on our testing, `pure_vl_text_tokens` will look IDENTICAL to `pure_qwen3_4b` because text tokens contain no VL info. This will DISPROVE the "deprecate Qwen3-4B" hypothesis.

#### Step 2: Layer Verification
Run layer sweep with -8 included.

**Status:** Our sweeps include -6, -8 is close. Could add explicit -8 test.

#### Step 3: Style Vector Injection (Gemini's Proposed Code)

```python
# Gemini's suggestion
def extract_with_style_injection(self, image, text, ratio=0.3):
    outputs = self.model(image, text, output_hidden_states=True)
    hidden = outputs.hidden_states[-8]

    # Separate components
    img_tokens = hidden[:, vision_start:vision_end, :]  # [1, 258, 2560]
    txt_tokens = hidden[:, vision_end:, :]              # [1, N, 2560]

    # Pool image tokens to single style vector
    style_vector = img_tokens.mean(dim=1, keepdim=True)

    # Add style to text tokens
    final_embeddings = txt_tokens + (ratio * style_vector)

    return final_embeddings
```

**Analysis:** This is essentially what our `blend_with_style_delta` does. We tried similar approaches:
- AdaIN: Transfers statistics, corrupts content (Homer -> Bart)
- Style delta: Adding style vector corrupts content too

**Why it fails:** The style vector contains "content" information too, not just style. Adding it shifts the semantic meaning, not just the visual style. The embedding space doesn't cleanly separate style from content.

**Potential fix:** Instead of mean pooling (which preserves all info), try:
- SVD to extract only top-k principal components (global structure)
- GeM pooling to emphasize larger activations
- Attention-weighted pooling to focus on "important" tokens

### Key Disagreement with Gemini

Gemini's entire analysis rests on the assumption that "text tokens absorb visual context via self-attention." This is **empirically false**.

When you extract hidden states at text token positions from Qwen3-VL after processing an image+text input:
- The attention mechanism DID condition on image tokens during forward pass
- But the HIDDEN STATES at text positions don't "store" that conditioning
- They're computed fresh each layer based on attention
- Extracting just those positions gives you text-only embeddings

The visual information is in the IMAGE TOKEN hidden states, period. There's no "In-Context LoRA" effect that transfers to text token positions in the extractable hidden states.

### What We Should Actually Try

1. **Fix image token distribution** - Whitening, better normalization
2. **Better pooling for style vector** - SVD, GeM, attention-weighted
3. **Surgical dimension fixing** - The 617x outlier in dim 396 might be fixable
4. **Different blend strategies** - Not AdaIN statistics transfer, but something else
5. **Minimal trained adapter** - If zero-shot hits a wall, one linear layer might bridge the gap
