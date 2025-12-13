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
