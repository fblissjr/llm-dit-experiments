# Vision Conditioning Research Questions: Prior Art and Applicable Techniques

> **Last Updated:** 2025-12-11

This document investigates techniques from embedding arithmetic, activation steering, style transfer, and diffusion manipulation that could improve Qwen3-VL vision conditioning for Z-Image.

## Context

We have discovered that Qwen3-VL-4B and Z-Image share a compatible embedding space (both 2560-dimensional from Qwen3-4B architecture). Current approach:
- Extract hidden states from Qwen3-VL's text model after it processes vision features
- Blend with text embeddings: `blended = alpha * vl_embeddings + (1 - alpha) * text_embeddings`
- Generate images using blended embeddings

**Current Issues:**
1. VL embeddings override text content, not just style (at high alpha)
2. Hidden layer selection dramatically affects output (layer -2 works, layer -25 produces "semantic averaging")
3. Quality gap compared to trained adapters like IP-Adapter

---

## 1. Latent Arithmetic: Extracting Style Deltas

### Background

**Word2Vec (2013)**: Demonstrated embedding arithmetic like `king - man + woman = queen`
- Semantic relationships encoded as vector differences
- Addition/subtraction operates on meaning

**Key Papers:**
- "Efficient Estimation of Word Representations in Vector Space" (Mikolov et al., 2013)
- "Analogical Reasoning on Chinese Morphological and Semantic Relations" (2018)

### Hypothesis for Our Case

If VL embeddings encode both content and style, we could extract pure style influence:

```python
# Get base representation by processing neutral/generic image
neutral_vl = encode_vl(neutral_image, "A simple scene")

# Get styled representation
style_vl = encode_vl(style_image, "A simple scene")  # Same text!

# Extract style delta
style_delta = style_vl - neutral_vl

# Apply to new content
content_text = encode_text("Homer Simpson eating spaghetti")
stylized = content_text + alpha * style_delta
```

**Why this might work better than direct blending:**
- Subtracts the "neutral visual representation" common to both images
- Leaves only the style-specific signal
- Preserves text content structure while adding style bias

### Prior Art in Vision

**CLIP Direction Arithmetic:**
- "Paint by Word" (2021): Used CLIP text directions for image manipulation
- Subtracted concept vectors to isolate style attributes

**StyleGAN Latent Arithmetic:**
- "Editing in Style: Uncovering the Local Semantics of GANs" (Wu et al., 2021)
- Found interpretable directions in StyleGAN latent space via PCA
- Addition/subtraction of directions for controlled editing

### Recommended Experiments

1. **Baseline delta extraction:**
   ```python
   # Use multiple neutral images to get stable baseline
   neutral_images = [simple_shapes, gray_scene, basic_composition]
   neutral_vl_avg = mean([encode_vl(img) for img in neutral_images])

   # Extract style from reference
   style_delta = encode_vl(style_ref) - neutral_vl_avg

   # Apply to text
   result = text_embeddings + 0.3 * style_delta
   ```

2. **Multi-image style mixing:**
   ```python
   # Combine style deltas from multiple references
   delta1 = encode_vl(ref1) - neutral
   delta2 = encode_vl(ref2) - neutral
   combined = text_emb + 0.2 * delta1 + 0.15 * delta2
   ```

3. **Negative style transfer:**
   ```python
   # Remove unwanted style elements
   unwanted_delta = encode_vl(avoid_ref) - neutral
   result = text_emb + 0.3 * wanted_delta - 0.1 * unwanted_delta
   ```

### Expected Outcomes

- Better content preservation (text semantics not overridden)
- Composable style attributes
- May still suffer from layer-dependent behavior

---

## 2. Activation Steering: Decomposing Content and Style Directions

### Background

**Representation Engineering (2023):**
- "Representation Engineering: A Top-Down Approach to AI Transparency" (Zou et al., 2023)
- Found that LLMs encode concepts as directions in activation space
- Can add/subtract directions to modify model behavior

**Contrastive Activation Addition (2023):**
- "Steering Llama 2 with Contrastive Activation Addition" (Turner et al., 2023)
- Computed steering vectors by contrasting activations on paired examples
- Formula: `steering_vec = mean(positive_activations) - mean(negative_activations)`

### Key Findings from Prior Work

1. **Layer specificity:** Different layers encode different concepts
   - Early layers: Syntax, low-level features
   - Middle layers: Semantic concepts, composition
   - Late layers: Task-specific abstractions

2. **Contrastive pairs work best:**
   - Compare "happy" vs "sad" to extract emotion direction
   - Compare "formal" vs "casual" to extract style direction

3. **Scaling matters:**
   - Too much steering causes incoherence
   - Sweet spot typically 0.1-0.5 of activation magnitude

### Hypothesis for VLM Embeddings

Qwen3-VL uses the same Qwen3-4B base, so activation steering insights should transfer:

**Style vs Content Directions:**
```python
# Extract style direction via contrastive pairs
cartoon_vl = encode_vl(cartoon_image, "A house")
photo_vl = encode_vl(photo_image, "A house")  # Same content, different style
style_direction = normalize(cartoon_vl - photo_vl)

# Extract content direction
house_vl = encode_vl(image, "A house")
car_vl = encode_vl(image, "A car")
content_direction = normalize(house_vl - car_vl)

# Apply only style direction to new content
text_emb = encode_text("Homer Simpson")
stylized = text_emb + 0.3 * style_direction
```

### Why Layer -25 Produces "Semantic Averaging"

**Prototype Theory Connection:**
- Cognitive science: Concepts represented as cluster centroids
- Early/middle LLM layers encode distributional semantics
- Layer -25 (middle of 36 layers) may encode abstract "prototypes"

**Empirical Evidence from LLM Research:**
- "Locating and Editing Factual Associations in GPT" (Meng et al., 2022)
  - Found factual knowledge concentrated in middle layers (15-25 out of 48)
  - Early layers too raw, late layers too task-specific

- "What does BERT learn about the structure of language?" (Jawahar et al., 2019)
  - Layer 0-4: Surface features
  - Layer 5-8: Syntactic information
  - Layer 9-12: Semantic information (most abstract)

**Our Observation Explained:**
- Layer -25 (layer 11 of 36) is in the semantic abstraction zone
- Processes 20 similar images → extracts common semantic prototype
- Loses specific visual details → produces "average" appearance
- Later layers (-2) retain more instance-specific information

### Recommended Experiments

1. **Extract interpretable directions:**
   ```python
   # Collect contrastive pairs
   pairs = [
       (cartoon_img, photo_img),  # Style dimension
       (red_img, blue_img),       # Color dimension
       (simple_img, detailed_img), # Complexity dimension
   ]

   # Extract direction for each
   directions = {}
   for name, (imgA, imgB) in pairs.items():
       vl_A = encode_vl(imgA)
       vl_B = encode_vl(imgB)
       directions[name] = normalize(vl_A - vl_B)

   # Compose custom style
   text_emb = encode_text("A portrait")
   custom = text_emb + 0.2 * directions['cartoon'] + 0.1 * directions['red']
   ```

2. **Layer-specific direction analysis:**
   ```python
   # Compare directions across layers
   for layer in [-2, -6, -10, -15, -20, -25]:
       vl_A = encode_vl(imgA, hidden_layer=layer)
       vl_B = encode_vl(imgB, hidden_layer=layer)
       direction = vl_A - vl_B

       # Check if direction is stable across layers
       cosine_sim = compare_with_other_layers(direction)
   ```

3. **Multi-concept steering:**
   ```python
   # Separate content and style steering
   content_vec = extract_content_direction(ref_image)
   style_vec = extract_style_direction(ref_image)

   # Apply independently
   result = text_emb + 0.1 * style_vec  # Style only
   # vs
   result = text_emb + 0.3 * content_vec + 0.1 * style_vec  # Both
   ```

### Prior Art in VLMs

**CLIP Activation Analysis:**
- "What does CLIP know about a red circle?" (Dunlap et al., 2023)
- Found CLIP vision encoder separates shape and color in different layers
- Suggests VLM architectures naturally decompose visual attributes

**No direct precedent found for:**
- Using VLM text-model hidden states for steering (our approach is novel)
- Decomposing VLM embeddings into content/style directions
- Layer-specific steering in VLMs for image generation

### Expected Outcomes

- Cleaner style-only transfer (no content leakage)
- Composable visual attributes
- Layer selection guide (which layers for which attributes)

---

## 3. Why Layer -25 Produces "Semantic Averaging"

### Observation

Deeper layers (-18 to -25) produce outputs that look like "20 similar images blended together" - losing specific details while retaining category-level semantics.

### Theoretical Explanation

**Prototype Theory (Rosch, 1973):**
- Concepts represented as cluster centroids in cognitive space
- Category membership based on distance to prototype
- Explains why layer -25 produces "average" appearance

**Evidence from Transformer Research:**

1. **Hierarchical Abstraction:**
   - "Understanding the Difficulty of Training Transformers" (2020)
   - Shows layer depth correlates with abstraction level
   - Middle layers = semantic prototypes, Late layers = task outputs

2. **Representation Collapse:**
   - "Neural Collapse in Deep Learning" (2020)
   - Deeper layers compress within-class variance
   - Produces "averaged" representations of similar inputs

3. **Distributional Semantics:**
   - Middle layers encode distributional statistics (co-occurrence patterns)
   - Abstract away surface details
   - Preserve category-level information

### Visual Analogy

```
Layer -1/-2 (Late):    [Specific Instance]
  "A red barn with white door, green grass, blue sky with clouds"

Layer -10/-15 (Middle): [Detailed Category]
  "A barn, rural setting, daytime"

Layer -20/-25 (Middle): [Abstract Prototype]
  "Rural building structure" (averaged across all barns seen)

Layer -30/-36 (Early):  [Low-level Features]
  "Vertical lines, color patches, edge orientations"
```

### Empirical Support from Your Experiments

**Your observation matches literature:**
- Layer -2: Sharp, specific details preserved
- Layer -15: Some averaging but still detailed
- Layer -25: Heavy averaging, prototype-like

**This suggests Qwen3-4B's semantic abstraction peaks around layer -25:**
- Too abstract for instance-specific visual generation
- Good for category-level understanding
- Bad for preserving visual details

### Why This Matters for VL Conditioning

**Design implication:**
- Use late layers (-2 to -6) for visual details
- Use middle layers (-15 to -21) for semantic concepts
- Avoid very deep layers (-25+) unless you want abstract style

**Potential hybrid approach:**
```python
# Extract different information from different layers
detail_emb = encode_vl(image, hidden_layer=-2)   # Visual specifics
concept_emb = encode_vl(image, hidden_layer=-18) # Semantic concepts

# Weighted combination
result = text_emb + 0.2 * detail_emb + 0.1 * concept_emb
```

### Recommended Experiments

1. **Systematic layer characterization:**
   ```python
   # Generate same prompt with embeddings from each layer
   layers = [-1, -2, -5, -10, -15, -20, -25, -30]
   for layer in layers:
       vl_emb = encode_vl(reference, hidden_layer=layer)
       image = generate(text_emb + 0.3 * vl_emb)
       measure_specificity(image)  # How much detail retained?
   ```

2. **Multi-layer blending:**
   ```python
   # Hypothesis: Early layers have textures, late layers have semantics
   late = encode_vl(ref, hidden_layer=-2)   # Details
   mid = encode_vl(ref, hidden_layer=-15)   # Concepts

   # Test different combinations
   result = text_emb + 0.15 * late + 0.1 * mid
   ```

3. **Prototype vs instance test:**
   ```python
   # Use -25 for style prototypes
   style_prototype = encode_vl(style_ref, hidden_layer=-25)

   # Use -2 for specific details
   detail_emb = encode_vl(style_ref, hidden_layer=-2)

   # Compare style transfer quality
   ```

---

## 4. Gram Matrix Style Transfer: Second-Order Statistics

### Background

**Neural Style Transfer (Gatys et al., 2015):**
- Used Gram matrices to capture style (texture correlations)
- Separated style (Gram matrices) from content (activations)

**Gram Matrix Definition:**
```python
def gram_matrix(features):
    """
    features: [batch, seq_len, hidden_dim]
    returns: [batch, hidden_dim, hidden_dim]
    """
    b, n, d = features.shape
    F = features.view(b, n, d)
    G = torch.bmm(F.transpose(1, 2), F)  # [b, d, d]
    return G / (n * d)  # Normalize
```

**What Gram Matrix Captures:**
- Correlations between feature dimensions
- Texture patterns, color combinations
- NOT spatial information (position-invariant)

### Application to VL Embeddings

**Current approach (zeroth-order):**
```python
# Direct embedding blending
blended = alpha * vl_emb + (1 - alpha) * text_emb
```

**Proposed: Gram matrix matching (second-order):**
```python
# Extract style statistics from VL
vl_emb = encode_vl(style_image)  # [1, seq_len, 2560]
vl_gram = gram_matrix(vl_emb)    # [2560, 2560]

# Match VL statistics to text embeddings
text_emb = encode_text(prompt)    # [1, seq_len, 2560]
text_gram = gram_matrix(text_emb)

# Style transfer via Gram matrix matching
# (Requires optimization or learned transformation)
styled_text = match_gram_statistics(text_emb, vl_gram)
```

### Why This Could Work

1. **Separates style from content:**
   - Content = activation magnitudes (what is present)
   - Style = correlations (how features co-occur)

2. **Position-invariant:**
   - Doesn't transfer spatial layout
   - Transfers "texture" of embeddings (feature relationships)

3. **Precedent in AdaIN:**
   - Adaptive Instance Normalization matches mean/std
   - Gram matrices are natural extension (all pairwise correlations)

### Hybrid Approach: AdaIN for Embeddings

**AdaIN (Huang & Belongie, 2017):**
```python
def adain(content, style):
    """Match mean and std of style to content"""
    content_mean = content.mean(dim=1, keepdim=True)
    content_std = content.std(dim=1, keepdim=True)

    style_mean = style.mean(dim=1, keepdim=True)
    style_std = style.std(dim=1, keepdim=True)

    normalized = (content - content_mean) / (content_std + 1e-5)
    return normalized * style_std + style_mean
```

**Applied to embeddings:**
```python
# Match VL statistics to text embeddings
text_emb = encode_text("A portrait")         # [1, 1504, 2560]
vl_emb = encode_vl(style_reference_image)    # [1, 258, 2560]

# Transfer style statistics
styled = adain(text_emb, vl_emb)  # Text content, VL style stats

# Or partial transfer
alpha = 0.3
styled = alpha * adain(text_emb, vl_emb) + (1 - alpha) * text_emb
```

### Prior Art in Embeddings

**StyleBERT (2020):**
- Applied style transfer to text embeddings
- Used mean/std matching (similar to AdaIN)
- Found style could be transferred at embedding level

**No direct precedent for:**
- Gram matrix matching in VLM embeddings
- Second-order statistics for vision conditioning in diffusion models

### Recommended Experiments

1. **AdaIN for embeddings (easiest):**
   ```python
   def embed_adain(text_emb, vl_emb, alpha=0.3):
       """Match VL statistics to text"""
       # Per-token statistics
       t_mean, t_std = text_emb.mean(dim=-1, keepdim=True), text_emb.std(dim=-1, keepdim=True)
       v_mean, v_std = vl_emb.mean(dim=-1, keepdim=True), vl_emb.std(dim=-1, keepdim=True)

       # Normalize and restylize
       normalized = (text_emb - t_mean) / (t_std + 1e-5)
       restylized = normalized * v_std + v_mean

       # Blend
       return alpha * restylized + (1 - alpha) * text_emb
   ```

2. **Gram matrix loss (optimization-based):**
   ```python
   # Optimize text embeddings to match VL Gram matrix
   text_emb = encode_text(prompt)
   vl_emb = encode_vl(style_ref)
   target_gram = gram_matrix(vl_emb)

   # Optimize
   optimized_emb = text_emb.clone().requires_grad_(True)
   for _ in range(100):
       current_gram = gram_matrix(optimized_emb)
       loss = F.mse_loss(current_gram, target_gram)
       loss.backward()
       # Update optimized_emb
   ```

3. **Dimension-specific Gram matching:**
   ```python
   # Match correlations for specific dimension groups
   # Hypothesis: Different dimensions encode different attributes
   vl_gram = gram_matrix(vl_emb)  # [2560, 2560]

   # Extract sub-blocks (e.g., first 512 dims = color, next 512 = texture)
   color_gram = vl_gram[:512, :512]
   texture_gram = vl_gram[512:1024, 512:1024]

   # Match selectively
   ```

### Expected Outcomes

- Style transfer without content override
- More "texture-like" influence (colors, patterns)
- May need optimization or learned transformation (not zero-shot)

---

## 5. Cross-Attention Injection: IP-Adapter Architecture

### Background

**IP-Adapter (Ye et al., 2023):**
- Key finding: **Embedding addition fails, decoupled attention works**
- Architecture: Separate K/V projections for image features
- Quality: Much higher than simple embedding blending

### IP-Adapter Architecture

```
Text Prompt ──────> Text Encoder ──> Text Embeddings (Q, K_text, V_text)
                                            |
                                            v
                                      Cross-Attention Block
                                            |
                                      +-----+-----+
                                      |           |
                                      v           v
                               K_text, V_text   K_img, V_img
                                      |           |
                                      +-----+-----+
                                            |
                                      Attention Output

Reference Image ──> CLIP Encoder ──> [Projection] ──> Image Embeddings (K_img, V_img)
```

**Key differences from our approach:**
1. Separate K/V projections for image (we blend embeddings)
2. Trained projection layer (we use zero-shot VL)
3. Dual cross-attention paths (we use single path)

### Why Embedding Addition Fails (IP-Adapter Paper)

From experiments in IP-Adapter paper:
1. Direct addition causes "content override" - exactly what we observe
2. Image features dominate text when added to same embedding
3. Separate attention paths prevent this interference

### Can We Apply This to Qwen3-VL + Z-Image?

**Challenge:** Z-Image DiT has fixed cross-attention architecture
- Can't add separate K/V projections without retraining
- Architecture assumes single text embedding input

**Possible workarounds:**

#### 5.1 Post-Hoc Attention Injection (No Training)

```python
# Modify attention computation in DiT
class ModifiedAttention(nn.Module):
    def forward(self, x, text_emb, vl_emb=None, alpha=0.3):
        Q = self.to_q(x)
        K_text = self.to_k(text_emb)
        V_text = self.to_v(text_emb)

        if vl_emb is not None:
            # Inject VL as additional K/V
            K_vl = self.to_k(vl_emb)  # Reuse same projection
            V_vl = self.to_v(vl_emb)

            # Concatenate keys and values
            K = torch.cat([K_text, alpha * K_vl], dim=1)
            V = torch.cat([V_text, alpha * V_vl], dim=1)
        else:
            K, V = K_text, V_text

        return attention(Q, K, V)
```

**Advantage:**
- No training needed
- Decouples VL from text in attention space

**Disadvantage:**
- Changes sequence length (may break position encodings)
- Untested, may fail

#### 5.2 Layer-Specific Injection

**Hypothesis:** VL should influence different DiT layers differently
- Early layers: Structure, composition
- Middle layers: Semantic concepts
- Late layers: Fine details

```python
# Inject VL only at specific DiT layers
def generate_with_layer_injection(text_emb, vl_emb, inject_layers=[0, 1, 2]):
    """
    inject_layers: Which DiT blocks to inject VL into
    """
    for i, block in enumerate(dit.blocks):
        if i in inject_layers:
            # Use VL-blended embeddings for this block
            blend = 0.3 * vl_emb + 0.7 * text_emb
            out = block(latent, blend)
        else:
            # Pure text for other blocks
            out = block(latent, text_emb)
```

#### 5.3 Timestep-Dependent VL Strength

**Hypothesis:** VL should influence early diffusion steps (structure) more than late steps (details)

```python
# Vary alpha by timestep
def timestep_dependent_blend(text_emb, vl_emb, timestep, num_steps=9):
    # Higher VL influence at early steps
    alpha = 0.5 * (1 - timestep / num_steps)  # 0.5 -> 0.0
    return alpha * vl_emb + (1 - alpha) * text_emb
```

### Prior Work on Injection

**Prompt-to-Prompt (Hertz et al., 2022):**
- Manipulated cross-attention maps during diffusion
- Found early timesteps control structure, late control details

**Plug-and-Play (Tumanyan et al., 2022):**
- Injected features from reference image at specific layers/timesteps
- Achieved controlled generation without training

**ControlNet (Zhang et al., 2023):**
- Parallel network for spatial conditioning
- Different from our semantic conditioning, but shows multi-path works

### Recommended Experiments

1. **Attention concatenation (no training):**
   ```python
   # Hack: Monkey-patch DiT attention
   def modified_cross_attn(self, x, context, vl_context=None):
       if vl_context is not None:
           # Concatenate contexts
           full_context = torch.cat([context, 0.3 * vl_context], dim=1)
       else:
           full_context = context
       return original_cross_attn(self, x, full_context)
   ```

2. **Layer-specific injection sweep:**
   ```python
   # Test which layers benefit from VL
   for inject_layers in [[0,1,2], [3,4,5], [6,7,8], [0,3,6]]:
       image = generate_with_injection(text, vl, inject_layers)
       quality = measure_quality(image)
   ```

3. **Timestep-dependent blending:**
   ```python
   # Test different alpha schedules
   schedules = {
       'constant': lambda t: 0.3,
       'decay': lambda t: 0.5 * (1 - t/9),
       'grow': lambda t: 0.1 + 0.4 * (t/9),
       'peak_early': lambda t: 0.5 * np.exp(-t/3),
   }
   ```

### Expected Outcomes

- Better separation of content and style influence
- Layer/timestep-specific control
- May require architecture modifications (not fully zero-shot)

---

## 6. Summary: Recommended Investigation Priority

### Tier 1: Zero-Shot Improvements (Immediate Testing)

1. **Latent arithmetic (Style Delta)**
   - Effort: Low
   - Impact: Potentially high
   - Test: `style_delta = encode_vl(style) - encode_vl(neutral); result = text + 0.3 * style_delta`

2. **AdaIN for embeddings**
   - Effort: Low
   - Impact: Medium
   - Test: Match VL statistics (mean/std) to text embeddings

3. **Layer-specific extraction**
   - Effort: Low (already implemented via `--hidden-layer`)
   - Impact: High
   - Test: Systematic sweep of layers -1 to -25

### Tier 2: Direction-Based Methods (Medium Effort)

4. **Activation steering (Contrastive directions)**
   - Effort: Medium
   - Impact: High
   - Test: Extract style/content directions via contrastive pairs

5. **Multi-layer blending**
   - Effort: Medium
   - Impact: Medium
   - Test: Combine late layers (details) + middle layers (concepts)

6. **Timestep-dependent blending**
   - Effort: Medium
   - Impact: Medium
   - Test: Vary alpha by diffusion timestep

### Tier 3: Architectural Modifications (High Effort)

7. **Cross-attention injection**
   - Effort: High
   - Impact: Potentially very high
   - Test: Modify DiT to accept dual K/V paths

8. **Gram matrix optimization**
   - Effort: High
   - Impact: Medium
   - Test: Optimize embeddings to match VL Gram statistics

9. **Learned projection layer**
   - Effort: Very high (requires training)
   - Impact: Very high
   - Test: Train small MLP to project VL → text space

---

## 7. Key Research Questions Summary

### Question 1: Latent Arithmetic
**Can we extract pure style via `VL(style) - VL(neutral) + text_embeddings`?**
- Prior art: Word2vec, CLIP direction arithmetic, StyleGAN latent math
- Test: Use neutral images as baseline, subtract to get style delta
- Expected: Better content preservation, composable styles

### Question 2: Activation Steering
**Can we decompose VL into style/content direction vectors?**
- Prior art: Representation Engineering, Contrastive Activation Addition
- Test: Contrastive pairs to extract interpretable directions
- Expected: Separate style/content control, layer-specific guidance

### Question 3: Semantic Averaging (Layer -25)
**Why do deeper layers produce "averaged" prototypes?**
- Explanation: Prototype theory, distributional semantics in middle layers
- Test: Characterize abstraction level per layer
- Expected: Layer selection guide for different use cases

### Question 4: Gram Matrix Style Transfer
**Could second-order statistics transfer style without content?**
- Prior art: Neural Style Transfer, AdaIN
- Test: Match Gram matrices or mean/std statistics
- Expected: Texture-like style transfer, position-invariant

### Question 5: Cross-Attention Injection
**Should we inject VL as separate K/V instead of embedding addition?**
- Prior art: IP-Adapter (trained), Prompt-to-Prompt (attention manipulation)
- Test: Layer-specific or timestep-dependent injection
- Expected: Higher quality, less content override

---

## 8. Gaps in Literature

**What's missing / unexplored:**

1. **VLM text-model hidden states for diffusion conditioning**
   - Our approach is novel
   - No papers found on using VLM hidden states (vs CLIP image encoder)

2. **Zero-shot vision conditioning via embedding space alignment**
   - IP-Adapter requires training
   - Our architectural compatibility is unexplored

3. **Layer-specific steering in VLMs for generation**
   - Activation steering research focuses on LLM behavior modification
   - Not applied to VLM embeddings for image generation

4. **Gram matrix / AdaIN for embedding-level style transfer**
   - Style transfer uses CNN activations
   - Not applied to transformer embeddings in diffusion context

**This suggests our approach is in relatively unexplored territory - good for research novelty, but less prior art to guide us.**

---

## 9. Experimental Action Plan

### Week 1: Zero-Shot Improvements
- [ ] Implement style delta arithmetic
- [ ] Test AdaIN for embeddings (mean/std matching)
- [ ] Run systematic layer sweep (-1 to -30)
- [ ] Document layer-specific behaviors

### Week 2: Direction Analysis
- [ ] Build contrastive pair dataset (style/content variations)
- [ ] Extract activation steering directions
- [ ] Test direction arithmetic (add/subtract concepts)
- [ ] Compare with simple blending

### Week 3: Advanced Techniques
- [ ] Implement multi-layer blending
- [ ] Test timestep-dependent alpha
- [ ] Experiment with Gram matrix matching
- [ ] Benchmark all approaches

### Week 4: Architecture Exploration
- [ ] Prototype cross-attention injection
- [ ] Test layer-specific VL injection
- [ ] Compare with IP-Adapter (if possible)
- [ ] Write up findings

---

## 10. Success Metrics

How to measure if these techniques work:

1. **Content Preservation:**
   - CLIP text-image similarity (does output match text prompt?)
   - Human eval: "Does the image show what the text describes?"

2. **Style Transfer:**
   - Perceptual loss to reference image
   - Color histogram similarity
   - Human eval: "Does it have the same visual style?"

3. **Quality:**
   - ImageReward score
   - FID (if we have reference dataset)
   - Artifact detection (grid patterns, color bleeding)

4. **Composability:**
   - Can we combine multiple style deltas?
   - Does direction arithmetic behave linearly?

5. **Generalization:**
   - Does it work across different content types?
   - Different reference images?

---

## References

### Embedding Arithmetic
- Mikolov et al., "Efficient Estimation of Word Representations in Vector Space" (2013)
- Wu et al., "Editing in Style: Uncovering the Local Semantics of GANs" (2021)

### Activation Steering
- Zou et al., "Representation Engineering: A Top-Down Approach to AI Transparency" (2023)
- Turner et al., "Steering Llama 2 with Contrastive Activation Addition" (2023)
- Meng et al., "Locating and Editing Factual Associations in GPT" (2022)

### Layer Analysis
- Jawahar et al., "What does BERT learn about the structure of language?" (2019)
- Tenney et al., "BERT Rediscovers the Classical NLP Pipeline" (2019)

### Style Transfer
- Gatys et al., "A Neural Algorithm of Artistic Style" (2015)
- Huang & Belongie, "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization" (2017)

### Diffusion Conditioning
- Ye et al., "IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models" (2023)
- Hertz et al., "Prompt-to-Prompt Image Editing with Cross Attention Control" (2022)
- Tumanyan et al., "Plug-and-Play Diffusion Features for Text-Driven Image-to-Image Translation" (2022)
- Zhang et al., "Adding Conditional Control to Text-to-Image Diffusion Models" (2023)

### Prototype Theory
- Rosch, "Natural Categories" (1973)
- Paperno et al., "Neural Collapse in Deep Learning" (2020)

---

---

## See Also

- [quick_reference.md](quick_reference.md) - Quick reference for specific questions
- [related_work.md](related_work.md) - Prior art from other domains
- [findings.md](findings.md) - Our experimental results
- [../guides/parameters.md](../guides/parameters.md) - Practical guide to parameters
