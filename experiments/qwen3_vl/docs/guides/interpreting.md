# Interpreting VL Token Position Experiment Results

> **Last Updated:** 2025-12-12

**CRITICAL CONTEXT**: These are experimental approaches. Even with optimal settings, expect visible artifacts compared to pure text generation. This guide helps interpret results and understand artifact patterns.

**UPDATE 2025-12-12:** Per-dimension analysis showed VL text tokens have 0.999 correlation with Qwen3-4B, while image tokens have extreme outliers (617x in dim 396). Text tokens produce fewer artifacts than image tokens, but quality degradation persists.

## What to Look For

### Visual Quality Indicators

#### Prompt Adherence
**Less Corrupted:** Generated image recognizably contains the requested subject (with artifacts)
- Prompt: "Homer Simpson eating spaghetti"
- Less corrupted: Image shows character resembling Homer with spaghetti (visible quality loss)
- More corrupted: Subject is unrecognizable or completely different
- Baseline: Pure text generation (artifact-free reference)

#### Style Transfer
**Good:** Generated image matches reference visual style
- Reference: Cartoon house with flat colors, simple shapes
- Good: Output uses similar flat colors, simple geometric style
- Poor: Output is photorealistic or completely different aesthetic

#### Artifacts
**Common issues with VL conditioning:**
- Grid patterns (especially at high alpha)
- Color bleeding between objects
- Blurry or smeared regions
- Ghosting or double images

#### Coherence
**Good:** Image is internally consistent
- Objects have proper proportions
- Lighting is consistent
- No impossible geometry
- Clear subject-background separation

### Metadata Statistics

Every experiment generates `metadata.json` with these fields:

```json
{
  "vl_shape": [258, 2560],        // Sequence length, hidden dim
  "vl_std": 70.0,                 // Scaled VL std (target ~70.0 for Qwen3-4B match)
  "token_selection": "text_only", // Which tokens were used
  "normalization_mode": "global", // Normalization method used
  "blended_std": 70.0,            // Final embedding std
  "generation_time": 12.3,        // Seconds to generate
  "alpha": 1.0,                   // Blend ratio used
  "hidden_layer": -8              // VL layer extracted from (recommended)
}
```

**Key metrics (UPDATED 2025-12-12):**
- `vl_std` close to 70.0 → Good scaling, compatible with DiT (was 58.75, now 70.0)
- `blended_std` close to 70.0 → Final embeddings in expected range
- `hidden_layer`: -8 recommended (not -2)
- `normalization_mode`: "global" for text tokens, "per_dim" for image tokens
- `generation_time` → Performance (should be ~8-15s on RTX 4090)

## Experiment-Specific Analysis

### 1. vl_only_vs_qwen3 Results

**4 outputs to compare:**

| Output | What to Check |
|--------|--------------|
| `pure_qwen3_4b.png` | Baseline text quality - should be excellent |
| `pure_vl_text_tokens.png` | KEY TEST - does this match Qwen3-4B quality? |
| `pure_vl_full_seq.png` | Full VL - likely has artifacts |
| `blend_30_vl_70_qwen3.png` | Current best practice - known good |

**Reality check:**
- `pure_qwen3_4b` will likely look significantly better (no artifacts)
- `pure_vl_text_tokens` will have visible artifacts even though correlation is 0.999
- Compare artifact severity, not absolute quality

**What to assess:**
- Are VL text token artifacts tolerable for any use case?
- Is the quality gap vs pure text acceptable?
- Does blending meaningfully reduce artifacts?
- **Expected outcome:** Pure text remains superior for production use

### 2. vl_layer_by_token Results

**10 outputs (5 layers x 2 token types):**

Compare image tokens across layers:
- `image_tokens_layer-1.png`
- `image_tokens_layer-2.png` (current default)
- `image_tokens_layer-5.png`
- `image_tokens_layer-10.png`
- `image_tokens_layer-15.png`

Compare text tokens across layers:
- `text_tokens_layer-1.png`
- `text_tokens_layer-2.png` (current default)
- `text_tokens_layer-5.png`
- `text_tokens_layer-10.png`
- `text_tokens_layer-15.png`

**What to check:**
- **Style retention** (image tokens): Which layer preserves reference style best?
- **Prompt adherence** (text tokens): Which layer follows prompt most accurately?
- **Artifact presence**: Do deeper/shallower layers introduce artifacts?

**Hypotheses to test:**
- Image tokens: Layer -5 to -10 might be better than -2 (more visual, less abstract)
- Text tokens: Layer -10 to -15 might be better than -2 (better semantics, pre-SFT)
- Current -2 default might be suboptimal for both token types

**Success criteria:**
- If image tokens and text tokens have DIFFERENT optimal layers → Implement multi-layer extraction
- If -2 is best for both → Current default is correct
- If middle layers (-10 to -15) consistently win → Update default layer recommendation

### 3. vl_prompt_variations Results

**6 outputs testing different VL text inputs:**

| Output | VL Text Input | What It Tests |
|--------|--------------|--------------|
| `no_text_image_tokens.png` | None (pure visual) | Image tokens without text influence |
| `no_text_text_tokens.png` | None | Can text tokens work without VL text? |
| `with_prompt_image_tokens.png` | Exact prompt | Current approach, image tokens |
| `with_prompt_text_tokens.png` | Exact prompt | Current approach, text tokens |
| `with_desc_text_tokens.png` | "Describe what you see" | Generic description |
| `with_instruction_text_tokens.png` | "Generate..." | Instruction format |

**What to check:**
- Compare `with_prompt_text_tokens` vs `with_desc_text_tokens` vs `with_instruction_text_tokens`
  - Which VL text input produces best prompt adherence?
- Compare `no_text_text_tokens` vs `with_prompt_text_tokens`
  - Do text tokens need VL text input to work?

**Expected findings:**
- Exact prompt in VL → Best prompt adherence in text tokens (our current finding)
- No text in VL → Poor prompt adherence, pure visual influence
- Generic description → Middle ground

**Optimization:**
If different VL text inputs significantly affect quality → Document best practices

### 4. vl_intra_token_blend Results (Baseline Only)

**3 outputs (no actual blending yet):**

| Output | Tokens Used | Expected Result |
|--------|------------|-----------------|
| `baseline_vl_full.png` | All tokens | Compromise - some style, some prompt |
| `baseline_vl_image_only.png` | Image tokens | Strong style transfer, weak prompt |
| `baseline_vl_text_only.png` | Text tokens | Strong prompt adherence |

**What to check:**
- Does `baseline_vl_text_only` show BOTH prompt adherence AND some visual influence?
  - If yes → Text tokens alone might be sufficient (supports vl_only_vs_qwen3 findings)
  - If no → Need to blend with image tokens or Qwen3-4B

- Does `baseline_vl_image_only` show style transfer with poor prompt adherence?
  - If yes → Confirms image tokens carry style, not content
  - If no → Image tokens might carry mixed information

**Future (Phase 2):**
When intra-blending is implemented, look for:
- 30% image + 70% text → Style transfer WITH prompt adherence
- Optimal ratio that balances both

### 5. vl_double_conditioning Results (Baseline Only)

**2 outputs (no dual extraction yet):**

| Output | Configuration | Expected Result |
|--------|--------------|-----------------|
| `single_image_tokens.png` | Image tokens, no VL text | Pure style, no prompt |
| `single_text_tokens.png` | Text tokens, with prompt | Prompt adherence |

**What to check:**
- How different are these two outputs?
- Does `single_image_tokens` show zero prompt influence?
- Does `single_text_tokens` show zero style influence from reference?

**Future (Phase 3):**
When dual extraction is implemented:
- Extract both separately, blend at 30/70
- Hypothesis: Cleaner separation → Better blending quality

## Comparison Grid Analysis

Each experiment generates `comparison_grid.png` with all outputs side-by-side.

### What to Look For in Grid

1. **Progressive changes** - Do outputs change smoothly across parameter values?
   - Alpha sweep: Gradual style increase from 0.0 to 1.0
   - Layer sweep: Gradual abstraction changes across layers

2. **Sweet spots** - Which config looks best to your eye?
   - Mark the best-looking output
   - Note its parameters (alpha, layer, token_selection)

3. **Failure modes** - Which configs produce artifacts?
   - Grid patterns → Too much VL influence
   - Color bleeding → Incompatible embeddings
   - Blur → Poor layer selection

4. **Consistency** - Do similar configs produce similar results?
   - alpha=0.3 vs alpha=0.4 should be very similar
   - If wildly different → Unstable region, avoid

## Statistical Analysis (Advanced)

If you want quantitative metrics:

### Embedding Statistics
```python
# Load metadata
import json
with open('experiments/results/{experiment}/metadata.json') as f:
    data = json.load(f)

# Compare std values
for result in data['results']:
    print(f"{result['name']}: vl_std={result['vl_std']:.2f}, blended_std={result['blended_std']:.2f}")
```

**Ideal values:**
- `vl_std` ≈ 58.75 (after scaling)
- `blended_std` ≈ 58.75
- If far from 58.75 → Embeddings may be out-of-distribution

### Performance Metrics
```python
# Compare generation times
times = [(r['name'], r['generation_time']) for r in data['results']]
times.sort(key=lambda x: x[1])
print("Fastest to slowest:")
for name, time in times:
    print(f"  {name}: {time:.1f}s")
```

**Expected:**
- Similar times across configs (~8-15s on RTX 4090)
- If one is much slower → Check for issues

## Decision Tree

### After Running vl_only_vs_qwen3

```
Does pure_vl_text_tokens match pure_qwen3_4b quality?
│
├─ YES → ✅ VL can replace Qwen3-4B
│         Next: Test vl_prompt_variations to optimize VL text input
│         Then: Consider removing Qwen3-4B from pipeline
│
└─ NO  → ❌ Still need Qwen3-4B blending
          Next: Test vl_layer_by_token to optimize layer selection
          Then: Stick with current blending approach
```

### After Running vl_layer_by_token

```
Do image tokens and text tokens have different optimal layers?
│
├─ YES → ✅ Implement multi-layer extraction
│         Next: Extract image tokens from layer X, text tokens from layer Y
│         Then: Blend the two layer extractions
│
└─ NO  → ❌ Layer -2 is optimal for both
          Next: Keep current layer=-2 default
```

### After Running vl_prompt_variations

```
Does VL text input significantly affect text token quality?
│
├─ YES → ✅ Document best VL text input
│         Next: Use optimal VL text format in all future experiments
│         Then: Update documentation with recommended VL text
│
└─ NO  → ❌ Any VL text works
          Next: Keep using exact prompt (simplest approach)
```

## Recording Your Findings

After analyzing results, update these files:

1. **experiments/qwen3_vl/docs/research/findings.md**
   - Add new section with your findings
   - Include comparison images or links
   - State conclusions clearly

2. **experiments/qwen3_vl/docs/research/token_position.md**
   - Update "Expected Research Outcomes" with actual outcomes
   - Mark experiments as COMPLETED
   - Add "Actual Results" section

3. **Internal session log**
   - Document methodology
   - Record parameter sweeps tested
   - Note any surprises or unexpected results

## Sharing Results

If results are significant:

1. **Generate summary document**
   - Best configurations found
   - Quality comparisons (include images)
   - Recommendations for users

2. **Update main documentation**
   - CLAUDE.md if pipeline changes
   - README.md if new features
   - config.toml.example if new parameters

3. **Consider community sharing**
   - HuggingFace discussion post
   - Research notes blog post
   - GitHub issue with findings

## Common Patterns to Watch For

### Pattern 1: Layer Sweet Spot
- Layers -10 to -15 often outperform -1 to -5
- Matches "semantic sweet spot" hypothesis from research
- If you see this: Document and update default layer recommendation

### Pattern 2: Token Type Separation
- Image tokens → Style transfer, weak prompt
- Text tokens → Prompt adherence, weak style
- If clean separation: Intra-VL blending will work well (Phase 2)

### Pattern 3: Alpha Diminishing Returns
- Alpha 0.0 to 0.3: Gradual style increase
- Alpha 0.3 to 0.7: Minimal change
- Alpha 0.7 to 1.0: Artifacts increase
- If you see this: Optimal alpha is 0.2-0.4 range

### Pattern 4: VL Text Input Importance
- Exact prompt → Best prompt adherence
- Generic description → Middle ground
- No text → Pure style, zero prompt
- If you see this: VL text input is critical for text token quality

---

**Remember:** Visual inspection is the primary metric. Trust your eyes over statistics. If an output looks good, it IS good, regardless of std values or generation time.
