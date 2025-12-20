last updated: 2025-12-19 (Bagel investigation + training requirements analysis)

# Z-Image Omni Research Findings

## Executive Summary

Z-Image Omni (diffusers PR 12857) adds image conditioning to Z-Image Turbo using SigLIP2 vision encoder. After extensive analysis, **the siglip_embedder and siglip_refiner components require trained weights that have NOT been released**. The architecture is understood and implemented, but omni mode produces noise without these weights.

## Table of Contents

1. [Background](#background)
2. [Architecture Analysis](#architecture-analysis)
3. [Implementation Work](#implementation-work)
4. [Initialization Experiments](#initialization-experiments)
5. [Key Findings](#key-findings)
6. [Open Questions](#open-questions)
7. [Files and Resources](#files-and-resources)

---

## Background

### What is Z-Image Omni?

Z-Image Omni extends Z-Image Turbo with **image conditioning** - the ability to use reference images to guide generation. This is implemented via a SigLIP2 vision encoder that extracts image embeddings which are then processed and concatenated with text embeddings in the transformer.

### Source Materials

- **Diffusers PR 12857**: https://github.com/huggingface/diffusers/pull/12857
- **RuoyiDu's Commit** (Z-Image dev): https://github.com/RuoyiDu/diffusers/commit/3435ba2a9f686e8adff8dbaaf5ed2746c2bd0327
- **Local PR Copy**: `./coderef/diffusers/src/diffusers/`

### Key Assertion

The user confirmed: **"its the same zimage turbo model"** - Z-Image Omni uses the identical Z-Image Turbo weights for the core DiT, with added siglip components.

---

## Architecture Analysis

See detailed breakdown: [diffusers_pr_12857_breakdown.md](./diffusers_pr_12857_breakdown.md)

### Dual Encoder System

```
Text Input                    Reference Image
    |                              |
Qwen3-4B                     SigLIP2 Vision Model
(~/Storage/Qwen3-4B)         (google/siglip2-so400m-patch14-384)
    |                              |
2560-dim embeddings          1152-dim embeddings
    |                              |
cap_embedder (trained)       siglip_embedder (NEW - needs training)
    |                              |
    +-------> 3840-dim <-----------+
                |
        Z-Image Turbo DiT
                |
          Generated Image
```

### New Transformer Components

From `transformer_z_image.py` in PR 12857:

```python
# Config parameter
siglip_feat_dim = 1152  # Default fallback (line 855)

# New components (require trained weights)
siglip_embedder = nn.Sequential(
    RMSNorm(siglip_feat_dim, eps=1e-5),
    nn.Linear(siglip_feat_dim, 3840, bias=True)
)
siglip_refiner = nn.ModuleList([
    ZImageTransformerBlock(2000 + layer_id, ...)
    for layer_id in range(2)
])
siglip_pad_token = nn.Parameter(torch.empty((1, 3840)))
```

### Processing Flow

1. **Dual Timestep Embeddings**: `t_noisy` for target, `t_clean=1.0` for conditioning
2. **Noise Mask**: Per-token mask distinguishing noisy (target) vs clean (condition) tokens
3. **Unified Sequence**: `[cap_feats] + [x_latents] + [siglip_feats]`
4. **Main Layers**: 30 transformer layers with noise-mask-based modulation

### SigLIP2 Model Specification

| Property | Value |
|----------|-------|
| Model | `google/siglip2-so400m-patch14-384` |
| Hidden Dimension | **1152** |
| Input Size | 384x384 |
| Patch Size | 14x14 |
| Patches | 27x27 = 729 tokens |

The 1152 dimension comes from PR line 855:
```python
sig_pad_dim = self.config.siglip_feat_dim or 1152
```

---

## Implementation Work

### SigLIP Encoder

Created `siglip_encoder.py` - a wrapper class for SigLIP2 vision model:

```python
class SigLIPEncoder:
    def __init__(self, model_path=None, device="cpu", dtype=torch.float32):
        if model_path is None:
            model_path = str(Path.home() / "Storage/google_siglip2-so400m-patch14-384")

        full_model = AutoModel.from_pretrained(model_path)
        self.model = full_model.vision_model.to(device)
        self.hidden_size = self.model.config.hidden_size  # 1152

    def encode(self, images):
        # Returns (H_patches, W_patches, hidden_size) spatial grid
        ...
```

**Key Fix**: The model type is "siglip" (not "siglip2"), requiring `AutoModel` then `.vision_model` extraction.

### Test Files

| File | Purpose | Status |
|------|---------|--------|
| `test_siglip_encoding.py` | SigLIP embedding extraction | Working |
| `test_omni_pipeline.py` | Minimal omni forward pass | Working |
| `test_full_omni_generation.py` | End-to-end generation | Runs, but omni produces noise |

### Tensor Format Fixes

The PR uses unconventional tensor formats:
- Latents: `(C, F, H, W)` not `(B, C, H, W)`
- F=1 for single-frame images

Fixed VAE decode:
```python
# latents is (C, F, H, W) = (16, 1, 32, 32)
latents_for_decode = latents.squeeze(1).unsqueeze(0)  # -> (1, 16, 32, 32)
```

---

## Initialization Experiments

Since trained siglip weights are not available, we tried multiple initialization strategies:

### 1. Random Initialization

```python
for name, param in transformer.named_parameters():
    if "siglip" in name:
        if param.dim() >= 2:
            torch.nn.init.xavier_uniform_(param)
        else:
            torch.nn.init.zeros_(param)
```

**Result**: Pure noise output

### 2. Statistics-Matched Initialization

Analyzed cap_embedder output statistics and initialized siglip_embedder to produce similar distributions:

```python
# cap_embedder output stats: mean≈0, std≈2.7
# Initialize to match
scale = 2.7 / (1152 ** 0.5)  # Account for input dim
nn.init.normal_(linear.weight, mean=0, std=scale)
```

**Result**: Still noise output

### 3. Cap_embedder Subset Initialization

Used first 1152 dimensions of cap_embedder's weight matrix:

```python
# siglip_embedder Linear: (1152 -> 3840)
# cap_embedder Linear: (2560 -> 3840)
siglip_linear.weight.data = cap_linear.weight[:, :1152].clone()
siglip_linear.bias.data = cap_linear.bias.clone()
```

**Result**: Still noise output

### Why Initialization Fails

The siglip components are **trained end-to-end** with the model. They learn:
1. How to project 1152-dim SigLIP features to 3840-dim transformer space
2. How to refine these features in the siglip_refiner (2 layers)
3. The proper siglip_pad_token for variable-length sequences

Without training, the projections produce embeddings that don't align with the text embedding space, causing the DiT to produce noise.

---

## Key Findings

### 1. Architecture is Understood

The Z-Image Omni architecture is fully documented in [diffusers_pr_12857_breakdown.md](./diffusers_pr_12857_breakdown.md). Key insights:
- Same Z-Image Turbo DiT (521 parameters, NO siglip weights)
- Dual encoder: Qwen3-4B (text) + SigLIP2-so400m (vision)
- Dual timestep embeddings for clean/noisy token distinction
- Per-token noise mask for conditioning

### 2. Trained Weights Required

Z-Image Turbo checkpoint analysis (521 keys):
```
cap_embedder.0.weight, cap_embedder.1.weight, cap_embedder.1.bias
context_refiner (2 layers)
noise_refiner (2 layers)
cap_pad_token, x_pad_token

NO siglip_embedder
NO siglip_refiner
NO siglip_pad_token
```

The siglip components require trained weights that **have not been released**.

### 3. Basic Mode Works

When omni mode is disabled (no reference images), the pipeline produces correct output using existing Z-Image Turbo weights. This confirms our implementation is correct.

### 4. SigLIP2 Encoder Works

Our SigLIP encoder correctly extracts 1152-dim embeddings:
```
Embedding shape: torch.Size([27, 27, 1152])
Hidden size: 1152
Mean: -0.0234
Std: 0.4892
```

### 5. Default Dimension is 1152

From PR line 855:
```python
sig_pad_dim = self.config.siglip_feat_dim or 1152
```

This indicates they use SigLIP2-so400m-patch14-384 (1152 dim), not the base model (768 dim).

---

## Open Questions

### 1. Different SigLIP2 Variant?

Could there be a fine-tuned SigLIP2 model we're missing? The PR uses stock `google/siglip2-so400m-patch14-384` but perhaps:
- Custom fine-tuned weights exist
- A different model checkpoint is used

**Available SigLIP2 so400m Models** (from HuggingFace):

| Model | Hidden Dim | Resolution | Notes |
|-------|-----------|------------|-------|
| [google/siglip2-so400m-patch14-384](https://huggingface.co/google/siglip2-so400m-patch14-384) | 1152 | 384x384 | **Likely candidate** |
| [google/siglip2-so400m-patch14-224](https://huggingface.co/google/siglip2-so400m-patch14-224) | 1152 | 224x224 | Lower resolution |
| [google/siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384) | 1152 | 384x384 | SigLIP v1 (not v2) |
| google/siglip2-so400m-patch16-naflex | 1152 | Variable | NaFlex variant |

All so400m models have 1152 hidden dimensions, matching the PR's default. The PR likely uses the stock `siglip2-so400m-patch14-384` model.

**Evidence against custom model**: The PR code shows standard model loading with no special weights.

### 2. When Will Weights Release?

The diffusers PR discussion doesn't mention a timeline for omni weight release. The PR focuses on code architecture, suggesting weights may come later.

### 3. Training Data Requirements?

To train siglip components ourselves, we would need:
- Paired (image, text, target) training data
- The Z-Image training recipe
- Significant compute resources

---

## Files and Resources

### Created Files

| File | Description |
|------|-------------|
| `experiments/siglip/README.md` | Overview and architecture summary |
| `experiments/siglip/diffusers_pr_12857_breakdown.md` | **Detailed PR analysis** |
| `experiments/siglip/siglip_encoder.py` | SigLIP2 encoder wrapper |
| `experiments/siglip/test_siglip_encoding.py` | Encoder unit tests |
| `experiments/siglip/test_omni_pipeline.py` | Minimal pipeline test |
| `experiments/siglip/test_full_omni_generation.py` | End-to-end generation test |
| `experiments/siglip/findings.md` | This document |

### External Resources

| Resource | URL |
|----------|-----|
| Diffusers PR 12857 | https://github.com/huggingface/diffusers/pull/12857 |
| RuoyiDu's Commit | https://github.com/RuoyiDu/diffusers/commit/3435ba2a9f686e8adff8dbaaf5ed2746c2bd0327 |
| SigLIP2 Model | https://huggingface.co/google/siglip2-so400m-patch14-384 |

### Local Model Paths

| Model | Path |
|-------|------|
| Z-Image Turbo | `~/Storage/Tongyi-MAI_Z-Image-Turbo` |
| SigLIP2-so400m | `~/Storage/google_siglip2-so400m-patch14-384` |
| Qwen3-4B | `~/Storage/Qwen3-4B` |

---

## Alternative Approach: Qwen3-VL

An important insight from our previous VL research: **Qwen3-VL may be a better path than waiting for SigLIP2 weights**.

### Dimension Comparison

| Vision Encoder | Output Dim | Projection Needed | Status |
|----------------|-----------|-------------------|--------|
| SigLIP2-so400m | 1152 | Yes (1152 → 3840) | **Missing trained weights** |
| Qwen3-VL-4B | 2560 | **No** - matches text encoder | **Working code exists** |

### Why Qwen3-VL May Be Better

1. **Direct compatibility**: 2560 dim matches Qwen3-4B text embeddings exactly
2. **No projection layer**: Eliminates the siglip_embedder problem entirely
3. **Working implementation**: Already proven in `experiments/qwen3_vl/`
4. **Known-good settings**: Layer -6, `text_tokens_only=False`, alpha 0.3-0.5

### Previous Qwen3-VL Results

From `internal/research/qwen3_vl_integration.md`:
- Layer -6 produces cleaner results than default -2
- `text_tokens_only=False` is critical (preserves image tokens)
- Alpha 0.3-0.5 provides good style transfer without corrupting content
- Higher alpha (0.7-1.0) reconstructs reference scene

### The Irony

We dismissed Qwen3-VL because Z-Image Omni uses SigLIP2. But:
- SigLIP2 requires trained siglip_embedder weights we don't have
- Qwen3-VL already works with existing code
- Qwen3-VL has better dimension matching (no projection needed)

See: `experiments/qwen3_vl/README.md` for full documentation.

---

## Bagel Connector Investigation (The Hail Mary)

### Background

The Z-Image developer starred ByteDance's Bagel repo. Bagel uses the same SigLIP2 vision encoder (1152 hidden dim). We investigated whether Bagel's trained connector weights could be adapted for Z-Image Omni.

### Dimension Analysis

| Component | Bagel | Z-Image | Gap |
|-----------|-------|---------|-----|
| SigLIP input | 1152 | 1152 | Match |
| Connector output | 3584 | 3840 | +256 dims |
| LLM hidden | 3584 (Qwen2.5-7B) | 2560 (Qwen3-4B) | Different |

### Adaptation Strategy

```python
# Load Bagel's connector fc1 weights
fc1_weight = f.get_tensor('connector.fc1.weight')  # (3584, 1152)
fc1_bias = f.get_tensor('connector.fc1.bias')      # (3584,)

# Pad to Z-Image dimensions
pad_size = 3840 - 3584  # 256
pad_weight = torch.randn(pad_size, 1152) * (std * 0.1)  # Small noise
adapted_weight = torch.cat([fc1_weight, pad_weight], dim=0)  # (3840, 1152)
```

### Results

**Good news**: Coherent images generated (not noise!)
- Text-only baseline: Perfect red apple, realistic portraits
- With reference image: Produces valid images

**Bad news**: **Style transfer doesn't work**
- Reference style is completely ignored
- Output is always photorealistic regardless of anime/oil painting/pixel art reference
- See: `experiments/results/bagel_connector_v2/`

### Root Cause Analysis

**Bagel's connector was trained for UNDERSTANDING, not GENERATION:**

1. **Architecture mismatch**: Bagel's MLPconnector projects SigLIP features into an LLM space for VL understanding tasks (image captioning, VQA)

2. **Training objective**: Bagel uses joint CE (next token prediction) + MSE (velocity prediction) loss. The connector is trained so the LLM can "understand" images, not to extract stylistic features.

3. **Information encoded**: The connector learns "what's in the image" (semantic content) not "how it looks" (style/appearance)

4. **Generation path is separate**: Bagel's actual image generation uses VAE latent tokens through MoE experts, not SigLIP directly:
   ```python
   # Understanding: SigLIP -> connector -> LLM hidden states
   # Generation: VAE latents -> vae2llm -> MoE experts -> llm2vae -> velocity
   ```

### Bagel Architecture Deep Dive

From `coderef/Bagel/modeling/bagel/bagel.py`:

```python
# Vision understanding path (connector)
packed_vit_token_embed = self.vit_model(...)  # SigLIP forward
packed_vit_token_embed = self.connector(packed_vit_token_embed)  # Project to LLM
# -> These go into KV cache for the LLM to attend to

# Image generation path (separate)
x_t = self.vae2llm(packed_latent) + timestep_embeds + pos_embed
output = self.language_model.forward_inference(mode="gen", ...)  # MoE routes to gen experts
v_t = self.llm2vae(output)  # Predict velocity for flow matching
```

### Key Insight

The connector is a "translator" that helps the LLM understand images. It does NOT preserve style/appearance information needed for generation. Z-Image's siglip_embedder needs to do something fundamentally different - preserve visual style features that influence the diffusion process.

### Why Coherent Images But No Style?

The adapted Bagel weights:
1. Project SigLIP features to a valid embedding space
2. siglip_refiner (initialized from context_refiner) processes them
3. DiT receives valid conditioning but ignores the "style" because:
   - The features encode semantic content ("woman with hair") not style ("anime style")
   - The DiT wasn't trained to extract style from these features
   - siglip_refiner wasn't trained to preserve style through refinement

### Files Created

| File | Description |
|------|-------------|
| `test_bagel_connector.py` | Initial weight adaptation and testing |
| `test_bagel_generation.py` | Full generation pipeline with Bagel weights |
| `sweep_bagel_connector.py` | Parameter sweeps (scale factors, styles) |
| `sweep_bagel_v2.py` | Improved version with scaling and init fixes |
| `bagel_adapted_siglip_embedder.pt` | Saved adapted weights |

### Results Directory

- `experiments/results/bagel_connector/` - v1 sweep results
- `experiments/results/bagel_connector_v2/` - v2 sweep results (with fixes)

---

## Conclusions

1. **Architecture**: Z-Image Omni's architecture is fully understood and documented
2. **Implementation**: Pipeline runs end-to-end, basic mode works correctly
3. **Blocker**: siglip_embedder/siglip_refiner require trained weights not yet released
4. **No Workaround**: Random/heuristic initialization produces noise - proper training required
5. **Bagel Investigation**: Adapter weights produce coherent images but no style transfer (wrong training objective)
6. **Alternative**: Qwen3-VL provides working vision conditioning without missing weights
7. **Next Steps**: Consider Qwen3-VL path, train custom adapter, or wait for official omni weight release

### Paths Forward

| Approach | Effort | Likelihood | Notes |
|----------|--------|------------|-------|
| Wait for official weights | Low | Unknown | May never release |
| Train custom adapter | High | Medium | Need paired data + compute |
| Use Qwen3-VL | Low | **High** | Already working in experiments |
| Extract from Bagel MoE | Medium | Low | Complex architecture |

---

## Parallel to Qwen3-VL Experiments

### The Core Problem is the Same

Both SigLIP/Bagel and Qwen3-VL were trained for **understanding**, not **generation**:

| Model | Training Objective | What it Learned |
|-------|-------------------|-----------------|
| Qwen3-VL | VL understanding (captioning, VQA) | Content + style entangled |
| Bagel connector | VL understanding (captioning, VQA) | Content only, style lost |
| Z-Image Omni (official) | Style conditioning (presumed) | Style disentangled (unavailable) |

### Different Symptoms, Same Root Cause

**Qwen3-VL:**
- Style info IS present in embeddings
- But it's **entangled** with content
- Blending transfers style BUT ALSO corrupts content
- Higher alpha = more style = more corruption
- Best results: layer -6, alpha 0.3-0.5, but content still degrades

**Bagel connector:**
- Style info is **NOT preserved**
- Connector was trained to encode "what's in the image" (semantic)
- Blending has no effect on style at all
- Output ignores reference completely
- Coherent images, zero style transfer

### Visual Comparison

```
Qwen3-VL (alpha=0.5):
  Reference: Anime girl
  Prompt: "Homer Simpson"
  Output: Homer with anime-ish style BUT face corrupted

Bagel connector (any scale):
  Reference: Anime girl
  Prompt: "Portrait of woman"
  Output: Photorealistic woman (reference completely ignored)
```

### The Fundamental Insight

Neither approach answers the question: **"Extract visual style features that can condition a diffusion model without affecting semantic content"**

This is exactly what:
- IP-Adapter was trained to do
- Z-Image Omni's siglip components were (presumably) trained to do
- We cannot do without training data + compute

### Why Qwen3-VL Works Better Than Bagel

Qwen3-VL gives *some* style transfer because the style info is at least **present** (even if entangled with content). Bagel's connector strips style info out entirely during training - it only learned to encode semantic content for VL understanding tasks.

| Aspect | Qwen3-VL | Bagel Connector |
|--------|----------|-----------------|
| Style info present | Yes (entangled) | No (lost) |
| Content info present | Yes | Yes |
| Style transfer possible | Partial (with corruption) | None |
| Best use case | Low-alpha blending | None for style |

---

## What We Need to Succeed

### 1. Training Data

The key dataset structure - triplets of:
```
(style_reference, content_prompt, target_image)
```

Where `target_image` has the **style** of `style_reference` but the **content** of `content_prompt`.

**Potential sources:**

| Source | Pros | Cons |
|--------|------|------|
| Artist portfolios | Real style consistency | Limited scale, copyright |
| Synthetic pairs | Unlimited scale | Style transfer quality varies |
| Same-scene different-style | Clear style difference | Hard to find at scale |
| Video frames + style | Consistent content | Temporal artifacts |

**Minimum viable dataset:** ~10-50k triplets with diverse styles

### 2. Training Objective

```python
# Frozen: SigLIP encoder, most of DiT
# Trainable: siglip_embedder, siglip_refiner (maybe DiT LoRA)

def training_step(style_ref, prompt, target):
    # Extract style features
    siglip_feats = siglip_encoder(style_ref)
    siglip_feats = siglip_embedder(siglip_feats)
    siglip_feats = siglip_refiner(siglip_feats)

    # Encode prompt
    text_feats = text_encoder(prompt)

    # Standard diffusion loss on target
    noise = torch.randn_like(target_latent)
    noisy = scheduler.add_noise(target_latent, noise, t)
    pred = dit(noisy, t, text_feats, siglip_feats)

    loss = F.mse_loss(pred, noise)  # or velocity
    return loss
```

### 3. What to Train

| Option | Params | Compute | Risk |
|--------|--------|---------|------|
| siglip_embedder only | ~4.4M | Low | May not be enough capacity |
| embedder + refiner | ~25M | Medium | Good balance |
| + DiT LoRA | ~50M | Higher | Best quality, harder to tune |

**Missing weights breakdown:**

| Component | Shape | Params |
|-----------|-------|--------|
| `siglip_embedder[0]` (RMSNorm) | (1152,) | ~1K |
| `siglip_embedder[1]` (Linear) | (3840, 1152) + (3840,) | ~4.4M |
| `siglip_refiner[0]` | Full transformer layer | ~12M |
| `siglip_refiner[1]` | Full transformer layer | ~12M |
| `siglip_pad_token` | (1, 3840) | ~4K |
| **Total** | | **~25-30M** |

### 4. Compute Requirements

**Rough estimates:**
- 1x A100 (80GB): ~1-2 days for minimal adapter
- 4x A100: ~1 day for full siglip components
- 8x A100: Hours, or include DiT LoRA

**RTX 4090 (24GB) feasibility:**
- siglip_embedder with gradient checkpointing: Feasible
- Full siglip_refiner: Needs CPU offload or smaller batch
- DiT LoRA: Probably not feasible

### 5. Evaluation Metrics

| Metric | What it Measures |
|--------|------------------|
| Style similarity (CLIP/DINO) | Output vs reference style match |
| Content preservation (CLIP) | Output vs prompt content match |
| FID | Overall image quality |
| ImageReward | Aesthetic quality |
| Human eval | Subjective style transfer success |

### 6. The Bootstrapping Problem

To generate training data, we need a style transfer model. To train a style transfer model, we need training data.

**Workarounds:**
1. Use existing neural style transfer to generate pairs
2. Use artist datasets where style is consistent across works
3. Use ControlNet + style LoRAs to generate synthetic pairs
4. Fine-tune on small manually curated set first, then scale

### 7. Minimum Viable Training Path

```
1. Curate ~10k style triplets (synthetic or artist-based)
2. Freeze: SigLIP, text encoder, DiT
3. Train: siglip_embedder + siglip_refiner (~25M params)
4. Loss: Standard diffusion MSE/velocity loss
5. Compute: 1-2 days on single A100 (or longer on 4090)
6. Evaluate: Style similarity vs content preservation tradeoff
```

### 8. Is Training Worth It?

**Pros of training custom adapter:**
- Full control over style conditioning behavior
- Could achieve clean style transfer without content corruption
- Reusable for any style reference

**Cons:**
- Significant data curation effort
- Compute cost (A100 time or days on 4090)
- Risk of not matching official quality

**Alternative:** Qwen3-VL already gives *some* style transfer at alpha 0.3-0.5 with layer -6, just with the entanglement tradeoff. For many use cases, this may be sufficient without any training.

---

## Appendix: Z-Image Turbo Checkpoint Keys

All 521 keys from the transformer checkpoint:

```
cap_embedder.0.weight
cap_embedder.1.weight, cap_embedder.1.bias
cap_pad_token
context_refiner.0.* (attention, ffn, norms)
context_refiner.1.* (attention, ffn, norms)
final_layer.adaLN_modulation.1.weight, .bias
final_layer.linear.weight, .bias
layers.0-29.* (30 main transformer layers)
noise_refiner.0.* (attention, ffn, norms, adaln)
noise_refiner.1.* (attention, ffn, norms, adaln)
t_embedder.mlp.0.weight, .bias
t_embedder.mlp.2.weight, .bias
x_embedder.proj.weight, .bias
x_pad_token

NO siglip_* keys
```
