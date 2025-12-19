last updated: 2025-12-19

# Diffusers PR 12857: Z-Image Omni Analysis

## Overview

Z-Image Omni adds **image conditioning** to Z-Image Turbo using SigLIP2 vision encoder.
The base DiT architecture remains unchanged - they add new components for vision processing.

## Architecture

```
                           Reference Image(s)
                                  |
                        +---------+---------+
                        |                   |
                   SigLIP2              VAE Encode
                        |                   |
                  768/1152-dim          16-ch latents
                        |                   |
               siglip_embedder      x_embedder (existing)
                        |                   |
                  3840-dim              3840-dim
                        |                   |
                siglip_refiner        noise_refiner
                   (2 layers)           (2 layers)
                        |                   |
                        v                   v
                        +-------+   +-------+
                                |   |
Text Prompt                     |   |
     |                          |   |
 Qwen3-4B                       |   |
     |                          |   |
 2560-dim                       |   |
     |                          |   |
cap_embedder                    |   |
     |                          |   |
 3840-dim                       |   |
     |                          |   |
context_refiner                 |   |
 (2 layers)                     |   |
     |                          |   |
     +----------+---------------+   |
                |                   |
                v                   v
    Unified Sequence: [cap_feats] + [x_latents] + [siglip_feats]
                            |
                    30 Transformer Layers
                            |
                      Final Layer
                            |
                     Unpatchify
                            |
                    Target Image
```

## Key Components Added

### 1. Transformer Config (`transformer_z_image.py:389`)

```python
siglip_feat_dim=None  # Set to 768 or 1152 to enable
```

When `siglip_feat_dim` is set, these components are created:

```python
# Projection: SigLIP hidden dim -> transformer dim
self.siglip_embedder = nn.Sequential(
    RMSNorm(siglip_feat_dim, eps=1e-5),
    nn.Linear(siglip_feat_dim, 3840, bias=True)
)

# 2-layer refiner (same arch as context_refiner)
self.siglip_refiner = nn.ModuleList([
    ZImageTransformerBlock(2000 + layer_id, ...)
    for layer_id in range(2)
])

# Padding token for variable-length sequences
self.siglip_pad_token = nn.Parameter(torch.empty((1, 3840)))
```

### 2. Pipeline Components (`pipeline_z_image_omni.py`)

```python
class ZImageOmniPipeline:
    def __init__(
        self,
        scheduler,
        vae,
        text_encoder,        # Qwen3-4B
        tokenizer,
        transformer,         # Z-Image DiT
        siglip,              # Siglip2VisionModel (NEW)
        siglip_processor,    # Siglip2ImageProcessorFast (NEW)
    )
```

## Processing Flow

### Step 1: Prompt Encoding (lines 207-267)

The prompt format changes based on whether images are provided:

**Without images:**
```
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
```

**With N condition images:**
```
<|im_start|>user
<|vision_start|><|vision_end|>  # Repeated N times with separators
<|vision_start|>...<|vision_end|>{prompt}<|im_end|>
<|im_start|>assistant
<|vision_start|><|vision_end|><|im_end|>  # Output marker
```

Each segment is tokenized separately and embeddings extracted with `hidden_states[-2]`.

### Step 2: Image Latent Encoding (lines 293-312)

Reference images are VAE-encoded to latents:

```python
image_latent = (
    vae.encode(image).latent_dist.mode()[0] - vae.config.shift_factor
) * vae.config.scaling_factor
# Shape: (1, 16, H/8, W/8)
```

### Step 3: SigLIP Embedding (lines 314-334)

```python
def prepare_siglip_embeds(self, images, batch_size, device, dtype):
    siglip_embeds = []
    for image in images:
        siglip_inputs = self.siglip_processor(images=[image], return_tensors="pt")
        shape = siglip_inputs.spatial_shapes[0]  # (H_patches, W_patches)

        hidden_state = self.siglip(**siglip_inputs).last_hidden_state
        # Shape: (1, num_patches, hidden_dim)

        # Reshape to spatial grid
        hidden_state = hidden_state[:, :shape[0] * shape[1]]
        hidden_state = hidden_state.view(shape[0], shape[1], hidden_dim)
        # Shape: (H_patches, W_patches, hidden_dim)

        siglip_embeds.append(hidden_state)
    return siglip_embeds
```

### Step 4: Transformer Forward - Omni Mode (lines 1062-1300)

#### 4a. Dual Timestep Embeddings (lines 1078-1083)

```python
# Create two timestep embeddings:
t_noisy = t_embedder(t * t_scale)       # For noisy (target) tokens
t_clean = t_embedder(1.0 * t_scale)     # For clean (condition) tokens
```

This tells the model which tokens are being denoised vs. which are conditioning.

#### 4b. Combine Latents (lines 1086-1087)

```python
# Condition latents + target latent
x = [cond_latents[i] + [target_latent[i]] for i in range(bsz)]

# Create noise mask: 0 = condition (clean), 1 = target (noisy)
image_noise_mask = [[0] * num_cond_images + [1] for i in range(bsz)]
```

#### 4c. Patchify and Embed (line 1105)

All three streams are patchified:
- **x**: Image latents -> x_embedder -> 3840-dim
- **cap_feats**: Text -> cap_embedder -> 3840-dim
- **siglip_feats**: Vision -> siglip_embedder -> 3840-dim

#### 4d. Refiner Layers

Each stream goes through its own 2-layer refiner:
- **noise_refiner**: Processes image latents (with adaln modulation)
- **context_refiner**: Processes text (no modulation)
- **siglip_refiner**: Processes vision (no modulation)

#### 4e. Unified Sequence (lines 1219-1252)

```python
unified = [cap_feats] + [x_latents] + [siglip_feats]
unified_noise_mask = [cap_mask] + [x_mask] + [siglip_mask]
```

#### 4f. Main Layers with Noise Mask (lines 1266-1289)

The 30 transformer layers use `noise_mask` to apply different modulation:
- `noise_mask=1` (noisy): Uses `adaln_noisy` (actual timestep)
- `noise_mask=0` (clean): Uses `adaln_clean` (t=1, no noise)

```python
for layer in self.layers:
    unified = layer(
        unified,
        attn_mask,
        freqs_cis,
        noise_mask=unified_noise_mask_tensor,
        adaln_noisy=t_noisy,
        adaln_clean=t_clean,
    )
```

#### 4g. Unpatchify (lines 1295-1298)

Only the target image tokens are extracted and unpatchified:

```python
# x_pos_offsets marks where target image tokens are in unified sequence
x = self.unpatchify(unified, x_size, patch_size, f_patch_size, x_pos_offsets)
```

## Noise Mask Implementation (ZImageTransformerBlock)

The block uses noise_mask to select which adaln to apply per-token:

```python
def forward(self, x, attn_mask, freqs_cis, adaln_input=None,
            noise_mask=None, adaln_noisy=None, adaln_clean=None):

    if noise_mask is not None:
        # Per-token modulation selection
        # noise_mask=1 -> adaln_noisy (being denoised)
        # noise_mask=0 -> adaln_clean (conditioning)
        adaln = torch.where(
            noise_mask.unsqueeze(-1) == 1,
            adaln_noisy.unsqueeze(1),
            adaln_clean.unsqueeze(1)
        )
    else:
        adaln = adaln_input
```

## SigLIP Model Details

Default fallback dimension: **1152** (line 855)

```python
sig_pad_dim = self.config.siglip_feat_dim or 1152
```

This suggests they use a SigLIP model with 1152 hidden dimensions (likely SigLIP-Large or similar).

## Data Flow Summary

| Stage | Text | Image Latents | SigLIP |
|-------|------|---------------|--------|
| Input | Prompt string | Reference images | Reference images |
| Encode | Qwen3-4B hidden[-2] | VAE encode | SigLIP2 last_hidden |
| Dim | 2560 | 16 channels | 768/1152 |
| Embed | cap_embedder | x_embedder | siglip_embedder |
| Dim | 3840 | 3840 | 3840 |
| Refine | context_refiner | noise_refiner | siglip_refiner |
| Layers | 2 | 2 | 2 |
| Modulation | None | adaln (dual) | None |
| In Unified | cap_feats | x (cond + target) | siglip_feats |
| Noise Mask | 0 (clean) | 0/1 (cond/target) | 0 (clean) |

## CFG Implementation (lines 643-667)

For classifier-free guidance:
1. Duplicate inputs: positive + negative
2. Run both through transformer
3. CFG: `pred = pos + scale * (pos - neg)`

Condition images and SigLIP embeddings are cloned for negative path:
```python
negative_condition_latents = [[lat.clone() for lat in batch] for batch in condition_latents]
negative_condition_siglip_embeds = [[se.clone() for se in batch] for batch in condition_siglip_embeds]
```

## What We Need to Implement

1. **SigLIP Encoder**: Load Siglip2VisionModel, extract embeddings
2. **Transformer Changes**: Add siglip_embedder, siglip_refiner, siglip_pad_token
3. **Pipeline Changes**:
   - New prompt format with vision tokens
   - prepare_siglip_embeds()
   - prepare_image_latents()
   - Dual timestep handling
   - Noise mask per-token

## File References

- `coderef/diffusers/src/diffusers/pipelines/z_image/pipeline_z_image_omni.py`
- `coderef/diffusers/src/diffusers/models/transformers/transformer_z_image.py`
