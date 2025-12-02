# Z-Image DiffSynth-Studio Analysis Report

**Date**: 2025-12-01
**Source**: `/coderef/DiffSynth-Studio` (WIP branch)
**Purpose**: Identify new Z-Image features/patterns for potential adoption

---

## Executive Summary

Analysis of DiffSynth-Studio's Z-Image implementation reveals several important differences from our ComfyUI implementation. The most significant finding is that **DiffSynth always enables thinking tokens with empty tags**, even when no thinking content is provided. Other key differences include no system prompt by default, explicit embedding layer selection, and specific attention mask handling.

---

## 1. Relevant Files Found

| File | Description |
|------|-------------|
| `diffsynth/models/z_image_text_encoder.py` | Qwen3 model wrapper (42 lines) |
| `diffsynth/models/z_image_dit.py` | S3-DiT implementation (622 lines) |
| `diffsynth/pipelines/z_image.py` | Full inference pipeline (258 lines) |
| `diffsynth/diffusion/flow_match.py` | Flow match scheduler with Z-Image template |
| `diffsynth/configs/model_configs.py` | Model hash/class mappings |
| `examples/z_image/model_inference/` | Basic inference example |
| `examples/z_image/model_inference_low_vram/` | FP8 + disk offload example |
| `examples/z_image/model_training/` | LoRA training scripts |

---

## 2. Key Technical Findings

### 2.1 Text Encoder Configuration

```python
# From z_image_text_encoder.py
config = Qwen3Config(**{
    "hidden_size": 2560,           # Matches our implementation
    "num_hidden_layers": 36,       # Matches our implementation
    "num_attention_heads": 32,
    "num_key_value_heads": 8,      # GQA - 8 KV heads
    "intermediate_size": 9728,
    "head_dim": 128,
    "max_position_embeddings": 40960,
    "rope_theta": 1000000,
    "vocab_size": 151936,
})
```

DiffSynth uses raw `Qwen3Model` (not `Qwen3ForCausalLM`), extracting `hidden_states[-2]` for embeddings.

### 2.2 Prompt Encoding - CRITICAL FINDING

```python
# From z_image.py - lines 162-196
messages = [
    {"role": "user", "content": prompt_item},
]
prompt_item = pipe.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,    # <-- ALWAYS TRUE
)

# Embedding extraction
prompt_embeds = pipe.text_encoder(
    input_ids=text_input_ids,
    attention_mask=prompt_masks,
    output_hidden_states=True,
).hidden_states[-2]          # Second-to-last hidden layer

# Filter by attention mask (remove padding)
for i in range(len(prompt_embeds)):
    embeddings_list.append(prompt_embeds[i][prompt_masks[i]])
```

**Critical finding**: DiffSynth uses `enable_thinking=True` in `apply_chat_template()`. This generates:
```
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
<think>

</think>

```

The thinking tags are ALWAYS included (empty) even when no thinking content is provided.

### 2.3 DiT Architecture Parameters

```python
# From z_image_dit.py
class ZImageDiT(nn.Module):
    def __init__(
        self,
        in_channels=16,          # 16-channel latents
        dim=3840,                # Hidden dimension
        n_layers=30,             # Main transformer layers
        n_refiner_layers=2,      # Pre-processing refinement
        n_heads=30,
        n_kv_heads=30,
        cap_feat_dim=2560,       # Caption feature dim (matches text encoder)
        rope_theta=256.0,
        axes_dims=[32, 48, 48],  # RoPE dimensions
        axes_lens=[1024, 512, 512],
    ):
```

Architecture notes:
- Separate `noise_refiner` (2 layers) and `context_refiner` (2 layers) before main transformer
- 30 main transformer blocks
- AdaLN modulation with timestep embedding
- 3D RoPE position encoding (time, height, width)
- Sequences padded to multiple of 32

### 2.4 Flow Match Scheduler

```python
# From flow_match.py
@staticmethod
def set_timesteps_z_image(num_inference_steps=100, denoising_strength=1.0, shift=None):
    shift = 3 if shift is None else shift     # Default shift = 3
    sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    timesteps = sigmas * num_train_timesteps
    return sigmas, timesteps
```

Uses shift=3 (vs Flux's shift=3, Wan's shift=5).

### 2.5 Model Function - Timestep Handling

```python
# From z_image.py
def model_fn_z_image(dit, latents, timestep, prompt_embeds, **kwargs):
    timestep = (1000 - timestep) / 1000    # Inverted timestep
    model_output = dit(latents, timestep, prompt_embeds, ...)[0][0]
    model_output = -model_output           # Negated output
    return model_output
```

**Note**: Model output is NEGATED and timestep is inverted.

### 2.6 VAE Configuration

```python
# Uses Flux VAE with modification
"extra_kwargs": {"use_conv_attention": False},  # Different from standard Flux
```

- Flux VAE architecture with `use_conv_attention=False`
- 16-channel latent space
- 8x spatial compression

---

## 3. Implementation Differences

| Aspect | DiffSynth | Our Implementation |
|--------|-----------|-------------------|
| **Thinking tokens** | Always `enable_thinking=True` with empty tags | User controls `add_think_block` toggle |
| **System prompt** | None (user message only) | Full system prompt support with templates |
| **Chat template** | Uses tokenizer's `apply_chat_template()` | Manual formatting with special tokens |
| **Embedding extraction** | `hidden_states[-2]` explicitly | Uses ComfyUI CLIP abstraction |
| **Padding handling** | Removes padding with attention mask | Not explicitly shown |
| **Max sequence length** | 512 tokens | Not explicitly limited |

---

## 4. Recommendations

### 4.1 HIGH PRIORITY - Default Thinking Tokens

DiffSynth always enables thinking with empty tags. Consider:
- Changing default `add_think_block` to `True`
- Or documenting that empty thinking blocks produce different results

### 4.2 MEDIUM PRIORITY - Verify Embedding Layer

DiffSynth uses `hidden_states[-2]` explicitly. We should verify ComfyUI's CLIP abstraction extracts from the same layer.

### 4.3 MEDIUM PRIORITY - Padding/Mask Handling

DiffSynth filters embeddings by attention mask to remove padding. Investigate if ComfyUI handles this automatically.

### 4.4 LOW PRIORITY - DiffSynth-Compatible Template

Consider adding a "minimal" template that matches DiffSynth's bare format (user message only + empty thinking tags).

---

## 5. Capabilities Gap Analysis

### DiffSynth Has, We Don't:

1. **LoRA Training Support**
   - Target modules: `"to_q,to_k,to_v,to_out.0,w1,w2,w3"`
   - Pre-trained adapter for distillation quality: `zimage_turbo_training_adapter_v1.safetensors`

2. **Low VRAM Inference**
   - FP8 computation with CPU/disk offload
   - Dynamic VRAM management

3. **Direct Distillation Training**
   - Task type `direct_distill`
   - FlowMatch SFT Loss

### We Have, DiffSynth Doesn't:

1. **Multi-turn conversations** - `ZImageTurnBuilder`
2. **Template system** - File-based templates with thinking content
3. **System prompt support** - DiffSynth uses none by default
4. **Strip key quotes** - JSON prompt filtering
5. **LLM Output Parser** - Generic bridge for LLM outputs

---

## 6. Important Training Caveat

From DiffSynth LoRA training comments:

> Z-Image-Turbo is a distilled model. After training, it loses its distillation-based acceleration capability, leading to degraded generation quality at fewer inference steps. This issue can be mitigated by using a pre-trained LoRA model to assist the training process.

**Implication**: Fine-tuning Z-Image-Turbo without the training adapter LoRA will degrade 8-step inference quality.

---

## 7. Action Items

### Completed
- [x] Change `add_think_block` default to `True` (matches DiffSynth)
- [x] Update documentation with reference implementation notes
- [x] Add tooltip explaining DiffSynth/diffusers default behavior
- [x] **Attention mask filtering** - Added `filter_padding` parameter (default: True)
  - Pattern: `embeddings_list.append(prompt_embeds[i][prompt_masks[i]])`
  - Added to ZImageTextEncoder, ZImageTextEncoderSimple, ZImageTurnBuilder
  - Verified both diffusers (HuggingFace) AND DiffSynth filter padding - ComfyUI is the outlier
  - Default True matches reference implementations; set False for stock ComfyUI behavior

### High Priority (Quality Impact)
2. [ ] **Max sequence length = 512** - Hard limit in both DiffSynth and diffusers
   - Add parameter to tokenization
3. [ ] **Resolution alignment = 16 pixels** - Z-Image uses 16, not 32
   - Z-Image VAE uses 8x scale with vae_scale_factor*2=16
   - Our 32-pixel may unnecessarily constrain resolutions

### Medium Priority
4. [ ] Verify ComfyUI CLIP uses `hidden_states[-2]` for Qwen3
5. [ ] Create "diffsynth_compatible" template (no system prompt, empty thinking)
6. [ ] Document flow match scheduler shift=3 for Z-Image

### Low Priority / Research
7. [ ] Investigate CFG normalization/truncation options
8. [ ] Document distillation preservation requirements for fine-tuning

---

## 8. Additional Deep Dive Findings (December 2024)

### 8.1 Attention Mask Filtering Pattern

DiffSynth filters embeddings by attention mask after encoding:

```python
# From z_image.py lines 191-196
prompt_masks = text_inputs.attention_mask.to(device).bool()
prompt_embeds = pipe.text_encoder(
    input_ids=text_input_ids,
    attention_mask=prompt_masks,
    output_hidden_states=True,
).hidden_states[-2]

# Filter embeddings by attention mask - removes padding
embeddings_list = []
for i in range(len(prompt_embeds)):
    embeddings_list.append(prompt_embeds[i][prompt_masks[i]])
```

**Impact**: Returns variable-length embedding tensors (list), not fixed-size padded batch.

### 8.2 Max Sequence Length

Both DiffSynth and diffusers enforce 512-token limit:
```python
max_sequence_length: int = 512
```

This aligns with DiT position embedding limits (`axes_lens: [1024, 512, 512]`).

### 8.3 Resolution Alignment

Z-Image uses 16-pixel alignment:
```python
# DiffSynth z_image.py line 24-25
height_division_factor=16, width_division_factor=16
```

Our documentation incorrectly states 32-pixel alignment for Z-Image. The 32 applies to Qwen-Image-Edit (different VAE).

### 8.4 SEQ_MULTI_OF = 32

DiT internally pads sequences to multiples of 32 for efficient attention:
```python
# z_image_dit.py line 15
SEQ_MULTI_OF = 32
```

This is DiT-internal, not encoder concern.

### 8.5 Flow Match Scheduler

Z-Image uses shift=3 (same as FLUX):
```python
# flow_match.py line 105
shift = 3 if shift is None else shift
```

ComfyUI's FlowMatchEulerDiscreteScheduler should handle this correctly.

---

## 9. Technical Reference

### Resolution Requirements
- Height/width divisible by 16 (vs our 32-pixel VAE alignment)

### RoPE Configuration (DiT)
```python
axes_dims=[32, 48, 48],    # Dimensions per axis
axes_lens=[1024, 512, 512], # Maximum positions per axis
rope_theta=256.0           # Much lower than text model's 1000000
```

### Sequence Padding
```python
SEQ_MULTI_OF = 32  # Pad to multiple of 32
```
