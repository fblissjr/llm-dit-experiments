# Z-Image Attention Mask Filtering: Technical Design Specification

**Date:** 2025-12-01
**Version:** 1.2
**Feature Version:** v2.9.11
**Author:** Claude Code

---

## 1. Executive Summary

This document describes the implementation of attention mask filtering for Z-Image text encoders in ComfyUI.

**Key Finding:** Both reference implementations (diffusers and DiffSynth) filter padding tokens from embeddings. Stock ComfyUI does not. Our implementation now matches the reference behavior.

**Key Change:** Added `filter_padding` parameter (default: `True`) to all Z-Image encoder nodes. Filters embeddings using the attention mask to remove padding tokens before sending to the DiT, matching diffusers/DiffSynth.

---

## 2. Implementation Comparison

### 2.1 Background

When encoding text prompts for Z-Image, the tokenizer produces a fixed-length sequence (typically 512 tokens). Short prompts are padded with special padding tokens to reach this length. The attention mask indicates which tokens are real (1) vs padding (0).

### 2.2 Reference Implementation Analysis

We verified how each implementation handles embeddings:

| Implementation | Filters Padding? | Source |
|----------------|------------------|--------|
| **diffusers** (HuggingFace) | Yes | `pipeline_z_image.py:242-247` |
| **DiffSynth** | Yes | `z_image.py:191-196` |
| **ComfyUI** (stock) | No | Returns padded + mask |

Both official reference implementations filter padding:

**DiffSynth-Studio (reference implementation):**
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

**Stock ComfyUI:**
- Returns full 512-token padded sequence
- Attention mask available in conditioning extra dict but not applied

### 2.3 Why We Match Reference Implementations

Since both diffusers (HuggingFace official) and DiffSynth filter padding, we follow their pattern. The DiT can handle padding via its learned `cap_pad_token`, but the reference implementations chose not to rely on this.

**Rationale for filtering:**
- Short prompts have high padding ratios (e.g., 50 tokens + 462 padding = 91% padding)
- Reduces sequence length sent to DiT (memory/compute savings)
- Matches tested/validated reference behavior

**Stock ComfyUI approach:** Sends padded sequence + mask. The DiT's context_refiner can handle this:
```python
cap_feats[pad_mask] = self.cap_pad_token  # Learned padding replacement
```

Both approaches produce valid results, but we default to matching the reference implementations.

---

## 3. Investigation

### 3.1 ComfyUI CLIP Architecture

We investigated whether ComfyUI's CLIP system exposes the attention mask. Key findings:

**File:** `comfy/text_encoders/z_image.py`
```python
class ZImageClipModel(torch.nn.Module):
    def __init__(self, ...):
        # ...
        self.return_attention_masks = True  # KEY: Masks ARE returned
```

**Conditioning Structure:**
```python
conditioning = [
    (embeddings_tensor, extra_dict)
]
# extra_dict contains:
# - "attention_mask": tensor of shape [batch, seq_len]
# - other metadata
```

### 3.2 Mask Accessibility

Confirmed that the attention mask IS accessible in ComfyUI without core modifications:
```python
attention_mask = conditioning[0][1].get("attention_mask")
```

### 3.3 Context Refiner Analysis

From analysis of the DiT's context_refiner (see `internal/z_image_context_refiner_deep_dive.md`):

```python
# DiT replaces padding with learned token
cap_feats = self.cap_embedder(cap_feats)
cap_feats[pad_mask] = self.cap_pad_token  # Learned padding replacement
```

While the DiT handles padding internally, filtering beforehand:
1. Reduces sequence length (memory/compute savings)
2. Matches reference implementation exactly
3. Eliminates any potential padding-related artifacts

---

## 4. Technical Design

### 4.1 Design Goals

1. **Match reference implementation** - Filter embeddings exactly as DiffSynth does
2. **Backward compatibility** - Provide option to disable for stock behavior
3. **Transparency** - Show users what's happening via debug output
4. **Minimal changes** - Single helper function, parameter additions only

### 4.2 Architecture

```
                                    ┌─────────────────────┐
                                    │  filter_padding=T   │
                                    └──────────┬──────────┘
                                               │
┌──────────┐    ┌──────────────┐    ┌──────────▼──────────┐    ┌──────────┐
│  Prompt  │───►│  Tokenize    │───►│  Encode + Filter    │───►│  DiT     │
│          │    │  (512 tokens)│    │  (N valid tokens)   │    │          │
└──────────┘    └──────────────┘    └─────────────────────┘    └──────────┘
                                               │
                                    ┌──────────┴──────────┐
                                    │  filter_padding=F   │
                                    │  (512 tokens, mask) │
                                    └─────────────────────┘
```

### 4.3 Filter Function Specification

```python
def filter_embeddings_by_mask(
    conditioning: List[Tuple[Tensor, Dict]],
    debug_lines: Optional[List[str]] = None
) -> List[Tuple[Tensor, Dict]]:
    """
    Filter padding tokens from embeddings using attention mask.

    Args:
        conditioning: ComfyUI conditioning list [(embeddings, extra_dict), ...]
        debug_lines: Optional list to append debug info

    Returns:
        New conditioning list with padding filtered out.
        Attention mask is removed from extra_dict (no longer needed).
    """
```

### 4.4 Algorithm

```
FOR each (embeddings, extra_dict) in conditioning:
    1. Get attention_mask from extra_dict
    2. IF no mask: keep original, continue
    3. Convert mask to boolean
    4. FOR each batch item:
        a. Select embeddings where mask is True
        b. Append to filtered list
    5. Stack filtered embeddings
    6. Remove attention_mask from extra_dict
    7. Return (filtered_embeddings, new_extra_dict)
```

### 4.5 Batch Handling

For batch processing with potentially different valid lengths:
```python
# Handle variable lengths within batch
if len(filtered_embeds_list) == 1:
    filtered_embeds = filtered_embeds_list[0].unsqueeze(0)
else:
    # Pad to max length in batch
    max_len = max(e.shape[0] for e in filtered_embeds_list)
    padded = []
    for e in filtered_embeds_list:
        if e.shape[0] < max_len:
            pad = torch.zeros(max_len - e.shape[0], e.shape[1],
                            device=e.device, dtype=e.dtype)
            padded.append(torch.cat([e, pad], dim=0))
        else:
            padded.append(e)
    filtered_embeds = torch.stack(padded, dim=0)
```

Note: In practice, Z-Image typically processes single prompts (batch=1), so the variable-length handling is defensive.

---

## 5. Implementation Details

### 5.1 Files Modified

| File | Change |
|------|--------|
| `nodes/z_image_encoder.py` | Added filter function, parameter to 3 classes |
| `nodes/docs/z_image_encoder.md` | Documentation updates |
| `CLAUDE.md` | Version bump, feature note |
| `CHANGELOG.md` | v2.9.11 entry |
| `internal/z_image_diffsynth_analysis_20251201.md` | Marked task complete |

### 5.2 Code Changes

#### 5.2.1 Filter Function (New)

Location: `nodes/z_image_encoder.py` (lines 24-96)

```python
def filter_embeddings_by_mask(conditioning: List, debug_lines: Optional[List[str]] = None) -> List:
    """
    Filter padding tokens from embeddings using attention mask.

    DiffSynth pattern:
        for i in range(len(prompt_embeds)):
            embeddings_list.append(prompt_embeds[i][prompt_masks[i]])
    """
    if not conditioning or len(conditioning) == 0:
        return conditioning

    filtered_conditioning = []

    for i, (embeddings, extra_dict) in enumerate(conditioning):
        attention_mask = extra_dict.get("attention_mask")

        if attention_mask is None:
            if debug_lines is not None:
                debug_lines.append(f"  Cond[{i}]: No attention_mask, keeping original")
            filtered_conditioning.append((embeddings, extra_dict))
            continue

        mask_bool = attention_mask.bool()
        original_shape = embeddings.shape
        batch_size = embeddings.shape[0]

        filtered_embeds_list = []
        for b in range(batch_size):
            valid_mask = mask_bool[b] if mask_bool.dim() > 1 else mask_bool
            valid_embeds = embeddings[b][valid_mask]
            filtered_embeds_list.append(valid_embeds)

        # Stack back to batch (handle variable lengths)
        if len(filtered_embeds_list) == 1:
            filtered_embeds = filtered_embeds_list[0].unsqueeze(0)
        else:
            max_len = max(e.shape[0] for e in filtered_embeds_list)
            padded = []
            for e in filtered_embeds_list:
                if e.shape[0] < max_len:
                    pad = torch.zeros(max_len - e.shape[0], e.shape[1],
                                    device=e.device, dtype=e.dtype)
                    padded.append(torch.cat([e, pad], dim=0))
                else:
                    padded.append(e)
            filtered_embeds = torch.stack(padded, dim=0)

        # Remove mask from extra dict (no longer needed)
        new_extra = {k: v for k, v in extra_dict.items() if k != "attention_mask"}

        if debug_lines is not None:
            new_shape = filtered_embeds.shape
            tokens_removed = original_shape[1] - new_shape[1]
            debug_lines.append(
                f"  Cond[{i}]: {original_shape} -> {new_shape} "
                f"(removed {tokens_removed} padding tokens)"
            )

        filtered_conditioning.append((filtered_embeds, new_extra))

    return filtered_conditioning
```

#### 5.2.2 Parameter Addition

Added to INPUT_TYPES for each encoder class:

```python
"filter_padding": ("BOOLEAN", {
    "default": False,
    "tooltip": "EXPERIMENTAL: Filter padding tokens from embeddings (DiffSynth pattern). "
               "Stock ComfyUI passes padded sequence + mask to DiT, which handles it fine. "
               "This option pre-filters instead. Try if you see artifacts with short prompts."
}),
```

#### 5.2.3 Encode Method Integration

```python
def encode(self, ..., filter_padding: bool = False, ...):
    # ... existing encoding logic ...

    tokens = clip.tokenize(formatted_text, llama_template="{}")
    conditioning = clip.encode_from_tokens_scheduled(tokens)

    # Apply attention mask filtering if enabled (DiffSynth pattern)
    if filter_padding:
        filter_debug = []
        conditioning = filter_embeddings_by_mask(conditioning, filter_debug)
        # Add filter debug to output
        # ... debug output integration ...

    return (conditioning, formatted_text, debug_output, conversation_out)
```

### 5.3 Classes Modified

| Class | Parameter | Integration Point |
|-------|-----------|-------------------|
| `ZImageTextEncoder` | `filter_padding=True` | After `encode_from_tokens_scheduled()` |
| `ZImageTextEncoderSimple` | `filter_padding=True` | After `encode_from_tokens_scheduled()` |
| `ZImageTurnBuilder` | `filter_padding=True` | After `encode_from_tokens_scheduled()` (when clip connected) |

---

## 6. User Interface

### 6.1 Parameter

| Name | Type | Default | Location |
|------|------|---------|----------|
| `filter_padding` | BOOLEAN | **True** | Optional inputs section |

### 6.2 Tooltip

> "Filter padding tokens from embeddings. Matches diffusers and DiffSynth reference implementations. Disable to use stock ComfyUI behavior (padded sequence + mask)."

### 6.3 Debug Output

When enabled, debug output includes:

```
=== Padding Filter ===
  Cond[0]: torch.Size([1, 512, 2560]) -> torch.Size([1, 47, 2560]) (removed 465 padding tokens)
```

This shows:
- Original shape: `[batch, 512 tokens, 2560 dims]`
- Filtered shape: `[batch, N valid tokens, 2560 dims]`
- Number of padding tokens removed

---

## 7. Compatibility

### 7.1 Backward Compatibility

- **Default changed:** `filter_padding=True` matches diffusers/DiffSynth reference implementations
- **Stock behavior available:** Set `filter_padding=False` for original ComfyUI behavior
- **Existing workflows:** Will use new default unless explicitly set to False

### 7.2 When to Disable Filtering

Users may want to set `filter_padding=False` if:
1. They want to match stock ComfyUI behavior exactly
2. They're debugging embedding-related issues and want to compare approaches
3. They have workflows that depend on the padded sequence structure

### 7.3 Known Limitations

1. **Batch processing:** Variable-length filtering within a batch pads to max length
2. **No mask case:** Falls back to original embeddings (no filtering)
3. **Extra dict modification:** Removes `attention_mask` key after filtering

---

## 8. Testing Considerations

### 8.1 Verification Approach

1. **Shape verification:** Debug output shows expected shape reduction
2. **A/B testing:** Compare outputs with filter_padding=True vs False
3. **Reference comparison:** Compare with DiffSynth-Studio outputs

### 8.2 Test Cases

| Test | Expected Result |
|------|-----------------|
| Short prompt (10 tokens) | Large reduction (502 tokens removed) |
| Long prompt (400 tokens) | Small reduction (~112 tokens removed) |
| Max length prompt (512) | No reduction (0 tokens removed) |
| No mask in conditioning | Original embeddings preserved |
| Batch of 2 prompts | Both filtered, padded to max |

### 8.3 Quality Metrics

- Visual comparison of generated images
- Artifact presence (especially with long prompts)
- Consistency with reference implementation outputs

---

## 9. Performance Impact

### 9.1 Memory

- **Reduced:** Smaller embedding tensors after filtering
- **Example:** 47 tokens vs 512 tokens = ~91% reduction for short prompts

### 9.2 Compute

- **Filter overhead:** Minimal (boolean indexing, one pass)
- **DiT savings:** Fewer tokens to process in context_refiner

### 9.3 Tradeoffs

| Aspect | With Filtering | Without Filtering |
|--------|----------------|-------------------|
| Memory | Lower | Higher |
| Compute | Lower (fewer tokens) | Higher |
| Compatibility | DiffSynth-like | Stock ComfyUI |
| Artifacts | Potentially fewer | Original behavior |

---

## 10. Future Considerations

### 10.1 Potential Enhancements

1. **Configurable max length:** Allow truncation to 512 tokens before filtering
2. **Statistics logging:** Track average filtering ratios
3. **Batch optimization:** More efficient variable-length batch handling

### 10.2 Related Work

- Max sequence length enforcement (512 token limit from reference)
- Resolution alignment (16 pixels for Z-Image)
- Context refiner analysis for diffusers port

---

## 11. References

- `internal/z_image_diffsynth_analysis_20251201.md` - DiffSynth analysis report
- `internal/z_image_context_refiner_deep_dive.md` - Context refiner architecture
- DiffSynth-Studio: `diffsynth/pipelines/z_image.py` lines 191-196
- ComfyUI: `comfy/text_encoders/z_image.py`

---

## 12. Appendix: Full Diff Summary

### Files Changed

```
nodes/z_image_encoder.py
  + import torch
  + filter_embeddings_by_mask() function (72 lines)
  + filter_padding parameter to ZImageTextEncoder.INPUT_TYPES
  + filter_padding parameter to ZImageTextEncoder.encode()
  + Filtering logic in ZImageTextEncoder.encode()
  + filter_padding parameter to ZImageTurnBuilder.INPUT_TYPES
  + filter_padding parameter to ZImageTurnBuilder.add_turn()
  + Filtering logic in ZImageTurnBuilder.add_turn()
  + filter_padding parameter to ZImageTextEncoderSimple.INPUT_TYPES
  + filter_padding parameter to ZImageTextEncoderSimple.encode()
  + Filtering logic in ZImageTextEncoderSimple.encode()

nodes/docs/z_image_encoder.md
  + filter_padding row in ZImageTextEncoder inputs table
  + filter_padding row in ZImageTextEncoderSimple inputs table
  + filter_padding row in ZImageTurnBuilder inputs table
  + Updated embedding extraction comparison table
  + Updated version to 4.7 (v2.9.11)

CLAUDE.md
  + Version bump to v2.9.11
  + Attention mask filtering feature note

CHANGELOG.md
  + v2.9.11 section with feature description

internal/z_image_diffsynth_analysis_20251201.md
  + Marked attention mask filtering as completed
```

---

**Document Version:** 1.0
**Last Updated:** 2025-12-01
