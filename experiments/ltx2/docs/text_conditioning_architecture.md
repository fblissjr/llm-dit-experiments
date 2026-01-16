# LTX-2 Text Conditioning Architecture

Last updated: 2026-01-16

## Overview

This document provides a detailed analysis of the LTX-2 text conditioning pathway, from Gemma-3 12B layer outputs through to DiT cross-attention. This is essential for understanding where and how to implement layer routing experiments.

## Architecture Flow

```
Prompt Text
    ↓
[Tokenizer: LTXVGemmaTokenizer]
    ↓
[Gemma-3 12B: 49 Decoder Layers]
    ↓
outputs.hidden_states: tuple[50]  # embedding + 49 layers
    ↓
[Feature Extractor: Normalization + Linear Projection]
    ↓
[Embeddings Connector: 1D Transformer with 128 Thinking Tokens]
    ↓
[Caption Projection: PixArtAlpha 2-layer MLP]
    ↓
[DiT Cross-Attention in each of 48 transformer blocks]
```

---

## 1. Gemma-3 12B Layer Structure

### File Location
`coderef/LTX-2/packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/base_encoder.py`

### Layer Architecture
Each of Gemma's 49 decoder layers (`Gemma3DecoderLayer`) contains:

```python
class Gemma3DecoderLayer:
    # Pre-normalization + Self-Attention
    input_layernorm: RMSNorm
    self_attn: Gemma3Attention  # Multi-head self-attention

    # Pre-normalization + Feed-Forward
    post_attention_layernorm: RMSNorm
    mlp: Gemma3MLP  # Feed-forward network
```

### Forward Pass Structure
```python
# Standard pre-norm transformer block:
x = x + self_attn(input_layernorm(x))
x = x + mlp(post_attention_layernorm(x))
```

### Hidden States Output
When calling `model(output_hidden_states=True)`:

```python
outputs.hidden_states: tuple[50, torch.Tensor]
    # [0]: Embedding layer output (before any decoder layers)
    # [1-49]: Output from each of 49 decoder layers
```

**Important for routing**: Each element in `hidden_states` is the **final output** of that layer (after both attention and FFN). There are no separate sub-layer outputs for attention vs. FFN.

### Dimensions
- Hidden state shape: `[batch, seq_len, 3840]`
- 49 decoder layers total
- Hidden dim: 3840

### Code Reference
```python
# Line 69-72 in base_encoder.py
outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
projected = self._run_feature_extractor(
    hidden_states=outputs.hidden_states, attention_mask=attention_mask, padding_side=padding_side
)
```

---

## 2. Feature Extractor: Layer Aggregation + Projection

### File Location
`coderef/LTX-2/packages/ltx-core/src/ltx_core/text_encoders/gemma/feature_extractor.py`

### Architecture

```python
class GemmaFeaturesExtractorProjLinear(torch.nn.Module):
    def __init__(self):
        # Linear projection: (3840 * 49) → 3840
        self.aggregate_embed = torch.nn.Linear(3840 * 49, 3840, bias=False)
```

### Processing Pipeline

#### Step 1: Stack Layers
```python
# Line 43 in base_encoder.py
encoded_text_features = torch.stack(hidden_states, dim=-1)
# Input: tuple[50] of [batch, seq, 3840]
# Output: [batch, seq, 3840, 50]
```

**Note**: All 50 hidden states (embedding + 49 layers) are stacked, but typically only layers 1-49 are used for the projection.

#### Step 2: Normalize Per Layer
```python
# Lines 157-213 in base_encoder.py - _norm_and_concat_padded_batch()
# For each layer independently:
# 1. Compute masked mean (respecting attention mask)
# 2. Compute masked min/max for range
# 3. Normalize: normed = 8 * (x - mean) / (range + eps)
# 4. Flatten to [batch, seq, 3840*49]
```

**Normalization formula**:
```python
mean = masked.sum(dim=(1,2), keepdim=True) / (seq_len * dim + eps)
range = max - min
normed = 8 * (encoded_text - mean) / (range + eps)
```

This normalization:
- Is computed **per batch item**
- Is computed **per layer** (independently for each of the 49 layers)
- Uses masked statistics (ignores padding tokens)
- Produces a roughly [-4, 4] range

#### Step 3: Linear Projection
```python
# Line 22 in feature_extractor.py
output = self.aggregate_embed(normed)  # [batch, seq, 3840*49] → [batch, seq, 3840]
```

### Key Insight for Routing

The `GemmaFeaturesExtractorProjLinear.aggregate_embed` layer is a **learnable weighted sum** across all 49 layers:

```python
# Conceptually:
weight_matrix: [3840*49, 3840]
# Can be reshaped to: [49, 3840, 3840]
# Each of 49 "slices" mixes one layer's features

output = sum(
    weight_matrix[layer_idx] @ hidden_states[layer_idx]
    for layer_idx in range(49)
)
```

**Routing opportunity**: Replace this fixed linear projection with:
- Dynamic routing weights per layer
- Attention-based routing
- Content-conditional routing

### Code Reference
```python
# Lines 40-51 in base_encoder.py
def _run_feature_extractor(
    self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, padding_side: str = "right"
) -> torch.Tensor:
    encoded_text_features = torch.stack(hidden_states, dim=-1)
    encoded_text_features_dtype = encoded_text_features.dtype

    sequence_lengths = attention_mask.sum(dim=-1)
    normed_concated_encoded_text_features = _norm_and_concat_padded_batch(
        encoded_text_features, sequence_lengths, padding_side=padding_side
    )

    return self.feature_extractor_linear(normed_concated_encoded_text_features.to(encoded_text_features_dtype))
```

---

## 3. Embeddings Connector: 128 Thinking Tokens

### File Location
`coderef/LTX-2/packages/ltx-core/src/ltx_core/text_encoders/gemma/embeddings_connector.py`

### Architecture

```python
class Embeddings1DConnector(torch.nn.Module):
    def __init__(
        self,
        attention_head_dim: int = 128,
        num_attention_heads: int = 30,
        num_layers: int = 2,  # 2 transformer layers
        num_learnable_registers: int = 128,  # "Thinking tokens"
        ...
    ):
        self.inner_dim = 30 * 128 = 3840
        self.learnable_registers = Parameter([128, 3840])
        self.transformer_1d_blocks = ModuleList[
            _BasicTransformerBlock1D (x2)
        ]
```

### Processing Flow

#### Step 1: Replace Padding with Thinking Tokens
```python
# Lines 131-157
# 1. Extract non-padded tokens
# 2. Pad with learnable_registers instead of zeros
# 3. Flip so registers are at the start (left side)
```

This is **critical**: The 128 learnable registers are **prepended** to the prompt embeddings, replacing padding tokens.

```
Before:
[PAD PAD PAD token1 token2 ... tokenN]  # 1024 total

After:
[reg1 reg2 ... reg128 token1 token2 ... tokenN]  # Non-padded tokens + registers
```

#### Step 2: 1D Bidirectional Transformer (2 layers)
```python
# Lines 189-190
for block in self.transformer_1d_blocks:
    hidden_states = block(hidden_states, attention_mask=attention_mask, pe=freqs_cis)
```

Each `_BasicTransformerBlock1D` contains:
- `attn1`: Self-attention with RoPE positional embeddings
- `ff`: Feed-forward network (MLP)
- Pre-norm architecture (RMS norm)

#### Step 3: Final Normalization
```python
# Line 192
hidden_states = rms_norm(hidden_states)
```

### Purpose of Thinking Tokens

The 128 learnable registers serve as:
1. **Compressed semantic representation**: Allow bidirectional attention to create global context
2. **Padding replacement**: Remove meaningless padding, replace with learnable features
3. **Cross-modal bridge**: Prepare text embeddings for cross-attention with video/audio

These tokens are **trained** alongside the DiT, not frozen from Gemma.

### Code Reference
```python
# Lines 159-194 in embeddings_connector.py
def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if self.num_learnable_registers:
        hidden_states, attention_mask = self._replace_padded_with_learnable_registers(hidden_states, attention_mask)

    # Compute RoPE positional embeddings
    indices_grid = torch.arange(hidden_states.shape[1], dtype=torch.float32, device=hidden_states.device)
    freqs_cis = precompute_freqs_cis(...)

    # Apply transformer blocks
    for block in self.transformer_1d_blocks:
        hidden_states = block(hidden_states, attention_mask=attention_mask, pe=freqs_cis)

    hidden_states = rms_norm(hidden_states)

    return hidden_states, attention_mask
```

---

## 4. Caption Projection: PixArtAlpha 2-Layer MLP

### File Location
`coderef/LTX-2/packages/ltx-core/src/ltx_core/model/transformer/text_projection.py`

### Architecture

```python
class PixArtAlphaTextProjection(torch.nn.Module):
    def __init__(self, in_features=3840, hidden_size=4096, out_features=4096):
        self.linear_1 = Linear(3840, 4096)
        self.act_1 = GELU(approximate="tanh")
        self.linear_2 = Linear(4096, 4096)

    def forward(self, caption):
        hidden_states = self.linear_1(caption)      # [batch, seq, 3840] → [batch, seq, 4096]
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)  # [batch, seq, 4096] → [batch, seq, 4096]
        return hidden_states
```

### Purpose

Projects text embeddings from Gemma's 3840 dim to DiT's 4096 dim cross-attention space. This is the **final transformation** before text becomes conditioning for the DiT.

### Integration Point

This projection is applied **inside the DiT model** during preprocessing:

```python
# Lines 73-84 in transformer_args.py
def _prepare_context(
    self,
    context: torch.Tensor,
    x: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Prepare context for transformer blocks."""
    batch_size = x.shape[0]
    context = self.caption_projection(context)  # Apply PixArt projection
    context = context.view(batch_size, -1, x.shape[-1])
    return context, attention_mask
```

### Code Reference
```python
# Lines 132-135 in model.py
self.caption_projection = PixArtAlphaTextProjection(
    in_features=caption_channels,  # 3840
    hidden_size=self.inner_dim,    # 4096
)
```

---

## 5. DiT Cross-Attention Integration

### File Location
`coderef/LTX-2/packages/ltx-core/src/ltx_core/model/transformer/transformer.py`

### Transformer Block Structure

Each of the 48 DiT blocks (`BasicAVTransformerBlock`) contains:

```python
class BasicAVTransformerBlock:
    # Video path
    attn1: Attention  # Self-attention (video tokens attend to video)
    attn2: Attention  # Cross-attention (video tokens attend to TEXT)
    ff: FeedForward   # Feed-forward network

    # Audio path (if enabled)
    audio_attn1: Attention  # Self-attention
    audio_attn2: Attention  # Cross-attention (audio tokens attend to TEXT)
    audio_ff: FeedForward

    # Audio-video cross-attention (if both enabled)
    audio_to_video_attn: Attention
    video_to_audio_attn: Attention
```

### Cross-Attention Flow (Video Only)

```python
# Line 165 in transformer.py
vx = vx + self.attn2(
    rms_norm(vx),
    context=video.context,        # Text embeddings from caption_projection
    mask=video.context_mask
)
```

**Key points**:
- `video.context`: Output of `caption_projection` (shape: `[batch, seq, 4096]`)
- This is where text **conditions** the video generation
- Cross-attention: `Q=video_tokens, K=text_embeddings, V=text_embeddings`

### Attention Mechanism

```python
# Cross-attention in Attention module:
# Q from video latents: [batch, num_video_patches, 4096]
# K, V from text: [batch, text_seq_len, 4096]

attn_output = softmax(Q @ K^T / sqrt(d)) @ V
```

Each of the 48 blocks applies this cross-attention, allowing text to influence video generation at multiple levels of abstraction.

### Code Reference
```python
# Lines 156-166 in transformer.py
if run_vx:
    vshift_msa, vscale_msa, vgate_msa = self.get_ada_values(
        self.scale_shift_table, vx.shape[0], video.timesteps, slice(0, 3)
    )
    if not perturbations.all_in_batch(PerturbationType.SKIP_VIDEO_SELF_ATTN, self.idx):
        norm_vx = rms_norm(vx, eps=self.norm_eps) * (1 + vscale_msa) + vshift_msa
        v_mask = perturbations.mask_like(PerturbationType.SKIP_VIDEO_SELF_ATTN, self.idx, vx)
        vx = vx + self.attn1(norm_vx, pe=video.positional_embeddings) * vgate_msa * v_mask

    vx = vx + self.attn2(rms_norm(vx, eps=self.norm_eps), context=video.context, mask=video.context_mask)
```

---

## Key Integration Points for Experiments

### 1. Layer Routing in Feature Extractor

**Location**: `GemmaFeaturesExtractorProjLinear.aggregate_embed`
**File**: `coderef/LTX-2/packages/ltx-core/src/ltx_core/text_encoders/gemma/feature_extractor.py:22`

**Current**: Fixed linear projection combining all 49 layers
**Opportunity**: Replace with:
- Learned routing weights (MLP that outputs per-layer weights)
- Attention-based routing (cross-attend to a query token)
- Content-conditional routing (different weights per prompt)

### 2. Thinking Token Integration

**Location**: `Embeddings1DConnector._replace_padded_with_learnable_registers`
**File**: `coderef/LTX-2/packages/ltx-core/src/ltx_core/text_encoders/gemma/embeddings_connector.py:131-157`

**Current**: Fixed 128 learnable registers prepended to text
**Opportunity**:
- Make register count dynamic based on prompt complexity
- Learn register initialization from layer routing signal
- Add routing-aware attention in the 1D transformer

### 3. Cross-Attention Context

**Location**: `BasicAVTransformerBlock.attn2`
**File**: `coderef/LTX-2/packages/ltx-core/src/ltx_core/model/transformer/transformer.py:165`

**Current**: Fixed text context for all 48 blocks
**Opportunity**:
- Different layer routing per DiT block depth
- Early blocks use early Gemma layers, late blocks use late Gemma layers
- Adaptive routing based on timestep/noise level

---

## Shape Tracing Reference

```python
# End-to-end shape flow:

# 1. Tokenization
input_text: str
input_ids: [1, seq_len]

# 2. Gemma forward
outputs.hidden_states: tuple[50, [1, seq_len, 3840]]

# 3. Feature extraction
stacked: [1, seq_len, 3840, 50]
normalized: [1, seq_len, 3840*50]
projected: [1, seq_len, 3840]

# 4. Embeddings connector (thinking tokens)
with_registers: [1, 128+seq_len, 3840]  # Registers prepended
after_1d_transformer: [1, 128+seq_len, 3840]

# 5. Caption projection
caption_projected: [1, 128+seq_len, 4096]

# 6. DiT cross-attention (in each of 48 blocks)
video_tokens: [1, num_patches, 4096]
cross_attn(Q=video_tokens, K=caption_projected, V=caption_projected)
```

---

## References

### Primary Source Files

1. **Base encoder**: `coderef/LTX-2/packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/base_encoder.py`
2. **Feature extractor**: `coderef/LTX-2/packages/ltx-core/src/ltx_core/text_encoders/gemma/feature_extractor.py`
3. **Embeddings connector**: `coderef/LTX-2/packages/ltx-core/src/ltx_core/text_encoders/gemma/embeddings_connector.py`
4. **Video encoder**: `coderef/LTX-2/packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/video_only_encoder.py`
5. **DiT model**: `coderef/LTX-2/packages/ltx-core/src/ltx_core/model/transformer/model.py`
6. **DiT transformer blocks**: `coderef/LTX-2/packages/ltx-core/src/ltx_core/model/transformer/transformer.py`
7. **Transformer args**: `coderef/LTX-2/packages/ltx-core/src/ltx_core/model/transformer/transformer_args.py`
8. **Text projection**: `coderef/LTX-2/packages/ltx-core/src/ltx_core/model/transformer/text_projection.py`

### Next Steps

For implementing layer routing experiments:
1. Start with `feature_extractor.py` - replace the linear projection
2. Extract routing weights/patterns from trained model
3. Analyze which layers contribute most to different prompt types
4. Design dynamic routing strategies based on findings
