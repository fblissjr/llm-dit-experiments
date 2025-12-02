# heylookitsanllm Hidden States Endpoint Specification

## Overview

This document specifies a new endpoint for heylookitsanllm that returns raw hidden states from LLM models, enabling use as a text encoder backend for image generation models like Z-Image.

## Motivation

The existing `/v1/embeddings` endpoint pools token embeddings into a single vector per input. Z-Image and similar DiT-based models require:

1. **Raw hidden states** - Full sequence of embeddings, not pooled
2. **Specific layer** - Second-to-last layer (`hidden_states[-2]`), not final layer
3. **Attention mask filtering** - Return only non-padding tokens
4. **Chat template preservation** - The formatted prompt with special tokens

## Endpoint Specification

### `POST /v1/hidden_states`

Extract raw hidden states from a specific layer of the model.

#### Request

```json
{
    "input": "string",           // Required: Text to encode (with chat template applied)
    "model": "string",           // Required: Model ID
    "layer": -2,                 // Optional: Which layer to extract (default: -2)
    "max_length": 512,           // Optional: Max sequence length (default: 512)
    "return_attention_mask": false,  // Optional: Also return attention mask
    "encoding_format": "float"   // Optional: "float" (default) or "base64"
}
```

#### Response

**When `encoding_format: "float"` (default):**
```json
{
    "hidden_states": [[float, ...], ...],  // [seq_len, hidden_dim] as nested list
    "shape": [int, int],                    // [seq_len, hidden_dim]
    "model": "string",                      // Model ID used
    "layer": int,                           // Layer extracted
    "dtype": "string",                      // "float32", "float16", "bfloat16"
    "attention_mask": [int, ...]            // Optional: [seq_len] of 0/1
}
```

**When `encoding_format: "base64"`:**
```json
{
    "hidden_states": "base64_string",       // Base64-encoded bytes (see below)
    "shape": [int, int],                    // [seq_len, hidden_dim]
    "model": "string",                      // Model ID used
    "layer": int,                           // Layer extracted
    "dtype": "string",                      // Source dtype before encoding
    "encoding_format": "base64",            // Confirms format used
    "attention_mask": [int, ...]            // Optional: [seq_len] of 0/1
}
```

**Base64 Encoding Details:**
- Data is flattened to 1D, converted to float32, then base64-encoded
- Decode: `base64.b64decode(data)` -> `np.frombuffer(..., dtype=np.float32)` -> `reshape(shape)`
- ~25% smaller than JSON float arrays, much faster to parse
- Matches OpenAI embeddings API pattern

#### Example

```bash
curl -X POST http://localhost:8000/v1/hidden_states \
  -H "Content-Type: application/json" \
  -d '{
    "input": "<|im_start|>user\nA beautiful sunset over the ocean<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n",
    "model": "qwen3-4b",
    "layer": -2,
    "max_length": 512
  }'
```

Response:
```json
{
    "hidden_states": [
        [0.123, -0.456, ...],  // Token 0: <|im_start|>
        [0.789, 0.012, ...],   // Token 1: user
        ...                     // 21 tokens total for this prompt
    ],
    "shape": [21, 2560],
    "model": "qwen3-4b",
    "layer": -2,
    "dtype": "bfloat16"
}
```

## Implementation Notes

### 1. Model Requirements

For Z-Image, the model must be **Qwen3-4B** (specifically `Qwen/Qwen3-4B` or compatible):
- Hidden dimension: 2560
- Layers: 36
- Vocab: 151936 tokens

### 2. Hidden State Extraction

```python
# Key difference from embeddings endpoint:
# 1. Use output_hidden_states=True
# 2. Extract specific layer (not final)
# 3. NO pooling

outputs = model(
    input_ids,
    attention_mask=attention_mask,
    output_hidden_states=True,
)

# Get second-to-last layer (matches diffusers/DiffSynth)
hidden_states = outputs.hidden_states[-2]  # [batch, seq_len, hidden_dim]

# Filter by attention mask (remove padding tokens)
# IMPORTANT: Only return actual tokens, not padding
seq_len = attention_mask.sum().item()
hidden_states = hidden_states[:, :seq_len, :]  # [batch, actual_seq_len, hidden_dim]
```

### 3. Attention Mask Filtering

The attention mask filtering is critical for variable-length sequences:

```python
def filter_by_attention_mask(hidden_states, attention_mask):
    """
    Remove padding tokens from hidden states.

    Args:
        hidden_states: [batch, max_seq_len, hidden_dim]
        attention_mask: [batch, max_seq_len] with 1 for real tokens, 0 for padding

    Returns:
        List of [seq_len_i, hidden_dim] tensors (one per batch item)
    """
    result = []
    for i in range(hidden_states.shape[0]):
        mask = attention_mask[i].bool()
        filtered = hidden_states[i][mask]  # [actual_seq_len, hidden_dim]
        result.append(filtered)
    return result
```

### 4. MLX-Specific Implementation

For MLX models:

```python
class MLXHiddenStatesExtractor:
    """Extract hidden states from MLX models (Qwen3-4B)."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def extract(
        self,
        text: str,
        layer: int = -2,
        max_length: int = 512,
    ) -> dict:
        import mlx.core as mx

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = mx.array(inputs["input_ids"])
        attention_mask = mx.array(inputs["attention_mask"])

        # Forward pass with hidden states
        # Note: Need to check MLX model's forward signature
        outputs = self.model(
            input_ids,
            output_hidden_states=True,
        )

        # Get requested layer
        if hasattr(outputs, "hidden_states"):
            all_hidden_states = outputs.hidden_states
        else:
            # Fallback: run through layers manually
            all_hidden_states = self._get_hidden_states_manual(input_ids)

        hidden_states = all_hidden_states[layer]  # [batch, seq_len, dim]

        # Filter by attention mask
        seq_len = int(mx.sum(attention_mask[0]).item())
        hidden_states = hidden_states[:, :seq_len, :]

        return {
            "hidden_states": hidden_states[0].tolist(),
            "shape": [seq_len, hidden_states.shape[-1]],
            "layer": layer,
            "dtype": str(hidden_states.dtype),
        }
```

### 5. llama.cpp Implementation

For GGUF models via llama-cpp-python:

```python
class LlamaCppHiddenStatesExtractor:
    """Extract hidden states from llama.cpp models."""

    def __init__(self, model):
        self.model = model

    def extract(
        self,
        text: str,
        layer: int = -2,
        max_length: int = 512,
    ) -> dict:
        # Tokenize
        tokens = self.model.tokenize(text.encode("utf-8"))
        if len(tokens) > max_length:
            tokens = tokens[:max_length]

        # Reset and eval
        self.model.reset()

        # Note: Getting hidden states from llama.cpp requires:
        # 1. Model must be loaded with embeddings=True
        # 2. Use llama_get_embeddings_seq() or similar

        # This is model-dependent and may require patches to llama-cpp-python
        # or using the raw C API

        raise NotImplementedError(
            "Hidden state extraction from llama.cpp requires "
            "embeddings mode. Check model compatibility."
        )
```

## Error Handling

### Error Response Format

```json
{
    "error": {
        "message": "string",
        "type": "string",
        "code": "string"
    }
}
```

### Error Codes

| Code | Type | Description |
|------|------|-------------|
| 400 | `invalid_request_error` | Missing or invalid parameters |
| 404 | `model_not_found` | Requested model not loaded |
| 422 | `model_error` | Model doesn't support hidden state extraction |
| 500 | `internal_error` | Server error |

## Performance Considerations

1. **Memory**: Hidden states can be large. For 512 tokens x 2560 dim x float32 = ~5MB per request
2. **Batching**: Consider supporting batch requests for efficiency
3. **Caching**: Consider caching tokenization for repeated prefixes
4. **Base64 encoding**: Use `encoding_format: "base64"` for ~25% smaller responses and faster parsing

### Base64 Implementation

**Server-side encoding:**
```python
import base64
import numpy as np

def encode_hidden_states_base64(hidden_states: np.ndarray) -> str:
    """Encode hidden states as base64 string."""
    # Ensure float32 and C-contiguous
    data = np.ascontiguousarray(hidden_states, dtype=np.float32)
    return base64.b64encode(data.tobytes()).decode("ascii")

# In response handler:
if encoding_format == "base64":
    return {
        "hidden_states": encode_hidden_states_base64(hidden_states),
        "shape": list(hidden_states.shape),
        "encoding_format": "base64",
        ...
    }
```

**Client-side decoding:**
```python
import base64
import numpy as np
import torch

def decode_hidden_states_base64(data: str, shape: list[int]) -> torch.Tensor:
    """Decode base64 hidden states to torch tensor."""
    raw = base64.b64decode(data)
    arr = np.frombuffer(raw, dtype=np.float32).reshape(shape)
    return torch.from_numpy(arr)

# In API client:
if response.get("encoding_format") == "base64":
    hidden_states = decode_hidden_states_base64(
        response["hidden_states"],
        response["shape"]
    )
else:
    hidden_states = torch.tensor(response["hidden_states"])
```

## Integration with llm-dit-experiments

The `APIBackend` class in llm-dit-experiments expects this endpoint:

```python
# src/llm_dit/backends/api.py
response = self.client.post(
    "/v1/hidden_states",
    json={
        "input": formatted_prompt,  # Already has chat template
        "model": self.config.model_id,
        "layer": self.config.hidden_layer,  # -2
        "max_length": self.config.max_length,  # 512
    },
)
```

## Testing

Test with Qwen3-4B and Z-Image chat template:

```python
def test_hidden_states_endpoint():
    # Z-Image style prompt with chat template
    prompt = """<|im_start|>user
A beautiful sunset over the ocean<|im_end|>
<|im_start|>assistant
<think>
</think>
"""

    response = client.post(
        "/v1/hidden_states",
        json={"input": prompt, "model": "qwen3-4b", "layer": -2}
    )

    data = response.json()

    # Verify shape
    assert data["shape"][1] == 2560, "Qwen3-4B hidden dim is 2560"
    assert data["shape"][0] > 0, "Should have at least one token"
    assert data["shape"][0] < 512, "Should be less than max_length"

    # Verify hidden states match shape
    hs = data["hidden_states"]
    assert len(hs) == data["shape"][0]
    assert len(hs[0]) == data["shape"][1]
```

## Appendix: Z-Image Technical Context

Z-Image uses Qwen3-4B as a text encoder for image generation:

- **Architecture**: S3-DiT (Single-Stream DiT), 6B parameters
- **Text encoder**: Qwen3-4B instruct model
- **Embedding extraction**: `hidden_states[-2]` (second-to-last layer)
- **Chat template**: Qwen3 format with `<|im_start|>`, `<|im_end|>`, `<think>` tags
- **Max length**: 512 tokens
- **CFG**: Not used (baked in via Decoupled DMD distillation)

The full generation flow:
1. Format prompt with Qwen3 chat template
2. Extract hidden_states[-2] from Qwen3-4B
3. Pass embeddings to Z-Image DiT transformer
4. 9 denoising steps with FlowMatchEuler scheduler
5. Decode latents with 16-channel VAE
