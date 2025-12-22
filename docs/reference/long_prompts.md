# text sequence length limits

*last updated: 2025-12-22*

The DiT transformer has a **maximum text sequence length of 1504 tokens**. Exceeding this causes CUDA kernel crashes.

**This is an active research area.** See `internal/research/long_prompt_research.md` for detailed notes.

## current behavior (default)

- Prompts exceeding 1504 tokens are **automatically truncated** with a warning
- Truncation happens after encoding, preserving the first 1504 tokens of embeddings

## experimental compression modes

| Mode | CLI Flag | Status | Quality Impact |
|------|----------|--------|----------------|
| `truncate` | `--long-prompt-mode truncate` | Stable | Predictable, loses tail content |
| `interpolate` | `--long-prompt-mode interpolate` | **Default** | Preserves all content, smooth resampling |
| `pool` | `--long-prompt-mode pool` | Experimental | Under evaluation |
| `attention_pool` | `--long-prompt-mode attention_pool` | Experimental | Cosine similarity weighting |

## token count guidelines

| Content Type | Approximate Tokens |
|--------------|-------------------|
| Simple prompt ("A cat sleeping") | 10-20 tokens |
| Detailed prompt with style | 50-100 tokens |
| Template + system prompt | 100-200 tokens |
| Full format (system + think + assistant) | 150-300 tokens |
| Maximum safe prompt | ~1200-1300 tokens |

**Note:** The 1504 limit includes ALL tokens: system prompt, user prompt, think block, and assistant content.

## strategies for long prompts

1. **Omit system prompt**: Default templates add ~50-100 tokens
2. **Skip think block**: Empty think block adds ~10 tokens
3. **Use concise descriptions**: Focus on key visual elements
4. **Use experimental compression modes**

## checking token count

```python
from llm_dit import ZImageTextEncoder

encoder = ZImageTextEncoder.from_pretrained("/path/to/model")
output = encoder.encode("Your prompt here")
token_count = output.token_counts[0]
print(f"Token count: {token_count}/1504")
```

## why this limit exists

The Z-Image DiT uses multi-axis RoPE for position encoding:
- Axis 0 (1504 actual, 1536 configured): Text/time sequence positions
- Axis 1 (512): Image height positions
- Axis 2 (512): Image width positions

The actual limit is 1504 = 47 * 32, suggesting an off-by-one in the RoPE frequency table indexing. Exceeding 1504 causes CUDA kernel errors (`vectorized_gather_kernel: index out of bounds`).

**Future research directions**: RoPE extrapolation, hierarchical encoding, chunked attention.
