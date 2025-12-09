# Diffusers RoPE Off-By-One Bug Investigation

**Status**: Open Investigation
**Priority**: Medium
**Potential Impact**: 2.1% additional token capacity (1504 → 1536)
**Date Discovered**: 2025-12-09
**Discoverer**: Claude Opus 4.5

---

## Summary

Through systematic testing, we discovered that Z-Image can only handle **1504 tokens** despite the config specifying `axes_lens=[1536, 512, 512]`. This appears to be an off-by-one error in the RoPE frequency table generation in diffusers' `ZImageTransformer2DModel`.

## Evidence

### Test Results

Binary search testing in isolated subprocesses (2025-12-09):

```
512:  SUCCESS
1024: SUCCESS
1280: SUCCESS
1408: SUCCESS
1472: SUCCESS
1500: SUCCESS
1504: SUCCESS
1505: FAIL - RuntimeError: vectorized_gather_kernel: index out of bounds
1510: FAIL
1520: FAIL
1536: FAIL
```

### Mathematical Pattern

- Config specifies: `axes_lens=[1536, 512, 512]` and `axes_dims=[32, 48, 48]`
- 1536 = 48 × 32 (expected working limit)
- 1504 = 47 × 32 (actual working limit)
- Difference: 32 tokens (one "chunk")

This strongly suggests the frequency table is computed for indices 0-46 (47 values) instead of 0-47 (48 values).

### Ruled Out Hypotheses

1. **NOT Flash Attention related**
   - Same crash occurs with `--attention-backend sdpa`
   - Flash Attention is used in DiT blocks, not RoPE generation

2. **NOT image size related**
   - Tested at 128×128, 256×256, 512×512 (latent: 16×16, 32×32, 64×64)
   - Same 1505-token crash regardless of image dimensions
   - Image positions use axes 1 & 2, not axis 0 (text axis)

3. **NOT special token overhead**
   - Chat format adds only 8-19 tokens of overhead
   - Not enough to explain 32-token gap

## Root Cause Hypothesis

The bug is likely in diffusers' multi-axis RoPE implementation for Z-Image. Possible locations:

### 1. Frequency Table Generation

```python
# Hypothetical buggy code in diffusers/models/transformers/transformer_z_image.py
# (actual code needs inspection)

def _compute_rope_frequencies(axes_lens, axes_dims):
    """Compute RoPE frequencies for multi-axis positions."""
    freqs = []
    for axis_len, axis_dim in zip(axes_lens, axes_dims):
        # BUG: Off-by-one here?
        # Should be: range(axis_len // axis_dim)
        # Actually: range((axis_len // axis_dim) - 1)
        for i in range((axis_len // axis_dim) - 1):  # WRONG: 0-46 instead of 0-47
            freq = compute_freq(i, axis_dim)
            freqs.append(freq)
    return freqs
```

### 2. Index Bounds Check

```python
# Alternative hypothesis: bounds check is too strict
def apply_rope(positions, frequencies):
    """Apply rotary embeddings."""
    max_pos = len(frequencies) - 1  # BUG: Should be len(frequencies)?
    if positions.max() > max_pos:
        raise RuntimeError("index out of bounds")
```

## Investigation Plan

### Phase 1: Locate the Bug (2-4 hours)

1. **Find the relevant code in diffusers**
   ```bash
   # Clone diffusers source
   git clone https://github.com/huggingface/diffusers.git
   cd diffusers

   # Search for Z-Image transformer
   grep -r "ZImageTransformer2DModel" --include="*.py"
   grep -r "axes_lens" --include="*.py"
   grep -r "multi.*axis.*rope" --include="*.py" -i
   ```

2. **Examine RoPE implementation**
   - Look for frequency table generation
   - Check index calculations
   - Review bounds checks

3. **Add debug logging**
   - Print frequency table length
   - Print max position index
   - Verify dimensions match config

### Phase 2: Fix and Test (1-2 hours)

1. **Create a patch**
   ```python
   # Example fix (actual code will differ)
   - for i in range((axis_len // axis_dim) - 1):
   + for i in range(axis_len // axis_dim):
   ```

2. **Test the fix**
   ```bash
   # Install patched diffusers
   pip install -e /path/to/patched/diffusers

   # Test with 1505-1536 token prompts
   uv run scripts/generate.py \
     --model-path /path/to/z-image \
     "$(python -c 'print(\"A cat \" * 250)')"  # ~1520 tokens
   ```

3. **Verify stability**
   - Test at exactly 1536 tokens (should now work)
   - Test at 1537 tokens (should still fail)
   - Regression test: ensure 1504 still works

### Phase 3: Submit Fix (1 day)

1. **Create minimal reproducer**
   ```python
   # Standalone script demonstrating the bug
   import torch
   from diffusers import ZImageTransformer2DModel

   # Load model
   transformer = ZImageTransformer2DModel.from_pretrained(...)

   # Test with 1505 tokens
   text_embeddings = torch.randn(1, 1505, 3840)  # Should work but fails
   ```

2. **File issue on diffusers GitHub**
   - Title: "[Z-Image] RoPE off-by-one error limits text to 1504 tokens instead of 1536"
   - Include test results and mathematical analysis
   - Link to config showing axes_lens=[1536, ...]
   - Provide reproducer script

3. **Submit pull request**
   - Fix the bug
   - Add test case for 1536-token sequence
   - Update any relevant documentation

## Expected Outcomes

### If Fix Works

- Unlock full 1536 tokens (2.1% more capacity beyond current 1504)
- Total gain: 50.0% over conservative 1024 limit
- No more compression needed for prompts up to 1536 tokens

### If More Complex

The issue might be:
1. **Intentional limitation** - Training data capped at 1504 tokens
2. **Deeper bug** - Multiple places need fixing
3. **Hardware constraint** - Some GPUs can't handle 1536

In these cases, 1504 remains the safe limit, but we gain understanding.

## Alternative Workarounds

If the bug can't be fixed easily:

### 1. Monkey-Patch diffusers Locally

```python
# In llm_dit/patches/rope_fix.py
import diffusers.models.transformers.transformer_z_image as zt

# Save original function
_original_compute_freqs = zt._compute_rope_frequencies

def _fixed_compute_freqs(axes_lens, axes_dims):
    """Fixed version with correct bounds."""
    # ... fixed implementation
    pass

# Apply patch
zt._compute_rope_frequencies = _fixed_compute_freqs
```

### 2. Use Compression for 1505-1536 Range

```python
# Only compress if above 1504
if token_count > 1504:
    if token_count <= 1536:
        # Minimal compression (1.0x - 1.02x)
        compressed = interpolate_embeddings(embeddings, 1504)
    else:
        # Heavier compression
        compressed = attention_pool_embeddings(embeddings, 1504)
```

## Research Value

Even if the fix yields only 32 more tokens, the investigation is valuable:

1. **Validates our testing methodology** - Binary search found exact limit
2. **Deepens understanding** - How multi-axis RoPE works in DiT
3. **Opens extension path** - Once we understand the code, we can attempt NTK scaling
4. **Helps community** - Other Z-Image users face same limitation

## Related Research

- [RoPE Paper](https://arxiv.org/abs/2104.09864) - Original RoPE formulation
- [Multi-Axis RoPE for Vision](https://arxiv.org/abs/2305.13048) - DiT-specific RoPE
- [Diffusers Documentation](https://huggingface.co/docs/diffusers) - Implementation reference

## Next Steps

1. Clone diffusers and locate the bug (IMMEDIATE)
2. Create minimal reproducer (IMMEDIATE)
3. Test proposed fix locally (AFTER FINDING BUG)
4. Submit issue + PR to diffusers (AFTER TESTING)
5. Document workarounds in CLAUDE.md (IN PARALLEL)

---

**Note for Future Researchers:**

This is a low-hanging fruit investigation. The bug is likely a simple off-by-one error in a for loop. Even if you're not experienced with diffusers internals, the evidence is strong enough that you can grep for the relevant code and spot the issue. Total time investment: 4-8 hours for investigation, fix, and testing.
