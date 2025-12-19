last updated: 2025-12-19

# SigLIP2 Vision Encoder for Z-Image

## Summary

Z-Image Omni (diffusers PR 12857) reveals that Z-Image supports image conditioning
via SigLIP2 vision encoder. This was likely held back from initial release.

## Architecture

```
Text Input                    Reference Image
    |                              |
Qwen3-4B (~/Storage/Qwen3-4B)   SigLIP2 Vision Model (so400m)
    |                              |
2560-dim embeddings            1152-dim embeddings
    |                              |
cap_embedder (trained)         siglip_embedder (trained)
    |                              |
    +-------> 3840-dim <-----------+
                |
        Z-Image Turbo DiT
                |
          Generated Image
```

**Same DiT, dual encoders:**
- Text encoder: Qwen3-4B (unchanged, 2560 dim)
- Vision encoder: SigLIP2-so400m (NEW, 1152 dim)

## New Transformer Components

From diffusers PR 12857, these are added to `ZImageTransformer2DModel`:

```python
# Config parameter
siglip_feat_dim = 1152  # or None to disable (default fallback is 1152)

# New layers (trained weights needed)
siglip_embedder = RMSNorm + Linear(1152 -> 3840)
siglip_refiner = 2-layer transformer blocks
siglip_pad_token = learnable parameter
```

Sequence concatenation order:
```
[caption_tokens] + [latent_tokens] + [siglip_tokens]
```

## SigLIP2 Embedding Stats

```
SigLIP2-so400m-patch14-384:
  Hidden dim: 1152
  Patches: 27x27 = 729 tokens (384x384 input)
  Model: google/siglip2-so400m-patch14-384
```

## Status

- **Omni weights**: Not yet released
- **siglip_feat_dim**: Already in transformer config (set to None currently)
- **When released**: Load weights with siglip components trained

## Files

- `test_siglip_encoding.py` - SigLIP2 embedding analysis
- `../../coderef/diffusers/` - Full diffusers PR 12857
