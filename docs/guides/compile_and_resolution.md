last updated: 2026-02-07

# torch.compile and resolution guide

Performance and resolution reference for FLUX.2 Klein on RTX 4090.

## performance decision matrix

| scenario | compile transformer | compile vae | compile encoder | recommended steps |
|----------|-------------------|-------------|-----------------|-------------------|
| distilled, single resolution, repeated gen | YES | YES | NO | 4 |
| distilled, varied resolutions | dynamic | NO | NO | 4 |
| base, single resolution | YES | YES | optional | 50 |
| base, varied resolutions | dynamic | NO | NO | 50 |
| editing (ref images, resolution matching) | depends on pattern | NO | NO | 4-50 |

**Key insight:** compile only helps when you generate many images at the *same* resolution.
Distilled models run so few steps (4) that the ~90s warmup is hard to amortize.
Base models (50 steps) amortize warmup in ~5 generations.

## compile ROI math

### transformer

- warmup: ~90s per new resolution (full tracing + optimization)
- per-step speedup: ~2x (0.8s -> 0.4s per step)
- **distilled (4 steps):** saves ~1.6s/gen. break-even: ~56 generations at same resolution
- **base (50 steps):** saves ~20s/gen. break-even: ~5 generations at same resolution

### VAE

- warmup: ~89s (one-time, resolution-dependent)
- per-decode savings: ~0.5s
- break-even: ~178 generations (marginal value for most workflows)

### encoder

- runs once per request, then offloads to CPU (shuttle pattern)
- compile ROI is near zero unless batching many prompts sequentially
- **recommendation:** do not compile the encoder

## resolution constraints

### FLUX.2 VAE requirements

- **minimum granularity:** multiples of 16 pixels
- spatial compression: 8x (VAE) * 2x (patchify) = 16x total
- latent tokens = `(height / 16) * (width / 16)`
- this token count is what torch.compile traces -- changing it triggers recompilation

### LTX-2 VAE requirements

- **minimum granularity:** multiples of 32 pixels
- temporal compression: 8x, spatial: 32x

### reference images

- each reference image is preprocessed independently to 16px multiples
- pixel cap: 1 megapixel per image (4 refs), or 4 megapixels (1 ref)
- `match_image_size` selects ONE reference by index (0-3) and rounds to 16px multiples
- reference image processing does NOT trigger recompilation -- only the *output* resolution matters

## what triggers recompilation

| change | recompiles? | why |
|--------|------------|-----|
| output resolution (different width/height) | YES | different num_tokens tensor shape |
| prompt text | NO | `pad_to_max=True` ensures fixed `[1, 512]` shape |
| seed | NO | scalar, not a tensor shape |
| guidance scale | NO | scalar |
| num_steps | NO | loop count, not tensor shape |
| reference images (same output res) | NO | output shape unchanged |
| reference images (match_image_size changes output) | YES | different output resolution |

## recommended configs for RTX 4090

### distilled model (speed-first, single resolution)

```toml
[flux2]
compile = true
compile_vae = false           # marginal ROI (89s warmup, ~0.5s savings/decode)
compile_mode = "default"
offload_between_stages = true
block_offload = false
quantization = "fp8-weight-only"
```

### base model (quality-first, single resolution)

```toml
[flux2]
compile = true
compile_vae = true            # worth it -- 50 steps amortize well
compile_mode = "max-autotune-no-cudagraphs"
offload_between_stages = true
block_offload = false
quantization = "fp8-weight-only"
```

### varied resolutions with compile (dynamic shapes, experimental)

```toml
[flux2]
compile = true
compile_dynamic = true
compile_vae = false
compile_mode = "default"
offload_between_stages = true
block_offload = false
quantization = "fp8-weight-only"
```

Dynamic shapes (`compile_dynamic = true`) tells torch.compile to generate shape-generic
kernels that handle varying sequence lengths without retracing. This means changing
resolution (e.g., 1024x1024 to 768x1344) will NOT trigger the ~90s recompilation.

Tradeoffs:
- first compilation may be slightly slower than static (shape guards are more complex)
- per-step performance may be ~5-10% slower than static fullgraph=True
- uses `fullgraph=False` as a safety measure for data-dependent branches

Best for base models with varied resolutions where the 50-step per-gen speedup
outweighs the slight per-step overhead.

### varied resolutions without compile (flexibility-first)

```toml
[flux2]
compile = false
compile_vae = false
offload_between_stages = true
block_offload = false
quantization = "fp8-weight-only"
```

This is the default config. Best for workflows that change resolution frequently (editing, aspect ratio experiments).

## compatibility matrix

| feature | torch.compile | block_offload | FP8 quant | stage offload |
|---------|-------------|---------------|-----------|---------------|
| torch.compile | -- | INCOMPATIBLE | OK (static shapes) | OK |
| block_offload | INCOMPATIBLE | -- | INCOMPATIBLE | N/A (alternative) |
| FP8 quant | OK | INCOMPATIBLE | -- | OK |
| stage offload | OK | N/A | OK | -- |

**block_offload** moves individual transformer blocks to/from GPU. It cannot coexist with torch.compile (graph tracing requires all parameters on one device) or FP8 quantization (quantized blocks cannot be dynamically moved).

**stage offload** (`offload_between_stages = true`) moves entire components (encoder -> transformer -> VAE) sequentially. This is compatible with everything and is the recommended memory strategy for 24GB GPUs.

## VRAM budget breakdown (9B FP8, 1024x1024)

| stage | component | VRAM | notes |
|-------|-----------|------|-------|
| 1 | Qwen3-8B encoder | ~8GB | offloads to pinned CPU after encoding |
| 1.5 | VAE encode (ref images) | ~0.3GB | only in editing mode, shares with encoder stage |
| 2 | Transformer (FP8) | ~9GB | dominant VRAM consumer |
| 2 | Activations + compile cache | ~3-5GB | scales with resolution |
| 3 | VAE decode | ~0.3GB | minimal |
| **peak** | max(stage 1, stage 2, stage 3) | **~14GB** | well within 24GB |

With three-stage offloading, peak VRAM = max(any single stage), not the sum.

## attention backend priority

On RTX 4090 (SM89), the `auto` backend selects in this order:

1. **flash_attn_3** -- fastest if installed (Ada Lovelace native)
2. **flash_attn_2** -- fast, widely available
3. **sage_int8_fp16** -- SM80+, best memory efficiency
4. **xformers** -- good fallback
5. **sdpa** -- PyTorch built-in, always available

No configuration needed if `attention_backend = "auto"` (default).

## FAQ

**Does compile save VRAM?**
Marginally. Operator fusion reduces peak activation memory, but model weights (the dominant cost) stay the same size.

**Can I compile the text encoder?**
Yes, but ROI is near zero. The encoder runs once per request then offloads to CPU via the shuttle pattern. Compile warmup would be wasted.

**Does changing the prompt trigger recompilation?**
No. `pad_to_max=True` ensures all prompts produce the same `[1, 512]` tensor shape regardless of actual text length.

**Does uploading a reference image trigger recompilation?**
Only if `match_image_size` is active AND the matched image has a different aspect ratio than your current output resolution. The reference images themselves are encoded independently -- it is the *output* resolution that matters for compilation.

**Why is compile_vae recommended as false for distilled?**
89s warmup for ~0.5s savings per decode means you need ~178 generations at the same resolution to break even. For a 4-step model that generates in ~5s total, this is rarely worthwhile.

**Why not use `max-autotune-no-cudagraphs` for distilled?**
`max-autotune` adds extra warmup time for marginal per-step gains. With only 4 steps, the extra warmup cost outweighs the benefit. Use `"default"` compile mode for distilled.

**What about `fullgraph=True`?**
We compile with `fullgraph=True` to prevent silent graph breaks. If the model's forward pass contains code that breaks the graph (logging, Python-side conditionals), torch.compile will raise an error at compile time instead of silently falling back to eager mode. All our models use `torch.compiler.is_compiling()` guards around logging to prevent this.

**What is `compile_dynamic`?**
When `compile_dynamic = true`, torch.compile uses `dynamic=True` which generates shape-generic kernels. Instead of specializing on exact tensor sizes (e.g., `[1, 4096, 128]` for 1024x1024), dynamo creates symbolic shape guards that accept any valid size. This eliminates the ~90s recompilation when resolution changes. The tradeoff is slightly less aggressive optimization (~5-10% slower per step) since the compiler can't fully specialize on known dimensions. When enabled, `fullgraph=False` is used automatically as a safety measure. Only the image latent sequence length (dim 1 of `x` and `x_ids`) varies with resolution -- text embeddings are always `[1, 512, 4096]` due to padding.
