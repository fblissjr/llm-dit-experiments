last updated: 2026-01-23

# ltx-2 video generation

Comprehensive guide for LTX-2 text-to-video generation, including memory optimization strategies for 24GB GPUs.

## table of contents
- [overview](#overview)
- [architecture](#architecture)
- [quick start](#quick-start)
- [cli reference](#cli-reference)
- [memory optimization](#memory-optimization)
- [common workflows](#common-workflows)
- [troubleshooting](#troubleshooting)
- [model directory structure](#model-directory-structure)
- [related documentation](#related-documentation)

## overview

LTX-2 is Lightricks' 13B parameter text-to-video diffusion transformer that uses Gemma3-12B as its text encoder. The model is distilled for fast 12-step generation and supports FP8 quantization for memory-efficient inference.

### key capabilities

- Text-to-video generation at configurable resolutions (768x512 default)
- Variable frame counts (33-65 frames depending on VRAM)
- FP8 quantized transformer for 24GB GPU inference
- Multiple Gemma3 variants for text encoding (bf16, 8-bit, torchao int4)
- Precomputed embeddings for memory-constrained workflows
- LoRA support for style/subject customization

## architecture

| Component | Details |
|-----------|---------|
| Text encoder | Gemma3-12B vision-language model (3840 hidden dim) |
| Embedding connector | 2-layer bidirectional transformer with learnable registers |
| DiT | 13B parameter transformer, ~26GB in bf16, ~13GB in FP8 |
| VAE | Temporal VAE for video encoding/decoding |
| Scheduler | Flow matching with 12 distilled steps |

### embedding pipeline

```
Text Prompt → Gemma3-12B → Sublayer Extractor → Embeddings Connector → DiT → VAE → Video
              (3840 dim)    (per-layer routing)  (bidirectional xfmr)
```

The text encoder extracts hidden states from Gemma3, processes them through a learned connector, and conditions the DiT through cross-attention during denoising.

## quick start

### basic generation

```bash
uv run scripts/generate.py --model-type ltx2 \
    --model-path ~/Storage/LTX-2 \
    "A cat walking through a sunny garden"
```

### with custom output

```bash
uv run scripts/generate.py --model-type ltx2 \
    --model-path ~/Storage/LTX-2 \
    --output my_video.mp4 \
    --ltx2-num-frames 49 \
    --width 768 --height 512 \
    "Ocean waves crashing on rocky shore at sunset"
```

## cli reference

### model paths

| Flag | Description | Default |
|------|-------------|---------|
| `--model-path` | Path to LTX-2 model directory | Required |
| `--ltx2-model-path` | Alternative (takes precedence over --model-path) | - |
| `--text-encoder-path` | Separate text encoder path (for different Gemma variants) | `{model-path}/text_encoder` |
| `--ltx2-encoder-model-id` | Alternative text encoder path (takes precedence) | - |

### generation parameters

| Flag | Description | Default |
|------|-------------|---------|
| `--ltx2-num-frames` | Number of video frames | 33 |
| `--ltx2-fps` | Output framerate | 24 |
| `--ltx2-steps` | Diffusion steps | 12 |
| `--ltx2-guidance-scale` | CFG guidance scale | 3.5 |
| `--width` | Video width (multiple of 16) | 768 |
| `--height` | Video height (multiple of 16) | 512 |
| `--seed` | Random seed for reproducibility | Random |
| `--output` | Output video path | output.mp4 |

### optimization flags

| Flag | Description | Default |
|------|-------------|---------|
| `--ltx2-text-encoder-device` | Device for Gemma3 (cpu/cuda) | cpu |
| `--ltx2-transformer-device` | Device for DiT (cpu/cuda) | cuda |
| `--ltx2-vae-device` | Device for VAE (cpu/cuda) | cuda |
| `--ltx2-quantize` | Transformer quantization (none/fp8) | fp8 |
| `--ltx2-skip-cleanup` | Skip memory cleanup between stages | False |
| `--ltx2-gemma-variant` | Gemma3 variant (bf16/8bit/q4-qat) | bf16 |

### embeddings precomputation

| Flag | Description |
|------|-------------|
| `--ltx2-save-embeddings` | Save embeddings to file, skip video generation |
| `--ltx2-load-embeddings` | Load embeddings from file, skip text encoding |

### lora

| Flag | Description | Default |
|------|-------------|---------|
| `--ltx2-lora-path` | Path to LoRA safetensors | - |
| `--ltx2-lora-scale` | LoRA blend scale | 1.0 |

## memory optimization

### understanding the memory profile

LTX-2 has three major components that compete for VRAM:

| Component | bf16 Size | FP8/Quantized | Notes |
|-----------|-----------|---------------|-------|
| Gemma3-12B | ~24GB | 12GB (8bit) / 3GB (torchao int4) | Text encoder |
| DiT 13B | ~26GB | ~13GB (FP8) | Transformer |
| VAE | ~2GB | ~2GB | Video decoder |

**Key insight**: Components are loaded sequentially, so peak VRAM = max(encoder, transformer + VAE), not the sum.

**Note on torchao int4**: The q4-qat variant loads the model to CPU in bf16 format (~24GB system RAM), applies `int4_weight_only()` quantization, then moves to GPU (~3GB VRAM). This happens during the initial load phase.

### strategy 1: default (cpu text encoder)

Best for: RTX 4090 (24GB) with standard setup

```bash
uv run scripts/generate.py --model-type ltx2 \
    --model-path ~/Storage/LTX-2 \
    --ltx2-text-encoder-device cpu \
    "A cat walking"
```

**Memory profile:**
- Text encoding: ~24GB system RAM (CPU), 0GB VRAM
- Generation: ~15GB VRAM (FP8 transformer + VAE)

**Tradeoff:** Slower text encoding (~30s on CPU vs ~5s on GPU)

### strategy 2: 8-bit gemma on gpu

Best for: Faster encoding with moderate VRAM savings

```bash
uv run scripts/generate.py --model-type ltx2 \
    --model-path ~/Storage/LTX-2 \
    --ltx2-text-encoder-device cuda \
    --ltx2-gemma-variant 8bit \
    "A cat walking"
```

**Memory profile:**
- Text encoding: ~14GB VRAM (8-bit Gemma + connectors + activations)
- Generation: ~15GB VRAM

**Tradeoff:** Requires `bitsandbytes` library, slight quality loss

### strategy 3: torchao int4 quantization (smallest encoder)

Best for: Minimum encoder VRAM, faster GPU encoding, works with any Gemma model

```bash
uv run scripts/generate.py --model-type ltx2 \
    --model-path ~/Storage/LTX-2 \
    --ltx2-text-encoder-device cuda \
    --ltx2-gemma-variant q4-qat \
    "A cat walking"
```

**Memory profile:**
- Load phase: ~24GB system RAM (loads bf16 weights to CPU)
- Quantization: Applies torchao `int4_weight_only()` quantization
- Runtime: ~3GB VRAM (quantized Gemma + connectors + activations)
- Generation: ~15GB VRAM

**How it works:**
The `q4-qat` variant now uses torchao's dynamic int4 quantization:
1. Loads the standard bf16 Gemma model to CPU (~24GB system RAM)
2. Applies `int4_weight_only()` quantization in-place
3. Moves quantized model to GPU (~3GB VRAM)

This approach works with any Gemma model - no need for pre-quantized checkpoints. The original "qat-q4_0-unquantized" models from Google actually store weights in bf16 format and weren't truly quantized, which is why we now quantize at load time instead.

**Requirements:**
- ~24GB free system RAM during model loading
- torchao library (included in dependencies)

### strategy 4: precomputed embeddings (minimum peak vram)

Best for: Running same prompt with multiple seeds, absolute minimum VRAM

**Step 1: Encode prompt (one-time)**
```bash
uv run scripts/generate.py --model-type ltx2 \
    --model-path ~/Storage/LTX-2 \
    --ltx2-text-encoder-device cuda \
    --ltx2-gemma-variant q4-qat \
    --ltx2-save-embeddings embeddings/cat.safetensors \
    "A cat walking through a sunny garden"
```

**Step 2: Generate video (can run many times)**
```bash
uv run scripts/generate.py --model-type ltx2 \
    --model-path ~/Storage/LTX-2 \
    --ltx2-load-embeddings embeddings/cat.safetensors \
    --seed 42 \
    --output cat_seed42.mp4
```

**Memory profile:**
- Step 1 (encoding): ~24GB system RAM, ~3GB VRAM (torchao int4 encoder only, no transformer)
- Step 2 (generation): ~15GB VRAM (transformer + VAE only, no encoder)
- Peak VRAM: max(3GB, 15GB) = 15GB (never both at once)

**Use cases:**
- Generate multiple videos with different seeds from same prompt
- Distributed inference (encode on one machine, generate on another)
- Batch processing with prompt caching

### strategy 5: one-liner for constrained vram

Combine encode + generate in a single command chain:

```bash
uv run scripts/generate.py --model-type ltx2 \
    --model-path ~/Storage/LTX-2 \
    --ltx2-text-encoder-device cuda \
    --ltx2-gemma-variant q4-qat \
    --ltx2-save-embeddings /tmp/claude/prompt.safetensors \
    "A majestic eagle soaring" && \
uv run scripts/generate.py --model-type ltx2 \
    --model-path ~/Storage/LTX-2 \
    --ltx2-load-embeddings /tmp/claude/prompt.safetensors \
    --output eagle.mp4
```

## memory estimates by configuration

| Configuration | System RAM | Encoder VRAM | Generation VRAM | Peak VRAM | Notes |
|---------------|------------|--------------|-----------------|-----------|-------|
| bf16 on CPU | 24GB | 0GB | 15GB | 15GB | Slowest encoding |
| bf16 on CUDA | - | 26GB | 15GB | 26GB | OOM on 24GB GPU |
| 8bit on CUDA | - | 14GB | 15GB | 15GB | Requires bitsandbytes |
| q4-qat on CUDA (torchao) | 24GB | 3GB | 15GB | 15GB | Load-time quantization |
| Precomputed + q4-qat | 24GB | 3GB | 15GB | 15GB | Separate encode/gen phases |

**Note on q4-qat:** Uses torchao `int4_weight_only()` quantization applied at load time. Loads bf16 model to CPU (~24GB system RAM), quantizes, then moves to GPU (~3GB VRAM). Works with any Gemma model - no pre-quantized checkpoint needed.

## common workflows

### batch generation with same prompt

```bash
# Encode once
uv run scripts/generate.py --model-type ltx2 \
    --model-path ~/Storage/LTX-2 \
    --ltx2-save-embeddings prompts/sunset.safetensors \
    "Golden sunset over mountain peaks with clouds"

# Generate multiple variations
for seed in 1 2 3 4 5; do
    uv run scripts/generate.py --model-type ltx2 \
        --model-path ~/Storage/LTX-2 \
        --ltx2-load-embeddings prompts/sunset.safetensors \
        --seed $seed \
        --output "sunset_v${seed}.mp4"
done
```

### higher frame count (longer video)

```bash
# 49 frames = ~2 seconds at 24fps
uv run scripts/generate.py --model-type ltx2 \
    --model-path ~/Storage/LTX-2 \
    --ltx2-num-frames 49 \
    --ltx2-text-encoder-device cpu \
    "A river flowing through autumn forest"
```

**Note:** Higher frame counts require more VRAM. Maximum ~65 frames on 24GB GPU.

### higher resolution

```bash
# 1024x576 (16:9 widescreen)
uv run scripts/generate.py --model-type ltx2 \
    --model-path ~/Storage/LTX-2 \
    --width 1024 --height 576 \
    "Cinematic cityscape at night"
```

**Note:** Resolution increases VRAM usage. May need to reduce frame count.

### with lora

```bash
uv run scripts/generate.py --model-type ltx2 \
    --model-path ~/Storage/LTX-2 \
    --ltx2-lora-path ~/Storage/loras/anime_style.safetensors \
    --ltx2-lora-scale 0.8 \
    "An anime character running through cherry blossoms"
```

## troubleshooting

### out of memory (oom)

**Symptoms:** CUDA out of memory error during generation

**Solutions (in order of preference):**
1. Use precomputed embeddings workflow (Strategy 4)
2. Use `--ltx2-text-encoder-device cpu`
3. Reduce `--ltx2-num-frames` (try 33 → 25)
4. Reduce resolution (try 768x512 → 640x480)

### stale cache issues

**Symptoms:** CLI arguments not being applied, old behavior persisting

**Solution:**
```bash
./scripts/refresh_cache.sh
# Then retry your command
```

### text encoder path not found

**Symptoms:** Error about missing text encoder

**Check:**
```bash
ls ~/Storage/LTX-2/text_encoder/
# Should contain model files (config.json, model.safetensors, tokenizer files)
```

The default text encoder is included in the LTX-2 model directory. The `--ltx2-gemma-variant q4-qat` flag uses the same text encoder but applies torchao quantization at load time - no separate model download needed.

### slow generation

**Possible causes:**
1. Text encoder on CPU (`--ltx2-text-encoder-device cpu`) - expected ~30s for encoding
2. No FP8 quantization - ensure `--ltx2-quantize fp8` (default)
3. Memory pressure causing swapping

**Optimize:**
```bash
# Use torchao int4 encoder on GPU for fast encoding
uv run scripts/generate.py --model-type ltx2 \
    --model-path ~/Storage/LTX-2 \
    --ltx2-text-encoder-device cuda \
    --ltx2-gemma-variant q4-qat \
    "Your prompt"
```

### embeddings file incompatible

**Symptoms:** Shape mismatch when loading embeddings

**Cause:** Embeddings were saved with different encoder configuration

**Solution:** Re-encode with current configuration:
```bash
uv run scripts/generate.py --model-type ltx2 \
    --model-path ~/Storage/LTX-2 \
    --ltx2-save-embeddings embeddings/prompt.safetensors \
    "Your prompt"
```

## model directory structure

Expected LTX-2 model directory layout:

```
~/Storage/LTX-2/
├── text_encoder/
│   ├── config.json
│   ├── model.safetensors (or sharded files)
│   ├── tokenizer.json
│   └── ...
├── transformer/
│   ├── config.json
│   ├── diffusion_pytorch_model.safetensors
│   └── ...
├── vae/
│   ├── config.json
│   ├── diffusion_pytorch_model.safetensors
│   └── ...
└── scheduler/
    └── scheduler_config.json
```

## related documentation

- [quantization.md](../reference/quantization.md) - FP8 and other quantization methods
- [distributed.md](../guides/distributed.md) - Multi-GPU and distributed inference
- [cli_flags.md](../reference/cli_flags.md) - Complete CLI reference
