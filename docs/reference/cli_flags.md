# cli flags reference

*last updated: 2026-02-14*

CLI flags for `scripts/generate.py` and `web/server.py`. Both use the same base parser from `cli.py`. Use `--model-type` to select which pipeline to run.

> **Note:** `scripts/generate.py` is planned for deprecation. See [entry_points.md](entry_points.md) for the strategic direction.

## model and config

| Flag | Description |
|------|-------------|
| `--model-type` | Model type: zimage (default), flux2, ltx2, qwenimage-layered, qwenimage-t2i, qwenimage-edit |
| `--model-path` | Path to Z-Image model |
| `--qwen-image-model-path` | Path to Qwen-Image-Layered model |
| `--config` | Path to TOML config file |
| `--profile` | Config profile to use (default: "default") |
| `--templates-dir` | Path to templates directory |

```bash
uv run scripts/generate.py --config config.toml --profile rtx4090 "A cat"
```

## device placement

| Flag | Description |
|------|-------------|
| `--text-encoder-device` | cpu/cuda/mps/auto |
| `--dit-device` | cpu/cuda/mps/auto |
| `--vae-device` | cpu/cuda/mps/auto |

## api backend

| Flag | Description |
|------|-------------|
| `--api-url` | URL for heylookitsanllm API |
| `--api-model` | Model ID for API backend |
| `--local-encoder` | Force local encoder even when --api-url is set (for A/B testing) |

## optimization

| Flag | Description |
|------|-------------|
| `--cpu-offload` | Enable CPU offload for transformer |
| `--flash-attn` | Enable Flash Attention |
| `--compile` | Compile transformer with torch.compile |
| `--debug` | Enable debug logging (embedding stats, token IDs) |

## pytorch native

| Flag | Description |
|------|-------------|
| `--attention-backend` | auto/flash_attn_2/flash_attn_3/sage/xformers/sdpa |
| `--use-custom-scheduler` | Use pure PyTorch FlowMatchScheduler |
| `--tiled-vae` | Enable tiled VAE decode for 2K+ images |
| `--tile-size` | Tile size in pixels (default: 512) |
| `--tile-overlap` | Overlap between tiles (default: 64) |
| `--embedding-cache` | Enable embedding cache for repeated prompts |
| `--cache-size` | Max cached embeddings (default: 100) |
| `--long-prompt-mode` | How to handle prompts >1504 tokens: truncate/interpolate/pool/attention_pool |
| `--hidden-layer` | Which hidden layer to extract embeddings from (default: -2, penultimate) |

## dype (high-resolution)

| Flag | Description |
|------|-------------|
| `--dype` | Enable DyPE position extrapolation for high-res generation |
| `--dype-method` | Method: vision_yarn/yarn/ntk (default: vision_yarn) |
| `--dype-scale` | DyPE magnitude lambda_s (default: 2.0) |
| `--dype-exponent` | DyPE decay speed lambda_t (default: 2.0 = quadratic) |
| `--dype-start-sigma` | When to start DyPE decay (0-1, 1.0 = from start) |
| `--dype-base-shift` | Noise schedule shift at base resolution (default: 0.5) |
| `--dype-max-shift` | Noise schedule shift at max resolution (default: 1.15) |
| `--dype-base-resolution` | Training resolution (Z-Image: 1024, Qwen: 1328) |
| `--dype-anisotropic` | Use per-axis scaling for extreme aspect ratios (16:9, 9:16) |
| `--dype-multipass` | Generation mode: single/twopass/threepass (default: single) |
| `--dype-pass2-strength` | img2img strength for second pass (default: 0.5) |
| `--dype-pass3-strength` | img2img strength for third pass (default: 0.4) |
| `--dype-frequency-modulation` | Enable timestep-based RoPE frequency scaling (experimental) |

## generation

| Flag | Description |
|------|-------------|
| `--width` | Image width in pixels (default: 1024, must be divisible by 16) |
| `--height` | Image height in pixels (default: 1024, must be divisible by 16) |
| `--steps` | Inference steps (default: 9) |
| `--guidance-scale` | CFG scale (default: 0.0) |
| `--cfg-normalization` | CFG norm clamping (0.0 = disabled, 1.0-2.0 typical). Prevents over-amplification. |
| `--cfg-truncation` | CFG truncation threshold (1.0 = never, 0.5-0.8 typical). Stops CFG at this progress. |
| `--shift` | Scheduler shift/mu (default: 3.0) |
| `--dynamic-shift` | Calculate shift based on resolution (overrides --shift). Uses linear interpolation: base_shift=0.5 at 512x512, max_shift=1.15 at 2048x2048. |
| `--d-noise` | Sigma schedule scaling factor. <1.0 = sharper/more detail (try 0.95-0.98), >1.0 = softer/deeper colors (try 1.02-1.05). Default: 1.0 (no scaling). |
| `--seed` | Random seed |
| `--img2img` | Input image path for img2img generation |
| `--strength` | img2img strength: 0.0 (no change) to 1.0 (full regeneration) (default: 0.7) |

## prompt control

| Flag | Description |
|------|-------------|
| `--system-prompt` | System message |
| `--thinking-content` | Content inside `<think>...</think>` (triggers think block) |
| `--assistant-content` | Content after `</think>` |
| `--enable-thinking` | Add `<think></think>` structure to prompt |
| `--template` | Template name to use |

## lora

| Flag | Description |
|------|-------------|
| `--lora` | LoRA path with optional scale (path:scale). Repeatable. |

## skip layer guidance (slg)

| Flag | Description |
|------|-------------|
| `--slg-scale` | SLG scale (default: 0.0, recommended: 2.8) |
| `--slg-layers` | Layers to skip (default: 15,16,17,18,19) |
| `--slg-start` | Start SLG at this fraction of steps (default: 0.01) |
| `--slg-stop` | Stop SLG at this fraction of steps (default: 0.20) |

## rewriter

| Flag | Description |
|------|-------------|
| `--rewriter-use-api` | Use API backend for prompt rewriting |
| `--rewriter-api-url` | API URL for rewriter (defaults to --api-url) |
| `--rewriter-api-model` | Model ID for rewriter API (default: Qwen3-4B) |
| `--rewriter-vl-api-model` | Model ID for VL rewriting via API (e.g., qwen2.5-vl-72b-mlx) |
| `--rewriter-temperature` | Sampling temperature (default: 0.6) |
| `--rewriter-top-p` | Nucleus sampling threshold (default: 0.95) |
| `--rewriter-min-p` | Minimum probability threshold (default: 0.0, disabled) |
| `--rewriter-max-tokens` | Maximum tokens to generate (default: 512) |
| `--rewriter-timeout` | API request timeout in seconds (default: 120.0, VL models may need longer) |
| `--rewriter-no-vl` | Disable VL model selection in rewriter UI |
| `--rewriter-preload-vl` | Preload Qwen3-VL at startup for rewriting |

## vision conditioning (qwen3-vl)

| Flag | Description |
|------|-------------|
| `--vl-model-path` | Path to Qwen3-VL model (enables vision conditioning) |
| `--vl-device` | Device for VL model: cpu/cuda/auto (cpu recommended to save VRAM) |
| `--vl-alpha` | VL influence ratio (0.0=pure text, 1.0=pure VL, default: 0.3) |
| `--vl-hidden-layer` | Hidden layer to extract from VL model (default: -2) |
| `--vl-no-auto-unload` | Keep VL model loaded after extraction (uses more VRAM) |
| `--vl-blend-mode` | Blend strategy: interpolate/adain_per_dim/adain/linear/style_only/graduated/attention_weighted |

## FLUX.2 Klein

Use `--model-type flux2` to select.

| Flag | Description |
|------|-------------|
| `--flux2-model-name` | Model variant name (e.g., klein-4b, klein-9b) |
| `--flux2-model-path` | Path to FLUX.2 model directory |
| `--flux2-encoder-path` | Path to Qwen3 encoder |
| `--flux2-vae-path` | Path to VAE decoder |
| `--flux2-num-steps` | Number of inference steps |
| `--flux2-guidance` | Guidance scale |
| `--flux2-seed` | Random seed |
| `--flux2-output` | Output file path |
| `--flux2-input-image` | Input image path(s) for image editing (multiple allowed) |
| `--flux2-offload` | Enable CPU offload |
| `--flux2-no-offload` | Disable CPU offload |
| `--flux2-block-offload` | Enable block-level offload (incompatible with torch.compile and torchao) |

```bash
uv run scripts/generate.py --model-type flux2 \
  --flux2-model-name klein-4b \
  --flux2-model-path /path/to/flux2-klein \
  "A cat sitting on a windowsill"
```

## LTX-2

Use `--model-type ltx2` to select.

| Flag | Description |
|------|-------------|
| `--ltx2-model-path` | Path to LTX-2 model directory |
| `--ltx2-encoder-model-id` | Gemma3 encoder model ID or path |
| `--ltx2-num-frames` | Number of frames to generate (must be 8n+1) |
| `--ltx2-fps` | Frames per second for output video |
| `--ltx2-guidance-scale` | Guidance scale |
| `--ltx2-steps` | Number of inference steps |
| `--ltx2-lora-path` | Path to LoRA weights |
| `--ltx2-lora-scale` | LoRA scale factor |
| `--ltx2-audio` | Enable audio generation |
| `--ltx2-output` | Output file path |
| `--ltx2-save-embeddings` | Save precomputed embeddings to file |
| `--ltx2-load-embeddings` | Load precomputed embeddings from file |

### LTX-2 optimization flags

| Flag | Description |
|------|-------------|
| `--ltx2-text-encoder-device` | Device for Gemma3 encoder: cpu/cuda |
| `--ltx2-transformer-device` | Device for DiT: cpu/cuda |
| `--ltx2-vae-device` | Device for VAE: cpu/cuda |
| `--ltx2-quantize` | Transformer quantization: none/fp8 |
| `--ltx2-skip-cleanup` | Skip model cleanup after generation (keep in memory) |
| `--ltx2-gemma-variant` | Gemma3 encoder variant: bf16/8bit/q4-qat |

```bash
uv run scripts/generate.py --model-type ltx2 \
  --ltx2-model-path /path/to/ltx-video-2 \
  --ltx2-num-frames 41 --ltx2-fps 24 \
  --ltx2-gemma-variant q4-qat \
  "A golden retriever playing fetch in a park"
```

## qwen-image (all variants)

Unified configuration for all Qwen-Image variants. Use `--model-type` to select:
- `qwenimage-t2i` - Text-to-image generation (60-layer DiT)
- `qwenimage-edit` - Instruction-based image editing (8B DiT)
- `qwenimage-layered` - Multi-layer decomposition (deprioritized)

| Flag | Description |
|------|-------------|
| `--qwen-image-model-path` | Path to any Qwen-Image model |
| `--qwen-image-cpu-offload` | Enable CPU offload (required for RTX 4090) |
| `--qwen-image-layers` | Number of decomposition layers (layered variant only, default: 4) |
| `--qwen-image-steps` | Diffusion steps (variant default: t2i=40, edit=25, layered=50) |
| `--qwen-image-cfg-scale` | CFG scale (default: 4.0) |
| `--qwen-image-resolution` | Resolution (variant default: t2i=1024, edit=640, layered=640) |
| `--qwen-image-quantize-text-encoder` | Quantization for text encoder (Qwen2.5-VL-7B): none/4bit/8bit |
| `--qwen-image-quantize-transformer` | Quantization for DiT (variant default: t2i=fp8, edit/layered=diffsynth-fp8) |

### variant-aware defaults

When not specified, parameters use variant-specific defaults:

| Variant | Steps | Resolution | Transformer Quantization |
|---------|-------|------------|-------------------------|
| `qwenimage-t2i` | 40 | 1024 | fp8 |
| `qwenimage-edit` | 25 | 640 | diffsynth-fp8 |
| `qwenimage-layered` | 50 | 640 | diffsynth-fp8 |

### quantization recommendations (rtx 4090)

For RTX 4090 (24GB VRAM):
- **Text encoder**: Use `none` with CPU offload (best quality, 0 GPU VRAM)
- **Transformer**: Use variant default (fp8 for T2I, diffsynth-fp8 for edit/layered)

```bash
# T2I example (uses variant defaults)
uv run scripts/generate.py --model-type qwenimage-t2i \
  --qwen-image-model-path /path/to/Qwen-Image-2512 \
  "A mountain landscape"

# Edit example
uv run scripts/generate.py --model-type qwenimage-edit \
  --qwen-image-model-path /path/to/Qwen-Image-Edit-2511 \
  --img2img input.png "Change the sky to sunset"
```
