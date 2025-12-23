# cli flags reference

*last updated: 2025-12-23*

Shared between `web/server.py` and `scripts/generate.py`.

## model and config

| Flag | Description |
|------|-------------|
| `--model-type` | Model type: zimage (default) or qwenimage |
| `--model-path` | Path to Z-Image model |
| `--qwen-image-model-path` | Path to Qwen-Image-Layered model |
| `--config` | Path to TOML config file |
| `--profile` | Config profile to use (default: "default") |
| `--templates-dir` | Path to templates directory |

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
| `--use-api-encoder` | Use API for encoding (local is default) |

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
| `--dype-scale` | Scaling factor: 2.0 for 2K, 4.0 for 4K (default: 2.0) |
| `--dype-alpha` | Vision-YaRN alpha parameter (default: 1.0) |
| `--dype-beta` | Vision-YaRN beta parameter (default: 32.0) |

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
| `--seed` | Random seed |
| `--img2img` | Input image path for img2img generation |
| `--strength` | img2img strength: 0.0 (no change) to 1.0 (full regeneration) (default: 0.7) |

## prompt control

| Flag | Description |
|------|-------------|
| `--system-prompt` | System message |
| `--thinking-content` | Content inside `<think>...</think>` (triggers think block) |
| `--assistant-content` | Content after `</think>` |
| `--force-think-block` | Add empty think block even without content |
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
| `--vl-model-path` | Path to Qwen3-VL-4B model (Instruct or Thinking) |
| `--vl-model-variant` | Model variant: instruct/thinking/both (auto-detected from template) |
| `--vl-device` | Device for VL model (cpu recommended to save VRAM) |
| `--vl-alpha` | Default VL influence (0.0-1.0, default: 0.3) |
| `--vl-hidden-layer` | Hidden layer for VL extraction (default: -2, recommend: -6 for VL) |
| `--vl-auto-unload` | Unload VL model after extraction (default: true) |
| `--vl-blend-mode` | Blend mode: linear/adain/adain_per_dim/style_delta |
| `--vl-outlier-masking` | Outlier masking mode: none/zero/clamp/scale (default: none) |
| `--vl-outlier-threshold` | Std ratio threshold for outlier masking (default: 10.0) |

## qwen-image-layered

| Flag | Description |
|------|-------------|
| `--qwen-image-model-path` | Path to Qwen-Image-Layered model |
| `--qwen-image-edit-model-path` | Path to Qwen-Image-Edit model (auto-downloads if empty) |
| `--qwen-image-cpu-offload` | Enable CPU offload (recommended, ~5 GB VRAM) |
| `--qwen-image-layers` | Number of layers to decompose (1-10, default: 4) |
| `--qwen-image-steps` | Diffusion steps (default: 50) |
| `--qwen-image-cfg-scale` | CFG scale for Qwen-Image (default: 4.0) |
| `--qwen-image-resolution` | Resolution: 640 (recommended) or 1024 |
