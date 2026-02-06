last updated: 2026-02-06

# changelog

All notable changes to this project will be documented in this file.
Uses [Semantic Versioning](https://semver.org/).

## 0.6.3

### changed
- Default `compile_mode` from `max-autotune-no-cudagraphs` to `default` for FLUX.2 and global optimization
  - Eliminates 5+ min Triton autotune warmup when combined with FP8 quantization
  - `default` mode still applies Inductor graph optimizations (kernel fusion, dead code elimination)
  - Users wanting maximum throughput can still set `max-autotune-no-cudagraphs` in config.toml

### added
- Compile+FP8 autotune warning in ModelCard: warns when `max-autotune` modes are used with quantization

## 0.6.2

### added
- FLUX.2 config visibility in ModelCard: colored badges for active optimization settings (FP8, compile, block_offload)
- Proactive config validation: incompatible settings (compile+block_offload, quantization+block_offload) shown as warnings before loading
- Generic data-driven config tag/warning system: backend provides tags, frontend renders them (pipeline-agnostic)

### fixed
- ModelCard VRAM display: backend now returns `vram_mb` alias (frontend expected this but only `total_vram_mb` was returned)

## 0.6.1

### added
- FLUX.2 persistent model loading: models stored in memory across requests, eliminating ~5-10s load per request
- FLUX.2 torchao FP8 quantization: transformer VRAM drops from ~18GB (BF16) to ~9GB (FP8)
- FLUX.2 encoder pinned memory shuttle: DMA-based CPU<->GPU transfers for encoder, ~2-3x faster than default
- FLUX.2 torch.compile support for transformer and VAE decoder
- `get_encoder_preset()` helper in FLUX.2 constants for DRY preset resolution
- Loading lock (`_flux2_loading_lock`) for concurrent `/api/vram/load-flux2` request safety

### fixed
- Pinned memory lost after first CUDA round-trip: `offload()` now uses shadow buffer pattern to copy CUDA tensors directly into pre-allocated pinned buffers (0 allocations vs 2N per cycle)
- Partial model cleanup on load failure: leaked encoder/transformer memory now explicitly freed
- DMA transfer now overlaps with tokenization in encoder `forward()` (~0.5-1s latency reduction)

### changed
- `quantize_to` + `block_offload` now raises `ValueError` instead of silently skipping quantization
- Added `RuntimeError` guard in transformer `forward()` for compile + block_offload incompatibility
- All logging in transformer `forward()` wrapped with `torch.compiler.is_compiling()` guards

## 0.2.0

### added
- Z-Image pipeline with Qwen3-4B encoder
- LTX-2 video pipeline with Gemma3-12B encoder
- Unified Qwen3 encoder (`qwen3_unified.py`) with preset system
- Multi-pipeline VRAM management with load/unload endpoints
- Block offload support for FLUX.2 transformer

## 0.1.0

### added
- Initial FLUX.2 Klein pipeline (4B and 9B variants)
- FastAPI server with React frontend
- Config system (TOML -> dataclass -> RuntimeConfig)
