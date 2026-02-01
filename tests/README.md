# tests

*last updated: 2026-01-24*

Comprehensive test suite with **~1030 tests** protecting against regressions.

## quick start

```bash
# Run all tests
uv run pytest tests/ -v

# Quick smoke test (30s, requires GPU)
# NOTE: LTX-2 only right now
uv run pytest tests/e2e/test_baseline_portable.py::TestBaselineSmoke -v -s

# Unit tests only (no GPU needed)
uv run pytest tests/unit/ -v
```

## documentation

| Document | Purpose |
|----------|---------|
| **[CLAUDE.md](CLAUDE.md)** | Comprehensive testing guide for agents |
| [backends/README.md](backends/README.md) | Portable backend system for 1:1 comparison |

## structure

```
tests/
├── CLAUDE.md              # Testing guide for agents (start here)
├── README.md              # This file
├── unit/                  # Fast, no GPU required (~500 tests)
│   ├── test_ltx2_*.py     # LTX-2 components
│   ├── test_gemma3_*.py   # Gemma3 encoder
│   ├── test_conditioning.py
│   ├── test_scheduler.py
│   └── ...
├── integration/           # Cross-component tests (~200 tests)
│   ├── test_ltx2_*.py
│   ├── test_performance.py
│   └── ...
├── e2e/                   # End-to-end pipeline tests (~50 tests)
│   ├── test_baseline_portable.py
│   ├── test_ltx2_reference.py
│   └── ...
├── backends/              # Portable test infrastructure
│   ├── protocol.py        # Backend interface
│   ├── llm_dit_backend.py
│   ├── ltx2_backend.py
│   └── README.md
├── fixtures/              # Test data
└── conftest.py            # Shared pytest fixtures
```

## test categories

| Category | Tests | GPU | Time | What's Validated |
|----------|-------|-----|------|------------------|
| **Unit** | ~500 | No | ~30s | Component logic, shapes, constraints |
| **Integration** | ~200 | Sometimes | ~2min | Cross-component interaction, memory |
| **E2E** | ~50 | Yes | ~5min | Full pipeline, visual quality |

## common commands

```bash
# LTX-2 specific tests
uv run pytest tests/ -v -k ltx2

# Run with verbose output
uv run pytest tests/unit/test_ltx2_transformer.py -v --tb=long

# Skip slow tests (default behavior)
uv run pytest tests/ -v

# Include slow tests
uv run pytest tests/ -v --runslow

# Collect tests without running
uv run pytest tests/ --collect-only
```

## key test files

| File | What It Tests |
|------|---------------|
| `unit/test_ltx2_transformer.py` | RoPE, attention, FFN, key mapping |
| `unit/test_ltx2_video_vae.py` | VAE compression, tiling, convolutions |
| `unit/test_gemma3_encoder.py` | Gemma3 connector, feature extraction |
| `unit/test_conditioning.py` | LatentState, I2V, denoise masks |
| `unit/test_scheduler.py` | Sigma schedule, dynamic shift |
| `integration/test_performance.py` | Memory leaks, timing bounds |
| `e2e/test_baseline_portable.py` | Full T2V pipeline |

## flux.2 test inventory

| File | Scenario | Status |
|------|----------|--------|
| `unit/test_flux2_transformer.py` | Double/single stream blocks, modulation | Tested |
| `unit/test_flux2_rope.py` | 4D RoPE, attention | Tested |
| `unit/test_flux2_vae.py` | VAE encode/decode | Tested |
| `integration/test_flux2_encoder.py` | Qwen3 encoder integration | Tested |
| `e2e/test_flux2_generation.py` | Klein-4B/9B basic generation | Tested |
| `e2e/test_flux2_fp8_offload.py` | **FP8 + block offload (OOM fix)** | **New** |

### fp8 + block offload test coverage (new)

| Scenario | Test | Status |
|----------|------|--------|
| Klein-9B-FP8 load with offload | `test_fp8_loading_with_block_offload_no_oom` | Tested |
| Klein-9B-FP8 load without offload | `test_fp8_loading_without_block_offload` | Tested |
| FP8 + offload smoke generation | `test_fp8_offload_smoke_generation` | Tested |
| VRAM bounded during denoising | `test_vram_stays_bounded_during_denoising` | Tested |
| FP8 variants (distilled, base) | `test_fp8_variants_with_offload` | Tested |

## success criteria

**Tests passing is necessary but not sufficient.**

| Level | Criteria | How to Verify |
|-------|----------|---------------|
| **Technical** | No errors, correct shapes | pytest assertions |
| **Semantic** | Output matches prompt | Visual inspection |
| **Temporal** | Motion is coherent | Watch video |

See [CLAUDE.md](CLAUDE.md) for detailed verification workflow.
