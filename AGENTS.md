# agent context

*last updated: 2026-01-22*

Quick reference for LLM agents working on this codebase.

## multi-model platform

This repo supports **multiple LLM-DiT pipelines**, not just LTX-2:

| Pipeline | Task | Encoder | Status |
|----------|------|---------|--------|
| Z-Image | text-to-image, img2img | Qwen3-4B | Production |
| LTX-2 | text-to-video | Gemma3-12B | Active development |
| Qwen-Image-Layered | image decomposition | Qwen2.5-VL-7B | Production |
| Qwen-Image-Edit-2511 | instruction editing | Qwen2.5-VL-7B | Production |

LTX-2 has been the focus recently due to its complexity (pure PyTorch implementation instead of diffusers).

## start here

| Doc | Purpose |
|-----|---------|
| **[internal/state/current.md](internal/state/current.md)** | Current project state (read first) |
| **[experiments/AGENTS.md](experiments/AGENTS.md)** | Research status tracking |
| **[internal/guides/](internal/guides/)** | Model-specific guides (z_image_*, ltx2_*) |

## critical rules

- **no emojis** in code, docs, or output
- **use `uv`** for all Python ops (`uv add`, `uv run`, `uv sync`)
- **never commit** without explicit user approval
- **dtype conventions** - libraries differ:
  - transformers: use `dtype=`
  - diffusers: use `torch_dtype=`
- **always update `internal/state/`** after significant work (see below)

## state management (REQUIRED)

After completing significant work, ALWAYS update these files:

| File | When to Update | What to Update |
|------|----------------|----------------|
| `internal/state/current.md` | After any milestone or major change | Status, completion %, blockers, recent work |
| `internal/state/lessons_learned.md` | After debugging sessions, bug fixes | New lessons with full context |
| `internal/log/log_YYYY-MM-DD.md` | Every session | What was done, decisions made |

**What counts as "significant work":**
- Bug fixes (especially tricky ones)
- New features or components
- Architecture changes
- Documentation updates
- Test additions or fixes

**Do not skip this.** Future sessions depend on accurate state to avoid repeating work or missing context.

## architecture

```
Text Prompt -> TextEncoder -> hidden_states[layer(s)] -> DiT -> VAE -> Image/Video
```

Models use different text encoders (Qwen3-4B, Gemma3-12B, UMT5-XXL) and DiT variants. See [internal/models/overview.md](internal/models/overview.md) for comparison.

## key parameters

### z-image (turbo distilled)

| param | value | notes |
|-------|-------|-------|
| encoder | Qwen3-4B | 2560 hidden dim, 36 layers |
| layer | -2 | penultimate (`--hidden-layer`) |
| max tokens | 1504 | DiT RoPE limit |
| cfg | 0.0 | baked in (Decoupled-DMD) |
| steps | 8-9 | turbo distilled |
| scheduler | FlowMatchEuler | shift=3.0 |

### ltx-2 (video, active development)

| param | value | notes |
|-------|-------|-------|
| encoder | Gemma3-12B (Q4) | 3840 hidden dim, 49 layers |
| transformer | LTX-2 13B | 48 blocks, 32 heads |
| cfg | 4.0 | default with latent normalization |
| steps | 40 | T2V default |
| resolution | 512x768 | default, up to 1280 |
| frames | 121 | default (8*15+1), 33 for quick tests |
| rope | 3D INTERLEAVED | (T, H, W) positions |
| quantization | FP8-quanto | 26GB->12GB, fits 24GB GPU |

## navigation

### primary (read first)
- **[current.md](internal/state/current.md)** - current project state
- **[internal/guides/](internal/guides/)** - model-specific guides (Z-Image, LTX-2)
- [models/](internal/models/) - per-model knowledge base

### session state
- [todos](internal/state/todos.md) - pending work across sessions
- [lessons learned](internal/state/lessons_learned.md) - aggregated insights

### architecture
- [guiding principles](internal/principles/guiding_principles.md) - architectural north star
- [analysis/architecture/](internal/analysis/architecture/) - pipeline audits and summaries

### production generation reference docs (for all models in steady state / prod state)
- [cli flags](docs/reference/cli_flags.md) - all CLI arguments
- [api endpoints](docs/reference/api_endpoints.md) - REST API reference
- [configuration](docs/reference/configuration.md) - DRY config principles

## research status system

All research documentation uses a consistent status tracking system:

| Symbol | Status | Meaning |
|--------|--------|---------|
| ✅ | **Validated** | Confirmed through experiments or architecture analysis |
| 🔬 | **Open** | Hypothesis needs testing or re-testing |
| ⚠️ | **Needs Verification** | Previous results may have bugs |
| 🚫 | **Dead-End** | Tested, doesn't work |

**Where to find EXPERIMENTS and RESEARCH (not core code) tracking:**
- [experiments/AGENTS.md](experiments/AGENTS.md) - Top-level research navigation
- [experiments/ltx2/docs/findings/](experiments/ltx2/docs/findings/) - Consolidated research findings

## testing protocol

**Full testing guide:** See **[tests/AGENTS.md](tests/AGENTS.md)** for comprehensive testing documentation.

| Document | Purpose |
|----------|---------|
| **[tests/AGENTS.md](tests/AGENTS.md)** | Complete testing guide for agents |
| [tests/README.md](tests/README.md) | Test suite overview |
| [tests/backends/README.md](tests/backends/README.md) | Portable backend system for 1:1 comparison |

### test structure

| Category | Purpose | Location | Tests |
|----------|---------|----------|-------|
| **unit** | component-level tests | `tests/unit/` | ~500 |
| **integration** | cross-component tests | `tests/integration/` | ~200 |
| **e2e** | end-to-end pipeline tests | `tests/e2e/` | ~50 |
| **backends** | portable test backends | `tests/backends/` | - |

### quick test commands

```bash
# Quick smoke test (30s, requires GPU)
uv run pytest tests/e2e/test_baseline_portable.py::TestBaselineSmoke -v -s

# Unit tests only (no GPU)
uv run pytest tests/unit/ -v

# LTX-2 specific tests
uv run pytest tests/ -v -k ltx2

# Full test suite with slow tests
uv run pytest tests/ -v --runslow
```

### when adding new encoders/backends

1. **Read the guide:** [tests/AGENTS.md](tests/AGENTS.md) for testing patterns
2. **Shape validation:** Add tests to verify tensor shapes through the pipeline
3. **Weight initialization:** Test that weights are non-zero after loading
4. **Reference comparison:** Compare against official implementation when available

## LTX-2 T2V Test Scripts

Pure PyTorch T2V pipeline tests. No diffusers dependency.

| Script | Backend | Purpose | Command |
|--------|---------|---------|---------|
| `tests/e2e/test_baseline_portable.py` | llm_dit | Pure PyTorch e2e test | `uv run pytest tests/e2e/test_baseline_portable.py -v -s` |
| `tests/e2e/test_baseline_portable.py` | ltx2 | Reference baseline | `LLM_DIT_TEST_BACKEND=ltx2 uv run pytest ...` |

### Test Configs (tests/backends/protocol.py)

| Config | Frames | Resolution | Steps | CFG | FP8 | Est. Time |
|--------|--------|------------|-------|-----|-----|-----------|
| SMOKE_CONFIG | 33 | 512x768 | 30 | 3.0 | Yes | ~3min |
| SHORT_CONFIG | 33 | 512x768 | 30 | 3.0 | No | ~4min |
| REFERENCE_CONFIG | 121 | 512x768 | 40 | 4.0 | No | ~10min |

### Output Location

All test outputs go to: `outputs/tests/runs/{backend}_{test}_{timestamp}/`
- `video.mp4` - Generated video
- `inputs.json` - Complete generation parameters
- `metadata.json` - Timing and memory stats
- `generation.log` - Debug logs

### Code Independence

| Path | Diffusers | coderef/LTX-2 |
|------|-----------|---------------|
| `src/llm_dit/encoders/` | No | No |
| `src/llm_dit/models/ltx2/` | No | No |
| `src/llm_dit/pipelines/generate.py` | No | No |
| `tests/backends/llm_dit_backend.py` | No | No |
| `tests/backends/ltx2_backend.py` | Yes | Yes (intentional) |

### Success Criteria (CRITICAL)

**A test "passing" means NOTHING without visual inspection.**

| Level | Criteria | How to verify |
|-------|----------|---------------|
| **Technical** | Pipeline completes, shapes correct, no NaN/Inf | pytest assertions |
| **Semantic** | Output matches prompt (cat = cat, not muppet) | Human inspection of frames |
| **Temporal** | Motion is coherent (walking = movement across frames) | Watch video or compare frames |
| **Reference** | Matches official LTX-2 output (same seed) | Side-by-side comparison |

**Current status (2026-01-22):**
- Technical: PASS
- Semantic: PASS (fixed tokenizer path bug - now produces correct output)
- Temporal: PASS (cat shows walking motion across frames)
- Reference: NOT TESTED

**Verification workflow:**
```bash
# 1. Run test
uv run pytest tests/e2e/test_baseline_portable.py::TestBaselineSmoke -v -s

# 2. Extract frames for inspection
ffmpeg -i outputs/tests/runs/llm_dit_*/video.mp4 -vf "select=eq(n\,0)+eq(n\,16)+eq(n\,32)" -vsync vfr /tmp/frames_%02d.png

# 3. Compare to reference (requires ltx2 backend)
LLM_DIT_TEST_BACKEND=ltx2 uv run pytest tests/e2e/test_baseline_portable.py::TestBaselineSmoke -v -s
```

**Do not claim "working" until semantic criteria passes.**

## LTX-2 Comprehensive Test Suite

**~150+ tests protecting against regressions.** Run all critical paths with:

```bash
# All LTX-2 tests (unit + integration + e2e)
uv run pytest tests/ -v -k ltx2

# Quick smoke test only (~1 min on 24GB GPU)
uv run pytest tests/e2e/test_baseline_portable.py::TestBaselineSmoke -v -s
```

### Test Coverage Map

| File | What It Tests | Tests |
|------|---------------|-------|
| `tests/unit/test_ltx2_transformer.py` | RoPE, attention, FFN, AdaLN, timesteps, key mapping | ~20 |
| `tests/unit/test_gemma3_encoder.py` | Connector, feature extractor, pipeline shapes | ~25 |
| `tests/unit/test_conditioning.py` | LatentState, I2V/keyframe, denoise mask | ~45 |
| `tests/unit/test_ltx2_video_vae.py` | Video VAE compression, tiling, components | ~15 |
| `tests/unit/test_scheduler.py` | Sigma schedule, timestep conversion, callbacks | ~8 |
| `tests/integration/test_ltx2_connectors.py` | RoPE, state dict mapping, layer variance | ~15 |
| `tests/integration/test_ltx2_gpu_integration.py` | Weight loading, forward pass, memory | ~10 |
| `tests/integration/test_ltx2_numerical_equivalence.py` | Diffusers vs our impl equivalence | ~12 |
| `tests/integration/test_ltx2_e2e_generation.py` | Config, positions, modality, full pipeline | ~15 |
| `tests/integration/test_performance.py` | Timing, memory bounds, no regressions | ~8 |
| `tests/e2e/test_ltx2_reference.py` | Smoke, T2V reference, I2V conditioning | ~4 |
| `tests/e2e/test_baseline_portable.py` | Portable backend, 1:1 comparison | ~4 |

### What's Validated

- **Transformer internals** - RoPE, attention, FFN, AdaLN, timestep embedding
- **Key mapping** - Diffusers state dict to our format
- **Numerical equivalence** - Against official LTX-2 implementation
- **Conditioning system** - I2V, keyframe continuation, denoise mask mechanics
- **Pipeline shapes** - Token counts, latent dimensions, position indices
- **Memory efficiency** - Leak detection, GPU cleanup verification
- **VAE components** - Compression ratios, causal convolutions, tiling

### Reference Values

Tests validate against official LTX-2 constants:
- `DEFAULT_SEED = 10`
- `LATENT_DIM = 128`
- `HIDDEN_DIM = 4096`
- Spatial compression: 32x
- Temporal compression: 8x

## adding parameters

Config flows through single chain:
```
config.toml -> Config dataclass -> RuntimeConfig -> startup.py -> Backend configs
```

See `docs/reference/configuration.md` for checklist.
