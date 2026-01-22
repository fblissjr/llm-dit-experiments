# agent context

*last updated: 2026-01-22*

Quick reference for LLM agents working on this codebase.

## start here

| Doc | Purpose |
|-----|---------|
| **[internal/hub.md](internal/hub.md)** | Central documentation hub |
| **[internal/state/current.md](internal/state/current.md)** | Current project state (read first) |
| **[experiments/AGENTS.md](experiments/AGENTS.md)** | Research status tracking |

## critical rules

- **no emojis** in code, docs, or output
- **use `uv`** for all Python ops (`uv add`, `uv run`, `uv sync`)
- **never commit** without explicit user approval
- **dtype conventions** - libraries differ:
  - transformers: use `dtype=`
  - diffusers: use `torch_dtype=`

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
- **[hub.md](internal/hub.md)** - central documentation hub
- **[current.md](internal/state/current.md)** - current project state
- [models/](internal/models/) - per-model knowledge base

### session state
- [todos](internal/state/todos.md) - pending work across sessions
- [lessons learned](internal/state/lessons_learned.md) - aggregated insights

### architecture
- [guiding principles](internal/principles/guiding_principles.md) - architectural north star
- [full docs index](internal/index.md) - complete navigation map

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

**Critical for new encoder/backend work:** See [tests/backends/README.md](tests/backends/README.md) for the testing protocol.

### test structure

| category | purpose | location |
|----------|---------|----------|
| **unit** | component-level tests | `tests/unit/` |
| **integration** | cross-component tests | `tests/integration/` |
| **e2e** | end-to-end pipeline tests | `tests/e2e/` |
| **backends** | portable test backends | `tests/backends/` |

### key test files

| file | purpose |
|------|---------|
| `tests/backends/README.md` | Backend protocol testing requirements |
| `tests/e2e/test_pipeline_shapes.py` | Pipeline shape validation (traces tensors through stages) |
| `tests/e2e/test_ltx2_reference.py` | Reference comparison against official implementation |
| `tests/e2e/test_baseline_portable.py` | Baseline tests that work with both implementations |
| `tests/unit/test_gemma3_encoder.py` | Example unit tests for encoder components |

### when adding new encoders/backends

1. **Read the protocol:** `tests/backends/README.md` defines required tests
2. **Shape validation:** Add tests to verify tensor shapes through the pipeline
3. **Weight initialization:** Test that weights are non-zero after loading
4. **Reference comparison:** Compare against official implementation when available

### running tests

```bash
# Quick smoke test (30s)
uv run pytest tests/e2e/test_baseline_portable.py::TestBaselineSmoke -v -s

# Shape validation for pipeline stages
uv run pytest tests/e2e/test_pipeline_shapes.py -v

# Full reference test with slow tests enabled (10min)
uv run pytest tests/e2e/test_baseline_portable.py --runslow -v -s

# Unit tests for specific component
uv run pytest tests/unit/test_gemma3_encoder.py -v
```

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

## adding parameters

Config flows through single chain:
```
config.toml -> Config dataclass -> RuntimeConfig -> startup.py -> Backend configs
```

See `docs/reference/configuration.md` for checklist.
