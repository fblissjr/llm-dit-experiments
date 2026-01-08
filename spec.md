# llm-dit-experiments specification

*last updated: 2026-01-08 (backlog reorganized)*

Standalone diffusers-based experimentation platform for LLM-DiT image generation.

## current state

### implemented features

- [x] Z-Image pipeline (Qwen3-4B encoder, turbo distilled 8-9 steps)
- [x] Qwen-Image-Layered pipeline (decomposition, editing)
- [x] Qwen-Image-T2I pipeline (text-to-image)
- [x] Qwen-Image-Edit pipeline (instruction-based editing)
- [x] Web UI with REST API
- [x] LoRA support with automatic weight fusion
- [x] DyPE high-resolution generation (2K+)
- [x] Skip Layer Guidance (SLG)
- [x] FMTT test-time scaling
- [x] Long prompt compression (4 modes)
- [x] Attention backend auto-detection (FA2/3, Sage, xFormers, SDPA)
- [x] FP8/INT8/4bit quantization options
- [x] Config management UI
- [x] D-noise sigma schedule scaling

### known limitations

- VL zero-shot style transfer corrupts image content (non-viable without training)
- Max tokens: 1504 (DiT RoPE limit)
- Qwen-Image-Layered layer 0 often buggy

---

## backlog

### p0: verification and testing

- [x] **Clean up scripts/ directory** (completed 2026-01-08)
  - Archived 6 one-off test scripts to `scripts/archive/`
  - Kept 11 core/utility scripts
  - **Kept**: generate.py, train.py, profiler.py, embeddings.py, smoke_test.py, check_fp8_compatibility.py, quantize_model.py, clear-cuda.py, install_sageattention.sh, start-server.sh, stop-server.sh
  - **Archived**: test_qwen_image_2512.py, test_qwen_image_edit_2511.py, test_fp8_load.py, test_quantized_dit.py, convert_image_encoders.py, test_qwen_image_2512_standalone.py

- [ ] **Test DyPE multipass at 2K+ resolution**
  - Real-world testing with twopass and threepass modes
  - Verify frequency modulation works correctly
  - Document quality vs performance tradeoffs
  - Tests: `tests/integration/test_dype_multipass.py`

- [ ] **Benchmark VRAM reduction strategies**
  - Compare group offloading vs model offloading
  - Test TorchAO FP8 vs BitsAndBytes 4bit quality
  - Measure latency impact of CPU offload
  - Document recommendations by GPU tier

### p1: web ui improvements

- [ ] **Web UI DyPE controls**
  - Add missing multipass UI controls
  - Add frequency modulation toggle
  - Show DyPE recommendations based on resolution
  - Tests: `tests/unit/test_web_dype_controls.py`

- [ ] **History panel improvements**
  - Add batch operations (select multiple, delete)
  - Add export to folder functionality
  - Add metadata display on hover

### p2: performance optimizations

- [ ] **Community best practices research**
  - Z-Image Turbo: optimal shift, d_noise, DyPE params for 2K+
  - Qwen-Image-Edit-2511: best CFG, steps, quantization tradeoffs
  - Qwen-Image-2512: prompt enhancement, resolution constraints
  - Sources: GitHub issues, ComfyUI (RES4LYF, DyPE), DiffSynth examples, HuggingFace
  - Output: `internal/research/community_best_practices.md`

- [ ] **Coderef updates analysis**
  - Review `./coderef/diffusers` - Z-Image pipelines, Qwen-Image support
  - Review `./coderef/transformers` - Qwen model updates
  - Review `./coderef/DiffSynth-Engine` - FP8, forward block cache, Nunchaku
  - Review `./coderef/DiffSynth-Studio` - pipeline patterns
  - Key findings: forward block cache, Z-Image Omni variant, vision_yarn
  - Output: `internal/research/coderef_updates_2026_01.md`

- [ ] **RoPE caching (LightX2V pattern)**
  - Implement two-level caching (base freq + resolution-specific)
  - Use `@functools.lru_cache` for resolution cache
  - Expected: 5-10% inference speedup
  - Tests: `tests/unit/test_rope_caching.py`

- [ ] **Device capability detection**
  - Add cleaner hardware capability detection
  - Auto-configure based on GPU compute capability
  - Support Ada (8.x), Hopper (9.x), Blackwell (12.x)
  - Tests: `tests/unit/test_device_caps.py`

- [ ] **Pinned memory for async offload**
  - Implement pinned CPU tensors for faster transfer
  - Foundation for stream manager
  - Tests: `tests/unit/test_pinned_memory.py`

### p3: new features

- [ ] **LightX2V distillation LoRAs**
  - Test Qwen-Image-2512-Lightning LoRA
  - Test Qwen-Image-Edit-2511-Lightning LoRA
  - Implement fixed timestep scheduler (4 steps)
  - Document quality tradeoffs
  - Tests: `tests/integration/test_distillation_lora.py`

- [ ] **VL step-based conditioning**
  - Try VL influence only at specific timesteps
  - Experiment with lower alpha (0.1-0.3)
  - Research trained adapter approaches

- [ ] **Advanced sampler techniques (RES4LYF)**
  - Add noise_mode parameter (hard/soft/sinusoidal)
  - Add s_noise multiplier
  - Add DetailBoost-style noise lying
  - Tests: `tests/unit/test_sampler_modes.py`

### p4: architecture improvements

- [ ] **Unified QwenImagePipeline**
  - Consolidate 3+ pipeline wrappers into one
  - Follow DiffSynth single-class pattern
  - Mode selection via optional parameters
  - Tests: `tests/unit/test_unified_pipeline.py`

- [ ] **Parameter registry pattern**
  - Define params once with metadata
  - Auto-generate CLI args, Pydantic models
  - Reduce "places to add param" from 7-8 to 1-2
  - Tests: `tests/unit/test_param_registry.py`

- [ ] **Stream manager for async offload**
  - Dual-buffer CUDA stream architecture
  - Transfer/compute overlap
  - Major rewrite - high effort
  - Expected: 30-50% speedup for CPU offload scenarios

### p5: research / exploration

- [ ] **LTX-2 video model for RTX 4090**
  - 19B DiT-based audio-video foundation model (Lightricks)
  - Generates synchronized video + audio
  - Model location: `~/Storage/LTX-2`
  - Variants:
    | Variant | VRAM Est. | Notes |
    |---------|-----------|-------|
    | ltx-2-19b-dev-fp8 | ~20-25GB | FP8, should fit RTX 4090 |
    | ltx-2-19b-dev-fp4 | ~10-12GB | NVFP4, definitely fits |
    | ltx-2-19b-distilled | ~20-30GB | 8 steps, CFG=1 |
  - Components: Gemma3 encoder, LTX2VideoTransformer3D, Video/Audio VAEs, Vocoder
  - Research tasks:
    - [ ] Analyze FP8/FP4 VRAM usage on RTX 4090
    - [ ] Test distilled variant performance
    - [ ] Understand pipeline architecture
    - [ ] Identify integration points with llm-dit-experiments
  - Output: `internal/research/ltx2_video_analysis.md`

---

### future / deferred

Items parked for later consideration:

- [ ] **FMTT real-world testing**
  - Test with SigLIP reward model
  - Benchmark quality improvement vs latency
  - Document when FMTT is worth the cost

- [ ] **Training infrastructure**
  - DiffSynth-based training setup
  - LoRA fine-tuning workflow
  - See: `internal/research/training_infrastructure_design.md`

---

## development workflow

### TDD process

1. Pick a backlog item
2. Create/update test file with failing test
3. Run: `uv run pytest -k "test_name" -v` (watch fail)
4. Implement the feature
5. Run: `uv run pytest -k "test_name" -v` (watch pass)
6. Update this spec (check off TODO)
7. Update relevant docs
8. Commit: `git commit -m "feat: description"`

### test commands

```bash
# Run all tests
uv run pytest tests/

# Run unit tests only
uv run pytest tests/unit/ -v

# Run specific test file
uv run pytest tests/unit/test_dype.py -v

# Run tests matching pattern
uv run pytest -k "dype" -v

# Run with coverage
uv run pytest --cov=src/llm_dit tests/
```

### key test files

| Area | Test File |
|------|-----------|
| DyPE | tests/unit/test_dype.py |
| Config | tests/unit/test_dry_config.py |
| Pipeline | tests/unit/test_z_image_pipeline.py |
| Scheduler | tests/unit/test_scheduler.py |
| Quantization | tests/unit/test_fp8_inference.py |

---

## references

- [Session state](internal/state/session_continuity.md) - Current focus
- [TODOs](internal/state/todos.md) - Persistent task tracking
- [Lessons learned](internal/state/lessons_learned.md) - Gotchas and solutions
- [Guiding principles](internal/principles/guiding_principles.md) - Architectural north star
