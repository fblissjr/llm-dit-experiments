# Session Continuity

> **Last Updated**: 2025-12-02 | **Status**: Phase 1 Complete - Backend Loading Verified

---

## Quick Start (New Session)

**"Where did we leave off?"** - Read this section first.

### Current State

- **Phase 1 COMPLETE**: All core infrastructure implemented and tested
- **Smoke test passing**: Imports, templates, conversation, AND backend loading verified
- **Backend tested on macOS MPS**: Loads model, encodes text, outputs correct dimensions
- **Ready for Phase 2**: Pipeline implementation and first experiments

### What's Working

- `TextEncoderBackend` Protocol with duck typing
- `TransformersBackend` with attention mask filtering (matches DiffSynth)
  - Auto-detects best device (CUDA, MPS, CPU)
  - Handles diffusers-style layout (separate tokenizer/ and text_encoder/ folders)
  - Fixed transformers subfolder=None bug
- `EncodingOutput` with variable-length embeddings
- `Conversation` types with `Message`, `Role`
- `Qwen3Formatter` for chat template formatting
- `format_prompt()` convenience function
- Template loader with YAML frontmatter support
- 144 Z-Image templates copied from ComfyUI
- Full research documentation in `docs/research/`

### What's Next (Phase 2)

1. Implement `ZImageTextEncoder` (wires backend + templates + formatter)
2. Implement `ZImagePipeline` (wraps diffusers with our encoder)
3. Run first experiment: thinking block effect on embeddings
4. Test full end-to-end generation on RTX 4090 Ubuntu server

---

## Architecture Summary

```
Prompt -> Qwen3Formatter -> TextEncoderBackend -> embeddings -> diffusers pipeline -> Image
           |                     |
           |                     +-- TransformersBackend (hidden_states[-2])
           |                         - Max 512 tokens
           |                         - Attention mask filtering
           |                         - Variable-length output
           +-- Chat template: <|im_start|>system/user/assistant<|im_end|>
               - Thinking: <think>...</think>
               - Empty think tags when enable_thinking=True but content empty
```

Key insight: We handle the LLM encoding layer, diffusers handles DiT + VAE.

---

## Key Technical Details (from research)

| Parameter | Value | Source |
|-----------|-------|--------|
| Text encoder | Qwen3-4B, 2560 dim | config |
| Embedding extraction | hidden_states[-2] | DiffSynth |
| Max sequence length | 512 tokens | DiffSynth/diffusers |
| Attention mask filtering | Remove padding tokens | DiffSynth |
| CFG scale | 0.0 (baked in via DMD) | Paper analysis |
| Steps | 9 (= 8 NFEs) | Official README |
| Scheduler | FlowMatchEuler, shift=3.0 | config |
| Timestep inversion | t = (1000 - t) / 1000 | DiffSynth |
| Output negation | output = -output | DiffSynth |
| VAE | 16-channel | config |
| Resolution alignment | 16 pixels | DiffSynth |

---

## Key Files

| File | Purpose | Status |
|------|---------|--------|
| `src/llm_dit/backends/protocol.py` | TextEncoderBackend Protocol | Done |
| `src/llm_dit/backends/transformers.py` | HuggingFace backend | Done |
| `src/llm_dit/backends/config.py` | BackendConfig | Done |
| `src/llm_dit/conversation/types.py` | Message, Conversation | Done |
| `src/llm_dit/conversation/formatter.py` | Qwen3Formatter | Done |
| `src/llm_dit/templates/loader.py` | Template loading | Done |
| `src/llm_dit/templates/registry.py` | Template registry | Done |
| `templates/z_image/` | 144 templates | Done |
| `docs/research/` | Research documentation | Done |

---

## Model Locations

- **Z-Image model**: `~/Storage/Tongyi-MAI_Z-Image-Turbo/` or HuggingFace
- **Reference code**: `~/workspace/ComfyUI-QwenImageWanBridge/coderef/`

---

## Commands

```bash
cd ~/workspace/llm-dit-experiments
uv sync                          # Install dependencies
uv run scripts/smoke_test.py     # Validate setup (passes)

# With model (on Ubuntu server with RTX 4090):
uv run scripts/smoke_test.py --model-path ~/Storage/Tongyi-MAI_Z-Image-Turbo
```

---

## Research Documents

Key research from ComfyUI project copied to `docs/research/`:

| Document | Key Findings |
|----------|--------------|
| `z_image_diffsynth_analysis_20251201.md` | DiffSynth patterns, attention mask filtering |
| `z_image_model_analysis_20251201.md` | 3D RoPE coordinate system, DiT architecture |
| `z_image_attention_mask_filtering_design.md` | Padding filter implementation spec |
| `decoupled_dmd_training_report.md` | CFG baking via CFG Augmentation |

---

## Recent Changes

### 2025-12-02: Backend Loading Verified
- Fixed TransformersBackend path handling for diffusers-style model layout
- Added auto-detect for CUDA/MPS/CPU device
- Fixed transformers subfolder=None bug (omit param entirely when None)
- Added `format_prompt()` convenience function
- Smoke test now passes with actual model loading and encoding on MPS
- Verified: 22 tokens encoded to torch.Size([22, 2560]) on mps:0

### 2025-12-01: Phase 1 Complete
- Smoke test passing
- All Phase 1 infrastructure implemented
- Research documentation integrated
- Ready for Phase 2 (pipeline + experiments)

### 2025-12-01: Research Integration
- Read DiffSynth analysis and model analysis reports
- Updated implementation to match reference (attention mask filtering)
- Copied research docs to project

### 2025-12-01: Project Initialization
- Created project structure
- Set up pyproject.toml with uv
- Created living documentation framework
- Implemented TextEncoderBackend Protocol

---

## Blockers / Open Questions

1. ~~Need to verify local model path for Z-Image~~ - Available on Ubuntu server
2. ~~Should we optimize for batch or single-prompt encoding first?~~ - Single prompt for now
3. Quantization effects on embedding quality (experiment later)
4. 512 token limit: hard architectural constraint or just conservative? (See model analysis)
