last updated: 2026-03-08

# changelog


All notable changes to this project will be documented in this file.
Uses [Semantic Versioning](https://semver.org/).

## 0.9.25

### removed
- **GGUF infrastructure (~1,380 lines across 16 files):** Deleted `gguf_dequant.py`, `gguf_loader.py`, `gguf_linear.py`, `gguf_tensor.py`, `audit_gguf_keys.py`, and all GGUF test files. Neither the official LTX-2 repo nor DiffSynth-Studio uses GGUF -- they use fp8-cast exclusively.
- **GGUF LoRA functions:** Removed `attach_lora_deltas()`, `detach_lora_deltas()`, `load_lora_for_gguf()`, `is_gguf_model()` from `lora.py` (~131 lines).
- **GGUF model loader:** Removed `load_ltx2_transformer_gguf()` from `models/ltx2/loader.py` (~76 lines).
- **GGUF config fields:** Removed `gguf_transformer_path` from `LTX2Config`, `config.toml`, `config.toml.example`.
- **GGUF ModelManager state:** Removed `_ltx2_gguf_model` attribute, `ltx2_gguf_model` property, `_preload_ltx2_gguf_transformer()` method.
- **torchao quantization for LTX-2 transformer:** Removed `quantize_component()` call from `_reconstruct_transformer_from_cache()`. LTX-2 transformer now uses fp8-cast (official approach) or plain bf16 only. torchao quantization remains available for FLUX.2, Gemma3, and other pipelines.
- **`gguf` dependency** from `pyproject.toml`.

### fixed
- **Connector gated attention bugs (garbled video):** Two errors in `embeddings_connector.py` Attention class: (1) missing `2.0x` multiplier on sigmoid gate -- `torch.sigmoid(gate)` should be `2.0 * torch.sigmoid(gate)`. Zero-initialized gates should produce identity (2*0.5=1.0), but without 2x, all attention outputs were attenuated by 0.5x per block. After 8 blocks: (0.5)^8 = 0.004x of correct magnitude. (2) Gating was applied AFTER `to_out` projection instead of BEFORE (as in the main DiT attention and DiffSynth/official reference). Moved gating into the `Attention.forward()` method before `to_out`, matching `attention.py:518-528`.
- **Connector RoPE max_pos bug (garbled video):** `connector_positional_embedding_max_pos` was hardcoded to `[1]` in `gemma3.py` and `embeddings_connector.py`, but the model was trained with `[4096]` (per safetensors metadata, DiffSynth-Studio, and Diffusers). With `[1]`, position 255 maps to raw value 255.0 in RoPE; with `[4096]` it maps to ~0.062 -- completely wrong positional encoding in the text embedding connector, scrambling all conditioning embeddings fed to the transformer. Fixed in `gemma3.py`, `embeddings_connector.py` (`from_config` default and `__init__` fallback).
- **Upsampler default config `mid_channels` mismatch:** `DEFAULT_UPSAMPLER_CONFIG` in `upsampler/loader.py` had `mid_channels: 512` but the actual `ltx-2.3-spatial-upscaler-x2-1.0.safetensors` metadata specifies `1024`. The metadata override masked this at runtime, but a metadata read failure would have created the wrong architecture. Fixed default to `1024`.
- **Config dataclass default upsampler filename:** `spatial_upsampler_file` default was `"ltx-2-spatial-upscaler-x2-1.0.safetensors"` (V2 naming) but the actual file is `"ltx-2.3-spatial-upscaler-x2-1.0.safetensors"`. `config.toml` had the correct name, masking the stale default.
- **Audio neg embeddings dimension bug (ROOT CAUSE of garbled video):** `generate.py` extracted `.embeddings` (video, 4096-dim) instead of `.audio_embeddings` (audio, 2048-dim) for audio negative prompt CFG. The `nn.Identity()` caption projection silently passed 4096-dim through, and `.view(B, -1, 2048)` reshaped to double-length garbled context. CFG amplified the garbage (x6 for audio), corrupting both audio and video via cross-modal attention. Fixed to use `.audio_embeddings`.
- **Cross-modal AdaLN uses wrong sigma source:** `_prepare_cross_modal_args` in `transformer.py` used each modality's OWN sigma for its cross-modal timestep AdaLN, but the reference uses the OTHER modality's sigma. Fixed: video cross-modal now uses audio's sigma and vice versa. Also added missing `av_ca_factor` gate scaling. Both bugs were masked for uniform sigma schedules (standard T2V+audio) but would affect per-token masking paths.
- **scipy dependency removed:** Replaced `scipy.io.wavfile` with stdlib `wave` module for WAV file writing in `web/routers/ltx2.py`.
- **`timesteps_from_mask` double-scaling bug:** Removed spurious `* 1000` from `conditioning/utils.py`. The transformer's `_prepare_timestep()` already multiplies by `timestep_scale_multiplier` (1000). This caused 1000x over-scaled AdaLN conditioning in i2v/keyframe paths. Standard text-to-video was unaffected. Matches official LTX-2 reference.
- **`load_ltx2_transformer()` missing `assign=True`:** Added `assign=True` to `load_state_dict()` in the bf16/standard model loading path. Prevents silent fp8->bf16 cast if mixed-dtype tensors land in this path. Matches official reference.
- **Gradient estimation `prev_velocity` storage bug:** GE was storing the corrected velocity as `prev_velocity` instead of the raw (pre-GE) model output. This caused compounding correction errors across steps when `ge_gamma > 0`. Fixed in both single-stage and AV denoising loops. Default `ge_gamma=0.0` means this bug was dormant in standard generation.
- **Audio noise not seeded:** `torch.randn_like(audio_latents)` for stage 2 re-noising ignored the generator seed. Replaced with `torch.randn(..., generator=generator)` for reproducible audio generation.
- **FPS hardcoded in two-stage pipeline:** Extracted `fps = 24.0` as a traced constant, replacing scattered `24.0` literals in `create_position_indices` and `compute_audio_latent_frames` calls.

### added
- **Two-stage 64-divisibility guard:** `_validate_two_stage_dimensions()` in `generate.py` raises `ValueError` if height/width are not divisible by 64. Called at the top of `generate_video_two_stage()`. Web schema already enforces this via `snap_to_64`; this guards direct pipeline callers (scripts, tests, CLI).
- **Connector config validation:** `_validate_connector_config()` in `gemma3.py` reads safetensors metadata at connector load time and warns on config mismatches. Guards against silent config drift (like the max_pos [1] vs [4096] bug).
- **Connector gated attention regression tests:** 10 tests in `test_connector_gated_attention.py` covering 2x sigmoid identity, gate placement, scaling range, and config validation.

### changed
- **`max_sequence_length` default 256 -> 512, now configurable:** Added `max_sequence_length` field to `LTX2Config` (config.toml, config.toml.example). Encoder now processes up to 384 real text tokens (512 minus 128 learnable registers). Previous default of 256 left only 128 real tokens, potentially truncating detailed prompts. Reference uses 1024; 512 is a practical middle ground for RTX 4090.
- **FPS plumbed through config:** Added `fps` field to `GenerationConfig`, populated from `LTX2Config.fps` in the router. Eliminated all hardcoded `fps=24.0` literals from `generate.py` (3 occurrences across single-stage, two-stage stage 1, and stage 2). FPS now flows: `config.toml` -> `LTX2Config.fps` -> `GenerationConfig.fps` -> `create_position_indices` / `compute_audio_latent_frames`.
- **`_load_transformer_and_lora()` signature:** Removed `gguf_model` parameter and `is_gguf` return value. 3-way dispatch (GGUF/cache/disk) simplified to 2-way (cache/disk).
- **`generate_video_with_offloading()` and `generate_video_two_stage()`:** Removed `gguf_model` parameter and GGUF cleanup branches.
- **Documentation cleanup:** Removed stale GGUF references from debugging_reference.md, post_refactor_guide.md, current.md, experiments/ltx2/CLAUDE.md. Backlog updated: 4 obsolete items removed/condensed, 3 new items from v0.9.25 reference verification (i2v E2E test, LinearQuadraticScheduler, skip_step optimization).

## 0.9.24

### added
- **Modality guidance (4th forward pass):** `modality_scale` parameter controls cross-modal attention guidance strength. When > 1.0, an additional transformer pass runs with cross-modal attention skipped (`SKIP_A2V_CROSS_ATTN` + `SKIP_V2A_CROSS_ATTN`), isolating each modality. Critical for lipsync and AV coherence. Default 3.0 (matching official reference).
- **Audio STG perturbation:** STG pass now creates both `SKIP_VIDEO_SELF_ATTN` and `SKIP_AUDIO_SELF_ATTN` perturbations (was video-only). Matches official pipeline behavior.
- **Audio embed dimension safety:** Guard in `generate_video_two_stage()` prevents 4096-dim video embeddings from being used as 2048-dim audio context when encoder fails to produce audio embeddings. Falls back to video-only with error log.
- **Config/schema/router:** `modality_scale` wired through `LTX2Config`, `config.toml`, `config.toml.example`, `LTX2GenerateRequest`, `resolve_param()` in router, and pipeline schema (UI slider, conditional on `enable_audio`).
- **Tests:** 5 new unit tests: modality guidance 4th pass activation/skip, cross-attn perturbation types, STG dual perturbation types, single-pass no-guidance.

### fixed
- **CFG audio negative embeddings (v0.9.23 followup):** Removed unsafe `pos_embeds` fallback in stage 1 and stage 2 AV denoising calls. Stage 2 AV call no longer passes `pos_audio_embeds if ... else pos_embeds` -- the dimension guard above ensures `pos_audio_embeds` is always valid when audio is enabled.
- **Modality guidance `blocks=None`:** Changed `list(range(48))` to `blocks=None` for cross-modal perturbations, matching the official reference. `None` means "all blocks" per `Perturbation.is_perturbed()` and is architecture-independent.
- **Eager `cond_video`/`cond_audio` cleanup:** Moved deletion immediately after Pass 2 model call (was deferred to STG/else branches). Frees memory before STG/modality passes allocate their own modality objects.
- **CLI audio params:** Added `--audio-guidance-scale`, `--audio-negative-prompt`, `--modality-scale`, `--rescale-scale`, and `--stg-blocks` to `scripts/gen.py` ltx2 subcommand.

## 0.9.23

### added
- **`_normalize_lora_args()` helper**: Extracted from duplicate inline blocks in `generate_video_with_offloading()` and `generate_video_two_stage()`. Normalizes flexible LoRA path/scale args (str|Path|list, float|list|None) into canonical parallel lists. Also fixes missing length validation in the two-stage path.
- **`_load_transformer_and_lora()` helper**: Extracted ~90 duplicated lines of 3-branch transformer loading (GGUF/cache/disk) + LoRA application into a single reusable helper with `video_only` parameter.
- **`_apply_distilled_lora_fp8()` helper**: State-dict-level LoRA fusion for live models with native fp8 weights. Extracts state dict, fuses deltas, reloads with `assign=True`, re-patches forwards.
- **Tests**: 9 `_normalize_lora_args` tests, 7 `_load_transformer_and_lora` tests, 1 fp8 distilled LoRA test, 1 FP8 guard test.

### fixed
- **Stage 2 distilled LoRA crash on fp8-cast models**: `load_lora()` called directly on live fp8-cast model would crash or silently upcast to bf16. Now detects native fp8 weights and routes through state-dict-level fusion.
- **FP8 preservation guard**: Upgraded from `logger.error()` (warn-only) to `raise RuntimeError` with actionable message suggesting `quantize='fp8-dynamic'` as alternative.
- **4 pre-existing test failures**: `test_resolution_validators.py` snap_to_32->snap_to_64 (2), `test_pipeline.py` OSError (1), `test_config_consistency.py` dataclass format (1).
- **Missing LoRA length validation in two-stage**: The two-stage path was missing the `len(paths) != len(scales)` check that the offloading path had. Now both use `_normalize_lora_args()`.

## 0.9.22

### added
- **State-dict LoRA fusion**: `fuse_lora_to_state_dict()` in `lora.py` -- fuses LoRA deltas into a state dict before `load_state_dict`, supporting fp8 + bf16 weights. Matches official LTX-2 `fuse_loras.py` pattern. fp8 weights are upcast to bf16, delta added, then downcast back.
- **FP8 preservation guard**: `_reconstruct_transformer_from_cache()` now validates fp8 weights survive device transfer, logging error if all parameters silently promoted to bf16.
- **Tests**: 9 new `fuse_lora_to_state_dict` tests (bf16, fp8, multi-LoRA, LoKR, immutability, scale), 2 fp8-cast LoRA integration tests.

### fixed
- **LoRA fusion crash on fp8-cast models**: `fuse_lora_to_base_model()` used `type(base_weight) is not torch.Tensor` to detect quantization, but native `float8_e4m3fn` tensors ARE `torch.Tensor`, causing fp8+bf16 addition crash. Now routes fp8-cast models through state-dict fusion path automatically.
- **Stale stg_blocks default**: `TwoStageConfig.__post_init__` defaulted to `[29]`, now `[28]` matching v0.9.20 config.
- **Stale single-stage fallback steps**: `web/routers/ltx2.py` fallback was `40` steps, now `30` matching v0.9.20 config.

### removed
- **Dead code**: `load_ltx2_transformer_from_fp8()` (~140 lines) removed from `loader.py` and `__init__.py`. Replaced by `load_ltx2_transformer_fp8_cast()` in v0.9.21.

## 0.9.21

### changed
- **FP8-cast loading**: Replaced eager dequantization (`load_ltx2_transformer_from_fp8`) with official Lightricks fp8-cast approach (`load_ltx2_transformer_fp8_cast`). Keeps FP8 weights as-is, patches nn.Linear forwards for per-forward upcast. Peak memory ~12GB (was ~42GB with dequant+requant cycle).
- **V2.3 defaults**: `num_inference_steps` 40->30, `stage1_num_inference_steps` 40->30, `stg_blocks` "29"->"28" to match official LTX-2.3 reference constants.
- **Cache/reconstruct fp8-cast**: `_reconstruct_transformer_from_cache()` detects fp8-cast cached state dicts and applies `amend_forward_with_upcast` instead of torchao quantization.

### added
- **`fp8_cast.py`**: New `src/llm_dit/quantization/fp8_cast.py` -- port of official Lightricks per-forward upcast pattern. `amend_forward_with_upcast()` patches all nn.Linear layers, skipping norms/embeddings.
- **V2.3 VAE loader**: Rewrote `src/llm_dit/models/ltx2/vae/loader.py` for V2.3 native key format. Hardcoded V2.3 decoder_blocks config (reverse-engineered from checkpoint shapes). Supports `compress_time` and `compress_space` blocks with multiplier.
- **V2.3 VAE architecture**: `VideoDecoder.__init__` and `_make_decoder_block` now handle `compress_time`/`compress_space` block types with `multiplier` and `out_channels_reduction_factor` parameters.
- **Tests**: 7 fp8-cast tests, 9 V2.3 VAE architecture + loading tests.

### fixed
- **V2.3 VAE channel computation**: `VideoDecoder.__init__` reverse-walk now handles all compress block types (was only `compress_all` and `res_x_y`), fixing conv_in channel mismatch (256 vs 1024).
- **Audio key filtering**: `is_audio_key()` now catches `av_ca_*` prefixed keys in addition to `av_cross_attn`.

## 0.9.20

### changed
- **LTX-2.3 only**: Dropped all V1 (LTX-2, 19B) support. V2.3 (22B) is the only supported LTX model.
- **Transformer loading**: `load_ltx2_transformer()`, `load_ltx2_transformer_from_fp8()`, `load_ltx2_transformer_gguf()` now always create V2.3 models (gated attention + cross-attention AdaLN).
- **Encoder**: `Gemma3Encoder` no longer has `_model_version` or V1 `_feature_extractor`. Always V2.3 with `_feature_extractor_v2`.
- **Config**: Removed `model_version` from `LTX2Config`. Added `connectors_file` (default: `ltx-2.3-connectors.safetensors`).
- **Config defaults**: `transformer_file` default updated to `ltx-2.3-transformer-fp8.safetensors`, `model_path` to `models/LTX-2.3`.
- **Embeddings connector**: Renamed `transformer_blocks` to `transformer_1d_blocks`, `norm_q`/`norm_k` to `q_norm`/`k_norm` to match V2.3 checkpoint key naming.

### added
- **Split script**: `scripts/split_ltx23_safetensors.py` splits the official 28GB bundled fp8 checkpoint into 5 component files (transformer, connectors, video-vae, audio-vae, vocoder).
- **Gated attention in connectors**: `Embeddings1DConnector` and `BasicTransformerBlock1D` now support `apply_gated_attention` for V2.3 8-block connectors.
- **V2.3 connector loading**: All encoder variants (bf16, fp8, fp8-safetensors, 8bit, q4-qat) load V2.3 connectors from `connectors_file`.

### fixed
- **fp8-safetensors crash**: `'Gemma3Encoder' object has no attribute '_feature_extractor_v2'` -- the `__new__()` bypass in fp8-safetensors variant never initialized V2 attributes. Now calls `_load_connector_weights()` after construction.
- **Reconstruct cache V2.3**: `_reconstruct_transformer_from_cache()` now always creates V2.3 models, matching the cached state dict format.

### polish (v0.9.20)
- **Config**: Updated config.toml and config.toml.example section headers, comments, and lora paths to reference LTX-2.3.
- **Tests**: Added 20 tests for `split_ltx23_safetensors.py`, 5 V2.3 config/architecture tests, removed 3 stale V1 tests.
- **Debug logging**: `--debug` now suppresses noisy third-party loggers (transformers, httpx, torch, etc.) at WARNING level.
- **Stale docstrings**: Fixed "19B" references in `models/__init__.py` and `av_block.py`.
- **Frontend**: Regenerated OpenAPI types; `model_version` confirmed absent from schema.

### removed
- V1 (19B) code paths in encoder, loader, and model manager.
- `model_version` config field and parameter threading.
- V1 `FeatureExtractorLinear` usage in encoder (kept for research `encode_multilayer()`).
- V1 feature extraction and normalization functions from `Gemma3Encoder.encode()`.
- Stale V1 model creation tests (`test_create_v1_model_default`, `test_v1_prepare_prompt_timestep_is_none`).
- `model_version = "2.3"` from GGUF smoke test TOML.

## 0.9.19

### added
- **GGUF pipeline integration**: Full pipeline wiring for GGUF-quantized LTX-2.3 transformers. Persistent model pattern (no cache/reconstruct), per-forward LoRA application via `GGMLLinear.lora_delta`, ModelManager GGUF preloading, and router integration.
- **GGUF-aware LoRA**: `load_lora_for_gguf()`, `attach_lora_deltas()`, `detach_lora_deltas()` in `utils/lora.py`. Pre-computes `lora_B @ lora_A` deltas and applies during dequant -- no base weight mutation.
- **GGUF key audit**: `scripts/audit_gguf_keys.py` verifies GGUF key mapping against model architecture. 1457/1457 video keys match (0 missing, 66 audio-only extras).
- **Tests**: 16 GGUF pipeline integration tests, 3 V2 VideoOnly state_dict verification tests (47 total new tests).

### fixed
- **V2 prompt_timestep crash**: `TransformerArgsPreprocessor.prepare()` now computes `prompt_timestep` for V2 models via `prompt_adaln_single`. Previously left as None, causing crash in `BasicTransformerBlock._apply_cross_attention_adaln()`.
- **model_version not passed to encoder**: `_load_ltx2()` now passes `model_version` to both `create_gemma3_encoder()` and `Gemma3Encoder()`. Previously always auto-detected, defeating explicit config control.
- **model_version not passed to GGUF loader**: `_preload_ltx2_gguf_transformer()` now accepts `model_version` and converts to `LTXModelType` override for the GGUF loader.
- **LoRA silent failure**: `attach_lora_deltas()` now logs warning when 0 of N delta keys match GGMLLinear layers.
- **Status endpoint GGUF path**: `ltx2_status()` now checks `gguf_transformer_path` in addition to `model_path`. GGUF-only configurations no longer report `available=false`.
- **Config session endpoint 500**: `get_session_config()` crashed when `current_profile` was None (attribute exists but is None, so `getattr` default wasn't used). Fixed with `or "default"` fallback.
- **V2 VideoOnly**: `BasicTransformerBlock` now supports V2 features (gated attention, cross-attention AdaLN, 9-param scale_shift_table, prompt_scale_shift_table). Previously V2 flags only worked in `BasicAVTransformerBlock`.
- **V2 caption_projection**: V2 models use `nn.Identity()` instead of `PixArtAlphaTextProjection` (projection moved to FeatureExtractorV2 encoder). Eliminates 4 missing GGUF keys.
- **GGUF loader**: Fixed `gguf_sd_loader()` tuple unpacking bug in `load_ltx2_transformer_gguf()`.
- **FFN slice**: `BasicTransformerBlock` FFN AdaLN slice changed from `slice(3, None)` to `slice(3, 6)` for correct indexing with 9-param V2 tables.

## 0.9.18

### added
- **GGUF**: Full GGUF quantization infrastructure (`quantization/gguf_dequant.py`, `gguf_tensor.py`, `gguf_loader.py`, `gguf_linear.py`). Supports Q2_K through Q8_0, IQ4_NL, IQ4_XS dequantization. GGMLLinear dequantizes per-forward, keeping quantized weights resident in VRAM.
- **LTX-2.3 (V2)**: Gated attention (`apply_gated_attention`) -- per-head sigmoid gate on attention output: `2.0 * sigmoid(gate_logits)`. Added to `attention.py`.
- **LTX-2.3 (V2)**: Cross-attention AdaLN (`cross_attention_adaln`) -- scale_shift_table grows from 6 to 9 params per block, with separate `prompt_scale_shift_table` for KV modulation. Added to `av_block.py`.
- **LTX-2.3 (V2)**: `prompt_timestep` and `self_attention_mask` fields on `TransformerArgs` for V2 conditioning.
- **LTX-2.3 (V2)**: `prompt_adaln_single` and `audio_prompt_adaln_single` modules in `LTX2Transformer` for cross-attention KV modulation.
- **LTX-2.3 (V2)**: `FeatureExtractorV2` (`encoders/gemma3_feature_extractor_v2.py`) with per-token RMSNorm and dual projections (video 4096, audio 2048).
- **LTX-2.3 (V2)**: Auto-detection of V1 vs V2 from checkpoint keys (`detect_v2_from_state_dict`).
- **LTX-2.3 (V2)**: `load_ltx2_transformer_gguf()` in `loader.py` for loading GGUF-quantized transformers with GGMLLinear layers.
- **Config**: `gguf_transformer_path` and `model_version` fields on `LTX2Config`.
- **Config**: `model_version` param on `Gemma3Encoder` for V2 encoder dispatch.
- **Protocol**: `audio_embeddings` field on `EncodingOutput` for V2 dual-stream audio embeddings.
- **Tests**: 28 new unit tests in `test_v2_architecture.py` covering gated attention, cross-attention AdaLN, FeatureExtractorV2, V2 detection, GGMLTensor, GGMLLinear, and V2 model creation.
- **Dependency**: `gguf>=0.18.0` (pure Python)

## 0.9.17

### added
- **CLI**: `scripts/gen.py` -- CLI-over-API tool that talks to the running server via HTTP. Subcommands: `flux2`, `zimage`, `ltx2`, `qwen`, `status`. Supports streaming (SSE), JSON output, and PNG download. 43 unit tests.

### deprecated
- **CLI**: `scripts/generate.py` emits `DeprecationWarning` at startup, directing users to `scripts/gen.py`. Will be removed in v1.0. Still needed for CLI-only features (embedding precompute, encoder-only mode).

### changed
- **Docs**: Updated `entry_points.md` and `feature_parity_matrix.md` to reflect three entry points (gen.py, Web API, generate.py) and deprecation status.

## 0.9.16

### changed
- **Core**: Replace 15 inline `gc.collect()` + `torch.cuda.empty_cache()` sequences in `model_manager.py` with centralized `cleanup_memory()` from `utils/memory.py`
- **Core**: Centralize `QUANT_ALIASES` constant in `quantization/__init__.py` -- single source of truth replacing 3 duplicated dicts (LTX2Config, Flux2Config, pipelines/generate.py)
- **Core**: Consolidate duplicate `DyPEConfig` -- canonical definition in `utils/dype.py`, re-exported from `config.py`
- **FLUX.2**: Extract scheduler functions (`get_schedule`, `compute_empirical_mu`, `generalized_time_snr_shift`) from `flux2_generate.py` to `schedulers/flux2_scheduler.py`
- **FLUX.2**: Replace local `cleanup_memory()` in `flux2_generate.py` with import from `utils/memory.py`
- **FLUX.2**: Extract `_resolve_flux2_params()` helper in `web/routers/flux2.py` -- DRYs duplicated param resolution between sync and streaming endpoints

### removed
- **LTX-2**: Dead `get_total_steps()` method, `estimate_vram_usage()` method + 9 ClassVar VRAM constants from `LTX2Config`
- **LTX-2**: Legacy distillation fields (`use_distilled`, `distilled_steps_stage1`, `distilled_steps_stage2`) from `LTX2Config`
- **LTX-2**: Deprecated encoder fields (`encoder_quantization`, `encoder_cpu_offload`) from `LTX2Config`
- **Config**: Dead `PipelineConfig` dataclass (not wired into RuntimeConfig)
- **Config**: Dead `flux2_seed`, `flux2_output_path`, `flux2_input_images` backward-compat properties from RuntimeConfig
- **Config**: `EnhancementConfig` preset classmethods (`quality_preset`, `speed_preset`, `all_preset`)

### fixed
- **Core**: `_unload_qwen_image()` and `_unload_qwen_image_t2i()` missing `torch.cuda.is_available()` guard before `torch.cuda.empty_cache()` (fixed via cleanup_memory adoption)

## 0.9.15

### changed
- **Core**: Extract `PinnedShuttleMixin` from duplicated pinned-memory shuttle code across AutoEncoder (VAE), Qwen3UnifiedEncoder, and Gemma3Encoder into `src/llm_dit/utils/shuttle.py`

### fixed
- **Gemma3**: `offload_to_pinned()` now includes `gc.collect()` + `torch.cuda.empty_cache()` (was missing, unlike VAE and Qwen3)
- **FLUX.2**: Fix OOM on 24GB GPUs with persistent model loading -- temporarily offload transformer to CPU during text encoding, reload after encoder offloads
- **FLUX.2**: Wire `[flux2].quantization` to `get_pipeline_quant_config()` via `quant_transformer` property (was dead -- `getattr` returned None, falling through to global default)

## 0.9.14

### added
- **FLUX.2**: Auto-load on generate -- pipeline loads on first request like LTX-2 and Z-Image, removing the need for manual pre-loading
- **Frontend**: "Unload All Models" button in SettingsMenu (ConfirmDialog-guarded, calls `/api/models/unload-all`)
- **Frontend**: Model status auto-refreshes after every generation (catches auto-load state changes)

### changed
- **Frontend**: ModelManager converted from interactive load/unload panel to read-only status display -- models auto-load on generate, manual management unnecessary
- **Z-Image**: SLG start/stop sliders always visible (removed broken `{"gt": 0}` conditional that frontend couldn't evaluate)

### removed
- **Z-Image**: Dead `compile` checkbox from schema (never wired to router or pipeline)
- **LTX-2**: Dead `compile` checkbox from schema (never wired to router or pipeline)
- **Frontend**: `loadPipelineModel` and `unloadPipelineModel` actions from appStore (replaced by auto-load + `unloadAllModels`)

### infrastructure
- **Playwright E2E**: Initial setup with chromium, Vite dev server integration, 3 test suites covering schema rendering, model status panel, and settings menu

## 0.9.13

### changed
- **Schemas**: `ParamSchema.config_mapped` field replaces global `EXCLUDED_PARAMS` set in `test_dry_config.py` -- each param self-declares per-pipeline whether it maps to a config dataclass, eliminating a ~30-entry hardcoded set that was imprecise (global across all pipelines) and required manual updates
- **LTX-2**: `LTX2_DEFAULT_NEGATIVE_PROMPT` constant extracted to `constants.py` -- replaces 3 inline copies in config.py, schemas.py, and generate.py
- **LTX-2**: `stg_enabled` checkbox now functionally gates STG -- when unchecked, forces `stg_scale=0.0` in router (previously hid sliders but STG stayed active at config default)

### removed
- **LTX-2**: Dead enhancement schema params (`latent_norm_enabled`, `nag_enabled`, `nag_scale`, `feta_enabled`, `feta_scale`, `teacache_enabled`, `teacache_threshold`) -- rendered UI controls with no router/pipeline wiring

## 0.9.12

### added
- **Frontend**: "Reset Storage" button in Settings menu (ConfirmDialog-guarded IndexedDB wipe + reload)
- **Frontend**: ConfirmDialog on "Clear all" history (destructive action guard)
- **Frontend**: MediaItem type and media utility functions (`utils/media.ts`) -- unified media display vocabulary with `detectKind`, `mediaItemFromResult`, `mediaItemFromHistory`
- **Frontend**: Extracted VideoViewer component from MediaViewer inline function
- **Frontend**: Shared VRAMBar component (`components/common/VRAMBar.tsx`) -- consolidates 2 identical VRAM progress bar implementations

### changed
- **Frontend**: Migrated from npm to bun (package manager only, runtime unchanged)
- **Frontend**: MediaViewer accepts `MediaItem` instead of separate url/mediaType/audioUrl props
- **Frontend**: ResultDisplay and HistoryCard use `mediaItemFromResult`/`mediaItemFromHistory` factories
- **Frontend**: Base64 data URLs stripped from persisted history params and form values (IndexedDB bloat fix)
- **Frontend**: BottomSheet close button touch target increased from p-1 to p-2 (44px minimum)
- **Frontend**: Moved `formatRelativeTime` from HistoryCard to shared `utils/format.ts`

### fixed
- **Frontend**: Audio post-seek resume -- audio now resumes playback after seeking while video is playing
- **Frontend**: ErrorBoundary around ResultDisplay prevents form loss on render crash

### removed
- **Frontend**: Dead utility functions removed from source files (`estimateDataUrlSize`, `validateForm`, `isParamVisible`)
- **Frontend**: ~470 lines of dead code (components/models/ directory, PipelineSelector, TabBar, unused API functions, unused types, dead store state)
- **Frontend**: Unused type fields: `VRAMStatus.freeMb`/`breakdown`, `ModelStatusResponse.totalVramMb`/`displayName`/`loraSummary`, `GenerationContext.loraSummary`/`fmttCached`/`historyCount`/`sessionModifiedFields`, `PipelinesResponse.defaults`, `GenerationError`, `GenerationProgress`, `OutputType.'layers'`

- **LTX-2**: Audio VAE decode pipeline (Phase 1). Pure PyTorch port of AudioDecoder (latents to stereo mel) and HiFiGAN Vocoder (mel to 24kHz waveform). Includes AudioPatchifier for 1D temporal patchify/unpatchify, PerChannelStatistics for latent denormalization, CausalConv2d blocks, and weight loaders for both models. 31 unit tests covering shapes, round-trips, weight loading, and full pipeline validation. New package: `src/llm_dit/models/ltx2/audio_vae/`.
- **LTX-2**: Transformer audio support (Phase 2). New `BasicAVTransformerBlock` handles video-only, audio-only, or dual-stream audio-video processing with bidirectional cross-modal attention (A2V, V2A). Extended `LTX2Transformer` with audio initialization, per-modality FBCache tracking, and dual-stream forward pass. Weight loading supports audio key mappings. Includes STG perturbation model (`PerturbationType`, `PerturbationConfig`, `BatchedPerturbationConfig`) for per-sample attention skipping. 41 unit tests. New file: `src/llm_dit/models/ltx2/av_block.py`.
- **LTX-2**: Pipeline integration for audio generation (Phase 3). Full end-to-end audio support: config fields (`audio_vae_path`, `vocoder_path`), ModelManager caching with pinned-memory shuttle for audio decoder and vocoder, `generate_video_two_stage()` extended with dual-stream denoising (`_compute_av_velocity`, `_denoise_av_stage`), cross-modal positional embeddings in transformer, audio utility functions (`compute_audio_latent_frames`, `create_audio_position_indices`, `create_audio_modality`), web layer wiring (parameter resolution, tuple return handling, WAV file saving). 29 new pipeline tests.

- **Frontend**: Audio playback support for LTX-2 video generation. Synced `<audio>` element with `<video>` in both ResultDisplay and fullscreen MediaViewer (onPlay/onPause/onSeeked sync). Audio indicator badge in metadata footer and history cards. Separate audio download button. SSE event parsing extracts `has_audio` and `audio_url` from completion events. OpenAPI types regenerated with `audio_negative_prompt` field.
- **Docs**: Composability analysis document (`internal/docs/architecture/composability_analysis.md`). Evaluates 6 patterns from the LTX-2 audio work against VISION.md's L1-L6 hierarchy. Includes summary matrix, cross-pipeline duplication map (sigma shift formulas, pinned memory shuttle, cleanup_memory), 3 prioritized actionable extractions (PerturbationConfig, ModalityPreprocessor protocol, sigma dedup), and roadmap alignment with DiTProtocol/UniversalFlowMatchLoop prerequisites.

### changed
- **LTX-2**: Removed ~106 lines of dead debug prints from `BasicTransformerBlock.forward()` and `Attention.forward()`. These were guarded by never-set debug attributes and included expensive operations (attention weight recomputation for entropy).
- **LTX-2**: Parameterized `LTX2Transformer._process_output()` to accept scale_shift_table, norm_out, proj_out as parameters (enables audio reuse).
- **LTX-2**: Collapsed redundant FBCache init branches in transformer forward pass.
- **LTX-2**: Fixed eager debug log evaluation in transformer -- `logger.debug()` with f-string `.float().mean()` now wrapped in `logger.isEnabledFor(logging.DEBUG)` guard.
- **LTX-2**: FBCache (Forward-Backward Cache) for transformer block skipping during denoising. Tracks L1 norm of residual changes between steps; blocks below threshold are skipped. First/last steps always compute fully. Config: `fbcache_threshold` (0.0 = disabled, 0.05 = recommended). Expected 10-30% speedup on denoising loop.
- **LTX-2**: Distilled sigma schedule mode. When `use_distilled_sigmas` is enabled, Stage 1 uses predefined sigma values from official LTX-2 constants instead of dynamic scheduler computation. Forces guidance_scale=1.0 (no CFG, no STG) -- guidance is baked into the distilled model.
- **LTX-2**: Meta-device skip init utility (`src/llm_dit/utils/meta_init.py`). Context manager that allocates model parameters on meta device (0 bytes) during construction. Combined with `load_state_dict(assign=True)`, eliminates 2x peak memory spike when reconstructing transformer from cached state dict.
- Optimization research document (`internal/docs/research/coderef_optimization_analysis.md`) -- cross-repo survey of 16 techniques across 5 codebases with RTX 4090-specific assessment

### fixed
- FA3 documentation (`docs/guides/compile_and_resolution.md`) incorrectly claimed Flash Attention 3 is "Ada Lovelace native". FA3 requires Hopper (SM90+). FA2 is the fastest available on RTX 4090.

## 0.9.11

### added
- **LTX-2**: Encoder persistence between generations. Gemma3 encoder now loads once at server startup (via ModelManager), caches on CPU with pinned memory, and shuttles to GPU per-request (~2s) instead of reloading from disk (~117s). Uses the same pinned memory DMA pattern as FLUX.2's Qwen3 encoder.
- **LTX-2**: Native fp8 layerwise casting for Gemma3 encoder (`src/llm_dit/quantization/layerwise_fp8.py`). Pure PyTorch forward hooks store `nn.Linear` weights as `float8_e4m3fn` (~50% memory savings) and cast to `bfloat16` on-the-fly during forward pass. No torchao or diffusers dependency. Norms and embeddings stay in bf16 for numerical stability.
- **LTX-2**: New `"fp8"` Gemma3 variant in `gemma3_variants.py`. Loads encoder in bf16, applies fp8 layerwise casting hooks, moves to target device. Replaces `"8bit"` (torchao int8) as the default variant. Same ~12GB VRAM footprint, but no torchao dependency and compatible with `torch.inference_mode()`.
- **LTX-2**: Transformer state_dict caching with pinned memory. ModelManager pre-loads transformer weights from disk at startup and caches as pinned bf16 tensors. Each generation reconstructs from cache (~2s `load_state_dict` + quantize) instead of reading from NVMe (~10-15s). Handles both regular and FP8 checkpoint formats.
- **LTX-2**: VAE decoder caching with pinned memory shuttle. ModelManager pre-loads VAE at startup (~2GB). Shuttles to GPU for decode, returns to CPU after. Eliminates two separate disk loads in two-stage pipeline (Stage 1.5 for `per_channel_statistics`, Stage 3 for full decode).
- **LTX-2**: `offload_to_pinned()` method on `Gemma3Encoder`. Pins all parameter and buffer memory for fast CPU-to-GPU DMA transfers. Models the same pattern as `Qwen3UnifiedEncoder.offload_to_pinned()`.
- **LTX-2**: Pre-converted fp8 safetensors script (`scripts/convert_gemma3_fp8.py`). One-time conversion of Gemma3 encoder weights from bf16 to `float8_e4m3fn` as a single safetensors file. Eliminates bf16 load + fp8 conversion overhead at runtime.

### changed
- **LTX-2**: Default `gemma_variant` changed from `"8bit"` (torchao int8) to `"fp8"` (native layerwise casting) in config.py and config.toml. Removes torchao dependency for encoder quantization.
- **LTX-2**: STG (Spatio-Temporal Guidance) now enabled by default in pipeline schema. The `stg_enabled` checkbox defaults to `True` (was `False`), aligning frontend with backend defaults.
- **LTX-2**: Added `distilled_lora_scale` slider to pipeline schema (advanced group). Controls blend strength for Stage 2 refinement LoRA. Sliding to 0 effectively disables the LoRA.

## 0.9.10

### added
- **LTX-2**: Local prompt enhancement via Gemma3. When `enhance_prompt` is enabled, the Gemma3 encoder uses its `.generate()` capability to expand terse user prompts into detailed video descriptions with motion, lighting, and audio cues before encoding -- no external API needed. Uses the official LTX-2 T2V system prompt and `apply_chat_template` for proper Gemma3 chat formatting. Available as config toggle, API field, and UI checkbox.
- **Quantization**: `fp8` alias now maps to `fp8-dynamic` (FP8 weights + FP8 activations) instead of `fp8-weight-only`. Utilizes RTX 4090's FP8 tensor cores for ~1.2-1.5x compute throughput. `fp8-weight-only` remains available as explicit override.
- **Quantization**: `granularity` parameter threaded from config through router to `quantize_component()`. Default changed from `per-tensor` to `per-row` for better numerical accuracy with fp8-dynamic.
- **Quantization**: `quantize_component()` now detects already-quantized weights (Float8Tensor, AffineQuantizedTensor, native FP8 dtypes) and skips redundant re-quantization.
- **LTX-2**: `transformer_file` config now wired to the loading path. When set to an FP8 safetensors file (e.g., `ltx-2-19b-dev-fp8.safetensors`), loads and dequantizes FP8 weights with scale factors. Falls back to `transformer/` directory if file not found.
- **LTX-2**: New `load_ltx2_transformer_from_fp8()` function for loading pre-quantized FP8 checkpoints with proper scale-factor dequantization. Modeled after FLUX.2's FP8 loading pattern.

### fixed
- **LTX-2**: Existing `load_ltx2_transformer()` now safely handles FP8 tensors -- preserves FP8 dtype instead of silently casting to BF16 without scale dequantization.
- **Quantization**: `get_recommended_method()` now returns `fp8-dynamic` (was `fp8-weight-only`) for FP8-capable hardware.

### removed
- **LTX-2**: Dead `load_ltx2_transformer_fp8_native()` function that imported from non-existent `fp8_native.py` module. Replaced by `load_ltx2_transformer_from_fp8()`.
- **Frontend**: `MediaViewer` component -- thin dispatcher that routes images to `ImageViewer` (zoom/pan/keyboard) and videos to a fullscreen `<video>` modal with native controls, Escape-to-close, and backdrop dismiss
- **Frontend**: Video expand button in `ResultDisplay` -- overlaid fullscreen icon (top-left) since native `<video>` click is play/pause
- **Frontend**: Play triangle overlay on video history thumbnails in `HistoryCard`

### fixed
- **Frontend**: LTX-2 video thumbnails broken in history. SSE parser now extracts `thumbnail_url` from backend "complete" event and maps to `GenerationResult.thumbnailUrl`. History cards show server-provided first-frame PNG instead of broken `<img src="video.mp4">`
- **Frontend**: Legacy IndexedDB history items with `.mp4`/`.webm` in `thumbnailUrl` migrated to empty string on hydration (shows placeholder icon instead of broken image)

## 0.9.9

### added
- Unified parameter resolution: `resolve_param()` helper in `web/param_resolver.py`. Establishes `client-sent > config.toml > schema default` precedence across all 4 pipelines using Pydantic v2's `model_fields_set`. Foundation for composable workflow orchestration (L1 vision).
- `csv_to_int_list()` helper for parsing comma-separated config strings (e.g., `stg_blocks = "29,30"`)
- 18 unit tests for `resolve_param()` covering: falsy value preservation (0, 0.0, ""), `skip_none` behavior, list fields, config-None graceful handling
- DRY tests for defaults endpoint: `TestDefaultsEndpointNameMappings` validates `_PARAM_NAME_MAPS` and `_PIPELINE_CONFIG_KEYS` against real schema/config fields (4 tests)
- ComfyUI workflow analysis docs: `internal/research/comfyui_workflows/` with LTX-2 SharksSampling (175-node) and FLUX.2 Klein Inpaint (72-node) analyses
- Architectural decision #14: three-layer parameter model (v0.9.9) with full rationale

### fixed
- **LTX-2**: Two-stage pipeline OOM during Stage 2 distilled LoRA fusion. The rank-384 LoRA's dequant/merge/requant cycle fragmented CUDA memory within ~20 layers. Fixed by eagerly deleting transient tensors before re-quantization and calling `empty_cache()` every layer instead of every 100.
- **LTX-2**: config.toml generation defaults (`guidance_scale`, `stg_scale`, `stg_blocks`, `rescale_scale`, `ge_gamma`, `negative_prompt`, `num_frames`, `width`, `height`, `fps`, `distilled_lora_path`, `distilled_lora_scale`, `stage1_steps`, `stage2_steps`) now respected for API requests when client omits them. Previously, schema defaults always won.
- **LTX-2**: `stg_scale=0` (disable STG) via API was silently ignored due to `or`-operator fallback treating 0 as falsy
- **Z-Image**: variant defaults (`steps`, `guidance_scale`, `shift`) now use `model_fields_set` instead of equality comparison against hardcoded Pydantic defaults. Fixes bug where client explicitly sending `steps=9` for the base variant would get it silently overwritten.
- **FLUX.2**: `num_steps` and `guidance` resolution now consults `config.toml` (`flux2.default_steps`, `flux2.default_guidance`) as intermediate layer between client and model-specific defaults. Fixed-param validation for distilled models now uses `model_fields_set` instead of `is not None`.
- **Qwen-Image**: `steps` and `cfg_scale` on all 3 endpoints (edit-layer, edit-multi, T2I generate) now respect `config.toml` values (`qwen_image.num_inference_steps`, `qwen_image.cfg_scale`). Previously, schema default (40 steps, 4.0 CFG) always won.

### fixed (defaults endpoint)
- **Defaults endpoint**: `GET /api/pipelines/{id}/defaults` now correctly resolves config.toml values for all pipelines. Fixed two bugs: (1) schema/config name mismatches (e.g., `stage1_steps` vs `stage1_num_inference_steps`, `num_steps` vs `default_steps`) via per-pipeline `_PARAM_NAME_MAPS`; (2) Qwen-Image pipeline IDs (`qwenimage-t2i`, `qwenimage-edit`) now correctly resolve to the `qwen_image` sub-config via `_PIPELINE_CONFIG_KEYS`.

### removed
- `_ZIMAGE_PYDANTIC_DEFAULTS` dict in `web/routers/core.py` (replaced by `model_fields_set` detection)
- `param_to_config_map` dict in `web/routers/config_mgmt.py` (replaced by per-pipeline `_PARAM_NAME_MAPS`)

## 0.9.8

### added
- Spatio-Temporal Guidance (STG): 3rd forward pass in denoising loop where self-attention is skipped at specified transformer blocks. The delta between conditioned and perturbed predictions drives spatial coherence and temporal consistency guidance. Reference formula: `v = v_cond + (cfg-1)*(v_cond-v_uncond) + stg*(v_cond-v_perturbed)`. Default: `stg_scale=1.0`, `stg_blocks=[29]` (matches reference).
- `skip_self_attn` parameter on `BasicTransformerBlock.forward()` for STG perturbed pass
- `stg_blocks` parameter on `LTX2Transformer.forward()` -- set of block indices where self-attention is skipped
- `stg_scale` and `stg_blocks` fields on `StepContext` and `constant_schedule()`
- `stg_blocks` field on `LTX2GenerateRequest` schema (API parameter)
- STG unit tests: transformer perturbation tests (4 tests) and pipeline config tests (3 tests)
- Reference doc sections: STG (4.8), Gradient Estimation (4.9), MultiModalGuider architecture (4.10), implementation gaps (4.11)

### fixed
- Reference doc: global `scale_shift_table` shape was `[6, 4096]` (per-block), corrected to `[2, 4096]` (output projection shift+scale)
- Reference doc: Stage 2 distilled schedule was "Fixed 8-step", corrected to "Fixed 3-step" (4 sigma values = 3 steps)

### changed
- `stg_scale` default changed from `0.0` to `1.0` across all layers (config.py, generate.py TwoStageConfig, schemas.py, config.toml, config.toml.example) to match reference `DEFAULT_VIDEO_GUIDER_PARAMS.stg_scale`
- STG wired through `constant_schedule()` call in Stage 1 of two-stage pipeline (was dead code)
- `stg_blocks` parsed from config string to list in ltx2 router (config uses comma-separated string, pipeline uses `list[int]`)

## 0.9.7

### fixed
- Two-stage noise formula: fixed flow-matching interpolation in Stage 2 noise addition. Was `x_0 + sigma * eps` (additive), now `(1-sigma) * x_0 + sigma * eps` (correct flow-matching interpolation). The wrong formula put the clean signal 11x too strong at sigma=0.909, causing green garbage output.
- LoRA fusion OOM: re-quantize each layer to fp8 immediately after fusion instead of batching at end. Prevents VRAM ballooning from ~13GB (fp8) toward ~26GB (bf16) when fusing large LoRAs (e.g., rank-384 distilled LoRA). Also updated from deprecated `float8_weight_only()` function API to `Float8WeightOnlyConfig`.
- CUDA fragmentation OOM during two-stage LoRA fusion: added `cleanup_memory()` between Stage 1 and Stage 1.5 to release denoising intermediate buffers (5-8 GB reserved). Added periodic `empty_cache()` every 100 layers during 487-layer fusion loop to prevent CUDA pool fragmentation.
- torchao availability check: updated `is_torchao_available()` to import current API (`quantize_`) instead of removed `int8_weight_only` function. Fixes "torchao not available" false negative with torchao 0.16+.
- Negative prompt: replaced 3-word placeholder with full reference DEFAULT_NEGATIVE_PROMPT (~1,300 chars) tuned for suppressing common diffusion failure modes
- Gemma config: cleared mismatched `encoder_model_id` in config.toml.example (Q4 QAT path doesn't apply to 8bit variant)
- Default `guidance_scale` 3.5 -> 3.0 to match reference `DEFAULT_VIDEO_GUIDER_PARAMS.cfg_scale` (was 17% over reference)
- Default `distilled_lora_scale` 0.8 -> 1.0 to match reference (Stage 2 was under-weighted)
- Default `height x width` 768x512 -> 512x768 to match reference landscape orientation (`DEFAULT_1_STAGE_HEIGHT=512, DEFAULT_1_STAGE_WIDTH=768`)
- Smoke test dimensions 384x256 -> 256x384 (landscape, consistent with reference orientation)

### changed
- Test constants: renamed legacy distilled field names to two-stage (`use_two_stage`, `stage1_num_inference_steps`, `stage2_num_inference_steps`) in SMOKE/STANDARD/FULL dicts and TOML overlays
- Test infrastructure: deleted duplicate `tests/e2e/conftest.py` (556-line copy of integration/pipeline/conftest.py)
- Test infrastructure: moved orphan test files to proper locations (`test_web_server.py` -> integration, `test_rewriter_parsing.py` -> unit, `test_qwen3_think_tokens.py` -> scripts)
- Z-Image API tests: rewrote from external `requests.get()` to in-process TestClient pattern; removed obsolete tests against dead endpoints (`/api/generation-config`, `/api/status`)

### added
- VRAM diagnostic logging at stage transitions in two-stage pipeline (post-encoder, post-stage1, post-stage1.5, pre-stage2). Logs allocated/reserved/free GPU memory for debugging OOM failures from logs alone.
- Qwen-Image E2E smoke test: `tests/e2e/api/test_qwenimage_smoke.py` with T2I generation, status/config endpoints, and validation
- Qwen-Image test config overlay: `tests/configs/qwenimage_smoke.toml`
- Z-Image API test: `tests/e2e/api/test_zimage_api.py` (config defaults, variant checks, generation, seed reproducibility)
- Config factory: expanded `qwen_image` model path extraction to include `cpu_offload`, `quantize_text_encoder`, `quantize_transformer`

## 0.9.6

### fixed
- Two-stage pipeline: set `distilled_lora_path` in config.toml (was empty, causing neon blob artifacts from base model doing 3-step distilled denoising without the distilled LoRA)
- Two-stage pipeline: wired `encoder_model_id` config through router to pipeline (was dead config, never passed as `text_encoder_path`)

### added
- Distilled LoRA guard in pipeline: raises clear `ValueError` when `distilled_lora_path` is empty instead of silently producing garbage
- Distilled LoRA guard in router: validates file exists before starting expensive generation

### changed
- Two-stage pipeline: eliminated second transformer load by reusing Stage 1 model for Stage 2 (~15-25s savings per generation). Distilled LoRA is fused into the existing model instead of reloading from disk.

## 0.9.5

### fixed
- LTX-2 video rendering in frontend: backend SSE response now uses standard `urls` array format (was `video_url` string)
- Config defaults endpoint: extracts nested pipeline sub-dicts from `RuntimeConfig.to_dict()` (LTX-2 config.toml values now reach frontend form defaults)
- Static file serving: added `/outputs` mount for LTX-2 video files (was returning 404)

### added
- Two-stage generation controls in LTX-2 pipeline schema: `use_two_stage`, `stage1_steps`, `stage2_steps`, `rescale_scale`, `ge_gamma`

### changed
- LTX-2 pipeline schema: replaced non-functional `num_inference_steps` param (no matching Pydantic field) with `stage1_steps` and `stage2_steps`

## 0.9.4

### fixed
- LTX-2 text encoder now runs on CUDA (was stuck on CPU due to config not reaching generation functions)

### removed
- `LTX2OptimizationConfig` eliminated -- dual-state config gap replaced with explicit parameters matching FLUX.2 pattern
- Dead config.toml fields: `encoder_quantization`, `encoder_cpu_offload` (never used)

### changed
- `generate_video_with_offloading()` and `generate_video_two_stage()` now take explicit `text_encoder_device`, `transformer_device`, `vae_device`, `quantize` (str), `skip_cleanup` parameters instead of `optimization: LTX2OptimizationConfig`
- Web router (`ltx2.py`) now passes `config.ltx2.*` optimization settings to generation functions
- Added `[LTX2]` and `[LTX2:TwoStage]` entry logging with device placement and quantization

## 0.9.3

### added
- Frontend: `React.memo` on `ParamControl` with custom comparator (prevents 20+ controls re-rendering on single value change)
- Frontend: slider debounce (50ms) with pointer-up commit pattern for smooth dragging
- Frontend: per-param memoized `onChange` callbacks in `PipelineForm` (stable refs for memo comparator)
- Frontend: `ErrorBoundary` wrapping `PipelineView` with retry button
- Frontend: `getResolvedValues()` module-level reference-equality cache (avoids recomputing on every selector call)
- Frontend: shared `PIPELINE_COLOR_MAP` constant (deduplicates appStore + HistoryCard)
- Frontend: lazy-loading on history thumbnails (`loading="lazy"`)
- Frontend: `React.memo` on `HistoryCard` with `item.id` comparator (prevents 500 cards re-rendering)
- Frontend: smoke render test (`App.smoke.test.tsx`) with route-based fetch mock and jsdom setup
- Backend: `GZipMiddleware` (Starlette built-in, minimum_size=1000, SSE auto-excluded)
- Backend: `Cache-Control` headers on `/api/pipelines` (5min), `/api/presets/preset/{name}` (5min), `/api/context` (5s)

### changed
- Frontend: unified duplicate `validateParam` (merged formStore + utils/validation.ts into single source of truth)
- Frontend: `PipelineForm.handleChange` no longer depends on `pipeline` reactive selector (uses `getState()` instead)
- Frontend: production sourcemaps disabled (saves ~744KB from dist)
- Upgraded `@hey-api/openapi-ts` 0.92.3 -> 0.92.4
- Upgraded `immer` 10 -> 11 (enableMapSet still required and functional)
- Upgraded `jsdom` 27 -> 28 (devDependency only)
- Upgraded `vite` 6 -> 7 (baseline-widely-available default target)
- Upgraded `@vitejs/plugin-react` 4 -> 5
- Upgraded `react` 18 -> 19, `react-dom` 18 -> 19, `@types/react` 18 -> 19, `@types/react-dom` 18 -> 19
- Migrated Tailwind CSS 3 -> 4: `@tailwind` directives -> `@import "tailwindcss"`, config.js -> CSS `@theme`, `@layer components` -> `@utility`, PostCSS -> `@tailwindcss/vite` plugin
- Tailwind 4 class name updates: `rounded` -> `rounded-sm`, `outline-none` -> `outline-hidden`, `flex-shrink-0` -> `shrink-0`, `min-w-[3.5rem]` -> `min-w-14`
- Frontend: replaced dynamic `import('./formStore')` in appStore with static import (eliminates Vite build warning)
- Frontend: removed Tailwind 4 border-color compat shim (all usages already have explicit colors)

### fixed
- Frontend: `StatusBar.tsx` crash when `ctx.pendingRestartFields` is undefined (added optional chaining)
- FLUX.2 prompt upsampling now reads `api_model` from `config.toml [rewriter].api_model` instead of using hardcoded default (both sync and streaming endpoints)

## 0.9.2

### added
- `fixed_params` field in FLUX.2 model registry (`constants.py`) -- distilled models declare which params are baked into weights
- `get_fixed_params()` and `is_distilled()` helper functions for FLUX.2 model introspection
- Fixed params validation in FLUX.2 generation endpoints -- overrides invalid params to model defaults with user-facing warnings
- `GET /api/flux2/models/{model_name}` endpoint returning model metadata (distilled, fixed_params, defaults, fp8)
- `warnings` field on `ImageGenerationResult` response model -- propagated through both POST and SSE endpoints
- `denoise_cfg()` function for FLUX.2 base models implementing true classifier-free guidance (doubled batch, uncond+cond forward passes, CFG formula)
- Unconditional text embedding preparation for base model CFG (encodes empty string, concatenates with prompt embeddings)
- `Flux2PromptUpsampler` class using BFL's official T2I and I2I system prompts for prompt enrichment via heylookitsanllm API
- `upsample_prompt` request field and pipeline schema checkbox for optional prompt upsampling before generation
- Frontend: distilled model controls (steps, guidance) disabled with "Fixed for distilled models" label when non-base model selected
- Frontend: generation warnings displayed in amber banner above result metadata

### changed
- FLUX.2 default resolution from 1024x1024 to 1360x768 (matches BFL's official default)
- FLUX.2 dimension preset list reordered with 1360x768 as first option
- FLUX.2 base model denoising now uses `denoise_cfg()` (explicit CFG) instead of `denoise()` (guidance embedding)

## 0.9.1

### added
- LTX-2 two-stage video generation pipeline (`generate_video_two_stage()`) matching reference `TI2VidTwoStagesPipeline` architecture
- `TwoStageConfig` dataclass for two-stage pipeline parameters (guidance, STG, rescaling, gradient estimation, distilled LoRA)
- `_denoise_stage()` shared denoising kernel with CFG, CFG rescaling, and gradient estimation support
- `load_spatial_upsampler()` loader for spatial upsampler model from safetensors checkpoints
- Distilled sigma schedule constants: `DISTILLED_SIGMA_VALUES`, `STAGE_2_DISTILLED_SIGMA_VALUES`
- Two-stage config fields in `LTX2Config`: `use_two_stage`, `stage1_num_inference_steps`, `stage2_num_inference_steps`, `spatial_upsampler_file`, `distilled_lora_path`, `distilled_lora_scale`, `stg_scale`, `stg_blocks`, `rescale_scale`, `ge_gamma`, `negative_prompt`
- Two-stage request fields in `LTX2GenerateRequest`: `use_two_stage`, `stage1_steps`, `stage2_steps`, `stg_scale`, `rescale_scale`, `distilled_lora_path`, `distilled_lora_scale`, `ge_gamma`
- Enhanced `vram_load_ltx2()` file validation: checks transformer, encoder, VAE, spatial upsampler, and distilled LoRA existence
- 17 new unit tests for two-stage config, distilled sigma schedules, half-resolution latents, and position indices

### changed
- LTX-2 default encoder quantization from `"none"` to `"fp8-weight-only"` (RTX 4090 has native FP8 tensor cores; INT4 is emulated)
- LTX-2 default transformer from distilled (`ltx-2-19b-distilled-fp8.safetensors`) to dev (`ltx-2-19b-dev-fp8.safetensors`)
- LTX-2 resolution snapping from 32-divisible to 64-divisible (two-stage requires half-res dimensions divisible by 32)
- `config.toml` `[ltx2]` section: full 1:1 alignment with `LTX2Config` dataclass fields
- `get_ltx2_model_path()` and `ltx2_status()`: removed TOML re-parsing, now use injected RuntimeConfig directly
- `ModelManager._load_ltx2()`: removed TOML re-parsing, uses injected config, validates all required files

### fixed
- `Config.load()` bug in ltx2 router (method is `Config.from_toml()`, but now bypassed entirely via injected config)

## 0.9.0

### added
- IndexedDB storage adapter (`idbStorage.ts`) for zustand persist middleware -- replaces localStorage with ~50MB+ async storage
- One-time `migrateFromLocalStorage()` function for seamless upgrade of existing history data
- OpenAPI TypeScript codegen pipeline: `bun run export-openapi && bun run gen-api` generates frontend types from FastAPI OpenAPI spec
- 3 new Pydantic response models: `ParamSchemaResponse`, `PipelineSchemaResponse`, `PresetDetailResponse`
- `_ensure_qwen_image_loaded()` and `_ensure_qwen_image_t2i_loaded()` helpers for on-demand pipeline loading via ModelManager
- `_get_zimage_encoder()` and `_ensure_zimage_loaded()` helpers in `core.py` for ModelManager access
- `_LOADED_PIPELINE_NAMES` mapping in `config_mgmt.py` for canonical ModelManager ID -> frontend API name translation
- `internal/state/backlog.md` -- prioritized improvement backlog

### changed
- **Dual state unification complete:** ModelManager is now the sole source of truth for all pipeline state across all 7 routers
- `core.py`: all 71 `srv.*` references migrated to ConfigDep/ManagerDep dependency injection
- `qwen_image.py`: 3 direct pipeline instantiation sites replaced with ModelManager `load()`/`get_pipeline()`
- `vram.py`: all unload functions use `manager.unload()` instead of server.py shims; all "is loaded?" checks use `manager.is_loaded()`
- `flux2.py`: all ~20 `srv.flux2_pipeline` references replaced with `manager.is_loaded("flux2")` / `manager.get_pipeline("flux2")`; `import web.server as srv` removed entirely
- `config_mgmt.py`: all ~10 pipeline reads replaced with `manager.is_loaded()` loop; `import web.server as srv` removed entirely
- Frontend types: hybrid strategy -- generated types re-exported where fit, hand-written kept where generated are too loose
- `server.py`: reduced from ~491 to ~296 lines
- History storage backend: localStorage -> IndexedDB (async, ~50MB+ quota, no main thread blocking)
- `MAX_HISTORY_ITEMS` increased from 100 to 500 (IndexedDB quota supports this comfortably)

### removed
- 6 pipeline globals from `server.py`: `pipeline`, `encoder`, `qwen_image_pipeline`, `qwen_image_t2i_pipeline`, `ltx2_pipeline`, `flux2_pipeline` -- ModelManager owns all pipeline state now
- 6 dead functions from `server.py` (~180 lines): `unload_zimage_pipeline()`, `unload_qwen_image_pipeline()`, `unload_qwen_image_t2i_pipeline()`, `unload_ltx2_pipeline()`, `get_vram_status()`, `load_zimage_pipeline_on_demand()`
- `_sync_globals_after_load()` and `_sync_globals_after_unload()` shim functions from `vram.py` + all 9 call sites
- Inline sync writes to `srv.pipeline`, `srv.encoder`, `srv.qwen_image_pipeline`, `srv.qwen_image_t2i_pipeline` from `core.py` and `qwen_image.py` helper functions
- `gc` and `torch` imports from `server.py` (no longer needed after unload function removal)
- `import web.server as srv` from `flux2.py` and `config_mgmt.py` (no longer access any server globals)
- `quotaHandlingStorage` from `sessionStore.ts` (~67 lines of localStorage quota error handling) -- replaced by IndexedDB adapter

### fixed
- `/api/context` returning 500 when server started without `--profile` flag -- `getattr(config, "current_profile")` returned `None` which Pydantic rejected as non-string
- `RuntimeError: Cannot set version_counter for inference tensor` during FP8-quantized generation -- reverted `@torch.inference_mode()` to `@torch.no_grad()` on all 4 generation executor functions (torchao's Float8Tensor dispatch requires version counter support that inference_mode disables)

## 0.8.9

### added
- `torch.no_grad()` wrappers on all generation executor functions (Z-Image, FLUX.2, LTX-2, Qwen-Image) to prevent autograd graph accumulation during inference
- `finally` cleanup blocks (`gc.collect()` + `torch.cuda.empty_cache()`) on all generation endpoints (streaming and non-streaming) to recover VRAM after errors
- `torch._dynamo.reset()` in FLUX.2 unload path to release compiled CUDA kernel cache (~3-5GB)
- `gc.collect()` before `empty_cache()` in Qwen-Image unload paths (server.py + model_manager.py)
- `AbortController` signal support in `generateStream()` for SSE cancellation
- 4 new Pydantic response models: `DyPEStatusResponse`, `PipelinesResponse`, `PipelineDefaultsResponse`, `ResolutionConfigResponse`
- `response_model=` applied to 4 remaining untyped endpoints (`/api/dype/status`, `/api/pipelines`, `/api/pipelines/{id}/defaults`, `/api/resolution-config`)
- `create_app()` factory function in server.py for OpenAPI spec extraction
- `scripts/export_openapi.py` for headless OpenAPI JSON export
- `openapi-ts.config.ts` and bun scripts (`export-openapi`, `gen-api`) for TypeScript codegen scaffolding

### changed
- Frontend context polling switched from `setInterval` to `setTimeout` chaining (prevents request pile-up during slow responses)
- Backend history entries no longer store `image_b64` (frontend IndexedDB is the image source of truth, saves ~150-250MB heap at 50 entries)
- Qwen-Image history entries no longer store base64 image data

### fixed
- CUDA memory leak: generation without `torch.no_grad()` built autograd graphs holding intermediate tensors (estimated 2-8GB per generation)
- CUDA memory leak: failed generations never called `empty_cache()`, leaving dead tensors in VRAM
- CUDA memory leak: FLUX.2 unload skipped `_dynamo.reset()`, leaving compiled kernels (~3-5GB) in VRAM
- CUDA memory leak: Qwen-Image unload skipped `gc.collect()`, leaving Python-held CUDA tensors unreclaimable

## 0.8.8

### added
- Pydantic `CamelModel` base class with automatic camelCase JSON serialization (`alias_generator=to_camel`)
- ~27 typed response models in `web/schemas.py` covering all JSON endpoints
- `response_model=` applied to all JSON endpoints across 7 routers (OpenAPI schema now fully typed)
- Shared `get_lora_info()` utility in `web/utils.py` for LoRA extraction
- Shared `formatUptime()` utility and `RestartWarning` component in frontend
- `LoRAFile`, `LoRAListResponse`, `ClearCacheResponse`, `PresetsResponse` types in `types.ts`

### changed
- All API responses now serialize as camelCase (e.g., `uptimeSeconds` instead of `uptime_seconds`)
- Frontend `client.ts` simplified: eliminated all manual snake-to-camel mapping functions (~60 lines removed)
- `fetchGenerationContext()`, `fetchVRAMStatus()`, `fetchModelStatus()`, `fetchPresets()`, `clearCache()` now direct typed passthrough
- `VRAMStatus` interface updated: `usedMB` -> `usedMb` (matches Pydantic `to_camel` output)
- `ModelStatusResponse` interface expanded to match full backend schema
- `LoRAInfo.layers_updated` -> `layersUpdated` across frontend
- `ModelStatusResponse.loras` field in schemas.py changed from `List[Dict]` to `List[LoRAInfo]`
- Consolidated 3 `if compile_enabled:` blocks in `vram.py` into single block
- Duplicate `LoRAFile`/`LoRAListResponse` types removed from `client.ts` (now in `types.ts` only)

### removed
- 2 stub endpoints: `GET /api/configs/available`, `POST /api/configs/load`
- 10 legacy load/unload routes from `vram.py` (superseded by unified `/api/models/{id}/load|unload`)
- 4 overlapping status endpoints merged into `/api/context`: `GET /api/system/status`, `GET /api/server/status`, `GET /api/generation-config`, `GET /api/rewriter-models`
- `web/static/` (dead v1 frontend) and `web/archive-frontend/` directories deleted
- Duplicate `_get_lora_info()` from `system.py` (replaced by shared `web/utils.get_lora_info()`)

## 0.8.7

### added
- `GET /api/context` endpoint: composite status aggregating model variant, LoRA fusion state, VRAM, quantization, compile, and session state
- `LoRAInfo` and `GenerationContextResponse` Pydantic models in `web/schemas.py`
- StatusBar component: persistent compact strip showing loaded model, LoRA badges, quant badge, VRAM bar with expand/collapse
- SettingsMenu component: server restart (with confirmation dialog), clear CUDA cache, system info, pending restart warnings
- ConfirmDialog reusable component for destructive actions
- Gear icon in LeftNav header for settings access on desktop
- Generation context polling (15s interval) in App.tsx
- ModelManager cards enriched with model variant name, LoRA badges, and config tags

### changed
- LoRA slider UX: added stepper buttons (44px touch targets with long-press acceleration), preset pills (0.25/0.50/0.75/1.00), slider moved to desktop-only secondary control
- `get_model_status()` in vram.py now returns `model_variant`, `loras`, `lora_summary` fields
- Model load/unload actions now refresh generation context
- VRAM poll augmented with composite context poll (15s interval)

## 0.8.6

### fixed
- LoRA post-fusion OOM: re-quantizes affected layers to fp8 after LoRA merge, reclaiming ~8GB VRAM on persistent models
- LoRA spec format mismatch: filters out empty-path LoRA entries from frontend before comparison
- FLUX.2 block_offload default: schema default changed from `true` to `false` to match `config.toml`

### added
- `log_prompts` config option: toggle prompt text logging (default: true) via `[logging]` section
- `log_generation_params` now actually gated in generation routers (was defined but never checked)
- HTTPS support for frontend-v2 dev server via `VITE_BACKEND_URL`, `VITE_SSL_CERT`, `VITE_SSL_KEY` env vars
- `.env.example` in `web/frontend-v2/` documenting HTTPS env vars

### removed
- Qwen-Image-Layered pipeline: all decomposition code, endpoints, schemas, tests, and docs deleted (~15 files modified/removed)
- `/api/qwen-image/decompose` endpoint
- `/api/qwen-image/status` and `/api/qwen-image/config` endpoints
- `QwenImagePipeline` (legacy pure-Python layered pipeline)
- `QwenImageDecomposeRequest` schema
- `qwenimage-layered` model type from CLI, config, and model manager

### changed
- README.md: updated pipeline table, added HTTPS setup section, removed Qwen-Image-Layered docs link
- CLAUDE.md: updated to v0.8.6, added DRY Configuration Principles section, HTTPS nav link

## 0.8.5

### fixed
- LoRA re-fusion OOM on persistent models: second request no longer dequantizes fp8 (9GB) to bf16 (18GB) again, preventing 26GB OOM when encoder shuttles to GPU
- `_infer_model_device_dtype` returns bfloat16 (compute dtype) instead of uint8/float8 (storage dtype) for quantized models, so LoRA math happens in correct precision
- Z-Image `load_lora()` no longer passes raw storage dtype to LoRA loader (delegates to `_infer_model_device_dtype`)

### added
- HTTPS support via `ssl_certfile` / `ssl_keyfile` config fields and CLI args (uvicorn-native TLS)
- Optional `ssl_ca_certs` for mutual TLS client certificate verification
- `FusedLoRAState` / `LoRAFusionRecord` dataclasses for pipeline-agnostic LoRA fusion tracking
- `get_fused_state(model)` attaches tracking state to any `nn.Module` -- works regardless of how the pipeline stores the model
- LoRA fusion guard in `flux2_generate.py`: skips re-fusion when LoRAs already match, raises `RuntimeError` on mismatch
- `_ensure_correct_model()` now checks LoRA specs in addition to model name; auto-reloads on LoRA mismatch
- HTTP 409 response for LoRA mismatch errors in FLUX.2 endpoints

## 0.8.4

### fixed
- FLUX.2 model switching: frontend model dropdown now actually triggers model reload instead of silently using whatever was loaded at startup
- LoRA crash on fp8-quantized models: `Float8Tensor + Tensor` now dequantizes before merge instead of hitting unimplemented `aten.add`
- VRAM race between generate and unload: mid-request unload returns 503 "model was unloaded" instead of OOM cascade

### added
- `ModelManager.reload_flux2(model_name)` for model-switching with proper lock coordination
- Loaded vs requested model_name logging in FLUX.2 generate endpoints

## 0.8.3

### added
- Preset card browser: horizontal scroll strips with visual cards replace the old `<select>` dropdown
- Active preset indicator with three states: none / active (checkmark) / modified (warning + Restore)
- Preset modification detection: compares active preset's original params against current resolved values
- `clearPreset()` and `restorePreset()` actions in formStore

### changed
- `applyPreset()` signature expanded to `(pipelineId, presetName, params)` -- records active preset and clears `userModified` for preset-touched params (synergy fix with dependent_defaults)
- appStore `loadPresets()` updated to use new `applyPreset` signature
- PipelineForm now renders `<PresetBrowser>` instead of a `<select>` dropdown

### fixed
- Preset + dependent_defaults synergy bug: applying a preset then switching models could leave stale preset values because `userModified` incorrectly blocked dependent_defaults updates

## 0.8.2

### added
- `compile_dynamic` config field for FLUX.2: `torch.compile(dynamic=True)` eliminates ~90s recompilation when resolution changes
- `dependent_defaults` on ParamSchema: schema-driven system for auto-updating form values when a trigger param changes (e.g., switching FLUX.2 model updates steps/guidance)
- FLUX.2 presets: `presets/flux2/distilled_fast.md` and `presets/flux2/base_quality.md`
- NumberInput auto-snap on blur: misaligned values (e.g., 1000 for a step=16 field) auto-correct when focus leaves the input
- `tests/unit/test_resolution_validators.py`: 51 tests for Pydantic and dataclass resolution snapping
- `snapToStep()` shared utility in `web/frontend-v2/src/utils/numbers.ts`
- `userModified` tracking in formStore for smart dependent default application
- Dynamic shapes section in `docs/guides/compile_and_resolution.md`

### changed
- Slider `commitInputValue()` refactored to use shared `snapToStep()` utility
- `getResolvedValues()` now layers dependent defaults between schema and server defaults
- PipelineForm `handleChange` triggers `applyDependentDefaults()` when a trigger param changes
- torch.compile uses `fullgraph=False` when `compile_dynamic=true` (safety for data-dependent branches)

## 0.8.1

### added
- `docs/guides/compile_and_resolution.md` -- comprehensive torch.compile and resolution guide with ROI math, compatibility matrix, VRAM budgets, and RTX 4090 config recommendations
- Resolution validation: Pydantic `@field_validator` snaps width/height to nearest VAE multiple (16 for FLUX.2, 32 for LTX-2) with `Field()` min/max constraints
- Defense-in-depth: `Flux2GenerationConfig.__post_init__` snaps invalid resolutions with warning
- Compile-aware logging in FLUX.2 router (latent token count, warmup notice)
- `compile_enabled` and `compile_vae_enabled` fields in `/api/flux2/status` response
- Compile warmup warning in pipeline config metadata (shown in model manager UI)
- Frontend: `dimension_preset` dropdown now drives width/height values (one-way sync)
- Frontend: per-image dimension display for multi-reference uploads in ImageUpload
- Frontend: step alignment validation warns when values are not multiples of step size

### changed
- torch.compile calls now use `fullgraph=True` to catch graph breaks at compile time
- FLUX.2 default config: `compile = false`, `compile_vae = false` (better default for 4-step distilled model)
- FLUX.2 schema: width/height step changed from 64 to 16, min from 512 to 256
- Dimension presets list now includes "Custom" option

## 0.8.0

### added
- 7 domain routers in `web/routers/`: core (Z-Image), flux2, ltx2, qwen_image, vram, config_mgmt, system
- `web/schemas.py`: all Pydantic request/response models extracted from server.py
- `web/utils.py`: shared helpers (output dirs, image saving, config merging)
- `web/dependencies.py`: FastAPI `Depends()` for `ConfigDep` and `ManagerDep`

### removed
- 5 dead load functions from server.py: `load_pipeline()`, `load_encoder_only()`, `load_api_encoder()`, `load_hybrid_pipeline()`, `load_api_pipeline()` (replaced by ModelManager)
- 17 unused backward-compat `@property` shims from RuntimeConfig: all `wan_*`, `flux2_quantization`, `flux2_encoder_device`, `fmtt_scale`
- Broken integration test fixtures (`client_with_model`, `client_with_pipeline`) and their test classes (`TestEncodeEndpoint`, `TestGenerateEndpoint`)

### changed
- `web/server.py` decomposed from 5744 lines to 465 lines (globals, unload functions, startup logic)
- All 68 API endpoints moved from monolithic server.py into 7 domain-specific router files
- Router registration deferred to `_register_routers()` in `main()` to avoid circular imports with `web.server as srv`

## 0.7.1

### removed
- Entire VL (Vision Conditioning) module: `src/llm_dit/vl/`, VL endpoints in server.py, VLConfig, VL CLI args, VL frontend JS, VL documentation
- VL experiment files: `experiments/qwen3_vl/`, `experiments/test_vl_ablation.py`, `experiments/qwen3_vl_poc.py`
- VL schema entries from Z-Image ParamSchemas (vl_enabled, vl_image, vl_strength)
- VL constants (QWEN3_VL_4B_CONFIG, QWEN3_VL_GENERATION_DEFAULTS)
- `load_in_4bit`/`load_in_8bit` boolean flags from Gemma3Encoder (replaced by `quantization_variant: str`)
- BitsAndBytes `load_in_8bit` parameter from 14 script/test files

### changed
- `config.toml.example` rewritten: added `[quantization]` section, all methods use unified torchao names, fixed `compile_mode` default
- Gemma3Encoder variant metadata: two booleans replaced by `_quantization_variant: str` ("bf16", "int8", "q4_0")
- Config presets use unified names: "8bit"->"int8", "fp8"->"fp8-weight-only", "4bit"->"int4"
- CLAUDE.md: added "DRY Configuration Principles" section with `--hidden-layer` reference

### fixed
- 27 unit test failures resolved (wrong mock patch targets, argument mismatches, outdated expectations)
- 3 missed VL CLI args in cli.py (`--rewriter-no-vl`, `--rewriter-preload-vl`, `--rewriter-vl-api-model`)
- config.toml/config.py warning messages referencing deleted method names

## 0.7.0

### added
- Unified `quantize_component()` entry point for all model components across all pipelines
- `ComponentQuantConfig` and `PipelineQuantConfig` dataclasses for type-safe quantization config
- Global `[quantization]` TOML section with per-pipeline overrides (resolution: pipeline override > global default)
- `get_pipeline_quant_config()` on RuntimeConfig for resolving effective quantization per pipeline
- `get_quant_compile_warnings()` for detecting dangerous quant + compile combinations
- `VALID_METHODS` constant: `none`, `fp8-dynamic`, `fp8-weight-only`, `int8`, `int4`
- New unit tests for `quantize_component()`, `VALID_METHODS`, compile warnings, and stats dict shape

### removed
- `fp8_native.py` (manual FP8 casting with allowlist) -- replaced by `Float8WeightOnlyConfig`
- `fp8_inference.py` (DiffSynth-style `F.linear` patching context manager) -- replaced by `Float8DynamicActivationFloat8WeightConfig`
- `quantization/config.py` (`QuantizationMethod` enum and BitsAndBytes helpers) -- no longer needed
- `quantize_model_torchao()` and `quantize_model_torchao_filtered()` from torchao_utils
- `create_fp8_filter_fn()` and `analyze_fp8_compatibility()` from torchao_utils
- All BitsAndBytes quantization paths (4bit, 8bit NF4) across all pipelines
- DiffSynth FP8 context manager usage from Qwen-Image pipelines
- `_build_quantization_config()` from `qwen_image_2512.py`
- `QUANTIZATION_PRESETS` dict from QwenImageConfig
- `test_fp8_inference.py` test file

### changed
- All 4 pipelines (FLUX.2, LTX-2, Z-Image, Qwen-Image) now use `quantize_component()` as sole quantization entry point
- torchao is the sole quantization backend (BitsAndBytes dependency removed from quantization paths)
- `get_recommended_method()` returns unified method names (`"fp8-weight-only"` instead of `"fp8"`, `"int8"` instead of `"8bit"`)
- Encoder quantization uses post-load pattern (load BF16 then quantize) instead of BnB during-load
- Config field names unified: removed `flux2_quantization`, `ltx2_quantize`, `qwen_image_quantize_*` in favor of `<pipeline>_quant_<component>`
- Updated `docs/reference/quantization.md` with migration table from old to new API
- **Missed layers cleanup**: Fixed remaining ~40% of codebase still referencing old method names
  - Fixed runtime crash bugs: `generate.py` default param, CLI choices, Qwen variant defaults, QwenImageConfig validation
  - Removed BnB from: EncoderConfig, BackendConfig, backends/qwen_image.py, encoders/gemma3.py, backends/transformers.py
  - Deleted dead code: `utils/quantization.py` (quanto module), `is_bitsandbytes_available()`, BnB migration tests
  - Fixed config wiring: `ltx2_quantize` TOML default from `"fp8"` to `"fp8-weight-only"`
  - Removed `bitsandbytes` from `pyproject.toml` dependencies
  - Updated `vae_utils.py` to remove `"8bit"` BnB method, only `"int8"` remains for VAE

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
