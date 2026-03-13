# scripts inventory

*last updated: 2026-03-03*

Complete inventory of `scripts/` directory with current status.

## active scripts (~12)

These are in active use and should be maintained.

| Script | Purpose | Notes |
|--------|---------|-------|
| `gen.py` | CLI-over-API generation tool | Thin httpx client. Subcommands: flux2, zimage, ltx2, qwen, status. See [entry_points.md](entry_points.md). |
| `generate.py` | Legacy CLI generation entry point | **Deprecated (v0.9.17)** -- use `gen.py` instead. Removal planned for v1.0. Still needed for embedding precompute, encoder-only mode. |
| `export_openapi.py` | Export OpenAPI spec from FastAPI app | Used by frontend codegen: `bun run export-openapi` |
| `quantize_model.py` | Quantize model weights to fp8/int8/int4 | Offline quantization utility |
| `train.py` | Training script | Training entry point |
| `convert_to_safetensors.py` | Convert model checkpoints to safetensors format | One-time conversion utility, still useful for new models |
| `smoke_test.py` | Quick smoke test for pipeline functionality | Fast validation |
| `clear-cuda.py` | Clear CUDA memory / reset GPU state | Debugging utility |
| `start-server.sh` | Start the web server | Shell wrapper |
| `stop-server.sh` | Stop the web server | Shell wrapper |
| `install_sageattention.sh` | Install SageAttention from source | Setup utility |
| `run_tests.sh` | Run test suite | Shell wrapper |
| `refresh_cache.sh` | Refresh model/config caches | Maintenance utility |

## research scripts (~7)

Useful for benchmarking, profiling, and reference comparisons. Kept for future use.

| Script | Purpose |
|--------|---------|
| `profiler.py` | Active profiling tool for pipeline performance |
| `benchmark_optimizations.py` | Performance benchmarking across configurations |
| `analyze_test_runs.py` | Experiment result analysis |
| `inspect_image.py` | Image inspection and metadata extraction |
| `generate_reference_video.py` | LTX-2 reference video generation for comparisons |
| `generate_encoder_baselines.py` | Encoder output comparison across variants |

## deleted (2026-02-14)

These one-time debug, trace, and verification scripts were created during the LTX-2 port from diffusers to pure PyTorch. The port is complete and in production. Git history preserves them.

### debug scripts (7)

| Script | Original Purpose |
|--------|------------------|
| `debug_connector.py` | Debug connector weight loading |
| `debug_pre_connector.py` | Debug pre-connector tensor shapes |
| `debug_encoding_pipeline.py` | Debug text encoding pipeline flow |
| `debug_final_masking.py` | Debug attention masking in transformer |
| `debug_ltx2_embeddings.py` | Debug LTX-2 embedding extraction |
| `debug_ltx2_pipeline.py` | Debug full LTX-2 pipeline execution |
| `debug_text_vs_registers.py` | Debug text token vs register token behavior |

### trace scripts (4)

| Script | Original Purpose |
|--------|------------------|
| `trace_connector_forward.py` | Trace connector forward pass shapes |
| `trace_actual_forward.py` | Trace transformer forward pass |
| `trace_block0_detailed.py` | Detailed trace of transformer block 0 |
| `trace_per_dim_means.py` | Per-dimension mean analysis of embeddings |

### verification scripts (6)

| Script | Original Purpose |
|--------|------------------|
| `verify_connector_weights.py` | Verify connector weight loading correctness |
| `verify_encoder_output.py` | Verify encoder output matches reference |
| `verify_feature_extractor.py` | Verify feature extractor behavior |
| `verify_gemma_weights.py` | Verify Gemma3 weight loading |
| `verify_layer_alignment.py` | Verify layer alignment between our port and reference |
| `verify_tokenizer_fix.py` | Verify tokenizer fix for special tokens |
| `validate_ltx2_port.py` | Full validation of LTX-2 port against reference |

### one-off test scripts (14)

| Script | Original Purpose |
|--------|------------------|
| `test_vae_comparison.py` | Compare VAE output against reference |
| `test_vae_decode_only.py` | Test VAE decode path in isolation |
| `test_vae_direct_compare.py` | Direct VAE comparison (tensor-level) |
| `test_vae_layer_trace.py` | Trace VAE layer activations |
| `test_vae_reconstruction.py` | Test VAE encode-decode round-trip |
| `test_transformer_compare.py` | Compare transformer output against reference |
| `test_transformer_forward.py` | Test transformer forward pass |
| `test_weight_loading.py` | Test weight loading from safetensors |
| `test_cross_attn_weights.py` | Test cross-attention weight shapes |
| `test_key_mapping.py` | Test state dict key mapping |
| `test_embedding_pipeline.py` | Test embedding pipeline end-to-end |
| `test_embedding_distribution.py` | Test embedding output distribution statistics |
| `test_connector_output.py` | Test connector output correctness |
| `test_orchestration.py` | Test pipeline orchestration flow |
| `test_quality_reference_code.py` | Test quality against reference implementation |
| `test_reference_vae_only.py` | Test reference VAE in isolation |

### comparison/audit scripts (5)

| Script | Original Purpose |
|--------|------------------|
| `compare_connector_outputs.py` | Compare connector outputs between implementations |
| `compare_caption_projection.py` | Compare caption projection layer outputs |
| `check_fp8_compatibility.py` | Check fp8 quantization compatibility |
| `check_learnable_registers.py` | Check learnable register token behavior |
| `audit_connector_interface.py` | Audit connector interface compliance |

### superseded scripts (4)

| Script | Superseded By |
|--------|---------------|
| `generate_ltx2_baseline.py` | Imports deprecated `ltx2_diffusers.py`; use `generate.py` with `--model-type ltx2` |
| `generate_with_enhanced_prompts.py` | Prompt rewriter in `src/llm_dit/utils/prompt_rewriter.py` |
| `embeddings.py` | Encoder architecture in `src/llm_dit/encoders/` |
| `run_reference_pipeline.py` | Reference comparison no longer needed (port validated) |
