# Quantization Reference

last updated: 2026-02-17

Reference for the unified quantization system in llm-dit-experiments. All pipelines (FLUX.2, LTX-2, Z-Image, Qwen-Image) use a single torchao-based `quantize_component()` function with consistent config.

## Important implementation note:
Any operation that temporarily dequantizes quantized weights should do so in a streaming fashion (one layer at a time, re-quantize immediately) rather than accumulating dequantized state. This is especially critical on 24GB cards where the full bf16 model doesn't fit.

## Quick Start

Configure quantization globally in `config.toml`:

```toml
[quantization]
# Global defaults for all pipelines
# Methods: none, fp8-dynamic, fp8-weight-only, int8, int4
encoder = "none"
transformer = "fp8-dynamic"
vae = "none"
granularity = "per-row"
```

Override per-pipeline:

```toml
[flux2]
quantization_transformer = "fp8-dynamic"  # Override global default for FLUX.2

[ltx2]
quantization_encoder = "int8"  # Quantize LTX-2's Gemma3 encoder
```

## Quantization Methods

All methods use [torchao](https://github.com/pytorch/ao) as the sole quantization backend.

| Config string | torchao class | What it does | Compile safe? | VRAM reduction |
|---|---|---|---|---|
| `none` | N/A | BF16 (no quantization) | Yes | 0% |
| `fp8-dynamic` | `Float8DynamicActivationFloat8WeightConfig` | FP8 weights + FP8 activations | NO (autotune) | ~50% |
| `fp8-weight-only` | `Float8WeightOnlyConfig` | FP8 weights, BF16 activations | Yes | ~50% |
| `int8` | `Int8WeightOnlyConfig` | INT8 weights, original activations | Yes | ~50% |
| `int4` | `Int4WeightOnlyConfig` | INT4 weights (max compression) | Yes | ~75% |

### Method Details

#### `fp8-dynamic` (recommended default)

**Best for:** RTX 4090+ (SM89) -- maximum throughput with FP8 tensor cores

- FP8 weights AND FP8 activations at runtime
- Uses `scaled_mm` GEMM ops for native FP8 tensor core acceleration
- ~1.2-1.5x compute throughput over BF16 matmuls
- `per-row` granularity recommended (one scale per matrix row, better numerical accuracy)
- NOT compatible with `torch.compile` `reduce-overhead` mode (autotune causes graph breaks)
- `"fp8"` is an alias for `"fp8-dynamic"`

```toml
[quantization]
transformer = "fp8-dynamic"
granularity = "per-row"
```

#### `fp8-weight-only`

**Best for:** RTX 4090+ with `torch.compile` enabled, or as a fallback from fp8-dynamic

- Stores weights in FP8, computes in BF16 (tensor cores idle)
- Requires compute capability 8.9+ (Ada Lovelace or newer)
- Requires Linear layer dimensions divisible by 16 (auto-filtered)
- Fully compatible with `torch.compile` (no autotune graph breaks)
- Same VRAM savings as fp8-dynamic but without compute speedup

```toml
[quantization]
transformer = "fp8-weight-only"
```

#### `int8`

**Best for:** Any GPU without FP8 support, or when maximum quality is needed

- INT8 weights, original dtype activations
- Works on any CUDA device
- Minimal quality loss
- Compatible with `torch.compile`

```toml
[quantization]
encoder = "int8"
```

#### `int4`

**Best for:** Maximum VRAM savings

- INT4 weight-only quantization
- ~75% VRAM reduction
- Some quality loss in fine details
- Compatible with `torch.compile`

```toml
[quantization]
transformer = "int4"
```

## Component-Specific Behavior

`quantize_component()` applies different filter logic depending on `component_type`:

### Encoder filtering
- Skips: `embed_tokens`, `*norm*`, `lm_head`, `rotary_emb`
- These layers are sensitive to quantization or have incompatible shapes

### Transformer filtering
- Skips: `*norm*` layers
- For FP8 methods: additionally skips Linear layers with dimensions not divisible by 16

### VAE filtering
- Delegates to `quantize_vae()` which applies INT8 dynamic quantization to Conv2d layers only
- FP8 is not supported for Conv2d (poorly supported in hardware)

## VAE Quantization

VAE uses Conv2d layers which have different quantization requirements:

| Method | Supported | Notes |
|--------|-----------|-------|
| `int8` | Yes | TorchAO INT8 dynamic, works for Conv2d |
| `fp8-*` | No | FP8 Conv2d poorly supported |
| `int4` | No | Quality degradation too severe for decoder |

```toml
[quantization]
vae = "int8"  # Only int8 is supported for VAE
```

## Config Hierarchy

Quantization config resolves as: **per-pipeline override > global default**.

```
config.toml [quantization] section   <-- global defaults
    |
    v
config.toml [pipeline] section      <-- per-pipeline overrides (optional)
    |
    v
RuntimeConfig.get_pipeline_quant_config("pipeline_name")
    |
    v
PipelineQuantConfig(encoder=..., transformer=..., vae=...)
```

## torch.compile Compatibility

| Method | `torch.compile` | `reduce-overhead` mode |
|--------|-----------------|------------------------|
| `none` | Yes | Yes |
| `fp8-weight-only` | Yes | Yes |
| `fp8-dynamic` | Yes (default mode) | NO (autotune graph breaks) |
| `int8` | Yes | Yes |
| `int4` | Yes | Yes |

Use `get_quant_compile_warnings()` to check for dangerous combinations:

```python
from llm_dit.quantization import get_quant_compile_warnings

warnings = get_quant_compile_warnings("fp8-dynamic", "reduce-overhead")
# Returns: ["fp8-dynamic uses autotune which is incompatible with reduce-overhead..."]
```

## Hardware Requirements

### FP8 Quantization

Requires compute capability 8.9+:
- NVIDIA RTX 4090, 4080, 4070 series (Ada Lovelace)
- NVIDIA H100, H200 (Hopper)

Check support:

```python
from llm_dit.quantization import check_fp8_support
print(check_fp8_support())  # True/False
```

### Memory Estimation

For a ~22B parameter DiT transformer (e.g., LTX-2.3):

| Configuration | VRAM Required | Speed |
|---------------|---------------|-------|
| `none` (BF16) | ~40GB | 1x |
| `fp8-weight-only` | ~20GB | ~1.5x |
| `fp8-dynamic` | ~20GB | ~2x |
| `int8` | ~20GB | ~1x |
| `int4` | ~10GB | ~0.8x |

## API Reference

### `quantize_component()`

Unified quantization entry point for any model component:

```python
from llm_dit.quantization import quantize_component

model = load_model()
model, stats = quantize_component(
    model,
    method="fp8-dynamic",            # any VALID_METHODS value
    component_type="transformer",    # "encoder", "transformer", or "vae"
    granularity="per-row",           # "per-tensor" or "per-row" (FP8 only)
    verbose=True,
)

print(f"Quantized {stats['quantized_layers']}/{stats['total_layers']} layers")
print(f"Skipped {stats['skipped_layers']} layers")
```

### `quantize_vae()`

Quantize VAE for Conv2d layers (called internally by `quantize_component` for vae component_type):

```python
from llm_dit.quantization import quantize_vae

vae = load_vae()
vae = quantize_vae(vae, "int8")
```

### `get_quant_compile_warnings()`

Check for dangerous quantization + compile combinations:

```python
from llm_dit.quantization import get_quant_compile_warnings

warnings = get_quant_compile_warnings("fp8-dynamic", "reduce-overhead")
for w in warnings:
    print(f"WARNING: {w}")
```

### `VALID_METHODS`

Tuple of all valid quantization method strings:

```python
from llm_dit.quantization import VALID_METHODS
# ("none", "fp8-dynamic", "fp8-weight-only", "int8", "int4")
```

### `get_recommended_method()`

Auto-detect best quantization method for current hardware:

```python
from llm_dit.quantization import get_recommended_method

method = get_recommended_method()
# Returns "fp8-dynamic" on RTX 4090, "int8" on older GPUs
```

## Already-Quantized Detection

`quantize_component()` automatically detects if a model's weights are already quantized and skips redundant re-quantization. This handles:

- **torchao tensor subclasses:** `Float8Tensor`, `AffineQuantizedTensor`
- **Native FP8 dtypes:** `torch.float8_e4m3fn`, `torch.float8_e5m2`

When detected, returns `stats["method"] = "already_quantized"` with zero layers quantized. This is useful when loading pre-quantized FP8 checkpoints via `transformer_file` config -- the weights arrive already dequantized to BF16, but if they were kept in FP8, the guard prevents double-quantization.

## Troubleshooting

### FP8 Dimension Errors

Error: `RuntimeError: Dimensions must be multiples of 16 for FP8`

This should not happen with the unified system -- `quantize_component()` automatically skips incompatible layers. If it does occur, check that you are using `quantize_component()` and not calling torchao directly.

### torch.compile + fp8-dynamic

Error: Graph breaks or autotune warnings

Solution: Use `fp8-weight-only` instead (compile-safe):

```toml
[quantization]
transformer = "fp8-weight-only"  # Safe with torch.compile
```

### torch.compile + Offload Errors

Error: `RuntimeError: Expected all tensors to be on the same device`

Solution: Disable compilation or use `compile_mode = "default"`:

```toml
compile = true
compile_mode = "default"  # NOT "reduce-overhead" with offloading
```

## Alternatives to Runtime Quantization

Runtime torchao quantization (applying `quantize_component()` at load time) is the current approach for all pipelines. A 2026-02-17 research survey identified two alternatives worth evaluating:

### Official pre-quantized fp8 checkpoints

Both LTX-2 and FLUX.2 Klein have official fp8 safetensors published by the model authors:

| Model | Repository | File |
|-------|-----------|------|
| LTX-2.3 dev fp8 | `Lightricks/LTX-2.3` | Split fp8 safetensors in `models/LTX-2.3/` |
| LTX-2.3 GGUF | `unsloth/LTX-2.3-GGUF` | `ltx-2.3-22b-dev-Q6_K.gguf` |
| FLUX.2 Klein 9B fp8 | `black-forest-labs/FLUX.2-klein-9b-fp8` | HF repo |
| FLUX.2 Klein 4B fp8 | `black-forest-labs/FLUX.2-klein-4b-fp8` | HF repo |

These weights load as standard `float8_e4m3fn` tensors -- no torchao required. Quantization was done by the model authors with architecture knowledge, which is strictly better quality than generic runtime quantization.

**Status:** Pending evaluation (see `internal/state/backlog.md` -- "evaluate official fp8 checkpoints + layerwise casting").

### Diffusers `enable_layerwise_casting`

Stores weights as `float8_e4m3fn` (standard PyTorch dtype, not a torchao tensor subclass). Hooks cast each layer to bf16 before its forward pass. Provides the same VRAM savings as `fp8-weight-only` without any torchao dependency.

```python
transformer.enable_layerwise_casting(
    storage_dtype=torch.float8_e4m3fn,
    compute_dtype=torch.bfloat16
)
```

Eliminates the `torch.inference_mode()` incompatibility, the `isinstance()` vs `type()` gotcha, the torchao API churn risk, and CUDA fragmentation during the LoRA requantization loop.

**When to prefer over torchao:** For any new pipeline where fp8 tensor core compute speedup (W8A8) is not required.

**When to keep torchao:** The fp8-dynamic (W8A8) path (`Float8DynamicActivationFloat8WeightConfig`) provides genuine compute speedup from FP8 tensor cores on sm89. This cannot be replicated by layerwise casting.

Full analysis: `internal/docs/research/quantization_alternatives_survey.md`

## Migration from Old API

The following old APIs have been removed:

| Old API | Replacement |
|---------|-------------|
| `quantize_model_torchao(model, "fp8")` | `quantize_component(model, "fp8-dynamic", "transformer")` |
| `quantize_model_torchao_filtered(model, "fp8")` | `quantize_component(model, "fp8-weight-only", "transformer")` |
| `fp8_inference()` context manager | Removed -- `fp8-dynamic` handles this internally |
| `enable_fp8_weights(model)` | Removed -- `fp8-weight-only` handles this internally |
| `create_fp8_filter_fn()` | Removed -- filtering is automatic in `quantize_component()` |
| `analyze_fp8_compatibility(model)` | Removed -- use `is_fp8_compatible_layer()` if needed |
| BitsAndBytes `"4bit"`, `"8bit"` | Use `"int4"`, `"int8"` (torchao equivalents) |
| `"diffsynth-fp8"` | Use `"fp8-dynamic"` |
| `"fp8-filtered"` | Use `"fp8-weight-only"` (filtering is automatic) |
