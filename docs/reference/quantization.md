# Quantization Reference

last updated: 2026-02-06

Reference for the unified quantization system in llm-dit-experiments. All pipelines (FLUX.2, LTX-2, Z-Image, Qwen-Image) use a single torchao-based `quantize_component()` function with consistent config.

## Quick Start

Configure quantization globally in `config.toml`:

```toml
[quantization]
# Global defaults for all pipelines
# Methods: none, fp8-dynamic, fp8-weight-only, int8, int4
encoder = "none"
transformer = "fp8-weight-only"
vae = "none"
granularity = "per-tensor"
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

#### `fp8-weight-only` (recommended default)

**Best for:** RTX 4090+ with `torch.compile` enabled

- Stores weights in FP8, computes in BF16
- Requires compute capability 8.9+ (Ada Lovelace or newer)
- Requires Linear layer dimensions divisible by 16 (auto-filtered)
- Fully compatible with `torch.compile` (no autotune graph breaks)
- Good balance of VRAM savings and quality

```toml
[quantization]
transformer = "fp8-weight-only"
```

#### `fp8-dynamic`

**Best for:** Maximum throughput when `torch.compile` is disabled

- FP8 weights AND FP8 activations at runtime
- Uses `scaled_mm` GEMM ops for native FP8 tensor core acceleration
- NOT compatible with `torch.compile` `reduce-overhead` mode (autotune causes graph breaks)
- ~2x inference speedup on Ada Lovelace / Hopper GPUs

```toml
[quantization]
transformer = "fp8-dynamic"
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

For a ~20B parameter DiT transformer:

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
    method="fp8-weight-only",        # any VALID_METHODS value
    component_type="transformer",    # "encoder", "transformer", or "vae"
    granularity="per-tensor",        # "per-tensor" or "per-row" (FP8 only)
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
# Returns "fp8-weight-only" on RTX 4090, "int8" on older GPUs
```

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
