# Quantization Reference

last updated: 2025-01-06

Reference for quantization options in llm-dit-experiments, covering VRAM reduction techniques for running Qwen-Image models on consumer GPUs like the RTX 4090.

## Quick Start

For RTX 4090 (24GB), use the `rtx4090_fp8` preset:

```python
from llm_dit.config import QwenImageConfig

config = QwenImageConfig.from_preset(
    "rtx4090_fp8",
    model_path="/path/to/Qwen-Image-Edit-2511",
)
# Result: ~13GB VRAM, 2x faster inference
```

Or configure via TOML:

```toml
[rtx4090.qwen_image]
model_path = "/path/to/Qwen-Image-Edit-2511"
quantize_text_encoder = "fp8"
quantize_transformer = "fp8"
quantize_vae = "int8"
offload_type = "none"
```

## Quantization Methods

### Comparison Table

| Method | Backend | VRAM Reduction | Speed | Quality | Hardware |
|--------|---------|----------------|-------|---------|----------|
| `fp8` | TorchAO | ~50% | 2x faster | 98%+ | RTX 4090+, H100 |
| `fp8-filtered` | TorchAO | ~50% | 2x faster | 98%+ | RTX 4090+ |
| `diffsynth-fp8` | DiffSynth | ~50% | 1.5x faster | 98%+ | RTX 4090+, AMD MI300 |
| `int8` | TorchAO | ~50% | ~1x | 99%+ | Any GPU |
| `8bit` | BitsAndBytes | ~50% | ~0.9x | 99%+ | Any GPU |
| `4bit` | BitsAndBytes | ~75% | ~0.8x | 95% | Any GPU |
| `none` | N/A | 0% | 1x | 100% | Requires >48GB |

### Method Details

#### `fp8` - TorchAO FP8 Dynamic

**Best for:** RTX 4090+ with maximum performance needs

- Uses `float8_e4m3fn` dtype with dynamic scaling
- Requires compute capability 8.9+ (Ada Lovelace or newer)
- Requires tensor dimensions divisible by 16
- Fully compatible with `torch.compile`
- ~2x inference speedup due to native FP8 tensor cores

```python
quantize_text_encoder = "fp8"
quantize_transformer = "fp8"
```

#### `fp8-filtered` - TorchAO FP8 with Layer Filtering

**Best for:** Models with non-16-aligned layers (e.g., Qwen2.5-VL vision encoder)

- Same as `fp8` but explicitly skips incompatible layers
- Logs which layers were skipped
- Use when `fp8` produces warnings about dimension alignment

```python
quantize_text_encoder = "fp8-filtered"  # Skips vision encoder's 3584-dim layers
```

#### `diffsynth-fp8` - DiffSynth-Style FP8

**Best for:** AMD ROCm support, flexibility

- Runtime `F.linear` patching via `torch._scaled_mm`
- No dimension alignment requirements
- AMD MI300 support via `float8_e4m3fnuz` dtype
- Slightly more overhead than TorchAO FP8

```python
quantize_transformer = "diffsynth-fp8"
# Usage: automatically wraps inference in fp8_inference() context
```

#### `int8` - TorchAO INT8 Weight-Only

**Best for:** Any GPU without FP8 support

- INT8 weights, FP16/BF16 activations
- Works on any CUDA device
- No quality loss for most use cases

```python
quantize_text_encoder = "int8"
```

#### `8bit` - BitsAndBytes INT8

**Best for:** Production stability

- Well-tested, widely used
- Requires model reload (can't apply post-load)
- Good compatibility with all GPUs

```python
quantize_text_encoder = "8bit"
```

#### `4bit` - BitsAndBytes NF4

**Best for:** Maximum VRAM savings

- Normal Float 4-bit quantization
- ~75% VRAM reduction
- Some quality loss (visible in fine details)
- Best combined with group offloading

```python
quantize_text_encoder = "4bit"
quantize_transformer = "4bit"
```

## VAE Quantization

VAE uses Conv2d layers which have different quantization requirements:

| Method | Supported | Notes |
|--------|-----------|-------|
| `int8` | Yes | TorchAO INT8 dynamic, works for Conv2d |
| `8bit` | Yes | BitsAndBytes, requires reload |
| `fp8` | No | FP8 Conv2d poorly supported |
| `4bit` | No | Quality degradation too severe |

```python
quantize_vae = "int8"  # Recommended: ~50% reduction, minimal quality loss
```

## Offloading Strategies

### Comparison

| Type | VRAM Usage | Speed | Use Case |
|------|------------|-------|----------|
| `none` | Full model | 100% | >48GB VRAM |
| `model` | ~20GB | ~70% | RTX 4090 with quantization |
| `group` | ~4-6GB | ~50% | Memory-constrained |
| `sequential` | ~2-3GB | ~20% | Minimum VRAM |

### `model` - Component-Level Offload

Moves entire components (text encoder, transformer, VAE) between CPU and GPU:

```python
offload_type = "model"
cpu_offload = True
```

### `group` - Block-Level Offload

Streams DiT transformer blocks in groups, keeping only a few on GPU:

```python
offload_type = "group"
num_blocks_per_group = 2  # Keep 2 blocks on GPU at a time
```

- Uses async data transfer with CUDA streams
- Best balance of VRAM savings and speed
- Requires diffusers with group offloading support

### `sequential` - Layer-Level Offload

Moves individual layers, minimum VRAM but slowest:

```python
offload_type = "sequential"
```

## Optimization Presets

Pre-configured settings for common scenarios:

### `balanced`

Good defaults for most systems:

```python
config = QwenImageConfig.from_preset("balanced")
# quantize_text_encoder = "8bit"
# quantize_transformer = "none"
# offload_type = "model"
# VRAM: ~20GB
```

### `rtx4090_fp8`

Maximum performance on RTX 4090:

```python
config = QwenImageConfig.from_preset("rtx4090_fp8")
# quantize_text_encoder = "fp8"
# quantize_transformer = "fp8"
# quantize_vae = "int8"
# offload_type = "none"
# VRAM: ~13GB, Speed: 2x
```

### `rtx4090_group`

RTX 4090 with group offloading for larger batches:

```python
config = QwenImageConfig.from_preset("rtx4090_group")
# quantize_text_encoder = "8bit"
# offload_type = "group"
# num_blocks_per_group = 2
# VRAM: ~16-18GB
```

### `max_vram_savings`

Minimum VRAM (~8-10GB):

```python
config = QwenImageConfig.from_preset("max_vram_savings")
# quantize_text_encoder = "4bit"
# quantize_transformer = "4bit"
# quantize_vae = "int8"
# offload_type = "group"
# num_blocks_per_group = 1
# VRAM: ~8-10GB, Quality: 95%
```

### `amd_mi300`

AMD ROCm support:

```python
config = QwenImageConfig.from_preset("amd_mi300")
# quantize_text_encoder = "8bit"
# quantize_transformer = "diffsynth-fp8"
# offload_type = "model"
```

## torch.compile Compatibility

**Important:** `torch.compile` is incompatible with CPU offloading.

| Offload Type | torch.compile | CUDA Graphs |
|--------------|---------------|-------------|
| `none` | Yes | Yes (reduce-overhead) |
| `model` | No | No |
| `group` | No | No |
| `sequential` | No | No |

When using offloading, set `compile_mode = "default"` (no CUDA graphs):

```toml
[rtx4090.optimization]
compile = true
compile_mode = "default"  # NOT "reduce-overhead" with offloading

[rtx4090.qwen_image]
offload_type = "model"
```

## Hardware Requirements

### FP8 Quantization

Requires compute capability 8.9+:
- NVIDIA RTX 4090, 4080, 4070 series (Ada Lovelace)
- NVIDIA H100, H200 (Hopper)
- AMD MI300 series (with DiffSynth FP8 only)

Check support:

```python
from llm_dit.quantization import check_fp8_support
print(check_fp8_support())  # True/False
```

### Memory Estimation

For Qwen-Image-Edit-2511 (20.43B DiT):

| Configuration | VRAM Required | Speed |
|---------------|---------------|-------|
| Full precision, no offload | >48GB | 100% |
| FP8 all, no offload | ~13GB | 200% |
| 8bit + model offload | ~20GB | 70% |
| 4bit + group offload | ~8-10GB | 50% |

## API Reference

### `quantize_model_torchao()`

Apply TorchAO quantization to a model:

```python
from llm_dit.quantization import quantize_model_torchao

model = load_model()
quantize_model_torchao(model, "fp8")  # or "int8"
```

### `quantize_model_torchao_filtered()`

Apply FP8 with automatic layer filtering:

```python
from llm_dit.quantization import quantize_model_torchao_filtered

model, stats = quantize_model_torchao_filtered(model, "fp8")
print(f"Quantized {stats['quantized_layers']}/{stats['total_linear_layers']} layers")
```

### `quantize_vae()`

Quantize VAE for Conv2d layers:

```python
from llm_dit.quantization import quantize_vae

vae = load_vae()
vae = quantize_vae(vae, "int8")  # Only int8/8bit for Conv2d
```

### `fp8_inference()` Context Manager

DiffSynth-style FP8 inference:

```python
from llm_dit.quantization import fp8_inference, enable_fp8_weights

# Pre-convert weights for memory savings
enable_fp8_weights(model)

# Runtime FP8 computation
with fp8_inference():
    output = model(input)
```

## Troubleshooting

### FP8 Dimension Errors

Error: `RuntimeError: Dimensions must be multiples of 16 for FP8`

Solution: Use `fp8-filtered` instead of `fp8`:

```python
quantize_text_encoder = "fp8-filtered"  # Skips non-aligned layers
```

### torch.compile + Offload Errors

Error: `RuntimeError: Expected all tensors to be on the same device`

Solution: Disable compilation or use `compile_mode = "default"`:

```python
offload_type = "model"
compile = False  # Or compile_mode = "default"
```

### Out of Memory with 4-bit

If OOM even with 4-bit, use sequential offloading:

```python
quantize_text_encoder = "4bit"
quantize_transformer = "4bit"
offload_type = "sequential"
```
