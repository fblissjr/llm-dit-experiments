# Profiler Script

The profiler script (`scripts/profiler.py`) provides stability testing and performance benchmarking for the Z-Image pipeline.

## Quick Start

```bash
# Run all tests
uv run scripts/profiler.py --model-path /path/to/z-image-turbo

# Show system info only
uv run scripts/profiler.py --show-info

# Verbose output
uv run scripts/profiler.py --model-path /path/to/z-image-turbo -v
```

## Features

- **Stability testing**: Verify encoder and pipeline work correctly
- **Performance profiling**: Measure encoding and generation times
- **Memory tracking**: Monitor GPU memory usage and detect leaks
- **Configuration sweeping**: Test multiple optimization settings automatically
- **Device placement testing**: Compare CPU vs GPU encoder performance
- **System info reporting**: Collect environment details for debugging

## Command Line Options

### Basic Options

| Flag | Description |
|------|-------------|
| `--model-path` | Path to Z-Image model (required for tests) |
| `--config` | Path to TOML config file |
| `--profile` | Config profile to use |
| `--hidden-layer` | Which hidden layer to extract embeddings from (default: -2) |
| `-v`, `--verbose` | Enable verbose output |

### Test Selection

| Flag | Description |
|------|-------------|
| `--tests` | Comma-separated list of tests to run |
| `--sweep` | Run tests with different optimization configs |
| `--sweep-devices` | Run tests with different device placements |
| `--repeat N` | Repeat test suite N times |

### Output

| Flag | Description |
|------|-------------|
| `--output`, `-o` | Save results to JSON file |
| `--show-info` | Show system info and exit (no tests) |

## Available Tests

### Encoder Tests

| Test | Description |
|------|-------------|
| `load_encoder` | Load text encoder and verify initialization |
| `encode_short` | Encode a short prompt (~20 tokens) |
| `encode_medium` | Encode a medium prompt (~100 tokens) |
| `encode_with_template` | Encode with template applied |
| `encode_with_thinking` | Encode with thinking block content |
| `generate_text` | Test text generation (prompt rewriting) |
| `repeated_encode` | Encode same prompt 5x to check for memory leaks |

### Pipeline Tests

| Test | Description |
|------|-------------|
| `load_pipeline` | Load full pipeline (encoder + DiT + VAE) |
| `full_generation` | Generate a complete image |

### System Tests

| Test | Description |
|------|-------------|
| `cuda_sync` | Measure CUDA synchronization overhead |

## Usage Examples

### Run Specific Tests

```bash
# Only encoder tests
uv run scripts/profiler.py --model-path /path/to/model --tests encode_short,encode_medium

# Include pipeline tests
uv run scripts/profiler.py --model-path /path/to/model --tests pipeline,full_generation
```

### Configuration Sweep

Test multiple optimization configurations automatically:

```bash
uv run scripts/profiler.py --model-path /path/to/model --sweep
```

This tests combinations of:
- Baseline configuration
- Embedding cache enabled
- CPU vs CUDA encoder
- FP16 vs BF16 dtype
- torch.compile enabled
- Flash Attention 2 (if available)
- xFormers (if available)
- SDPA

### Device Placement Sweep

Test different device configurations:

```bash
uv run scripts/profiler.py --model-path /path/to/model --sweep-devices
```

Configurations tested:
- All components on CUDA
- Encoder on CPU, DiT/VAE on CUDA
- VAE on CPU, encoder/DiT on CUDA
- Encoder and VAE on CPU, DiT on CUDA

### Save Results

```bash
# Save to JSON for later analysis
uv run scripts/profiler.py --model-path /path/to/model --output results.json

# Multiple runs for statistical analysis
uv run scripts/profiler.py --model-path /path/to/model --repeat 3 --output results.json
```

### System Information

Display detailed system and library information:

```bash
uv run scripts/profiler.py --show-info
```

Output includes:
- Platform and Python version
- PyTorch version and CUDA availability
- GPU details (name, VRAM, compute capability)
- Available attention backends (FA2, FA3, xFormers, Sage, SDPA)
- ML library versions (transformers, diffusers, etc.)
- System memory

## Output Format

### Console Output

```
============================================================
PROFILING SESSION
============================================================
Model: /path/to/z-image-turbo
Encoder device: cpu
DiT device: cuda
VAE device: cuda
Dtype: bfloat16
============================================================

[TEST] cuda_sync
  [PASS] 0.1ms | Memory: 0 -> 0 MB (peak: 0 MB)

[TEST] load_encoder
  [PASS] 5234.2ms | Memory: 0 -> 0 MB (peak: 0 MB)

[TEST] encode_short
  [PASS] 45.3ms | Memory: 0 -> 0 MB (peak: 0 MB)

============================================================
SUMMARY
============================================================
Tests: 8/8 passed
Total time: 12345.6ms
Peak memory: 8192MB
```

### JSON Output

```json
[
  {
    "timestamp": "2024-01-15T10:30:00",
    "config": {
      "model_path": "/path/to/model",
      "encoder_device": "cpu",
      "dit_device": "cuda",
      "torch_dtype": "bfloat16"
    },
    "system_info": {
      "platform": "Linux-5.15.0-x86_64",
      "torch_version": "2.5.0",
      "gpu_name": "NVIDIA GeForce RTX 4090"
    },
    "tests": [
      {
        "name": "encode_short",
        "success": true,
        "duration_ms": 45.3,
        "memory_before_mb": 0,
        "memory_after_mb": 0,
        "memory_peak_mb": 0,
        "extra": {
          "token_count": 21,
          "tokens_per_sec": 463.5
        }
      }
    ],
    "summary": {
      "total_tests": 8,
      "passed": 8,
      "failed": 0,
      "total_time_ms": 12345.6,
      "peak_memory_mb": 8192
    }
  }
]
```

## Troubleshooting

### Common Errors

**"No encoder loaded"**
- The `load_encoder` test failed
- Check model path and ensure model files exist
- Check verbose output for the underlying error

**Memory issues**
- Use `--text-encoder-device cpu` to reduce GPU memory
- Enable `--cpu-offload` for lower memory usage

**Import errors**
- Ensure all dependencies are installed: `uv sync`
- Check Python version compatibility

### Debug Mode

Use verbose mode to see detailed timing and memory information:

```bash
uv run scripts/profiler.py --model-path /path/to/model -v
```
