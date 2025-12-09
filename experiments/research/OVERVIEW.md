# Z-Image Research Directions

This directory contains structured research questions, ablation study designs, and experimentation plans for the llm-dit-experiments platform.

## Overview

Z-Image is a 6B parameter LLM-DiT model that uses Qwen3-4B as a text encoder. The architecture is unique:

```
Text Prompt
    |
    v
Qwen3-4B (2560 hidden dim, 36 layers)
    |
    v
hidden_states[-2] extraction (configurable via --hidden-layer)
    |
    v
Linear projection (2560 -> 3840)
    |
    v
Context Refiner (2 layers, no timestep modulation)
    |
    v
DiT Main Blocks (with timestep modulation)
    |
    v
VAE Decode (16-channel Wan-family)
    |
    v
Image Output
```

Key characteristics:
- **Max text tokens**: 1504 (DiT RoPE limit - empirically verified 2025-12-09)
- **CFG scale**: 0.0 (baked in via Decoupled-DMD training)
- **Inference steps**: 8-9 (turbo distilled)
- **Scheduler shift**: 3.0 (default for turbo)

## Recent Discoveries (2025-12-09)

### Token Limit Discovery (CRITICAL)

Through systematic binary search testing, we discovered the actual maximum sequence length is **1504 tokens**, not 1024 (conservative DiffSynth choice) or 1536 (config value).

**Key findings:**
- 1504 = 47 × 32, while config specifies 1536 = 48 × 32
- This is likely an **off-by-one bug** in RoPE frequency table indexing in diffusers ZImageTransformer2DModel
- NOT related to Flash Attention (tested with SDPA backend)
- NOT related to image size (tested multiple resolutions)
- Gives 46.9% more capacity than previously thought (1024 → 1504)

**Impact:** Many prompts that previously required compression can now fit within the limit.

### Compression Algorithm Upgrade

Upgraded `attention_pool` mode from L2 norm to **cosine similarity** based importance weighting:
- Measures semantic distinctiveness (how different a token is from neighbors)
- Uses ±2 token window for context
- Importance = 1 - avg_cosine_similarity_to_neighbors
- Much better at preserving salient concepts vs filler content

### RoPE Architecture Insights

- `rope_theta=256` is intentionally low for precise spatial discrimination (not a mistake)
- LLMs use theta=1,000,000 for extrapolation; DiT uses 256 for local precision
- Multi-axis RoPE design means text and image don't compete for positions
  - Text: uses axis 0 sequentially (0-1503)
  - Image: shares ONE axis 0 position, uses axes 1 & 2 spatially
- NTK scaling and YaRN are potential avenues for extending beyond 1504

## Research Documents

| Document | Description |
|----------|-------------|
| [Ablation Studies](./ablation_studies.md) | Controlled experiments varying single parameters |
| [Metrics and Data](./metrics_and_data.md) | What to measure and how to collect it |
| [Future Directions](./future_directions.md) | Ambitious research paths and novel ideas |
| [Assumptions to Challenge](./assumptions_to_challenge.md) | Testing conventional wisdom |
| [Open Questions](./open_questions.md) | Unanswered questions requiring investigation |
| [Diffusers RoPE Bug](./diffusers_rope_bug.md) | Investigation of off-by-one bug limiting tokens to 1504 |

## Priority Ranking

For hobbyist time constraints, prioritize in this order:

1. **Shift parameter sweep** - High impact, easy to run, actionable results
2. **Think block content experiments** - Novel, not documented elsewhere
3. **Hidden layer extraction comparison** - Could reveal architectural insights
4. **Long prompt compression evaluation** - Validate existing experimental code
5. **Embedding space steering** - Most ambitious, most publishable if successful

## Experiment Framework

A proposed experiment runner for systematic ablations:

```python
from dataclasses import dataclass
from typing import List, Tuple
import pandas as pd

@dataclass
class ExperimentConfig:
    name: str
    variable: str           # Parameter being varied
    values: list            # Values to test
    metrics: List[str]      # fid, clip_score, lpips, user_pref
    n_samples: int          # Images per condition
    seed_range: Tuple[int, int]  # Seed range for reproducibility
    baseline: dict          # Baseline configuration

class ExperimentRunner:
    def __init__(self, pipeline, encoder):
        self.pipeline = pipeline
        self.encoder = encoder

    def run(self, config: ExperimentConfig) -> pd.DataFrame:
        results = []
        for value in config.values:
            for seed in range(*config.seed_range):
                # Generate with this configuration
                image = self.generate(
                    **config.baseline,
                    **{config.variable: value},
                    seed=seed
                )
                # Compute metrics
                metrics = self.compute_metrics(image, config.metrics)
                results.append({
                    config.variable: value,
                    "seed": seed,
                    **metrics
                })
        return pd.DataFrame(results)
```

## Quick Reference: Key Parameters

| Parameter | Location | Default | Range to Test |
|-----------|----------|---------|---------------|
| `shift` | Scheduler | 3.0 | 1.0 - 5.0 |
| `steps` | Pipeline | 9 | 4, 6, 8, 9, 12, 16, 25 |
| `hidden_layer` | Encoder | -2 | -1, -2, -3, -4 |
| `long_prompt_mode` | Utils | truncate | truncate, interpolate, pool, attention_pool |
| `force_think_block` | Formatter | False | True, False |
| `thinking_content` | Formatter | None | Various content types |
| `system_prompt` | Formatter | Default | None, custom |

## Getting Started

1. Pick an experiment from [Ablation Studies](./ablation_studies.md)
2. Check [Metrics and Data](./metrics_and_data.md) for measurement approach
3. Run with the existing profiler or build custom script
4. Document findings in this directory

## Contributing Findings

When you complete an experiment, add your findings:

```
experiments/research/results/
    shift_parameter_sweep_2025-12-XX.md
    think_block_ablation_2025-12-XX.md
    ...
```

Include:
- Experimental setup (exact commands, configs)
- Raw data or summary statistics
- Visualizations if applicable
- Conclusions and next questions
