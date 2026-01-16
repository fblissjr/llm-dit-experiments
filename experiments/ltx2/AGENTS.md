# Experiments Agent Context

Last updated: 2026-01-16

## Quick Links

- **LTX-2 Prompting Guide**: [ltx2/prompting_guide.md](ltx2/prompting_guide.md) - Required reading before running LTX-2 experiments

## Directory Structure

```
experiments/
├── ltx2/                    # LTX-2 specific experiments
│   ├── prompting_guide.md   # How to write prompts for LTX-2
│   ├── layer_profile_sweep.py
│   ├── dimension_analysis.py
│   └── layer_extraction_comparison.py
├── metrics/                 # Scoring modules
│   ├── siglip_score.py      # SigLIP2 text-image alignment
│   └── image_reward.py      # Human preference alignment
├── compare/                 # Comparison infrastructure
│   ├── discovery.py         # Auto-discover experiment results
│   ├── models.py            # Data models
│   └── grid.py              # Grid generation
├── viewer/                  # Interactive web viewer
│   └── server.py            # Gradio-based viewer
└── results/                 # Experiment outputs (auto-generated)
```

## LTX-2 Architecture Reference

LTX-2 uses Gemma-3 for text encoding with multi-layer extraction:

```
Gemma-3 text_encoder(output_hidden_states=True)  → 49-tuple of [B, T, 3840]
torch.stack(dim=-1)                               → [B, T, 3840, 49]
mean-centered scaling (per-layer)                 → normalized
_pack_text_embeds() (projection W)                → [B, T, 188160]
```

Key parameters:
- **49 layers** (0-48): Gemma-3 hidden state layers
- **3840 hidden dim**: Per-layer embedding dimension
- **188160 packed dim**: 3840 × 49 flattened after projection

## Running Experiments

```bash
# Layer profile sweep (quick test)
uv run python experiments/ltx2/layer_profile_sweep.py --quick

# Full sweep (490 generations)
uv run python experiments/ltx2/layer_profile_sweep.py

# View results
uv run experiments/viewer/server.py
# Navigate to localhost:7861
```

## Viewer-Compatible Output Format

Experiments should output to `experiments/results/<experiment_name>_<timestamp>/` with:

```
<experiment>/
├── images/           # First frames as PNG
├── videos/           # Full MP4s (optional)
├── metadata/         # Per-sample JSON
│   └── <sample_name>.json
└── <experiment>_summary.json
```

Metadata JSON schema:
```json
{
  "config": {
    "prompt_id": "animal_001",
    "variable_name": "layer_index",
    "variable_value": 23,
    "seed": 42
  },
  "siglip_score": 0.287,
  "image_reward": 0.45,
  "generation_time_seconds": 12.3,
  "output_path": "images/layer_23_animal_001.png"
}
```

## Methodology Notes

### Layer Masking

When testing layer subsets, use **weighted blending** (not zeroing):

```python
# Wrong - produces out-of-distribution input
hidden_states[:, :, :, inactive_layer] = 0.0

# Better - weight compensation
weights = torch.zeros(49)
weights[active_layers] = 1.0 / len(active_layers)
hidden_states *= weights.view(1, 1, 1, -1) * 49
```

### Mean-Centering

LTX-2 applies mean-centered scaling per layer:
```python
centered = hidden - hidden.mean(dim=(1, 2), keepdim=True)
scaled = centered / (centered.std() + 1e-6)
```
