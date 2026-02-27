# LTX-2 Experiments Agent Context

*last updated: 2026-02-27*

---

## research status legend

| Tag | Status | Meaning |
|-----|--------|---------|
| `[VALIDATED]` | **Validated** | Confirmed through experiments or architecture analysis |
| `[OPEN]` | **Open** | Hypothesis needs testing or re-testing |
| `[NEEDS_CHECK]` | **Needs Verification** | Previous results may have bugs |
| `[DEAD_END]` | **Dead-End** | Tested, doesn't work |

**Consolidated findings:** [docs/findings/](docs/findings/)

---

## CRITICAL: Prompt Standardization

**All experiment prompts have been centralized.** Do NOT define inline prompts.

```python
# REQUIRED - import from centralized module
from experiments.ltx2.prompts import CATEGORY_PROMPTS, get_all_prompts

TEST_PROMPTS = CATEGORY_PROMPTS  # 8 prompts, 100+ words each
# or
TEST_PROMPTS = get_all_prompts(quick=True)  # 5 prompts for fast testing
```

**Why?** Prior experiments used short 60-80 word prompts that are out-of-distribution for LTX-2's training data (which uses 100-300 word prose with dialogue, scene headings, and camera directions).

**Full details**: [prompting_fix_summary.md](prompting_fix_summary.md)

---

## Quick Links

**Navigation:**
- **[../CLAUDE.md](../CLAUDE.md)** - Parent navigation (all experiments)
- **[docs/findings/](docs/findings/)** - Consolidated research findings with status

**Prompts:**
- **Prompts Module**: [prompts.py](prompts.py) - ALL experiment prompts (MUST USE)
- **Prompting Fix Summary**: [prompting_fix_summary.md](prompting_fix_summary.md) - Why prompts changed
- **LTX-2 Prompting Guide**: [prompting_guide.md](prompting_guide.md) - Format requirements

**Research:**
- **[docs/findings/apollo_analysis.md](docs/findings/apollo_analysis.md)** - Apollo paper insights
- **[docs/findings/research_synthesis.md](docs/findings/research_synthesis.md)** - Consolidated research status
- `internal/research/ltx2/` - Working research notes

## Directory Structure

```
experiments/
├── ltx2/                    # LTX-2 specific experiments
│   ├── prompts.py                # CENTRALIZED PROMPTS (MUST USE)
│   ├── prompting_guide.md        # How to write prompts for LTX-2
│   ├── prompting_fix_summary.md  # Why prompts were standardized
│   ├── layer_profile_sweep.py    # Full 49-layer sweep with soft masking
│   ├── layer_extraction_comparison.py  # Layer subset comparison
│   ├── layer_blend_sweep.py      # Weighted layer blending experiments
│   ├── prompt_format_ablation.py # Test structured formats vs prose
│   ├── analyze_projection_matrix.py    # Zero-cost W analysis
│   ├── analyze_projection_deeper.py    # Activation-weighted contribution
│   ├── thinking_token_analysis.py      # Register token analysis
│   ├── dimension_analysis.py
│   └── chunk_boundary_analysis.py      # VAE temporal chunk boundary hypothesis test
├── metrics/                 # Scoring modules
│   ├── siglip_score.py      # SigLIP2 text-image alignment (local models supported)
│   └── image_reward.py      # Human preference alignment
├── compare/                 # Comparison infrastructure
│   ├── discovery.py         # Auto-discover experiment results
│   ├── models.py            # Data models
│   └── grid.py              # Grid generation
├── viewer/                  # Interactive web viewer
│   └── server.py            # Gradio-based viewer
└── results/                 # Experiment outputs (auto-generated)
```

---

## Gemma-3 to LTX-2 Adapter Architecture

### Overview

LTX-2 does NOT use Gemma-3 as a traditional text encoder (final layer only). Instead, it extracts **ALL 49 hidden states** and projects them through a learned adapter. This is critical for understanding what happens with quantized models.

### Full Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                         GEMMA-3 (2B)                                │
│                                                                     │
│  Input: "A golden retriever runs through a park..."                 │
│         ↓                                                           │
│  Tokenizer → [B, T] token IDs                                       │
│         ↓                                                           │
│  Embedding layer                                                    │
│         ↓                                                           │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  48 Transformer Layers + Embedding (output_hidden_states=True) │   │
│  │                                                             │   │
│  │  Layer 0  → hidden_state_0  [B, T, 3840]  (embeddings)     │   │
│  │  Layer 1  → hidden_state_1  [B, T, 3840]  (transformer 1)  │   │
│  │  ...                                                        │   │
│  │  Layer 47 → hidden_state_47 [B, T, 3840]  (low norm!)      │   │
│  │  Layer 48 → hidden_state_48 [B, T, 3840]  (final output)   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Output: tuple of 49 hidden states (ALL are used, including embeddings)
└─────────────────────────────────────────────────────────────────────┘
                              ↓
                    torch.stack(dim=-1)
                              ↓
                    [B, T, 3840, 49]
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    _pack_text_embeds()                              │
│                    (NO LEARNED PARAMETERS)                          │
│                                                                     │
│  1. Compute masked statistics per layer (mean, min, max)           │
│  2. Mean-center: x - mean                                          │
│  3. Scale to [-8, +8] range: x / (max - min) * 8                   │
│  4. Flatten: [B, T, 3840, 49] → [B, T, 188160]                     │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                 connectors.text_proj_in                             │
│                 (LEARNED PROJECTION W)                              │
│                                                                     │
│  nn.Linear(188160, 3840, bias=False)                               │
│                                                                     │
│  W shape: [3840, 188160] = [3840, 49 × 3840]                       │
│                                                                     │
│  This W was trained via MSE loss against DiT denoising.            │
│  It learned which layers matter and how to combine them.           │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
                    [B, T, 3840]
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│              LTX2ConnectorTransformer1d                             │
│              (Video or Audio Connector)                             │
│                                                                     │
│  1. Replace padding positions with 128 learnable "thinking tokens" │
│  2. Full bidirectional attention (no causal masking)               │
│  3. Transform to modality-specific representation                  │
│                                                                     │
│  Separate connectors for video (4096 dim) and audio (2048 dim)     │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
                    DiT Cross-Attention
```

### Key Insight: Why ALL 49 Layers?

From the LTX-2 paper (Section 3.2.1):
> "We extract hidden states from ALL layers of Gemma-3, not just the final layer. Different layers encode different types of information - early layers capture syntax and surface features, middle layers capture semantics, late layers capture abstract concepts."

The projection W learns to combine these optimally for video generation.

---

## Quantized Model Considerations

### CRITICAL: Quantization Affects Hidden State Distributions

The learned projection W expects hidden states with specific statistical properties from the original Gemma-3. Quantized models produce different distributions.

### Unsloth Quantized Models

**Compatibility**: **PARTIALLY COMPATIBLE** (with caveats)

Unsloth 4-bit models can still expose hidden states via `output_hidden_states=True`, but:

```python
# This works but produces different hidden states
model = AutoModelForCausalLM.from_pretrained(
    "unsloth/gemma-3-2b-it-bnb-4bit",
    output_hidden_states=True,
)
outputs = model(input_ids, output_hidden_states=True)
hidden_states = outputs.hidden_states  # tuple of 49 tensors
```

**Issues**:
1. **Distribution shift**: Quantized activations have different mean/variance
2. **Layer-specific effects**: Some layers affected more than others
3. **Calibration mismatch**: W was trained on full-precision hidden states

**Workarounds**:
```python
# Option 1: Scale to match original distribution
hidden_states = [h * scale_factor for h in hidden_states]

# Option 2: Fine-tune the projection W on quantized hidden states
# (Requires training data and compute)

# Option 3: Use higher bit quantization (8-bit less affected than 4-bit)
```

### GGUF Models

**Compatibility**: **NOT COMPATIBLE**. GGUF/llama.cpp does not expose intermediate hidden states. Would require C++ modifications. Not worth pursuing.

### Recommendation Matrix

| Model Type | Hidden State Access | Distribution Match | Recommended |
|------------|--------------------|--------------------|-------------|
| Original Gemma-3 (BF16) | Full | Perfect | Yes |
| Gemma-3 (FP16) | Full | Very close | Yes |
| Unsloth 8-bit | Full | Minor shift | Test first |
| Unsloth 4-bit | Full | Significant shift | May need calibration |
| GGUF (any quant) | None | N/A | No |
| AWQ/GPTQ | Full | Varies | Test first |

### Testing Quantized Models

If you want to test a quantized model, run this diagnostic first:

```python
# Compare hidden state distributions
def compare_hidden_states(full_model, quant_model, prompt):
    full_hidden = get_all_hidden_states(full_model, prompt)
    quant_hidden = get_all_hidden_states(quant_model, prompt)

    for layer_idx in range(49):
        full_stats = {
            "mean": full_hidden[layer_idx].mean().item(),
            "std": full_hidden[layer_idx].std().item(),
            "norm": full_hidden[layer_idx].norm().item(),
        }
        quant_stats = {
            "mean": quant_hidden[layer_idx].mean().item(),
            "std": quant_hidden[layer_idx].std().item(),
            "norm": quant_hidden[layer_idx].norm().item(),
        }
        # Compare and flag significant deviations
```

---

## LTX-2 Architecture Quick Reference

Key parameters:
- **49 layers** (0-48): Gemma-3 hidden state layers
- **3840 hidden dim**: Per-layer embedding dimension
- **188160 packed dim**: 3840 × 49 flattened for projection
- **128 thinking tokens**: Learnable registers for global context
- **48 DiT blocks**: Transformer blocks in the diffusion model (14B video + 5B audio)
- **Dual-stream AV**: `BasicAVTransformerBlock` extends DiT blocks with bidirectional cross-modal attention (A2V, V2A). Three modes: video-only, audio-only, dual-stream
- **Audio connector**: 2048 dim (vs 4096 for video). Separate `LTX2ConnectorTransformer1d` instance
- **STG perturbation**: Per-sample attention skipping via `PerturbationConfig` for spatio-temporal guidance

---

## Research Findings

See **[docs/findings/](docs/findings/)** for consolidated research with status tracking:

- [apollo_analysis.md](docs/findings/apollo_analysis.md) - Apollo paper transfer analysis
- [research_synthesis.md](docs/findings/research_synthesis.md) - All findings with status tracking

---

## Running Experiments

```bash
# Layer profile sweep (quick test)
uv run python experiments/ltx2/layer_profile_sweep.py --quick

# Full sweep (490 generations)
uv run python experiments/ltx2/layer_profile_sweep.py

# Layer blend sweep (weighted combinations)
uv run python experiments/ltx2/layer_blend_sweep.py --quick  # 3 blends × 2 prompts
uv run python experiments/ltx2/layer_blend_sweep.py          # Full: 10 blends × 5 prompts

# Zero-cost projection analysis
uv run python experiments/ltx2/analyze_projection_matrix.py
uv run python experiments/ltx2/analyze_projection_deeper.py  # Includes activation-weighted analysis

# Thinking token analysis
uv run python experiments/ltx2/thinking_token_analysis.py

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
