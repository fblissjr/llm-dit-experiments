# Experiments Agent Context

Last updated: 2026-01-16

---

## CRITICAL: Prompt Standardization (2026-01-16)

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

- **Prompts Module**: [prompts.py](prompts.py) - ALL experiment prompts (MUST USE)
- **Prompting Fix Summary**: [prompting_fix_summary.md](prompting_fix_summary.md) - Why prompts changed
- **LTX-2 Prompting Guide**: [prompting_guide.md](prompting_guide.md) - Format requirements
- **Official Repo Analysis**: `internal/research/ltx2/official_repo_analysis.md` - Infrastructure we discovered in the official LTX-2 repo

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
│  │  49 Transformer Layers (output_hidden_states=True)          │   │
│  │                                                             │   │
│  │  Layer 0  → hidden_state_0  [B, T, 3840]  (embeddings)     │   │
│  │  Layer 1  → hidden_state_1  [B, T, 3840]                   │   │
│  │  ...                                                        │   │
│  │  Layer 47 → hidden_state_47 [B, T, 3840]  (low norm!)      │   │
│  │  Layer 48 → hidden_state_48 [B, T, 3840]  (final)          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Output: tuple of 49 hidden states                                  │
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

### ⚠️ CRITICAL: Quantization Affects Hidden State Distributions

The learned projection W expects hidden states with specific statistical properties from the original Gemma-3. Quantized models produce different distributions.

### Unsloth Quantized Models

**Compatibility**: ⚠️ **PARTIALLY COMPATIBLE** (with caveats)

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

**Compatibility**: ❌ **NOT COMPATIBLE** (without significant work)

GGUF is a format for llama.cpp that does NOT expose intermediate hidden states by default.

```python
# This does NOT work - GGUF models via llama-cpp-python don't expose hidden states
from llama_cpp import Llama
model = Llama("gemma-3-2b.Q4_K_M.gguf")
# No output_hidden_states parameter available
```

**Why GGUF Won't Work**:
1. llama.cpp optimizes for inference speed, not intermediate extraction
2. Hidden states are computed but immediately discarded after each layer
3. No Python API to capture intermediate activations
4. Would require modifying llama.cpp C++ code

**Theoretical Workarounds** (all require significant development):
```python
# Option 1: Modify llama.cpp to expose hidden states (C++ changes required)
# Option 2: Use a GGUF-to-PyTorch converter, then use PyTorch model
# Option 3: Run GGUF for text generation only, use full model for hidden states
```

### Recommendation Matrix

| Model Type | Hidden State Access | Distribution Match | Recommended |
|------------|--------------------|--------------------|-------------|
| Original Gemma-3 (BF16) | ✅ Full | ✅ Perfect | ✅ Yes |
| Gemma-3 (FP16) | ✅ Full | ✅ Very close | ✅ Yes |
| Unsloth 8-bit | ✅ Full | ⚠️ Minor shift | ⚠️ Test first |
| Unsloth 4-bit | ✅ Full | ❌ Significant shift | ⚠️ May need calibration |
| GGUF (any quant) | ❌ None | N/A | ❌ No |
| AWQ/GPTQ | ✅ Full | ⚠️ Varies | ⚠️ Test first |

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

---

## Key Research Finding: Layer Contribution Analysis

### The Deeper Analysis (analyze_projection_deeper.py)

Initial analysis of projection W showed uniform Frobenius norms (~2% variation). Deeper analysis revealed:

1. **Projection W is nearly uniform** - each layer block has similar weight magnitude
2. **Hidden state magnitudes vary dramatically** - late layers have much higher norms
3. **When accounting for activations** (`||W_layer @ h_layer||`):

| Rank | Layer | Contribution |
|------|-------|--------------|
| 1 | Layer 45 | **5.60%** |
| 2 | Layer 46 | **5.42%** |
| 3 | Layer 47 | **5.03%** |
| 4 | Layer 44 | **4.98%** |
| 5 | Layer 43 | **4.75%** |
| ... | ... | ... |
| 49 | Layer 0 | **0.00%** |

**Key insight**: Late layers (43-47) contribute ~25% of signal. Early layers (0-4) contribute <1%.

**Layer 48 (final) is paradoxically low (0.02%)** - possibly LM head prediction layer, not semantic.

### Implications for Experiments

- **Layer blending should focus on layers 40-47** for maximum impact
- **Early layers can likely be downweighted/excluded** with minimal effect
- **Uniform W suggests** layer importance comes from Gemma activations, not learned projection

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
