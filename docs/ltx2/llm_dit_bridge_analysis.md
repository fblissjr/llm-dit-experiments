# LTX-2 LLM-DiT Bridge Analysis

*Last Updated: 2026-01-17*

This document analyzes how LLM techniques are bridged to the LTX-2 Diffusion Transformer, identifying what transfers from LLM research, where implementations live, and opportunities for enhancement.

## Table of Contents

1. [Text Conditioning Architecture](#1-text-conditioning-architecture)
2. [Core vs Experiment Implementation](#2-core-vs-experiment-implementation)
3. [Hidden State Handling](#3-hidden-state-handling)
4. [LLM-DiT Transfer Patterns](#4-llm-dit-transfer-patterns)
5. [Opportunities for Enhancement](#5-opportunities-for-enhancement)
6. [Gaps and Missing Functionality](#6-gaps-and-missing-functionality)

---

## 1. Text Conditioning Architecture

### Overview

LTX-2 uses **Gemma3-12B** as a frozen text encoder to condition a dual-stream DiT (14B video, 5B audio). The bridging architecture extracts hidden states from all 49 decoder layers, normalizes them per-layer, concatenates, and projects to the DiT's expected dimension.

```
Text Prompt
    |
    v
[Gemma3-12B Text Encoder]
    |
    v
49 decoder layers -> [B, T, 3840, 49]
    |
    v
[Per-layer normalization: 8*(x-mean)/range]
    |
    v
[Concatenate: B, T, 188160]
    |
    v
[FeatureExtractorLinear: 188160 -> 3840]
    |
    v
[Diffusers text_proj_in]
    |
    +---> [Video Connector] -> 4096-dim -> Video DiT
    |
    +---> [Audio Connector] -> 2048-dim -> Audio DiT
```

### Key Components

| Component | Location | Dimension | Notes |
|-----------|----------|-----------|-------|
| Text Encoder | `Gemma3ForConditionalGeneration` | 3840 per layer | Frozen, Q4 QAT quantized |
| Layer Stack | `encode()` / `encode_multilayer()` | [B, T, 3840, 49] | All 49 decoder layers |
| Normalization | `_norm_and_concat_layers()` | [B, T, 188160] | Per-layer: 8*(x-mean)/range |
| Feature Extractor | `FeatureExtractorLinear` | 188160 -> 3840 | Learned projection |
| Video Connector | diffusers `connectors.video_connector` | 3840 -> 4096 | Bidirectional + thinking tokens |
| Audio Connector | diffusers `connectors.audio_connector` | 3840 -> 2048 | Separate from video |

### Normalization Formula

The per-layer normalization from LTX-2 paper:

```python
# Per layer i:
mean_i = hidden_states[:, :, :, i].mean(over valid tokens)
range_i = max_i - min_i
normalized_i = 8 * (x - mean_i) / (range_i + eps)
```

This differs from standard LayerNorm - it uses range-based scaling and a fixed multiplier of 8.

### Thinking Tokens (Registers)

The text connectors use **128 learnable "thinking tokens"** that:
- Replace padding positions in the text sequence
- Serve as global information carriers with bidirectional attention
- Allow aggregation across the sequence before DiT cross-attention

This mirrors the "register tokens" concept from LLM research (ViT attention sinks).

---

## 2. Core vs Experiment Implementation

### Core Implementation

**File: `/home/fbliss/workspace/llm-dit-experiments/src/llm_dit/encoders/gemma3.py`**

The core encoder implements:

1. **Basic Encoding** (`encode()`):
   - Standard multi-layer extraction
   - Returns projected [B, T, 3840] embeddings
   - Used by `LTX2Pipeline` for generation

2. **Multi-layer Access** (`encode_multilayer()`):
   - Exposes full [B, T, 3840, 49] layer stack
   - Supports selective layer extraction via `layer_indices`
   - Optional sub-layer extraction (attention/MLP outputs)
   - Critical for routing experiments

3. **SubLayerExtractor** class:
   - Forward hooks to capture attention and MLP outputs separately
   - Extracts outputs after `post_attention_layernorm` and `post_feedforward_layernorm`
   - Enables sub-layer routing experiments

```python
# Core API for experiments
result = encoder.encode_multilayer(
    texts=["A cat walking"],
    layer_indices=[10, 20, 30, 40, 48],  # Selective extraction
    return_projected=True,
    extract_sub_layers=True,  # Get attention/MLP separately
)
# result['layer_stack']: [B, T, 3840, 5]
# result['attention_stack']: [B, T, 3840, 5]
# result['mlp_stack']: [B, T, 3840, 5]
```

**File: `/home/fbliss/workspace/llm-dit-experiments/src/llm_dit/router/token_layer_router.py`**

Per-token layer routing module:

- `TokenLayerRouter`: Learns per-token layer weights (~250K params)
- `extract_router_input()`: Extracts router input from layer stack
- `SparsityLoss`: Encourages compute-efficient routing
- Supports soft, top-k, and gumbel-softmax routing modes

**File: `/home/fbliss/workspace/llm-dit-experiments/src/llm_dit/pipelines/ltx2.py`**

Pipeline wrapper with:

- `generate_with_embeddings()`: Bypass text encoder with pre-computed embeddings
- Integration with enhancement techniques (latent normalization, FFN chunking)
- Memory management for 24GB VRAM constraint

### Experiment Implementation

| Experiment | File | Technique | Status |
|------------|------|-----------|--------|
| Activation Steering | `experiments/ltx2/activation_steering.py` | Contrastive embedding steering | Complete |
| Layer Extraction | `experiments/ltx2/layer_extraction_comparison.py` | Layer subset ablation | Complete |
| Router Training | `experiments/ltx2/train_router.py` | Per-token routing optimization | Scaffold only |
| Thinking Tokens | `experiments/ltx2/thinking_token_analysis.py` | Register analysis | Partial |
| Projection Analysis | `experiments/ltx2/analyze_projection_deeper.py` | Weight structure analysis | Complete |
| Entropy Encoding | `experiments/ltx2/entropy_guided_encoding.py` | Confidence weighting | Complete |

---

## 3. Hidden State Handling

### Layer Selection Options

The codebase supports multiple layer selection strategies:

| Strategy | Code Location | Description |
|----------|---------------|-------------|
| All Layers | `encode()` default | Uniform blend of all 49 layers |
| Selective Layers | `encode_multilayer(layer_indices=[...])` | Specific layer subset |
| Single Layer | `encode_multilayer(layer_indices=[i])` | One layer only |
| Sub-layers | `SubLayerExtractor` | Attention/MLP outputs separately |

### Router Input Modes

From `token_layer_router.py`:

```python
RouterInputMode = Literal[
    "layer_0",      # First decoder layer
    "layer_24",     # Middle layer
    "layer_47",     # High-contribution layer
    "layer_48",     # Final layer (may be LM-biased)
    "mean",         # Average across all layers (recommended)
    "weighted",     # Weighted average using pre-computed weights
    "attention",    # Attention outputs only (requires SubLayerExtractor)
    "mlp",          # MLP outputs only (requires SubLayerExtractor)
]
```

### Token-Level vs Sequence-Level

Current implementation applies operations at different granularities:

| Operation | Granularity | Location |
|-----------|-------------|----------|
| Layer normalization | Per-layer, all tokens | `_norm_and_concat_layers()` |
| Feature projection | Per-token | `FeatureExtractorLinear` |
| Router weights | Per-token, per-layer | `TokenLayerRouter` |
| Activation steering | Per-token (uniform direction) | `activation_steering.py` |
| Entropy weighting | Per-token | `entropy_guided_encoding.py` |

---

## 4. LLM-DiT Transfer Patterns

### Transfer Assessment Table

| LLM Technique | DiT Analog | Transfer Status | Implementation Location |
|---------------|------------|-----------------|------------------------|
| **Attention sinks** | Thinking tokens / registers | Transfers conceptually | diffusers connector |
| **Layer probing** | Multi-layer extraction | Direct transfer | `encode_multilayer()` |
| **Activation steering** | Embedding manipulation | Transfers well | `activation_steering.py` |
| **KV cache** | Cached text embeddings | Direct transfer | `generate_with_embeddings()` |
| **Chain-of-thought** | Extended prompting | Partial transfer | Prompt engineering only |
| **Hidden state extraction** | Layer stack access | Direct transfer | `SubLayerExtractor` |
| **Per-token routing** | Layer weight learning | Novel adaptation | `TokenLayerRouter` |
| **Entropy-based filtering** | Confidence weighting | Transfers with caveats | `entropy_guided_encoding.py` |

### What Transfers Well

1. **Layer Probing / Feature Hierarchy**
   - LLM insight: Different layers capture different linguistic levels
   - DiT application: Early layers for syntax, late layers for semantics
   - Evidence: `layer_extraction_comparison.py` ablations

2. **Activation Steering**
   - LLM insight: Contrastive pairs define semantic directions
   - DiT application: Steer embeddings toward "detailed" or "bright"
   - Evidence: `activation_steering.py` with multiple direction types

3. **Register/Sink Tokens**
   - LLM insight: Models need "attention sinks" for numerical stability
   - DiT application: 128 thinking tokens aggregate global info
   - Evidence: Architecture design in LTX-2 paper

### What Requires Adaptation

1. **Per-Token Routing**
   - LLM context: Mixture-of-Experts (MoE) routes tokens to experts
   - DiT adaptation: Route tokens to optimal layers instead of experts
   - Challenge: Reward signal requires generation (non-differentiable)
   - Status: Scaffold exists, reward computation TODO

2. **Entropy-Based Weighting**
   - LLM context: Token prediction entropy indicates uncertainty
   - DiT adaptation: Use embedding variance as proxy (no logits)
   - Caveat: Proxy metric may not correlate with actual uncertainty

### What Doesn't Transfer

1. **Causal Attention Patterns**
   - LLM: Causal masking for autoregressive generation
   - DiT: Bidirectional attention in connectors (solves this)
   - The thinking tokens explicitly enable bidirectional aggregation

2. **Token-Level Generation**
   - LLM: Generate one token at a time
   - DiT: Generate all latents simultaneously
   - Implication: Can't do token-level steering during generation

---

## 5. Opportunities for Enhancement

### Immediate Opportunities

1. **Complete Router Training**
   - File: `experiments/ltx2/train_router.py`
   - Gap: `compute_reward()` returns dummy zeros
   - Solution: Implement SigLIP-based reward or proxy metric
   - Effort: Medium

2. **Thinking Token Manipulation**
   - File: `experiments/ltx2/thinking_token_analysis.py`
   - Gap: Analysis exists but manipulation experiments incomplete
   - Solution: Implement generation with/without thinking tokens
   - Effort: Medium

3. **Unified Routing API**
   - Gap: Custom routing requires manual embedding computation
   - Solution: Add `generate_with_routed_embeddings()` to pipeline
   - Effort: Low

### Research Opportunities

1. **Learned Layer Blending**
   - Current: Uniform blend or manual selection
   - Opportunity: Learn optimal blend weights per-prompt-type
   - Approach: Train lightweight predictor on prompt categories

2. **Sub-Layer Routing**
   - Current: Route to full layer outputs
   - Opportunity: Route attention vs MLP outputs differently
   - Hypothesis: Attention captures relations, MLP captures features

3. **Cross-Modal Routing**
   - Current: Same embeddings go to video and audio connectors
   - Opportunity: Different layer selections for video vs audio
   - Rationale: Audio may need different semantic levels than video

### Code Consolidation

1. **Experiment → Core Promotion**
   - `activation_steering.py` patterns → reusable `SteeringDirection` class
   - `layer_extraction_comparison.py` hooks → `LayerMaskingHook` utility

2. **Shared Infrastructure**
   - Many experiments duplicate pipeline loading code
   - Solution: Create `ExperimentBase` class with common setup

---

## 6. Gaps and Missing Functionality

### Critical Gaps

| Gap | Impact | Files Affected | Suggested Fix |
|-----|--------|----------------|---------------|
| Router reward computation | Can't train router | `train_router.py` | Implement SigLIP or proxy reward |
| Audio connector access | Can't experiment with audio conditioning | `ltx2.py` | Expose `audio_connector` in wrapper |
| Thinking token control | Can't ablate registers | `thinking_token_analysis.py` | Add register masking hooks |

### Implementation Gaps

1. **No Direct Connector Access**
   - The diffusers pipeline encapsulates connectors
   - Experiments hook into `encode_prompt` instead
   - Makes clean experiments harder

2. **Missing Gradient Flow**
   - `generate_with_embeddings()` is inference-only
   - Can't backprop through DiT for embedding optimization
   - Would need custom training loop

3. **Incomplete Sub-Layer Support**
   - `SubLayerExtractor` captures outputs but can't inject modified versions
   - Need bidirectional hooks for full sub-layer routing

### Documentation Gaps

1. **No architecture diagram** showing embedding flow
2. **Missing benchmark results** for layer ablations
3. **No guidance** on when to use which routing mode

---

## Appendix: Key File References

### Core Files

| File | Purpose | Key Functions |
|------|---------|---------------|
| `/home/fbliss/workspace/llm-dit-experiments/src/llm_dit/encoders/gemma3.py` | Text encoding | `encode()`, `encode_multilayer()`, `SubLayerExtractor` |
| `/home/fbliss/workspace/llm-dit-experiments/src/llm_dit/pipelines/ltx2.py` | Pipeline wrapper | `generate_with_embeddings()` |
| `/home/fbliss/workspace/llm-dit-experiments/src/llm_dit/router/token_layer_router.py` | Per-token routing | `TokenLayerRouter`, `extract_router_input()` |
| `/home/fbliss/workspace/llm-dit-experiments/src/llm_dit/encoders/protocol.py` | Encoder interface | `TextEncoderProtocol` |

### Experiment Files

| File | Technique | LLM Analog |
|------|-----------|------------|
| `/home/fbliss/workspace/llm-dit-experiments/experiments/ltx2/activation_steering.py` | Embedding steering | Activation patching |
| `/home/fbliss/workspace/llm-dit-experiments/experiments/ltx2/layer_extraction_comparison.py` | Layer ablation | Layer probing |
| `/home/fbliss/workspace/llm-dit-experiments/experiments/ltx2/train_router.py` | Router training | MoE routing |
| `/home/fbliss/workspace/llm-dit-experiments/experiments/ltx2/thinking_token_analysis.py` | Register analysis | Attention sinks |
| `/home/fbliss/workspace/llm-dit-experiments/experiments/ltx2/entropy_guided_encoding.py` | Confidence weighting | Entropy filtering |
| `/home/fbliss/workspace/llm-dit-experiments/experiments/ltx2/analyze_projection_deeper.py` | Weight analysis | Probing classifiers |

---

## Summary

The LTX-2 codebase demonstrates a sophisticated LLM-DiT bridge with:

- **Strong foundation**: Multi-layer extraction, sub-layer hooks, flexible routing API
- **Active research**: Multiple experiment scripts exploring LLM technique transfer
- **Clear transfer patterns**: Layer probing, activation steering, and register tokens transfer well
- **Key gaps**: Router training incomplete, audio conditioning not exposed, thinking token manipulation limited

The primary research direction - per-token layer routing - has solid scaffolding but needs reward computation to complete. The codebase is well-positioned to explore whether LLM-style routing can improve DiT conditioning quality.
