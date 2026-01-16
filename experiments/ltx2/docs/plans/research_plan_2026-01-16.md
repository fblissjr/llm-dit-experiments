# LTX-2 Research Plan: Validated Path to Per-Token Layer Routing

Last updated: 2026-01-16

---

## Overview

This plan prioritizes **validation before investment**. Based on synthesis of community insights, Apollo paper findings, and our architectural analysis, we proceed in phases that minimize wasted effort.

**Goal**: Determine if per-token layer routing improves LTX-2 video generation quality.

**Timeline**: ~1 week of focused effort (RTX 4090, 24GB VRAM)

---

## Phase 0: Quick Wins (Today)

### 0.1 Run Chunk Boundary Experiment

**Purpose**: Test the 8-frame temporal boundary hypothesis.

**Command**:
```bash
uv run python experiments/ltx2/chunk_boundary_analysis.py --quick
```

**Time**: ~30 minutes (4 generations)

**What to look for**:
- Motion hitches at frames 8, 16, 24
- Boundary diff > mid-chunk diff by >10%
- Visual discontinuities in metronome/walking person videos

**Outcome**:
- ✓ If supported: Document finding, proceed with temporal-aware routing
- ✗ If not supported: Note that chunk boundaries are soft, focus on semantic routing

### 0.2 Review Generated Videos

**Purpose**: Visual inspection complements quantitative metrics.

**Command**:
```bash
uv run experiments/viewer/server.py
# Navigate to localhost:7861
```

**Checklist**:
- [ ] Watch each video at 0.5x speed
- [ ] Note any motion stutters
- [ ] Compare frame 8 vs frame 4 transitions
- [ ] Document observations in log

---

## Phase 1: Validation Experiments (Days 1-2)

### 1.1 Projection W Analysis

**Purpose**: Determine if learned projection differs from uniform averaging.

**Method**:
```python
# Pseudocode
uniform_output = mean(all_49_layers, dim=-1)  # [B, T, 3840]
projected_output = W @ concat(all_49_layers)   # [B, T, 3840]

# Compare
cosine_sim = F.cosine_similarity(uniform_output, projected_output)
output_diff = (uniform_output - projected_output).norm()
```

**If similar (cosine_sim > 0.95)**:
- Projection hasn't learned meaningful layer differentiation
- Routing may not help
- Pivot to alternative directions

**If different (cosine_sim < 0.9)**:
- Projection has learned layer-specific weighting
- Routing has potential
- Proceed to Phase 2

**Script**: Create `experiments/ltx2/projection_comparison.py`

### 1.2 DiT Cross-Attention Analysis

**Purpose**: Check if different tokens attend differently during generation.

**Method**:
1. Hook into DiT cross-attention layers
2. Extract attention maps for 5-10 prompts
3. Compute attention entropy per token
4. Look for patterns: Do noun tokens attend differently than verb tokens?

**Expected outcome**:
- High variance in attention entropy → tokens have different roles → routing justified
- Uniform entropy → tokens treated similarly → routing may not help

**Script**: Create `experiments/ltx2/cross_attention_analysis.py`

### 1.3 Random Routing Sanity Check

**Purpose**: Verify that routing CAN affect output (not just noise).

**Method**:
```python
# Generate same prompt with:
# 1. Uniform layer weights (baseline)
# 2. Random per-token layer weights
# 3. Different random weights

# Compare outputs
# If all identical → routing has no effect
# If different → routing mechanism works
```

**This is a sanity check**: If random routing produces identical outputs to uniform, the routing mechanism is broken or ineffective.

---

## Phase 2: Router Development (Days 3-5)

*Only proceed if Phase 1 validation passes*

### 2.1 Router Architecture

**Design** (from previous work):
```python
class TokenLayerRouter(nn.Module):
    """
    Per-token routing over 49 Gemma layers.
    ~249K parameters (trainable)
    """
    def __init__(self, hidden_dim=3840, num_layers=49):
        self.query_proj = nn.Linear(hidden_dim, 256)
        self.layer_keys = nn.Parameter(torch.randn(num_layers, 256))
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, token_features):
        # token_features: [B, T, 3840] from last layer or pooled
        queries = self.query_proj(token_features)  # [B, T, 256]
        logits = queries @ self.layer_keys.T       # [B, T, 49]
        weights = F.softmax(logits / self.temperature, dim=-1)
        return weights  # [B, T, 49]
```

**Input options** (test all):
1. Last layer only (layer 48)
2. Mean of all layers
3. Mean of late layers (43-47)
4. Learned weighted combination

### 2.2 Training Strategy

**Objective**: REINFORCE with SigLIP reward

```python
# Pseudocode
for prompt in training_prompts:
    # Sample routing weights
    routing_weights = router(token_features)

    # Generate with routing
    video = generate_with_routing(prompt, routing_weights)

    # Compute reward
    reward = siglip_score(prompt, video.first_frame)

    # REINFORCE gradient
    loss = -reward * log_prob(routing_weights)
    loss.backward()
```

**Key considerations**:
- Use baseline subtraction to reduce variance
- Start with frozen DiT, only train router
- Monitor for collapse (all tokens same routing)
- Add entropy regularization if collapse detected

### 2.3 Evaluation Protocol

**Metrics**:
| Metric | Tool | Purpose |
|--------|------|---------|
| SigLIP score | `experiments/metrics/siglip_score.py` | Spatial quality |
| Routing entropy | Custom | Detect collapse |
| Layer usage distribution | Custom | Understand what router learned |
| Visual inspection | Viewer | Qualitative assessment |

**Comparison**:
1. Baseline (uniform routing)
2. Trained router
3. Oracle (if we had per-token labels)

---

## Phase 3: Analysis & Iteration (Days 6-7)

### 3.1 If Router Improves Quality

**Document**:
- Which tokens route to which layers?
- Do nouns → late layers, verbs → middle layers?
- Does routing vary by prompt type?

**Next steps**:
- Scale up training data
- Test on longer videos
- Explore timestep-conditional routing

### 3.2 If Router Doesn't Help

**Pivot options** (in priority order):

1. **Brightness steering refinement**
   - Already showed +6.8% improvement
   - Low-hanging fruit
   - Refine the direction finding

2. **Timestep-conditional layers**
   - Different denoising stages may need different layers
   - Early steps: semantic (late layers)
   - Late steps: detail (middle layers?)

3. **Thinking token analysis**
   - What do the 128 registers capture?
   - Can we add domain-specific registers?

4. **Dual encoder exploration**
   - Add temporal encoder (like Apollo's InternVideo2)
   - More invasive but Apollo shows ~7% gain

---

## Decision Tree

```
START
  │
  ▼
Phase 0: Chunk boundary experiment
  │
  ├─[Supported]─► Document temporal finding
  │               │
  │               ▼
  │             Consider temporal-aware routing
  │
  └─[Not supported]─► Note soft boundaries
                      │
                      ▼
                    Focus on semantic routing
  │
  ▼
Phase 1: Validation experiments
  │
  ├─[All pass]─► Proceed to Phase 2 (router development)
  │               │
  │               ▼
  │             Train router, evaluate
  │               │
  │               ├─[Improves]─► Document, scale up
  │               │
  │               └─[No improvement]─► Pivot to alternatives
  │
  └─[Any fail]─► Pivot immediately
                  │
                  ▼
                Brightness steering OR
                Timestep conditioning OR
                Thinking token analysis
```

---

## Resource Allocation

### Hardware
- RTX 4090 (24GB VRAM)
- Memory-optimized pipeline (8-bit quantization + offloading)

### Time Budget

| Phase | Task | Time |
|-------|------|------|
| 0 | Chunk boundary experiment | 1 hour |
| 1.1 | Projection W analysis | 2 hours |
| 1.2 | Cross-attention analysis | 3 hours |
| 1.3 | Random routing sanity | 1 hour |
| 2.1 | Router implementation | 2 hours |
| 2.2 | Training loop | 4 hours |
| 2.3 | Evaluation | 2 hours |
| 3 | Analysis & documentation | 3 hours |
| **Total** | | **~18 hours** |

### Prompt Budget

| Phase | Prompts | Purpose |
|-------|---------|---------|
| 0 | 4-20 | Chunk boundary |
| 1 | 20-50 | Validation |
| 2 | 200-500 | Router training |
| 3 | 50-100 | Evaluation |

---

## Success Criteria

### Minimum Viable Success
- [ ] Chunk boundary finding documented (positive or negative)
- [ ] Validation experiments completed
- [ ] Clear go/no-go decision on routing

### Target Success
- [ ] Router trained and evaluated
- [ ] ≥3% SigLIP improvement over baseline
- [ ] Understanding of what routing learns

### Stretch Success
- [ ] ≥7% improvement (matching Apollo dual-encoder gain)
- [ ] Publishable finding about layer usage patterns
- [ ] Generalization to other prompts/styles

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Router collapses to uniform | Entropy regularization, early stopping |
| VRAM OOM during training | Gradient checkpointing, smaller batches |
| SigLIP doesn't capture improvement | Add temporal metrics, visual inspection |
| Validation fails | Clear pivot plan to alternatives |
| Time overrun | Strict phase gates, cut scope if needed |

---

## Immediate Next Steps

1. **Now**: Run `chunk_boundary_analysis.py --quick`
2. **Today**: Visual inspection of results, document findings
3. **Tomorrow**: Begin Phase 1 validation experiments
4. **Day 3**: Go/no-go decision on router development

---

## Appendix: Scripts to Create

| Script | Purpose | Priority |
|--------|---------|----------|
| `projection_comparison.py` | Compare W vs uniform | High |
| `cross_attention_analysis.py` | Extract DiT attention patterns | High |
| `random_routing_test.py` | Sanity check routing mechanism | High |
| `router_training.py` | REINFORCE training loop | Medium |
| `temporal_metrics.py` | Optical flow, frame interpolation | Medium |

---

## Appendix: Key Files Reference

| File | Purpose |
|------|---------|
| `experiments/ltx2/chunk_boundary_analysis.py` | Phase 0 experiment |
| `experiments/ltx2/memory_utils.py` | Memory-efficient generation |
| `experiments/ltx2/prompts.py` | Standardized prompts |
| `experiments/metrics/siglip_score.py` | Quality metric |
| `internal/log/log_2026-01-16.md` | Session documentation |
