# experiments agent context

*last updated: 2026-02-19*

---

## research status legend

- `[VALIDATED]` - Confirmed through experiments or architecture analysis
- `[OPEN]` - Hypothesis needs testing or re-testing
- `[NEEDS_CHECK]` - Previous results may have bugs
- `[DEAD_END]` - Tested, doesn't work

---

## quick navigation

| Area | Entry Point | Status |
|------|-------------|--------|
| **LTX-2** | [ltx2/CLAUDE.md](ltx2/CLAUDE.md) | Active |
| **Research Findings** | [ltx2/docs/findings/](ltx2/docs/findings/) | Reference |
| **Metrics** | [metrics/](metrics/) | Active |
| **Viewer** | [viewer/](viewer/) | Active |

---

## ltx-2 research status

### validated findings

| Finding | Status | Notes |
|---------|--------|-------|
| 49 layers extracted from Gemma-3 | `[VALIDATED]` | Architecture confirmed |
| 8-frame VAE temporal unit | `[VALIDATED]` | Architecture (valid frames: 9, 17, 25...) |
| Bidirectional connector (not causal) | `[VALIDATED]` | 2 transformer layers + 128 thinking tokens |
| Prompt ordering doesn't matter for semantics | `[VALIDATED]` | Bidirectional attention handles all orders |
| SigLIP appropriate for spatial evaluation | `[VALIDATED]` | Apollo paper validates this |
| Scaling consistency enables small-scale experiments | `[VALIDATED]` | Apollo paper (R^2 > 0.8) |
| Text encoder matches reference output | `[VALIDATED]` | mean=0.003, std=1.0, per-dim range [-8.5, 12.2] (Jan 20) |
| FP8 transformer forward pass works | `[VALIDATED]` | mean=0.06, std=1.05, 0.09% zeros (Jan 20) |
| -9.5 GELU bias is architecturally intentional | `[VALIDATED]` | DiT produces valid output despite 98% sparsity (Jan 20) |
| Context dimension must be 3840 | `[VALIDATED]` | Gemma hidden size, caption_projection maps to 4096 (Jan 20) |

### open hypotheses

| Hypothesis | Status | Notes |
|------------|--------|-------|
| Per-token layer routing improves quality | `[OPEN]` | **READY** - layer masking fixed Jan 17; run `train_router.py` |
| Chunk-aligned prompting improves transitions | `[OPEN]` | Not yet tested |
| Activation steering can improve quality | `[OPEN]` | Zero-training approach |
| Thinking tokens capture global context | `[OPEN]` | Connector internals unexplored |
| Temporal tokens benefit from different layers | `[OPEN]` | Derived from Apollo findings |

### needs verification

| Finding | Status | Notes |
|---------|--------|-------|
| Late layers (43-47) contribute ~25% | `[VALIDATED]` | **REFUTED** - Layer 48 contributes ~0% when isolated (Jan 17) |
| Layer 47 anomaly (near-zero norm) | `[NEEDS_CHECK]` | Needs re-verification with full 49-layer run |
| Projection weights uniform | `[NEEDS_CHECK]` | Depends on correct layer extraction |

### new findings (jan 20)

| Finding | Status | Notes |
|---------|--------|-------|
| Text encoder output validated | `[VALIDATED]` | Matches reference exactly (mean=0.003, std=1.0) |
| FP8 transformer validated | `[VALIDATED]` | Forward pass produces healthy output (mean=0.06, std=1.05) |
| "Blurry blob" is downstream | `[VALIDATED]` | Core components work, issue in pipeline integration |
| ModelLedger architecture documented | `[VALIDATED]` | Full analysis in `internal/analysis/model_ledger_*.md` |

### new findings (jan 17)

| Finding | Status | Notes |
|---------|--------|-------|
| Layer 0 (embedding) contributes 51% isolated | `[VALIDATED]` | Highest contribution when only layer active |
| Layer 24 contributes 49% isolated | `[VALIDATED]` | Mid-layers carry semantic content |
| Layer 48 contributes ~0% isolated | `[VALIDATED]` | Final layer alone contributes nothing |
| L2 normalization destroys masking info | `[VALIDATED]` | Must use min-max normalization `8*(x-mean)/(max-min)` |

## output organization

*Updated: 2026-01-18*

### directory structure

All experiment outputs go to `experiments/results/{pipeline}/{experiment}_{timestamp}/`:

```
experiments/results/
├── ltx2/                                  # LTX-2 video experiments
│   ├── layer_ablation_20260115_191023/
│   ├── activation_steering_20260115_214600/
│   └── ltx2_layer_blend_20260116_105604/
├── z_image/                               # Z-Image experiments
│   └── hidden_layer_blend_20260101_172411/
├── wan/                                   # Wan video experiments (future)
└── archive/                               # Superseded/analysis files
```

### naming convention

- **Directories**: `{experiment_name}_{YYYYMMDD_HHMMSS}`
- **Files**: `{config}_{prompt}.{ext}` or `{variant}_seed{N}.{ext}`
- **Metadata**: `metadata/{filename}.json` alongside outputs

### discovery

Experiments are auto-discovered by `compare/discovery.py`:

```python
from experiments.compare.discovery import discover_experiments
experiments = discover_experiments()  # Returns all experiments
```

### creating new experiments

Inherit from the appropriate base class to get correct output paths:

```python
from experiments.ltx2.base import LTX2ExperimentBase

class MyExperiment(LTX2ExperimentBase):
    def __init__(self):
        super().__init__("my_experiment")
        # Output: experiments/results/ltx2/my_experiment_{timestamp}/
```

---

## codebase directory structure

```
experiments/
├── CLAUDE.md              # This file (navigation hub)
├── base.py                # Shared experiment infrastructure
├── ltx2/                  # LTX-2 experiments (ACTIVE)
│   ├── CLAUDE.md          # LTX-2 specific context
│   ├── base.py            # LTX2ExperimentBase (inherits from experiments/base.py)
│   ├── prompts.py         # CENTRALIZED PROMPTS (MUST USE)
│   ├── docs/              # Documentation
│   │   ├── findings/      # Consolidated research findings
│   │   └── reports/       # Session reports
│   └── *.py               # Experiment scripts
├── qwen3_vl/              # Qwen3-VL experiments (DEAD-END)
│   └── ...
├── metrics/               # Scoring modules
│   ├── siglip_score.py    # SigLIP2 text-image alignment
│   └── image_reward.py    # Human preference alignment
├── compare/               # Comparison infrastructure
│   ├── discovery.py       # Auto-discovery (supports pipeline structure)
│   └── models.py          # ExperimentRun, ExperimentImage dataclasses
├── viewer/                # Interactive web viewer
├── results/               # Experiment outputs (pipeline-organized)
│   ├── ltx2/              # LTX-2 experiments
│   ├── z_image/           # Z-Image experiments
│   └── archive/           # Superseded content
└── archive/               # Superseded/dated content
    ├── dated_reports/     # Historical dated reports
    ├── drafts/            # Draft versions
    └── superseded/        # Superseded documents
```

---

## key documents

### ltx-2 research

| Document | Purpose |
|----------|---------|
| [ltx2/docs/findings/apollo_analysis.md](ltx2/docs/findings/apollo_analysis.md) | Apollo paper insights and transfer analysis |
| [ltx2/docs/findings/research_synthesis.md](ltx2/docs/findings/research_synthesis.md) | Consolidated research status |
| [ltx2/docs/text_conditioning_architecture.md](ltx2/docs/text_conditioning_architecture.md) | Full architecture reference |
| [ltx2/prompting_guide.md](ltx2/prompting_guide.md) | How to write prompts for LTX-2 |

### archived content

Historical dated reports: [archive/dated_reports/](archive/dated_reports/)

---

## research priorities

### tier 1: validation (do first)

| Task | Status | Purpose |
|------|--------|---------|
| ~~Re-verify layer contribution patterns~~ | `[VALIDATED]` | Validated with corrected extraction (Jan 17) |
| Layer 47 anomaly verification | `[OPEN]` | Confirm near-zero norm (needs full 49-layer run) |
| GPU numerical equivalence test | `[OPEN]` | Compare pure PyTorch vs diffusers output |

### tier 2: high-value experiments (NOW UNBLOCKED)

| Task | Status | Expected Impact |
|------|--------|-----------------|
| Train per-token layer router | `[OPEN]` Ready | 3-10% on complex prompts |
| Activation steering | `[OPEN]` | Zero-training quality boost |
| Chunk-aligned prompting | `[OPEN]` | Improved transitions |

---

## critical reminders

### prompt standardization

**All experiment prompts must be imported from centralized module:**

```python
from experiments.ltx2.prompts import CATEGORY_PROMPTS, get_all_prompts

TEST_PROMPTS = CATEGORY_PROMPTS  # 8 prompts, 100+ words each
```

**Why?** LTX-2's training data uses 100-300 word prose. Short prompts are out-of-distribution.

### evaluation metrics

- **SigLIP**: Primary spatial quality metric (Apollo-validated)
- **Temporal metrics**: Needed separately (SigLIP is spatial-focused)
- **Human preference**: For final validation

---

## related documentation

- **Internal hub**: `internal/hub.md` - Project-wide navigation
- **LTX-2 paper**: arXiv:2601.03233
- **Apollo paper**: arXiv:2412.10360
