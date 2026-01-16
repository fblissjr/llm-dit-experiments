# LTX-2 Prompting Fix Summary

Last updated: 2026-01-16

## TL;DR for Other Agents

**CRITICAL**: All experiment prompts have been standardized. You MUST import from `prompts.py` instead of defining inline prompts.

```python
# OLD (WRONG - out-of-distribution)
TEST_PROMPTS = ["A dog running in a park"]

# NEW (CORRECT - matches training data)
from experiments.ltx2.prompts import CATEGORY_PROMPTS, get_all_prompts
TEST_PROMPTS = list(CATEGORY_PROMPTS.values())
```

---

## What We Fixed

### The Problem

LTX-2's training data uses **detailed prose captions** (100-300 words) with:
- Multi-paragraph structure
- Scene headings (`EXT. PARK – GOLDEN HOUR`)
- Dialogue in "quotation marks"
- Explicit camera directions
- Style tags at the end

Our experiments were using **short 60-80 word prompts** without these elements. This created **out-of-distribution inputs** that likely invalidated prior experiment results.

### The Evidence

From the LTX-2 arXiv paper (2601.03233):
> "We developed a new video captioning system capable of describing both the visual and auditory tracks of a clip in exhaustive detail... comprehensive yet factual"

The official LTX-2 prompting guide (`ltx2_official_prompting_guide.md`) shows examples ranging from 70-237 words with screenplay-style formatting.

---

## What Changed

### New Centralized Module: `prompts.py`

All prompts are now in `experiments/ltx2/prompts.py`:

| Collection | Count | Word Range | Purpose |
|------------|-------|------------|---------|
| `OFFICIAL_PROMPTS` | 7 | 71-237 | Verbatim from official guide |
| `CATEGORY_PROMPTS` | 8 | 143-158 | Rewritten experiment prompts |
| `STRUCTURED_PROMPTS` | 5 | varies | Format ablation (JSON, XML, etc.) |
| `LEGACY_SHORT_PROMPTS` | 5 | 10-15 | Old prompts (DO NOT USE) |

### Helper Functions

```python
from experiments.ltx2.prompts import (
    get_all_prompts,      # Returns official + category (15 total)
    get_category_prompts, # Returns 8 category prompts
    get_official_prompts, # Returns 7 official prompts
    validate_prompts,     # Checks word counts and formatting
)

# Quick mode support
prompts = get_all_prompts(quick=True)  # 5 prompts for fast testing
```

### Updated Experiment Files

These files now import from `prompts.py`:
- `layer_profile_sweep.py`
- `layer_blend_sweep.py`
- `layer_extraction_comparison.py`
- `layer_ablation.py`
- `entropy_guided_encoding.py`
- `extract_embeddings.py`

### New Experiment

`prompt_format_ablation.py` - Tests whether structured formats (markdown, JSON, XML, YAML) work as well as prose. Hypothesis: prose outperforms structured formats.

---

## Example: Before vs After

### Before (Out-of-Distribution)
```
"A golden retriever runs through a sun-dappled park, its fur gleaming
in warm afternoon light."
```
**60 words, no dialogue, no scene heading, no camera direction**

### After (In-Distribution)
```
"EXT. SUBURBAN PARK – GOLDEN HOUR. A golden retriever bounds across
sun-dappled grass, its honey-colored fur catching the warm afternoon
light as it runs toward the camera. The dog's tongue lolls happily,
ears flopping with each energetic stride. The camera tracks alongside
at a low angle, capturing the athletic grace of the dog's movement.

A young woman's voice calls out from off-screen: "Max! Come here, boy!"
The retriever's ears perk up and it changes direction, kicking up small
tufts of grass as it pivots..."
```
**156 words, dialogue, scene heading, camera direction, physical cues**

---

## Action Required for Other Agents

1. **DO NOT** define inline prompts in experiment files
2. **DO** import from `experiments.ltx2.prompts`
3. **DO** use `validate_prompts()` to verify any new prompts
4. **DO** check that prompts are 100+ words with dialogue

### Quick Migration

```python
# Replace this:
TEST_PROMPTS = {
    "animal": "A dog running...",
    "scene": "A city street...",
}

# With this:
from experiments.ltx2.prompts import CATEGORY_PROMPTS
TEST_PROMPTS = CATEGORY_PROMPTS
```

---

## Why This Matters

Prior experiments may have been measuring artifacts of out-of-distribution inputs rather than genuine model behavior. Results from experiments using short prompts should be re-run with the new standardized prompts before drawing conclusions.
