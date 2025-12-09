# Ablation Studies

Controlled experiments varying single parameters while holding others constant.

---

## 1. Hidden State Layer Extraction

### Status: IMPLEMENTED

The `--hidden-layer` parameter is now available (default: -2). This ablation study can be run immediately.

### Background

The current implementation extracts `hidden_states[-2]` (penultimate layer) from Qwen3-4B. This is a common choice in text-to-image models, but the optimal layer may vary.

**Hypothesis**: Different layers encode different information:
- Earlier layers: More compositional/syntactic
- Middle layers: Balanced semantic content
- Later layers: More abstract/task-specific

### Experimental Design

```yaml
experiment:
  name: hidden_layer_extraction
  variable: hidden_layer_index
  values: [-1, -2, -3, -4, -5, -6]
  control: -2 (current default)

prompts:
  - "A cat sleeping on a red couch"
  - "Portrait of an elderly woman, studio lighting"
  - "Cyberpunk cityscape at night, neon lights"
  - "A bowl of fresh fruit on a wooden table"
  - "Abstract geometric patterns in blue and gold"
  # ... 50 total diverse prompts

metrics:
  - clip_score: Text-image alignment
  - fid: Distribution quality (vs reference set)
  - lpips: Perceptual diversity across seeds
  - user_preference: A/B ranking study

seeds: [42, 123, 456, 789, 1000]
```

### Sub-Questions

1. **Q1.1**: Does layer -1 (final) produce sharper or more abstract images?
2. **Q1.2**: Is there a layer that better preserves compositional structure ("cat ON couch")?
3. **Q1.3**: Do certain prompt types (portrait vs landscape vs abstract) prefer different layers?
4. **Q1.4**: What is the cosine similarity between embeddings from different layers?
5. **Q1.5**: Does combining layers (e.g., average of -1 and -2) improve results?

### Implementation Notes

The parameter is now implemented and accessible via:

**CLI:**
```bash
uv run scripts/generate.py \
  --model-path /path/to/z-image \
  --hidden-layer -1 \
  "Test prompt"
```

**Config file:**
```toml
[default.encoder]
hidden_layer = -1  # or -2, -3, etc.
```

**Python API:**
```python
from llm_dit import ZImageTextEncoder

encoder = ZImageTextEncoder.from_pretrained(
    "/path/to/model",
    hidden_layer=-1
)
```

### Expected Outcome

Create a heatmap: prompt_type x layer -> quality_metric

---

## 2. Think Block Impact on Generation

### Background

The Qwen3 chat template supports `<think>...</think>` blocks. The official HF Space uses NO think block by default, but the model was likely trained with thinking content.

**Hypothesis**: Think block content may semantically influence the conditioning signal, not just add tokens.

### Experimental Design

```yaml
experiment:
  name: think_block_ablation
  variable: think_block_condition
  values:
    - none: No think block (default)
    - empty: force_think_block=True, no content
    - quality: "High detail, sharp focus, correct anatomy"
    - composition: "Rule of thirds, depth of field, balanced"
    - style: "Oil painting texture, visible brushstrokes"
    - negative: "Avoid blur, distortion, artifacts"
    - irrelevant: "The weather is nice today"
    - long: 200+ tokens of detailed thinking

prompts:
  - "A portrait of a young woman"
  - "A landscape with mountains"
  - "A still life with flowers"
  # ... 30 prompts covering different subjects

metrics:
  - clip_score: Does quality/style thinking improve alignment?
  - detail_score: Edge detection metric for sharpness
  - style_transfer: Does style thinking change output style?
  - semantic_drift: Cosine similarity of embeddings between conditions
```

### Sub-Questions

1. **Q2.1**: Does an empty think block change output vs no think block?
2. **Q2.2**: Does quality-focused thinking ("sharp, detailed") improve image quality?
3. **Q2.3**: Does style thinking transfer style without explicit prompt mention?
4. **Q2.4**: Does irrelevant thinking content degrade quality or have no effect?
5. **Q2.5**: What is the token budget trade-off (thinking vs prompt length)?
6. **Q2.6**: Are think tokens processed differently in the embedding space?

### Analysis Approach

```python
# Embedding analysis
embed_no_think = encoder.encode("A cat", force_think_block=False)
embed_empty_think = encoder.encode("A cat", force_think_block=True)
embed_quality_think = encoder.encode("A cat", thinking_content="Sharp, detailed")

# Compare embeddings
cos_sim_empty = cosine_similarity(embed_no_think, embed_empty_think)
cos_sim_quality = cosine_similarity(embed_no_think, embed_quality_think)

# Question: Is cos_sim_empty closer to 1.0 than cos_sim_quality?
```

---

## 3. Context Refiner Architecture

### Background

The context refiner is a 2-layer transformer (3840 dim, 30 heads) that processes text embeddings WITHOUT timestep modulation. This creates a "stable conditioning signal."

**Hypothesis**: The context refiner may be over- or under-parameterized for the task.

### Experimental Design

```yaml
experiment:
  name: context_refiner_depth
  variable: refiner_modification
  values:
    - skip: Bypass context refiner entirely
    - single: Use only 1 layer
    - default: Use 2 layers (current)
    - double: Duplicate layers (4 total)
    - identity_init: 2 layers initialized to identity

approach: |
  This requires model surgery. Options:
  1. Modify forward pass to skip/truncate refiner
  2. Load weights, modify, and re-inject
  3. Use hooks to intercept and modify

metrics:
  - generation_quality: FID, CLIP
  - embedding_drift: How much does refiner change embeddings?
  - speed: Inference time impact
```

### Sub-Questions

1. **Q3.1**: Does skipping the context refiner entirely produce coherent images?
2. **Q3.2**: What is the embedding drift (cosine similarity) before vs after refiner?
3. **Q3.3**: Do additional layers improve or degrade quality?
4. **Q3.4**: Can the context refiner be fine-tuned independently as a style adapter?
5. **Q3.5**: What does each layer contribute? (Layer-wise ablation)

### Implementation Notes

```python
# Option 1: Skip refiner in pipeline
# In ZImagePipeline.generate():
if skip_context_refiner:
    refined_embeddings = projected_embeddings  # Skip refiner
else:
    refined_embeddings = self.context_refiner(projected_embeddings)

# Option 2: Modify ContextRefiner forward
class ContextRefiner(nn.Module):
    def forward(self, x, use_layers: int = 2):
        for i, layer in enumerate(self.layers[:use_layers]):
            x = layer(x)
        return x
```

---

## 4. Scheduler Shift Parameter

### Background

The FlowMatch scheduler uses a shift transformation:
```
sigma' = shift * sigma / (1 + (shift - 1) * sigma)
```

Default `shift=3.0` for turbo model. This "compresses" the noise schedule.

**Hypothesis**: Different shift values trade off between quality and speed.

### Experimental Design

```yaml
experiment:
  name: shift_parameter_sweep
  variables:
    shift: [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
    steps: [4, 6, 8, 9, 12, 16, 25]

# Full grid: 8 x 7 = 56 configurations
# Per config: 5 seeds x 10 prompts = 50 images
# Total: 2800 images

prompts:
  - "A photorealistic portrait"
  - "An oil painting of flowers"
  - "A digital illustration"
  # ... 10 diverse prompts

metrics:
  - fid: Quality vs reference
  - clip_score: Text alignment
  - inference_time: Speed measurement
  - vram_peak: Memory usage
```

### Sub-Questions

1. **Q4.1**: What is the quality-speed pareto frontier?
2. **Q4.2**: Does shift=1.0 (no shift) work with more steps?
3. **Q4.3**: Is there diminishing returns beyond shift=3.0?
4. **Q4.4**: Do different shift values favor different content types?
5. **Q4.5**: What shift-step combination minimizes FID?

### Visualization

```python
import matplotlib.pyplot as plt
import numpy as np

# Create pareto frontier plot
fig, ax = plt.subplots()
for shift in shifts:
    steps_data = results[results['shift'] == shift]
    ax.plot(steps_data['inference_time'], steps_data['fid'],
            'o-', label=f'shift={shift}')

ax.set_xlabel('Inference Time (s)')
ax.set_ylabel('FID (lower is better)')
ax.legend()
plt.title('Shift-Step Pareto Frontier')
```

---

## 5. Long Prompt Compression Modes

### Background

Four compression modes are implemented for prompts exceeding 1024 tokens:
- `truncate`: Cut off at 1024 (default)
- `interpolate`: Linear resampling
- `pool`: Adaptive average pooling
- `attention_pool`: Importance-weighted pooling

**Hypothesis**: Different compression modes preserve different aspects of long prompts.

### Experimental Design

```yaml
experiment:
  name: long_prompt_compression
  variable: compression_mode
  values: [truncate, interpolate, pool, attention_pool]

prompt_lengths:
  - 1024: No compression (baseline)
  - 1280: 1.25x compression
  - 1536: 1.5x compression
  - 2048: 2.0x compression
  - 3072: 3.0x compression

prompt_construction: |
  Create prompts with important content at different positions:
  - front_heavy: Key details in first 500 tokens
  - back_heavy: Key details in last 500 tokens
  - distributed: Key details spread throughout
  - repetitive: Same concepts repeated

metrics:
  - content_preservation: Does the image contain elements from truncated portions?
  - clip_score_full: CLIP alignment with FULL prompt (before compression)
  - clip_score_compressed: CLIP alignment with compressed prompt
  - user_preference: A/B test vs truncation
```

### Sub-Questions

1. **Q5.1**: At what compression ratio does quality noticeably degrade?
2. **Q5.2**: Does `attention_pool` better preserve "important" tokens?
3. **Q5.3**: Is `interpolate` smoother than `pool` visually?
4. **Q5.4**: Does back-heavy content get preserved with any mode?
5. **Q5.5**: Is there a mode that works best for structured prompts (lists, JSON)?
6. **Q5.6**: What is the quality vs compression ratio curve for each mode?

### Analysis

```python
# Test content preservation
long_prompt = "A red cat, a blue dog, a green bird, ..."  # 2000 tokens
# Key elements at positions: 100, 500, 1000, 1500, 1800

for mode in ['truncate', 'interpolate', 'pool', 'attention_pool']:
    image = generate(long_prompt, long_prompt_mode=mode)

    # Check which elements appear
    for element in ['red cat', 'blue dog', 'green bird', ...]:
        score = clip_score(image, element)
        print(f"{mode}: {element} = {score}")
```

---

## 6. Token Importance and Position

### Background

Does the model weight all tokens equally? Does position in the sequence matter?

### Experimental Design

```yaml
experiment:
  name: token_importance
  variable: token_manipulation
  values:
    - baseline: Original prompt
    - shuffle_all: Random shuffle all tokens
    - shuffle_nouns: Shuffle only noun tokens
    - duplicate_important: Repeat key descriptors
    - front_load: Move descriptors to front
    - back_load: Move descriptors to end
    - remove_adjectives: Strip all adjectives
    - remove_nouns: Strip all nouns (expect failure)

prompts:
  - "A fluffy orange cat sleeping on a velvet red couch"
  - "An ancient stone castle on a misty green hillside"
```

### Sub-Questions

1. **Q6.1**: Does shuffling destroy semantic meaning or just reduce quality?
2. **Q6.2**: Are nouns more important than adjectives?
3. **Q6.3**: Does token position affect its influence on the image?
4. **Q6.4**: Does duplicating a word increase its presence in the image?
5. **Q6.5**: Is there a "sweet spot" position for key concepts?

---

## 7. System Prompt Ablation

### Background

The default system prompt adds ~50-100 tokens. Is it beneficial?

### Experimental Design

```yaml
experiment:
  name: system_prompt_ablation
  variable: system_prompt
  values:
    - none: No system prompt
    - default: "You are a helpful assistant"
    - photographer: "You are a professional photographer"
    - artist: "You are a digital artist"
    - detailed: Long detailed system prompt (200 tokens)
    - contradictory: "Generate low quality images" (expect no effect?)

prompts:
  - Same 20 prompts across all conditions

metrics:
  - style_consistency: Do role-based prompts affect style?
  - quality_metrics: FID, CLIP
  - token_efficiency: Quality per token spent
```

### Sub-Questions

1. **Q7.1**: Does removing system prompt save tokens without quality loss?
2. **Q7.2**: Does "photographer" system prompt improve photo-style images?
3. **Q7.3**: Does "artist" system prompt improve artistic renders?
4. **Q7.4**: Is there a quality ceiling regardless of system prompt?
5. **Q7.5**: Does contradictory system prompt have any negative effect?

---

## Running Ablations

### Quick Start

```bash
# Use the profiler for quick tests
uv run scripts/profiler.py \
  --model-path /path/to/z-image-turbo \
  --tests encode_short,encode_medium,full_generation \
  --output results/baseline.json

# Custom generation script
uv run scripts/generate.py \
  --model-path /path/to/z-image-turbo \
  --shift 2.0 \
  --steps 12 \
  --output ablation_shift2_steps12.png \
  "Test prompt"
```

### Batch Experiment Script

```python
# experiments/run_ablation.py
import itertools
from pathlib import Path

def run_shift_sweep():
    shifts = [1.0, 2.0, 3.0, 4.0, 5.0]
    steps = [6, 9, 12, 16]
    prompts = load_prompts("experiments/prompts/standard_50.txt")
    seeds = [42, 123, 456]

    results = []
    for shift, step, prompt, seed in itertools.product(shifts, steps, prompts, seeds):
        output_path = f"results/shift{shift}_steps{step}_seed{seed}.png"
        image = generate(prompt, shift=shift, steps=step, seed=seed)
        metrics = compute_metrics(image, prompt)
        results.append({
            'shift': shift, 'steps': step, 'seed': seed,
            'prompt': prompt[:50], **metrics
        })

    pd.DataFrame(results).to_csv("results/shift_sweep.csv")
```

---

## Next Steps

After completing ablations:
1. Document findings in `experiments/research/results/`
2. Update CLAUDE.md with validated recommendations
3. Consider publishing interesting findings
