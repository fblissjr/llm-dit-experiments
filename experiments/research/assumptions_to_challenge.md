# Assumptions to Challenge

Conventional wisdom from Z-Image, DiffSynth-Studio, and diffusers that may not be optimal.

---

## 1. Hidden State Extraction: Layer -2

### Status: READY TO TEST

The `--hidden-layer` parameter is now implemented (default: -2), making this experiment immediately runnable.

### The Assumption

Z-Image extracts embeddings from `hidden_states[-2]` (penultimate layer of Qwen3-4B).

### Source

This is a common pattern in text-to-image models, based on intuition that:
- Final layer (-1) is too task-specific (next-token prediction)
- Earlier layers are too low-level (syntactic)
- Penultimate layer balances semantics and generality

### Why Challenge It?

1. **No empirical validation**: The choice may be cargo-culted
2. **Model-specific**: Qwen3's architecture may differ from models where this was validated
3. **Task mismatch**: Image generation may benefit from different layers than NLP tasks

### Experiment

```bash
# Test layers -1 through -6
for layer in -1 -2 -3 -4 -5 -6; do
  uv run scripts/generate.py \
    --model-path /path/to/z-image \
    --hidden-layer $layer \
    --seed 42 \
    --output "layer${layer}.png" \
    "A beautiful sunset over mountains"
done
```

### Sub-Questions

- Q: Is layer -1 sharper but less compositional?
- Q: Is layer -3 better for complex scenes?
- Q: Does the optimal layer depend on prompt type?
- Q: Could we ensemble multiple layers?

---

## 2. CFG Scale: 0.0 is Optimal

### The Assumption

Z-Image uses `guidance_scale=0.0` because CFG is "baked in" via Decoupled-DMD training.

### Source

The Decoupled-DMD paper claims to distill CFG behavior into the model weights, eliminating the need for runtime CFG.

### Why Challenge It?

1. **Distillation is imperfect**: Some CFG benefit may be lost
2. **Different prompts**: Baked-in CFG may be calibrated for certain prompt types
3. **User preference**: Some users may want stronger/weaker guidance

### Experiment

```python
# Test small positive CFG values
for cfg in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
    # Even with baked-in CFG, runtime CFG might help or hurt
    image = generate(prompt, guidance_scale=cfg)
    quality = measure_quality(image)
    print(f"CFG {cfg}: {quality}")
```

### Sub-Questions

- Q: Does small CFG (0.5-1.5) improve text alignment?
- Q: Does CFG > 2.0 cause oversaturation?
- Q: Do certain prompt types benefit from runtime CFG?
- Q: Is there interaction between CFG and shift parameter?

---

## 3. Scheduler Shift: 3.0 for Turbo

### The Assumption

Z-Image-Turbo uses `shift=3.0` with 8-9 steps.

### Source

The shift value compresses the noise schedule:
```
sigma' = shift * sigma / (1 + (shift - 1) * sigma)
```

Higher shift = more aggressive compression = fewer steps needed.

### Why Challenge It?

1. **Quality trade-off**: Shift was tuned for speed, not maximum quality
2. **Content dependent**: Different content may need different schedules
3. **Step interaction**: Shift and steps are coupled; other combinations may work

### Experiment

```python
# Grid search shift x steps
results = []
for shift in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
    for steps in [4, 6, 8, 9, 12, 16, 25, 50]:
        image = generate(prompt, shift=shift, steps=steps)
        fid = compute_fid(image)
        time = measure_time()
        results.append({'shift': shift, 'steps': steps, 'fid': fid, 'time': time})

# Find pareto frontier
```

### Sub-Questions

- Q: What is the quality ceiling with more steps and lower shift?
- Q: Is shift=1.0 + 50 steps better than shift=3.0 + 9 steps?
- Q: Do portraits vs landscapes prefer different shifts?

---

## 4. Context Refiner: Required

### The Assumption

The 2-layer context refiner is essential for processing text embeddings.

### Source

Z-Image architecture includes it, so diffusers/DiffSynth preserve it.

### Why Challenge It?

1. **May be redundant**: Text encoder already processes text
2. **Adds latency**: 2 transformer layers add computation
3. **Unknown contribution**: Its actual benefit is unmeasured

### Experiment

```python
# Bypass context refiner
class ModifiedPipeline:
    def encode(self, text):
        embeddings = self.text_encoder.encode(text)
        projected = self.linear_projection(embeddings)  # 2560 -> 3840

        # Skip context refiner
        if self.skip_refiner:
            return projected
        else:
            return self.context_refiner(projected)

# Compare quality
image_with_refiner = generate(skip_refiner=False)
image_without_refiner = generate(skip_refiner=True)
```

### Sub-Questions

- Q: Does skipping the refiner produce coherent images?
- Q: What is the embedding drift (cosine sim) from the refiner?
- Q: Is the refiner just adding noise or meaningful transformation?
- Q: Can we make the refiner optional for speed?

---

## 5. Think Block: Optional/Cosmetic

### The Assumption

The `<think>...</think>` block is optional and doesn't affect generation semantically.

### Source

Official HF Space uses `enable_thinking=True` (confusingly named) which produces NO think block.

### Why Challenge It?

1. **Training data**: Model may have been trained WITH think blocks
2. **Embedding impact**: Think tokens may prime the embedding space
3. **Semantic content**: Thinking content may guide generation

### Experiment

```python
# Test think block variations
conditions = {
    'no_think': {'force_think_block': False, 'thinking_content': None},
    'empty_think': {'force_think_block': True, 'thinking_content': None},
    'quality_think': {'thinking_content': 'Sharp, detailed, high quality'},
    'style_think': {'thinking_content': 'Oil painting, textured, artistic'},
    'random_think': {'thinking_content': 'The weather is nice today'},
}

for name, config in conditions.items():
    image = generate(prompt, **config)
    quality = measure_quality(image)
    print(f"{name}: {quality}")
```

### Sub-Questions

- Q: Does empty think block differ from no think block?
- Q: Does quality-focused thinking improve image quality?
- Q: Does irrelevant thinking degrade results?
- Q: What token budget is optimal for thinking vs prompt?

---

## 6. Left Padding for Text

### The Assumption

Qwen3 tokenizer uses `padding_side="left"` for batch processing.

### Source

This is standard for causal LMs to keep attention patterns consistent.

### Why Challenge It?

1. **Image generation is different**: We're not doing next-token prediction
2. **Attention patterns**: Left padding may bias attention to later tokens
3. **Batch effects**: Different padding may affect batch consistency

### Experiment

```python
# Test padding sides
tokenizer.padding_side = "left"
embed_left = encode("A cat")

tokenizer.padding_side = "right"
embed_right = encode("A cat")

# Compare embeddings
cos_sim = cosine_similarity(embed_left, embed_right)
print(f"Embedding similarity: {cos_sim}")

# Compare generated images
image_left = generate_from_embeddings(embed_left)
image_right = generate_from_embeddings(embed_right)
```

### Sub-Questions

- Q: Do left and right padding produce identical embeddings for same-length inputs?
- Q: In batched generation, does padding side affect shorter prompts?
- Q: Is there a "center padding" that works better?

---

## 7. 1024 Token Limit: Hard Constraint

### Status: PARTIALLY RESOLVED (2025-12-09)

### The Assumption

The DiT cannot handle more than 1024 text tokens due to RoPE.

### Source

DiT config: `axes_lens=[1536, 512, 512]` for RoPE position encoding. DiffSynth-Studio uses 1024 conservatively.

### What We Discovered

Through systematic binary search testing (2025-12-09), the actual limit is **1504 tokens**:

**Test Results:**
- 1504 tokens: SUCCESS
- 1505 tokens: FAIL (CUDA kernel error)
- Config specifies 1536 = 48 × 32
- Actual limit is 1504 = 47 × 32

**Root Cause**: Likely an **off-by-one bug** in diffusers ZImageTransformer2DModel's RoPE frequency table indexing. The table is computed for indices 0-46 (47 values) instead of 0-47 (48 values).

**Impact**: 46.9% more capacity than the conservative 1024 limit (1024 → 1504).

### Remaining Challenges

1. **Fixing the off-by-one bug**: Could unlock full 1536 tokens (requires diffusers patch)
2. **RoPE interpolation**: LLM research shows RoPE can be extended beyond training range
3. **Position encoding modification**: May be possible without retraining
4. **Quality vs length trade-off**: Some degradation may be acceptable

### Updated Experiments

```python
# Test current limit thoroughly
for length in [1400, 1450, 1500, 1504, 1505, 1510]:
    try:
        image = generate_prompt_of_length(length)
        print(f"Length {length}: SUCCESS")
    except RuntimeError as e:
        print(f"Length {length}: FAIL - {e}")

# Test RoPE interpolation beyond 1504
class InterpolatedRoPE:
    def __init__(self, original_max=1504, extended_max=2048):
        self.scale = original_max / extended_max

    def get_positions(self, seq_len):
        # Scale positions to fit within original range
        positions = torch.arange(seq_len) * self.scale
        return positions

# Test extension with NTK scaling
for length in [1504, 1800, 2048, 3008]:
    prompt = generate_prompt_of_length(length)
    image = generate_with_ntk_scaling(prompt)
    quality = measure_quality(image)
    print(f"Length {length}: {quality}")
```

### Updated Sub-Questions

- Q: Can we patch diffusers to fix the off-by-one bug and unlock 1536? (PRIORITY)
- Q: At what extension ratio beyond 1504 does quality noticeably degrade?
- Q: Does NTK scaling work better than linear interpolation for extending beyond 1504?
- Q: Can we chunk long prompts and combine embeddings?
- Q: Is the theta=256 choice limiting extension potential?

---

## 8. Inference Steps: 8-9 Sufficient

### The Assumption

Z-Image-Turbo produces good results with 8-9 steps.

### Source

Turbo distillation targets fast inference.

### Why Challenge It?

1. **Quality ceiling**: More steps may improve quality
2. **Difficult prompts**: Complex scenes may need more steps
3. **Trade-off curve**: The marginal benefit of more steps is unknown

### Experiment

```python
# Test step counts
for steps in [4, 6, 8, 9, 12, 16, 25, 50, 100]:
    image = generate(prompt, steps=steps)
    fid = compute_fid(image)
    time = measure_time()
    print(f"Steps {steps}: FID={fid:.2f}, Time={time:.2f}s")
```

### Sub-Questions

- Q: Is there a quality ceiling regardless of steps?
- Q: Do certain prompt types benefit from more steps?
- Q: What is the time-quality pareto frontier?

---

## 9. No Timestep Modulation in Context Refiner

### The Assumption

The context refiner processes embeddings WITHOUT timestep information.

### Source

Architecture inspection shows no AdaLN or timestep injection in context refiner (unlike main DiT blocks).

### Why Challenge It?

1. **Design choice, not necessity**: This was a training decision
2. **Dynamic conditioning**: Timestep-aware refining might help
3. **Comparison**: Other models use timestep everywhere

### Experiment

```python
# Modify context refiner to accept timestep
class TimestepAwareRefiner(ContextRefiner):
    def __init__(self, ...):
        super().__init__(...)
        self.timestep_embed = nn.Embedding(1000, self.dim)

    def forward(self, x, timestep=None):
        if timestep is not None:
            t_embed = self.timestep_embed(timestep)
            x = x + t_embed.unsqueeze(1)  # Add timestep info
        return super().forward(x)

# Would need fine-tuning to be useful
```

### Sub-Questions

- Q: Why was timestep modulation omitted from the refiner?
- Q: Does the "stable conditioning signal" concept help or hurt?
- Q: Could timestep-aware refining improve early/late step behavior?

---

## 10. VAE: 16-Channel Wan-Family

### The Assumption

The VAE is fixed and optimal.

### Source

Z-Image uses a specific 16-channel VAE from the Wan family.

### Why Challenge It?

1. **VAE bottleneck**: VAE quality limits final image quality
2. **Alternative VAEs**: Other VAEs may decode better
3. **Fine-tuning**: VAE decoder could potentially be improved

### Experiment

```python
# This is more exploratory - VAE replacement is non-trivial
# But we can analyze VAE behavior

# Encode-decode roundtrip
original_image = load_image("test.png")
latent = vae.encode(original_image)
reconstructed = vae.decode(latent)

# Measure reconstruction quality
psnr = compute_psnr(original_image, reconstructed)
ssim = compute_ssim(original_image, reconstructed)
print(f"VAE reconstruction: PSNR={psnr:.2f}, SSIM={ssim:.4f}")
```

### Sub-Questions

- Q: What is the VAE reconstruction quality?
- Q: Are there artifacts specific to this VAE?
- Q: Could tiled decoding introduce seams? (Already tested: blending helps)

---

## Summary: Assumption Risk Assessment

| Assumption | Risk Level | Effort to Test | Potential Impact | Status (2025-12-09) |
|------------|------------|----------------|------------------|---------------------|
| Layer -2 extraction | Medium | Low | Medium | Ready to test |
| CFG = 0.0 | Low | Low | Low-Medium | Open |
| Shift = 3.0 | Medium | Low | Medium | Open |
| Context refiner required | Medium | Medium | High | Open |
| Think block optional | Medium | Low | Medium | Open |
| Left padding | Low | Low | Low | Open |
| ~~1024 token limit~~ | ~~High~~ | ~~High~~ | ~~High~~ | **SOLVED: Limit is 1504** |
| 8-9 steps sufficient | Low | Low | Low | Open |
| No timestep in refiner | Medium | High | Medium | Open |
| Fixed VAE | Low | High | Low | Open |
| RoPE theta=256 | Low | Medium | Medium | **EXPLAINED: Intentional design** |

**Major Updates:**
- **Token limit**: Discovered actual limit is 1504 (not 1024), due to off-by-one bug
- **RoPE theta**: Explained as intentional choice for local precision vs extrapolation

**Recommended first challenges:**
1. Shift parameter (Low effort, medium impact) - UNCHANGED
2. Think block testing (Low effort, medium impact) - UNCHANGED
3. Hidden layer extraction (Low effort, medium impact) - UNCHANGED
4. **Fixing off-by-one bug** (Medium effort, high impact) - NEW PRIORITY
5. Context refiner bypass (Medium effort, high impact) - UNCHANGED
