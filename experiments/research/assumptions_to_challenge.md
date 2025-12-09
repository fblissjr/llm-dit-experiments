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

### The Assumption

The DiT cannot handle more than 1024 text tokens due to RoPE.

### Source

DiT config: `axes_lens=[1024, 512, 512]` for RoPE position encoding.

### Why Challenge It?

1. **RoPE interpolation**: LLM research shows RoPE can be extended
2. **Position encoding**: May be modifiable without retraining
3. **Quality vs length**: Some degradation may be acceptable

### Experiment

```python
# Test RoPE interpolation
class InterpolatedRoPE:
    def __init__(self, original_max=1024, extended_max=2048):
        self.scale = original_max / extended_max

    def get_positions(self, seq_len):
        # Scale positions to fit within original range
        positions = torch.arange(seq_len) * self.scale
        return positions

# Generate with 1500+ token prompts
for length in [1024, 1280, 1536, 2048]:
    prompt = generate_prompt_of_length(length)
    image = generate_with_interpolation(prompt)
    quality = measure_quality(image)
    print(f"Length {length}: {quality}")
```

### Sub-Questions

- Q: At what extension ratio does quality noticeably degrade?
- Q: Does NTK scaling work better than linear interpolation?
- Q: Can we chunk long prompts and combine embeddings?

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

| Assumption | Risk Level | Effort to Test | Potential Impact |
|------------|------------|----------------|------------------|
| Layer -2 extraction | Medium | Low | Medium |
| CFG = 0.0 | Low | Low | Low-Medium |
| Shift = 3.0 | Medium | Low | Medium |
| Context refiner required | Medium | Medium | High |
| Think block optional | Medium | Low | Medium |
| Left padding | Low | Low | Low |
| 1024 token limit | High | High | High |
| 8-9 steps sufficient | Low | Low | Low |
| No timestep in refiner | Medium | High | Medium |
| Fixed VAE | Low | High | Low |

**Recommended first challenges:**
1. Shift parameter (Low effort, medium impact)
2. Think block testing (Low effort, medium impact)
3. Hidden layer extraction (Low effort, medium impact)
4. Context refiner bypass (Medium effort, high impact)
