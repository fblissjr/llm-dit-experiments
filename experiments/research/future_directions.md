# Future Directions

Ambitious research paths beyond basic ablations. These require more time and may produce publishable results.

---

## 1. Embedding Space Steering

### Concept

Analogous to activation steering in LLMs, manipulate embeddings to transfer styles or concepts without explicit prompting.

### Background

In LLM interpretability research, steering vectors can be computed as:
```
steering_vector = mean(concept_A_activations) - mean(concept_B_activations)
```

Applying this vector to new inputs transfers the concept.

### Approach for Z-Image

```python
def compute_style_vector(
    style_a_prompts: List[str],  # e.g., "photorealistic" prompts
    style_b_prompts: List[str],  # e.g., "oil painting" prompts
    encoder
) -> torch.Tensor:
    """Compute a style transfer vector."""
    # Encode all prompts
    embeds_a = [encoder.encode(p).embeddings for p in style_a_prompts]
    embeds_b = [encoder.encode(p).embeddings for p in style_b_prompts]

    # Mean-pool to single vectors
    mean_a = torch.stack([e.mean(dim=0) for e in embeds_a]).mean(dim=0)
    mean_b = torch.stack([e.mean(dim=0) for e in embeds_b]).mean(dim=0)

    # Style direction
    return mean_a - mean_b  # [hidden_dim]

def apply_steering(
    embedding: torch.Tensor,
    steering_vector: torch.Tensor,
    alpha: float = 1.0
) -> torch.Tensor:
    """Apply steering to an embedding."""
    # Add steering vector to each token
    return embedding + alpha * steering_vector.unsqueeze(0)
```

### Experiments

1. **Style transfer**: Can we make "a cat" look like an oil painting by steering?
2. **Concept injection**: Can we add "cyberpunk" to any prompt?
3. **Concept removal**: Can we subtract "blurry" to improve sharpness?
4. **Steering strength**: What alpha values work? Is there a saturation point?

### Sub-Questions

- Q1: Do steering vectors generalize across different base prompts?
- Q2: Can we identify "suppressor features" when steering fails?
- Q3: Is there a basis of style vectors that spans common styles?
- Q4: Do steering vectors compose? (style_a + style_b?)
- Q5: Can we steer in the context refiner output space instead?

### Failure Analysis

When steering doesn't work (e.g., model outputs "I'm the Eiffel Tower. No actually I'm not" equivalent):
- What happens in the embedding space?
- Are there regulatory/suppressor activations?
- Can we detect steering "overflow"?

---

## 2. RoPE Position Hacking

### Concept

The 1024 token limit comes from RoPE position encoding. Can we extend it?

### Background

RoPE interpolation techniques from LLM research:
- **Linear interpolation**: Scale position indices down
- **NTK-aware scaling**: Adjust the theta base frequency
- **YaRN**: Combined approach with attention scaling

### DiT-Specific Considerations

The Z-Image DiT uses multi-axis RoPE:
- Axis 0 (1024): Text sequence positions
- Axis 1 (512): Image height positions
- Axis 2 (512): Image width positions

Only Axis 0 limits text length.

### Approach

```python
class ExtendedRoPE:
    """RoPE with position interpolation for longer sequences."""

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 1024,
        extended_len: int = 2048,
        interpolation: str = "linear"  # linear, ntk, yarn
    ):
        self.original_max = max_seq_len
        self.extended_max = extended_len
        self.scale = extended_len / max_seq_len

        if interpolation == "linear":
            # Scale positions: pos' = pos / scale
            self.position_scale = 1.0 / self.scale
        elif interpolation == "ntk":
            # Adjust theta: theta' = theta * scale^(dim/(dim-2))
            self.theta_scale = self.scale ** (dim / (dim - 2))
        # ...

    def get_positions(self, seq_len: int) -> torch.Tensor:
        positions = torch.arange(seq_len)
        if self.interpolation == "linear":
            positions = positions * self.position_scale
        return positions
```

### Experiments

1. **Quality vs extension**: How does quality degrade at 1.5x, 2x, 4x extension?
2. **Interpolation methods**: Which method (linear, NTK, YaRN) works best?
3. **Training-free**: Does interpolation work without fine-tuning?
4. **Sliding window**: Alternative - encode chunks with overlap

### Sub-Questions

- Q1: What is the quality cliff? At what extension does it break?
- Q2: Does the context refiner's RoPE also need modification?
- Q3: Can we fine-tune just the position encodings?
- Q4: Is chunked encoding (overlapping windows) viable?

---

## 3. Context Refiner Fine-Tuning

### Concept

The context refiner is small (~50M params). It could be fine-tuned as a "style adapter" without touching the large DiT.

### Approach

```python
from llm_dit.models import ContextRefiner
from peft import LoraConfig, get_peft_model

# Load context refiner
refiner = ContextRefiner.from_pretrained("/path/to/z-image")

# Add LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
)
refiner = get_peft_model(refiner, lora_config)

# Training loop
optimizer = torch.optim.AdamW(refiner.parameters(), lr=1e-4)

for batch in dataloader:
    embeddings = encoder.encode(batch["prompts"])
    refined = refiner(embeddings)

    # Loss: match target style or reconstruction
    loss = compute_loss(refined, batch["targets"])
    loss.backward()
    optimizer.step()
```

### Training Objectives

1. **Style matching**: Fine-tune to produce style-consistent outputs
2. **Quality improvement**: Train on high-quality prompt-image pairs
3. **Concept learning**: Teach specific concepts (character, scene)
4. **Denoising**: Train to clean up noisy/bad embeddings

### Sub-Questions

- Q1: What loss function works for embedding-space training?
- Q2: How much data is needed for effective fine-tuning?
- Q3: Do LoRA adapters compose (multiple styles)?
- Q4: Can we train without generating images (embedding-only)?
- Q5: Does gradient checkpointing enable full fine-tuning on consumer GPUs?

---

## 4. Negative Prompting Without CFG

### Concept

Z-Image has CFG baked in (scale=0.0). Can we achieve negative prompting via embedding manipulation?

### Approach

```python
def negative_guidance(
    positive_embed: torch.Tensor,
    negative_embed: torch.Tensor,
    strength: float = 0.5
) -> torch.Tensor:
    """Apply negative guidance in embedding space."""
    # Option 1: Subtraction
    guided = positive_embed - strength * negative_embed

    # Option 2: Orthogonalization
    # Project out the negative direction
    neg_norm = negative_embed / negative_embed.norm()
    projection = (positive_embed * neg_norm).sum() * neg_norm
    guided = positive_embed - strength * projection

    # Option 3: Contrastive
    # Push away from negative in normalized space
    pos_norm = F.normalize(positive_embed, dim=-1)
    neg_norm = F.normalize(negative_embed, dim=-1)
    guided = pos_norm - strength * neg_norm
    guided = guided * positive_embed.norm()  # Restore magnitude

    return guided
```

### Experiments

1. **Artifact removal**: Can "blurry, distorted" negative improve quality?
2. **Style avoidance**: Can we avoid specific styles?
3. **Content removal**: Can we remove unwanted elements?
4. **Strength tuning**: What negative strength works without artifacts?

### Sub-Questions

- Q1: Does subtraction work or does it create artifacts?
- Q2: Is orthogonalization more stable than subtraction?
- Q3: How does embedding-space negative compare to real CFG?
- Q4: Can we negative-guide at the context refiner output?

---

## 5. Multi-Seed Ensemble

### Concept

Average predictions across seeds to improve consistency.

### Approach

```python
def ensemble_generate(
    prompt: str,
    seeds: List[int],
    ensemble_mode: str = "prediction_avg"
) -> Image:
    """Generate with multi-seed ensemble."""

    if ensemble_mode == "prediction_avg":
        # Average noise predictions at each step
        for t in timesteps:
            predictions = []
            for seed in seeds:
                torch.manual_seed(seed)
                pred = model(latent, t, embedding)
                predictions.append(pred)
            avg_pred = torch.stack(predictions).mean(dim=0)
            latent = scheduler.step(avg_pred, t, latent)

    elif ensemble_mode == "latent_avg":
        # Generate full latents, average before decode
        latents = []
        for seed in seeds:
            latent = generate_latent(prompt, seed=seed)
            latents.append(latent)
        avg_latent = torch.stack(latents).mean(dim=0)
        image = vae.decode(avg_latent)

    elif ensemble_mode == "image_avg":
        # Generate images, blend
        images = [generate(prompt, seed=seed) for seed in seeds]
        # Pixel-wise average
        avg_image = np.stack([np.array(img) for img in images]).mean(axis=0)

    return image
```

### Sub-Questions

- Q1: Does prediction averaging improve anatomical consistency?
- Q2: Which ensemble mode produces best results?
- Q3: How many seeds are needed for stable improvement?
- Q4: Does ensembling reduce diversity too much?
- Q5: Can we selectively ensemble only certain timesteps?

---

## 6. Token Attribution and Interpretability

### Concept

Build a complete attribution map: which tokens affect which pixels?

### Approach

```python
class TokenAttributor:
    """Compute token-to-pixel attribution maps."""

    def __init__(self, encoder, pipeline):
        self.encoder = encoder
        self.pipeline = pipeline

    def attribute(
        self,
        prompt: str,
        seed: int,
        method: str = "leave_one_out"
    ) -> np.ndarray:
        """
        Compute attribution: [num_tokens, height, width]
        """
        tokens = self.encoder.tokenizer.encode(prompt)
        baseline_image = self.generate(prompt, seed)

        if method == "leave_one_out":
            attributions = []
            for i in range(len(tokens)):
                # Remove token i
                modified = self.remove_token(prompt, i)
                modified_image = self.generate(modified, seed)

                # Per-pixel difference
                diff = np.abs(
                    np.array(baseline_image).astype(float) -
                    np.array(modified_image).astype(float)
                )
                attributions.append(diff.mean(axis=-1))  # Grayscale diff

            return np.stack(attributions)

        elif method == "gradient":
            # Gradient-based attribution (requires differentiable pipeline)
            pass

    def visualize(self, attribution: np.ndarray, tokens: List[str]):
        """Create interactive visualization."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(tokens), figsize=(4*len(tokens), 4))
        for i, (ax, token) in enumerate(zip(axes, tokens)):
            ax.imshow(attribution[i], cmap='hot')
            ax.set_title(f"Token: {token}")
            ax.axis('off')
        plt.tight_layout()
        return fig
```

### Research Questions

1. **Spatial patterns**: Do nouns map to object locations?
2. **Style influence**: Do adjectives affect global properties?
3. **Positional bias**: Do early tokens have more influence?
4. **Compositional understanding**: Does "red cat blue dog" work?
5. **Failure modes**: What makes some prompts fail?

---

## 7. Cross-Model Embedding Transfer

### Concept

Can embeddings from other LLMs (same hidden dim) work with Z-Image's DiT?

### Requirements

- Same hidden dimension (2560)
- Compatible tokenization (or re-alignment)

### Candidate Models

| Model | Hidden Dim | Compatible? |
|-------|------------|-------------|
| Qwen3-4B | 2560 | Yes (native) |
| Qwen2.5-3B | 2048 | No |
| Qwen3-1.8B | 2048 | No |
| Phi-3-mini | 3072 | No |

### Approach for Incompatible Dims

```python
class EmbeddingProjector(nn.Module):
    """Project embeddings from one dim to another."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

# Train projector to align embedding spaces
# Loss: MSE between projected and Qwen3-4B embeddings for same text
```

### Sub-Questions

- Q1: Can a linear projection align embedding spaces?
- Q2: Does a smaller model (Qwen3-1.8B + projection) produce viable results?
- Q3: Can we use distillation to train a smaller encoder?
- Q4: Do instruction-tuned variants (Qwen3-4B-Instruct) work?

---

## 8. Diffusion Trajectory Analysis

### Concept

Analyze what happens at each denoising step.

### Approach

```python
def analyze_trajectory(
    prompt: str,
    seed: int,
    pipeline
) -> List[dict]:
    """Capture intermediate states during generation."""
    trajectory = []

    latent = torch.randn(...)
    embedding = pipeline.encode(prompt)

    for i, t in enumerate(pipeline.scheduler.timesteps):
        # Capture pre-step state
        pre_state = {
            'step': i,
            'timestep': t.item(),
            'latent_norm': latent.norm().item(),
            'latent_mean': latent.mean().item(),
            'latent_std': latent.std().item(),
        }

        # Model prediction
        noise_pred = pipeline.transformer(latent, t, embedding)

        # Capture prediction stats
        pre_state['pred_norm'] = noise_pred.norm().item()
        pre_state['pred_mean'] = noise_pred.mean().item()

        # Step
        latent = pipeline.scheduler.step(noise_pred, t, latent).prev_sample

        # Decode intermediate (expensive but informative)
        if i % 3 == 0:  # Every 3rd step
            intermediate_image = pipeline.vae.decode(latent)
            pre_state['intermediate_image'] = intermediate_image

        trajectory.append(pre_state)

    return trajectory
```

### Research Questions

1. **Convergence**: When does the image "lock in"?
2. **Early vs late**: Which steps affect composition vs detail?
3. **Noise schedule**: Do different shifts change the trajectory shape?
4. **Failure prediction**: Can we detect bad generations early?

---

## Priority for Hobbyist Researcher

| Direction | Difficulty | Novelty | Time Investment |
|-----------|------------|---------|-----------------|
| Embedding steering | Medium | High | 1-2 weeks |
| Context refiner fine-tuning | High | High | 2-4 weeks |
| Token attribution | Low | Medium | 1 week |
| RoPE extension | High | High | 2-4 weeks |
| Negative prompting | Low | Medium | 3-5 days |
| Multi-seed ensemble | Low | Low | 2-3 days |
| Cross-model transfer | High | High | 3-4 weeks |
| Trajectory analysis | Low | Medium | 1 week |

**Recommended starting point**: Token attribution or embedding steering - both are tractable and produce interesting visualizations.
