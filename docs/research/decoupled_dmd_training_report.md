# Decoupled DMD: Z-Image Training Methodology Report

## Executive Summary

The "Decoupled DMD" paper provides critical insights into how the Z-Image 8-step turbo model achieves high-quality generation with minimal inference steps. The key revelation is that the success of Distribution Matching Distillation (DMD) is not primarily due to distribution matching as conventionally understood, but rather due to a previously overlooked mechanism called **CFG Augmentation (CA)** that "bakes" Classifier-Free Guidance into the model weights during training.

**Paper Citation**: Liu et al., "Decoupled DMD: CFG Augmentation as the Spear, Distribution Matching as the Shield" - Tongyi Lab, Alibaba Group / CUHK

---

## 1. Training Methodology Deep Dive

### 1.1 How DMD Works: The Conventional View vs. Reality

**Conventional Understanding (Incorrect)**

The traditional view held that DMD succeeds by minimizing the Integral Kullback-Leibler (IKL) divergence between the student's output distribution and the teacher's:

```
L_IKL(p_real, p_fake) = integral from 0 to 1 of KL(p_real,tau || p_fake,tau) d_tau
```

**The Paper's Key Discovery**

Through rigorous decomposition, the authors reveal that the practical DMD objective decomposes into two distinct components:

```
gradient_theta L_DMD = E[-(Delta_real-fake + Delta_cfg_real) * partial_G_theta(z_t)/partial_theta]

Where:
- Delta_real-fake = s_cond_real(x_tau) - s_cond_fake(x_tau)  [Distribution Matching]
- Delta_cfg_real = (alpha - 1)(s_cond_real(x_tau) - s_uncond_real(x_tau))  [CFG Augmentation]
```

### 1.2 The Engine and Regularizer Decomposition

**CFG Augmentation (CA) - "The Engine"**

- Directly applies scaled CFG signal as a gradient to the student's output
- Responsible for converting multi-step model into few-step generator
- Acts at specific noise levels to enhance corresponding frequency content:
  - Noisy timesteps (tau near 0): Enhances low-frequency (composition, color blocks)
  - Clean timesteps (tau near 1): Enhances high-frequency (textures, edges)

**Distribution Matching (DM) - "The Regularizer"**

- Matches the theoretical derivation of IKL divergence
- Primary function: Prevent training collapse and artifact accumulation
- Acts as a "corrective mechanism" by canceling out artifacts learned by the fake model
- Without DM, CA alone leads to:
  - Monotonic increase in output variance
  - Over-saturation
  - High-frequency checkerboard artifacts
  - Eventually, complete training collapse

### 1.3 Key Mathematical Formulations

**Equation 1 - IKL Divergence:**
```
L_IKL(p_real, p_fake) = integral_0^1 KL(p_real,tau || p_fake,tau) d_tau
```

**Equation 2 - Theoretical DMD Gradient:**
```
gradient_theta L_DMD-theory = E[-( s_cond_real(x_tau) - s_cond_fake(x_tau) ) * partial_G_theta(z_t)/partial_theta]
```

**Equation 3 - Practical DMD Gradient (with CFG):**
```
gradient_theta L_DMD = E[-( s_cfg_real(x_tau) - s_cond_fake(x_tau) ) * partial_G_theta(z_t)/partial_theta]
```

**Equation 4 - CFG Definition:**
```
s_cfg_real(x_tau) = s_uncond_real(x_tau) + alpha * (s_cond_real(x_tau) - s_uncond_real(x_tau))
```

**Equation 6 - The Core Decomposition:**
```
gradient_theta L_DMD = E[-((s_cond_real - s_cond_fake) + (alpha-1)(s_cond_real - s_uncond_real)) * dG/d_theta]
                          |---- DM (Regularizer) ----|   |-------- CA (Engine) --------------|
```

**Equation 8 - Decoupled DMD Gradient:**
```
gradient_theta L_d-DMD = E[-( (s_cond_real(x_tau_DM) - s_cond_fake(x_tau_DM)) +
                              (alpha-1)(s_cond_real(x_tau_CA) - s_uncond_real(x_tau_CA)) ) *
                            partial_G_theta(z_t)/partial_theta]
```

### 1.4 The Re-noising Schedule Innovation

The paper's practical contribution is the **Decoupled-Hybrid schedule**:

| Configuration | CA Schedule | DM Schedule | Result |
|--------------|-------------|-------------|--------|
| Original DMD | tau in [0,1] | tau in [0,1] (shared) | Baseline |
| Decoupled-Full | tau in [0,1] | tau in [0,1] (independent) | Negligible improvement |
| Decoupled-Constrained | tau > t | tau > t | Good detail, oversaturation |
| **Decoupled-Hybrid** | **tau > t** | **tau in [0,1]** | **Best results** |

**Why Decoupled-Hybrid Works:**
- CA constrained to tau > t: Acts as "focused engine" on unresolved aspects only
- DM spanning full range: Acts as "comprehensive regularizer" to correct global issues

---

## 2. CFG Baking Mechanism

### 2.1 How CFG Gets "Baked In"

The CA term `(alpha - 1)(s_cond_real - s_uncond_real)` directly applies the CFG pattern to the student generator's predictions during training. This effectively:

1. **Internalizes the guidance signal**: The model learns to produce outputs that already incorporate CFG-like enhancement
2. **Collapses the decision tree**: What was previously a probabilistic sampling process becomes a deterministic prediction
3. **Eliminates the need for runtime guidance**: Since the model has "learned" the CFG pattern, no CFG is needed at inference

### 2.2 Why CFG=0 Works at Inference

The paper draws an analogy to LLMs: Just as an LLM cannot predict multiple tokens simultaneously because each depends on probabilistic sampling of the previous token, diffusion models require multiple steps because CFG introduces an "external intervention" the model cannot predict.

By training with the CA term, the model learns: "Given this input, the external CFG process will always produce this specific shift." The model internalizes this pattern, allowing single-pass prediction.

**Key insight**: CFG represents a "deterministic decision pattern" that gets baked into the weights, transforming an uncontrollable external force into predictable behavior.

### 2.3 Training CFG Scale vs. Inference Behavior

**During DMD training:**
- CFG scale alpha is typically set high (alpha > 1, often 7-8 for SDXL-like models)
- This high CFG is applied only to the "real" model's predictions
- The "fake" model never sees CFG

**At inference:**
- CFG scale should be 0 (or effectively 1, meaning no guidance)
- The guidance effect is already embedded in the model weights
- Using CFG > 0 at inference would "double-dip" and cause over-saturation

---

## 3. Practical Training Details

### 3.1 Number of Steps

For Z-Image specifically: **8 steps (8 NFEs - Number of Function Evaluations)**

The paper validates on:
- 1-step SDXL
- 4-step SDXL
- 4-step Lumina-Image-2.0

### 3.2 Training Data Requirements

The DMD process does not require the original training data. It uses:
- Text prompts (sampled from datasets like COCO-10k for evaluation)
- The pre-trained teacher model to generate "real" scores
- The fake model trained on student outputs

### 3.3 Loss Functions

**Primary Loss (Proxy Loss):**
```
L_proxy = ||G_theta(z_t) - stop_grad(G_theta(z_t) + lambda * Delta_total)||^2
```

**Fake Model Loss:**
```
L_denoise = ||s_fake(x'_tau', tau') - x'_gen||^2
```

### 3.4 The Role of the "Fake Model"

The fake model serves several critical purposes:

1. **Tracks student distribution**: Continuously trained on student generator outputs
2. **Enables DM gradient computation**: Provides s_cond_fake for the distribution matching term
3. **Acts as artifact detector**: Learns to replicate student's failure modes, which then get "subtracted out" by the DM gradient
4. **Initialized from teacher**: Both student generator and fake model start from pre-trained teacher weights

**Training procedure (from Algorithm 1):**
1. Initialize G_theta and s_fake from s_real (teacher)
2. For each iteration:
   - Generate image with student: x_gen = G_theta(z_t)
   - Re-noise to get x_tau
   - Compute CA and DM gradients
   - Update generator with proxy loss
   - Update fake model with denoising loss (can run multiple times per generator update - TTUR)

---

## 4. LoRA Training Implications

### 4.1 The CFG Baking Problem for LoRAs

This is a critical consideration that the paper does not directly address, but we can infer:

**The Challenge:**
The Z-Image model has CFG "baked in" at a specific scale during distillation. When training a LoRA:
- The LoRA sees a model that already has CFG internalized
- Training with additional CFG would compound the effect
- Training without CFG means the LoRA learns on top of the baked-in guidance

### 4.2 Recommendations for LoRA Training

**Option A: Train with CFG=0 (Recommended)**
- Rationale: The base model already has CFG baked in
- The LoRA should learn style/concept modifications without additional guidance
- At inference, keep CFG=0 as intended

**Option B: Train with low CFG (0.5-1.5)**
- Only if you observe the LoRA not following prompts well
- May cause slight oversaturation
- Monitor for "greasy" or oversaturated outputs

**What NOT to do:**
- Do NOT train with high CFG (3+) as this will almost certainly cause severe artifacts
- Do NOT expect to "dial in" CFG at inference - the model fundamentally expects CFG=0

### 4.3 Distillation Effects on Fine-tuning

The distillation process creates specific characteristics:
- **Fixed aesthetic bias**: The baked CFG enforces a particular "look"
- **Reduced flexibility**: Unlike non-distilled models, you cannot trade off prompt adherence vs. diversity via CFG
- **Potentially brittle**: LoRA training may disrupt the carefully balanced CA/DM equilibrium

**Practical advice for hobbyists:**
1. Start with very low LoRA ranks (4-8) to minimize disruption
2. Use lower learning rates than typical SDXL LoRA training
3. Train on data that matches the model's expected prompt structure (Qwen3 chat format)
4. Monitor for variance explosion (over-saturation) during training

---

## 5. Prompt Alignment Insights

### 5.1 Training Prompt Structure

Based on the paper and Z-Image's use of Qwen3-4B:

**Expected prompt format during distillation:**
```
<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
<think>
{thinking_content}
</think>

{assistant_content}
```

The model was likely trained on prompts that:
- Use the complete Qwen3-4B chat template
- Include system prompts for conditioning style/quality
- May include thinking blocks for complex compositions
- Were curated/filtered for quality (aesthetic filtering common in T2I training)

### 5.2 Thinking/Reasoning Structure Impact

The `<think>...</think>` structure in Qwen3-4B serves a specific purpose:
- During text encoder training, thinking content helps the model "reason about" the image
- This reasoning is embedded into the text conditioning
- The distilled model has learned to expect and use this structure

**Implications:**
- Including thinking content may improve compositional understanding
- Empty thinking blocks still activate the associated attention patterns
- The `assistant_content` after thinking acts as a "refined" version of the request

### 5.3 Inference Prompt Recommendations

Based on the training methodology:

1. **Use the full chat template** - The model expects this structure
2. **Include system prompts** - These were part of training conditioning
3. **Use thinking blocks for complex scenes** - Helps with compositional reasoning
4. **Keep prompts descriptive but natural** - The Qwen3-4B encoder prefers natural language over tag soup
5. **Avoid CFG-dependent tricks** - Things like prompt weighting may not work as expected since CFG is already baked in

---

## 6. Connection to Z-Image Architecture

### 6.1 Application to S3-DiT

The S3-DiT (Single-Stream DiT) architecture used in Z-Image has specific characteristics:
- **6B parameters**: Large enough for high-quality generation
- **Single-stream design**: Simpler than cross-attention designs, text embeddings directly concatenated
- **Flow matching formulation**: Uses t=0 for noise, t=1 for clean data (as noted in the paper)

The Decoupled DMD method applies by:
1. Using the pre-trained S3-DiT as the teacher
2. Initializing student and fake model from the teacher
3. Running the CA+DM distillation process
4. Result: 8-step student that matches 50-step teacher quality

### 6.2 Qwen3-4B Text Encoder Role

**Specifications:**
- 2560 hidden dimension
- 36 layers
- Uses `hidden_states[-2]` for embeddings (penultimate layer)

**In the distillation process:**
- Text encoder is frozen during DMD training
- Only the DiT (student generator) is updated
- Text embeddings serve as conditioning for both real/fake score estimation

**Important**: The text encoder's understanding of the prompt structure is fixed. The distillation process only changes how the DiT responds to these embeddings.

### 6.3 16-Channel VAE Considerations

Z-Image uses the Wan-family 16-channel VAE:
- Same latent space as Wan2.1 models
- 8x spatial compression
- Compatible with upscale VAE variants

**During distillation:**
- VAE is frozen
- All operations happen in latent space
- The re-noising happens on VAE latents, not pixels

**For users:**
- Can swap VAE (e.g., Wan2.1-VAE-upscale2x) without affecting the distillation
- The 16-channel architecture is preserved regardless of VAE choice
- Quality improvements from better VAE decoders are orthogonal to distillation benefits

---

## 7. Key Takeaways

### For Users

| Setting | Recommended Value | Reason |
|---------|------------------|--------|
| CFG Scale | 0 | CFG is baked in during training |
| Steps | 8-9 | Model is distilled for this range |
| Scheduler | Euler or similar flow-based | Matches training formulation |
| Prompt Format | Full Qwen3 chat template | Expected by text encoder |

### For Developers

1. **Do not expose CFG controls** for Z-Image (or default to 0)
2. **Support thinking blocks** in prompt templates
3. **Document the baked-in nature** of CFG for LoRA trainers
4. **Validate step counts** - using more or fewer than 8 may degrade quality

### What the Distillation Means

**Advantages:**
- Fast inference (8x fewer NFEs than teacher)
- No CFG overhead (single model evaluation per step vs. two for CFG)
- Quality approaches 50-step teacher

**Limitations:**
- Fixed "baked in" aesthetic from training CFG scale
- Cannot trade prompt adherence for diversity via CFG
- May be more sensitive to prompt structure than non-distilled models

---

## 8. Confirmed Model Specifications

From the official Z-Image-Turbo model files:

### Text Encoder (Qwen3-4B)

```json
{
  "architectures": ["Qwen3ForCausalLM"],
  "hidden_size": 2560,
  "num_hidden_layers": 36,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "intermediate_size": 9728,
  "max_position_embeddings": 40960,
  "vocab_size": 151936,
  "model_type": "qwen3",
  "torch_dtype": "bfloat16"
}
```

### Scheduler

```json
{
  "_class_name": "FlowMatchEulerDiscreteScheduler",
  "num_train_timesteps": 1000,
  "use_dynamic_shifting": false,
  "shift": 3.0
}
```

### Pipeline Components

| Component | Class | Library |
|-----------|-------|---------|
| text_encoder | Qwen3Model | transformers |
| tokenizer | Qwen2Tokenizer | transformers |
| transformer | ZImageTransformer2DModel | diffusers |
| vae | AutoencoderKL | diffusers |
| scheduler | FlowMatchEulerDiscreteScheduler | diffusers |

### Official Inference Settings

From the README:
```python
pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    num_inference_steps=9,  # Results in 8 DiT forwards
    guidance_scale=0.0,     # Guidance should be 0 for Turbo models
    generator=torch.Generator("cuda").manual_seed(42),
)
```

**Key confirmation**: `guidance_scale=0.0` is explicitly documented as required for Turbo models.

---

## 9. DMDR: Additional Post-Training

The README also mentions **DMDR** (DMD + Reinforcement Learning), which further refines the model after Decoupled-DMD distillation:

> "To achieve further improvements in terms of semantic alignment, aesthetic quality, and structural coherence - while producing images with richer high-frequency details - we present DMDR."

This suggests Z-Image-Turbo may have undergone additional RL-based refinement after the core DMD distillation.

---

## 10. References

- Liu, D., Gao, P., et al. "Decoupled DMD: CFG Augmentation as the Spear, Distribution Matching as the Shield." Tongyi Lab, Alibaba Group / CUHK. arXiv:2511.22677
- Z-Image Technical Report: arXiv:2511.22699
- DMDR Paper: arXiv:2511.13649
- Z-Image GitHub: https://github.com/Tongyi-MAI/Z-Image
- Related works: DMD, DMD2, Diff-Instruct, SDXL-Lightning, LCM
