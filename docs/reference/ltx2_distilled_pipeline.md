Last Updated: 2026-03-13

> **v0.9.31:** `pipeline_mode="distilled"` and `DISTILLED_SIGMA_VALUES` removed.
> The distilled code path required a pre-distilled checkpoint that doesn't exist.
> We use `TI2VidTwoStagesPipeline` (base + distilled LoRA) exclusively.
> Stage 2 sigma schedule (`STAGE_2_DISTILLED_SIGMA_VALUES`) is kept.

# LTX-2.3 Distilled Pipeline Reference

Reference documentation for the official LTX-2.3 distilled and non-distilled two-stage pipelines, based on the code at `coderef/LTX-2/`. Covers pipeline selection, sigma schedules, guidance mechanics, Euler stepping, noise initialization, and known divergences from our implementation.

---

## 1. Distilled vs Non-Distilled Pipelines

The reference repo provides two distinct two-stage pipeline classes:

### DistilledPipeline (`coderef/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/distilled.py`)

- Uses a **distilled checkpoint** (model weights already baked with distillation -- no separate base+LoRA).
- Both stage 1 and stage 2 use `simple_denoising_func` -- a **single forward pass per step** with no CFG, STG, or any guidance.
- No negative prompt encoding. Only the positive prompt is encoded (line 95: `encode_prompts([prompt], ...)`).
- Stage 1: 8 denoising steps using `DISTILLED_SIGMA_VALUES`.
- Stage 2: 3 denoising steps using `STAGE_2_DISTILLED_SIGMA_VALUES`.

### TI2VidTwoStagesPipeline (`coderef/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages.py`)

- Uses a **non-distilled base checkpoint** for stage 1, then loads the same model **with additional distilled LoRA(s)** for stage 2 (line 73: `self.stage_1_model_ledger.with_additional_loras(loras=distilled_lora)`).
- Stage 1 uses `multi_modal_guider_factory_denoising_func` with full **CFG + STG + modality guidance** -- up to 4 forward passes per step.
- Stage 2 uses `simple_denoising_func` -- **single forward pass**, no guidance.
- Both positive and negative prompts are encoded (line 105: `encode_prompts([prompt, negative_prompt], ...)`).
- Stage 1: `num_inference_steps` steps (30 for V2.3, 40 for V2.0) with dynamic sigma schedule from `LTX2Scheduler`.
- Stage 2: 3 steps using `STAGE_2_DISTILLED_SIGMA_VALUES`.

### When to Use Which

| Scenario | Pipeline | Stage 1 Guidance | Stage 1 Steps |
|----------|----------|-----------------|---------------|
| Fast generation (distilled model checkpoint) | DistilledPipeline | None (single pass) | 8 |
| High quality (base model + distilled LoRA) | TI2VidTwoStagesPipeline | Full (CFG+STG+modality) | 30 (V2.3) / 40 (V2.0) |

Both pipelines share the same stage 2 behavior: distilled sigma schedule, `simple_denoising_func`, no guidance.

---

## 2. Sigma Schedules

Source: `coderef/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py`, lines 15-18.

### Stage 1 Distilled (8 steps)

```python
DISTILLED_SIGMA_VALUES = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
```

9 values define 8 denoising steps. Sigma decreases from 1.0 to 0.0, with most values clustered near 1.0 (first 5 values span only 0.025 of sigma space), then larger jumps toward the end.

### Stage 2 Distilled (3 steps)

```python
STAGE_2_DISTILLED_SIGMA_VALUES = [0.909375, 0.725, 0.421875, 0.0]
```

4 values define 3 denoising steps. This is a strict suffix of `DISTILLED_SIGMA_VALUES` (the last 4 values). Stage 2 starts at sigma 0.909375, not 1.0, because the upsampled stage 1 output is re-noised to this level.

### Stage 1 Non-Distilled (dynamic schedule)

For the non-distilled pipeline, stage 1 uses `LTX2Scheduler` (`coderef/LTX-2/packages/ltx-core/src/ltx_core/components/schedulers.py`, line 14):

```python
sigmas = LTX2Scheduler().execute(steps=num_inference_steps).to(dtype=torch.float32, device=self.device)
```

`LTX2Scheduler.execute()` generates `steps + 1` linearly spaced values from 1.0 to 0.0, applies a token-count-dependent shift (interpolated between `base_shift=0.95` at 1024 tokens and `max_shift=2.05` at 4096 tokens), then optionally stretches so the last non-zero sigma maps to `terminal=0.1`. Default `default_number_of_tokens=4096` when no latent is provided (line 29).

---

## 3. Guidance

### Distilled: No Guidance

`simple_denoising_func` (`coderef/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/helpers.py`, lines 315-328):

```python
def simple_denoising_func(video_context, audio_context, transformer):
    def simple_denoising_step(video_state, audio_state, sigmas, step_index):
        sigma = sigmas[step_index]
        pos_video = modality_from_latent_state(video_state, video_context, sigma)
        pos_audio = modality_from_latent_state(audio_state, audio_context, sigma)
        denoised_video, denoised_audio = transformer(video=pos_video, audio=pos_audio, perturbations=None)
        return denoised_video, denoised_audio
    return simple_denoising_step
```

One forward pass. No negative prompts, no CFG delta, no perturbations. The transformer receives positive-prompt context only and returns denoised (x0) predictions directly.

### Non-Distilled: Multi-Modal Guider

`multi_modal_guider_denoising_func` (`helpers.py`, lines 361-465) runs up to 4 forward passes per step:

1. **Conditional pass** (always): positive context for both modalities.
2. **Unconditional pass** (when `cfg_scale != 1.0`): negative context for both modalities.
3. **Perturbed pass** (when `stg_scale != 0.0`): positive context but with self-attention skipped at specified blocks.
4. **Modality-isolated pass** (when `modality_scale != 1.0`): positive context but with cross-modal attention (A2V + V2A) skipped.

The combined formula in `MultiModalGuider.calculate()` (`coderef/LTX-2/packages/ltx-core/src/ltx_core/components/guiders.py`, lines 256-261):

```python
pred = (
    cond
    + (cfg_scale - 1) * (cond - uncond_text)
    + stg_scale * (cond - uncond_perturbed)
    + (modality_scale - 1) * (cond - uncond_modality)
)
```

With optional CFG rescaling when `rescale_scale != 0`:

```python
factor = cond.std() / pred.std()
factor = rescale_scale * factor + (1 - rescale_scale)
pred = pred * factor
```

### V2.3 Default Guider Parameters

Source: `constants.py`, lines 68-73.

| Parameter | Video | Audio |
|-----------|-------|-------|
| `cfg_scale` | 3.0 | 7.0 |
| `stg_scale` | 1.0 | 1.0 |
| `rescale_scale` | 0.7 | 0.7 |
| `modality_scale` | 3.0 | 3.0 |
| `skip_step` | 0 | 0 |
| `stg_blocks` | [28] | [28] |

---

## 4. V2.3 vs V2.0 Differences

Source: `constants.py`, lines 65-73.

| Parameter | V2.0 (`LTX_2_PARAMS`) | V2.3 (`LTX_2_3_PARAMS`) |
|-----------|----------------------|------------------------|
| `num_inference_steps` | 40 | 30 |
| `stg_blocks` (video) | [29] | [28] |
| `stg_blocks` (audio) | [29] | [28] |

All other parameters are inherited from the base `PipelineParams` defaults. The `stg_blocks` difference reflects the different number of transformer layers in V2.3 vs V2.0 (the reference perturbs different block indices).

Version detection (`detect_params()`, lines 104-124) reads `model_version` from safetensors metadata. Versions starting with `"2.3"` get `LTX_2_3_PARAMS`; all others fall back to `LTX_2_PARAMS`.

There is also an `LTX_2_3_HQ_PARAMS` variant (lines 74-94) with `num_inference_steps=15`, larger resolution (544x960 stage 1), `stg_scale=0.0` (no STG), and `stg_blocks=[]`.

---

## 5. Two-Stage Flow

Both pipelines follow the same two-stage structure:

### Stage 1: Half-Resolution Generation

1. Set output shape to `(height/2, width/2)` -- e.g., 512x768 for 1024x1536 final output.
2. Initialize video and audio latent states (see section 7 for noise init).
3. Run denoising loop (8 steps distilled, or 30/40 steps non-distilled).
4. Unpatchify latent states back to spatial format.

### Stage 1.5: Spatial Upsampling

1. `upsample_video()` takes the stage 1 video latent and upsamples it 2x using a learned spatial upsampler.
   - Reference: `ltx_core.model.upsampler.upsample_video(latent, video_encoder, upsampler)`.
   - The upsampler works in VAE-decoded pixel space: decode -> upsample -> re-encode.

### Stage 2: Full-Resolution Refinement

1. Set output shape to `(height, width)` -- full target resolution.
2. Re-noise the upsampled video latent at `noise_scale=stage_2_sigmas[0]` (0.909375).
3. Re-noise the stage 1 audio latent at the same noise scale.
4. Run denoising loop (3 steps) with distilled sigma schedule and `simple_denoising_func` -- no guidance.
5. VAE decode video and audio.

### Key: Same Transformer for Both Stages

- **Distilled pipeline**: Uses the same distilled transformer instance for both stages (no LoRA swapping).
- **Non-distilled pipeline**: Creates a `stage_2_model_ledger` that adds distilled LoRA(s) on top of the base model's LoRAs. Stage 2 loads a new transformer instance from this ledger (line 200: `transformer = self.stage_2_model_ledger.transformer()`).

---

## 6. Euler Step Mechanics

Source: `coderef/LTX-2/packages/ltx-core/src/ltx_core/components/diffusion_steps.py`, lines 7-22.

### The Transformer Predicts x0 (Denoised Sample)

The reference transformer returns **x0 predictions** (denoised samples), not velocity. The conversion happens inside `EulerDiffusionStep.step()`.

### EulerDiffusionStep

```python
class EulerDiffusionStep:
    def step(self, sample, denoised_sample, sigmas, step_index):
        sigma = sigmas[step_index]
        sigma_next = sigmas[step_index + 1]
        dt = sigma_next - sigma
        velocity = to_velocity(sample, sigma, denoised_sample)
        return (sample.float() + velocity.float() * dt).to(sample.dtype)
```

### Velocity Conversion

Source: `coderef/LTX-2/packages/ltx-core/src/ltx_core/utils.py`, lines 21-36.

```python
def to_velocity(sample, sigma, denoised_sample):
    return (sample - denoised_sample) / sigma
```

### Full Euler Update (Expanded)

Substituting `to_velocity` into the Euler step:

```
velocity = (sample - denoised) / sigma
x_next = sample + velocity * (sigma_next - sigma)
x_next = sample + (sample - denoised) / sigma * (sigma_next - sigma)
```

When `sigma_next = 0` (final step): `x_next = sample + (sample - denoised) / sigma * (0 - sigma) = sample - (sample - denoised) = denoised`.

### Post-Processing

After each step, `post_process_latent()` blends the stepped latent with the clean reference latent using the denoise mask:

```python
output = denoised * mask + clean * (1 - mask)
```

This preserves conditioned regions (mask=0) from the original latent.

---

## 7. Noise Initialization

### Stage 1: GaussianNoiser

Source: `coderef/LTX-2/packages/ltx-core/src/ltx_core/components/noisers.py`, lines 15-35.

```python
class GaussianNoiser:
    def __call__(self, latent_state, noise_scale=1.0):
        noise = torch.randn(...)
        scaled_mask = latent_state.denoise_mask * noise_scale
        latent = noise * scaled_mask + latent_state.latent * (1 - scaled_mask)
        return replace(latent_state, latent=latent)
```

For standard T2V (no image conditioning), `denoise_mask` is all 1s and `noise_scale` is 1.0, so the latent becomes pure Gaussian noise. For I2V, conditioned regions (mask=0) retain their clean encoded-image latent.

The initial `latent_state.latent` starts as either zeros (from `tools.create_initial_state()`) or a provided `initial_latent`.

### Stage 2: Flow-Matching Re-Noising

Stage 2 receives `initial_video_latent` (the upsampled stage 1 output) and `initial_audio_latent` (stage 1 audio output). These are passed through the same `GaussianNoiser` with `noise_scale=stage_2_sigmas[0]` (0.909375).

The noiser formula with `noise_scale < 1.0`:

```
scaled_mask = denoise_mask * noise_scale  # = 1.0 * 0.909375 = 0.909375
latent = noise * 0.909375 + initial_latent * (1 - 0.909375)
latent = 0.909375 * noise + 0.090625 * initial_latent
```

This is the standard flow-matching interpolation: `x_t = t * eps + (1 - t) * x_0`.

---

## 8. Divergences From Our Implementation

Our implementation: `src/llm_dit/pipelines/generate.py`.

### 8.1. Velocity vs x0 Prediction Semantics

**Reference**: The transformer returns **x0 predictions**. `EulerDiffusionStep` converts to velocity via `to_velocity(sample, sigma, denoised)` and then applies the Euler update.

**Ours**: `_compute_velocity()` and `_compute_av_velocity()` treat the model output as **velocity predictions** directly. The Euler step is applied as:

```python
# Our implementation (generate.py, line 1759)
dt = sigma_next - sigma
denoised = (latents.float() + velocity.float() * dt).to(dtype)
```

This is mathematically equivalent if our transformer's output is the velocity `v = (x - x0) / sigma` rather than x0. The expanded reference formula is:

```
x_next = x + (x - denoised) / sigma * (sigma_next - sigma)
```

If we define `velocity = (x - denoised) / sigma`, then `x_next = x + velocity * dt`, which matches our code. The key difference is **where the conversion happens**: the reference converts inside the stepper; we assume the model already outputs velocity. This is consistent as long as our transformer wraps its x0 output with the velocity conversion.

### 8.2. Stage 2 Re-Noising Formula

**Reference**: Uses `GaussianNoiser` with `noise_scale=0.909375` applied to `initial_video_latent`:

```python
scaled_mask = denoise_mask * noise_scale  # denoise_mask is all 1s for T2V
latent = noise * scaled_mask + initial_latent * (1 - scaled_mask)
# = 0.909375 * noise + 0.090625 * initial_latent
```

**Ours** (`generate.py`, lines 2464-2468):

```python
noise_scale = distilled_sigmas[0].item()  # 0.909375
noise = torch.randn_like(latents_flat, ...)
latents_noisy = (1 - noise_scale) * latents_flat + noise_scale * noise
# = 0.090625 * latents_flat + 0.909375 * noise
```

These are **identical** -- both implement `x_t = (1-t)*x_0 + t*eps` at `t=0.909375`.

### 8.3. Separate Audio Guidance Scale

**Reference**: Video and audio have **independent** `MultiModalGuiderParams` with different `cfg_scale` values (video=3.0, audio=7.0 by default). Each modality gets its own `MultiModalGuider` instance.

**Ours**: `_compute_av_velocity()` uses a single `ctx.guidance_scale` for video CFG and falls back to it when `ctx.audio_guidance_scale` is 0 (`generate.py`, line 1569). The two-stage config does pass `audio_guidance_scale` separately, but the default `TwoStageConfig` does not expose separate audio STG/rescale/modality scales -- they share the video values.

### 8.4. Stage 2 Audio Re-Noising Source

**Reference**: Stage 2 re-noises the stage 1 **denoised audio** latent (line 181 in `ti2vid_two_stages.py`: `initial_audio_latent=audio_state.latent`). This is the fully denoised audio from stage 1 (after unpatchify and clearing conditioning).

**Ours**: Generates a separate audio noise tensor at stage 1 init time (`audio_noise`, line 2242-2247) and uses flow-matching interpolation on the stage 1 audio output:

```python
audio_latents_noisy = (1 - noise_scale) * audio_latents + noise_scale * audio_noise
```

The reference passes `audio_state.latent` as `initial_audio_latent` to `denoise_audio_video()`, which feeds it into `GaussianNoiser`. The noiser generates **fresh** noise internally from its seeded generator. Our approach pre-generates the noise with a separate `torch.randn` call (seeded from `config.seed + 1`), which could produce different noise values from the reference's `GaussianNoiser` generator state.

### 8.5. Non-Distilled Stage 1 Sigma Schedule: Latent Shape

**Reference** (`ti2vid_two_stages.py`, line 138):

```python
sigmas = LTX2Scheduler().execute(steps=num_inference_steps).to(...)
```

No `latent` argument is passed, so `LTX2Scheduler` uses `default_number_of_tokens=4096` for the shift calculation.

**Ours** (`generate.py`, lines 2257-2265):

```python
mock_latent = torch.empty(1, 128, t_latent, h_latent, w_latent)
sigmas = scheduler.execute(
    steps=two_stage.stage1_steps,
    latent=mock_latent,
    max_shift=config.max_shift,
    ...
)
```

We pass a mock latent so the scheduler computes tokens from `math.prod(latent.shape[2:])` -- the actual stage 1 latent token count. At default 512x768 stage 1 resolution with 121 frames, this is `16 * 16 * 24 = 6144` tokens, which gives a **different shift** than the reference's 4096 default.

### 8.6. Gradient Estimation Placement

**Reference**: `gradient_estimating_euler_denoising_loop` (`samplers.py`, lines 69-126) operates on **denoised predictions** (x0). It converts to velocity, applies the GE correction on velocity, converts back to denoised, and then passes to the stepper. Importantly, when `sigma_next == 0`, it skips GE entirely and returns the raw denoised prediction (line 113-114).

**Ours**: `_denoise_stage()` applies GE correction directly on velocity predictions before the Euler step (`generate.py`, lines 1750-1755). We do not have the `sigma_next == 0` short-circuit.

### 8.7. Transformer Lifecycle in Non-Distilled Pipeline

**Reference**: Deletes the stage 1 transformer (line 176: `del transformer`) and loads a **new** transformer from `stage_2_model_ledger` with distilled LoRA pre-fused (line 200). Two separate model loads.

**Ours**: Reuses the stage 1 transformer and applies the distilled LoRA on top (in-place fusion) for stage 2 (`generate.py`, lines 2408-2438). One model load, LoRA fused incrementally. This saves load time but means the base LoRA + distilled LoRA are both fused into the same weights.

### 8.8. Resolution Validation

**Reference** (`helpers.py`, lines 640-651): Two-stage pipelines require height and width divisible by **64**. One-stage requires divisible by 32.

**Ours**: `_validate_two_stage_dimensions()` validates but the specific divisor check may differ. The reference's 64-divisibility requirement accounts for the 2x upsampling (32 VAE compression * 2 upscale factor).

### 8.9. STG Perturbation Targeting

**Reference** (`helpers.py`, lines 416-431): Video and audio STG perturbations can target **different** perturbation types (`SKIP_VIDEO_SELF_ATTN` and `SKIP_AUDIO_SELF_ATTN`), and each guider has its own `stg_blocks` list.

**Ours** (`generate.py`, lines 1584-1587): Both video and audio STG perturbations use the same `ctx.stg_blocks` list. There is no separate `audio_stg_blocks` parameter.

### 8.10. Skip Step Logic

**Reference**: `MultiModalGuider.should_skip_step()` (`guiders.py`, lines 282-286) implements step skipping: when `skip_step > 0`, the guider skips steps where `step % (skip_step + 1) != 0`, reusing the previous denoised output.

**Ours**: No skip-step mechanism is implemented.

---

## Summary Table

| Aspect | Reference (Distilled) | Reference (Non-Distilled) | Our Implementation |
|--------|----------------------|--------------------------|-------------------|
| Stage 1 steps | 8 (fixed sigmas) | 30 V2.3 / 40 V2.0 (dynamic) | Configurable (`use_distilled_sigmas` flag) |
| Stage 1 guidance | None | CFG+STG+modality (up to 4 passes) | Configurable (matches non-distilled) |
| Stage 2 steps | 3 (fixed sigmas) | 3 (fixed sigmas) | 3 (fixed sigmas) |
| Stage 2 guidance | None | None | None |
| Model output | x0 prediction | x0 prediction | Velocity (equivalent) |
| Audio CFG scale | N/A | 7.0 (independent) | Separate param, defaults to video scale |
| GE support | No | Via separate loop | Inline in denoising loop |
| Transformer reuse | Same instance | Separate loads per stage | Same instance, LoRA fused in-place |
