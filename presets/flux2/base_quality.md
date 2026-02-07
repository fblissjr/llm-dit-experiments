---
name: base_quality
description: Base model defaults - 50 steps, higher guidance for maximum quality
category: model_defaults
pipelines: [flux2]
variant: base

num_steps: 50
guidance: 4.0
---

Default settings for FLUX.2 Klein base models (klein-base-9b, klein-base-9b-fp8,
klein-base-4b, klein-base-4b-fp8). Base models are not distilled, so they need
full denoising (50 steps) and explicit classifier-free guidance (4.0). These
produce higher quality results but take ~12x longer than distilled.

Guidance range: 3.0-5.0 is the recommended range. Below 3.0 produces blurry
results. Above 5.0 can oversaturate.
