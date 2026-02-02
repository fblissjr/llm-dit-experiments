---
name: ltx2_smoke_test
description: LTX-2 smoke test baseline (fastest validation)
category: testing
pipelines: [ltx2]
variant: base

# Generation parameters - optimized for quick validation
guidance_scale: 3.0
steps: 30
num_frames: 33
height: 512
width: 768
frame_rate: 24.0

# Negative prompt - based on official LTX-2 defaults
negative_prompt: |
  blurry, out of focus, overexposed, underexposed, low contrast, washed out colors,
  excessive noise, grainy texture, poor lighting, flickering, motion blur,
  distorted proportions, unnatural skin tones, deformed facial features,
  asymmetrical face, missing facial features, extra limbs, disfigured hands,
  wrong hand count, artifacts around text, inconsistent perspective

# Test-specific fields (stored in metadata)
purpose: smoke_test
prompt: "A cat walking"
seed: 42
min_variance: 500
max_variance: 8000
---

LTX-2 smoke test baseline for quick pipeline validation.

Uses minimal parameters to enable fast iteration while still verifying
the pipeline produces coherent output.

Key parameters:
- guidance_scale: 3.0 (official default)
- steps: 30 (reduced from 40 for faster validation)
- num_frames: 33 (~1.4 seconds at 24fps)
- resolution: 512x768 (official 1-stage default)

Variance thresholds:
- min_variance=500: Below this suggests solid color/failed generation
- max_variance=8000: Above this suggests noise/artifacts

Expected performance:
- Time: ~2-3 minutes on RTX 4090
- VRAM: ~14-16 GB (FP8 quantized)
