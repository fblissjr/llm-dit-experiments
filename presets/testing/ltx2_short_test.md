---
name: ltx2_short_test
description: LTX-2 short test baseline (balanced quality/speed)
category: testing
pipelines: [ltx2]
variant: base

# Generation parameters - balanced for quality validation
guidance_scale: 3.0
steps: 40
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
  wrong hand count, artifacts around text, inconsistent perspective, camera shake,
  incorrect depth of field, background clutter, harsh shadows, color banding,
  cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect

# Test-specific fields (stored in metadata)
purpose: quality_validation
prompt: "A golden retriever puppy runs through a sun-dappled forest trail, kicking up fallen autumn leaves. The camera follows low to the ground, capturing the dog's joyful expression as it bounds forward. Warm afternoon light filters through the canopy above, creating dynamic shadows that dance across the scene."
seed: 42
min_variance: 500
max_variance: 8000
---

LTX-2 short test baseline for quality validation.

Uses official parameter defaults with a more complex prompt to validate
that the pipeline handles realistic generation scenarios.

Key parameters:
- guidance_scale: 3.0 (official default)
- steps: 40 (official default)
- num_frames: 33 (~1.4 seconds at 24fps)
- resolution: 512x768 (official 1-stage default)

Prompt style:
- Narrative screenplay format
- Camera movement description
- Lighting and atmosphere details
- Natural action sequence

Expected performance:
- Time: ~3-5 minutes on RTX 4090
- VRAM: ~14-16 GB (FP8 quantized)
