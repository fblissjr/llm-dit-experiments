---
name: zimage_turbo_test
description: Z-Image Turbo visual verification baseline
category: testing
pipelines: [zimage]
variant: turbo

# Generation parameters (Turbo-specific)
negative_prompt: ""
guidance_scale: 0.0
steps: 9
shift: 3.0

# Test-specific fields (stored in metadata)
purpose: visual_verification
prompt: "A photo of a man eating spaghetti"
height: 1024
width: 1024
seed: 42
min_variance: 500
max_variance: 6000
---

Visual verification baseline for Z-Image Turbo model.

Turbo model has CFG baked into weights, so guidance_scale=0.0.

Key parameters for Turbo (vs Base):
- guidance_scale: 0.0 (CFG baked into distilled weights)
- steps: 9 (fewer steps needed due to distillation)
- shift: 3.0 (Turbo default)

Same variance thresholds as Base since output quality expectations are similar.
