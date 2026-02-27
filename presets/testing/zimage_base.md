---
name: zimage_base_test
description: Z-Image Base visual verification baseline
category: testing
pipelines: [zimage]
variant: base

# Generation parameters - aligned with presets/zimage/portrait.md
negative_prompt: |
  plastic skin, doll, smooth skin, airbrushed, wax figure, semi-realistic,
  makeup, deformed, bad anatomy, bad hands, extra fingers, missing fingers,
  mutation, floating limbs, disconnected limbs, illustration, 3d render,
  painting, artwork, drawing, anime, cartoon, sketch, simple background,
  low quality, worst quality
guidance_scale: 4.0
steps: 40
shift: 6.0

# Test-specific fields (stored in metadata)
purpose: visual_verification
prompt: "A photo of a man eating spaghetti"
height: 1024
width: 1024
seed: 42
min_variance: 500
max_variance: 6000
---

Visual verification baseline for Z-Image Base model.

Uses seed=42 for reproducibility. Parameters aligned with production portrait preset.

Key parameters for Base model (vs Turbo):
- guidance_scale: 4.0 (required - Base doesn't bake in CFG)
- steps: 40 (matches production presets)
- shift: 6.0 (Base default, vs Turbo's 3.0)

Negative prompt aligned with presets/zimage/portrait.md (includes research-backed
tokens: wax figure, semi-realistic, simple background).

Variance thresholds:
- min_variance=500: Below this suggests solid color/failed generation
- max_variance=6000: Above this suggests noise/artifacts
