---
name: flux2_distilled_test
description: FLUX.2 Klein distilled visual verification baseline
category: testing
pipelines: [flux2]
variant: distilled

# Generation parameters (distilled model defaults)
guidance_scale: 1.0
steps: 4

# Test-specific fields (stored in metadata)
purpose: visual_verification
prompt: "A photo of a man eating spaghetti"
height: 1024
width: 1024
seed: 42
min_variance: 500
max_variance: 6000
offload_between_stages: true
---

Visual verification baseline for FLUX.2 Klein distilled models.

Distilled models (Klein-4B, Klein-9B) use minimal steps and low guidance.

Key parameters:
- guidance_scale: 1.0 (distilled models need minimal guidance)
- steps: 4 (distilled models converge quickly)
- offload_between_stages: true (three-stage memory offloading)

These settings match the production defaults in Flux2GenerationConfig.
