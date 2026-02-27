---
name: flux2_base_test
description: FLUX.2 base (non-distilled) visual verification baseline
category: testing
pipelines: [flux2]
variant: base

# Generation parameters (base model defaults)
guidance_scale: 3.5
steps: 28

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

Visual verification baseline for FLUX.2 base (non-distilled) models.

Base models require more steps and guidance than distilled variants.

Key parameters:
- guidance_scale: 3.5 (base models need proper CFG)
- steps: 28 (more steps for quality generation)

Note: The FLUX.2 Klein models in this codebase are distilled. This preset
exists for potential future support of non-distilled FLUX variants.
