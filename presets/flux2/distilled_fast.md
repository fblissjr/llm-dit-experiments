---
name: distilled_fast
description: Distilled model defaults - 4 steps, low guidance for fast generation
category: model_defaults
pipelines: [flux2]
variant: distilled

num_steps: 4
guidance: 1.0
---

Default settings for FLUX.2 Klein distilled models (klein-9b, klein-9b-fp8,
klein-4b, klein-4b-fp8). Distilled models have CFG baked in during training,
so guidance=1.0 effectively disables additional CFG. 4 steps is the training
default -- going higher adds diminishing returns with distilled weights.
