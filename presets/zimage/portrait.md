---
name: portrait
description: Portrait/human optimized - natural skin texture and anatomy
category: negative_prompt
pipelines: [zimage]
variant: base

negative_prompt: |
  plastic skin, doll, smooth skin, airbrushed, makeup, deformed, bad anatomy,
  bad hands, extra fingers, missing fingers, mutation, floating limbs,
  disconnected limbs, illustration, 3d render, painting, artwork, drawing,
  anime, cartoon, sketch, low quality, worst quality

guidance_scale: 4.0
steps: 40
shift: 6.0
---

Optimized for portrait and human subject photography.

Z-Image Base can produce images with an "AI look" or smoothness characteristic of
early Flux or SDXL models. This preset counteracts the "waxy" effect by:

- Targeting artificial skin textures (plastic skin, doll, smooth skin, airbrushed)
- Addressing anatomy issues (bad hands, extra fingers, deformed)
- Including universal photorealism negatives

The result should render skin pores, imperfections, and natural texture variance.

Based on: internal/research/z-image/zimage_base_negative_prompts_research_by_gemini.md
