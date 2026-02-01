---
name: photorealistic
description: Universal photorealism - removes artistic/digital artifacts
category: negative_prompt
pipelines: [zimage]
variant: base

negative_prompt: |
  illustration, 3d render, painting, artwork, drawing, anime, cartoon, sketch,
  graphic, cgi, low quality, worst quality, text, watermark

guidance_scale: 4.0
steps: 40
shift: 6.0
---

Research-backed negative prompt for photorealistic generation.

The core strategy is "subtraction" - Z-Image Base preserves a vast spectrum of
visual languages including anime, illustration, and 3D rendering. For photorealism,
you must explicitly tell the model NOT to use its artistic capabilities.

This preset targets:
- Artistic styles (illustration, 3d render, painting, anime, cartoon)
- Quality issues (low quality, artifacts, watermarks)

Based on: internal/research/z-image/zimage_base_negative_prompts_research_by_gemini.md
