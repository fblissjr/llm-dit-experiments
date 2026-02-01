---
name: architecture
description: Architecture/interior - sharp lines and correct perspective
category: negative_prompt
pipelines: [zimage]
variant: base

negative_prompt: |
  curved lines, distorted perspective, surreal, fantasy, painting, illustration, drawing, sketch, blur, blurry, bokeh, depth of field, tilt shift, simple background, low quality, worst quality, text, watermark, signature

guidance_scale: 4.0
steps: 40
shift: 6.0
---

Optimized for architectural and interior photography.

Key considerations for architecture:
- Perspective accuracy (targets distorted perspective, curved lines)
- Sharp focus throughout (targets blur, depth of field, tilt shift)
- Realistic style (removes fantasy, surreal, painting elements)

Note: Z-Image Base tends to apply strong depth of field (bokeh). If you need deep
depth of field (everything in focus), this preset explicitly includes those negatives.
