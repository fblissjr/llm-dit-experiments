---
name: bilingual
description: Bilingual Chinese/English - enhanced negative prompt adherence
category: negative_prompt
pipelines: [zimage]
variant: base

negative_prompt: |
  illustration, 3d render, painting, artwork, drawing, anime, cartoon, sketch,
  graphic, cgi, low quality, worst quality, text, watermark,
  plastic skin, doll, smooth skin, airbrushed,
  deformed, bad anatomy, bad hands, extra fingers, missing fingers,
  mutation, floating limbs, disconnected limbs,
  bokeh, blur, depth of field,
  bad composition, cropped, out of frame,
  ugly, disfigured,
  oversaturated, undersaturated,
  jpg artifacts, compression artifacts,
  noise, film grain,
  插画, 绘画, 卡通, 3D渲染

guidance_scale: 4.0
steps: 40
shift: 6.0
---

Bilingual preset with Chinese negative terms for improved adherence.

Z-Image is trained on a bilingual mix of high-quality art and photos. If English
negatives alone fail to suppress certain styles, appending Chinese terms can
improve adherence:

- 插画 (illustration)
- 绘画 (painting)
- 卡通 (cartoon)
- 3D渲染 (3D render)

This preset is comprehensive, including:
- Universal photorealism negatives
- Portrait/skin texture negatives
- Blur/depth of field control
- Quality and artifact control
- Chinese equivalents of key terms

Based on: internal/research/z-image/zimage_base_negative_prompts_research_by_gemini.md
