---
name: universal
description: Maximum photorealism - comprehensive AI artifact suppression
category: negative_prompt
pipelines: [zimage]
variant: base

negative_prompt: |
  illustration, 3d render, painting, artwork, drawing, anime, cartoon, sketch,
  graphic, cgi, vector art, digital art, unreal engine, video game, fantasy style,
  plastic, plastic skin, smooth skin, airbrushed, wax figure, semi-realistic,
  doll like, over-smooth, retouching, makeup, glossy, fake skin, flat lighting,
  deformed, bad anatomy, bad hands, extra fingers, missing fingers, mutation,
  floating limbs, disconnected limbs, disfigured, extra limbs, missing limbs, ugly,
  simple background, bokeh, depth of field,
  low quality, worst quality, text, watermark, signature, username, error,
  jpeg artifacts, blurry, noise, grain

guidance_scale: 4.0
steps: 50
shift: 6.0
---

Comprehensive negative prompt combining all research-backed interventions.

This is the "kitchen sink" preset for maximum photorealism, combining:

1. **Style Suppression** (Universal Photorealism)
   - illustration, 3d render, painting, artwork, drawing
   - anime, cartoon, sketch, graphic, cgi
   - vector art, digital art, unreal engine, video game, fantasy style

2. **Texture/Skin Artifacts** (Portrait)
   - plastic, plastic skin, smooth skin, airbrushed, wax figure
   - doll like, over-smooth, retouching, makeup, glossy
   - fake skin, flat lighting

3. **Style Ambiguity**
   - semi-realistic (forces commitment to true photorealism)

4. **Anatomy Issues**
   - deformed, bad anatomy, bad hands, extra/missing fingers
   - mutation, floating/disconnected/extra/missing limbs, disfigured, ugly

5. **Environmental Control**
   - simple background (forces realistic scene complexity)
   - bokeh, depth of field (for sharp focus throughout)

6. **Quality Control**
   - low quality, worst quality, text, watermark, signature
   - username, error, jpeg artifacts, blurry, noise, grain

Use this preset when:
- Other presets still produce AI-looking output
- Subject matter spans multiple categories (e.g., person in architectural setting)
- Maximum realism is more important than prompt adherence

Note: This preset uses 50 steps (research maximum) for best quality.
May slightly reduce prompt adherence due to the comprehensive negative list.
