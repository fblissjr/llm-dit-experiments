---
name: ltx2_reference_test
description: LTX-2 reference test baseline (high quality, full length)
category: testing
pipelines: [ltx2]
variant: base

# Generation parameters - official defaults for maximum quality
guidance_scale: 3.0
steps: 40
num_frames: 121
height: 512
width: 768
frame_rate: 24.0

# STG (Spatio-Temporal Guidance) parameters
stg_scale: 1.0
rescale_scale: 0.7
a2v_guidance_scale: 3.0
stg_blocks: [29]

# Negative prompt - full official negative prompt
negative_prompt: |
  blurry, out of focus, overexposed, underexposed, low contrast, washed out colors,
  excessive noise, grainy texture, poor lighting, flickering, motion blur,
  distorted proportions, unnatural skin tones, deformed facial features,
  asymmetrical face, missing facial features, extra limbs, disfigured hands,
  wrong hand count, artifacts around text, inconsistent perspective, camera shake,
  incorrect depth of field, background too sharp, background clutter,
  distracting reflections, harsh shadows, inconsistent lighting direction,
  color banding, cartoonish rendering, 3D CGI look, unrealistic materials,
  uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions,
  wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice,
  robotic voice, echo, background noise, off-sync audio, incorrect dialogue,
  added dialogue, repetitive speech, jittery movement, awkward pauses,
  incorrect timing, unnatural transitions, inconsistent framing, tilted camera,
  flat lighting, inconsistent tone, cinematic oversaturation, stylized filters,
  or AI artifacts

# Test-specific fields (stored in metadata)
purpose: reference_baseline
prompt: "The sun rises over a misty mountain range, casting golden light across alpine meadows dotted with wildflowers. A lone hiker emerges from a forest path, pausing to take in the breathtaking vista. The camera slowly pans right, revealing a crystal-clear lake reflecting the surrounding peaks. Morning birdsong fills the air as wisps of fog drift lazily across the water's surface. The hiker adjusts their backpack and continues along the ridge trail, their silhouette framed against the brightening sky."
seed: 42
min_variance: 500
max_variance: 8000
---

LTX-2 reference test baseline for comprehensive quality validation.

Uses full official parameter defaults with a complex multi-scene prompt
to validate the pipeline at production quality levels.

Key parameters:
- guidance_scale: 3.0 (official default)
- steps: 40 (official default)
- num_frames: 121 (~5 seconds at 24fps)
- resolution: 512x768 (official 1-stage default)
- STG parameters: Full official configuration

Prompt characteristics:
- Multi-scene narrative with natural transitions
- Dynamic camera movement (pan)
- Atmospheric elements (fog, light)
- Human subject interaction with environment
- Audio description (birdsong)

Expected performance:
- Time: ~10-15 minutes on RTX 4090
- VRAM: ~16-20 GB (FP8 quantized)

Use this baseline for:
- Pre-release quality validation
- Reference comparison against official repo
- Regression testing after significant changes
