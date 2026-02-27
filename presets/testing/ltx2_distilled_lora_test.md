---
name: ltx2_distilled_lora_test
description: LTX-2 with distilled LoRA (official recommended setup)
category: testing
pipelines: [ltx2]
variant: dev_lora

# Generation parameters - smoke-tier for fast testing
guidance_scale: 3.0
steps: 30
num_frames: 33
height: 512
width: 768
frame_rate: 24.0

# LoRA configuration
lora_path: /home/fbliss/Storage/LTX-2/ltx-2-19b-distilled-lora-384.safetensors
lora_scale: 0.8

# Negative prompt - based on official LTX-2 defaults
negative_prompt: |
  blurry, out of focus, overexposed, underexposed, low contrast, washed out colors,
  excessive noise, grainy texture, poor lighting, flickering, motion blur,
  distorted proportions, unnatural skin tones, deformed facial features,
  asymmetrical face, missing facial features, extra limbs, disfigured hands,
  wrong hand count, artifacts around text, inconsistent perspective

# Test-specific fields (stored in metadata)
purpose: lora_test
prompt: "A cat walking"
seed: 42
min_variance: 500
max_variance: 8000
---

LTX-2 with distilled LoRA test preset.

Uses the official `ltx-2-19b-distilled-lora-384.safetensors` LoRA to enhance
generation quality. This is the recommended setup for production use:
- Dev checkpoint (base model)
- Distilled LoRA at 0.8 scale

The distilled LoRA was trained to improve visual quality and temporal coherence
when used with the development checkpoint. It's particularly effective for:
- Reducing artifacts
- Improving motion smoothness
- Enhancing detail consistency

Key parameters:
- guidance_scale: 3.0 (official default)
- steps: 30 (reduced from 40 for faster validation)
- num_frames: 33 (~1.4 seconds at 24fps)
- resolution: 512x768 (official 1-stage default)
- lora_scale: 0.8 (recommended for distilled LoRA)

Expected performance:
- Time: ~2-3 minutes on RTX 4090
- VRAM: ~14-16 GB (FP8 quantized + LoRA)

Note: LoRA is fused into the model weights during generation. To use a different
LoRA or no LoRA, the model must be reloaded.
