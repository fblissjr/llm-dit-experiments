---
name: ltx2_two_stage_smoke
description: LTX-2 two-stage smoke test (10 stage1 steps + 3 stage2 steps)
category: testing
pipelines: [ltx2]
variant: two_stage
guidance_scale: 3.5
steps: 10
num_frames: 33
height: 512
width: 768
frame_rate: 24.0
stage1_steps: 10
stage2_steps: 3
rescale_scale: 0.7
distilled_lora_path: ltx-2-19b-distilled-lora-384.safetensors
distilled_lora_scale: 0.8
spatial_upsampler_file: ltx-2-spatial-upscaler-x2-1.0.safetensors
negative_prompt: "worst quality, blurry, distorted"
purpose: two_stage_smoke_test
prompt: "A cat walking through a sunny garden"
seed: 42
---

LTX-2 two-stage smoke test for quick pipeline validation.

Uses reduced stage 1 steps (10 vs 40) to enable fast iteration while
still exercising the full two-stage flow: encode, low-res denoise,
spatial upsample, high-res refine, decode.

Key parameters:
- guidance_scale: 3.5 (official two-stage default)
- stage1_steps: 10 (reduced from 40 for speed)
- stage2_steps: 3 (official distilled LoRA default)
- rescale_scale: 0.7 (CFG rescaling)
- resolution: 512x768 (full output, stage 1 runs at 256x384)

Expected performance:
- Time: ~3-5 minutes on RTX 4090
- VRAM: ~14-16 GB (FP8 quantized)
