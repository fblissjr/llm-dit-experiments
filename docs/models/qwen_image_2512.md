# Qwen-Image-2512 Text-to-Image Pipeline

last updated: 2025-12-31

## Overview

Qwen-Image-2512 is a pure text-to-image generation model from Alibaba. Unlike Qwen-Image-Edit, this model generates images from scratch based on text prompts without requiring an input image.

## Model Architecture

| Component | Size | Description |
|-----------|------|-------------|
| Transformer | 39GB (60 layers) | DiT with 3840 hidden dim, 24 heads @ 128 dim |
| Text Encoder | 16GB | Qwen2.5-VL-7B-Instruct (same as Qwen-Image-Edit) |
| VAE | 243MB | AutoencoderKLQwenImage |

**Total model size**: ~55GB (requires quantization for 24GB GPUs)

## VRAM Requirements

### RTX 4090 (24GB) - Recommended Configuration

```python
from llm_dit.pipelines import QwenImage2512Pipeline

pipe = QwenImage2512Pipeline.from_pretrained(
    "/path/to/Qwen-Image-2512",
    quantize_transformer="fp8",      # ~20GB -> essential for 24GB
    quantize_text_encoder="4bit",    # 16GB -> ~4GB
    cpu_offload=True,                # Sequential: text_encoder -> transformer -> vae
)
```

**Memory breakdown with recommended settings:**
- Transformer (FP8): ~20GB peak during inference
- Text encoder (4bit): ~4GB
- VAE: ~0.5GB
- With CPU offload, only one component is on GPU at a time

### Higher VRAM GPUs (48GB+)

```python
pipe = QwenImage2512Pipeline.from_pretrained(
    "/path/to/Qwen-Image-2512",
    quantize_transformer=None,       # Full bf16
    quantize_text_encoder=None,      # Full bf16
    cpu_offload=False,               # All on GPU
)
```

## Quick Start

```python
from llm_dit.pipelines import QwenImage2512Pipeline

# Load with recommended settings for RTX 4090
pipe = QwenImage2512Pipeline.from_pretrained(
    "~/Storage/Qwen-Image-2512",
    quantize_transformer="fp8",
    quantize_text_encoder="4bit",
    cpu_offload=True,
)

# Generate an image
image = pipe(
    prompt="A serene mountain lake at sunset, photorealistic",
    negative_prompt="blurry, low quality",
    height=1024,
    width=1024,
    num_inference_steps=40,
    cfg_scale=4.0,
    seed=42,
)

image.save("output.png")
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `prompt` | required | Text description of desired image |
| `negative_prompt` | " " | What to avoid in the image |
| `height` | 1024 | Output height (pixels) |
| `width` | 1024 | Output width (pixels) |
| `num_inference_steps` | 40 | Diffusion steps |
| `cfg_scale` | 4.0 | Classifier-free guidance scale |
| `seed` | None | Random seed for reproducibility |
| `max_sequence_length` | 512 | Maximum prompt tokens |

## Quantization Options

### Transformer Quantization

| Option | VRAM | Quality | Speed |
|--------|------|---------|-------|
| `None` | 39GB | Best | Baseline |
| `"fp8"` | ~20GB | Excellent | Slightly faster |
| `"int8"` | ~20GB | Very good | Faster |
| `"8bit"` | ~20GB | Good | Faster |
| `"4bit"` | ~10GB | Acceptable | Fastest |

### Text Encoder Quantization

| Option | VRAM | Quality |
|--------|------|---------|
| `None` | 16GB | Best |
| `"8bit"` | ~8GB | Very good |
| `"4bit"` | ~4GB | Good |

## Prompt Rewriting

Qwen-Image-2512 works best with detailed, descriptive prompts. The official Alibaba implementation uses a prompt rewriter powered by Qwen-Plus API. We document the system prompts here for local replication.

### Prompt Categories

The rewriter classifies prompts into three categories:
1. **Portrait** - Human-focused images
2. **Text-containing** - Images with visible text
3. **General** - Landscapes, still life, abstract

### English System Prompt

For local prompt rewriting, use the following system prompt with any capable LLM (e.g., Qwen3-4B):

```
# Image Prompt Rewriting Expert
You are a world-class expert in crafting image prompts, fluent in both Chinese and English, with exceptional visual comprehension and descriptive abilities.
Your task is to automatically classify the user's original image description into one of three categories: portrait, text-containing image, or general image, and then rewrite it naturally, precisely, and aesthetically in English.

## Core Requirements
1. Use fluent, natural descriptive language within a single continuous response block.
2. Enrich visual details appropriately - supplement environment, lighting, texture, or atmosphere when needed.
3. Never modify proper nouns: names, brands, locations, IPs, titles, URLs, phone numbers.
4. If the image contains text, enclose displayed text in double quotation marks (" ").
5. Clearly specify the overall artistic style (realistic photography, anime, 3D rendering, etc.)

## Portrait Images
Include: ethnicity, gender, age, facial features, expression, skin, makeup, clothing, hairstyle, accessories, pose, background, lighting.

## Text-containing Images
Faithfully reproduce all text content with position, layout, font style, color, size. Describe the relationship between text and its carrier.

## General Images
Cover: subject type, quantity, form, color, material, spatial layering, lighting, textures, scene atmosphere.

Output only the rewritten prompt text with no explanations.
```

### Chinese System Prompt

For Chinese prompts, the rewriter uses a parallel Chinese system prompt. The language is automatically detected based on the presence of CJK characters in the input.

## Comparison with Z-Image

| Aspect | Qwen-Image-2512 | Z-Image |
|--------|-----------------|---------|
| Text Encoder | Qwen2.5-VL-7B (3584 dim) | Qwen3-4B (2560 dim) |
| DiT Size | 60 layers, 39GB | 30 layers, ~8GB |
| Default Steps | 40 | 8-9 (turbo) |
| CFG | 4.0 (explicit) | 0.0 (baked in) |
| Max Tokens | 512 (configurable) | 1504 (RoPE limit) |
| Quantization | Required for 24GB | Optional |

## Reference Implementation

Based on:
- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) reference implementation
- [Qwen-Image official repo](https://github.com/QwenLM/Qwen-Image)
- diffusers `QwenImagePipeline` from coderef

## See Also

- [Qwen-Image-Edit-2511 Documentation](qwen_image_edit_2511.md) - For image editing
- [Prompt Rewriting Guide](../guides/prompt_rewriting.md) - General prompt expansion
