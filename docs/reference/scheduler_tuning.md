# scheduler tuning guide

*last updated: 2026-03-03*

This guide explains how to tune the FlowMatch scheduler parameters (`shift` and `d_noise`) for Z-Image generation. Understanding these parameters helps you control image sharpness, detail level, and overall aesthetic.

> **CLI note:** The examples below use `scripts/gen.py` (requires a running server). The `--shift` parameter is available as a gen.py flag. The `--d-noise` parameter is a server-side setting -- configure it in `config.toml` under `[default.scheduler]` or via the web UI. The deprecated `scripts/generate.py` supported both as CLI flags directly.

## Overview

The FlowMatch scheduler controls how noise is removed during generation. Two key parameters affect the denoising trajectory:

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `shift` | 3.0 | 0.5-10.0 | Controls sigma schedule compression |
| `d_noise` | 1.0 | 0.8-1.2 | Scales sigma values (detail/softness) |

## Shift Parameter

### What Shift Does

Shift (also called "mu") compresses the sigma schedule, affecting how aggressively the model denoises at each step.

**Formula:**
```
sigma' = shift * sigma / (1 + (shift - 1) * sigma)
```

Higher shift values compress the schedule, causing more denoising to happen in early steps.

### Shift Values and Their Effects

| Shift | Effect | Visual Result | Best For |
|-------|--------|---------------|----------|
| < 1.5 | Incomplete denoising | Blurry, unfinished look | Not recommended |
| 1.5-2.5 | Light denoising | Softer, more painterly | Artistic styles, dreams |
| **3.0** | **Default** | **Balanced detail/smoothness** | **General use** |
| 3.5-4.5 | Aggressive denoising | Sharper, more contrast | Detailed subjects |
| 5.0-6.0 | Very aggressive | High detail, potential artifacts | Technical images |
| > 6.0 | Over-aggressive | Artifacts likely | Not recommended |

### Shift Examples

```bash
# Softer, more artistic (shift 2.5)
uv run scripts/gen.py zimage --prompt "Watercolor painting of a garden" --shift 2.5

# Default balanced (shift 3.0)
uv run scripts/gen.py zimage --prompt "Portrait of a woman" --shift 3.0

# Sharper, more detail (shift 4.0)
uv run scripts/gen.py zimage --prompt "Architectural photograph of a building" --shift 4.0
```

### Visual Guide: Shift Effects

```
Shift 2.0          Shift 3.0          Shift 4.0          Shift 5.0
-----------        -----------        -----------        -----------
Softer edges       Balanced           Sharper edges      Very sharp
Less contrast      Natural look       More contrast      High contrast
Painterly feel     Photo-realistic    Crisp details      May clip highlights
Good for art       General purpose    Technical work     Use with caution
```

## D-Noise Parameter

### What D-Noise Does

D-noise scales the entire sigma schedule, effectively "lying" to the model about the noise level:

- **d_noise < 1.0**: Model thinks there's less noise than reality → works harder → sharper output
- **d_noise > 1.0**: Model thinks there's more noise than reality → works less → softer output

This is a technique from the RES4LYF/ClownSharkSampler research.

### D-Noise Values and Their Effects

| D-Noise | Effect | Visual Result | Best For |
|---------|--------|---------------|----------|
| 0.90-0.94 | Strong sharpening | Very crisp, possible artifacts | Rarely needed |
| 0.95-0.97 | Moderate sharpening | Enhanced detail, clean edges | Text, architecture |
| 0.98-0.99 | Subtle sharpening | Slightly crisper | Portraits, products |
| **1.00** | **Default** | **Model's trained output** | **General use** |
| 1.01-1.02 | Subtle softening | Slightly smoother | Skin, soft subjects |
| 1.03-1.05 | Moderate softening | Blended colors, dreamy | Artistic, atmospheric |
| 1.06-1.10 | Strong softening | Very soft, painterly | Abstract, impressionist |

### D-Noise Examples

Set `d_noise` in `config.toml` (not available as a gen.py flag):

```toml
# config.toml -- sharper details
[default.scheduler]
d_noise = 0.96

# config.toml -- default
[default.scheduler]
d_noise = 1.0

# config.toml -- softer, dreamier
[default.scheduler]
d_noise = 1.04
```

Then generate:

```bash
uv run scripts/gen.py zimage --prompt "Detailed mechanical watch"
uv run scripts/gen.py zimage --prompt "A cat sitting on a windowsill"
uv run scripts/gen.py zimage --prompt "Misty morning in a forest"
```

### Visual Guide: D-Noise Effects

```
d_noise 0.95       d_noise 1.00       d_noise 1.05
------------       ------------       ------------
Sharper edges      Balanced           Softer edges
More texture       Natural            Smoother gradients
Enhanced detail    Model default      Blended colors
Crisper text       Photo-like         Painterly feel
```

## Combining Shift and D-Noise

These parameters work together but affect different aspects of the generation:

- **Shift**: Controls the denoising schedule shape (when denoising happens)
- **D-Noise**: Scales the noise levels (how much denoising happens)

### Recommended Combinations

For combined tuning, set `d_noise` in `config.toml` and pass `--shift` via gen.py.

#### Maximum Sharpness (Technical/Detailed Subjects)

```toml
# config.toml
[default.scheduler]
d_noise = 0.96
```

```bash
uv run scripts/gen.py zimage --prompt "Macro photograph of a circuit board, sharp details" --shift 4.0
```

**Expect:** Very crisp edges, enhanced fine detail, high contrast. May introduce subtle artifacts on smooth gradients.

#### Balanced Sharp (Portraits, Products)

```toml
# config.toml
[default.scheduler]
d_noise = 0.98
```

```bash
uv run scripts/gen.py zimage --prompt "Professional headshot portrait" --shift 3.0
```

**Expect:** Natural look with slightly enhanced detail. Good skin texture without being harsh.

#### Default/Neutral

```toml
# config.toml
[default.scheduler]
d_noise = 1.0
```

```bash
uv run scripts/gen.py zimage --prompt "A landscape at sunset" --shift 3.0
```

**Expect:** The model's trained optimum. Balanced between detail and smoothness.

#### Soft/Artistic (Dreamy, Painterly)

```toml
# config.toml
[default.scheduler]
d_noise = 1.04
```

```bash
uv run scripts/gen.py zimage --prompt "Impressionist painting of a garden" --shift 2.8
```

**Expect:** Softer edges, blended colors, dreamy atmosphere. Less sharp detail but more cohesive mood.

#### Maximum Softness (Abstract, Atmospheric)

```toml
# config.toml
[default.scheduler]
d_noise = 1.06
```

```bash
uv run scripts/gen.py zimage --prompt "Abstract watercolor, flowing colors" --shift 2.5
```

**Expect:** Very soft, painterly quality. Minimal hard edges, smooth color transitions.

### Quick Reference Matrix

| Goal | Shift | D-Noise | Expected Output |
|------|-------|---------|-----------------|
| Technical/Mechanical | 4.0 | 0.95-0.97 | Crisp, detailed, high contrast |
| Architecture | 3.5-4.0 | 0.96-0.98 | Sharp lines, clear structure |
| Portraits | 3.0 | 0.98-1.02 | Natural skin, balanced detail |
| Landscapes | 3.0 | 1.0 | Natural, photo-realistic |
| Soft portraits | 3.0 | 1.02-1.03 | Smoother skin, flattering |
| Artistic/Painterly | 2.5-3.0 | 1.03-1.05 | Soft, blended, dreamy |
| Abstract | 2.5 | 1.05-1.08 | Very soft, flowing |

## High Resolution (2K+) Considerations

### Dynamic Shift

For high-resolution generation, `dynamic_shift` calculates shift based on resolution:

| Resolution | Dynamic Shift Value |
|------------|---------------------|
| 512x512 | ~0.5 |
| 1024x1024 | ~0.8 |
| 2048x2048 | ~1.15 |

**Important:** Dynamic shift values are much lower than the default (3.0). This is a different operating regime optimized for high-res.

### Recommendations by Resolution

#### 1024x1024 (Standard)

```toml
# config.toml
[default.scheduler]
d_noise = 0.98
```

```bash
uv run scripts/gen.py zimage --prompt "Your prompt" --shift 3.0 --width 1024 --height 1024
```

#### 2048x2048+ (High-Res with DyPE)

DyPE and d_noise are server-side config parameters. Set them in `config.toml`:

```toml
# config.toml
[default.scheduler]
d_noise = 0.97

[default.dype]
enabled = true
method = "vision_yarn"
```

```bash
uv run scripts/gen.py zimage --prompt "Your prompt" --width 2048 --height 2048
```

DyPE has its own shift parameters (`base_shift`, `max_shift`) that handle the resolution scaling internally.

#### 2048x2048+ (Experimental: Dynamic Shift Only)

```toml
# config.toml
[default.scheduler]
d_noise = 0.98
dynamic_shift = true
```

```bash
uv run scripts/gen.py zimage --prompt "Your prompt" --width 2048 --height 2048
```

## Interaction with Other Parameters

### Steps

More steps give finer control but Z-Image is optimized for 8-9 steps:

| Steps | Shift | D-Noise | Notes |
|-------|-------|---------|-------|
| 8-9 | 3.0 | 1.0 | Optimal for turbo model |
| 6-7 | 3.5 | 0.98 | Compensate with higher shift |

### CFG Scale

Z-Image has CFG baked in (guidance_scale=0.0), but if using CFG:

- Higher CFG + lower d_noise = very sharp, potential artifacts
- Higher CFG + higher d_noise = balanced sharpening

### SLG (Skip Layer Guidance)

SLG adds detail without changing the sigma schedule. Can combine with d_noise. Both are server-side config parameters:

```toml
# config.toml
[default.scheduler]
d_noise = 0.98

[default.slg]
scale = 2.5
```

```bash
uv run scripts/gen.py zimage --prompt "Detailed subject"
```

## Troubleshooting

### Image Too Sharp/Artifacts

- Decrease shift (try 2.8-3.0)
- Increase d_noise (try 1.02-1.03)

### Image Too Soft/Blurry

- Increase shift (try 3.5-4.0)
- Decrease d_noise (try 0.97-0.98)

### Inconsistent Results

- Start from defaults (shift=3.0, d_noise=1.0)
- Adjust one parameter at a time
- Use fixed seed for A/B testing

### High-Res Looks Different

- High-res needs different settings than 1024x1024
- Use DyPE for 2K+ (it handles shift internally)
- Slightly sharper d_noise (0.97-0.99) helps with upscaled detail

## Config File Examples

### config.toml - Sharp Profile

```toml
[sharp.scheduler]
shift = 3.5
d_noise = 0.97
dynamic_shift = false
```

### config.toml - Soft Profile

```toml
[soft.scheduler]
shift = 2.8
d_noise = 1.04
dynamic_shift = false
```

### config.toml - High-Res Profile

```toml
[highres.scheduler]
shift = 3.0
d_noise = 0.98
dynamic_shift = false  # Use DyPE instead

[highres.dype]
enabled = true
method = "vision_yarn"
dype_scale = 2.0
```

## Summary

1. **Start with defaults** (shift=3.0, d_noise=1.0) for the model's trained optimum
2. **Adjust shift** for overall denoising aggressiveness (2.5-4.5 safe range)
3. **Adjust d_noise** for texture/sharpness control (0.95-1.05 safe range)
4. **Use DyPE** for high-resolution, not dynamic_shift
5. **Test with fixed seeds** to isolate parameter effects
