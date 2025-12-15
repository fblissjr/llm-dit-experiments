# VL-4B Model Variant Comparison: Instruct vs Thinking

last updated: 2025-12-15

## Overview

This document compares Qwen3-VL-4B-Instruct vs Qwen3-VL-4B-Thinking for Z-Image vision conditioning to determine if the "thinking" training objective produces better embeddings.

## Hypothesis

The VL-4B-Thinking model might produce better embeddings because:
1. **Training objective**: Thinking models preserve information for multi-step reasoning
2. **Layer preservation**: SFT for thinking may not overwrite visual concepts as aggressively
3. **Native think blocks**: Template natively supports `<think>` tokens

## Architecture Verification

Both models are architecturally identical and compatible with Z-Image:

| Parameter | VL-4B-Instruct | VL-4B-Thinking |
|-----------|----------------|----------------|
| hidden_size | 2560 | 2560 |
| num_hidden_layers | 36 | 36 |
| vision encoder depth | 24 | 24 |
| vision out_hidden_size | 2560 | 2560 |
| enable_thinking template | No | **Yes** |

## Chat Template Difference

The key functional difference is in the chat template:

**Instruct model** (with `add_generation_prompt=True`):
```
<|im_start|>assistant\n
```

**Thinking model** (with `add_generation_prompt=True`):
```
<|im_start|>assistant\n<think>\n
```

The Thinking model's template automatically adds the `<think>` block - no manual injection needed.

## Experiment Configuration

**Sweep parameters:**
- Model variants: Instruct, Thinking
- Alphas: 0.3, 0.5, 1.0
- Hidden layers: -2, -6, -8
- Token modes: text_only, full (with image tokens)
- Blend mode: adain_per_dim
- Steps: 4 (turbo mode)

**Prompts tested:**
1. "Homer Simpson eating spaghetti"
2. "A cartoon house with a red roof"

**Reference image:** `experiments/inputs/style_anime_girl.png` (anime girl)

**Total images:** 76 (18 per model variant per prompt + comparison grids)

## Results

### Observations

**Visual Quality:**
Both model variants produce similar quality results. At matched settings (same layer, alpha, token mode), the outputs are nearly indistinguishable.

**Prompt Adherence:**
- Both models correctly generate "Homer Simpson eating spaghetti"
- Both models correctly generate "cartoon house with red roof"
- No significant difference in prompt following ability

**Style Transfer:**
- Both models transfer anime style from reference at similar rates
- Higher alpha (0.5, 1.0) produces stronger style influence
- Layer -6 and -8 produce cleaner results than -2

**Token Mode Impact:**
- `text_only`: Uses only text tokens from VL embedding, cleaner but less style
- `full`: Includes image tokens, stronger style transfer

### Key Metrics

From metadata analysis:

| Model | Token Mode | Typical VL Std | Blended Std |
|-------|------------|----------------|-------------|
| Instruct | text_only | ~12.4 | ~61 |
| Instruct | full | ~12.4 | ~258-390 |
| Thinking | text_only | ~12.4 | ~61 |
| Thinking | full | ~12.4 | ~258-390 |

The embedding statistics are essentially identical between model variants.

## Conclusion

**No significant difference observed** between Instruct and Thinking model variants for VL conditioning. Both produce:
- Similar visual quality
- Similar prompt adherence
- Similar style transfer characteristics
- Nearly identical embedding statistics

The hypothesis that Thinking models might preserve more visual information in later layers was not supported by this experiment. The VL fine-tuning appears to have similar effects on both model variants.

## Recommendations

1. **Use either model** - Both work equally well for VL conditioning
2. **Prefer Thinking model** if you need native `<think>` block support (eliminates manual injection)
3. **Use layer -6** for best results (cleaner than -2, well-tested)
4. **Stick with txt2img** - VL conditioning works well without img2img

## Files

Results directory: `experiments/results/vl_thinking_comparison_20251215_140437/`

```
Homer_Simpson_eating_spaghetti/
  instruct/comparison_grid.png
  thinking/comparison_grid.png
A_cartoon_house_with_a_red_roof/
  instruct/comparison_grid.png
  thinking/comparison_grid.png
```

## Future Work

1. Test with more diverse prompts
2. Compare embedding correlation with Qwen3-4B reference statistics
3. Test at higher step counts (9 steps vs 4)
4. Investigate if thinking models behave differently at extreme alphas
