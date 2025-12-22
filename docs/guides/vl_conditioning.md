# vision conditioning guide (qwen3-vl)

*last updated: 2025-12-22*

**Status: EXPERIMENTAL** - VL conditioning works but produces visible artifacts compared to pure text.

Zero-shot vision conditioning using Qwen3-VL embeddings. A reference image influences the generated output's style/content without training.

## why it works (in theory)

Qwen3-VL-4B's text model shares architecture with Qwen3-4B:
- Both have `hidden_size=2560` (matching Z-Image's expected embedding dimension)
- Qwen3-VL projects vision features into the same embedding space via `PatchMerger.linear_fc2`
- Interpolating VL + text embeddings should produce coherent conditioning

## correct configuration (critical)

**Previous documentation was WRONG.** The correct settings are:

| Setting | Wrong | Correct |
|---------|-------|---------|
| `text_tokens_only` | True | **False** |
| `hidden_layer` | -2 | **-6** |

**Why `text_tokens_only=False` is critical:**
- Image info is in IMAGE TOKEN positions (~1026 tokens for 1024x1024)
- `text_tokens_only=True` discards all image tokens
- With True, VL embeddings = pure text embeddings (VL does nothing)

## python usage

```python
vl_result = vl_extractor.extract(
    image,
    text=prompt,
    hidden_layer=-6,          # NOT -2 (has glitch artifacts)
    text_tokens_only=False,   # MUST be False to include image info!
    scale_to_text=True
)
blended = blend_embeddings(vl_result.embeddings, text_emb, alpha=0.5)
```

## alpha effects

| Alpha | Effect |
|-------|--------|
| 0.3 | Slight style influence |
| 0.5 | Strong style transfer (photorealistic from photo ref) |
| 0.7 | Reference strongly influences output |
| 1.0 | Reconstructs reference scene |

## toml config

```toml
[rtx4090.vl]
model_path = "/path/to/Qwen3-VL-4B-Instruct"
device = "cpu"                  # Recommended to save VRAM
default_alpha = 0.3             # Start conservative
default_hidden_layer = -6       # Layer -6 produces cleaner results than -2
text_tokens_only = false        # MUST be false for VL to work!
auto_unload = true              # Unload after extraction
```

## cli flags

```bash
uv run web/server.py \
  --model-path /path/to/z-image \
  --vl-model-path /path/to/Qwen3-VL-4B-Instruct \
  --vl-device cpu \
  --vl-hidden-layer -6 \
  --vl-alpha 0.3
```

## web ui usage

1. Open "Vision Conditioning (Qwen3-VL)" section
2. Upload a reference image
3. Use settings: layer -6, **text_tokens_only=False** (critical!)
4. Adjust alpha: 0.3-0.5 for style influence, higher for scene reconstruction

## memory management

Recommended VL extraction workflow:
1. Load Qwen3-VL on CPU
2. Extract embeddings from reference image
3. Cache embeddings (keyed by image hash)
4. Unload Qwen3-VL
5. Generate using cached embeddings

This keeps VRAM free for the DiT.

## research findings (2025-12-12)

**Layer Selection:**
- Layer -2 (Z-Image default) loses text prompt content with VL conditioning
- Layer -6 produces crisper images and preserves text prompt
- VL fine-tuning overwrote later layers for vision tasks, making middle layers better

**Core Insight:**
The embedding space doesn't cleanly separate "style" from "content". VL influence strong enough to transfer style also corrupts semantic content.

**What FAILED:**
- Style Delta: Even at alpha 0.3, destroys content
- AdaIN per_dim: Transfers colors but corrupts subject

**What WORKS:**
- Linear blending with layer -6, text_tokens_only=false
- Conservative alpha (0.3-0.5) for style influence

## blending modes

| Mode | Description | Result |
|------|-------------|--------|
| `linear` | Simple interpolation | Default, works best |
| `adain` | Per-token normalization | Preserves content, no visible style |
| `adain_per_dim` | Per-dimension normalization | Strong style, corrupts content |
| `style_delta` | Arithmetic style transfer | FAILED - destroys content |

## outlier dimension masking

VL image tokens have extreme per-dimension outliers at layer -2:
- **Dimension 396:** 617x std ratio
- **Dimension 4:** 42x std ratio
- **Layer -6:** NO outliers (naturally cleaner)

Three masking modes in `mask_outlier_dimensions()`:
- `zero`: Zero out outlier dimensions entirely
- `clamp`: Scale to threshold level
- `scale`: Proportionally reduce values

## experiment runner

```bash
cd experiments/qwen3_vl/scripts

# Run style transfer test
uv run run_comparison.py \
    -i experiments/inputs/style_anime_girl.png \
    -p "Homer Simpson" \
    --sweep style_transfer \
    --steps 4 \
    -o experiments/results/style_test

# Run blend mode comparison
uv run run_comparison.py \
    -i experiments/inputs/style_anime_girl.png \
    -p "Your prompt" \
    --sweep blend_comparison \
    --steps 4
```

**Profile Configuration:**
- Use `default` profile for VL experiments (`interpolate` long_prompt_mode)
- `rtx4090` profile uses `attention_pool` which can cause issues with VL

## files

| File | Description |
|------|-------------|
| `src/llm_dit/vl/qwen3_vl.py` | VLEmbeddingExtractor class |
| `src/llm_dit/vl/blending.py` | Blending utilities |
| `experiments/qwen3_vl/scripts/run_comparison.py` | Experiment runner |

## future research directions

1. **Trained minimal adapter**: Single linear layer to bridge VL and text space
2. **Layer interpolation**: Blend multiple hidden layers
3. **Attention-based selection**: Use attention weights to select important tokens
4. **IP-Adapter comparison**: Measure quality gap vs trained methods

## related research

- `internal/research/qwen3_vl_integration.md` - Integration details
- `internal/research/vl_conditioning_hypotheses.md` - Theory and hypotheses
- `experiments/qwen3_vl/README.md` - Experiment-specific details
