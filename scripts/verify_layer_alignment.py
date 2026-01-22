#!/usr/bin/env python3
"""
Verify layer alignment between Gemma hidden states and text_proj_in weights.

Last Updated: 2026-01-20

HYPOTHESIS:
If our Gemma hidden states don't correlate with the projection weights,
the linear output will be dominated by bias (y ≈ b = -9.4), causing
GELU(-9.4) ≈ 0 → signal death → "blurry blob" output.

TEST APPROACH:
1. Load text_proj_in.weight [3840, 188160]
2. Load caption_projection weights (DiT side)
3. Run Gemma on a test prompt, get 49 hidden states
4. Check correlation patterns to verify alignment

FLATTENING GEOMETRY (verified in code audit):
- Input shape: [B, T, D, L] where D=3840, L=49
- After reshape: [B, T, D*L] = [B, T, 188160]
- Layout is DIMENSION-MAJOR: indices 0-48 = dim 0 across all layers
  - weight[:, 0:49] reads from dimension 0 of all 49 layers
  - weight[:, 49:98] reads from dimension 1 of all 49 layers

SMOKING GUN TEST:
If layer stacking is wrong, certain weight chunks will show zero correlation
with the hidden states they're supposed to read from.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from safetensors import safe_open

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_projection_weights(connectors_path: str) -> dict:
    """Load text_proj_in weights from connectors checkpoint."""
    weights = {}
    with safe_open(connectors_path, framework="pt") as f:
        for key in f.keys():
            if "text_proj_in" in key:
                weights[key] = f.get_tensor(key)
                logger.info(f"Loaded {key}: shape={weights[key].shape}")
    return weights


def load_caption_projection_weights(dit_path: str) -> dict:
    """Load caption_projection weights from DiT checkpoint."""
    weights = {}
    with safe_open(dit_path, framework="pt") as f:
        for key in f.keys():
            if "caption_projection" in key:
                weights[key] = f.get_tensor(key)
                logger.info(f"Loaded {key}: shape={weights[key].shape}")
    return weights


def analyze_weight_structure(weight: torch.Tensor, num_layers: int = 49, hidden_dim: int = 3840):
    """
    Analyze the structure of text_proj_in.weight.

    Weight shape: [output_dim, input_dim] = [3840, 188160]

    If input is dimension-major (verified in code audit):
    - Columns 0:49 correspond to dim 0 across all 49 layers
    - Columns 49:98 correspond to dim 1 across all 49 layers
    - etc.

    We can check which "layer chunks" have the most weight magnitude,
    which tells us which layers the model learned to attend to.
    """
    logger.info("\n" + "=" * 70)
    logger.info("WEIGHT STRUCTURE ANALYSIS")
    logger.info("=" * 70)

    output_dim, input_dim = weight.shape
    expected_input = hidden_dim * num_layers

    logger.info(f"Weight shape: [{output_dim}, {input_dim}]")
    logger.info(f"Expected input dim: {hidden_dim} × {num_layers} = {expected_input}")

    if input_dim != expected_input:
        logger.error(f"MISMATCH: Input dim {input_dim} != expected {expected_input}")
        return

    # Convert to float32 for analysis
    weight = weight.float()

    # Method 1: Per-layer importance (sum across dimensions)
    # Reshape weight to [output_dim, hidden_dim, num_layers]
    # Wait, the flattening is dimension-major, so columns are organized as:
    # [dim0_layer0, dim0_layer1, ..., dim0_layer48, dim1_layer0, ...]
    # So we need to reshape as [output_dim, num_layers, hidden_dim] to group by layer

    # Actually, let's think about this more carefully.
    # Input tensor after reshape: [B, T, D*L] where D=3840, L=49
    # The reshape from [B, T, D, L] puts dimension d before layer l:
    # flattened index = d * L + l
    # So index 0 = dim0_layer0, index 1 = dim0_layer1, ..., index 48 = dim0_layer48
    # index 49 = dim1_layer0, etc.

    # To analyze per-layer importance, we want to sum over all dimensions for each layer
    # Weight columns for layer l: indices l, l+49, l+98, ..., l+49*3839

    logger.info("\n--- Per-Layer Weight Magnitude ---")
    logger.info("(Higher magnitude = model learned this layer matters more)")

    layer_magnitudes = []
    for layer_idx in range(num_layers):
        # Get all columns that read from this layer
        # These are indices: layer_idx, layer_idx + num_layers, layer_idx + 2*num_layers, ...
        layer_cols = torch.arange(layer_idx, input_dim, num_layers)
        layer_weights = weight[:, layer_cols]  # [output_dim, hidden_dim]

        # Compute L2 norm across all weights for this layer
        layer_mag = layer_weights.norm().item()
        layer_magnitudes.append(layer_mag)

    # Print top and bottom layers
    sorted_indices = sorted(range(num_layers), key=lambda x: layer_magnitudes[x], reverse=True)

    logger.info(f"\nTop 10 layers by weight magnitude:")
    for idx in sorted_indices[:10]:
        logger.info(f"  Layer {idx:2d}: {layer_magnitudes[idx]:.4f}")

    logger.info(f"\nBottom 10 layers by weight magnitude:")
    for idx in sorted_indices[-10:]:
        logger.info(f"  Layer {idx:2d}: {layer_magnitudes[idx]:.4f}")

    # Statistics
    mags = torch.tensor(layer_magnitudes)
    logger.info(f"\nLayer magnitude statistics:")
    logger.info(f"  Mean: {mags.mean():.4f}")
    logger.info(f"  Std:  {mags.std():.4f}")
    logger.info(f"  Min:  {mags.min():.4f} (Layer {mags.argmin().item()})")
    logger.info(f"  Max:  {mags.max():.4f} (Layer {mags.argmax().item()})")

    return layer_magnitudes


def run_gemma_and_check_correlation(
    connectors_path: str,
    model_id: str = "models/LTX-2/text_encoder",
    test_prompt: str = "A fluffy orange cat sleeping peacefully on a soft red couch.",
):
    """
    Run Gemma on a test prompt and check correlation with projection weights.
    """
    logger.info("\n" + "=" * 70)
    logger.info("GEMMA HIDDEN STATE CORRELATION CHECK")
    logger.info("=" * 70)

    # Load Gemma
    logger.info(f"\nLoading Gemma from {model_id}...")
    from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Tokenize
    encoded = tokenizer(
        test_prompt,
        padding="max_length",
        max_length=256,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoded.input_ids.to(model.device)
    attention_mask = encoded.attention_mask.to(model.device)

    logger.info(f"Prompt: '{test_prompt}'")
    logger.info(f"Token count: {attention_mask.sum().item()}")

    # Forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

    hidden_states = outputs.hidden_states
    num_layers = len(hidden_states)

    logger.info(f"\nHidden states: {num_layers} layers")
    logger.info(f"Each layer shape: {hidden_states[0].shape}")

    # Stack to [B, T, D, L]
    stacked = torch.stack(hidden_states[:49], dim=-1)
    logger.info(f"Stacked shape: {stacked.shape}")

    # Analyze per-layer statistics
    logger.info("\n--- Per-Layer Hidden State Statistics ---")
    for i in [0, 1, 24, 47, 48]:
        if i < num_layers:
            hs = hidden_states[i].float()
            valid_mask = attention_mask[0].bool()
            valid_hs = hs[0, valid_mask]
            logger.info(
                f"  Layer {i:2d}: mean={valid_hs.mean():.4f}, std={valid_hs.std():.4f}, "
                f"min={valid_hs.min():.4f}, max={valid_hs.max():.4f}"
            )

    # Move stacked to CPU and free GPU memory
    stacked = stacked.cpu()
    attention_mask = attention_mask.cpu()
    del model, tokenizer
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    # Load projection weights (keep on CPU)
    with safe_open(connectors_path, framework="pt") as f:
        proj_weight = f.get_tensor("text_proj_in.weight").float()  # [3840, 188160]

    logger.info(f"\nProjection weight shape: {proj_weight.shape}")

    # Test: Apply projection manually and check output
    # First, normalize and flatten the stacked hidden states
    b, t, d, l = stacked.shape

    # Simple flatten for testing (skip normalization for raw correlation)
    flat_raw = stacked.reshape(b, t, -1).float()  # [1, 256, 188160]

    # Apply projection
    proj_output_raw = torch.matmul(flat_raw, proj_weight.t())  # [1, 256, 3840]

    valid_mask = attention_mask[0].bool()
    valid_output = proj_output_raw[0, valid_mask]

    logger.info(f"\nRaw projection output (no normalization):")
    logger.info(f"  Shape: {valid_output.shape}")
    logger.info(f"  Mean:  {valid_output.mean():.4f}")
    logger.info(f"  Std:   {valid_output.std():.4f}")
    logger.info(f"  Min:   {valid_output.min():.4f}")
    logger.info(f"  Max:   {valid_output.max():.4f}")

    # Now with normalization (matching _norm_and_concat_layers)
    logger.info("\n--- With Normalization (matching reference) ---")

    # Per-layer normalization: 8 * (x - mean) / (range + eps)
    eps = 1e-6
    mask_expanded = attention_mask[:, :, None, None].bool()  # [B, T, 1, 1]
    seq_lengths = attention_mask.sum(dim=1)  # [B]

    masked_states = stacked.float().masked_fill(~mask_expanded, 0.0)
    denom = (seq_lengths * d).view(b, 1, 1, 1)
    mean = masked_states.sum(dim=(1, 2), keepdim=True) / (denom + eps)

    x_min = stacked.float().masked_fill(~mask_expanded, float("inf")).amin(dim=(1, 2), keepdim=True)
    x_max = (
        stacked.float().masked_fill(~mask_expanded, float("-inf")).amax(dim=(1, 2), keepdim=True)
    )
    range_val = x_max - x_min

    normed = 8.0 * (stacked.float() - mean) / (range_val + eps)
    normed_flat = normed.reshape(b, t, -1)

    # Zero out padding
    mask_flat = attention_mask[:, :, None].bool().expand(-1, -1, d * l)
    normed_flat = normed_flat.masked_fill(~mask_flat, 0.0)

    logger.info(f"Normalized input:")
    logger.info(f"  Mean: {normed_flat[mask_flat].mean():.4f}")
    logger.info(f"  Std:  {normed_flat[mask_flat].std():.4f}")

    # Apply projection
    proj_output_normed = torch.matmul(normed_flat, proj_weight.t())  # [1, 256, 3840]
    valid_normed_output = proj_output_normed[0, valid_mask]

    logger.info(f"\nNormalized projection output:")
    logger.info(f"  Mean:  {valid_normed_output.mean():.4f}")
    logger.info(f"  Std:   {valid_normed_output.std():.4f}")
    logger.info(f"  Min:   {valid_normed_output.min():.4f}")
    logger.info(f"  Max:   {valid_normed_output.max():.4f}")

    # KEY DIAGNOSTIC: Per-layer correlation test
    logger.info("\n" + "=" * 70)
    logger.info("CORRELATION TEST: Does layer order matter?")
    logger.info("=" * 70)

    # Test 1: Normal order (layer 0 first)
    normal_output = proj_output_normed[0, valid_mask].mean().item()

    # Test 2: Reversed order (layer 48 first)
    reversed_stack = torch.flip(stacked, dims=[-1])
    reversed_normed = 8.0 * (reversed_stack.float() - mean) / (range_val + eps)
    reversed_flat = reversed_normed.reshape(b, t, -1)
    reversed_flat = reversed_flat.masked_fill(~mask_flat, 0.0)
    reversed_output = torch.matmul(reversed_flat, proj_weight.t())
    reversed_mean = reversed_output[0, valid_mask].mean().item()

    logger.info(f"\nLayer order comparison:")
    logger.info(f"  Normal order (0→48) mean output:   {normal_output:.4f}")
    logger.info(f"  Reversed order (48→0) mean output: {reversed_mean:.4f}")

    if abs(reversed_mean - normal_output) > 0.1:
        logger.info(f"  ⚠️  Significant difference detected! Order matters.")
    else:
        logger.info(f"  ✓ Similar outputs - layer order is probably correct")

    # Test 3: Random shuffle
    perm = torch.randperm(49)
    shuffled_stack = stacked[..., perm]
    shuffled_normed = 8.0 * (shuffled_stack.float() - mean) / (range_val + eps)
    shuffled_flat = shuffled_normed.reshape(b, t, -1)
    shuffled_flat = shuffled_flat.masked_fill(~mask_flat, 0.0)
    shuffled_output = torch.matmul(shuffled_flat, proj_weight.t())
    shuffled_mean = shuffled_output[0, valid_mask].mean().item()

    logger.info(f"  Shuffled order mean output:        {shuffled_mean:.4f}")

    return {
        "normal_mean": normal_output,
        "reversed_mean": reversed_mean,
        "shuffled_mean": shuffled_mean,
    }


def analyze_caption_projection(dit_path: str):
    """Analyze the caption_projection in the DiT (where -9.4 bias comes from)."""
    logger.info("\n" + "=" * 70)
    logger.info("CAPTION PROJECTION ANALYSIS (DiT Side)")
    logger.info("=" * 70)

    weights = {}
    with safe_open(dit_path, framework="pt") as f:
        for key in f.keys():
            if "caption_projection" in key:
                weights[key] = f.get_tensor(key).float()

    if not weights:
        logger.warning("No caption_projection weights found in DiT checkpoint")
        return

    for name, w in weights.items():
        logger.info(f"\n{name}:")
        logger.info(f"  Shape: {w.shape}")
        logger.info(f"  Mean:  {w.mean():.6f}")
        logger.info(f"  Std:   {w.std():.6f}")

        if "bias" in name.lower() or w.dim() == 1:
            logger.info(f"  Min:   {w.min():.4f}")
            logger.info(f"  Max:   {w.max():.4f}")
            logger.info(f"  🎯 This is likely the -9.4 bias causing GELU death!")

            # Check what percentage of values would kill GELU
            threshold = -5.0  # GELU(-5) ≈ 0
            dead_ratio = (w < threshold).float().mean().item()
            logger.info(f"  % below {threshold}: {dead_ratio * 100:.1f}% (would cause GELU ≈ 0)")


def main():
    parser = argparse.ArgumentParser(description="Verify Gemma layer alignment")
    parser.add_argument(
        "--connectors-path",
        type=str,
        default="models/LTX-2/connectors/diffusion_pytorch_model.safetensors",
        help="Path to connectors checkpoint",
    )
    parser.add_argument(
        "--dit-path",
        type=str,
        default="models/LTX-2/transformer/diffusion_pytorch_model.safetensors",
        help="Path to DiT transformer checkpoint",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="models/LTX-2/text_encoder",
        help="Gemma model ID or path",
    )
    parser.add_argument(
        "--skip-gemma",
        action="store_true",
        help="Skip Gemma loading (just analyze weights)",
    )
    args = parser.parse_args()

    # Analyze connector weights
    logger.info("Loading connector weights...")
    proj_weights = load_projection_weights(args.connectors_path)

    if "text_proj_in.weight" in proj_weights:
        analyze_weight_structure(proj_weights["text_proj_in.weight"])

    # Analyze DiT caption projection
    if Path(args.dit_path).exists():
        analyze_caption_projection(args.dit_path)
    else:
        logger.warning(f"DiT checkpoint not found at {args.dit_path}")

    # Run Gemma correlation check
    if not args.skip_gemma:
        run_gemma_and_check_correlation(
            args.connectors_path,
            args.model_id,
        )
    else:
        logger.info("\n(Skipping Gemma loading as requested)")


if __name__ == "__main__":
    main()
