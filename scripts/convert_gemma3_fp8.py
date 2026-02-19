#!/usr/bin/env python3
"""
Convert Gemma3 encoder weights to fp8 safetensors for faster loading.

Last Updated: 2026-02-17

Pre-converts Gemma3 linear layer weights from bf16 to float8_e4m3fn and saves
as a single safetensors file. This eliminates the bf16 load + fp8 conversion
overhead at runtime (~40s saved on first generation).

Non-linear layers (norms, embeddings) remain in bf16 for numerical stability.

Usage:
    # Convert LTX-2's bundled Gemma3 encoder
    uv run python scripts/convert_gemma3_fp8.py models/LTX-2

    # Convert with custom text encoder path
    uv run python scripts/convert_gemma3_fp8.py models/LTX-2 --encoder-path ~/Storage/gemma-3-12b

    # Custom output path
    uv run python scripts/convert_gemma3_fp8.py models/LTX-2 -o models/LTX-2/text_encoder_fp8.safetensors

    # Dry run (show what would be converted, don't save)
    uv run python scripts/convert_gemma3_fp8.py models/LTX-2 --dry-run
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import save_file as save_safetensors
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Patterns that should stay in bf16 for numerical stability
SKIP_PATTERNS = ("norm", "embed", "lm_head")


def convert_gemma3_to_fp8(
    model_path: str,
    encoder_path: str | None = None,
    output_path: str | None = None,
    dry_run: bool = False,
) -> Path | None:
    """Convert Gemma3 encoder weights to fp8 safetensors.

    Steps:
    1. Load Gemma3 via the existing Gemma3Encoder loader (handles LTX-2's
       sharded format + key remapping automatically)
    2. Walk all nn.Linear modules and convert weights to float8_e4m3fn
    3. Save the entire state_dict (fp8 linears + bf16 norms/embeds) as
       a single safetensors file

    Args:
        model_path: Path to LTX-2 model directory (e.g., "models/LTX-2").
        encoder_path: Override path for Gemma3 weights. Defaults to
            model_path/text_encoder/.
        output_path: Where to save the fp8 safetensors. Defaults to
            model_path/text_encoder_fp8.safetensors.
        dry_run: If True, show conversion stats without saving.

    Returns:
        Path to the saved fp8 safetensors file, or None if dry_run.
    """
    model_path_obj = Path(model_path).expanduser()
    if not model_path_obj.exists():
        logger.error(f"Model path not found: {model_path_obj}")
        sys.exit(1)

    # Resolve encoder path
    if encoder_path:
        enc_path = str(Path(encoder_path).expanduser())
    else:
        enc_path = str(model_path_obj / "text_encoder")

    if not Path(enc_path).exists():
        logger.error(f"Encoder path not found: {enc_path}")
        sys.exit(1)

    # Resolve output path
    if output_path:
        out_path = Path(output_path).expanduser()
    else:
        out_path = model_path_obj / "text_encoder_fp8.safetensors"

    if out_path.exists() and not dry_run:
        logger.error(f"Output already exists: {out_path}")
        logger.error("Delete it first or use -o to specify a different path.")
        sys.exit(1)

    overall_start = time.time()

    # Step 1: Load the encoder on CPU in bf16
    logger.info("Loading Gemma3 encoder on CPU in bf16...")
    logger.info(f"  Encoder path: {enc_path}")
    start = time.time()

    from llm_dit.encoders.gemma3 import Gemma3Encoder

    encoder = Gemma3Encoder(
        model_id=enc_path,
        device="cpu",
        dtype=torch.bfloat16,
        tokenizer_path=str(model_path_obj / "tokenizer"),
        connectors_path=str(
            model_path_obj / "text_encoder"
            / "diffusion_pytorch_model-00011-of-00012.safetensors"
        ),
    )
    encoder._load_model()
    load_time = time.time() - start
    logger.info(f"  Loaded in {load_time:.1f}s")

    if encoder._model is None:
        logger.error("Model failed to load (encoder._model is None)")
        sys.exit(1)

    # Step 2: Convert linear weights to fp8
    model = encoder._model
    state_dict: dict[str, torch.Tensor] = {}

    # Pre-scan: count total params and linear layers for progress bar
    total_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()

    linear_modules = []
    skip_modules = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if any(pat in name.lower() for pat in SKIP_PATTERNS):
            skip_modules.append(name)
        else:
            linear_modules.append((name, module))

    logger.info(
        f"Converting {len(linear_modules)} linear layers to float8_e4m3fn "
        f"(skipping {len(skip_modules)} norm/embed layers)..."
    )

    converted = 0
    fp8_params = 0
    convert_start = time.time()

    for name, module in tqdm(
        linear_modules,
        desc="  bf16 -> fp8",
        unit="layer",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    ):
        module.weight.data = module.weight.data.to(torch.float8_e4m3fn)
        converted += 1
        fp8_params += module.weight.numel()

    convert_time = time.time() - convert_start
    skipped = len(skip_modules)

    bf16_params = total_params - fp8_params
    logger.info(f"  Converted: {converted} linear layers in {convert_time:.1f}s")
    logger.info(f"  Skipped:   {skipped} layers (norms/embeddings)")
    logger.info(
        f"  Params:    {total_params / 1e9:.2f}B total "
        f"({fp8_params / 1e9:.2f}B fp8, {bf16_params / 1e9:.2f}B bf16)"
    )

    # Estimate sizes
    fp8_bytes = fp8_params * 1  # 1 byte per fp8
    bf16_bytes = bf16_params * 2  # 2 bytes per bf16
    original_bytes = total_params * 2  # all bf16
    total_bytes = fp8_bytes + bf16_bytes
    savings = 1.0 - (total_bytes / original_bytes)
    logger.info(
        f"  Size:      {original_bytes / 1e9:.1f}GB (bf16) -> "
        f"{total_bytes / 1e9:.1f}GB (mixed fp8/bf16) "
        f"({savings:.0%} savings)"
    )

    if dry_run:
        logger.info("[DRY RUN] Would save to: %s", out_path)
        return None

    # Step 3: Build state dict and save
    # Get full state dict (weights are already fp8 where converted)
    # Exclude lm_head.weight -- it's tied to embed_tokens.weight (same tensor).
    # safetensors rejects shared memory. The loader detects the missing key
    # and calls model.tie_weights() to restore the tie.
    logger.info("Building state dict...")
    build_start = time.time()

    raw_state = model.state_dict()
    total_keys = len(raw_state)
    state_dict = {}
    for name, param in tqdm(
        raw_state.items(),
        desc="  model",
        unit="key",
        total=total_keys,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    ):
        if name == "lm_head.weight":
            continue
        # safetensors requires contiguous tensors
        state_dict[name] = param.contiguous()

    # Also save feature extractor and connector weights
    if encoder._feature_extractor is not None:
        fe_state = encoder._feature_extractor.state_dict()
        logger.info(f"  Adding {len(fe_state)} feature extractor keys")
        for name, param in fe_state.items():
            state_dict[f"feature_extractor.{name}"] = param.contiguous()

    if encoder._embeddings_connector is not None:
        conn_state = encoder._embeddings_connector.state_dict()
        logger.info(f"  Adding {len(conn_state)} embeddings connector keys")
        for name, param in conn_state.items():
            state_dict[f"embeddings_connector.{name}"] = param.contiguous()

    build_time = time.time() - build_start
    logger.info(f"  {len(state_dict)} total keys assembled in {build_time:.1f}s")

    logger.info(f"Writing {out_path}...")
    write_start = time.time()
    save_safetensors(state_dict, str(out_path))
    write_time = time.time() - write_start

    file_size = out_path.stat().st_size / 1e9
    logger.info(f"  Written in {write_time:.1f}s ({file_size:.2f}GB)")
    logger.info(f"Output: {out_path}")

    # Summary
    overall_time = time.time() - overall_start
    logger.info("")
    logger.info(f"Done in {overall_time:.1f}s (load: {load_time:.1f}s, "
                f"convert: {convert_time:.1f}s, save: {build_time + write_time:.1f}s)")
    logger.info("")
    logger.info("To use this checkpoint, set in config.toml:")
    logger.info(f'  gemma_variant = "fp8-safetensors"')
    logger.info(f'  # and ensure the file is at: {out_path}')

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Gemma3 encoder weights to fp8 safetensors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to LTX-2 model directory (e.g., models/LTX-2)",
    )
    parser.add_argument(
        "--encoder-path",
        type=str,
        default=None,
        help="Override path for Gemma3 weights (default: model_path/text_encoder/)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output path for fp8 safetensors (default: model_path/text_encoder_fp8.safetensors)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show conversion stats without saving",
    )

    args = parser.parse_args()
    convert_gemma3_to_fp8(
        model_path=args.model_path,
        encoder_path=args.encoder_path,
        output_path=args.output,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
