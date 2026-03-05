#!/usr/bin/env python3
"""Audit GGUF key mapping against LTX-2 model architecture.

Loads a GGUF file via our loader, applies map_key() to every key,
builds a V2 model on meta device, and compares mapped keys against
the model's state_dict keys.

Usage:
    uv run python scripts/audit_gguf_keys.py /path/to/transformer.gguf

Reports: matched, missing (model has, GGUF doesn't), unexpected (GGUF has, model doesn't).
Runs in seconds -- no GPU, no dequant.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def audit_gguf_keys(gguf_path: str, video_only: bool = True) -> dict:
    """Audit GGUF keys against model architecture.

    Args:
        gguf_path: Path to GGUF file.
        video_only: If True, build VideoOnly model (skip audio keys).

    Returns:
        Dict with "matched", "missing", "unexpected", "is_v2" keys.
    """
    from llm_dit.quantization.gguf_loader import gguf_sd_loader, detect_v2_from_state_dict
    from llm_dit.models.ltx2.loader import (
        create_model_from_config,
        load_config,
        map_key,
        is_audio_key,
        LTXModelType,
    )
    from llm_dit.utils.meta_init import meta_init

    # Step 1: Load GGUF state dict (strips model.diffusion_model. prefix)
    logger.info(f"Loading GGUF: {gguf_path}")
    gguf_sd, extra = gguf_sd_loader(gguf_path)
    logger.info(f"GGUF: {len(gguf_sd)} keys, arch={extra.get('arch_str', 'unknown')}")

    # Step 2: Detect V2
    is_v2 = detect_v2_from_state_dict(gguf_sd)
    logger.info(f"V2 detected: {is_v2}")

    # Step 3: Apply map_key() and filter audio
    mapped_keys = set()
    skipped_audio = 0
    key_mapping = {}  # mapped -> original for debugging
    for key in gguf_sd:
        if video_only and is_audio_key(key):
            skipped_audio += 1
            continue
        mapped = map_key(key)
        mapped_keys.add(mapped)
        key_mapping[mapped] = key

    logger.info(f"Mapped: {len(mapped_keys)} keys (skipped {skipped_audio} audio)")

    # Step 4: Build model on meta device
    config = load_config(Path(gguf_path))
    model_type = LTXModelType.VideoOnly if video_only else LTXModelType.AudioVideo

    v2_kwargs = {}
    if is_v2:
        v2_kwargs["apply_gated_attention"] = True
        v2_kwargs["cross_attention_adaln"] = True

    with meta_init():
        model = create_model_from_config(
            config, torch.bfloat16, model_type=model_type, **v2_kwargs
        )

    model_keys = set(model.state_dict().keys())
    logger.info(f"Model: {len(model_keys)} keys")

    # Step 5: Compare
    matched = mapped_keys & model_keys
    missing = model_keys - mapped_keys  # model has, GGUF doesn't
    unexpected = mapped_keys - model_keys  # GGUF has, model doesn't

    return {
        "matched": sorted(matched),
        "missing": sorted(missing),
        "unexpected": sorted(unexpected),
        "is_v2": is_v2,
        "key_mapping": key_mapping,
        "total_gguf": len(gguf_sd),
        "total_model": len(model_keys),
    }


def print_report(result: dict) -> None:
    """Print audit report to stdout."""
    print(f"\n{'='*60}")
    print(f"GGUF Key Audit Report")
    print(f"{'='*60}")
    print(f"Model version: {'2.3 (22B)' if result['is_v2'] else '1.0 (19B)'}")
    print(f"GGUF keys:     {result['total_gguf']}")
    print(f"Model keys:    {result['total_model']}")
    print(f"Matched:       {len(result['matched'])}")
    print(f"Missing:       {len(result['missing'])} (model has, GGUF doesn't)")
    print(f"Unexpected:    {len(result['unexpected'])} (GGUF has, model doesn't)")

    if result["missing"]:
        print(f"\n--- Missing keys (model expects, GGUF missing) ---")
        for k in result["missing"][:30]:
            print(f"  {k}")
        if len(result["missing"]) > 30:
            print(f"  ... and {len(result['missing']) - 30} more")

    if result["unexpected"]:
        print(f"\n--- Unexpected keys (GGUF has, model doesn't) ---")
        mapping = result["key_mapping"]
        for k in result["unexpected"][:30]:
            orig = mapping.get(k, "?")
            print(f"  {k}")
            if orig != k:
                print(f"    (from GGUF: {orig})")
        if len(result["unexpected"]) > 30:
            print(f"  ... and {len(result['unexpected']) - 30} more")

    if not result["missing"] and not result["unexpected"]:
        print(f"\nPerfect match -- all keys aligned.")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Audit GGUF key mapping against model architecture")
    parser.add_argument("gguf_path", help="Path to GGUF file")
    parser.add_argument("--audio", action="store_true", help="Include audio keys (build AudioVideo model)")
    args = parser.parse_args()

    if not Path(args.gguf_path).exists():
        logger.error(f"File not found: {args.gguf_path}")
        sys.exit(1)

    result = audit_gguf_keys(args.gguf_path, video_only=not args.audio)
    print_report(result)

    # Exit with error code if mismatches found
    if result["missing"] or result["unexpected"]:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
