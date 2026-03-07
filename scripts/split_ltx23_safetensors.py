#!/usr/bin/env python3
"""Split bundled LTX-2.3 fp8 safetensors into individual component files.

Last Updated: 2026-03-06

The official LTX-2.3 fp8 checkpoint bundles ALL components (transformer, VAE,
audio VAE, vocoder, connectors, aggregate embeds) in a single ~28GB file.
This script splits it into individual component files for our loading pipeline.

Usage:
    uv run python scripts/split_ltx23_safetensors.py \\
        models/LTX-2.3-fp8/ltx-2.3-22b-dev-fp8.safetensors \\
        --output models/LTX-2.3/

Output files:
    ltx-2.3-transformer-fp8.safetensors   -- transformer (mixed fp8+bf16+f32)
    ltx-2.3-connectors.safetensors         -- connectors + aggregate embeds
    ltx-2.3-video-vae.safetensors          -- video VAE
    ltx-2.3-audio-vae.safetensors          -- audio VAE
    ltx-2.3-vocoder.safetensors            -- HiFi-GAN vocoder + BWE
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

logger = logging.getLogger(__name__)

# Prefix groups and their output filenames + strip prefixes
COMPONENT_GROUPS = [
    {
        "name": "transformer",
        "output": "ltx-2.3-transformer-fp8.safetensors",
        "prefixes": ["model.diffusion_model."],
        "strip": "model.diffusion_model.",
        "exclude_prefixes": [
            "model.diffusion_model.video_embeddings_connector.",
            "model.diffusion_model.audio_embeddings_connector.",
        ],
    },
    {
        "name": "connectors",
        "output": "ltx-2.3-connectors.safetensors",
        "prefixes": [
            "model.diffusion_model.video_embeddings_connector.",
            "model.diffusion_model.audio_embeddings_connector.",
            "text_embedding_projection.",
        ],
        "strip": None,  # Keep prefixes as-is for the connector loader
    },
    {
        "name": "video-vae",
        "output": "ltx-2.3-video-vae.safetensors",
        "prefixes": ["vae."],
        "strip": "vae.",
    },
    {
        "name": "audio-vae",
        "output": "ltx-2.3-audio-vae.safetensors",
        "prefixes": ["audio_vae."],
        "strip": "audio_vae.",
    },
    {
        "name": "vocoder",
        "output": "ltx-2.3-vocoder.safetensors",
        "prefixes": ["vocoder."],
        "strip": "vocoder.",
    },
]


def _build_component_metadata(full_config: dict, component_name: str) -> dict[str, str]:
    """Extract the relevant config section for a component and return as metadata dict.

    Args:
        full_config: Full config dict from bundled checkpoint metadata.
        component_name: One of the COMPONENT_GROUPS names.

    Returns:
        Metadata dict with a "config" key (JSON string), or empty if no config available.
    """
    import json

    # Map component names to config sections
    section_map = {
        "transformer": "transformer",
        "connectors": "transformer",  # connectors use transformer config for dims
        "video-vae": "vae",
        "audio-vae": "audio_vae",
        "vocoder": "vocoder",
    }

    section_key = section_map.get(component_name)
    if not section_key or section_key not in full_config:
        return {}

    section = full_config[section_key]

    # For transformer: also include model_version from parent if available
    meta = {"config": json.dumps(section)}

    return meta


def split_safetensors(input_path: Path, output_dir: Path, dry_run: bool = False) -> dict:
    """Split a bundled safetensors file into component files.

    Preserves config metadata from the bundled file: each split file gets
    the relevant config section embedded in its safetensors metadata.

    Args:
        input_path: Path to the bundled safetensors file.
        output_dir: Directory to write component files.
        dry_run: If True, only report what would be written.

    Returns:
        Dict mapping component name to key count.
    """
    import json

    logger.info(f"Opening {input_path} ({input_path.stat().st_size / 1e9:.1f} GB)")

    results = {}

    with safe_open(str(input_path), framework="pt") as f:
        all_keys = list(f.keys())
        logger.info(f"Total keys: {len(all_keys)}")

        # Extract full config from bundled metadata
        source_meta = f.metadata() or {}
        full_config = {}
        if "config" in source_meta:
            full_config = json.loads(source_meta["config"])
            logger.info(f"Source config sections: {list(full_config.keys())}")
        else:
            logger.warning("No config metadata in source file -- split files will have no config")

        claimed_keys: set[str] = set()

        for group in COMPONENT_GROUPS:
            component_tensors = {}
            exclude_prefixes = tuple(group.get("exclude_prefixes", []))

            for key in all_keys:
                # Skip keys claimed by exclude prefixes
                if exclude_prefixes and key.startswith(exclude_prefixes):
                    continue

                for prefix in group["prefixes"]:
                    if key.startswith(prefix):
                        strip = group.get("strip")
                        new_key = key[len(strip):] if strip and key.startswith(strip) else key
                        component_tensors[new_key] = f.get_tensor(key)
                        claimed_keys.add(key)
                        break

            if not component_tensors:
                logger.warning(f"  {group['name']}: no keys found")
                continue

            output_path = output_dir / group["output"]
            results[group["name"]] = len(component_tensors)

            # Report dtypes
            dtypes = {}
            for t in component_tensors.values():
                dt = str(t.dtype)
                dtypes[dt] = dtypes.get(dt, 0) + 1
            dtype_str = ", ".join(f"{c} {d}" for d, c in sorted(dtypes.items()))

            logger.info(f"  {group['name']}: {len(component_tensors)} keys ({dtype_str})")

            if not dry_run:
                component_meta = _build_component_metadata(full_config, group["name"])
                if component_meta:
                    logger.info(f"    embedding config metadata ({group['name']})")
                save_file(component_tensors, str(output_path), metadata=component_meta)
                size_gb = output_path.stat().st_size / 1e9
                logger.info(f"    -> {output_path} ({size_gb:.2f} GB)")

        # Report unclaimed keys
        unclaimed = set(all_keys) - claimed_keys
        if unclaimed:
            logger.warning(f"  Unclaimed keys ({len(unclaimed)}): {sorted(unclaimed)[:5]}...")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Split bundled LTX-2.3 fp8 safetensors into component files."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to the bundled safetensors file (e.g. ltx-2.3-22b-dev-fp8.safetensors)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/LTX-2.3"),
        help="Output directory (default: models/LTX-2.3/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be written without writing files",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)

    start = time.time()
    results = split_safetensors(args.input, args.output, dry_run=args.dry_run)
    elapsed = time.time() - start

    total_keys = sum(results.values())
    logger.info(f"\nSplit {total_keys} keys into {len(results)} components in {elapsed:.1f}s")
    if args.dry_run:
        logger.info("(dry run -- no files written)")


if __name__ == "__main__":
    main()
