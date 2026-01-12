#!/usr/bin/env python3
"""
Convert PyTorch checkpoint files to safetensors format.

Last Updated: 2026-01-12

Safetensors is a safer and faster format for storing model weights:
- No arbitrary code execution (unlike .pth/.pt files)
- Faster loading (memory-mapped file access)
- Smaller file sizes in some cases

Usage:
    # Single file
    uv run python scripts/convert_to_safetensors.py ~/Storage/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth

    # Directory (recursive)
    uv run python scripts/convert_to_safetensors.py ~/Storage/Wan2.1-T2V-1.3B/ --recursive

    # With verification and cleanup
    uv run python scripts/convert_to_safetensors.py ~/Storage/ --recursive --verify --delete-originals
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from safetensors.torch import save_file as save_safetensors
from safetensors.torch import load_file as load_safetensors


def convert_file(
    input_path: Path,
    output_path: Optional[Path] = None,
    verify: bool = True,
    delete_original: bool = False,
) -> bool:
    """
    Convert a single .pth/.pt file to .safetensors format.

    Args:
        input_path: Path to input .pth or .pt file
        output_path: Output path (default: same name with .safetensors extension)
        verify: Whether to verify conversion by comparing state dicts
        delete_original: Whether to delete original file after successful conversion

    Returns:
        True if conversion successful, False otherwise
    """
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        return False

    if input_path.suffix not in (".pth", ".pt"):
        print(f"Skipping non-PyTorch file: {input_path}")
        return False

    # Default output path
    if output_path is None:
        output_path = input_path.with_suffix(".safetensors")

    # Skip if already converted
    if output_path.exists():
        print(f"Skipping (already exists): {output_path}")
        return True

    print(f"Converting: {input_path}")
    print(f"       to: {output_path}")

    try:
        # Load with weights_only=True for safety (avoids arbitrary code execution)
        state_dict = torch.load(str(input_path), map_location="cpu", weights_only=True)

        # Handle nested state dicts (some checkpoints wrap in 'state_dict' key)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            print("  Note: Extracting nested 'state_dict' key")
            state_dict = state_dict["state_dict"]

        # Validate all values are tensors
        non_tensor_keys = [k for k, v in state_dict.items() if not isinstance(v, torch.Tensor)]
        if non_tensor_keys:
            print(f"  Warning: Skipping {len(non_tensor_keys)} non-tensor entries")
            state_dict = {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}

        # Convert to contiguous tensors (required by safetensors)
        state_dict = {k: v.contiguous() for k, v in state_dict.items()}

        # Save as safetensors
        save_safetensors(state_dict, str(output_path))

        # Verify conversion
        if verify:
            if not verify_conversion(input_path, output_path, state_dict):
                print(f"  ERROR: Verification failed!")
                output_path.unlink()  # Remove failed conversion
                return False
            print(f"  Verified: {len(state_dict)} tensors match")

        # Optionally delete original
        if delete_original:
            input_path.unlink()
            print(f"  Deleted original: {input_path}")

        # Report size difference
        orig_size = input_path.stat().st_size if input_path.exists() else 0
        new_size = output_path.stat().st_size
        if orig_size > 0:
            ratio = new_size / orig_size
            print(f"  Size: {orig_size / 1e6:.1f}MB -> {new_size / 1e6:.1f}MB ({ratio:.2%})")

        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        if output_path.exists():
            output_path.unlink()
        return False


def verify_conversion(
    original_path: Path,
    safetensors_path: Path,
    original_state_dict: Optional[Dict[str, torch.Tensor]] = None,
) -> bool:
    """
    Verify that safetensors file matches original checkpoint.

    Args:
        original_path: Path to original .pth/.pt file
        safetensors_path: Path to converted .safetensors file
        original_state_dict: Pre-loaded state dict (avoids reloading)

    Returns:
        True if verification passes
    """
    try:
        # Load safetensors
        converted = load_safetensors(str(safetensors_path))

        # Load original if not provided
        if original_state_dict is None:
            original_state_dict = torch.load(str(original_path), map_location="cpu", weights_only=True)
            if isinstance(original_state_dict, dict) and "state_dict" in original_state_dict:
                original_state_dict = original_state_dict["state_dict"]
            original_state_dict = {k: v for k, v in original_state_dict.items() if isinstance(v, torch.Tensor)}

        # Compare keys
        orig_keys = set(original_state_dict.keys())
        conv_keys = set(converted.keys())

        if orig_keys != conv_keys:
            missing = orig_keys - conv_keys
            extra = conv_keys - orig_keys
            if missing:
                print(f"  Missing keys: {list(missing)[:5]}...")
            if extra:
                print(f"  Extra keys: {list(extra)[:5]}...")
            return False

        # Compare values
        for key in orig_keys:
            orig_tensor = original_state_dict[key]
            conv_tensor = converted[key]

            if orig_tensor.shape != conv_tensor.shape:
                print(f"  Shape mismatch for {key}: {orig_tensor.shape} vs {conv_tensor.shape}")
                return False

            if not torch.allclose(orig_tensor.float(), conv_tensor.float(), atol=1e-6):
                max_diff = (orig_tensor.float() - conv_tensor.float()).abs().max().item()
                print(f"  Value mismatch for {key}: max diff = {max_diff}")
                return False

        return True

    except Exception as e:
        print(f"  Verification error: {e}")
        return False


def find_checkpoint_files(directory: Path, recursive: bool = False) -> List[Path]:
    """Find all .pth and .pt files in directory."""
    pattern = "**/*.pth" if recursive else "*.pth"
    pth_files = list(directory.glob(pattern))

    pattern = "**/*.pt" if recursive else "*.pt"
    pt_files = list(directory.glob(pattern))

    return sorted(pth_files + pt_files)


def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch checkpoints to safetensors format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "path",
        type=Path,
        help="File or directory to convert",
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Recursively convert files in directory",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip verification after conversion",
    )
    parser.add_argument(
        "--delete-originals",
        action="store_true",
        help="Delete original files after successful conversion",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output path (for single file conversion)",
    )

    args = parser.parse_args()

    if args.path.is_file():
        # Single file conversion
        success = convert_file(
            args.path,
            output_path=args.output,
            verify=not args.no_verify,
            delete_original=args.delete_originals,
        )
        sys.exit(0 if success else 1)

    elif args.path.is_dir():
        # Directory conversion
        files = find_checkpoint_files(args.path, recursive=args.recursive)

        if not files:
            print(f"No .pth or .pt files found in {args.path}")
            sys.exit(0)

        print(f"Found {len(files)} checkpoint files to convert")
        print()

        success_count = 0
        fail_count = 0

        for filepath in files:
            if convert_file(
                filepath,
                verify=not args.no_verify,
                delete_original=args.delete_originals,
            ):
                success_count += 1
            else:
                fail_count += 1
            print()

        print("=" * 50)
        print(f"Conversion complete: {success_count} succeeded, {fail_count} failed")
        sys.exit(0 if fail_count == 0 else 1)

    else:
        print(f"Error: Path not found: {args.path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
