#!/usr/bin/env python3
"""
Quantize Qwen Image Edit model components.

last updated: 2026-02-06

This script creates YOUR OWN trusted quantized model checkpoints.
It loads the original model, applies quantization, and saves the result.

Supported quantization methods (unified torchao):
- fp8-dynamic: FP8 weights + FP8 activations (RTX 4090+, H100)
- fp8-weight-only: FP8 weights, BF16 activations
- int8: INT8 weight-only
- int4: INT4 weight-only (max compression)

Usage:
    # Quantize transformer with fp8-weight-only
    uv run scripts/quantize_model.py --model-path /path/to/model \
        --component transformer --method fp8-weight-only --output /path/to/output

    # Quantize text encoder with int8
    uv run scripts/quantize_model.py --model-path /path/to/model \
        --component text_encoder --method int8 --output /path/to/output

    # Hybrid: fp8 transformer + int8 text encoder
    uv run scripts/quantize_model.py --model-path /path/to/model \
        --component both --transformer-method fp8-weight-only --text-encoder-method int8 \
        --output /path/to/output
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Valid methods from the unified quantization system
VALID_METHODS = ["fp8-dynamic", "fp8-weight-only", "int8", "int4"]


def quantize_transformer(
    model_path: str,
    output_path: str,
    method: str = "fp8-weight-only",
    dtype: torch.dtype = torch.bfloat16,
    verbose: bool = False,
) -> dict:
    """Quantize the DiT transformer component."""
    logger.info(f"Loading transformer from {model_path}")
    logger.info(f"  Method: {method}")
    logger.info(f"  Dtype: {dtype}")

    from diffusers import QwenImageTransformer2DModel

    logger.info("Loading model weights...")
    transformer = QwenImageTransformer2DModel.from_pretrained(
        model_path,
        subfolder="transformer",
        dtype=dtype,
        low_cpu_mem_usage=True,
    )

    param_count = sum(p.numel() for p in transformer.parameters())
    logger.info(f"Model loaded: {param_count / 1e9:.2f}B parameters")

    from llm_dit.quantization import quantize_component

    logger.info(f"Starting {method} quantization...")
    transformer, stats = quantize_component(
        transformer, method=method, component_type="transformer", verbose=verbose
    )

    logger.info(f"Quantized {stats['quantized_layers']}/{stats['total_layers']} layers")
    if stats["skipped_layers"] > 0:
        logger.info(f"Skipped {stats['skipped_layers']} layers")

    # TorchAO quantized tensors use AffineQuantizedTensor subclasses that
    # cannot be serialized with safetensors. This is a known torchao limitation.
    # These are our OWN quantized checkpoints from trusted source models.
    output_dir = Path(output_path) / "transformer"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving quantized transformer to {output_dir}")
    logger.info("  Using torch format (TorchAO tensor subclasses)")
    transformer.save_pretrained(output_dir, safe_serialization=False)

    return stats


def quantize_text_encoder(
    model_path: str,
    output_path: str,
    method: str = "int8",
    dtype: torch.dtype = torch.bfloat16,
    verbose: bool = False,
) -> dict:
    """Quantize the text encoder component."""
    logger.info(f"Loading text encoder from {model_path}")
    logger.info(f"  Method: {method}")
    logger.info(f"  Dtype: {dtype}")

    from transformers import Qwen2_5_VLForConditionalGeneration

    logger.info("Loading model weights...")
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        subfolder="text_encoder",
        dtype=dtype,
        low_cpu_mem_usage=True,
    )

    param_count = sum(p.numel() for p in text_encoder.parameters())
    logger.info(f"Model loaded: {param_count / 1e9:.2f}B parameters")

    from llm_dit.quantization import quantize_component

    logger.info(f"Starting {method} quantization...")
    text_encoder, stats = quantize_component(
        text_encoder, method=method, component_type="encoder", verbose=verbose
    )

    logger.info(f"Quantized {stats['quantized_layers']}/{stats['total_layers']} layers")
    if stats["skipped_layers"] > 0:
        logger.info(f"Skipped {stats['skipped_layers']} layers")

    # TorchAO quantized tensors use AffineQuantizedTensor subclasses that
    # cannot be serialized with safetensors.
    output_dir = Path(output_path) / "text_encoder"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving quantized text encoder to {output_dir}")
    logger.info("  Using torch format (TorchAO tensor subclasses)")
    text_encoder.save_pretrained(output_dir, safe_serialization=False)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Quantize Qwen Image Edit model components",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quantize transformer with FP8 weight-only
  uv run scripts/quantize_model.py --model-path /path/to/model \\
      --component transformer --method fp8-weight-only --output /path/to/output

  # Hybrid quantization (FP8 transformer + INT8 text encoder)
  uv run scripts/quantize_model.py --model-path /path/to/model \\
      --component both --transformer-method fp8-weight-only --text-encoder-method int8 \\
      --output /path/to/output
        """,
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the source model directory",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output directory for quantized model",
    )
    parser.add_argument(
        "--component",
        type=str,
        choices=["transformer", "text_encoder", "both"],
        default="both",
        help="Which component(s) to quantize (default: both)",
    )
    parser.add_argument(
        "--method",
        "-m",
        type=str,
        choices=VALID_METHODS,
        default="fp8-weight-only",
        help="Quantization method for single component (default: fp8-weight-only)",
    )
    parser.add_argument(
        "--transformer-method",
        type=str,
        choices=VALID_METHODS,
        default="fp8-weight-only",
        help="Quantization method for transformer (when --component=both)",
    )
    parser.add_argument(
        "--text-encoder-method",
        type=str,
        choices=VALID_METHODS,
        default="int8",
        help="Quantization method for text encoder (when --component=both)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output (log each layer being quantized)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    args = parser.parse_args()

    if not Path(args.model_path).exists():
        print(f"Error: Model path does not exist: {args.model_path}")
        sys.exit(1)

    if not args.output:
        print("Error: --output is required for quantization")
        sys.exit(1)

    results = {}
    verbose = args.verbose

    if args.component == "both":
        transformer_method = args.transformer_method
        text_encoder_method = args.text_encoder_method

        logger.info(
            f"Hybrid quantization: transformer={transformer_method}, text_encoder={text_encoder_method}"
        )

        logger.info("=" * 50)
        logger.info("Quantizing transformer...")
        results["transformer"] = quantize_transformer(
            args.model_path, args.output, transformer_method, verbose=verbose
        )
        torch.cuda.empty_cache()

        logger.info("=" * 50)
        logger.info("Quantizing text encoder...")
        results["text_encoder"] = quantize_text_encoder(
            args.model_path, args.output, text_encoder_method, verbose=verbose
        )
        torch.cuda.empty_cache()

    elif args.component == "transformer":
        results["transformer"] = quantize_transformer(
            args.model_path, args.output, args.method, verbose=verbose
        )

    elif args.component == "text_encoder":
        results["text_encoder"] = quantize_text_encoder(
            args.model_path, args.output, args.method, verbose=verbose
        )

    # Copy other necessary files (scheduler, tokenizer, etc.)
    logger.info("Copying additional model files...")
    source_path = Path(args.model_path)
    output_path = Path(args.output)

    for subdir in ["scheduler", "tokenizer", "vae"]:
        src = source_path / subdir
        if src.exists():
            import shutil

            dst = output_path / subdir
            if not dst.exists():
                shutil.copytree(src, dst)
                logger.info(f"  Copied {subdir}/")

    # Copy model_index.json
    model_index = source_path / "model_index.json"
    if model_index.exists():
        import shutil

        shutil.copy2(model_index, output_path / "model_index.json")
        logger.info("  Copied model_index.json")

    # Save quantization metadata
    metadata = {
        "source_model": str(args.model_path),
        "quantization": results,
        "component": args.component,
    }
    with open(output_path / "quantization_info.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("=" * 50)
    logger.info(f"Quantization complete! Output saved to: {args.output}")
    logger.info("=" * 50)

    if args.json:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
