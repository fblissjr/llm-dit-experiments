#!/usr/bin/env python3
"""
Quantize Qwen Image Edit model components.

last updated: 2025-12-29

This script creates YOUR OWN trusted quantized model checkpoints.
It loads the original model, applies quantization, and saves the result.

Supported quantization methods:
- fp8: TorchAO FP8 dynamic (RTX 4090+, H100)
- fp8-filtered: FP8 with auto-skip of non-16-aligned layers
- int8: TorchAO INT8 weight-only
- 4bit: BitsAndBytes NF4
- 8bit: BitsAndBytes INT8

Usage:
    # Analyze FP8 compatibility first
    uv run scripts/quantize_model.py --model-path /path/to/model --analyze

    # Quantize transformer only (recommended for Qwen Image Edit)
    uv run scripts/quantize_model.py --model-path /path/to/model \\
        --component transformer --method fp8 --output /path/to/output

    # Quantize text encoder with filtered FP8 (skips incompatible layers)
    uv run scripts/quantize_model.py --model-path /path/to/model \\
        --component text_encoder --method fp8-filtered --output /path/to/output

    # Hybrid: FP8 transformer + 4bit text encoder
    uv run scripts/quantize_model.py --model-path /path/to/model \\
        --component both --transformer-method fp8 --text-encoder-method 4bit \\
        --output /path/to/output
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def analyze_model(model_path: str, component: str = "both") -> dict:
    """Analyze model's FP8 compatibility without quantizing."""
    from llm_dit.quantization import analyze_fp8_compatibility

    results = {}

    if component in ("transformer", "both"):
        logger.info("Analyzing transformer FP8 compatibility...")
        try:
            from diffusers import QwenImageTransformer2DModel

            transformer = QwenImageTransformer2DModel.from_pretrained(
                model_path,
                subfolder="transformer",
                dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
            results["transformer"] = analyze_fp8_compatibility(transformer)
            del transformer
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Failed to analyze transformer: {e}")
            results["transformer"] = {"error": str(e)}

    if component in ("text_encoder", "both"):
        logger.info("Analyzing text encoder FP8 compatibility...")
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration

            text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                subfolder="text_encoder",
                dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
            results["text_encoder"] = analyze_fp8_compatibility(text_encoder)
            del text_encoder
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Failed to analyze text encoder: {e}")
            results["text_encoder"] = {"error": str(e)}

    return results


def print_analysis_results(results: dict):
    """Pretty-print analysis results."""
    print("\n" + "=" * 70)
    print("FP8 COMPATIBILITY ANALYSIS")
    print("=" * 70)

    for component, analysis in results.items():
        print(f"\n--- {component.upper()} ---")

        if "error" in analysis:
            print(f"  Error: {analysis['error']}")
            continue

        total = analysis["total_linear_layers"]
        compatible = analysis["compatible_layers"]
        incompatible = analysis["incompatible_layers"]
        rate = analysis["compatibility_rate"]

        print(f"  Total Linear layers: {total}")
        print(f"  FP8-compatible:      {compatible} ({rate:.1f}%)")
        print(f"  Incompatible:        {incompatible}")

        if incompatible > 0:
            print("\n  Incompatible layers:")
            for info in analysis["incompatible_layer_info"][:5]:  # Show first 5
                print(f"    - {info['name']}: [{info['out_features']}, {info['in_features']}]")
                print(f"      (in%16={info['in_remainder']}, out%16={info['out_remainder']})")
            if incompatible > 5:
                print(f"    ... and {incompatible - 5} more")

    print("\n" + "=" * 70)

    # Recommendations
    print("\nRECOMMENDATIONS:")
    for component, analysis in results.items():
        if "error" in analysis:
            continue

        rate = analysis["compatibility_rate"]
        if rate == 100:
            print(f"  {component}: Use 'fp8' (fully compatible)")
        elif rate >= 90:
            print(
                f"  {component}: Use 'fp8-filtered' ({rate:.0f}% compatible, skips {analysis['incompatible_layers']} layers)"
            )
        else:
            print(f"  {component}: Use '4bit' or '8bit' instead (only {rate:.0f}% FP8-compatible)")

    print("=" * 70 + "\n")


def quantize_transformer(
    model_path: str,
    output_path: str,
    method: str = "fp8",
    dtype: torch.dtype = torch.bfloat16,
    verbose: bool = False,
) -> dict:
    """Quantize the DiT transformer component."""
    logger.info(f"Loading transformer from {model_path}")
    logger.info(f"  Method: {method}")
    logger.info(f"  Dtype: {dtype}")

    from diffusers import QwenImageTransformer2DModel

    if method in ("fp8", "fp8-filtered"):
        # Load first, then quantize with our filtered method
        logger.info("Loading model weights...")
        transformer = QwenImageTransformer2DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            dtype=dtype,
            low_cpu_mem_usage=True,
        )

        # Report model size
        param_count = sum(p.numel() for p in transformer.parameters())
        logger.info(f"Model loaded: {param_count / 1e9:.2f}B parameters")

        from llm_dit.quantization import quantize_model_torchao_filtered

        use_filtered = method == "fp8-filtered"
        logger.info(f"Starting quantization (filtered={use_filtered})...")
        transformer, stats = quantize_model_torchao_filtered(
            transformer, "fp8", skip_incompatible=use_filtered, verbose=verbose
        )

        logger.info(f"Quantized {stats['quantized_layers']}/{stats['total_linear_layers']} layers")
        if stats["skipped_layers"] > 0:
            logger.info(f"Skipped {stats['skipped_layers']} incompatible layers")

        # TorchAO FP8 requires pickle format (safetensors doesn't support tensor subclasses)
        use_safetensors = False

    elif method == "int8":
        logger.info("Loading model weights...")
        transformer = QwenImageTransformer2DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            dtype=dtype,
            low_cpu_mem_usage=True,
        )

        param_count = sum(p.numel() for p in transformer.parameters())
        logger.info(f"Model loaded: {param_count / 1e9:.2f}B parameters")

        from llm_dit.quantization import quantize_model_torchao

        logger.info("Starting INT8 quantization...")
        quantize_model_torchao(transformer, "int8")
        stats = {"method": "int8"}

        # TorchAO INT8 also requires pickle format
        use_safetensors = False

    elif method in ("4bit", "8bit"):
        from diffusers import BitsAndBytesConfig

        if method == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
            )
            logger.info("Loading with BitsAndBytes 4-bit quantization...")
        else:
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
            logger.info("Loading with BitsAndBytes 8-bit quantization...")

        transformer = QwenImageTransformer2DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            quantization_config=quant_config,
            dtype=dtype,
        )
        stats = {"method": method}

        # BitsAndBytes can use safetensors
        use_safetensors = True

    else:
        raise ValueError(f"Unknown method: {method}")

    # Save the quantized model
    output_dir = Path(output_path) / "transformer"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving quantized transformer to {output_dir}")
    if not use_safetensors:
        logger.info("  Using pickle format (TorchAO tensors don't support safetensors)")
    transformer.save_pretrained(output_dir, safe_serialization=use_safetensors)

    return stats


def quantize_text_encoder(
    model_path: str,
    output_path: str,
    method: str = "fp8-filtered",
    dtype: torch.dtype = torch.bfloat16,
    verbose: bool = False,
) -> dict:
    """Quantize the text encoder component."""
    logger.info(f"Loading text encoder from {model_path}")
    logger.info(f"  Method: {method}")
    logger.info(f"  Dtype: {dtype}")

    from transformers import Qwen2_5_VLForConditionalGeneration

    if method in ("fp8", "fp8-filtered"):
        # Load first, then quantize
        logger.info("Loading model weights...")
        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            subfolder="text_encoder",
            dtype=dtype,
            low_cpu_mem_usage=True,
        )

        # Report model size
        param_count = sum(p.numel() for p in text_encoder.parameters())
        logger.info(f"Model loaded: {param_count / 1e9:.2f}B parameters")

        from llm_dit.quantization import quantize_model_torchao_filtered

        use_filtered = method == "fp8-filtered"
        logger.info(f"Starting quantization (filtered={use_filtered})...")
        text_encoder, stats = quantize_model_torchao_filtered(
            text_encoder, "fp8", skip_incompatible=use_filtered, verbose=verbose
        )

        logger.info(f"Quantized {stats['quantized_layers']}/{stats['total_linear_layers']} layers")
        if stats["skipped_layers"] > 0:
            logger.info(f"Skipped {stats['skipped_layers']} incompatible layers")

        # TorchAO FP8 requires pickle format
        use_safetensors = False

    elif method == "int8":
        logger.info("Loading model weights...")
        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            subfolder="text_encoder",
            dtype=dtype,
            low_cpu_mem_usage=True,
        )

        param_count = sum(p.numel() for p in text_encoder.parameters())
        logger.info(f"Model loaded: {param_count / 1e9:.2f}B parameters")

        from llm_dit.quantization import quantize_model_torchao

        logger.info("Starting INT8 quantization...")
        quantize_model_torchao(text_encoder, "int8")
        stats = {"method": "int8"}

        # TorchAO INT8 also requires pickle format
        use_safetensors = False

    elif method in ("4bit", "8bit"):
        from transformers import BitsAndBytesConfig

        if method == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
            )
            logger.info("Loading with BitsAndBytes 4-bit quantization...")
        else:
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
            logger.info("Loading with BitsAndBytes 8-bit quantization...")

        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            subfolder="text_encoder",
            quantization_config=quant_config,
            dtype=dtype,
        )
        stats = {"method": method}

        # BitsAndBytes can use safetensors
        use_safetensors = True

    else:
        raise ValueError(f"Unknown method: {method}")

    # Save the quantized model
    output_dir = Path(output_path) / "text_encoder"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving quantized text encoder to {output_dir}")
    if not use_safetensors:
        logger.info("  Using pickle format (TorchAO tensors don't support safetensors)")
    text_encoder.save_pretrained(output_dir, safe_serialization=use_safetensors)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Quantize Qwen Image Edit model components",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze FP8 compatibility
  uv run scripts/quantize_model.py --model-path /path/to/model --analyze

  # Quantize transformer with FP8
  uv run scripts/quantize_model.py --model-path /path/to/model \\
      --component transformer --method fp8 --output /path/to/output

  # Hybrid quantization (FP8 transformer + 4bit text encoder)
  uv run scripts/quantize_model.py --model-path /path/to/model \\
      --component both --transformer-method fp8 --text-encoder-method 4bit \\
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
        choices=["fp8", "fp8-filtered", "int8", "4bit", "8bit"],
        default="fp8",
        help="Quantization method for single component (default: fp8)",
    )
    parser.add_argument(
        "--transformer-method",
        type=str,
        choices=["fp8", "fp8-filtered", "int8", "4bit", "8bit"],
        default="fp8",
        help="Quantization method for transformer (when --component=both)",
    )
    parser.add_argument(
        "--text-encoder-method",
        type=str,
        choices=["fp8", "fp8-filtered", "int8", "4bit", "8bit"],
        default="4bit",
        help="Quantization method for text encoder (when --component=both)",
    )
    parser.add_argument(
        "--analyze",
        "-a",
        action="store_true",
        help="Only analyze FP8 compatibility, don't quantize",
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

    # Analysis mode
    if args.analyze:
        results = analyze_model(args.model_path, args.component)
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print_analysis_results(results)
        return

    # Quantization mode
    if not args.output:
        print("Error: --output is required for quantization")
        sys.exit(1)

    results = {}
    verbose = args.verbose

    if args.component == "both":
        # Hybrid quantization with different methods for each component
        transformer_method = args.transformer_method
        text_encoder_method = args.text_encoder_method

        logger.info(
            f"Hybrid quantization: transformer={transformer_method}, text_encoder={text_encoder_method}"
        )

        # Quantize transformer
        logger.info("=" * 50)
        logger.info("Quantizing transformer...")
        results["transformer"] = quantize_transformer(
            args.model_path, args.output, transformer_method, verbose=verbose
        )
        torch.cuda.empty_cache()

        # Quantize text encoder
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
