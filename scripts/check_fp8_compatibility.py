#!/usr/bin/env python3
"""
Check FP8 compatibility for model layers.

last updated: 2025-12-29

FP8 quantization with TorchAO's _scaled_mm requires both dimensions
of Linear layers to be multiples of 16 for tensor core operations.

This script analyzes a model and reports which layers are FP8-compatible.

Usage:
    uv run scripts/check_fp8_compatibility.py --model-path /path/to/model
    uv run scripts/check_fp8_compatibility.py --model-path /path/to/model --component transformer
    uv run scripts/check_fp8_compatibility.py --model-path /path/to/model --verbose
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class LayerInfo:
    """Information about a single layer's FP8 compatibility."""

    name: str
    shape: tuple[int, int]
    in_features: int
    out_features: int
    in_aligned: bool
    out_aligned: bool
    fp8_compatible: bool
    params: int


@dataclass
class CompatibilityReport:
    """Summary report of FP8 compatibility analysis."""

    total_linear_layers: int = 0
    compatible_layers: int = 0
    incompatible_layers: int = 0
    total_params: int = 0
    compatible_params: int = 0
    incompatible_params: int = 0
    layers: list[LayerInfo] = field(default_factory=list)

    @property
    def compatibility_rate(self) -> float:
        if self.total_linear_layers == 0:
            return 0.0
        return self.compatible_layers / self.total_linear_layers * 100

    @property
    def param_coverage(self) -> float:
        if self.total_params == 0:
            return 0.0
        return self.compatible_params / self.total_params * 100


def is_fp8_compatible(in_features: int, out_features: int) -> bool:
    """Check if dimensions are compatible with FP8 _scaled_mm."""
    return in_features % 16 == 0 and out_features % 16 == 0


def analyze_model(model: nn.Module, prefix: str = "") -> CompatibilityReport:
    """Analyze a model's Linear layers for FP8 compatibility."""
    report = CompatibilityReport()

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            full_name = f"{prefix}.{name}" if prefix else name
            in_f = module.in_features
            out_f = module.out_features
            params = in_f * out_f

            in_aligned = in_f % 16 == 0
            out_aligned = out_f % 16 == 0
            compatible = in_aligned and out_aligned

            layer_info = LayerInfo(
                name=full_name,
                shape=(out_f, in_f),
                in_features=in_f,
                out_features=out_f,
                in_aligned=in_aligned,
                out_aligned=out_aligned,
                fp8_compatible=compatible,
                params=params,
            )

            report.layers.append(layer_info)
            report.total_linear_layers += 1
            report.total_params += params

            if compatible:
                report.compatible_layers += 1
                report.compatible_params += params
            else:
                report.incompatible_layers += 1
                report.incompatible_params += params

    return report


def print_report(report: CompatibilityReport, verbose: bool = False) -> None:
    """Print a formatted compatibility report."""
    print("\n" + "=" * 70)
    print("FP8 COMPATIBILITY REPORT")
    print("=" * 70)

    print(f"\nSummary:")
    print(f"  Total Linear layers: {report.total_linear_layers}")
    print(f"  FP8-compatible:      {report.compatible_layers} ({report.compatibility_rate:.1f}%)")
    print(f"  Incompatible:        {report.incompatible_layers}")
    print()
    print(f"  Total parameters:    {report.total_params:,}")
    print(f"  Compatible params:   {report.compatible_params:,} ({report.param_coverage:.1f}%)")
    print(f"  Incompatible params: {report.incompatible_params:,}")

    if report.incompatible_layers > 0:
        print("\n" + "-" * 70)
        print("INCOMPATIBLE LAYERS (dimensions not multiples of 16):")
        print("-" * 70)
        for layer in report.layers:
            if not layer.fp8_compatible:
                in_status = (
                    "OK"
                    if layer.in_aligned
                    else f"BAD ({layer.in_features} % 16 = {layer.in_features % 16})"
                )
                out_status = (
                    "OK"
                    if layer.out_aligned
                    else f"BAD ({layer.out_features} % 16 = {layer.out_features % 16})"
                )
                print(f"  {layer.name}")
                print(f"    Shape: {layer.shape}")
                print(f"    in_features: {in_status}")
                print(f"    out_features: {out_status}")
                print()

    if verbose:
        print("\n" + "-" * 70)
        print("ALL LAYERS:")
        print("-" * 70)
        for layer in report.layers:
            status = "OK" if layer.fp8_compatible else "SKIP"
            print(f"  [{status}] {layer.name}: {layer.shape}")

    print("\n" + "=" * 70)

    if report.compatibility_rate == 100:
        print("RESULT: Model is FULLY FP8 compatible!")
    elif report.param_coverage > 90:
        print(f"RESULT: Model is MOSTLY FP8 compatible ({report.param_coverage:.1f}% of params)")
        print("        Incompatible layers will remain in original precision.")
    else:
        print(
            f"RESULT: Model has LIMITED FP8 compatibility ({report.param_coverage:.1f}% of params)"
        )
        print("        Consider using 4bit/8bit quantization instead.")
    print("=" * 70 + "\n")


def load_transformer_from_diffusers(model_path: str) -> Optional[nn.Module]:
    """Load transformer component from a diffusers model."""
    try:
        from diffusers import QwenImageTransformer2DModel

        transformer = QwenImageTransformer2DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        return transformer
    except Exception as e:
        print(f"Error loading transformer: {e}")
        return None


def load_text_encoder(model_path: str) -> Optional[nn.Module]:
    """Load text encoder component."""
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration

        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            subfolder="text_encoder",
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        return text_encoder
    except Exception as e:
        print(f"Error loading text encoder: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Check FP8 compatibility for model layers")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model directory",
    )
    parser.add_argument(
        "--component",
        type=str,
        choices=["transformer", "text_encoder", "both"],
        default="both",
        help="Which component to analyze (default: both)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show all layers, not just incompatible ones",
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

    results = {}

    if args.component in ("transformer", "both"):
        print(f"\nLoading transformer from {args.model_path}...")
        transformer = load_transformer_from_diffusers(args.model_path)
        if transformer:
            report = analyze_model(transformer, prefix="transformer")
            results["transformer"] = report
            if not args.json:
                print("\n--- TRANSFORMER ---")
                print_report(report, verbose=args.verbose)

    if args.component in ("text_encoder", "both"):
        print(f"\nLoading text encoder from {args.model_path}...")
        text_encoder = load_text_encoder(args.model_path)
        if text_encoder:
            report = analyze_model(text_encoder, prefix="text_encoder")
            results["text_encoder"] = report
            if not args.json:
                print("\n--- TEXT ENCODER ---")
                print_report(report, verbose=args.verbose)

    if args.json:
        output = {}
        for name, report in results.items():
            output[name] = {
                "total_layers": report.total_linear_layers,
                "compatible_layers": report.compatible_layers,
                "incompatible_layers": report.incompatible_layers,
                "compatibility_rate": report.compatibility_rate,
                "total_params": report.total_params,
                "compatible_params": report.compatible_params,
                "param_coverage": report.param_coverage,
                "incompatible_layer_names": [l.name for l in report.layers if not l.fp8_compatible],
            }
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
