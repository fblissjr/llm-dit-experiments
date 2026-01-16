#!/usr/bin/env python3
"""
LTX-2 Enhancement Technique Ablation Study

Last Updated: 2026-01-16

Systematically tests each enhancement technique in isolation and in combination
to measure their impact on generation quality and performance.

Enhancement Techniques Tested:
1. Latent Normalization - Prevents CFG-induced drift
2. Audio Normalization - Per-step audio latent scaling
3. FFN Chunking - Memory reduction via chunked feedforward

Metrics Collected:
- Generation time (seconds)
- Peak VRAM usage (GB)
- Visual quality (subjective, can add SigLIP score)

Usage:
    # Full ablation (all combinations)
    uv run python experiments/ltx2/technique_ablation.py

    # Quick test (baseline + individual techniques)
    uv run python experiments/ltx2/technique_ablation.py --quick

    # Specific techniques only
    uv run python experiments/ltx2/technique_ablation.py --techniques latent_norm ffn_chunk

    # Custom prompt
    uv run python experiments/ltx2/technique_ablation.py --prompt "A dog running on a beach"
"""

import argparse
import gc
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Optional

import torch

from llm_dit.config import EnhancementConfig
from llm_dit.pipelines.ltx2 import LTX2Pipeline


@dataclass
class AblationResult:
    """Result from a single ablation run."""

    config_name: str
    techniques_enabled: list[str]
    generation_time_seconds: float
    peak_vram_gb: float
    prompt: str
    seed: int
    output_path: str
    enhancement_config: dict


@dataclass
class AblationStudy:
    """Complete ablation study results."""

    study_name: str
    timestamp: str
    model_path: str
    prompt: str
    seed: int
    resolution: tuple[int, int]
    num_frames: int
    num_steps: int
    results: list[AblationResult]

    def to_dict(self) -> dict:
        return {
            "study_name": self.study_name,
            "timestamp": self.timestamp,
            "model_path": self.model_path,
            "prompt": self.prompt,
            "seed": self.seed,
            "resolution": self.resolution,
            "num_frames": self.num_frames,
            "num_steps": self.num_steps,
            "results": [asdict(r) for r in self.results],
        }

    def summary_table(self) -> str:
        """Generate a summary table of results."""
        lines = []
        lines.append("=" * 80)
        lines.append("Ablation Study Summary")
        lines.append("=" * 80)
        lines.append(f"Prompt: {self.prompt}")
        lines.append(f"Resolution: {self.resolution[0]}x{self.resolution[1]}, {self.num_frames} frames")
        lines.append("-" * 80)
        lines.append(f"{'Config':<30} {'Time (s)':<12} {'VRAM (GB)':<12} {'Techniques'}")
        lines.append("-" * 80)

        for r in self.results:
            techniques = ", ".join(r.techniques_enabled) if r.techniques_enabled else "none"
            lines.append(
                f"{r.config_name:<30} {r.generation_time_seconds:<12.2f} "
                f"{r.peak_vram_gb:<12.2f} {techniques}"
            )

        lines.append("=" * 80)
        return "\n".join(lines)


def get_peak_vram() -> float:
    """Get peak VRAM usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    return 0.0


def reset_vram_stats():
    """Reset VRAM tracking."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    gc.collect()


def run_single_ablation(
    pipe: LTX2Pipeline,
    config_name: str,
    enhancement_config: EnhancementConfig,
    prompt: str,
    seed: int,
    output_dir: Path,
    height: int = 512,
    width: int = 768,
    num_frames: int = 33,
    num_steps: int = 12,
    guidance_scale: float = 3.5,
) -> AblationResult:
    """Run a single ablation configuration."""

    # Build list of enabled techniques
    techniques_enabled = []
    if enhancement_config.latent_norm_enabled:
        techniques_enabled.append("latent_norm")
    if enhancement_config.audio_norm_enabled:
        techniques_enabled.append("audio_norm")
    if enhancement_config.ffn_chunking_enabled:
        techniques_enabled.append("ffn_chunk")
    if enhancement_config.nag_enabled:
        techniques_enabled.append("nag")
    if enhancement_config.feta_enabled:
        techniques_enabled.append("feta")
    if enhancement_config.tea_cache_enabled:
        techniques_enabled.append("tea_cache")

    print(f"\n{'='*60}")
    print(f"Running: {config_name}")
    print(f"Techniques: {', '.join(techniques_enabled) if techniques_enabled else 'none'}")
    print(f"{'='*60}")

    # Reset VRAM tracking
    reset_vram_stats()

    # Create generator for reproducibility
    generator = torch.Generator(device="cuda").manual_seed(seed)

    # Output path
    safe_name = config_name.replace(" ", "_").replace("+", "_")
    output_path = output_dir / f"{safe_name}_seed{seed}.mp4"

    # Run generation
    start_time = time.time()

    output = pipe(
        prompt=prompt,
        negative_prompt="worst quality, blurry, distorted, inconsistent motion",
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        enhancement_config=enhancement_config,
        return_dict=True,
    )

    generation_time = time.time() - start_time
    peak_vram = get_peak_vram()

    # Save video
    output_dir.mkdir(parents=True, exist_ok=True)
    pipe.save_video(output, str(output_path))

    print(f"Generated in {generation_time:.2f}s, peak VRAM: {peak_vram:.2f}GB")
    print(f"Saved to: {output_path}")

    return AblationResult(
        config_name=config_name,
        techniques_enabled=techniques_enabled,
        generation_time_seconds=generation_time,
        peak_vram_gb=peak_vram,
        prompt=prompt,
        seed=seed,
        output_path=str(output_path),
        enhancement_config=enhancement_config.to_dict(),
    )


def build_ablation_configs(
    techniques: list[str],
    full_ablation: bool = True,
) -> list[tuple[str, EnhancementConfig]]:
    """
    Build list of ablation configurations.

    Args:
        techniques: List of technique names to include
        full_ablation: If True, include all combinations; if False, only individual

    Returns:
        List of (name, config) tuples
    """
    configs = []

    # Always include baseline
    configs.append(("baseline", EnhancementConfig()))

    # Map technique names to config attributes
    technique_attrs = {
        "latent_norm": ("latent_norm_enabled", True),
        "audio_norm": ("audio_norm_enabled", True),
        "ffn_chunk": ("ffn_chunking_enabled", True),
        "nag": ("nag_enabled", True),
        "feta": ("feta_enabled", True),
        "tea_cache": ("tea_cache_enabled", True),
    }

    # Filter to requested techniques
    available_techniques = [t for t in techniques if t in technique_attrs]

    if full_ablation:
        # All possible combinations
        for r in range(1, len(available_techniques) + 1):
            for combo in combinations(available_techniques, r):
                name = " + ".join(combo)
                config = EnhancementConfig()
                for tech in combo:
                    attr, value = technique_attrs[tech]
                    setattr(config, attr, value)
                configs.append((name, config))
    else:
        # Individual techniques only
        for tech in available_techniques:
            attr, value = technique_attrs[tech]
            config = EnhancementConfig()
            setattr(config, attr, value)
            configs.append((tech, config))

    return configs


def main():
    parser = argparse.ArgumentParser(description="LTX-2 Enhancement Technique Ablation")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/LTX-2",
        help="Path to LTX-2 model directory",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A cat walking through a sunny garden, realistic",
        help="Text prompt for video generation",
    )
    parser.add_argument("--height", type=int, default=512, help="Video height")
    parser.add_argument("--width", type=int, default=768, help="Video width")
    parser.add_argument("--num-frames", type=int, default=33, help="Number of frames")
    parser.add_argument("--steps", type=int, default=12, help="Inference steps")
    parser.add_argument("--guidance-scale", type=float, default=3.5, help="CFG scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/ltx2_ablation",
        help="Output directory for videos and results",
    )
    parser.add_argument(
        "--techniques",
        nargs="+",
        default=["latent_norm", "audio_norm", "ffn_chunk"],
        help="Techniques to include in ablation",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: baseline + individual techniques only (no combinations)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full mode: all possible combinations (can be slow)",
    )

    args = parser.parse_args()

    # Set up output directory
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"ablation_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LTX-2 Enhancement Technique Ablation Study")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Prompt: {args.prompt}")
    print(f"Resolution: {args.width}x{args.height}, {args.num_frames} frames")
    print(f"Steps: {args.steps}, Guidance: {args.guidance_scale}")
    print(f"Seed: {args.seed}")
    print(f"Techniques: {', '.join(args.techniques)}")
    print(f"Output: {run_dir}")
    print("=" * 60)

    # Load pipeline
    print("\nLoading LTX-2 pipeline...")
    pipe = LTX2Pipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        enable_cpu_offload=True,
    )

    # Build ablation configurations
    full_ablation = args.full and not args.quick
    configs = build_ablation_configs(args.techniques, full_ablation=full_ablation)

    print(f"\nRunning {len(configs)} configurations:")
    for name, _ in configs:
        print(f"  - {name}")

    # Run ablations
    results = []
    for config_name, enhancement_config in configs:
        result = run_single_ablation(
            pipe=pipe,
            config_name=config_name,
            enhancement_config=enhancement_config,
            prompt=args.prompt,
            seed=args.seed,
            output_dir=run_dir,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_steps=args.steps,
            guidance_scale=args.guidance_scale,
        )
        results.append(result)

    # Create study object
    study = AblationStudy(
        study_name=f"technique_ablation_{timestamp}",
        timestamp=timestamp,
        model_path=args.model_path,
        prompt=args.prompt,
        seed=args.seed,
        resolution=(args.width, args.height),
        num_frames=args.num_frames,
        num_steps=args.steps,
        results=results,
    )

    # Print summary
    print("\n" + study.summary_table())

    # Save results
    results_path = run_dir / "ablation_results.json"
    with open(results_path, "w") as f:
        json.dump(study.to_dict(), f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Cleanup
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    print("\nAblation study complete!")


if __name__ == "__main__":
    main()
