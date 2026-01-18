#!/usr/bin/env python3
"""
LTX-2 Layer Extraction Comparison Experiment

Last Updated: 2026-01-16

Zero-training technique: Test which Gemma-3 layers matter for which visual attributes.

Hypothesis from LLM research:
- Early layers (0-16): Phonetic/syntactic information
- Middle layers (17-32): Semantic meaning
- Late layers (33-48): Abstract/high-level concepts

For DiT conditioning, this may translate to:
- Early: Text rendering, literal descriptions
- Middle: Object/scene semantics
- Late: Style, composition, abstract qualities

LTX-2 uses all 49 Gemma layers with uniform blending. This experiment tests
selective layer usage by zeroing out contributions from layer subsets.

Usage:
    uv run python experiments/ltx2/layer_extraction_comparison.py
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import prompts from centralized module
# These match the official LTX-2 prompting guide format (100+ words, dialogue, etc.)
from experiments.ltx2.prompts import CATEGORY_PROMPTS

# Use category prompts for layer extraction comparison
# Map to prompt types: semantic (animal), spatial (urban), abstract (abstract)
TEST_PROMPTS = [
    CATEGORY_PROMPTS["animal"],    # Semantic/concrete - should benefit from middle layers
    CATEGORY_PROMPTS["urban"],     # Spatial/compositional
    CATEGORY_PROMPTS["abstract"],  # Abstract/style - should benefit from late layers
]

PROMPT_TYPES = ["semantic", "spatial", "abstract"]

# Layer configurations to test
# LTX-2 uses 49 Gemma layers (0-48)
LAYER_CONFIGS = {
    "all_layers": list(range(49)),  # Baseline: all layers
    "early_only": list(range(17)),  # Layers 0-16
    "middle_only": list(range(17, 33)),  # Layers 17-32
    "late_only": list(range(33, 49)),  # Layers 33-48
    "early_middle": list(range(33)),  # Layers 0-32
    "middle_late": list(range(17, 49)),  # Layers 17-48
    "no_early": list(range(17, 49)),  # Skip early layers
    "no_late": list(range(33)),  # Skip late layers
}


def compute_frame_statistics(frames: list) -> dict:
    """Compute statistics on generated frames."""
    frame_arrays = [np.array(f) for f in frames]
    stacked = np.stack(frame_arrays, axis=0)  # [T, H, W, C]

    return {
        "mean": float(stacked.mean()),
        "std": float(stacked.std()),
        "min": float(stacked.min()),
        "max": float(stacked.max()),
        "temporal_variance": float(stacked.var(axis=0).mean()),
    }


def run_layer_extraction_experiment(
    output_dir: str = "experiments/results/ltx2",
    model_path: str = "models/LTX-2",
    save_videos: bool = True,
    configs_to_test: list = None,
):
    """
    Run layer extraction comparison experiment.

    Tests which layer subsets contribute most to generation quality
    by masking out contributions from excluded layers.

    Architecture insight (LTX-2):
    - text_encoder(output_hidden_states=True) → 49-tuple of [B, T, 3840]
    - torch.stack(dim=-1) → [B, T, 3840, 49]
    - _pack_text_embeds() → normalize per-layer, flatten → [B, T, 188160]

    We hook into _get_gemma_prompt_embeds to mask specific layers.
    """
    from diffusers import LTX2Pipeline
    from diffusers.utils import export_to_video

    if configs_to_test is None:
        configs_to_test = ["all_layers", "early_only", "middle_only", "late_only"]

    print("=" * 60)
    print("LTX-2 Layer Extraction Comparison")
    print("=" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load pipeline
    print("\nLoading pipeline...")
    pipe = LTX2Pipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )
    # Use model_cpu_offload instead of sequential_cpu_offload for 2-3x speedup
    pipe.enable_model_cpu_offload()

    results = {}

    for config_name in configs_to_test:
        if config_name not in LAYER_CONFIGS:
            print(f"Unknown config: {config_name}, skipping")
            continue

        active_layers = LAYER_CONFIGS[config_name]
        print(f"\n{'='*50}")
        print(f"Config: {config_name}")
        print(f"Active layers: {len(active_layers)} ({min(active_layers)}-{max(active_layers)})")
        print("=" * 50)

        config_results = []

        # Create layer masking hook
        original_get_embeds = pipe._get_gemma_prompt_embeds

        def masked_get_embeds(*args, **kwargs):
            """Hook to mask specific layers after text encoder."""
            # Call original method's internal logic
            prompt = kwargs.get("prompt", args[0] if args else None)
            num_videos_per_prompt = kwargs.get("num_videos_per_prompt", 1)
            max_sequence_length = kwargs.get("max_sequence_length", 1024)
            scale_factor = kwargs.get("scale_factor", 8)
            device = kwargs.get("device", pipe._execution_device)
            dtype = kwargs.get("dtype", pipe.text_encoder.dtype)

            prompt = [prompt] if isinstance(prompt, str) else prompt
            batch_size = len(prompt)

            # Tokenize
            text_inputs = pipe.tokenizer(
                prompt,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids.to(device)
            prompt_attention_mask = text_inputs.attention_mask.to(device)

            # Get hidden states from all layers
            text_encoder_outputs = pipe.text_encoder(
                input_ids=text_input_ids,
                attention_mask=prompt_attention_mask,
                output_hidden_states=True,
            )
            text_encoder_hidden_states = text_encoder_outputs.hidden_states
            text_encoder_hidden_states = torch.stack(text_encoder_hidden_states, dim=-1)
            # Shape: [batch, seq, hidden_dim, num_layers] = [B, T, 3840, 49]

            # LAYER MASKING: Soft masking (replace with mean, not zero)
            # Zeroing creates OOD inputs; soft masking preserves distribution
            for layer_idx in range(49):
                if layer_idx not in active_layers:
                    # Replace with mean across sequence (preserves layer statistics)
                    layer_mean = text_encoder_hidden_states[:, :, :, layer_idx].mean(dim=1, keepdim=True)
                    text_encoder_hidden_states[:, :, :, layer_idx] = layer_mean

            sequence_lengths = prompt_attention_mask.sum(dim=-1)

            # Pack text embeds (normalize and flatten)
            prompt_embeds = pipe._pack_text_embeds(
                text_encoder_hidden_states,
                sequence_lengths,
                device=device,
                padding_side=pipe.tokenizer.padding_side,
                scale_factor=scale_factor,
            )
            prompt_embeds = prompt_embeds.to(dtype=dtype)

            # Duplicate for multiple videos per prompt
            _, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

            prompt_attention_mask = prompt_attention_mask.view(batch_size, -1)
            prompt_attention_mask = prompt_attention_mask.repeat(num_videos_per_prompt, 1)

            return prompt_embeds, prompt_attention_mask

        # Install hook
        pipe._get_gemma_prompt_embeds = masked_get_embeds

        for i, prompt in enumerate(TEST_PROMPTS):
            print(f"\n  [{i+1}/{len(TEST_PROMPTS)}] {prompt[:50]}...")

            start_time = time.time()

            try:
                generator = torch.Generator(device="cpu").manual_seed(42 + i)

                output = pipe(
                    prompt=prompt,
                    height=512,
                    width=768,
                    num_frames=33,
                    num_inference_steps=25,
                    guidance_scale=3.0,
                    generator=generator,
                )

                frames = output.frames[0]
                gen_time = time.time() - start_time

                # Compute statistics
                stats = compute_frame_statistics(frames)
                stats["generation_time"] = gen_time
                stats["prompt"] = prompt
                stats["prompt_type"] = PROMPT_TYPES[i]
                config_results.append(stats)

                print(f"  Time: {gen_time:.1f}s | Mean: {stats['mean']:.1f} | Std: {stats['std']:.1f}")

                # Save video
                if save_videos:
                    video_path = output_path / f"{config_name}_sample{i}.mp4"
                    export_to_video(frames, str(video_path), fps=24)
                    print(f"  Saved: {video_path.name}")

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                config_results.append({"error": str(e), "prompt": prompt})

            # Memory cleanup
            gc.collect()
            torch.cuda.empty_cache()

        # Restore original method
        pipe._get_gemma_prompt_embeds = original_get_embeds

        results[config_name] = config_results

    # Analysis
    print("\n" + "=" * 60)
    print("LAYER EXTRACTION ANALYSIS")
    print("=" * 60)

    # Summary table
    print("\n## Per-Config Statistics\n")
    print("| Config | Layers | Mean | Std | Temp Var |")
    print("|--------|--------|------|-----|----------|")

    for config_name, config_results in results.items():
        valid = [s for s in config_results if "mean" in s]
        if valid:
            mean_avg = np.mean([s["mean"] for s in valid])
            std_avg = np.mean([s["std"] for s in valid])
            temp_avg = np.mean([s["temporal_variance"] for s in valid])
            n_layers = len(LAYER_CONFIGS.get(config_name, []))
            print(f"| {config_name:14} | {n_layers:6} | {mean_avg:5.1f} | {std_avg:4.1f} | {temp_avg:8.1f} |")

    # Per-prompt-type analysis
    print("\n## Layer Effects by Prompt Type\n")

    for ptype in PROMPT_TYPES:
        print(f"\n### {ptype.capitalize()} prompt")
        print("| Config | Mean | Std |")
        print("|--------|------|-----|")
        for config_name, config_results in results.items():
            matching = [s for s in config_results if s.get("prompt_type") == ptype and "mean" in s]
            if matching:
                s = matching[0]
                print(f"| {config_name:14} | {s['mean']:5.1f} | {s['std']:4.1f} |")

    # Save results
    results_file = output_path / "layer_extraction_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # Cleanup
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="LTX-2 Layer Extraction Comparison")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results/ltx2",
        help="Output directory",
    )
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        default=["all_layers", "early_only", "middle_only", "late_only"],
        help="Layer configurations to test",
    )
    parser.add_argument(
        "--no-save-videos",
        action="store_true",
        help="Don't save output videos",
    )
    args = parser.parse_args()

    run_layer_extraction_experiment(
        output_dir=args.output_dir,
        configs_to_test=args.configs,
        save_videos=not args.no_save_videos,
    )


if __name__ == "__main__":
    main()
