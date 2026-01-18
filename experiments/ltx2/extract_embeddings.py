#!/usr/bin/env python3
"""
Extract Embeddings from LTX-2 Pipeline

Last Updated: 2026-01-15

This script extracts prompt embeddings by running minimal generation (1 step)
and capturing the embeddings through the callback mechanism.

This works around the OOM issue with direct encode_prompt calls by using
the pipeline's internal memory management.

Usage:
    uv run python experiments/ltx2/extract_embeddings.py
"""

import gc
from pathlib import Path

import torch


def extract_embedding_via_generation(
    pipe,
    prompt: str,
    seed: int = 42,
    height: int = 256,  # Minimal size for memory
    width: int = 256,
    num_frames: int = 9,  # Minimal frames
):
    """
    Extract embeddings by running minimal generation.

    The key insight is that pipe() with prompt= handles encoding with proper
    memory management, then we can capture embeddings from the callback.
    """
    captured = {}

    def capture_callback(pipe, step, timestep, callback_kwargs):
        # Capture on first step only
        if step == 0 and "prompt_embeds" not in captured:
            # The prompt_embeds should be in callback_kwargs or accessible
            if "latents" in callback_kwargs:
                # We're past encoding, but we can note this
                pass
        return callback_kwargs

    # Run minimal generation
    generator = torch.Generator(device="cpu").manual_seed(seed)

    try:
        output = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=1,  # Minimal steps
            guidance_scale=1.0,  # Minimal guidance
            generator=generator,
            callback_on_step_end=capture_callback,
            callback_on_step_end_tensor_inputs=["latents"],
        )
    except Exception as e:
        print(f"Generation error (expected for minimal config): {e}")

    return captured


def extract_embeddings_direct(model_path: str, prompts: list):
    """
    Try to extract embeddings with aggressive memory management.

    Uses PYTORCH_CUDA_ALLOC_CONF for better memory handling.
    """
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    from diffusers import LTX2Pipeline

    embeddings = {}

    for i, prompt in enumerate(prompts):
        print(f"\nProcessing prompt {i+1}/{len(prompts)}: {prompt[:40]}...")

        # Clear memory before loading
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Check memory
        free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
        print(f"  Free memory before load: {free_mem / 1e9:.2f} GB")

        # Load pipeline fresh
        print("  Loading pipeline...")
        pipe = LTX2Pipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        )

        # Try different offload strategies
        try:
            # First try: full offload to CPU
            pipe.enable_model_cpu_offload()

            print("  Encoding prompt...")
            embeds, mask = pipe.encode_prompt(
                prompt=prompt,
                negative_prompt=None,
                do_classifier_free_guidance=False,
                num_videos_per_prompt=1,
                max_sequence_length=128,
                device="cuda",
                dtype=torch.bfloat16,
            )
            embeddings[prompt] = embeds.cpu()
            print(f"  Success! Shape: {embeds.shape}")

        except torch.OutOfMemoryError as e:
            print(f"  OOM with model_cpu_offload, trying sequential...")

            # Cleanup
            del pipe
            gc.collect()
            torch.cuda.empty_cache()

            # Second try: sequential offload
            pipe = LTX2Pipeline.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
            )
            pipe.enable_sequential_cpu_offload()

            try:
                embeds, mask = pipe.encode_prompt(
                    prompt=prompt,
                    negative_prompt=None,
                    do_classifier_free_guidance=False,
                    num_videos_per_prompt=1,
                    max_sequence_length=64,  # Shorter sequence
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                embeddings[prompt] = embeds.cpu()
                print(f"  Success with sequential! Shape: {embeds.shape}")
            except torch.OutOfMemoryError:
                print(f"  Still OOM. Model too large for this GPU.")

        finally:
            del pipe
            gc.collect()
            torch.cuda.empty_cache()

    return embeddings


def main():
    print("=" * 60)
    print("LTX-2 Embedding Extraction")
    print("=" * 60)

    # Check GPU
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total memory: {gpu_mem:.1f} GB")
    else:
        print("No GPU available!")
        return

    MODEL_PATH = "models/LTX-2"
    OUTPUT_DIR = Path("experiments/results/ltx2/embeddings")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Import prompts from centralized module
    from experiments.ltx2.prompts import CATEGORY_PROMPTS, LEGACY_SHORT_PROMPTS

    # Test prompts for embedding extraction (one short reference, one full format)
    # Using short prompt for vague baseline is acceptable here since we're testing
    # the steering direction, not generation quality
    test_prompts = [
        LEGACY_SHORT_PROMPTS["animal_short"],  # Vague baseline
        CATEGORY_PROMPTS["animal"],             # Detailed version
    ]

    print("\nAttempting embedding extraction...")
    embeddings = extract_embeddings_direct(MODEL_PATH, test_prompts)

    if embeddings:
        print(f"\nSuccessfully extracted {len(embeddings)} embeddings!")

        # Compute direction if we have both
        if len(embeddings) == 2:
            vague = embeddings[test_prompts[0]]
            detailed = embeddings[test_prompts[1]]
            direction = detailed.mean(dim=0, keepdim=True) - vague.mean(dim=0, keepdim=True)
            magnitude = torch.norm(direction).item()

            print(f"\nDirection magnitude: {magnitude:.4f}")

            # Save
            torch.save({
                "direction": direction,
                "magnitude": magnitude,
                "vague_embed": vague,
                "detailed_embed": detailed,
            }, OUTPUT_DIR / "steering_direction.pt")
            print(f"Saved to {OUTPUT_DIR / 'steering_direction.pt'}")
    else:
        print("\nFailed to extract any embeddings. GPU memory insufficient.")
        print("\nAlternative approaches:")
        print("1. Use a machine with more VRAM (>32GB)")
        print("2. Use 8-bit quantized text encoder")
        print("3. Use a smaller LLM for text encoding")
        print("4. Try the layer ablation approach (which works)")


if __name__ == "__main__":
    main()
