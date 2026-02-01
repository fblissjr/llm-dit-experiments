#!/usr/bin/env python3
"""
Generate encoder baseline fixtures for regression testing.

Last Updated: 2026-02-01

This script generates pre-computed encoder outputs that are used by
tests/integration/test_encoder_baselines.py to detect regressions.

Usage:
    # Generate baselines (requires GPU and models)
    uv run scripts/generate_encoder_baselines.py

    # Custom output path
    uv run scripts/generate_encoder_baselines.py --output custom_path.pt

    # Specify models explicitly
    uv run scripts/generate_encoder_baselines.py --zimage-model /path/to/z-image --flux2-model Qwen/Qwen3-4B-FP8

Requirements:
    - CUDA GPU with sufficient VRAM (~8GB for one encoder at a time)
    - Z-Image model weights (local or HuggingFace)
    - Qwen3-4B-FP8 or Qwen3-8B-FP8 for FLUX.2

Output format:
    {
        "zimage": {
            "embeddings": Tensor[seq_len, 2560],
            "prompt": str,
            "config": {...}
        },
        "flux2": {
            "embeddings": Tensor[1, seq_len, 7680],
            "prompt": str,
            "config": {...}
        },
        "thinking_comparison": {
            "thinking_true": Tensor,
            "thinking_false": Tensor,
            "prompt": str
        },
        "metadata": {
            "generated_at": str,
            "torch_version": str,
            "cuda_version": str
        }
    }
"""

import argparse
import datetime
import gc
import sys
from pathlib import Path

import torch

# Default test prompt (short but representative)
TEST_PROMPT = "A photograph of a cat sitting on a windowsill, natural lighting"


def generate_zimage_baseline(
    model_path: str = "Tongyi-MAI/Z-Image-Turbo",
    prompt: str = TEST_PROMPT,
) -> dict:
    """Generate Z-Image encoder baseline."""
    print(f"[Z-Image] Loading encoder from {model_path}...")

    try:
        from llm_dit.encoders.qwen3 import Qwen3Encoder
    except ImportError as e:
        print(f"[Z-Image] SKIP: {e}")
        return None

    try:
        encoder = Qwen3Encoder.from_pretrained(
            model_path=model_path,
            device="cuda",
            dtype="bfloat16",
        )

        print(f"[Z-Image] Encoding prompt: {prompt[:50]}...")
        output = encoder.encode([prompt], layer_index=-2)

        # Get first (and only) embedding
        embeddings = output.embeddings[0].cpu()

        result = {
            "embeddings": embeddings,
            "prompt": prompt,
            "config": {
                "model_path": model_path,
                "layer_index": -2,
                "enable_thinking": True,  # Z-Image default
                "dtype": "bfloat16",
            },
            "shape": list(embeddings.shape),
            "mean": embeddings.float().mean().item(),
            "std": embeddings.float().std().item(),
        }

        print(f"[Z-Image] Generated baseline: shape={embeddings.shape}, "
              f"mean={result['mean']:.4f}, std={result['std']:.4f}")

        # Cleanup
        encoder.offload()
        del encoder
        gc.collect()
        torch.cuda.empty_cache()

        return result

    except Exception as e:
        print(f"[Z-Image] ERROR: {e}")
        return None


def generate_flux2_baseline(
    model_path: str = "Qwen/Qwen3-4B-FP8",
    prompt: str = TEST_PROMPT,
) -> dict:
    """Generate FLUX.2 Klein encoder baseline."""
    print(f"[FLUX.2] Loading encoder from {model_path}...")

    try:
        from llm_dit.encoders.qwen3_flux2 import Qwen3Flux2Encoder
    except ImportError as e:
        print(f"[FLUX.2] SKIP: {e}")
        return None

    try:
        encoder = Qwen3Flux2Encoder.from_pretrained(
            model_spec=model_path,
            device="cuda",
        )

        print(f"[FLUX.2] Encoding prompt: {prompt[:50]}...")
        embeddings = encoder.encode([prompt])

        # Move to CPU
        embeddings_cpu = embeddings.cpu()

        result = {
            "embeddings": embeddings_cpu,
            "prompt": prompt,
            "config": {
                "model_path": model_path,
                "output_layers": encoder.output_layers,
                "enable_thinking": False,  # CRITICAL for FLUX.2
                "output_dim": encoder.output_dim,
                "hidden_dim": encoder.hidden_dim,
            },
            "shape": list(embeddings_cpu.shape),
            "mean": embeddings_cpu.float().mean().item(),
            "std": embeddings_cpu.float().std().item(),
        }

        print(f"[FLUX.2] Generated baseline: shape={embeddings_cpu.shape}, "
              f"mean={result['mean']:.4f}, std={result['std']:.4f}")

        # Cleanup
        encoder.offload()
        del encoder
        gc.collect()
        torch.cuda.empty_cache()

        return result

    except Exception as e:
        print(f"[FLUX.2] ERROR: {e}")
        return None


def generate_thinking_comparison(
    model_path: str = "Qwen/Qwen3-4B-FP8",
    prompt: str = TEST_PROMPT,
) -> dict:
    """Generate comparison between enable_thinking=True vs False.

    This validates that the enable_thinking parameter actually affects output.
    """
    print(f"[Thinking Comparison] Loading model from {model_path}...")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        print(f"[Thinking Comparison] SKIP: {e}")
        return None

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Format with enable_thinking=True
        messages = [{"role": "user", "content": prompt}]
        text_true = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        inputs_true = tokenizer(text_true, return_tensors="pt", padding=True)

        # Format with enable_thinking=False
        text_false = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs_false = tokenizer(text_false, return_tensors="pt", padding=True)

        # Get embeddings for both
        with torch.no_grad():
            outputs_true = model(
                input_ids=inputs_true["input_ids"].to("cuda"),
                attention_mask=inputs_true["attention_mask"].to("cuda"),
                output_hidden_states=True,
            )
            outputs_false = model(
                input_ids=inputs_false["input_ids"].to("cuda"),
                attention_mask=inputs_false["attention_mask"].to("cuda"),
                output_hidden_states=True,
            )

        # Use layer 9 for comparison (middle-ish layer)
        layer_idx = 9
        emb_true = outputs_true.hidden_states[layer_idx].cpu()
        emb_false = outputs_false.hidden_states[layer_idx].cpu()

        result = {
            "thinking_true": emb_true,
            "thinking_false": emb_false,
            "prompt": prompt,
            "layer_idx": layer_idx,
            "text_with_thinking": text_true,
            "text_without_thinking": text_false,
        }

        # Report difference
        diff = (emb_true.float().mean() - emb_false.float().mean()).abs()
        print(f"[Thinking Comparison] Mean difference: {diff:.6f}")

        # Cleanup
        del model
        gc.collect()
        torch.cuda.empty_cache()

        return result

    except Exception as e:
        print(f"[Thinking Comparison] ERROR: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate encoder baseline fixtures for regression testing."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tests/fixtures/encoder_baselines.pt",
        help="Output path for baseline fixtures",
    )
    parser.add_argument(
        "--zimage-model",
        type=str,
        default="Tongyi-MAI/Z-Image-Turbo",
        help="Z-Image model path",
    )
    parser.add_argument(
        "--flux2-model",
        type=str,
        default="Qwen/Qwen3-4B-FP8",
        help="FLUX.2 encoder model path",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=TEST_PROMPT,
        help="Test prompt for baseline generation",
    )
    parser.add_argument(
        "--skip-zimage",
        action="store_true",
        help="Skip Z-Image baseline generation",
    )
    parser.add_argument(
        "--skip-flux2",
        action="store_true",
        help="Skip FLUX.2 baseline generation",
    )
    parser.add_argument(
        "--skip-thinking",
        action="store_true",
        help="Skip thinking comparison generation",
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. GPU required for baseline generation.")
        sys.exit(1)

    baselines = {}

    # Generate Z-Image baseline
    if not args.skip_zimage:
        zimage_baseline = generate_zimage_baseline(
            model_path=args.zimage_model,
            prompt=args.prompt,
        )
        if zimage_baseline:
            baselines["zimage"] = zimage_baseline

    # Generate FLUX.2 baseline
    if not args.skip_flux2:
        flux2_baseline = generate_flux2_baseline(
            model_path=args.flux2_model,
            prompt=args.prompt,
        )
        if flux2_baseline:
            baselines["flux2"] = flux2_baseline

    # Generate thinking comparison
    if not args.skip_thinking:
        thinking_comp = generate_thinking_comparison(
            model_path=args.flux2_model,  # Use same Qwen3 model
            prompt=args.prompt,
        )
        if thinking_comp:
            baselines["thinking_comparison"] = thinking_comp

    # Add metadata
    baselines["metadata"] = {
        "generated_at": datetime.datetime.now().isoformat(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "prompt": args.prompt,
    }

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(baselines, output_path)
    print(f"\nBaselines saved to: {output_path}")
    print(f"Keys: {list(baselines.keys())}")


if __name__ == "__main__":
    main()
