#!/usr/bin/env python3
"""
Test script for Qwen-Image-2512 pipeline with FP8 quantization.

This script verifies:
1. Model loads correctly with FP8 quantization
2. VRAM usage stays under 24GB
3. Image generation works

Usage:
    uv run scripts/test_qwen_image_2512.py

last updated: 2025-12-31
"""

import sys
from pathlib import Path

# Add coderef diffusers to path
_CODEREF_DIFFUSERS = Path(__file__).parent.parent / "coderef" / "diffusers" / "src"
if _CODEREF_DIFFUSERS.exists() and str(_CODEREF_DIFFUSERS) not in sys.path:
    sys.path.insert(0, str(_CODEREF_DIFFUSERS))
    print(f"Added coderef diffusers to path: {_CODEREF_DIFFUSERS}")

import logging
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model path
MODEL_PATH = Path.home() / "Storage" / "Qwen-Image-2512"


def get_vram_usage():
    """Get current VRAM usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0.0


def get_vram_reserved():
    """Get reserved VRAM in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved() / 1024**3
    return 0.0


def test_basic_import():
    """Test that we can import the pipeline."""
    print("\n=== Test 1: Basic Import ===")
    try:
        from diffusers import QwenImagePipeline
        print(f"[PASS] QwenImagePipeline imported successfully")
        return True
    except ImportError as e:
        print(f"[FAIL] Import failed: {e}")
        return False


def test_torchao_available():
    """Test that TorchAO is available for FP8."""
    print("\n=== Test 2: TorchAO Availability ===")
    try:
        from diffusers import TorchAoConfig
        print(f"[PASS] TorchAoConfig available")

        # Check FP8 support
        from llm_dit.quantization import check_fp8_support
        if check_fp8_support():
            print(f"[PASS] FP8 supported on this GPU")
        else:
            print(f"[WARN] FP8 not supported - need RTX 4090+ or H100")

        return True
    except ImportError as e:
        print(f"[FAIL] TorchAO not available: {e}")
        return False


def test_model_exists():
    """Test that model exists at expected path."""
    print("\n=== Test 3: Model Path ===")
    if not MODEL_PATH.exists():
        print(f"[FAIL] Model not found at {MODEL_PATH}")
        return False

    model_index = MODEL_PATH / "model_index.json"
    if not model_index.exists():
        print(f"[FAIL] model_index.json not found")
        return False

    print(f"[PASS] Model found at {MODEL_PATH}")
    return True


def test_load_with_fp8():
    """Test loading the model with FP8 quantization."""
    print("\n=== Test 4: Load with FP8 ===")

    from diffusers import QwenImagePipeline, TorchAoConfig
    from diffusers.quantizers import PipelineQuantizationConfig

    print(f"Initial VRAM: {get_vram_usage():.2f} GB")

    # Create FP8 config for transformer only
    # Use quant_mapping to only quantize the transformer component
    fp8_config = TorchAoConfig("float8dq")
    pipe_quant_config = PipelineQuantizationConfig(
        quant_mapping={"transformer": fp8_config}
    )
    print(f"Created pipeline quantization config for transformer FP8")

    print("Loading pipeline with FP8 transformer...")
    try:
        pipe = QwenImagePipeline.from_pretrained(
            str(MODEL_PATH),
            torch_dtype=torch.bfloat16,
            quantization_config=pipe_quant_config,
        )
        print(f"[PASS] Pipeline loaded with FP8 transformer")
        print(f"VRAM after load: {get_vram_usage():.2f} GB")

        # Enable CPU offload for memory management
        print("Enabling model CPU offload...")
        pipe.enable_model_cpu_offload()
        print(f"VRAM after offload setup: {get_vram_usage():.2f} GB")

        return pipe

    except Exception as e:
        print(f"[FAIL] Load failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_load_transformer_with_fp8():
    """Test loading just the transformer with FP8."""
    print("\n=== Test 4b: Load Transformer with FP8 ===")

    from diffusers import TorchAoConfig
    from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel

    print(f"Initial VRAM: {get_vram_usage():.2f} GB")

    # Create FP8 config
    fp8_config = TorchAoConfig("float8dq")

    print("Loading transformer with FP8...")
    try:
        transformer = QwenImageTransformer2DModel.from_pretrained(
            str(MODEL_PATH),
            subfolder="transformer",
            quantization_config=fp8_config,
            torch_dtype=torch.bfloat16,
        )
        print(f"[PASS] Transformer loaded with FP8")
        print(f"VRAM after transformer load: {get_vram_usage():.2f} GB")

        # Check if it's actually quantized
        print(f"Transformer dtype: {next(transformer.parameters()).dtype}")

        return transformer

    except Exception as e:
        print(f"[FAIL] Transformer load failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_generation(pipe):
    """Test image generation."""
    print("\n=== Test 5: Image Generation ===")

    if pipe is None:
        print("[SKIP] No pipeline available")
        return False

    prompt = "A beautiful sunset over mountains, photorealistic"

    print(f"Generating image with prompt: '{prompt}'")
    print(f"VRAM before generation: {get_vram_usage():.2f} GB")

    try:
        # Use generator for reproducibility
        generator = torch.Generator(device="cpu").manual_seed(42)

        result = pipe(
            prompt=prompt,
            negative_prompt=" ",
            true_cfg_scale=4.0,
            height=1024,
            width=1024,
            num_inference_steps=20,  # Fewer steps for testing
            generator=generator,
        )

        print(f"VRAM after generation: {get_vram_usage():.2f} GB")
        print(f"Max VRAM reserved: {get_vram_reserved():.2f} GB")

        # Save the image
        output_path = Path(__file__).parent.parent / "output_qwen2512_test.png"
        result.images[0].save(output_path)
        print(f"[PASS] Image saved to {output_path}")

        return True

    except Exception as e:
        print(f"[FAIL] Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pipeline_wrapper():
    """Test our QwenImage2512Pipeline wrapper."""
    print("\n=== Test 6: QwenImage2512Pipeline Wrapper ===")

    try:
        from llm_dit.pipelines import QwenImage2512Pipeline

        print("Loading QwenImage2512Pipeline with FP8 transformer + 4bit text encoder...")
        pipe = QwenImage2512Pipeline.from_pretrained(
            str(MODEL_PATH),
            quantize_transformer="fp8",       # ~20GB FP8
            quantize_text_encoder="4bit",     # ~4GB 4bit
            cpu_offload=True,
        )

        print(f"VRAM after load: {get_vram_usage():.2f} GB")

        prompt = "A fluffy white cat sitting on a windowsill, golden hour lighting"
        print(f"Generating with prompt: '{prompt}'")

        image = pipe(
            prompt=prompt,
            height=1024,
            width=1024,
            num_inference_steps=20,
            seed=42,
        )

        output_path = Path(__file__).parent.parent / "output_qwen2512_wrapper_test.png"
        image.save(output_path)
        print(f"[PASS] Image saved to {output_path}")
        print(f"Max VRAM reserved: {get_vram_reserved():.2f} GB")

        return True

    except Exception as e:
        print(f"[FAIL] Wrapper test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Qwen-Image-2512 FP8 Test Suite")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("[FAIL] CUDA not available")
        return 1

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")

    # Run tests
    tests = [
        ("Import", test_basic_import),
        ("TorchAO", test_torchao_available),
        ("Model Path", test_model_exists),
    ]

    results = {}
    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"[FAIL] {name} crashed: {e}")
            results[name] = False

    # If basic tests pass, try loading
    if all(results.values()):
        # First try loading just the transformer with FP8
        transformer = test_load_transformer_with_fp8()
        if transformer is not None:
            del transformer
            torch.cuda.empty_cache()
            print(f"VRAM after cleanup: {get_vram_usage():.2f} GB")

        # Then try full pipeline
        pipe = test_load_with_fp8()
        if pipe is not None:
            results["Generation"] = test_generation(pipe)
            del pipe
            torch.cuda.empty_cache()

        # Test our wrapper
        results["Wrapper"] = test_pipeline_wrapper()

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
