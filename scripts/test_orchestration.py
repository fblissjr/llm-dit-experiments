#!/usr/bin/env python3
"""
Test orchestration system at multiple levels.

Last Updated: 2026-01-12

Usage:
    uv run python scripts/test_orchestration.py --level unit      # No GPU
    uv run python scripts/test_orchestration.py --level component # Individual models
    uv run python scripts/test_orchestration.py --level full      # Full pipeline
"""

import argparse
import sys
from pathlib import Path


def test_unit():
    """Test orchestration mechanics without loading models."""
    print("\n=== UNIT TESTS (No GPU) ===\n")

    # Test imports
    print("1. Testing imports...")
    from llm_dit.orchestration import (
        FunctionStep,
        ModelPool,
        ModelSpec,
        Orchestrator,
        PipelineStep,
        StepInput,
        StepOutput,
    )
    from llm_dit.orchestration.adapters.wan_video import WanVideoAdapter

    print("   ✓ All imports successful")

    # Test ModelPool
    print("\n2. Testing ModelPool...")
    pool = ModelPool(vram_budget_gb=24.0)
    pool.register("test-model", ModelSpec())
    assert pool.status()["test-model"] == "unloaded"
    print("   ✓ ModelPool register/status works")

    # Test PipelineStep validation
    print("\n3. Testing PipelineStep validation...")

    class AddStep(PipelineStep):
        inputs = [StepInput("a", int), StepInput("b", int, default=10)]
        outputs = [StepOutput("sum", int)]
        required_models = []

        def execute(self, inputs, models):
            return {"sum": inputs["a"] + inputs["b"]}

    step = AddStep()
    validated = step.validate_inputs({"a": 5})
    assert validated == {"a": 5, "b": 10}, f"Got {validated}"
    result = step.execute(validated, {})
    assert result == {"sum": 15}
    print("   ✓ Input validation and defaults work")

    # Test Orchestrator chaining
    print("\n4. Testing Orchestrator chaining...")

    class DoubleStep(PipelineStep):
        inputs = [StepInput("sum", int)]
        outputs = [StepOutput("doubled", int)]
        required_models = []

        def execute(self, inputs, models):
            return {"doubled": inputs["sum"] * 2}

    orch = Orchestrator(pool)
    orch.add_step(AddStep())
    orch.add_step(DoubleStep())
    result = orch.run({"a": 5, "b": 3})
    assert result["doubled"] == 16, f"Got {result}"
    print("   ✓ Step chaining works (5+3=8, 8*2=16)")

    # Test FunctionStep
    print("\n5. Testing FunctionStep...")

    def triple(inputs, models):
        return {"tripled": inputs["doubled"] * 3}

    orch.add_step(
        FunctionStep(
            fn=triple,
            inputs=[StepInput("doubled", int)],
            outputs=[StepOutput("tripled", int)],
        )
    )
    result = orch.run({"a": 5, "b": 3})
    assert result["tripled"] == 48, f"Got {result}"
    print("   ✓ FunctionStep works (16*3=48)")

    # Test WanVideoAdapter declaration
    print("\n6. Testing WanVideoAdapter declaration...")
    adapter = WanVideoAdapter()
    assert "prompt" in [i.name for i in adapter.inputs]
    assert "video" in [o.name for o in adapter.outputs]
    # WanVideoAdapter manages its own models (doesn't use ModelPool)
    assert adapter.required_models == []
    print(f"   ✓ WanVideoAdapter: {len(adapter.inputs)} inputs, {len(adapter.outputs)} outputs")
    print(f"   ✓ Self-managed models (doesn't use ModelPool)")

    print("\n=== UNIT TESTS PASSED ===\n")
    return True


def test_component():
    """Test individual model components."""
    print("\n=== COMPONENT TESTS (GPU Required) ===\n")

    import torch

    if not torch.cuda.is_available():
        print("   ✗ CUDA not available, skipping component tests")
        return False

    print(f"   GPU: {torch.cuda.get_device_name()}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    wan_path = Path("./models/Wan2.1-T2V-1.3B").expanduser()
    if not wan_path.exists():
        print(f"   ✗ Wan weights not found at {wan_path}")
        return False

    # Test Text Encoder
    print("\n1. Testing Text Encoder (UMT5-XXL)...")
    try:
        import gc

        from llm_dit.models.wan_text_encoder import load_wan_text_encoder

        # Find weights file
        weights_path = wan_path / "models_t5_umt5-xxl-enc-bf16.safetensors"
        if not weights_path.exists():
            weights_path = wan_path / "models_t5_umt5-xxl-enc-bf16.pth"

        tokenizer_path = wan_path / "google" / "umt5-xxl"

        # Load on CPU first, then move to GPU
        print("   Loading on CPU first...")
        encoder = load_wan_text_encoder(
            checkpoint_path=str(weights_path),
            tokenizer_path=str(tokenizer_path),
            dtype=torch.bfloat16,
            device="cpu",  # Load on CPU first
        )
        print("   Moving to CUDA...")
        encoder = encoder.to("cuda")

        embeddings, mask = encoder.encode("A cat sitting on a table")
        print(f"   ✓ Text encoder loaded")
        print(f"   ✓ Embeddings shape: {embeddings.shape}")
        print(f"   ✓ VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

        del encoder
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"   ✗ Text encoder failed: {e}")

        traceback.print_exc()
        return False

    # Test VAE
    print("\n2. Testing VAE...")
    try:
        import gc

        from safetensors.torch import load_file

        from llm_dit.models.wan_vae import WanVAE

        vae = WanVAE()

        # Load checkpoint (keys are for inner VideoVAE model)
        vae_path = wan_path / "Wan2.1_VAE.safetensors"
        if not vae_path.exists():
            vae_path = wan_path / "Wan2.1_VAE.pth"

        state_dict = load_file(str(vae_path))
        # Load into inner model (checkpoint has no 'model.' prefix)
        vae.model.load_state_dict(state_dict)
        vae = vae.to("cuda", torch.bfloat16)

        # Test decode with random latents
        latents = torch.randn(1, 16, 3, 30, 52, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            video = vae.decode(latents)
        print(f"   ✓ VAE loaded")
        print(f"   ✓ Decoded shape: {video.shape}")
        print(f"   ✓ VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

        del vae
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"   ✗ VAE failed: {e}")
        import traceback

    except Exception as e:
        print(f"   ✗ VAE failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n=== COMPONENT TESTS PASSED ===\n")
    return True


def test_full():
    """Test full orchestration pipeline."""
    print("\n=== FULL INTEGRATION TEST ===\n")

    import torch

    if not torch.cuda.is_available():
        print("   ✗ CUDA not available")
        return False

    from llm_dit.orchestration import ModelPool, ModelSpec, Orchestrator
    from llm_dit.orchestration.adapters.wan_video import WanVideoAdapter

    wan_path = Path("./models/Wan2.1-T2V-1.3B").expanduser()
    humo_path = Path("./models/HuMo").expanduser()

    # Check for HuMo-1.7B (fits in 24GB VRAM) vs HuMo-17B (requires >24GB)
    humo_17b = humo_path / "HuMo-17B"
    humo_1_7b = humo_path / "HuMo-1.7B"

    if humo_1_7b.exists():
        humo_variant = "1.7B"
        print("   Using HuMo-1.7B (fits in 24GB VRAM)")
    elif humo_17b.exists():
        humo_variant = "17B"
        print("   ⚠ HuMo-17B found but requires >24GB VRAM")
        print("   ⚠ Test may fail on 24GB GPU")
    else:
        print(f"   ✗ No HuMo weights found at {humo_path}")
        return False

    print(f"\n1. Setting up ModelPool...")
    pool = ModelPool(vram_budget_gb=24.0, auto_offload=True)

    print(f"\n2. Creating WanVideoAdapter...")
    # Note: humo_path is base path, variant is separate config
    adapter = WanVideoAdapter(
        humo_path=str(humo_path),
        wan_path=str(wan_path),
        humo_variant=humo_variant,
    )

    print(f"\n3. Building Orchestrator...")
    orch = Orchestrator(pool)
    orch.add_step(adapter)

    # Note: VAE temporal upsampling is broken (see wan_vae.py Resample class),
    # so latent_frames = video_frames. 5 frames fits in 24GB VRAM.
    print(f"\n4. Running generation (small test: 5 frames, 480x832)...")
    try:
        result = orch.run(
            {
                "prompt": "A cat sitting peacefully on a sunny windowsill",
                "num_frames": 5,  # Limited by VRAM without temporal compression
                "height": 480,
                "width": 832,
                "num_inference_steps": 10,  # Fast test
                "guidance_scale": 5.0,
                "seed": 42,
            }
        )

        video = result["video"]
        print(f"   ✓ Generation complete!")
        print(f"   ✓ Video shape: {video.frames.shape}")
        print(f"   ✓ FPS: {video.fps}")
        print(f"   ✓ Duration: {video.duration:.2f}s")

        # Save test output
        output_path = Path("outputs/test_orchestration.mp4")
        output_path.parent.mkdir(exist_ok=True)
        video.save(str(output_path))
        print(f"   ✓ Saved to {output_path}")

    except Exception as e:
        print(f"   ✗ Generation failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n=== FULL INTEGRATION TEST PASSED ===\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="Test orchestration system")
    parser.add_argument(
        "--level",
        choices=["unit", "component", "full", "all"],
        default="unit",
        help="Test level: unit (no GPU), component (individual models), full (pipeline), all",
    )
    args = parser.parse_args()

    results = {}

    if args.level in ("unit", "all"):
        results["unit"] = test_unit()

    if args.level in ("component", "all"):
        results["component"] = test_component()

    if args.level in ("full", "all"):
        results["full"] = test_full()

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for level, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {level}: {status}")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
