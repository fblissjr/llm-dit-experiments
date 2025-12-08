#!/usr/bin/env python3
"""
Profiling and stability testing script for Z-Image pipeline.

Uses the same configuration system as web server and CLI to ensure
consistent behavior across all entry points.

Usage:
    # Run all tests with default config
    uv run scripts/profile.py --model-path /path/to/z-image-turbo

    # Run specific tests
    uv run scripts/profile.py --model-path /path/to/z-image-turbo --tests encode,generate

    # Test different config combinations
    uv run scripts/profile.py --model-path /path/to/z-image-turbo --sweep

    # Save results to JSON
    uv run scripts/profile.py --model-path /path/to/z-image-turbo --output results.json

    # Verbose output with all timings
    uv run scripts/profile.py --model-path /path/to/z-image-turbo -v
"""

import argparse
import gc
import json
import logging
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_dit.cli import RuntimeConfig, create_base_parser, load_runtime_config, setup_logging

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result of a single test run."""
    name: str
    success: bool
    duration_ms: float
    error: str | None = None
    memory_before_mb: float = 0.0
    memory_after_mb: float = 0.0
    memory_peak_mb: float = 0.0
    extra: dict = field(default_factory=dict)


@dataclass
class ProfileResults:
    """Complete profiling session results."""
    timestamp: str
    config: dict
    system_info: dict
    tests: list[TestResult]
    summary: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "config": self.config,
            "system_info": self.system_info,
            "tests": [asdict(t) for t in self.tests],
            "summary": self.summary,
        }


def get_system_info() -> dict:
    """Collect system information for debugging."""
    info = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_total_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9

    try:
        import transformers
        info["transformers_version"] = transformers.__version__
    except ImportError:
        pass

    try:
        import diffusers
        info["diffusers_version"] = diffusers.__version__
    except ImportError:
        pass

    return info


def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e6
    return 0.0


def get_gpu_memory_peak_mb() -> float:
    """Get peak GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e6
    return 0.0


def reset_memory_stats():
    """Reset GPU memory stats and run garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


class Profiler:
    """Main profiler class for running tests."""

    def __init__(self, config: RuntimeConfig, verbose: bool = False):
        self.config = config
        self.verbose = verbose
        self.pipeline = None
        self.encoder = None
        self.results: list[TestResult] = []

    def log(self, msg: str):
        """Log if verbose mode is enabled."""
        if self.verbose:
            print(f"  {msg}")

    def run_test(self, name: str, func, *args, **kwargs) -> TestResult:
        """Run a single test with timing and memory tracking."""
        print(f"\n[TEST] {name}")
        reset_memory_stats()
        memory_before = get_gpu_memory_mb()

        start_time = time.perf_counter()
        success = False
        error = None
        extra = {}

        try:
            result = func(*args, **kwargs)
            success = True
            if isinstance(result, dict):
                extra = result
        except Exception as e:
            error = f"{type(e).__name__}: {str(e)}"
            if self.verbose:
                traceback.print_exc()

        duration_ms = (time.perf_counter() - start_time) * 1000
        memory_after = get_gpu_memory_mb()
        memory_peak = get_gpu_memory_peak_mb()

        status = "PASS" if success else "FAIL"
        print(f"  [{status}] {duration_ms:.1f}ms | Memory: {memory_before:.0f} -> {memory_after:.0f} MB (peak: {memory_peak:.0f} MB)")
        if error:
            print(f"  Error: {error}")

        result = TestResult(
            name=name,
            success=success,
            duration_ms=duration_ms,
            error=error,
            memory_before_mb=memory_before,
            memory_after_mb=memory_after,
            memory_peak_mb=memory_peak,
            extra=extra,
        )
        self.results.append(result)
        return result

    def load_pipeline(self) -> dict:
        """Load the full pipeline."""
        from llm_dit.pipelines.z_image import ZImagePipeline

        self.log(f"Loading pipeline from {self.config.model_path}")
        self.log(f"  Encoder device: {self.config.encoder_device_resolved}")
        self.log(f"  DiT device: {self.config.dit_device_resolved}")
        self.log(f"  VAE device: {self.config.vae_device_resolved}")

        self.pipeline = ZImagePipeline.from_pretrained(
            self.config.model_path,
            torch_dtype=self.config.get_torch_dtype(),
            text_encoder_device=self.config.encoder_device_resolved,
            dit_device=self.config.dit_device_resolved,
            vae_device=self.config.vae_device_resolved,
            templates_dir=self.config.templates_dir,
        )

        return {
            "encoder_device": str(self.pipeline.encoder.backend.device),
            "dit_device": str(self.pipeline.dit_device),
            "vae_device": str(self.pipeline.vae_device),
        }

    def load_encoder_only(self) -> dict:
        """Load just the text encoder."""
        from llm_dit.encoders.z_image import ZImageTextEncoder

        self.log(f"Loading encoder from {self.config.model_path}")

        self.encoder = ZImageTextEncoder.from_pretrained(
            self.config.model_path,
            templates_dir=self.config.templates_dir,
            device=self.config.encoder_device_resolved,
            torch_dtype=self.config.get_torch_dtype(),
        )

        return {
            "device": str(self.encoder.backend.device),
            "dtype": str(self.encoder.backend.dtype),
        }

    def test_encode_short(self) -> dict:
        """Test encoding a short prompt."""
        enc = self.encoder or (self.pipeline.encoder if self.pipeline else None)
        if not enc:
            raise RuntimeError("No encoder loaded")

        prompt = "A cat sleeping in sunlight"
        self.log(f"Encoding: {prompt}")

        output = enc.encode(prompt)
        token_count = output.token_counts[0]
        embed_shape = output.embeddings[0].shape

        return {
            "prompt_length": len(prompt),
            "token_count": token_count,
            "embedding_shape": list(embed_shape),
        }

    def test_encode_medium(self) -> dict:
        """Test encoding a medium-length prompt."""
        enc = self.encoder or (self.pipeline.encoder if self.pipeline else None)
        if not enc:
            raise RuntimeError("No encoder loaded")

        prompt = (
            "A highly detailed photograph of a serene mountain landscape at golden hour, "
            "with snow-capped peaks reflecting warm orange and pink light from the setting sun. "
            "A crystal-clear alpine lake in the foreground perfectly mirrors the mountains and sky. "
            "Scattered pine trees frame the scene, and a light mist rises from the water's surface."
        )
        self.log(f"Encoding: {prompt[:50]}...")

        output = enc.encode(prompt)
        token_count = output.token_counts[0]

        return {
            "prompt_length": len(prompt),
            "token_count": token_count,
        }

    def test_encode_with_template(self) -> dict:
        """Test encoding with a template applied."""
        enc = self.encoder or (self.pipeline.encoder if self.pipeline else None)
        if not enc:
            raise RuntimeError("No encoder loaded")

        prompt = "A warrior princess"
        template = "photorealistic"
        self.log(f"Encoding with template '{template}': {prompt}")

        output = enc.encode(prompt, template=template)
        token_count = output.token_counts[0]

        return {
            "prompt_length": len(prompt),
            "template": template,
            "token_count": token_count,
        }

    def test_encode_with_thinking(self) -> dict:
        """Test encoding with thinking block."""
        enc = self.encoder or (self.pipeline.encoder if self.pipeline else None)
        if not enc:
            raise RuntimeError("No encoder loaded")

        prompt = "A portrait of a woman"
        thinking = "Soft lighting, natural expression, shallow depth of field"
        self.log(f"Encoding with thinking: {prompt}")

        output = enc.encode(prompt, thinking_content=thinking)
        token_count = output.token_counts[0]

        return {
            "prompt_length": len(prompt),
            "thinking_length": len(thinking),
            "token_count": token_count,
        }

    def test_generate_text(self) -> dict:
        """Test text generation (rewriting)."""
        enc = self.encoder or (self.pipeline.encoder if self.pipeline else None)
        if not enc:
            raise RuntimeError("No encoder loaded")

        backend = enc.backend
        if not backend.supports_generation:
            raise RuntimeError("Backend does not support generation")

        prompt = "A cat"
        system_prompt = "Expand this into a detailed image description."
        self.log(f"Generating from: {prompt}")

        generated = backend.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            max_new_tokens=64,  # Short for testing
            temperature=0.7,
        )

        return {
            "input_length": len(prompt),
            "output_length": len(generated),
            "output_preview": generated[:100] + "..." if len(generated) > 100 else generated,
        }

    def test_full_generation(self) -> dict:
        """Test full image generation."""
        if not self.pipeline:
            raise RuntimeError("Pipeline not loaded")

        prompt = "A cat sleeping in sunlight, photorealistic"
        self.log(f"Generating image: {prompt}")
        self.log(f"  Size: {self.config.width}x{self.config.height}")
        self.log(f"  Steps: {self.config.steps}")

        result = self.pipeline(
            prompt=prompt,
            height=self.config.height,
            width=self.config.width,
            num_inference_steps=self.config.steps,
            guidance_scale=self.config.guidance_scale,
            generator=torch.Generator(device="cpu").manual_seed(42),
        )

        image = result.images[0]
        return {
            "image_size": f"{image.width}x{image.height}",
            "steps": self.config.steps,
        }

    def test_repeated_encode(self, count: int = 5) -> dict:
        """Test repeated encoding to check for memory leaks."""
        enc = self.encoder or (self.pipeline.encoder if self.pipeline else None)
        if not enc:
            raise RuntimeError("No encoder loaded")

        prompt = "A test prompt for repeated encoding"
        times = []

        for i in range(count):
            start = time.perf_counter()
            enc.encode(prompt)
            times.append((time.perf_counter() - start) * 1000)

        return {
            "count": count,
            "times_ms": times,
            "avg_ms": sum(times) / len(times),
            "min_ms": min(times),
            "max_ms": max(times),
        }

    def test_cuda_sync(self) -> dict:
        """Test CUDA synchronization behavior."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")

        # Test sync timing
        start = time.perf_counter()
        torch.cuda.synchronize()
        sync_time = (time.perf_counter() - start) * 1000

        return {
            "sync_time_ms": sync_time,
            "current_device": torch.cuda.current_device(),
        }


def run_profile(config: RuntimeConfig, tests: list[str] | None = None, verbose: bool = False) -> ProfileResults:
    """Run profiling tests with the given configuration."""
    profiler = Profiler(config, verbose=verbose)

    # Default test suite
    all_tests = [
        ("load_encoder", profiler.load_encoder_only),
        ("encode_short", profiler.test_encode_short),
        ("encode_medium", profiler.test_encode_medium),
        ("encode_with_template", profiler.test_encode_with_template),
        ("encode_with_thinking", profiler.test_encode_with_thinking),
        ("generate_text", profiler.test_generate_text),
        ("repeated_encode", profiler.test_repeated_encode),
    ]

    # Add CUDA tests if available
    if torch.cuda.is_available():
        all_tests.insert(0, ("cuda_sync", profiler.test_cuda_sync))

    # Add full pipeline tests if not encoder-only
    pipeline_tests = [
        ("load_pipeline", profiler.load_pipeline),
        ("full_generation", profiler.test_full_generation),
    ]

    # Filter tests if specified
    if tests:
        test_names = [t.strip() for t in tests]
        if "pipeline" in test_names or "generate" in test_names:
            # Include pipeline loading and generation
            all_tests = [t for t in all_tests if t[0] in test_names or t[0].startswith("load")]
            all_tests.extend([t for t in pipeline_tests if t[0] in test_names or "pipeline" in test_names])
        else:
            all_tests = [t for t in all_tests if t[0] in test_names or t[0].startswith("load")]

    # Run tests
    print("\n" + "=" * 60)
    print("PROFILING SESSION")
    print("=" * 60)
    print(f"Model: {config.model_path}")
    print(f"Encoder device: {config.encoder_device}")
    print(f"DiT device: {config.dit_device}")
    print(f"VAE device: {config.vae_device}")
    print(f"Dtype: {config.torch_dtype}")
    print("=" * 60)

    for name, func in all_tests:
        profiler.run_test(name, func)

    # Build results
    results = ProfileResults(
        timestamp=datetime.now().isoformat(),
        config={
            "model_path": config.model_path,
            "encoder_device": config.encoder_device,
            "dit_device": config.dit_device,
            "vae_device": config.vae_device,
            "torch_dtype": config.torch_dtype,
            "flash_attn": config.flash_attn,
            "compile": config.compile,
            "cpu_offload": config.cpu_offload,
            "tiled_vae": config.tiled_vae,
            "embedding_cache": config.embedding_cache,
            "long_prompt_mode": config.long_prompt_mode,
        },
        system_info=get_system_info(),
        tests=profiler.results,
    )

    # Calculate summary
    passed = sum(1 for t in profiler.results if t.success)
    failed = sum(1 for t in profiler.results if not t.success)
    total_time = sum(t.duration_ms for t in profiler.results)

    results.summary = {
        "total_tests": len(profiler.results),
        "passed": passed,
        "failed": failed,
        "total_time_ms": total_time,
        "peak_memory_mb": max((t.memory_peak_mb for t in profiler.results), default=0),
    }

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Tests: {passed}/{len(profiler.results)} passed")
    print(f"Total time: {total_time:.1f}ms")
    print(f"Peak memory: {results.summary['peak_memory_mb']:.0f}MB")

    if failed > 0:
        print("\nFailed tests:")
        for t in profiler.results:
            if not t.success:
                print(f"  - {t.name}: {t.error}")

    return results


def run_config_sweep(base_config: RuntimeConfig, verbose: bool = False) -> list[ProfileResults]:
    """Run tests with different config combinations."""
    results = []

    # Define variations to test
    variations = [
        {"name": "baseline", "changes": {}},
        {"name": "cpu_encoder", "changes": {"encoder_device": "cpu"}},
        {"name": "flash_attn", "changes": {"flash_attn": True}},
        {"name": "compile", "changes": {"compile": True}},
        {"name": "embedding_cache", "changes": {"embedding_cache": True}},
    ]

    if torch.cuda.is_available():
        variations.append({"name": "fp16", "changes": {"torch_dtype": "float16"}})

    for var in variations:
        print(f"\n{'#' * 60}")
        print(f"# TESTING VARIATION: {var['name']}")
        print(f"{'#' * 60}")

        # Create modified config
        config = RuntimeConfig(**asdict(base_config))
        for key, value in var["changes"].items():
            setattr(config, key, value)

        try:
            result = run_profile(config, verbose=verbose)
            result.config["variation"] = var["name"]
            results.append(result)
        except Exception as e:
            print(f"[ERROR] Variation {var['name']} failed: {e}")
            if verbose:
                traceback.print_exc()

    return results


def main():
    parser = create_base_parser(
        description="Profile Z-Image pipeline for stability testing",
        include_generation_args=True,
    )

    # Add profiler-specific arguments
    parser.add_argument(
        "--tests",
        type=str,
        default=None,
        help="Comma-separated list of tests to run (e.g., encode,generate)",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run tests with different config combinations",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of times to repeat the test suite",
    )

    args = parser.parse_args()
    config = load_runtime_config(args)

    # Validate model path
    if not config.model_path:
        print("Error: --model-path is required")
        sys.exit(1)

    # Setup logging
    setup_logging(config)

    # Parse test list
    tests = args.tests.split(",") if args.tests else None

    # Run profiling
    if args.sweep:
        all_results = run_config_sweep(config, verbose=config.verbose)
    else:
        all_results = []
        for i in range(args.repeat):
            if args.repeat > 1:
                print(f"\n{'*' * 60}")
                print(f"* RUN {i + 1}/{args.repeat}")
                print(f"{'*' * 60}")
            result = run_profile(config, tests=tests, verbose=config.verbose)
            result.config["run_number"] = i + 1
            all_results.append(result)

    # Save results
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump([r.to_dict() for r in all_results], f, indent=2)
        print(f"\nResults saved to: {output_path}")

    # Exit with error code if any tests failed
    total_failed = sum(r.summary.get("failed", 0) for r in all_results)
    if total_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
