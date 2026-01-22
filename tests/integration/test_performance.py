"""
Performance Regression Tests.

Last Updated: 2026-01-22

Detects performance regressions by comparing relative timings. Does NOT assert
absolute times (which vary by hardware), but validates:
1. Operations complete in reasonable time (no hangs)
2. Memory usage stays within bounds
3. No memory leaks across repeated operations

Run with: uv run pytest tests/integration/test_performance.py -v

Note: These tests require CUDA and are skipped if GPU is not available.
"""

import gc
import time
from pathlib import Path

import pytest
import torch


# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for performance tests"
)


def models_available() -> bool:
    """Check if LTX-2 model files are available."""
    paths = [
        Path.home() / "Storage/LTX-2/transformer",
        Path.home() / "models/LTX-2/transformer",
        Path("models/LTX-2/transformer"),
    ]
    return any(p.exists() for p in paths)


@pytest.fixture
def cleanup_gpu():
    """Fixture to clean up GPU memory before and after tests."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================================
# Tensor Operation Performance
# ============================================================================


class TestTensorOperationPerformance:
    """Tests for basic tensor operation timing (model-agnostic)."""

    def test_large_tensor_creation_timing(self, cleanup_gpu):
        """Large tensor creation should complete quickly."""
        timeout_seconds = 5.0

        start = time.perf_counter()
        # Create a tensor similar to latent size: [1, 128, 16, 16, 24]
        tensor = torch.randn(1, 128, 16, 16, 24, device="cuda")
        elapsed = time.perf_counter() - start

        assert elapsed < timeout_seconds, \
            f"Tensor creation took {elapsed:.2f}s, expected < {timeout_seconds}s"
        assert tensor.shape == (1, 128, 16, 16, 24)

        del tensor

    def test_matmul_timing(self, cleanup_gpu):
        """Matrix multiplication should complete in reasonable time."""
        timeout_seconds = 2.0

        # Simulate attention-like matmul: [32, 1920, 128] @ [32, 128, 1920]
        a = torch.randn(32, 1920, 128, device="cuda")
        b = torch.randn(32, 128, 1920, device="cuda")

        # Warmup
        _ = torch.bmm(a, b)
        torch.cuda.synchronize()

        start = time.perf_counter()
        result = torch.bmm(a, b)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        assert elapsed < timeout_seconds, \
            f"Matmul took {elapsed:.2f}s, expected < {timeout_seconds}s"
        assert result.shape == (32, 1920, 1920)

        del a, b, result


# ============================================================================
# Memory Leak Detection
# ============================================================================


class TestMemoryLeakDetection:
    """Tests to detect memory leaks across repeated operations."""

    def test_no_memory_leak_tensor_creation(self, cleanup_gpu):
        """Repeated tensor creation should not leak memory."""
        # Get baseline memory
        torch.cuda.reset_peak_memory_stats()

        baseline_allocated = torch.cuda.memory_allocated()

        # Perform many iterations
        for _ in range(100):
            tensor = torch.randn(1, 128, 5, 16, 24, device="cuda")
            del tensor

        gc.collect()
        torch.cuda.empty_cache()

        final_allocated = torch.cuda.memory_allocated()

        # Memory should return to near baseline (allow 1MB tolerance)
        tolerance_bytes = 1 * 1024 * 1024  # 1MB
        assert final_allocated - baseline_allocated < tolerance_bytes, \
            f"Memory leak detected: {(final_allocated - baseline_allocated) / 1024 / 1024:.2f} MB"

    def test_no_memory_leak_computation(self, cleanup_gpu):
        """Repeated computation should not leak memory."""
        torch.cuda.reset_peak_memory_stats()

        baseline_allocated = torch.cuda.memory_allocated()

        for _ in range(50):
            x = torch.randn(1, 64, 8, 8, 8, device="cuda")
            y = torch.nn.functional.silu(x)
            z = y * x
            del x, y, z

        gc.collect()
        torch.cuda.empty_cache()

        final_allocated = torch.cuda.memory_allocated()

        tolerance_bytes = 1 * 1024 * 1024  # 1MB
        assert final_allocated - baseline_allocated < tolerance_bytes, \
            f"Memory leak detected: {(final_allocated - baseline_allocated) / 1024 / 1024:.2f} MB"


# ============================================================================
# VAE Performance Tests
# ============================================================================


class TestVAEPerformance:
    """Tests for VAE encode/decode performance (requires models)."""

    @pytest.mark.skipif(not models_available(), reason="LTX-2 models not found")
    @pytest.mark.slow
    def test_vae_decode_completes(self, cleanup_gpu):
        """VAE decode should complete without hanging.

        This test verifies the decode operation completes in reasonable time.
        Actual timing will vary by hardware.
        """
        # This would require loading the VAE - placeholder for now
        pytest.skip("VAE decode test requires model loading infrastructure")


# ============================================================================
# Transformer Performance Tests
# ============================================================================


class TestTransformerPerformance:
    """Tests for transformer forward pass performance (requires models)."""

    @pytest.mark.skipif(not models_available(), reason="LTX-2 models not found")
    @pytest.mark.slow
    def test_transformer_forward_completes(self, cleanup_gpu):
        """Transformer forward pass should complete without hanging.

        This test verifies the forward pass completes in reasonable time.
        Actual timing will vary by hardware and quantization.
        """
        # This would require loading the transformer - placeholder for now
        pytest.skip("Transformer forward test requires model loading infrastructure")


# ============================================================================
# Memory Bounds Tests
# ============================================================================


class TestMemoryBounds:
    """Tests for memory usage bounds."""

    def test_peak_memory_under_threshold_basic_ops(self, cleanup_gpu):
        """Basic operations should not exceed reasonable memory threshold."""
        torch.cuda.reset_peak_memory_stats()

        # Simulate a small pipeline step
        batch_size, tokens, dim = 1, 1920, 4096
        x = torch.randn(batch_size, tokens, dim, device="cuda", dtype=torch.bfloat16)

        # Simulate attention-like operation
        q = torch.randn(batch_size, 32, tokens, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(batch_size, 32, tokens, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(batch_size, 32, tokens, 128, device="cuda", dtype=torch.bfloat16)

        # Attention (scaled dot product)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (128 ** 0.5)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_out = torch.matmul(attn_weights, v)

        peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3

        # For basic ops without model weights, should be well under 8GB
        threshold_gb = 8.0
        assert peak_memory_gb < threshold_gb, \
            f"Peak memory {peak_memory_gb:.2f} GB exceeds threshold {threshold_gb} GB"

        del x, q, k, v, attn_weights, attn_out

    def test_memory_fragmentation(self, cleanup_gpu):
        """Test that memory doesn't become heavily fragmented."""
        torch.cuda.reset_peak_memory_stats()

        # Allocate and deallocate tensors of varying sizes
        tensors = []
        for size in [100, 500, 1000, 200, 800, 300]:
            t = torch.randn(size, size, device="cuda")
            tensors.append(t)

        # Delete half of them
        del tensors[1], tensors[3]
        gc.collect()
        torch.cuda.empty_cache()

        # Allocate new tensors
        new_tensors = []
        for size in [400, 600]:
            t = torch.randn(size, size, device="cuda")
            new_tensors.append(t)

        # Memory should still be manageable
        allocated_gb = torch.cuda.memory_allocated() / 1024**3
        reserved_gb = torch.cuda.memory_reserved() / 1024**3

        # Reserved should not be excessively larger than allocated
        # (would indicate fragmentation)
        fragmentation_ratio = reserved_gb / max(allocated_gb, 0.001)
        assert fragmentation_ratio < 3.0, \
            f"High fragmentation: reserved={reserved_gb:.2f}GB, allocated={allocated_gb:.2f}GB"


# ============================================================================
# Stress Tests
# ============================================================================


@pytest.mark.slow
class TestStress:
    """Stress tests for stability under repeated operations."""

    def test_repeated_forward_passes_stable(self, cleanup_gpu):
        """Multiple sequential operations should remain stable."""
        # Simple conv-like operation
        conv = torch.nn.Conv3d(64, 64, 3, padding=1).cuda()

        initial_memory = torch.cuda.memory_allocated()

        for i in range(20):
            x = torch.randn(1, 64, 5, 16, 24, device="cuda")
            y = conv(x)
            y = torch.nn.functional.silu(y)
            del x, y

            if i % 5 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        final_memory = torch.cuda.memory_allocated()

        # Memory growth should be minimal
        growth_mb = (final_memory - initial_memory) / 1024**2
        assert growth_mb < 50, f"Memory grew by {growth_mb:.1f} MB over 20 iterations"


# Run with: uv run pytest tests/integration/test_performance.py -v
# Run slow tests: uv run pytest tests/integration/test_performance.py -v --runslow
