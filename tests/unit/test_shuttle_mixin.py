"""
Tests for PinnedShuttleMixin -- shared pinned-memory GPU shuttle pattern.

Last Updated: 2026-03-02

Tests the mixin in isolation using tiny nn.Modules, without needing
real model weights or GPU (except for pinned-memory and CUDA tests).

Run with: uv run pytest tests/unit/test_shuttle_mixin.py -v
"""

import pytest
import torch
from torch import nn

from llm_dit.utils.shuttle import PinnedShuttleMixin


# ---------------------------------------------------------------------------
# Test helpers: tiny modules for testing the mixin
# ---------------------------------------------------------------------------


class DirectModule(PinnedShuttleMixin, nn.Module):
    """Module that IS the shuttle target (like AutoEncoder)."""

    def __init__(self):
        nn.Module.__init__(self)
        self._init_shuttle_state()
        self.linear = nn.Linear(4, 4)
        # Add a buffer for buffer-pinning tests
        self.register_buffer("running_mean", torch.zeros(4))


class WrappedModule(PinnedShuttleMixin):
    """Wrapper that delegates to an inner module (like encoders)."""

    def __init__(self):
        self._init_shuttle_state()
        self._model = nn.Linear(4, 4)

    def _shuttle_module(self) -> nn.Module | None:
        return self._model


class WrappedWithExtras(PinnedShuttleMixin):
    """Wrapper with extra modules (like Gemma3Encoder)."""

    def __init__(self):
        self._init_shuttle_state()
        self._model = nn.Linear(4, 4)
        self._extra_a = nn.Linear(4, 2)
        self._extra_b = nn.Linear(2, 4)

    def _shuttle_module(self) -> nn.Module | None:
        return self._model

    def _shuttle_extra_modules(self) -> list[nn.Module]:
        return [self._extra_a, self._extra_b]


class WrappedWithOptionalExtras(PinnedShuttleMixin):
    """Wrapper with optional extra modules (some may be None)."""

    def __init__(self, include_extra: bool = True):
        self._init_shuttle_state()
        self._model = nn.Linear(4, 4)
        self._extra = nn.Linear(4, 2) if include_extra else None

    def _shuttle_module(self) -> nn.Module | None:
        return self._model

    def _shuttle_extra_modules(self) -> list[nn.Module]:
        return [m for m in [self._extra] if m is not None]


def _snapshot(module: nn.Module) -> dict[str, torch.Tensor]:
    """Clone all params and buffers for comparison."""
    snap = {}
    for name, p in module.named_parameters():
        snap[name] = p.data.detach().cpu().clone()
    for name, b in module.named_buffers():
        snap[name] = b.data.detach().cpu().clone()
    return snap


# ---------------------------------------------------------------------------
# Tests: _init_shuttle_state
# ---------------------------------------------------------------------------


class TestInitShuttleState:
    """Tests for _init_shuttle_state."""

    def test_sets_defaults_direct(self):
        m = DirectModule()
        assert m._is_offloaded is False
        assert m._is_pinned is False
        assert m._pinned_shadows == {}

    def test_sets_defaults_wrapped(self):
        m = WrappedModule()
        assert m._is_offloaded is False
        assert m._is_pinned is False
        assert m._pinned_shadows == {}


# ---------------------------------------------------------------------------
# Tests: _shuttle_module default
# ---------------------------------------------------------------------------


class TestShuttleModule:
    """Tests for _shuttle_module hook."""

    def test_direct_module_returns_self(self):
        m = DirectModule()
        assert m._shuttle_module() is m

    def test_wrapped_module_returns_inner(self):
        m = WrappedModule()
        assert m._shuttle_module() is m._model

    def test_default_non_module_returns_none(self):
        """Plain PinnedShuttleMixin (not nn.Module) returns None by default."""

        class Bare(PinnedShuttleMixin):
            def __init__(self):
                self._init_shuttle_state()

        m = Bare()
        assert m._shuttle_module() is None


class TestShuttleExtraModules:
    """Tests for _shuttle_extra_modules hook."""

    def test_default_returns_empty(self):
        m = DirectModule()
        assert m._shuttle_extra_modules() == []

    def test_with_extras(self):
        m = WrappedWithExtras()
        extras = m._shuttle_extra_modules()
        assert len(extras) == 2
        assert m._extra_a in extras
        assert m._extra_b in extras

    def test_none_filtering(self):
        m = WrappedWithOptionalExtras(include_extra=False)
        assert m._shuttle_extra_modules() == []

        m2 = WrappedWithOptionalExtras(include_extra=True)
        assert len(m2._shuttle_extra_modules()) == 1


# ---------------------------------------------------------------------------
# Tests: offload_to_pinned
# ---------------------------------------------------------------------------


class TestOffloadToPinned:
    """Tests for offload_to_pinned."""

    def test_sets_flags(self):
        m = DirectModule()
        m.offload_to_pinned()
        assert m._is_offloaded is True
        assert m._is_pinned is True

    def test_all_params_on_cpu(self):
        m = DirectModule()
        m.offload_to_pinned()
        for name, p in m.named_parameters():
            assert p.device.type == "cpu", f"{name} not on CPU"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for pin_memory")
    def test_all_params_pinned(self):
        m = DirectModule()
        m.offload_to_pinned()
        for name, p in m.named_parameters():
            assert p.data.is_pinned(), f"{name} not pinned"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for pin_memory")
    def test_buffers_pinned(self):
        m = DirectModule()
        m.offload_to_pinned()
        for name, b in m.named_buffers():
            assert b.data.is_pinned(), f"buffer {name} not pinned"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for pin_memory")
    def test_shadow_buffers_stored(self):
        m = DirectModule()
        m.offload_to_pinned()
        param_names = {n for n, _ in m.named_parameters()}
        buffer_names = {n for n, _ in m.named_buffers()}
        assert set(m._pinned_shadows.keys()) == param_names | buffer_names

    def test_preserves_weights(self):
        m = DirectModule()
        before = _snapshot(m)
        m.offload_to_pinned()
        for name, p in m.named_parameters():
            torch.testing.assert_close(p.data.cpu(), before[name])

    def test_noop_when_no_module(self):
        """offload_to_pinned is a no-op when _shuttle_module returns None."""

        class Bare(PinnedShuttleMixin):
            def __init__(self):
                self._init_shuttle_state()

        m = Bare()
        m.offload_to_pinned()
        assert m._is_offloaded is False
        assert m._is_pinned is False

    def test_wrapped_module_pins_inner(self):
        m = WrappedModule()
        m.offload_to_pinned()
        assert m._is_offloaded is True
        assert m._is_pinned is True
        for name, p in m._model.named_parameters():
            assert p.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for pin_memory")
    def test_extras_moved_to_cpu(self):
        """Extra modules are moved to CPU (not pinned) during offload_to_pinned."""
        m = WrappedWithExtras()
        # Start extras on CUDA
        m._extra_a.to("cuda")
        m._extra_b.to("cuda")
        m.offload_to_pinned()
        for p in m._extra_a.parameters():
            assert p.device.type == "cpu"
        for p in m._extra_b.parameters():
            assert p.device.type == "cpu"


# ---------------------------------------------------------------------------
# Tests: offload (fast-path)
# ---------------------------------------------------------------------------


class TestOffload:
    """Tests for offload() -- fast return from GPU to CPU."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_moves_to_cpu_pinned_path(self):
        m = DirectModule()
        m.offload_to_pinned()
        m.to_device(torch.device("cuda"))
        torch.cuda.synchronize()

        m.offload()

        for name, p in m.named_parameters():
            assert p.device.type == "cpu", f"{name} on GPU after offload"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_sets_offloaded_flag(self):
        m = DirectModule()
        m.offload_to_pinned()
        m.to_device(torch.device("cuda"))
        torch.cuda.synchronize()

        m.offload()
        assert m._is_offloaded is True

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_preserves_pinned_state(self):
        m = DirectModule()
        m.offload_to_pinned()
        m.to_device(torch.device("cuda"))
        torch.cuda.synchronize()

        m.offload()
        assert m._is_pinned is True

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_params_pinned_after_offload(self):
        m = DirectModule()
        m.offload_to_pinned()
        m.to_device(torch.device("cuda"))
        torch.cuda.synchronize()

        m.offload()
        for name, p in m.named_parameters():
            assert p.data.is_pinned(), f"{name} not pinned after offload"

    def test_fallback_when_not_pinned(self):
        """Without prior pinning, offload() falls back to simple .to('cpu')."""
        m = DirectModule()
        m._is_offloaded = False  # simulate GPU state without actual CUDA
        m.offload()
        assert m._is_offloaded is True
        for name, p in m.named_parameters():
            assert p.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_wrapped_offload(self):
        m = WrappedModule()
        m.offload_to_pinned()
        m.to_device(torch.device("cuda"))
        torch.cuda.synchronize()

        m.offload()
        for name, p in m._model.named_parameters():
            assert p.device.type == "cpu"
        assert m._is_offloaded is True

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_extras_moved_to_cpu_on_offload(self):
        m = WrappedWithExtras()
        m.offload_to_pinned()
        m.to_device(torch.device("cuda"))
        torch.cuda.synchronize()

        m.offload()
        for p in m._extra_a.parameters():
            assert p.device.type == "cpu"
        for p in m._extra_b.parameters():
            assert p.device.type == "cpu"


# ---------------------------------------------------------------------------
# Tests: to_device
# ---------------------------------------------------------------------------


class TestToDevice:
    """Tests for to_device -- move from CPU pinned to GPU."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_moves_to_gpu(self):
        m = DirectModule()
        m.offload_to_pinned()

        needs_sync = m.to_device(torch.device("cuda"))
        if needs_sync:
            torch.cuda.synchronize()

        for name, p in m.named_parameters():
            assert p.device.type == "cuda", f"{name} not on GPU"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_clears_offloaded_flag(self):
        m = DirectModule()
        m.offload_to_pinned()
        m.to_device(torch.device("cuda"))
        torch.cuda.synchronize()
        assert m._is_offloaded is False

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_returns_needs_sync_when_pinned(self):
        m = DirectModule()
        m.offload_to_pinned()
        needs_sync = m.to_device(torch.device("cuda"))
        assert needs_sync is True

    def test_noop_when_not_offloaded(self):
        m = DirectModule()
        needs_sync = m.to_device(torch.device("cpu"))
        assert needs_sync is False

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_wrapped_moves_to_gpu(self):
        m = WrappedModule()
        m.offload_to_pinned()
        needs_sync = m.to_device(torch.device("cuda"))
        if needs_sync:
            torch.cuda.synchronize()
        for name, p in m._model.named_parameters():
            assert p.device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_extras_moved_to_device(self):
        m = WrappedWithExtras()
        m.offload_to_pinned()
        needs_sync = m.to_device(torch.device("cuda"))
        if needs_sync:
            torch.cuda.synchronize()
        for p in m._extra_a.parameters():
            assert p.device.type == "cuda"
        for p in m._extra_b.parameters():
            assert p.device.type == "cuda"


# ---------------------------------------------------------------------------
# Tests: round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Tests for full shuttle cycle: pinned -> GPU -> CPU -> GPU."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_weights_preserved_direct(self):
        m = DirectModule()
        before = _snapshot(m)

        # Cycle 1
        m.offload_to_pinned()
        m.to_device(torch.device("cuda"))
        torch.cuda.synchronize()
        m.offload()

        # Cycle 2
        m.to_device(torch.device("cuda"))
        torch.cuda.synchronize()
        m.offload()

        for name, p in m.named_parameters():
            torch.testing.assert_close(
                p.data.cpu(), before[name],
                msg=f"Parameter {name} drifted after round-trip",
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_buffers_preserved_direct(self):
        m = DirectModule()
        before = _snapshot(m)

        m.offload_to_pinned()
        m.to_device(torch.device("cuda"))
        torch.cuda.synchronize()
        m.offload()

        for name, b in m.named_buffers():
            torch.testing.assert_close(
                b.data.cpu(), before[name],
                msg=f"Buffer {name} drifted after round-trip",
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_weights_preserved_wrapped(self):
        m = WrappedModule()
        before = _snapshot(m._model)

        m.offload_to_pinned()
        m.to_device(torch.device("cuda"))
        torch.cuda.synchronize()
        m.offload()
        m.to_device(torch.device("cuda"))
        torch.cuda.synchronize()
        m.offload()

        for name, p in m._model.named_parameters():
            torch.testing.assert_close(
                p.data.cpu(), before[name],
                msg=f"Parameter {name} drifted",
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_weights_preserved_with_extras(self):
        m = WrappedWithExtras()
        before_main = _snapshot(m._model)
        before_a = _snapshot(m._extra_a)
        before_b = _snapshot(m._extra_b)

        m.offload_to_pinned()
        m.to_device(torch.device("cuda"))
        torch.cuda.synchronize()
        m.offload()

        for name, p in m._model.named_parameters():
            torch.testing.assert_close(p.data.cpu(), before_main[name])
        for name, p in m._extra_a.named_parameters():
            torch.testing.assert_close(p.data.cpu(), before_a[name])
        for name, p in m._extra_b.named_parameters():
            torch.testing.assert_close(p.data.cpu(), before_b[name])
