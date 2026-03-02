"""
Tests for FLUX.2 VAE pinned-memory shuttle pattern.

Last Updated: 2026-03-02

Verifies the VAE can shuttle between CPU (pinned memory) and GPU,
matching the encoder shuttle pattern from Qwen3UnifiedEncoder.

Run with: uv run pytest tests/unit/test_flux2_vae_shuttle.py -v
"""

import pytest
import torch

from llm_dit.models.flux2.vae import AutoEncoder, AutoEncoderParams


@pytest.fixture
def vae():
    """Create a small VAE on CPU for testing."""
    params = AutoEncoderParams(
        resolution=64,
        in_channels=3,
        ch=32,
        out_ch=3,
        ch_mult=[1, 2],
        num_res_blocks=1,
        z_channels=4,
    )
    model = AutoEncoder(params)
    model.eval()
    return model


def _snapshot_params(model: AutoEncoder) -> dict[str, torch.Tensor]:
    """Capture a detached CPU clone of all parameters."""
    snap = {}
    for name, p in model.named_parameters():
        snap[name] = p.data.detach().cpu().clone()
    for name, b in model.named_buffers():
        snap[name] = b.data.detach().cpu().clone()
    return snap


class TestOffloadToPinned:
    """Tests for offload_to_pinned() -- one-time setup."""

    def test_sets_flags(self, vae):
        """offload_to_pinned sets _is_offloaded and _is_pinned."""
        vae.offload_to_pinned()

        assert vae._is_offloaded is True
        assert vae._is_pinned is True

    def test_all_params_on_cpu(self, vae):
        """After offload_to_pinned, all parameters are on CPU."""
        vae.offload_to_pinned()

        for name, p in vae.named_parameters():
            assert p.device.type == "cpu", f"{name} not on CPU"

    def test_all_params_pinned(self, vae):
        """After offload_to_pinned, all parameter tensors are pinned."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for pin_memory")

        vae.offload_to_pinned()

        for name, p in vae.named_parameters():
            assert p.data.is_pinned(), f"{name} not pinned"

    def test_shadow_buffers_stored(self, vae):
        """offload_to_pinned populates _pinned_shadows for all params and buffers."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for pin_memory")

        vae.offload_to_pinned()

        param_names = {n for n, _ in vae.named_parameters()}
        buffer_names = {n for n, _ in vae.named_buffers()}
        all_names = param_names | buffer_names

        assert set(vae._pinned_shadows.keys()) == all_names

    def test_preserves_weights(self, vae):
        """offload_to_pinned does not change parameter values."""
        before = _snapshot_params(vae)
        vae.offload_to_pinned()

        for name, p in vae.named_parameters():
            torch.testing.assert_close(
                p.data.cpu(), before[name],
                msg=f"Parameter {name} changed after offload_to_pinned",
            )


class TestOffload:
    """Tests for offload() -- fast return from GPU to CPU."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_moves_to_cpu(self, vae):
        """offload() moves parameters from GPU back to CPU."""
        vae.offload_to_pinned()
        vae.to_device(torch.device("cuda"))
        torch.cuda.synchronize()

        vae.offload()

        for name, p in vae.named_parameters():
            assert p.device.type == "cpu", f"{name} still on GPU after offload"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_sets_offloaded_flag(self, vae):
        """offload() sets _is_offloaded to True."""
        vae.offload_to_pinned()
        vae.to_device(torch.device("cuda"))
        torch.cuda.synchronize()

        vae.offload()

        assert vae._is_offloaded is True

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_preserves_pinned_state(self, vae):
        """offload() keeps _is_pinned True after round-trip."""
        vae.offload_to_pinned()
        vae.to_device(torch.device("cuda"))
        torch.cuda.synchronize()

        vae.offload()

        assert vae._is_pinned is True

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_params_pinned_after_offload(self, vae):
        """After offload(), parameters are back in pinned CPU memory."""
        vae.offload_to_pinned()
        vae.to_device(torch.device("cuda"))
        torch.cuda.synchronize()

        vae.offload()

        for name, p in vae.named_parameters():
            assert p.data.is_pinned(), f"{name} not pinned after offload"


class TestToDevice:
    """Tests for to_device() -- move from CPU pinned to GPU."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_moves_to_gpu(self, vae):
        """to_device() moves all parameters to target device."""
        vae.offload_to_pinned()

        needs_sync = vae.to_device(torch.device("cuda"))
        if needs_sync:
            torch.cuda.synchronize()

        for name, p in vae.named_parameters():
            assert p.device.type == "cuda", f"{name} not on GPU"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_clears_offloaded_flag(self, vae):
        """to_device() sets _is_offloaded to False."""
        vae.offload_to_pinned()

        vae.to_device(torch.device("cuda"))
        torch.cuda.synchronize()

        assert vae._is_offloaded is False

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_returns_needs_sync_when_pinned(self, vae):
        """to_device() returns True when moving pinned tensors to CUDA."""
        vae.offload_to_pinned()

        needs_sync = vae.to_device(torch.device("cuda"))

        assert needs_sync is True

    def test_noop_when_not_offloaded(self, vae):
        """to_device() returns False when model is already on device."""
        # Model starts on CPU, not offloaded
        needs_sync = vae.to_device(torch.device("cpu"))

        assert needs_sync is False


class TestRoundTrip:
    """Tests for full shuttle cycle: pinned -> GPU -> CPU -> GPU."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_weights_preserved_after_round_trip(self, vae):
        """Full cycle: offload_to_pinned -> to_device -> offload -> to_device preserves weights."""
        before = _snapshot_params(vae)

        # Cycle 1
        vae.offload_to_pinned()
        vae.to_device(torch.device("cuda"))
        torch.cuda.synchronize()
        vae.offload()

        # Cycle 2
        vae.to_device(torch.device("cuda"))
        torch.cuda.synchronize()
        vae.offload()

        for name, p in vae.named_parameters():
            torch.testing.assert_close(
                p.data.cpu(), before[name],
                msg=f"Parameter {name} drifted after round-trip",
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_buffers_preserved_after_round_trip(self, vae):
        """BatchNorm running stats survive the shuttle cycle."""
        before = _snapshot_params(vae)

        vae.offload_to_pinned()
        vae.to_device(torch.device("cuda"))
        torch.cuda.synchronize()
        vae.offload()

        for name, buf in vae.named_buffers():
            torch.testing.assert_close(
                buf.data.cpu(), before[name],
                msg=f"Buffer {name} drifted after round-trip",
            )
