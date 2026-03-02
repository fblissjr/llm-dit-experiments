"""
PinnedShuttleMixin -- shared pinned-memory GPU shuttle pattern.

Last Updated: 2026-03-02

Provides fast CPU <-> GPU transfers via page-locked (pinned) memory and
pre-allocated shadow buffers. Used by AutoEncoder (VAE), Qwen3UnifiedEncoder,
and Gemma3Encoder.

Design:
    The mixin operates on a "shuttle module" -- the nn.Module whose parameters
    get pinned. Subclasses control which module via two hooks:

    _shuttle_module() -> nn.Module | None
        The main module to pin. Defaults to ``self`` if it IS an nn.Module
        (like AutoEncoder), or None otherwise.

    _shuttle_extra_modules() -> list[nn.Module]
        Additional modules to .to(device) alongside the main module but NOT
        pin (they're small enough that pinning isn't worth the complexity).
        Default: empty list. Gemma3Encoder overrides this for
        _feature_extractor and _embeddings_connector.
"""

import gc
import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


class PinnedShuttleMixin:
    """Mixin for pinned-memory GPU shuttle pattern.

    Provides three operations:
        offload_to_pinned() -- one-time setup: move to CPU, pin memory, store shadows
        offload()           -- fast-path return from GPU to pre-allocated pinned buffers
        to_device(device)   -- move from CPU pinned to target device, returns needs_sync
    """

    _is_offloaded: bool
    _is_pinned: bool
    _pinned_shadows: dict[str, torch.Tensor]

    def _init_shuttle_state(self) -> None:
        """Initialize shuttle state. Call from subclass __init__."""
        self._is_offloaded = False
        self._is_pinned = False
        self._pinned_shadows: dict[str, torch.Tensor] = {}

    def _shuttle_module(self) -> nn.Module | None:
        """Return the module whose params/buffers get pinned.

        Default: ``self`` if it's an nn.Module, else None.
        Override in wrappers that hold an inner ``_model``.
        """
        if isinstance(self, nn.Module):
            return self
        return None

    def _shuttle_extra_modules(self) -> list[nn.Module]:
        """Return additional modules to .to(device) but NOT pin.

        Default: empty list. Override for ancillary components
        (e.g. Gemma3's feature_extractor, embeddings_connector).
        """
        return []

    def offload_to_pinned(self) -> None:
        """Offload to CPU with pinned memory for fast GPU shuttle.

        Pinned (page-locked) memory enables direct DMA transfers between
        CPU and GPU, avoiding the intermediate staging-buffer copy. This
        makes subsequent ``.to("cuda", non_blocking=True)`` calls ~2-3x faster.

        Also stores references to pinned tensors as shadow buffers so that
        subsequent ``offload()`` calls can copy CUDA -> pinned directly
        without re-allocating pinned memory on every cycle.

        Called once after loading the model at startup.
        """
        module = self._shuttle_module()
        if module is None:
            return

        label = type(self).__name__
        logger.info(f"[{label}] Offloading to CPU with pinned memory...")
        module.to("cpu")

        pinned_count = 0
        self._pinned_shadows = {}
        for name, param in module.named_parameters():
            if not param.data.is_pinned():
                param.data = param.data.pin_memory()
                pinned_count += 1
            self._pinned_shadows[name] = param.data
        for name, buf in module.named_buffers():
            if not buf.data.is_pinned():
                buf.data = buf.data.pin_memory()
            self._pinned_shadows[name] = buf.data

        for extra in self._shuttle_extra_modules():
            extra.to("cpu")

        self._is_offloaded = True
        self._is_pinned = True
        logger.info(f"[{label}] Offloaded with {pinned_count} pinned tensors")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def offload(self) -> None:
        """Fast-path offload: copy CUDA tensors into pre-allocated pinned buffers.

        When shadow buffers exist (from ``offload_to_pinned``), copies CUDA
        tensors directly into the pre-allocated pinned buffers. This avoids
        the 2N allocation + 2N copy overhead of ``module.to("cpu")`` + re-pin
        on every cycle, replacing it with 0 allocations + N copies.
        """
        module = self._shuttle_module()
        if module is not None:
            if self._is_pinned and self._pinned_shadows:
                for name, param in module.named_parameters():
                    pinned = self._pinned_shadows.get(name)
                    if pinned is not None:
                        pinned.copy_(param.data)
                        param.data = pinned
                    else:
                        param.data = param.data.cpu().pin_memory()
                for name, buf in module.named_buffers():
                    pinned = self._pinned_shadows.get(name)
                    if pinned is not None:
                        pinned.copy_(buf.data)
                        buf.data = pinned
                    else:
                        buf.data = buf.data.cpu().pin_memory()
            else:
                module.to("cpu")

        for extra in self._shuttle_extra_modules():
            extra.to("cpu")

        self._is_offloaded = True

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def to_device(self, device: torch.device, non_blocking: bool = True) -> bool:
        """Move from CPU pinned memory to target device.

        Args:
            device: Target device (e.g. ``torch.device("cuda")``).
            non_blocking: Use async DMA transfer (default True).

        Returns:
            True if caller should synchronize (pinned -> CUDA transfer),
            False if no transfer was needed.
        """
        if not self._is_offloaded:
            return False

        module = self._shuttle_module()
        if module is not None:
            module.to(device, non_blocking=non_blocking)

        for extra in self._shuttle_extra_modules():
            extra.to(device)

        needs_sync = self._is_pinned and device.type == "cuda"
        self._is_offloaded = False
        return needs_sync
