"""
Skip Layer Guidance for diffusion transformers.

Last Updated: 2025-12-21

This module provides Skip Layer Guidance (SLG) for improving structure and anatomy
coherence in generated images. It works by comparing predictions with and without
specific transformer layers, then scaling the difference.

For Z-Image (CFG=0 baked in), the formula simplifies to:
    pred = pred_cond + scale * (pred_cond - pred_skip)

This moves the prediction away from the "degraded" version (with layers skipped),
which improves anatomical coherence.

Reference:
- StabilityAI SD3.5: https://github.com/Stability-AI/sd3.5
- Spatio-Temporal Guidance (STG): https://huggingface.co/papers/2411.18664
- Guiding a Diffusion Model with a Bad Version of Itself: https://huggingface.co/papers/2406.02507
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator, List, Optional, Set, Union

import torch
import torch.nn as nn


@dataclass
class LayerSkipConfig:
    """
    Configuration for which layers to skip in a transformer.

    Args:
        indices: Layer indices to skip (0-indexed). For Z-Image with 40 layers,
                 valid values are 0-39. Recommended: [7, 8, 9] for SD3.5-style,
                 or middle layers like [15, 16, 17, 18, 19] for Z-Image.
        fqn: Fully qualified name of the transformer blocks attribute.
             Use "auto" for automatic detection. For Z-Image, this is "blocks".
        skip_attention: Whether to skip attention operations in the layers.
        skip_ff: Whether to skip feed-forward operations in the layers.
        dropout: Probability of dropping layer outputs (1.0 = complete skip).
    """
    indices: List[int]
    fqn: str = "auto"
    skip_attention: bool = True
    skip_ff: bool = True
    dropout: float = 1.0

    def __post_init__(self):
        if not (0 <= self.dropout <= 1):
            raise ValueError(f"dropout must be between 0.0 and 1.0, got {self.dropout}")
        if not self.indices:
            raise ValueError("indices list cannot be empty")
        if not isinstance(self.indices, list):
            if isinstance(self.indices, int):
                self.indices = [self.indices]
            else:
                self.indices = list(self.indices)

    def to_dict(self) -> dict:
        return {
            "indices": self.indices,
            "fqn": self.fqn,
            "skip_attention": self.skip_attention,
            "skip_ff": self.skip_ff,
            "dropout": self.dropout,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LayerSkipConfig":
        return cls(**data)


class _SkipHook:
    """Forward hook that makes a layer return its input (skip the layer)."""

    def __init__(self, layer_idx: int, dropout: float = 1.0):
        self.layer_idx = layer_idx
        self.dropout = dropout
        self.handle: Optional[torch.utils.hooks.RemovableHandle] = None

    def __call__(
        self,
        module: nn.Module,
        args: tuple,
        kwargs: dict,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Hook that returns input instead of output (skips the layer)."""
        # Get hidden states from input
        hidden_states = kwargs.get("hidden_states", None)
        if hidden_states is None and len(args) > 0:
            hidden_states = args[0]

        if hidden_states is None:
            # Can't skip, return original output
            return output

        if self.dropout >= 1.0:
            # Full skip - return input
            # Handle tuple outputs (hidden_states, encoder_hidden_states)
            if isinstance(output, tuple):
                encoder_hidden_states = kwargs.get("encoder_hidden_states", None)
                if encoder_hidden_states is None and len(args) > 1:
                    encoder_hidden_states = args[1]
                return (hidden_states, encoder_hidden_states)
            return hidden_states
        else:
            # Partial skip with dropout
            if isinstance(output, tuple):
                return tuple(
                    torch.nn.functional.dropout(o, p=self.dropout) if o is not None else None
                    for o in output
                )
            return torch.nn.functional.dropout(output, p=self.dropout)

    def register(self, module: nn.Module) -> None:
        """Register this hook on a module."""
        self.handle = module.register_forward_hook(self, with_kwargs=True)

    def remove(self) -> None:
        """Remove this hook."""
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class SkipLayerGuidance:
    """
    Skip Layer Guidance for improving structure/anatomy in generated images.

    Usage:
        slg = SkipLayerGuidance(
            skip_layers=[15, 16, 17, 18, 19],  # Middle layers for Z-Image
            guidance_scale=2.8,
        )

        # In denoising loop:
        # 1. Normal forward pass
        pred_cond = model(latents, t, prompt_embeds)

        # 2. Forward pass with layers skipped
        with slg.skip_layers_context(model):
            pred_skip = model(latents, t, prompt_embeds)

        # 3. Combine predictions
        guided_pred = slg.guide(pred_cond, pred_skip)

    Args:
        skip_layers: Layer indices to skip, or LayerSkipConfig.
        guidance_scale: Scale for the guidance signal (default: 2.8).
        guidance_start: Start guidance at this fraction of steps (default: 0.01).
        guidance_stop: Stop guidance at this fraction of steps (default: 0.2).
        fqn: Fully qualified name for transformer blocks (default: "auto").
    """

    # Common block identifiers for auto-detection
    _BLOCK_IDENTIFIERS = ("blocks", "transformer_blocks", "single_transformer_blocks", "layers")

    def __init__(
        self,
        skip_layers: Union[List[int], LayerSkipConfig, List[LayerSkipConfig]],
        guidance_scale: float = 2.8,
        guidance_start: float = 0.01,
        guidance_stop: float = 0.2,
        fqn: str = "auto",
    ):
        self.guidance_scale = guidance_scale
        self.guidance_start = guidance_start
        self.guidance_stop = guidance_stop

        # Normalize config to list of LayerSkipConfig
        if isinstance(skip_layers, LayerSkipConfig):
            self.configs = [skip_layers]
        elif isinstance(skip_layers, list):
            if len(skip_layers) > 0 and isinstance(skip_layers[0], LayerSkipConfig):
                self.configs = skip_layers
            else:
                # List of integers
                self.configs = [LayerSkipConfig(indices=skip_layers, fqn=fqn)]
        else:
            raise ValueError(f"skip_layers must be list of ints or LayerSkipConfig, got {type(skip_layers)}")

        # Hooks storage
        self._active_hooks: List[_SkipHook] = []
        self._transformer_blocks: Optional[nn.ModuleList] = None

    def _find_transformer_blocks(self, model: nn.Module, fqn: str) -> nn.ModuleList:
        """Find the transformer blocks in the model."""
        if fqn != "auto":
            blocks = getattr(model, fqn, None)
            if blocks is None:
                # Try nested lookup
                for part in fqn.split("."):
                    model = getattr(model, part, None)
                    if model is None:
                        raise ValueError(f"Could not find '{fqn}' in model")
                blocks = model
            if not isinstance(blocks, nn.ModuleList):
                raise ValueError(f"'{fqn}' is not a ModuleList")
            return blocks

        # Auto-detect
        for identifier in self._BLOCK_IDENTIFIERS:
            blocks = getattr(model, identifier, None)
            if blocks is not None and isinstance(blocks, nn.ModuleList):
                return blocks

        raise ValueError(
            f"Could not auto-detect transformer blocks. "
            f"Tried: {self._BLOCK_IDENTIFIERS}. "
            f"Please specify fqn explicitly."
        )

    def _apply_hooks(self, model: nn.Module) -> None:
        """Apply skip hooks to the model."""
        for config in self.configs:
            blocks = self._find_transformer_blocks(model, config.fqn)
            self._transformer_blocks = blocks

            for idx in config.indices:
                if idx >= len(blocks):
                    raise ValueError(
                        f"Layer index {idx} out of range. "
                        f"Model has {len(blocks)} blocks (0-{len(blocks)-1})."
                    )

                hook = _SkipHook(idx, config.dropout)
                hook.register(blocks[idx])
                self._active_hooks.append(hook)

    def _remove_hooks(self) -> None:
        """Remove all active hooks."""
        for hook in self._active_hooks:
            hook.remove()
        self._active_hooks.clear()

    @contextmanager
    def skip_layers_context(self, model: nn.Module) -> Generator[None, None, None]:
        """
        Context manager that temporarily applies layer skipping to the model.

        Args:
            model: The transformer model to modify.

        Yields:
            None

        Example:
            with slg.skip_layers_context(transformer):
                pred_skip = transformer(latents, t, embeds)
        """
        try:
            self._apply_hooks(model)
            yield
        finally:
            self._remove_hooks()

    def is_active(self, step: int, num_steps: int) -> bool:
        """
        Check if guidance should be active at this step.

        Args:
            step: Current step index (0-indexed).
            num_steps: Total number of steps.

        Returns:
            True if guidance should be applied.
        """
        if num_steps <= 0:
            return True
        progress = step / num_steps
        return self.guidance_start <= progress < self.guidance_stop

    def guide(
        self,
        pred_cond: torch.Tensor,
        pred_skip: torch.Tensor,
        pred_uncond: Optional[torch.Tensor] = None,
        cfg_scale: float = 0.0,
    ) -> torch.Tensor:
        """
        Combine predictions using Skip Layer Guidance.

        For models with CFG (cfg_scale > 0):
            pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
                   + slg_scale * (pred_cond - pred_skip)

        For models without CFG (Z-Image, cfg_scale = 0):
            pred = pred_cond + slg_scale * (pred_cond - pred_skip)

        Args:
            pred_cond: Conditional prediction (normal forward pass).
            pred_skip: Prediction with layers skipped.
            pred_uncond: Unconditional prediction (for CFG models).
            cfg_scale: Classifier-free guidance scale (0 for Z-Image).

        Returns:
            Guided prediction tensor.
        """
        # Skip layer guidance shift
        slg_shift = pred_cond - pred_skip

        if pred_uncond is not None and cfg_scale > 0:
            # Standard CFG + SLG
            cfg_shift = pred_cond - pred_uncond
            return pred_uncond + cfg_scale * cfg_shift + self.guidance_scale * slg_shift
        else:
            # SLG only (Z-Image)
            return pred_cond + self.guidance_scale * slg_shift

    @property
    def skip_layer_indices(self) -> Set[int]:
        """Get all layer indices that will be skipped."""
        indices = set()
        for config in self.configs:
            indices.update(config.indices)
        return indices


def apply_layer_skip(model: nn.Module, config: LayerSkipConfig) -> List[_SkipHook]:
    """
    Apply layer skip hooks to a model.

    This is a lower-level function. For typical use, prefer SkipLayerGuidance
    with skip_layers_context().

    Args:
        model: The transformer model.
        config: Skip configuration.

    Returns:
        List of applied hooks (keep reference to remove later).
    """
    slg = SkipLayerGuidance(config)
    slg._apply_hooks(model)
    return slg._active_hooks


def remove_layer_skip(hooks: List[_SkipHook]) -> None:
    """
    Remove layer skip hooks from a model.

    Args:
        hooks: List of hooks returned by apply_layer_skip().
    """
    for hook in hooks:
        hook.remove()
