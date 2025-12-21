"""
Gradient checkpointing utilities for training.

Provides memory-efficient forward passes by trading compute for memory.
Based on DiffSynth-Studio implementation (Apache 2.0 license).

Usage:
    from llm_dit.training import gradient_checkpoint_forward

    output = gradient_checkpoint_forward(
        model,
        use_gradient_checkpointing=True,
        hidden_states=hidden_states,
        timestep=timestep,
    )
"""

import torch
from typing import Any, Callable


def create_custom_forward(module: torch.nn.Module) -> Callable:
    """
    Create a custom forward function for checkpointing.

    Args:
        module: The module to wrap

    Returns:
        Custom forward callable
    """
    def custom_forward(*inputs, **kwargs):
        return module(*inputs, **kwargs)
    return custom_forward


def gradient_checkpoint_forward(
    model: torch.nn.Module,
    use_gradient_checkpointing: bool = True,
    use_gradient_checkpointing_offload: bool = False,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """
    Forward pass with optional gradient checkpointing.

    Gradient checkpointing trades compute for memory by not storing
    intermediate activations during the forward pass. Instead, they
    are recomputed during the backward pass.

    With CPU offloading, intermediate activations are moved to CPU
    memory during the forward pass, further reducing GPU memory usage
    at the cost of CPU-GPU transfer overhead.

    Args:
        model: Module to run forward pass through
        use_gradient_checkpointing: Enable gradient checkpointing
        use_gradient_checkpointing_offload: Offload activations to CPU
        *args: Positional arguments for model
        **kwargs: Keyword arguments for model

    Returns:
        Model output

    Example:
        >>> output = gradient_checkpoint_forward(
        ...     transformer,
        ...     use_gradient_checkpointing=True,
        ...     hidden_states=latents,
        ...     timestep=timestep,
        ...     encoder_hidden_states=prompt_embeds,
        ... )
    """
    if use_gradient_checkpointing_offload:
        # Offload activations to CPU during forward pass
        with torch.autograd.graph.save_on_cpu():
            output = torch.utils.checkpoint.checkpoint(
                create_custom_forward(model),
                *args,
                **kwargs,
                use_reentrant=False,
            )
    elif use_gradient_checkpointing:
        # Standard gradient checkpointing
        output = torch.utils.checkpoint.checkpoint(
            create_custom_forward(model),
            *args,
            **kwargs,
            use_reentrant=False,
        )
    else:
        # No checkpointing
        output = model(*args, **kwargs)

    return output


def enable_gradient_checkpointing(model: torch.nn.Module) -> None:
    """
    Enable gradient checkpointing on a model if supported.

    Checks if the model has an enable_gradient_checkpointing method
    and calls it if available.

    Args:
        model: Model to enable checkpointing on
    """
    if hasattr(model, 'enable_gradient_checkpointing'):
        model.enable_gradient_checkpointing()
    elif hasattr(model, 'gradient_checkpointing_enable'):
        # HuggingFace models use this name
        model.gradient_checkpointing_enable()


def disable_gradient_checkpointing(model: torch.nn.Module) -> None:
    """
    Disable gradient checkpointing on a model if supported.

    Args:
        model: Model to disable checkpointing on
    """
    if hasattr(model, 'disable_gradient_checkpointing'):
        model.disable_gradient_checkpointing()
    elif hasattr(model, 'gradient_checkpointing_disable'):
        # HuggingFace models use this name
        model.gradient_checkpointing_disable()
