"""
Cross-attention extraction utilities for Diffusion Transformers.

Last Updated: 2026-01-17

Provides hooks and utilities for extracting attention maps from DiT models
during generation. Useful for understanding which text tokens influence
which spatial/temporal regions of the generated video.

Usage:
    from llm_dit.pipelines.utils.attention import AttentionExtractor

    # Create extractor
    extractor = AttentionExtractor(transformer)

    # Run generation with extraction
    with extractor:
        output = pipeline(prompt="A cat sleeping", ...)

    # Get attention maps
    attention_maps = extractor.get_attention_maps()
    # Shape: [num_steps, batch, heads, latent_seq, text_seq]

    # Aggregate per-token attention
    token_attention = extractor.aggregate_per_token(tokenizer, prompt)
    # Dict mapping tokens to their average attention weights
"""

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class AttentionMapInfo:
    """Metadata about extracted attention maps."""

    num_steps: int
    num_heads: int
    latent_seq_len: int
    text_seq_len: int
    block_indices: List[int]


class AttentionExtractorHook:
    """
    Forward hook for extracting cross-attention weights from transformer blocks.

    This hook captures the attention weights from cross-attention layers
    during the forward pass. Works with various DiT architectures including
    LTX-2 and Flux.
    """

    def __init__(
        self,
        block_idx: int,
        extract_cross_attention: bool = True,
        extract_self_attention: bool = False,
    ):
        """
        Initialize attention extraction hook.

        Args:
            block_idx: Index of the transformer block
            extract_cross_attention: Extract cross-attention (text->latent)
            extract_self_attention: Extract self-attention (latent->latent)
        """
        self.block_idx = block_idx
        self.extract_cross_attention = extract_cross_attention
        self.extract_self_attention = extract_self_attention

        self.cross_attention_weights: List[torch.Tensor] = []
        self.self_attention_weights: List[torch.Tensor] = []

    def __call__(
        self,
        module: nn.Module,
        input: Tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        """
        Hook callback called after attention forward pass.

        Different DiT architectures expose attention weights differently:
        - Some return (hidden_states, attention_weights) tuple
        - Some store attention_probs on the module
        - Some require custom extraction via attention_scores
        """
        # Try to extract attention weights from output
        if isinstance(output, tuple) and len(output) >= 2:
            # Some attention modules return (output, attention_weights)
            attention_weights = output[1]
            if attention_weights is not None:
                self._store_weights(attention_weights)
                return

        # Try to get from module attributes (some implementations store here)
        if hasattr(module, "attention_probs"):
            attention_weights = module.attention_probs
            if attention_weights is not None:
                self._store_weights(attention_weights)
                return

        # Fallback: module may not expose attention weights
        # This is common - many implementations don't store attention_probs
        pass

    def _store_weights(self, weights: torch.Tensor) -> None:
        """Store attention weights, moving to CPU to save VRAM."""
        # Clone and detach to avoid memory issues
        weights = weights.detach().cpu()

        if self.extract_cross_attention:
            self.cross_attention_weights.append(weights)

    def clear(self) -> None:
        """Clear stored attention weights."""
        self.cross_attention_weights.clear()
        self.self_attention_weights.clear()


class AttentionExtractor:
    """
    Context manager for extracting attention maps from DiT transformers.

    Registers forward hooks on attention layers to capture attention weights
    during generation. Supports both single-step and multi-step extraction.

    Example:
        extractor = AttentionExtractor(transformer)

        with extractor:
            output = pipeline(prompt="A cat", ...)

        # Get raw attention maps
        maps = extractor.get_attention_maps()

        # Aggregate by token
        token_attn = extractor.aggregate_per_token(tokenizer, prompt)
    """

    def __init__(
        self,
        transformer: nn.Module,
        block_indices: Optional[List[int]] = None,
        extract_cross_attention: bool = True,
        extract_self_attention: bool = False,
    ):
        """
        Initialize attention extractor.

        Args:
            transformer: DiT transformer module
            block_indices: Which blocks to extract from (default: all)
            extract_cross_attention: Extract cross-attention weights
            extract_self_attention: Extract self-attention weights
        """
        self.transformer = transformer
        self.block_indices = block_indices
        self.extract_cross_attention = extract_cross_attention
        self.extract_self_attention = extract_self_attention

        self._hooks: List = []
        self._hook_handles: List = []
        self._registered = False

    def _find_attention_modules(self) -> List[Tuple[int, nn.Module]]:
        """
        Find attention modules in the transformer.

        Searches for common attention module patterns:
        - attn, attn1, attn2 (diffusers naming)
        - attention, self_attn, cross_attn
        - Attention class types
        """
        attention_modules = []

        # Try to find transformer blocks
        blocks = None

        # Common block container names
        for attr_name in ["blocks", "transformer_blocks", "layers", "decoder_layers"]:
            if hasattr(self.transformer, attr_name):
                blocks = getattr(self.transformer, attr_name)
                break

        if blocks is None:
            logger.warning("Could not find transformer blocks. Searching entire module.")
            blocks = list(self.transformer.modules())

        # Search for attention modules
        for idx, block in enumerate(blocks):
            if self.block_indices is not None and idx not in self.block_indices:
                continue

            # Look for cross-attention modules
            if self.extract_cross_attention:
                for attr_name in ["attn2", "cross_attn", "attn"]:
                    if hasattr(block, attr_name):
                        attn_module = getattr(block, attr_name)
                        attention_modules.append((idx, attn_module))
                        break

        return attention_modules

    def register(self) -> None:
        """Register forward hooks on attention modules."""
        if self._registered:
            return

        attention_modules = self._find_attention_modules()

        if not attention_modules:
            logger.warning(
                "No attention modules found. Attention extraction may not work "
                "with this transformer architecture."
            )
            return

        for block_idx, attn_module in attention_modules:
            hook = AttentionExtractorHook(
                block_idx=block_idx,
                extract_cross_attention=self.extract_cross_attention,
                extract_self_attention=self.extract_self_attention,
            )

            handle = attn_module.register_forward_hook(hook)

            self._hooks.append(hook)
            self._hook_handles.append(handle)

        self._registered = True
        logger.debug(f"Registered {len(self._hooks)} attention extraction hooks")

    def unregister(self) -> None:
        """Remove all forward hooks."""
        for handle in self._hook_handles:
            handle.remove()

        self._hook_handles.clear()
        self._registered = False

    def clear(self) -> None:
        """Clear all stored attention weights."""
        for hook in self._hooks:
            hook.clear()

    def __enter__(self) -> "AttentionExtractor":
        """Context manager entry - register hooks."""
        self.register()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Context manager exit - unregister hooks."""
        self.unregister()
        return False

    def get_attention_maps(
        self,
        block_idx: Optional[int] = None,
    ) -> List[torch.Tensor]:
        """
        Get extracted attention maps.

        Args:
            block_idx: Get maps from specific block (default: all blocks)

        Returns:
            List of attention weight tensors [batch, heads, latent_seq, text_seq]
        """
        if block_idx is not None:
            for hook in self._hooks:
                if hook.block_idx == block_idx:
                    return hook.cross_attention_weights
            return []

        # Aggregate from all blocks
        all_weights = []
        for hook in self._hooks:
            all_weights.extend(hook.cross_attention_weights)

        return all_weights

    def get_stacked_maps(self) -> Optional[torch.Tensor]:
        """
        Get attention maps stacked into a single tensor.

        Returns:
            Tensor [num_captures, batch, heads, latent_seq, text_seq]
            or None if no maps captured
        """
        maps = self.get_attention_maps()
        if not maps:
            return None

        return torch.stack(maps, dim=0)

    def aggregate_per_token(
        self,
        tokenizer,
        prompt: str,
        method: str = "mean",
    ) -> Dict[str, float]:
        """
        Aggregate attention weights per token.

        Args:
            tokenizer: Tokenizer used for the prompt
            prompt: The text prompt
            method: Aggregation method ("mean", "max", "sum")

        Returns:
            Dict mapping token strings to their aggregated attention weight
        """
        maps = self.get_stacked_maps()
        if maps is None:
            return {}

        # Tokenize to get token strings
        tokens = tokenizer.tokenize(prompt)

        # Average over all dimensions except text sequence
        # [num_captures, batch, heads, latent_seq, text_seq] -> [text_seq]
        if method == "mean":
            per_token = maps.mean(dim=(0, 1, 2, 3))
        elif method == "max":
            per_token = maps.amax(dim=(0, 1, 2, 3))
        elif method == "sum":
            per_token = maps.sum(dim=(0, 1, 2, 3))
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

        # Map tokens to weights
        result = {}
        for i, token in enumerate(tokens):
            if i < len(per_token):
                result[token] = per_token[i].item()

        return result

    def get_info(self) -> Optional[AttentionMapInfo]:
        """Get metadata about captured attention maps."""
        maps = self.get_attention_maps()
        if not maps:
            return None

        sample = maps[0]  # [batch, heads, latent_seq, text_seq]

        return AttentionMapInfo(
            num_steps=len(maps),
            num_heads=sample.shape[1],
            latent_seq_len=sample.shape[2],
            text_seq_len=sample.shape[3],
            block_indices=[h.block_idx for h in self._hooks],
        )


def extract_cross_attention_on_step(
    pipe,
    step: int,
    callback_kwargs: dict,
    extractor: AttentionExtractor,
) -> dict:
    """
    Callback function for extracting attention at specific steps.

    Use with pipeline's callback_on_step_end parameter.

    Example:
        extractor = AttentionExtractor(pipe.transformer)

        def callback(pipe, step, timestep, callback_kwargs):
            return extract_cross_attention_on_step(
                pipe, step, callback_kwargs, extractor
            )

        output = pipe(prompt="...", callback_on_step_end=callback)
    """
    # The actual extraction happens via the forward hooks
    # This callback just ensures the extractor is active during the step
    return callback_kwargs


def visualize_attention_heatmap(
    attention_weights: torch.Tensor,
    tokens: List[str],
    output_path: str,
    title: str = "Cross-Attention Heatmap",
) -> None:
    """
    Visualize attention weights as a heatmap.

    Args:
        attention_weights: [text_seq] or [latent_seq, text_seq]
        tokens: List of token strings
        output_path: Path to save the visualization
        title: Plot title
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib required for visualization. pip install matplotlib")
        return

    weights = attention_weights.cpu().numpy()

    fig, ax = plt.subplots(figsize=(12, 4))

    if weights.ndim == 1:
        # Bar chart for per-token weights
        ax.bar(range(len(tokens)), weights[: len(tokens)])
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha="right")
        ax.set_ylabel("Attention Weight")
    else:
        # Heatmap for 2D weights
        im = ax.imshow(weights[:, : len(tokens)], aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha="right")
        ax.set_ylabel("Latent Position")
        plt.colorbar(im, ax=ax)

    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    logger.info(f"Saved attention heatmap to {output_path}")
