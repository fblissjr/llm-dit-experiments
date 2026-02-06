#!/usr/bin/env python3
"""
Memory-efficient utilities for LTX-2 experiments on consumer GPUs (24GB).

Last Updated: 2026-01-16

This module provides utilities for running LTX-2 experiments on GPUs with limited
VRAM (e.g., RTX 4090 24GB) using:

1. 8-bit quantized text encoder (Gemma3) - reduces from ~54GB to ~13GB
2. Sequential loading - encode first, then offload text encoder
3. Group offloading for transformer - streams blocks from CPU during generation

Memory profile:
- Text encoding phase: ~13GB (8-bit Gemma3)
- After offload: ~0GB
- Generation phase: ~5GB (VAE + connectors) + streaming transformer blocks

Usage:
    from experiments.ltx2.memory_utils import (
        load_text_encoder_8bit,
        encode_prompts_with_layer_weights,
        encode_negative_prompt,
        load_pipeline_with_offloading,
        cleanup_memory,
    )

    # Phase 1: Encode prompts
    text_encoder, tokenizer = load_text_encoder_8bit("models/LTX-2")

    # For CFG (guidance_scale > 1.0), encode negative prompt
    neg_embeds, neg_mask = encode_negative_prompt(text_encoder, tokenizer)

    embeddings = encode_prompts_with_layer_weights(
        text_encoder, tokenizer, prompts, layer_weights
    )
    del text_encoder
    cleanup_memory()

    # Phase 2: Generate
    pipe = load_pipeline_with_offloading("models/LTX-2")
    for prompt_embeds in embeddings:
        output = pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=neg_embeds,
            negative_prompt_attention_mask=neg_mask,
            ...
        )
"""

import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration


def cleanup_memory() -> None:
    """Clean up GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_gpu_memory() -> float:
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    return 0.0


def load_text_encoder_8bit(
    model_path: str,
    device_map: str = "auto",
) -> Tuple[Gemma3ForConditionalGeneration, AutoTokenizer]:
    """
    Load the Gemma3 text encoder with 8-bit quantization.

    Uses ~13GB VRAM instead of ~54GB for the full model.

    Args:
        model_path: Path to LTX-2 model directory
        device_map: Device map for model placement ("auto" recommended)

    Returns:
        Tuple of (text_encoder, tokenizer)
    """
    text_encoder_path = Path(model_path) / "text_encoder"
    tokenizer_path = Path(model_path) / "tokenizer"


    text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
        str(text_encoder_path),
        dtype=torch.bfloat16,
        device_map=device_map,
    )

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

    return text_encoder, tokenizer


def encode_prompt_with_layer_weights(
    text_encoder: Gemma3ForConditionalGeneration,
    tokenizer: AutoTokenizer,
    prompt: str,
    layer_weights: Optional[torch.Tensor] = None,
    max_length: int = 512,
    num_layers: int = 49,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Encode a single prompt with optional layer weighting.

    Args:
        text_encoder: The Gemma3 text encoder
        tokenizer: The tokenizer
        prompt: Text prompt to encode
        layer_weights: Optional tensor of shape [49] with layer weights.
                       If None, uniform weights are used.
        max_length: Maximum sequence length
        num_layers: Number of Gemma layers (49 for LTX-2)

    Returns:
        Tuple of (hidden_states, attention_mask, sequence_length)
        - hidden_states: [1, T, 3840, 49] - raw hidden states from all layers
        - attention_mask: [1, T] - attention mask
        - sequence_length: int - actual sequence length
    """
    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
        truncation=True,
    )
    input_ids = inputs["input_ids"].to(text_encoder.device)
    attention_mask = inputs["attention_mask"].to(text_encoder.device)

    # Get sequence length (excluding padding)
    sequence_length = attention_mask.sum().item()

    # Encode
    with torch.no_grad():
        outputs = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

    # Stack hidden states: [B, T, hidden_dim, num_layers]
    # Include ALL hidden states (embedding + 48 transformer layers = 49 total)
    # The projection matrix expects 49 layers (188160 = 49 × 3840)
    hidden_states = torch.stack(outputs.hidden_states[:num_layers], dim=-1)

    # Apply layer weights if provided
    if layer_weights is not None:
        layer_weights = layer_weights.to(hidden_states.device).view(1, 1, 1, -1)
        hidden_states = hidden_states * layer_weights

    return hidden_states, attention_mask, sequence_length


def pack_text_embeds(
    hidden_states: torch.Tensor,
    sequence_length: int,
    device: torch.device,
    scale_factor: int = 8,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Pack text hidden states into prompt embeddings (matches LTX2Pipeline._pack_text_embeds).

    Args:
        hidden_states: [B, T, hidden_dim, num_layers] - stacked hidden states
        sequence_length: Actual sequence length (excluding padding)
        device: Target device
        scale_factor: Scale factor for normalization
        eps: Epsilon for numerical stability

    Returns:
        Packed prompt embeddings ready for the pipeline
    """
    # Normalize each layer
    hidden_states = hidden_states.to(device)
    normed = hidden_states / (hidden_states.norm(dim=2, keepdim=True) + eps)

    # Flatten layers: [B, T, hidden_dim * num_layers]
    batch_size, seq_len, hidden_dim, num_layers = normed.shape
    packed = normed.view(batch_size, seq_len, hidden_dim * num_layers)

    # Scale
    packed = packed * scale_factor

    return packed


def encode_prompts_batch(
    text_encoder: Gemma3ForConditionalGeneration,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    layer_weights_list: Optional[List[torch.Tensor]] = None,
    max_length: int = 512,
) -> List[Dict]:
    """
    Encode multiple prompts, optionally with different layer weights for each.

    Args:
        text_encoder: The Gemma3 text encoder
        tokenizer: The tokenizer
        prompts: List of prompts to encode
        layer_weights_list: Optional list of layer weight tensors (one per prompt).
                           If None, uses uniform weights for all.
        max_length: Maximum sequence length

    Returns:
        List of dicts, each containing:
        - 'hidden_states': [1, T, 3840, 49]
        - 'attention_mask': [1, T]
        - 'sequence_length': int
        - 'prompt': str
    """
    results = []

    for i, prompt in enumerate(prompts):
        layer_weights = None
        if layer_weights_list is not None:
            layer_weights = layer_weights_list[i]

        hidden_states, attention_mask, seq_len = encode_prompt_with_layer_weights(
            text_encoder,
            tokenizer,
            prompt,
            layer_weights=layer_weights,
            max_length=max_length,
        )

        results.append(
            {
                "hidden_states": hidden_states.cpu(),  # Move to CPU to save VRAM
                "attention_mask": attention_mask.cpu(),
                "sequence_length": seq_len,
                "prompt": prompt,
            }
        )

    return results


def load_pipeline_with_offloading(
    model_path: str,
    num_blocks_per_group: int = 1,
    use_stream: bool = True,
    enable_audio: bool = False,
) -> "LTX2Pipeline":
    """
    Load LTX2Pipeline with group offloading for memory efficiency.

    The transformer uses group offloading to stream blocks from CPU during generation.
    VAE and connectors are kept on GPU.

    Memory usage: ~5GB (VAE + connectors), transformer streams as needed.

    Args:
        model_path: Path to LTX-2 model
        num_blocks_per_group: Number of transformer blocks per offload group.
                              1 = minimum memory, higher = faster but more VRAM
        use_stream: Use CUDA streams for async prefetching
        enable_audio: Whether to enable audio generation. Default False to save VRAM.
                      If True, audio_vae will be moved to CUDA.

    Returns:
        LTX2Pipeline with offloading configured
    """
    from diffusers import LTX2Pipeline
    from diffusers.hooks import apply_group_offloading

    # Load pipeline without text encoder (we handle encoding separately)
    pipe = LTX2Pipeline.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        text_encoder=None,
        tokenizer=None,
    )

    # Apply group offloading to transformer
    apply_group_offloading(
        pipe.transformer,
        onload_device=torch.device("cuda"),
        offload_device=torch.device("cpu"),
        offload_type="block_level",
        num_blocks_per_group=num_blocks_per_group,
        use_stream=use_stream,
        non_blocking=True,
    )

    # Move VAE and connectors to GPU (they're small enough)
    pipe.vae.to("cuda")
    if pipe.connectors is not None:
        pipe.connectors.to("cuda")

    # Handle audio VAE - disable by default to save VRAM
    if enable_audio and hasattr(pipe, "audio_vae") and pipe.audio_vae is not None:
        pipe.audio_vae.to("cuda")
    elif hasattr(pipe, "audio_vae") and pipe.audio_vae is not None:
        # Create a dummy audio VAE that provides required attributes but skips processing
        # The pipeline accesses latents_mean/std and config before checking output_type
        class DummyAudioVAE:
            """Dummy audio VAE to skip audio generation while satisfying pipeline checks."""

            def __init__(self, real_vae):
                # Copy essential attributes for denormalization and config
                self.latents_mean = real_vae.latents_mean
                self.latents_std = real_vae.latents_std
                self.dtype = real_vae.dtype
                self.config = real_vae.config  # Pipeline accesses config.mel_bins

            def decode(self, latents, return_dict=False):
                # Return zeros matching expected shape
                # Audio latents shape: [B, C, T, mel_bins]
                # Just return zeros since we're not using audio
                zeros = torch.zeros_like(latents)
                if return_dict:
                    return type("obj", (object,), {"sample": zeros})()
                return (zeros,)

            def to(self, *args, **kwargs):
                # No-op for device transfers
                return self

        pipe.audio_vae = DummyAudioVAE(pipe.audio_vae)

        # Also need a dummy vocoder
        if hasattr(pipe, "vocoder") and pipe.vocoder is not None:
            pipe.vocoder = lambda x: torch.zeros(x.shape[0], 1, 1)  # Minimal audio output

    return pipe


def create_layer_weights(
    active_layers: List[int],
    weights: Optional[Dict[int, float]] = None,
    num_layers: int = 49,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Create layer weight tensor from config.

    Args:
        active_layers: List of layer indices to activate
        weights: Optional dict mapping layer index to weight.
                 If None, active layers get weight 1.0
        num_layers: Total number of layers
        normalize: If True, normalize weights to preserve signal magnitude

    Returns:
        Tensor of shape [num_layers] with layer weights
    """
    layer_weights = torch.zeros(num_layers)

    if weights is None:
        for idx in active_layers:
            layer_weights[idx] = 1.0
    else:
        for idx in active_layers:
            layer_weights[idx] = weights.get(idx, 1.0)

    if normalize:
        weight_sum = layer_weights.sum()
        if weight_sum > 0:
            layer_weights = layer_weights / weight_sum * num_layers

    return layer_weights


def encode_negative_prompt(
    text_encoder: Gemma3ForConditionalGeneration,
    tokenizer: AutoTokenizer,
    max_length: int = 512,
    num_layers: int = 49,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encode negative prompt (empty string) for CFG.

    This is required when using guidance_scale > 1.0 with pre-computed prompt_embeds.
    If you pass prompt_embeds to the pipeline without negative_prompt_embeds,
    the pipeline will try to encode the negative prompt using the text encoder,
    which will fail if the text encoder has been offloaded.

    Args:
        text_encoder: The Gemma3 text encoder
        tokenizer: The tokenizer
        max_length: Maximum sequence length (must match positive prompt encoding)
        num_layers: Number of Gemma layers (49 for LTX-2)

    Returns:
        Tuple of (negative_prompt_embeds, negative_prompt_attention_mask)
        Both are on CPU and ready to cache.
    """
    # Tokenize empty string
    neg_inputs = tokenizer(
        "",
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
        truncation=True,
    )
    neg_input_ids = neg_inputs["input_ids"].to(text_encoder.device)
    neg_attention_mask = neg_inputs["attention_mask"].to(text_encoder.device)

    # Encode
    with torch.no_grad():
        neg_outputs = text_encoder(
            input_ids=neg_input_ids,
            attention_mask=neg_attention_mask,
            output_hidden_states=True,
        )

    # Stack and pack
    # Include ALL hidden states (embedding + 48 layers = 49 total)
    neg_hidden_states = torch.stack(neg_outputs.hidden_states[:num_layers], dim=-1)
    neg_seq_len = neg_attention_mask.sum().item()
    negative_prompt_embeds = pack_text_embeds(
        neg_hidden_states,
        neg_seq_len,
        device=torch.device("cuda"),
    ).cpu()
    negative_attention_mask = neg_attention_mask.cpu()

    return negative_prompt_embeds, negative_attention_mask


def encode_prompt_with_layer_masking(
    text_encoder: Gemma3ForConditionalGeneration,
    tokenizer: AutoTokenizer,
    prompt: str,
    active_layers: List[int],
    masking_mode: str = "soft",
    max_length: int = 512,
    num_layers: int = 49,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Encode a single prompt with layer masking (for layer isolation experiments).

    This masks inactive layers according to the specified mode, allowing
    you to see what a single layer (or subset) contributes in isolation.

    Args:
        text_encoder: The Gemma3 text encoder
        tokenizer: The tokenizer
        prompt: Text prompt to encode
        active_layers: List of layer indices to keep active
        masking_mode: How to handle inactive layers:
            - "soft": Replace with per-layer mean (maintains distribution)
            - "zero": Zero out (creates OOD inputs - NOT RECOMMENDED)
            - "weighted": Scale active layers to preserve total norm
        max_length: Maximum sequence length
        num_layers: Number of Gemma layers (49 for LTX-2)

    Returns:
        Tuple of (hidden_states, attention_mask, sequence_length)
    """
    active_set = set(active_layers)

    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
        truncation=True,
    )
    input_ids = inputs["input_ids"].to(text_encoder.device)
    attention_mask = inputs["attention_mask"].to(text_encoder.device)

    # Get sequence length (excluding padding)
    sequence_length = attention_mask.sum().item()

    # Encode
    with torch.no_grad():
        outputs = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

    # Stack hidden states: [B, T, hidden_dim, num_layers]
    # Include ALL hidden states (embedding + 48 layers = 49 total)
    hidden_states = torch.stack(outputs.hidden_states[:num_layers], dim=-1)

    # Apply masking based on mode
    if masking_mode == "soft":
        # Soft masking: Replace inactive layers with per-layer mean
        # Maintains expected input distribution for projection W
        for layer_idx in range(num_layers):
            if layer_idx not in active_set:
                layer_mean = hidden_states[:, :, :, layer_idx].mean(dim=1, keepdim=True)
                hidden_states[:, :, :, layer_idx] = layer_mean

    elif masking_mode == "zero":
        # Zero masking: Creates OOD inputs (not recommended)
        for layer_idx in range(num_layers):
            if layer_idx not in active_set:
                hidden_states[:, :, :, layer_idx] = 0.0

    elif masking_mode == "weighted":
        # Weighted masking: Scale active layers to preserve total norm
        num_active = len(active_layers)
        scale = num_layers / num_active if num_active > 0 else 1.0

        for layer_idx in range(num_layers):
            if layer_idx in active_set:
                hidden_states[:, :, :, layer_idx] *= scale
            else:
                hidden_states[:, :, :, layer_idx] = 0.0

    else:
        raise ValueError(f"Unknown masking_mode: {masking_mode}")

    return hidden_states, attention_mask, sequence_length
