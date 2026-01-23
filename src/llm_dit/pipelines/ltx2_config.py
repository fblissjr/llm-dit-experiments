"""
LTX-2 Optimization Configuration.

Last Updated: 2026-01-23

Memory and performance optimization settings for LTX-2 video generation.
Matches the reference LTX-2 repo's memory optimization approach, optimized
for RTX 4090 24GB VRAM.

Reference: coderef/LTX-2/packages/ltx-pipelines/README.md
- FP8 Transformer (--enable-fp8 flag)
- PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
- Memory cleanup between stages (default on, can skip if enough VRAM)
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class LTX2OptimizationConfig:
    """
    Memory and performance optimization settings for LTX-2.

    Matches reference repo patterns, optimized for RTX 4090 24GB.
    Extensible for future additions (torch.compile, flash_attn, etc.)

    Attributes:
        text_encoder_device: Device for Gemma3 text encoder. "cpu" recommended
            for 24GB GPUs since the 12B model in bf16 takes ~24GB.
        transformer_device: Device for DiT transformer. "cuda" for generation.
        vae_device: Device for VAE decoder. "cuda" for decoding.
        quantize_transformer: Whether to use FP8 quantization. Essential for
            fitting the 13B transformer on 24GB.
        precision: Quantization precision. "fp8-native" (recommended) or "bf16".
        cleanup_between_stages: Whether to run garbage collection and CUDA
            cache clearing between pipeline stages. Matches reference repo
            default (on). Disable for speed on high-VRAM systems.

    Future extensibility (placeholders, not yet implemented):
        enable_torch_compile: Enable torch.compile for transformer
        enable_flash_attn: Enable Flash Attention
        tiled_vae: Enable tiled VAE for large videos
        gradient_checkpointing: For training/fine-tuning
    """

    # Device placement
    text_encoder_device: str = "cpu"  # cpu recommended for 24GB
    transformer_device: str = "cuda"
    vae_device: str = "cuda"

    # Quantization
    quantize_transformer: bool = True  # FP8 quantization
    precision: Literal["fp8-native", "bf16", "fp8-quanto", "int8-quanto", "int4-quanto"] = (
        "fp8-native"
    )

    # Memory management
    cleanup_between_stages: bool = True  # Match ref repo default

    # Future extensibility (not implemented yet)
    enable_torch_compile: bool = False  # Placeholder
    enable_flash_attn: bool = False  # Placeholder
    tiled_vae: bool = False  # Placeholder for large videos
    gradient_checkpointing: bool = False  # Placeholder for training

    @classmethod
    def for_24gb_gpu(cls) -> "LTX2OptimizationConfig":
        """
        Optimized defaults for RTX 4090 / 24GB VRAM.

        This is the default configuration that matches the reference repo's
        recommended settings for consumer GPUs with 24GB VRAM.

        Returns:
            LTX2OptimizationConfig with 24GB-optimized settings
        """
        return cls(
            text_encoder_device="cpu",
            transformer_device="cuda",
            vae_device="cuda",
            quantize_transformer=True,
            precision="fp8-native",
            cleanup_between_stages=True,
        )

    @classmethod
    def for_high_vram(cls) -> "LTX2OptimizationConfig":
        """
        Configuration for GPUs with >40GB VRAM.

        Skips memory cleanup for speed and can optionally run full precision.
        Text encoder can stay on GPU since there's room.

        Returns:
            LTX2OptimizationConfig with high-VRAM settings
        """
        return cls(
            text_encoder_device="cuda",  # Can fit on GPU
            transformer_device="cuda",
            vae_device="cuda",
            quantize_transformer=False,  # Full precision
            precision="bf16",
            cleanup_between_stages=False,  # Skip cleanup for speed
        )

    @classmethod
    def for_low_vram(cls) -> "LTX2OptimizationConfig":
        """
        Configuration for GPUs with 16GB VRAM or less.

        More aggressive memory management with everything possible on CPU.
        May require further frame count reduction.

        Returns:
            LTX2OptimizationConfig with low-VRAM settings
        """
        return cls(
            text_encoder_device="cpu",
            transformer_device="cuda",
            vae_device="cpu",  # Decode on CPU
            quantize_transformer=True,
            precision="fp8-native",
            cleanup_between_stages=True,
        )
