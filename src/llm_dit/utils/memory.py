"""
Memory management utilities for LTX-2 and other video generation pipelines.

Last Updated: 2026-03-13

Provides memory cleanup, monitoring, and estimation functions for
running on memory-constrained GPUs (e.g., RTX 4090 24GB).

Usage:
    from llm_dit.utils.memory import cleanup_memory, get_gpu_memory, log_memory_usage

    # Clean up after encoding
    del text_encoder
    cleanup_memory()

    # Monitor during generation
    log_memory_usage("After transformer forward")

    # Estimate before loading
    estimate = estimate_vram_usage(model_size_gb=19, batch_size=1)
"""

import gc
import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

try:
    import psutil
    _psutil = psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _psutil = None
    _PSUTIL_AVAILABLE = False


def format_memory_gb(bytes_val: int | float) -> str:
    """Format memory value in GB with 2 decimal places."""
    return f"{bytes_val / 1e9:.2f}GB"


def log_memory_debug(
    prefix: str,
    *,
    component: str = "",
    device: torch.device | str | None = None,
) -> None:
    """Log GPU allocated/reserved + CPU RSS at DEBUG level.

    Gated on DEBUG -- zero cost when not debugging.

    Args:
        prefix: Label for this log point (e.g. "After load_state_dict").
        component: Component name for the log tag (e.g. "FLUX2:Loader").
        device: CUDA device for memory queries (None = default device).
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return

    tag = f"[{component}:{prefix}]" if component else f"[{prefix}]"
    msg_parts = [tag]

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        msg_parts.append(f"GPU allocated: {format_memory_gb(allocated)}")
        msg_parts.append(f"reserved: {format_memory_gb(reserved)}")

    if _PSUTIL_AVAILABLE and _psutil is not None:
        process = _psutil.Process()
        mem_info = process.memory_info()
        msg_parts.append(f"CPU RSS: {format_memory_gb(mem_info.rss)}")

    logger.debug(" -> ".join(msg_parts))


def cleanup_memory(label: str = "") -> None:
    """
    Clean up GPU memory by running garbage collection and clearing CUDA cache.

    Call this after deleting large models or tensors to ensure memory is
    actually freed. Includes synchronization to ensure all CUDA operations
    complete before clearing cache.

    When ``label`` is provided, logs VRAM state after cleanup at INFO level.
    This is the canonical VRAM diagnostic -- use it at every stage transition
    in multi-stage pipelines so OOM failures can be diagnosed from logs alone.

    The log includes three metrics:
    - **allocated**: memory actively used by PyTorch tensors
    - **reserved**: memory held by PyTorch's CUDA caching allocator
    - **cuda_free**: memory available according to the CUDA driver (what matters for OOM)

    Example:
        del text_encoder
        cleanup_memory("post_encoder_unload")
        # INFO [VRAM:post_encoder_unload] allocated=0.1GB, reserved=0.2GB, cuda_free=23.3GB, freed=14.9GB
    """
    before_allocated = 0.0
    if label and torch.cuda.is_available():
        before_allocated = torch.cuda.memory_allocated()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        if label:
            after_allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            cuda_free = torch.cuda.mem_get_info()[0]
            freed = before_allocated - after_allocated

            msg = (
                f"[VRAM:{label}] "
                f"allocated={after_allocated / 1024**3:.1f}GB, "
                f"reserved={reserved / 1024**3:.1f}GB, "
                f"cuda_free={cuda_free / 1024**3:.1f}GB"
            )
            if freed > 10 * 1024**2:  # only show freed if > 10 MB
                msg += f", freed={freed / 1024**3:.1f}GB"
            logger.info(msg)


def get_gpu_memory() -> float:
    """
    Get current GPU memory usage in GB.

    Returns:
        Memory currently allocated on CUDA in GB, or 0.0 if no CUDA device.

    Example:
        >>> print(f"Current memory: {get_gpu_memory():.2f}GB")
        Current memory: 12.34GB
    """
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    return 0.0


def get_gpu_memory_reserved() -> float:
    """
    Get GPU memory reserved by PyTorch in GB.

    Reserved memory includes both allocated and cached (for future allocations).
    This is closer to what nvidia-smi reports.

    Returns:
        Memory reserved on CUDA in GB, or 0.0 if no CUDA device.
    """
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved() / 1e9
    return 0.0


def get_gpu_memory_stats() -> dict:
    """
    Get comprehensive GPU memory statistics.

    Returns:
        Dict with allocated, reserved, max_allocated, and free memory in GB.
    """
    if not torch.cuda.is_available():
        return {
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
            "max_allocated_gb": 0.0,
            "free_gb": 0.0,
            "total_gb": 0.0,
        }

    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    max_allocated = torch.cuda.max_memory_allocated() / 1e9

    # Get total and free from device properties
    device = torch.cuda.current_device()
    total = torch.cuda.get_device_properties(device).total_memory / 1e9
    free = total - reserved

    return {
        "allocated_gb": allocated,
        "reserved_gb": reserved,
        "max_allocated_gb": max_allocated,
        "free_gb": free,
        "total_gb": total,
    }


def log_memory_usage(label: str = "", level: int = logging.INFO) -> None:
    """
    Log current GPU memory usage with an optional label.

    Args:
        label: Optional description of when this log was captured
        level: Logging level (default INFO)

    Example:
        log_memory_usage("After loading transformer")
        # INFO: [Memory] After loading transformer: 19.23GB allocated, 20.50GB reserved
    """
    stats = get_gpu_memory_stats()

    if label:
        msg = f"[Memory] {label}: "
    else:
        msg = "[Memory] "

    msg += f"{stats['allocated_gb']:.2f}GB allocated, {stats['reserved_gb']:.2f}GB reserved"

    if stats['total_gb'] > 0:
        pct = (stats['reserved_gb'] / stats['total_gb']) * 100
        msg += f" ({pct:.1f}% of {stats['total_gb']:.1f}GB)"

    logger.log(level, msg)


def reset_peak_memory_stats() -> None:
    """Reset the peak memory allocation tracking."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def estimate_vram_usage(
    model_size_gb: float,
    batch_size: int = 1,
    sequence_length: int = 256,
    hidden_dim: int = 4096,
    num_layers: int = 32,
    dtype_bytes: int = 2,  # bf16 = 2 bytes
    gradient_checkpointing: bool = False,
) -> dict:
    """
    Estimate VRAM usage for a model with given parameters.

    This is a rough estimate based on model weights + activations.
    Actual usage depends on many factors including batch size,
    sequence length, and whether gradient checkpointing is enabled.

    Args:
        model_size_gb: Model weights size in GB
        batch_size: Batch size for inference
        sequence_length: Input sequence length
        hidden_dim: Hidden dimension of model
        num_layers: Number of transformer layers
        dtype_bytes: Bytes per element (2 for bf16, 4 for fp32)
        gradient_checkpointing: If True, reduces activation memory

    Returns:
        Dict with estimated weights, activations, and peak memory in GB.

    Example:
        >>> estimate_vram_usage(model_size_gb=19.0, batch_size=1)
        {'weights_gb': 19.0, 'activations_gb': 1.5, 'peak_gb': 21.0}
    """
    # Weights are constant
    weights_gb = model_size_gb

    # Estimate activation memory per layer
    # For transformer: ~4 * batch * seq * hidden * dtype_bytes per layer
    # (Q, K, V, attention output)
    activation_per_layer = 4 * batch_size * sequence_length * hidden_dim * dtype_bytes / 1e9

    if gradient_checkpointing:
        # Only store activations for checkpointed layers
        activations_gb = activation_per_layer * (num_layers ** 0.5)
    else:
        activations_gb = activation_per_layer * num_layers

    # Add buffer for temporary tensors (attention scores, etc.)
    # Attention scores: batch * heads * seq * seq * dtype_bytes
    # Assume ~32 heads
    attention_buffer = batch_size * 32 * sequence_length * sequence_length * dtype_bytes / 1e9
    activations_gb += attention_buffer

    # Peak includes some overhead for PyTorch internals
    peak_gb = weights_gb + activations_gb * 1.2

    return {
        "weights_gb": weights_gb,
        "activations_gb": round(activations_gb, 2),
        "peak_gb": round(peak_gb, 2),
    }


def estimate_ltx2_vram(
    resolution: tuple[int, int] = (768, 512),
    num_frames: int = 33,
    use_8bit_encoder: bool = True,
    enable_audio: bool = False,
) -> dict:
    """
    Estimate VRAM usage for LTX-2 video generation.

    LTX-2 memory components:
    - Text encoder (Gemma3 12B): ~54GB (fp32), ~27GB (bf16), ~13GB (8-bit)
    - Transformer: ~19GB (bf16)
    - VAE: ~2GB
    - Audio VAE + vocoder: ~1GB (if enabled)

    With sequential loading (encode first, then offload):
    - Peak during encoding: ~13GB (8-bit) or ~27GB (bf16)
    - Peak during generation: ~22GB (transformer + VAE + activations)

    Args:
        resolution: (height, width) tuple
        num_frames: Number of video frames
        use_8bit_encoder: Use 8-bit quantized text encoder
        enable_audio: Include audio generation

    Returns:
        Dict with estimated memory per component and peak usage.
    """
    height, width = resolution

    # Base model sizes (bf16)
    encoder_full = 27.0  # Gemma3 12B in bf16
    encoder_8bit = 13.0  # Gemma3 12B in 8-bit
    transformer = 19.0
    vae = 2.0
    audio_vae = 0.5 if enable_audio else 0.0
    vocoder = 0.5 if enable_audio else 0.0

    # Text encoder memory
    encoder_gb = encoder_8bit if use_8bit_encoder else encoder_full

    # Activation memory scales with resolution and frames
    # Latent size: (frames / 8) * (H / 32) * (W / 32) * channels
    latent_size = (num_frames // 8) * (height // 32) * (width // 32) * 128
    latent_gb = latent_size * 2 / 1e9  # bf16

    # Cross-attention activations
    text_length = 256  # Typical prompt length
    cross_attn_gb = latent_size * text_length * 2 / 1e9  # bf16 attention scores

    # Transformer activations (rough estimate)
    transformer_activations = latent_gb * 4 + cross_attn_gb * 2

    # Peak estimates for different phases
    encoding_peak = encoder_gb + 2.0  # Encoder + tokenization overhead
    generation_peak = transformer + vae + audio_vae + transformer_activations + 3.0  # +3GB overhead

    return {
        "encoder_gb": encoder_gb,
        "transformer_gb": transformer,
        "vae_gb": vae,
        "audio_gb": audio_vae + vocoder,
        "activations_gb": round(transformer_activations, 2),
        "encoding_peak_gb": round(encoding_peak, 2),
        "generation_peak_gb": round(generation_peak, 2),
        "recommended_vram_gb": max(24.0, round(generation_peak, 0)),
    }


class MemoryTracker:
    """
    Context manager for tracking memory usage during operations.

    Example:
        with MemoryTracker("Transformer forward") as tracker:
            output = transformer(latents)
        print(f"Used {tracker.delta_gb:.2f}GB")
    """

    def __init__(self, label: str = "", log_on_exit: bool = True):
        """
        Initialize memory tracker.

        Args:
            label: Description for logging
            log_on_exit: Whether to log memory delta on exit
        """
        self.label = label
        self.log_on_exit = log_on_exit
        self.start_gb: float = 0.0
        self.end_gb: float = 0.0
        self.delta_gb: float = 0.0
        self.peak_gb: float = 0.0

    def __enter__(self) -> "MemoryTracker":
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.start_gb = get_gpu_memory()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.end_gb = get_gpu_memory()
        self.delta_gb = self.end_gb - self.start_gb

        if torch.cuda.is_available():
            self.peak_gb = torch.cuda.max_memory_allocated() / 1e9

        if self.log_on_exit and self.label:
            logger.info(
                f"[Memory] {self.label}: "
                f"{self.start_gb:.2f}GB -> {self.end_gb:.2f}GB "
                f"(delta: {self.delta_gb:+.2f}GB, peak: {self.peak_gb:.2f}GB)"
            )

        return False
