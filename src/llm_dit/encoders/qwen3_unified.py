"""
Unified Qwen3 Encoder with configurable behavior.

Last Updated: 2026-02-01

This encoder unifies the functionality of:
- qwen3.py (Z-Image) - single layer extraction, enable_thinking=True
- qwen3_flux2.py (FLUX.2 Klein) - multi-layer extraction, enable_thinking=False

Key configuration differences:
| Config | Z-Image | FLUX.2 Klein |
|--------|---------|--------------|
| layer_indices | [-2] | [9, 18, 27] |
| concat_mode | "none" | "concat" |
| enable_thinking | True | False |
| output_dim | 2560 | 7680/12288 |

CRITICAL: The enable_thinking parameter is essential!
- Z-Image: enable_thinking=True (allows Qwen3's thinking tokens)
- FLUX.2: enable_thinking=False (thinking tokens corrupt embeddings)

Usage:
    # Z-Image preset
    encoder = Qwen3UnifiedEncoder.from_preset("zimage", model_path="...")

    # FLUX.2 Klein 4B preset
    encoder = Qwen3UnifiedEncoder.from_preset("klein-4b")

    # Custom config
    config = Qwen3EncoderConfig(
        layer_indices=[10, 20, 30],
        concat_mode="concat",
        enable_thinking=False,
    )
    encoder = Qwen3UnifiedEncoder(config, model_path="...")
"""

import gc
import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, Union

import torch
import torch.nn as nn
from einops import rearrange

from llm_dit.encoders.qwen3_base import (
    KLEIN_DEFAULT_LAYERS,
    QWEN3_4B_HIDDEN_DIM,
    QWEN3_8B_HIDDEN_DIM,
    ZIMAGE_DEFAULT_LAYER,
    Qwen3EncoderMixin,
)

logger = logging.getLogger(__name__)

# Feature flag for migration
USE_UNIFIED_ENCODER = os.environ.get("LLM_DIT_UNIFIED_ENCODER", "false").lower() == "true"


@dataclass
class Qwen3EncoderConfig:
    """
    Configuration for Qwen3 text encoding.

    This config class encapsulates all the differences between
    Z-Image and FLUX.2 Klein encoder configurations.
    """

    # Layer extraction configuration
    layer_indices: List[int] = field(default_factory=lambda: [-2])

    # Output concatenation mode
    # "none": Return single layer output
    # "concat": Concatenate multiple layers along hidden dimension
    concat_mode: str = "none"

    # CRITICAL: Thinking token configuration
    # Z-Image uses True (default), FLUX.2 Klein MUST use False
    enable_thinking: bool = True

    # Sequence length configuration
    max_length: int = 512
    pad_to_max: bool = True

    # Output dimension (auto-calculated if None)
    # Set explicitly for validation
    output_dim: Optional[int] = None

    def __post_init__(self):
        """Validate configuration."""
        if self.concat_mode not in ("none", "concat"):
            raise ValueError(f"Invalid concat_mode: {self.concat_mode}")

        if self.concat_mode == "concat" and len(self.layer_indices) < 2:
            raise ValueError(
                f"concat_mode='concat' requires multiple layers, got {len(self.layer_indices)}"
            )


# Preset configurations
ZIMAGE_CONFIG = Qwen3EncoderConfig(
    layer_indices=[ZIMAGE_DEFAULT_LAYER],  # [-2]
    concat_mode="none",
    enable_thinking=True,  # Z-Image allows thinking
    max_length=512,
    pad_to_max=True,
    output_dim=QWEN3_4B_HIDDEN_DIM,  # 2560
)

KLEIN_4B_CONFIG = Qwen3EncoderConfig(
    layer_indices=KLEIN_DEFAULT_LAYERS,  # [9, 18, 27]
    concat_mode="concat",
    enable_thinking=False,  # CRITICAL: Must be False for FLUX.2
    max_length=512,
    pad_to_max=True,
    output_dim=3 * QWEN3_4B_HIDDEN_DIM,  # 7680
)

KLEIN_9B_CONFIG = Qwen3EncoderConfig(
    layer_indices=KLEIN_DEFAULT_LAYERS,  # [9, 18, 27]
    concat_mode="concat",
    enable_thinking=False,  # CRITICAL: Must be False for FLUX.2
    max_length=512,
    pad_to_max=True,
    output_dim=3 * QWEN3_8B_HIDDEN_DIM,  # 12288
)

# Preset lookup
PRESETS = {
    "zimage": ZIMAGE_CONFIG,
    "klein-4b": KLEIN_4B_CONFIG,
    "klein-9b": KLEIN_9B_CONFIG,
}


class Qwen3UnifiedEncoder(nn.Module, Qwen3EncoderMixin):
    """
    Unified Qwen3 encoder with configurable layer extraction and thinking mode.

    This class unifies the functionality of both qwen3.py and qwen3_flux2.py
    into a single implementation with configuration-driven behavior.

    Args:
        config: Encoder configuration (or use from_preset)
        model_path: HuggingFace model ID or local path
        device: Target device ("cuda", "cpu", "auto")

    Example:
        # Z-Image style (single layer, thinking enabled)
        encoder = Qwen3UnifiedEncoder.from_preset("zimage", model_path="...")

        # FLUX.2 style (multi-layer concat, thinking disabled)
        encoder = Qwen3UnifiedEncoder.from_preset("klein-4b")
    """

    def __init__(
        self,
        config: Qwen3EncoderConfig,
        model_path: str,
        device: Union[str, torch.device] = "cuda",
    ):
        super().__init__()

        self.config = config
        self.model_path = model_path
        self._target_device = torch.device(device) if isinstance(device, str) else device

        # Model components (loaded lazily or immediately)
        self._model = None
        self._tokenizer = None
        self._hidden_dim = None
        self._num_layers = None
        self._is_loaded = False
        self._is_offloaded = False
        self._is_pinned = False

    @classmethod
    def from_preset(
        cls,
        preset: str,
        model_path: Optional[str] = None,
        device: Union[str, torch.device] = "cuda",
    ) -> "Qwen3UnifiedEncoder":
        """
        Create encoder from preset name.

        Args:
            preset: Preset name ("zimage", "klein-4b", "klein-9b")
            model_path: Model path (uses default for preset if not specified)
            device: Target device

        Returns:
            Configured encoder instance
        """
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(PRESETS.keys())}")

        config = PRESETS[preset]

        # Default model paths for presets
        default_paths = {
            "zimage": "Tongyi-MAI/Z-Image-Turbo",
            "klein-4b": "Qwen/Qwen3-4B-FP8",
            "klein-9b": "Qwen/Qwen3-8B-FP8",
        }

        if model_path is None:
            model_path = default_paths.get(preset)
            if model_path is None:
                raise ValueError(f"No default model path for preset: {preset}")

        encoder = cls(config=config, model_path=model_path, device=device)
        encoder._load_model()

        return encoder

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        config: Optional[Qwen3EncoderConfig] = None,
        device: Union[str, torch.device] = "cuda",
        **kwargs,
    ) -> "Qwen3UnifiedEncoder":
        """
        Load encoder from pretrained model.

        Args:
            model_path: HuggingFace model ID or local path
            config: Encoder config (defaults to Z-Image config)
            device: Target device
            **kwargs: Additional config overrides

        Returns:
            Loaded encoder instance
        """
        if config is None:
            config = ZIMAGE_CONFIG

        # Apply any overrides from kwargs
        if kwargs:
            config_dict = {
                "layer_indices": config.layer_indices,
                "concat_mode": config.concat_mode,
                "enable_thinking": config.enable_thinking,
                "max_length": config.max_length,
                "pad_to_max": config.pad_to_max,
                "output_dim": config.output_dim,
            }
            config_dict.update(kwargs)
            config = Qwen3EncoderConfig(**config_dict)

        encoder = cls(config=config, model_path=model_path, device=device)
        encoder._load_model()

        return encoder

    def _load_model(self) -> None:
        """Load Qwen3 model and tokenizer."""
        if self._is_loaded:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading Qwen3 encoder from {self.model_path}")

        # Load model
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=None,  # Auto-detect (important for FP8)
            device_map=str(self._target_device),
        )

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        # Get model dimensions
        self._hidden_dim = self._model.config.hidden_size
        self._num_layers = self._model.config.num_hidden_layers

        # Validate layer indices
        for layer_idx in self.config.layer_indices:
            # Handle negative indices
            actual_idx = layer_idx if layer_idx >= 0 else self._num_layers + layer_idx + 1
            if actual_idx < 0 or actual_idx >= self._num_layers + 1:  # +1 for embedding layer
                raise ValueError(
                    f"Layer {layer_idx} out of range for model with {self._num_layers} layers"
                )

        self._is_loaded = True
        self._is_offloaded = False

        logger.info(
            f"Qwen3 encoder loaded: {self._hidden_dim} dim, "
            f"{self._num_layers} layers, "
            f"extracting layers {self.config.layer_indices}, "
            f"enable_thinking={self.config.enable_thinking}, "
            f"output dim = {self.output_dim}"
        )

    @property
    def output_dim(self) -> int:
        """
        Output dimension (depends on layer concatenation).

        Returns:
            - Single layer mode: hidden_dim
            - Concat mode: num_layers * hidden_dim
        """
        if self._hidden_dim is None:
            # Return expected output dim from config
            if self.config.output_dim is not None:
                return self.config.output_dim
            # Default assumption
            return QWEN3_4B_HIDDEN_DIM

        if self.config.concat_mode == "concat":
            return len(self.config.layer_indices) * self._hidden_dim
        return self._hidden_dim

    @property
    def hidden_dim(self) -> int:
        """Single-layer hidden dimension."""
        return self._hidden_dim or QWEN3_4B_HIDDEN_DIM

    @property
    def device(self) -> torch.device:
        """Current device."""
        if self._model is None:
            return self._target_device
        return self._get_device(self._model)

    @property
    def dtype(self) -> torch.dtype:
        """Model dtype."""
        if self._model is None:
            return torch.bfloat16
        return self._get_dtype(self._model)

    @torch.no_grad()
    def forward(self, txt: List[str]) -> torch.Tensor:
        """
        Encode text prompts to embeddings.

        Args:
            txt: List of text prompts

        Returns:
            Tensor with shape depending on config:
            - Single layer mode: [batch, seq_len, hidden_dim]
            - Concat mode: [batch, seq_len, num_layers * hidden_dim]
        """
        if not self._is_loaded:
            self._load_model()

        # Start async GPU transfer if offloaded (DMA runs in parallel with tokenization)
        needs_sync = False
        if self._is_offloaded:
            self._model.to(self._target_device, non_blocking=self._is_pinned)
            needs_sync = self._is_pinned and self._target_device.type == "cuda"
            self._is_offloaded = False

        # Tokenize on CPU while DMA transfer runs in the background
        all_input_ids = []
        all_attention_masks = []

        for prompt in txt:
            # Apply chat template with configurable thinking mode
            messages = [{"role": "user", "content": prompt}]
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.config.enable_thinking,  # CONFIGURABLE
            )

            # Tokenize
            padding_strategy = "max_length" if self.config.pad_to_max else "longest"
            model_inputs = self._tokenizer(
                text,
                return_tensors="pt",
                padding=padding_strategy,
                truncation=True if self.config.max_length else False,
                max_length=self.config.max_length,
            )

            all_input_ids.append(model_inputs["input_ids"])
            all_attention_masks.append(model_inputs["attention_mask"])

        # Synchronize after tokenization (DMA may already be done by this point)
        if needs_sync:
            torch.cuda.synchronize()

        # Batch inputs and move to model device
        input_ids = torch.cat(all_input_ids, dim=0).to(self._model.device)
        attention_mask = torch.cat(all_attention_masks, dim=0).to(self._model.device)

        logger.info(
            f"[Qwen3Encoder] Encoding {len(txt)} prompts, "
            f"seq_len={input_ids.shape[1]}, "
            f"layers={self.config.layer_indices}, "
            f"enable_thinking={self.config.enable_thinking}"
        )

        # Forward pass with hidden state extraction
        output = self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        # Extract specified layers
        # hidden_states is a tuple of (num_layers + 1) tensors
        # Index 0 is embeddings, indices 1-N are layer outputs
        layers = [output.hidden_states[k] for k in self.config.layer_indices]

        # Process based on concat mode
        if self.config.concat_mode == "concat":
            # Stack and reshape: [B, num_layers, L, D] -> [B, L, num_layers*D]
            out = torch.stack(layers, dim=1)
            out = rearrange(out, "b c l d -> b l (c d)")
        else:
            # Single layer mode - return first (and only) layer
            out = layers[0]

        logger.info(f"[Qwen3Encoder] Output shape: {list(out.shape)}")

        return out

    def encode(self, txt: List[str]) -> torch.Tensor:
        """Encode text prompts (alias for forward)."""
        return self.forward(txt)

    def encode_single(self, prompt: str) -> torch.Tensor:
        """Encode a single prompt."""
        return self.forward([prompt])

    def offload(self) -> None:
        """Offload model to CPU and free GPU memory.

        When pinned memory is enabled, re-pins parameters after the CPU
        round-trip. PyTorch allocates new (non-pinned) CPU tensors when
        moving from CUDA back to CPU, so the original pinned memory is lost.
        """
        if self._model is not None:
            self._model.to("cpu")
            if self._is_pinned:
                # Re-pin after CUDA round-trip (PyTorch allocates new non-pinned CPU tensors)
                for param in self._model.parameters():
                    if not param.data.is_pinned():
                        param.data = param.data.pin_memory()
                for buf in self._model.buffers():
                    if not buf.data.is_pinned():
                        buf.data = buf.data.pin_memory()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self._is_offloaded = True

    def offload_to_pinned(self) -> None:
        """Offload model to CPU with pinned memory for fast GPU shuttle.

        Pinned (page-locked) memory enables direct DMA transfers between
        CPU and GPU, avoiding the intermediate copy through a staging buffer.
        This makes subsequent .to("cuda", non_blocking=True) calls ~2-3x faster.

        Safe to call on systems with sufficient RAM (encoder is ~8GB for Qwen3-8B).
        """
        if self._model is None:
            return

        logger.info("Offloading encoder to CPU with pinned memory...")
        self._model.to("cpu")

        # Pin all parameter and buffer memory for DMA transfers
        pinned_count = 0
        for param in self._model.parameters():
            if not param.data.is_pinned():
                param.data = param.data.pin_memory()
                pinned_count += 1
        for buf in self._model.buffers():
            if not buf.data.is_pinned():
                buf.data = buf.data.pin_memory()

        self._is_offloaded = True
        self._is_pinned = True
        logger.info(f"Encoder offloaded with {pinned_count} pinned tensors")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def to(self, device: Union[str, torch.device]) -> "Qwen3UnifiedEncoder":
        """Move encoder to device."""
        device = torch.device(device) if isinstance(device, str) else device
        if self._model is not None:
            self._model.to(device)
        self._target_device = device
        self._is_offloaded = device.type == "cpu"
        return self


def get_unified_encoder(
    preset: str,
    model_path: Optional[str] = None,
    device: str = "cuda",
) -> Qwen3UnifiedEncoder:
    """
    Factory function to get a unified encoder.

    This is the recommended way to create encoders in new code.
    Use this instead of directly importing Qwen3Encoder or Qwen3Flux2Encoder.

    Args:
        preset: Encoder preset ("zimage", "klein-4b", "klein-9b")
        model_path: Optional model path override
        device: Target device

    Returns:
        Configured Qwen3UnifiedEncoder

    Example:
        # Z-Image encoder
        encoder = get_unified_encoder("zimage")

        # FLUX.2 Klein 4B encoder
        encoder = get_unified_encoder("klein-4b")
    """
    return Qwen3UnifiedEncoder.from_preset(
        preset=preset,
        model_path=model_path,
        device=device,
    )
