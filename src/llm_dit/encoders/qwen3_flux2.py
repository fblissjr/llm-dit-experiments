"""
-----
**IMPORTANT**:
QUESTION TO CLAUDE: IS THIS DEPRECATED???
-----

Qwen3 Encoder for FLUX.2 Klein models.

Last Updated: 2026-01-23

Implements multi-layer extraction from Qwen3 models (4B and 8B) for FLUX.2
text conditioning. Unlike the standard single-layer encoder, this extracts
3 specific hidden layers and concatenates them.

Key Features:
- Multi-layer extraction: Layers [9, 18, 27] for depth-aware representations
- Chat template with enable_thinking=False (critical for FLUX.2)
- Output dimension: 3 * model_dim (e.g., 12288 for 8B, 7680 for 4B)
- Memory-efficient offloading for staged inference

Ported from: coderef/flux2/src/flux2/text_encoder.py

Usage:
    from llm_dit.encoders.qwen3_flux2 import Qwen3Flux2Encoder

    # Create encoder for Klein 9B (uses Qwen3-8B)
    encoder = Qwen3Flux2Encoder.from_pretrained("Qwen/Qwen3-8B-FP8")

    # Encode text
    embeddings = encoder.encode(["A photo of a cat"])  # [1, seq_len, 12288]

    # Offload to free VRAM
    encoder.offload()
"""

import gc
import logging
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


# Default layers to extract for FLUX.2 conditioning
# These provide a mix of early, middle, and late representations
DEFAULT_OUTPUT_LAYERS = [9, 18, 27]

# Default max sequence length
DEFAULT_MAX_LENGTH = 512


class Qwen3Flux2Encoder(nn.Module):
    """
    Multi-layer Qwen3 encoder for FLUX.2 Klein models.

    Extracts hidden states from 3 specific layers and concatenates them,
    providing richer text representations for image generation.

    Output Dimensions:
        - Qwen3-8B: 3 x 4096 = 12288 (for Klein 9B)
        - Qwen3-4B: 3 x 2560 = 7680 (for Klein 4B)

    Args:
        model_spec: HuggingFace model ID or local path
        device: Target device (cuda, cpu)
        max_length: Maximum sequence length (default 512, set None to disable truncation)
        output_layers: Which hidden layers to extract and concatenate (default [9, 18, 27])
        pad_to_max: Whether to pad all sequences to max_length (default True)
    """

    def __init__(
        self,
        model_spec: str,
        device: str | torch.device = "cuda",
        max_length: int = DEFAULT_MAX_LENGTH,
        output_layers: Optional[list[int]] = None,
        pad_to_max: bool = True,
    ):
        super().__init__()

        self.model_spec = model_spec
        self.max_length = max_length
        self.output_layers = output_layers or DEFAULT_OUTPUT_LAYERS.copy()
        self.pad_to_max = pad_to_max
        self._device = torch.device(device)

        # Validate output_layers - must be exactly 3 for Klein models
        if len(self.output_layers) != 3:
            raise ValueError(
                f"output_layers must have exactly 3 layers for Klein models "
                f"(got {len(self.output_layers)}). The transformer expects "
                f"context_dim = 3 * hidden_dim."
            )

        # Load model
        logger.info(f"Loading Qwen3 encoder from {model_spec}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_spec,
            torch_dtype=None,  # Let HF auto-detect (important for FP8)
            device_map=str(device),
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_spec)

        # Get model dimension from config
        self._hidden_dim = self.model.config.hidden_size
        self._num_layers = self.model.config.num_hidden_layers

        # Validate layer indices
        for layer_idx in self.output_layers:
            if layer_idx >= self._num_layers:
                raise ValueError(
                    f"Layer {layer_idx} requested but model only has {self._num_layers} layers"
                )

        logger.info(
            f"Qwen3 encoder loaded: {self._hidden_dim} dim, "
            f"{self._num_layers} layers, "
            f"extracting layers {self.output_layers}, "
            f"output dim = {self.output_dim}"
        )

    @classmethod
    def from_pretrained(
        cls,
        model_spec: str,
        device: str | torch.device = "cuda",
        max_length: int = DEFAULT_MAX_LENGTH,
        output_layers: Optional[list[int]] = None,
        pad_to_max: bool = True,
    ) -> "Qwen3Flux2Encoder":
        """
        Load Qwen3 encoder from pretrained model.

        Args:
            model_spec: HuggingFace model ID (e.g., "Qwen/Qwen3-8B-FP8")
            device: Target device
            max_length: Maximum sequence length
            output_layers: Which hidden layers to extract (default [9, 18, 27])
            pad_to_max: Whether to pad all sequences to max_length

        Returns:
            Initialized encoder
        """
        return cls(
            model_spec=model_spec,
            device=device,
            max_length=max_length,
            output_layers=output_layers,
            pad_to_max=pad_to_max,
        )

    @property
    def output_dim(self) -> int:
        """Output dimension after layer concatenation (num_layers * hidden_dim)."""
        return len(self.output_layers) * self._hidden_dim

    @property
    def hidden_dim(self) -> int:
        """Single-layer hidden dimension."""
        return self._hidden_dim

    @property
    def device(self) -> torch.device:
        """Current device."""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Model dtype."""
        return next(self.model.parameters()).dtype

    @torch.no_grad()
    def forward(self, txt: list[str]) -> torch.Tensor:
        """
        Encode text prompts to multi-layer embeddings.

        Args:
            txt: List of text prompts

        Returns:
            Tensor [batch_size, seq_len, num_layers*hidden_dim]
        """
        all_input_ids = []
        all_attention_masks = []

        for prompt in txt:
            # Apply chat template with thinking disabled
            # This is critical for FLUX.2 - thinking tokens would corrupt the embeddings
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,  # CRITICAL: Disable thinking tokens
            )

            # Tokenize with configurable padding
            padding_strategy = "max_length" if self.pad_to_max else "longest"
            model_inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=padding_strategy,
                truncation=True if self.max_length else False,
                max_length=self.max_length,
            )

            all_input_ids.append(model_inputs["input_ids"])
            all_attention_masks.append(model_inputs["attention_mask"])

        # Batch inputs
        input_ids = torch.cat(all_input_ids, dim=0).to(self.model.device)
        attention_mask = torch.cat(all_attention_masks, dim=0).to(self.model.device)

        # Log sequence info
        logger.info(
            f"[Qwen3Encoder] Encoding {len(txt)} prompts, "
            f"seq_len={input_ids.shape[1]}, "
            f"layers={self.output_layers}, "
            f"pad_to_max={self.pad_to_max}"
        )

        # Forward pass with hidden state extraction
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        # Extract specified layers
        # hidden_states is a tuple of (num_layers + 1) tensors
        # Index 0 is embeddings, indices 1-N are layer outputs
        layers = [output.hidden_states[k] for k in self.output_layers]

        # Stack and reshape: [B, num_layers, L, D] -> [B, L, num_layers*D]
        out = torch.stack(layers, dim=1)
        out = rearrange(out, "b c l d -> b l (c d)")

        logger.info(f"[Qwen3Encoder] Output shape: {list(out.shape)}")

        return out

    def encode(self, txt: list[str]) -> torch.Tensor:
        """
        Encode text prompts (alias for forward).

        Args:
            txt: List of text prompts

        Returns:
            Tensor [batch_size, seq_len, output_dim]
        """
        return self.forward(txt)

    def encode_single(self, prompt: str) -> torch.Tensor:
        """
        Encode a single prompt.

        Args:
            prompt: Single text prompt

        Returns:
            Tensor [1, seq_len, output_dim]
        """
        return self.forward([prompt])

    def offload(self) -> None:
        """
        Offload model to CPU and free GPU memory.

        Call this after encoding to release VRAM for the transformer.
        """
        logger.info("Offloading Qwen3 encoder to CPU...")
        self.model.to("cpu")
        self._device = torch.device("cpu")

        # Force cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Qwen3 encoder offloaded, VRAM freed")

    def to(self, device: str | torch.device) -> "Qwen3Flux2Encoder":
        """Move encoder to device."""
        self.model.to(device)
        self._device = torch.device(device)
        return self


def load_qwen3_flux2_encoder(
    variant: str,
    device: str | torch.device = "cuda",
    max_length: int = DEFAULT_MAX_LENGTH,
    output_layers: Optional[list[int]] = None,
    pad_to_max: bool = True,
) -> Qwen3Flux2Encoder:
    """
    Load Qwen3 encoder for FLUX.2 by variant name.

    Args:
        variant: Model variant ("8B" for Klein 9B, "4B" for Klein 4B)
        device: Target device
        max_length: Maximum sequence length
        output_layers: Which hidden layers to extract (default [9, 18, 27])
        pad_to_max: Whether to pad all sequences to max_length

    Returns:
        Initialized encoder
    """
    variant = variant.upper()
    if variant not in ["8B", "4B"]:
        raise ValueError(f"Unknown variant: {variant}. Use '8B' or '4B'")

    # Use FP8 versions for efficiency
    model_spec = f"Qwen/Qwen3-{variant}-FP8"

    return Qwen3Flux2Encoder.from_pretrained(
        model_spec=model_spec,
        device=device,
        max_length=max_length,
        output_layers=output_layers,
        pad_to_max=pad_to_max,
    )
