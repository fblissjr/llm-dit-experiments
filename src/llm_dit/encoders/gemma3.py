"""
Gemma3 Encoder implementation for LTX-2.

Last Updated: 2026-01-09

Gemma 3-12B is used as the text encoder for LTX-2 video generation.
Architecture based on LTX-2 reference implementation.

Key Architecture:
- 49 decoder layers, each 3840-dimensional hidden states
- Multi-layer feature extraction: Stack -> Normalize -> Concatenate -> Project
- Output: 3840-dimensional embeddings (DiT projects to 4096/2048 internally)

Note: The 4096 (video) and 2048 (audio) dimensions are applied by the DiT's
internal projection layers, not by this encoder. We output the raw 3840-dim
features from the text encoder.
"""

import gc
import logging
import math
from typing import List, Optional, Union

import torch
from torch import nn
from PIL import Image

from llm_dit.encoders.protocol import (
    EncoderCapability,
    EncoderInfo,
    EncoderType,
    EncodingOutput,
    GenerativeEncoderProtocol,
    VisionLanguageEncoderProtocol,
)

logger = logging.getLogger(__name__)


# LTX-2 Architecture Constants
GEMMA3_HIDDEN_DIM = 3840  # Hidden dimension per layer
GEMMA3_NUM_LAYERS = 49  # Number of decoder layers to aggregate
GEMMA3_FEATURE_DIM = GEMMA3_HIDDEN_DIM * GEMMA3_NUM_LAYERS  # 188,160
GEMMA3_OUTPUT_DIM = 3840  # Final output dimension (DiT projects further)


def _norm_and_concat_layers(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    padding_side: str = "left",
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Normalize and concatenate multi-layer hidden states.

    LTX-2 applies per-layer normalization:
    normalized = 8 * (x - mean) / (range + eps)

    Args:
        hidden_states: [B, T, D, L] - stacked hidden states from all layers
        attention_mask: [B, T] - attention mask (1 for valid, 0 for padding)
        padding_side: "left" or "right" padding
        eps: Epsilon for numerical stability

    Returns:
        [B, T, D*L] - Normalized and flattened features
    """
    b, t, d, num_layers = hidden_states.shape
    device = hidden_states.device
    dtype = hidden_states.dtype

    # Build mask: [B, T, 1, 1] for broadcasting
    mask = attention_mask.bool()
    mask_expanded = mask[:, :, None, None]  # [B, T, 1, 1]

    # Get sequence lengths for mean calculation
    seq_lengths = attention_mask.sum(dim=1)  # [B]

    # Per-layer normalization
    # Mask invalid tokens for statistics
    masked_states = hidden_states.masked_fill(~mask_expanded, 0.0)

    # Mean per layer (over valid tokens only)
    denom = (seq_lengths * d).view(b, 1, 1, 1).clamp(min=eps)
    mean = masked_states.sum(dim=(1, 2), keepdim=True) / denom

    # Range per layer (over valid tokens only)
    x_min = hidden_states.masked_fill(~mask_expanded, float("inf")).amin(dim=(1, 2), keepdim=True)
    x_max = hidden_states.masked_fill(~mask_expanded, float("-inf")).amax(dim=(1, 2), keepdim=True)
    range_val = (x_max - x_min).clamp(min=eps)

    # Normalize: 8 * (x - mean) / range
    normed = 8.0 * (hidden_states - mean) / range_val

    # Flatten layers: [B, T, D, L] -> [B, T, D*L]
    normed = normed.reshape(b, t, -1)

    # Zero out padding positions
    mask_flat = mask[:, :, None].expand(-1, -1, d * num_layers)
    normed = normed.masked_fill(~mask_flat, 0.0)

    return normed.to(dtype)


class FeatureExtractorLinear(nn.Module):
    """
    Linear projection that aggregates multi-layer features.

    Input: [B, T, 3840 * 49] = [B, T, 188160]
    Output: [B, T, 3840]
    """

    def __init__(
        self,
        input_dim: int = GEMMA3_FEATURE_DIM,
        output_dim: int = GEMMA3_OUTPUT_DIM,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.aggregate_embed = nn.Linear(input_dim, output_dim, bias=False)
        self.aggregate_embed = self.aggregate_embed.to(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.aggregate_embed(x)


class Gemma3Encoder:
    """
    Gemma3 vision-language encoder for LTX-2.

    Implements:
    - TextEncoderProtocol: Text encoding
    - VisionLanguageEncoderProtocol: Vision-language encoding
    - GenerativeEncoderProtocol: Text generation (for captioning)

    LTX-2 Architecture:
    1. Gemma 3 backbone: text -> hidden states from all 49 layers
    2. Multi-layer extraction: Stack -> Normalize -> Concatenate
    3. Feature extractor: Linear(188160 -> 3840)
    4. Output: [B, T, 3840] - DiT projects to 4096/2048 internally

    Memory Strategy (RTX 4090):
    - Q4 QAT model: ~6GB VRAM
    - Offload after encoding to free memory for transformer
    - Sequential loading pattern for 24GB constraint
    """

    def __init__(
        self,
        model_id: str = "google/gemma-3-12b-it-qat-q4_0-unquantized",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        max_sequence_length: int = 256,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
    ):
        """
        Initialize Gemma3 encoder.

        Args:
            model_id: Gemma 3 model ID or path.
            device: Device to load on ("cuda", "cpu", "auto").
            dtype: Model dtype (typically bfloat16).
            max_sequence_length: Maximum sequence length for encoding.
            load_in_4bit: Apply additional 4-bit quantization (on top of QAT).
            load_in_8bit: Apply additional 8-bit quantization.
        """
        self._model_id = model_id
        self._device_str = device
        self._dtype = dtype
        self._max_sequence_length = max_sequence_length
        self._load_in_4bit = load_in_4bit
        self._load_in_8bit = load_in_8bit

        # Model components (lazy loaded)
        self._model = None
        self._tokenizer = None
        self._feature_extractor = None
        self._is_loaded = False
        self._is_offloaded = False

    @classmethod
    def from_pretrained(
        cls,
        model_path: str = "google/gemma-3-12b-it-qat-q4_0-unquantized",
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_sequence_length: int = 256,
        quantization: Optional[str] = None,
        **kwargs,
    ) -> "Gemma3Encoder":
        """
        Load Gemma3 encoder from pretrained model.

        Args:
            model_path: Path to model or HuggingFace ID.
            device: Device to load on.
            dtype: Model dtype as string.
            max_sequence_length: Max sequence length.
            quantization: Additional quantization ("4bit", "8bit", or None).
            **kwargs: Additional arguments.

        Returns:
            Initialized Gemma3Encoder.
        """
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(dtype, torch.bfloat16)

        encoder = cls(
            model_id=model_path,
            device=device,
            dtype=torch_dtype,
            max_sequence_length=max_sequence_length,
            load_in_4bit=(quantization == "4bit"),
            load_in_8bit=(quantization == "8bit"),
        )

        # Load model immediately
        encoder._load_model()

        return encoder

    def _load_model(self) -> None:
        """Load Gemma 3 model and tokenizer."""
        if self._is_loaded:
            return

        try:
            from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
        except ImportError:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            # Fallback for older transformers without Gemma3ForConditionalGeneration
            Gemma3ForConditionalGeneration = AutoModelForCausalLM

        logger.info(f"Loading Gemma 3 encoder from {self._model_id}")

        # Build quantization config if needed
        quantization_config = None
        if self._load_in_4bit or self._load_in_8bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=self._load_in_4bit,
                load_in_8bit=self._load_in_8bit,
                bnb_4bit_compute_dtype=self._dtype,
            )

        # Determine device map
        device_map = None
        if self._device_str == "auto":
            device_map = "auto"
        elif self._device_str != "cpu":
            device_map = {"": self._device_str}

        # Load model
        self._model = Gemma3ForConditionalGeneration.from_pretrained(
            self._model_id,
            torch_dtype=self._dtype,
            device_map=device_map,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
        )

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_id,
            model_max_length=self._max_sequence_length,
        )
        self._tokenizer.padding_side = "left"  # Gemma prefers left padding
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Initialize feature extractor projection
        self._feature_extractor = FeatureExtractorLinear(dtype=self._dtype)
        if self._device_str != "cpu":
            self._feature_extractor = self._feature_extractor.to(
                device=torch.device(self._device_str if self._device_str != "auto" else "cuda")
            )

        # Set model to mode without gradients
        self._model.requires_grad_(False)

        self._is_loaded = True
        self._is_offloaded = False
        logger.info(f"Gemma 3 encoder loaded: {self._model.device}")

    @property
    def info(self) -> EncoderInfo:
        """Get encoder information and capabilities."""
        capabilities = {
            EncoderCapability.TEXT_ENCODING,
            EncoderCapability.VISION_ENCODING,
            EncoderCapability.TEXT_GENERATION,
            EncoderCapability.HIDDEN_LAYER_SELECTION,
        }

        return EncoderInfo(
            encoder_type=EncoderType.GEMMA3,
            model_id=self._model_id,
            hidden_dim=self.embedding_dim,
            max_sequence_length=self._max_sequence_length,
            capabilities=capabilities,
            quantization="q4_0" if not self._load_in_4bit else "q4_bnb",
            device=self.device,
            dtype=self._dtype,
        )

    @property
    def embedding_dim(self) -> int:
        """
        Return embedding dimension.

        Gemma3 encoder outputs 3840-dimensional features.
        The DiT applies its own projection to 4096 (video) or 2048 (audio).
        """
        return GEMMA3_OUTPUT_DIM

    @property
    def max_sequence_length(self) -> int:
        """Return max sequence length."""
        return self._max_sequence_length

    @property
    def device(self) -> torch.device:
        """Return model device."""
        if self._model is None:
            return torch.device("cpu")
        # Handle device_map="auto" case
        if hasattr(self._model, "device"):
            return self._model.device
        return torch.device(self._device_str if self._device_str != "auto" else "cuda")

    @property
    def dtype(self) -> torch.dtype:
        """Return model dtype."""
        return self._dtype

    def encode(
        self,
        texts: Union[str, List[str]],
        images: Optional[List[Image.Image]] = None,
        return_padded: bool = False,
        layer_index: int = -1,  # Use all layers (ignored for Gemma3)
    ) -> EncodingOutput:
        """
        Encode text to embeddings using multi-layer feature extraction.

        LTX-2 encoding pipeline:
        1. Tokenize text
        2. Forward pass with output_hidden_states=True
        3. Stack all decoder layer hidden states
        4. Normalize per layer and concatenate
        5. Project via feature extractor linear

        Args:
            texts: Input text(s) to encode.
            images: Optional images (not yet implemented for I2V).
            return_padded: If True, return padded sequences.
            layer_index: Ignored (Gemma3 uses all layers).

        Returns:
            EncodingOutput with embeddings [B, T, 3840].
        """
        if not self._is_loaded:
            self._load_model()

        if self._is_offloaded:
            self._model.to(self.device)
            self._is_offloaded = False

        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]

        # Tokenize
        encoded = self._tokenizer(
            texts,
            padding="max_length",
            max_length=self._max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded.input_ids.to(self.device)
        attention_mask = encoded.attention_mask.to(self.device)

        # Forward pass with all hidden states
        with torch.no_grad():
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        # Stack hidden states: tuple of [B, T, D] -> [B, T, D, L]
        # Note: outputs.hidden_states includes embedding layer + decoder layers
        # Skip embedding layer (index 0), use decoder layers only
        hidden_states = outputs.hidden_states[1:]  # Skip embedding layer

        # Limit to expected number of layers (model may have different count)
        num_layers = min(len(hidden_states), GEMMA3_NUM_LAYERS)
        hidden_states = hidden_states[:num_layers]

        # Stack: [B, T, D, L]
        stacked = torch.stack(hidden_states, dim=-1)

        # Normalize and concatenate
        # Adjust feature extractor if needed for different layer count
        actual_feature_dim = stacked.shape[2] * num_layers
        if actual_feature_dim != GEMMA3_FEATURE_DIM:
            # Create adjusted feature extractor for this model
            logger.warning(
                f"Gemma3 layer count differs from expected: {num_layers} vs {GEMMA3_NUM_LAYERS}. "
                f"Feature dim: {actual_feature_dim} vs {GEMMA3_FEATURE_DIM}"
            )
            feature_extractor = FeatureExtractorLinear(
                input_dim=actual_feature_dim,
                output_dim=GEMMA3_OUTPUT_DIM,
                dtype=self._dtype,
            ).to(self.device)
        else:
            feature_extractor = self._feature_extractor

        normalized = _norm_and_concat_layers(stacked, attention_mask)

        # Project to output dimension
        embeddings = feature_extractor(normalized)

        # Apply attention mask
        embeddings = embeddings * attention_mask[:, :, None].to(embeddings.dtype)

        return EncodingOutput(
            embeddings=embeddings,
            attention_mask=attention_mask,
            pooled_output=None,  # Gemma3 doesn't use pooled output
            hidden_states=outputs.hidden_states if return_padded else None,
        )

    def encode_image(
        self,
        images: List[Image.Image],
        return_padded: bool = False,
    ) -> EncodingOutput:
        """
        Encode images only (for image-to-video).

        TODO: Implement image encoding for I2V mode.
        """
        raise NotImplementedError(
            "Gemma3Encoder.encode_image() not yet implemented. "
            "Image-to-video encoding requires vision encoder integration."
        )

    def get_image_tokens(self, image: Image.Image) -> int:
        """
        Get the number of tokens an image will consume.

        Based on Gemma3's vision encoder config.
        """
        # Gemma3's vision encoder typically uses ~576 tokens per image
        return 576

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        **kwargs,
    ) -> str:
        """
        Generate text using Gemma3.

        Useful for auto-captioning or prompt enhancement.

        Args:
            prompt: Input prompt.
            system_prompt: Optional system prompt.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            **kwargs: Additional generation arguments.

        Returns:
            Generated text.
        """
        if not self._is_loaded:
            self._load_model()

        if self._is_offloaded:
            self._model.to(self.device)
            self._is_offloaded = False

        # Build full prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt

        # Tokenize
        encoded = self._tokenizer(
            full_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_sequence_length - max_new_tokens,
        )
        input_ids = encoded.input_ids.to(self.device)
        attention_mask = encoded.attention_mask.to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                top_p=top_p,
                do_sample=(temperature > 0),
                pad_token_id=self._tokenizer.pad_token_id,
                **kwargs,
            )

        # Decode only new tokens
        generated_ids = outputs[0, input_ids.shape[1]:]
        generated_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

        return generated_text.strip()

    def offload(self) -> None:
        """Offload model to CPU and free GPU memory."""
        if self._model is not None:
            self._model.to("cpu")
        if self._feature_extractor is not None:
            self._feature_extractor.to("cpu")
        self._is_offloaded = True

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Gemma3 encoder offloaded to CPU")

    def to(self, device: torch.device) -> "Gemma3Encoder":
        """Move model to device."""
        if self._model is not None:
            self._model.to(device)
        if self._feature_extractor is not None:
            self._feature_extractor.to(device)
        self._is_offloaded = (device.type == "cpu")
        return self


# Convenience alias
LTX2Encoder = Gemma3Encoder
