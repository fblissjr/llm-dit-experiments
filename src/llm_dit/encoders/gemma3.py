"""
Gemma3 Encoder implementation for LTX-2.

Last Updated: 2026-01-19

Gemma 3-12B is used as the text encoder for LTX-2 video generation.
Architecture based on LTX-2 reference implementation.

Key Architecture:
- 49 decoder layers, each 3840-dimensional hidden states
- Multi-layer feature extraction: Stack -> Normalize -> Concatenate -> Project
- Embeddings1DConnector: 2-layer bidirectional transformer with RoPE
- Output: 3840-dimensional embeddings (DiT projects to 4096/2048 internally)

Pipeline stages:
1. Tokenize text
2. Gemma3 forward -> extract all 49 hidden states
3. Stack and normalize hidden states
4. Feature extractor linear projection (188160 -> 3840)
5. Embeddings1DConnector (2 transformer blocks + learnable registers)
6. Output to DiT
"""

import gc
import json
import logging
import math
from pathlib import Path
from typing import List, Literal, Optional, Union

import torch
from PIL import Image
from torch import nn

from llm_dit.encoders.protocol import (
    EncoderCapability,
    EncoderInfo,
    EncoderType,
    EncodingOutput,
    GenerativeEncoderProtocol,
    VisionLanguageEncoderProtocol,
)
from llm_dit.encoders.embeddings_connector import (
    Embeddings1DConnector,
    RopeType,
    load_connector_weights,
)

logger = logging.getLogger(__name__)

# Default path to LTX-2 model connectors checkpoint
DEFAULT_CONNECTORS_PATH = "models/LTX-2/connectors/diffusion_pytorch_model.safetensors"
DEFAULT_CONNECTORS_CONFIG = "models/LTX-2/connectors/config.json"


class SubLayerExtractor:
    """
    Extract attention and MLP outputs from Gemma3 using forward hooks.

    Last Updated: 2026-01-16

    Gemma3 uses a pre-norm architecture with 4 extraction points per layer:
    1. Attention output (after post_attention_layernorm, before residual)
    2. Post-attention state (after first residual, before MLP)
    3. MLP output (after post_feedforward_layernorm, before residual)
    4. Layer output (after second residual) - this is what output_hidden_states returns

    This extractor captures points 1 and 3, enabling sub-layer routing experiments.

    Important Model Notes:
    - Use full/quantized models (google/gemma-3-12b-it-qat-q4_0-unquantized), NOT distilled
    - Distilled/LoRA models may have different layer behaviors - avoid for experiments
    - Q4 QAT model preserves layer structure while reducing memory (~6GB)

    Memory overhead for full extraction (49 layers):
    - Attention outputs: ~92MB per batch (B=1, T=256, D=3840, L=49, bf16)
    - MLP outputs: ~92MB per batch
    - Total: ~184MB additional (acceptable for RTX 4090's 24GB)

    Usage:
        extractor = SubLayerExtractor(model, layer_indices=[0, 10, 20, 30, 40, 48])
        extractor.register()

        outputs = model(input_ids, attention_mask, output_hidden_states=True)

        sub_layers = extractor.get_stacked_outputs()
        # sub_layers['attention']: [B, T, 3840, num_selected_layers]
        # sub_layers['mlp']: [B, T, 3840, num_selected_layers]

        extractor.unregister()
    """

    def __init__(
        self,
        model,
        layer_indices: Optional[List[int]] = None,
    ):
        """
        Initialize sub-layer extractor.

        Args:
            model: Gemma3 model instance
            layer_indices: Which layers to extract from (default: all 49)
                          Example: [0, 10, 20, 30, 40, 48] for sparse extraction
        """
        self.model = model
        # Get layer count from model
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            num_model_layers = len(model.model.layers)
        else:
            num_model_layers = GEMMA3_NUM_LAYERS

        self.layer_indices = (
            layer_indices if layer_indices is not None else list(range(num_model_layers))
        )

        # Storage for captured outputs
        self.attention_outputs: dict[int, torch.Tensor] = {}
        self.mlp_outputs: dict[int, torch.Tensor] = {}
        self.hooks: List = []

    def _make_attention_hook(self, layer_idx: int):
        """Create hook to capture attention output (after post_attention_layernorm)."""

        def hook(module, input, output):
            # Output of post_attention_layernorm is normalized attention output
            # Clone to avoid modification by subsequent operations
            self.attention_outputs[layer_idx] = output.detach().clone()

        return hook

    def _make_mlp_hook(self, layer_idx: int):
        """Create hook to capture MLP output (after post_feedforward_layernorm)."""

        def hook(module, input, output):
            # Output of post_feedforward_layernorm is normalized MLP output
            self.mlp_outputs[layer_idx] = output.detach().clone()

        return hook

    def register(self):
        """Register hooks on specified layers."""
        # Access layers through model.model.layers (Gemma3 structure)
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers = self.model.model.layers
        else:
            raise RuntimeError("Cannot find decoder layers in model. Expected model.model.layers")

        for idx in self.layer_indices:
            if idx >= len(layers):
                logger.warning(f"Layer index {idx} exceeds model layers ({len(layers)}), skipping")
                continue

            layer = layers[idx]

            # Hook after post_attention_layernorm (captures attention output)
            if hasattr(layer, "post_attention_layernorm"):
                attn_hook = layer.post_attention_layernorm.register_forward_hook(
                    self._make_attention_hook(idx)
                )
                self.hooks.append(attn_hook)
            else:
                logger.warning(f"Layer {idx} missing post_attention_layernorm")

            # Hook after post_feedforward_layernorm (captures MLP output)
            if hasattr(layer, "post_feedforward_layernorm"):
                mlp_hook = layer.post_feedforward_layernorm.register_forward_hook(
                    self._make_mlp_hook(idx)
                )
                self.hooks.append(mlp_hook)
            else:
                logger.warning(f"Layer {idx} missing post_feedforward_layernorm")

        logger.debug(
            f"Registered {len(self.hooks)} sub-layer hooks for layers {self.layer_indices}"
        )

    def unregister(self):
        """Remove all hooks and clear stored outputs."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.attention_outputs.clear()
        self.mlp_outputs.clear()

    def get_stacked_outputs(self) -> dict:
        """
        Stack captured outputs into tensors.

        Returns:
            Dict with:
            - 'attention': [B, T, D, L] - Attention outputs (after layernorm, before residual)
            - 'mlp': [B, T, D, L] - MLP outputs (after layernorm, before residual)
            - 'layer_indices': List[int] - Which layers were extracted
        """
        if not self.attention_outputs:
            raise RuntimeError(
                "No outputs captured. Did you call register() and run a forward pass?"
            )

        # Stack in sorted order
        sorted_indices = sorted(self.attention_outputs.keys())

        attention_stack = torch.stack(
            [self.attention_outputs[i] for i in sorted_indices], dim=-1
        )  # [B, T, 3840, L]

        mlp_stack = torch.stack(
            [self.mlp_outputs[i] for i in sorted_indices], dim=-1
        )  # [B, T, 3840, L]

        return {
            "attention": attention_stack,
            "mlp": mlp_stack,
            "layer_indices": sorted_indices,
        }

    def __enter__(self):
        """Context manager entry - register hooks."""
        self.register()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - unregister hooks."""
        self.unregister()
        return False


# LTX-2 Architecture Constants
GEMMA3_HIDDEN_DIM = 3840  # Hidden dimension per layer
GEMMA3_NUM_LAYERS = 49  # Number of decoder layers to aggregate
GEMMA3_FEATURE_DIM = GEMMA3_HIDDEN_DIM * GEMMA3_NUM_LAYERS  # 188,160
GEMMA3_OUTPUT_DIM = 3840  # Final output dimension (DiT projects further)

# Layer masking modes for ablation experiments
LayerMaskingMode = Literal["soft", "zero", "weighted"]


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


def pack_text_embeds(
    hidden_states: torch.Tensor,
    sequence_length: int,
    device: torch.device,
    scale_factor: float = 8.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Pack text hidden states into prompt embeddings for LTX2Pipeline.

    This matches the LTX2Pipeline._pack_text_embeds method in diffusers,
    allowing pre-computed embeddings to be passed directly to the pipeline.

    Args:
        hidden_states: [B, T, hidden_dim, num_layers] - stacked hidden states
                      from all Gemma layers (e.g., shape [1, 256, 3840, 49])
        sequence_length: Actual sequence length (excluding padding)
        device: Target device for output
        scale_factor: Scale factor for normalization (default 8.0, matches LTX-2)
        eps: Epsilon for numerical stability

    Returns:
        Packed prompt embeddings [B, T, hidden_dim * num_layers] ready for the pipeline.
        Shape example: [1, 256, 188160] for 49 layers of 3840 hidden dim.

    Example:
        >>> text_encoder, tokenizer = load_text_encoder_8bit("models/LTX-2")
        >>> outputs = text_encoder(input_ids, output_hidden_states=True)
        >>> hidden_states = torch.stack(outputs.hidden_states[:49], dim=-1)
        >>> packed = pack_text_embeds(hidden_states, seq_len, torch.device("cuda"))
        >>> # Now pass packed to pipeline as prompt_embeds
    """
    # Move to target device
    hidden_states = hidden_states.to(device)

    # Normalize each layer: divide by L2 norm
    # Shape: [B, T, D, L] -> norm over D dimension
    normed = hidden_states / (hidden_states.norm(dim=2, keepdim=True) + eps)

    # Flatten layers: [B, T, D, L] -> [B, T, D * L]
    batch_size, seq_len, hidden_dim, num_layers = normed.shape
    packed = normed.view(batch_size, seq_len, hidden_dim * num_layers)

    # Apply scale factor
    packed = packed * scale_factor

    return packed


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
        max_memory: Optional[dict] = None,
        connectors_path: Optional[str] = None,
        use_connector: bool = True,
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
            max_memory: Memory limits per device for CPU offloading.
                       Example: {0: "18GiB", "cpu": "32GiB"} limits GPU 0 to 18GB.
            connectors_path: Path to connectors checkpoint (safetensors).
                            Defaults to models/LTX-2/connectors/.
            use_connector: Whether to use the Embeddings1DConnector (default True).
                          Set False for debugging feature extractor only.
        """
        self._model_id = model_id
        self._device_str = device
        self._dtype = dtype
        self._max_sequence_length = max_sequence_length
        self._load_in_4bit = load_in_4bit
        self._load_in_8bit = load_in_8bit
        self._max_memory = max_memory
        self._connectors_path = connectors_path or DEFAULT_CONNECTORS_PATH
        self._use_connector = use_connector

        # Model components (lazy loaded)
        self._model = None
        self._tokenizer = None
        self._feature_extractor = None
        self._embeddings_connector: Optional[Embeddings1DConnector] = None
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
        max_memory: Optional[dict] = None,
        connectors_path: Optional[str] = None,
        use_connector: bool = True,
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
            max_memory: Memory limits for CPU offloading. Example: {0: "18GiB", "cpu": "32GiB"}.
            connectors_path: Path to connectors safetensors checkpoint.
            use_connector: Whether to use Embeddings1DConnector (default True).
            **kwargs: Additional arguments.

        Returns:
            Initialized Gemma3Encoder.
        """
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        dtype_torch = dtype_map.get(dtype, torch.bfloat16)

        encoder = cls(
            model_id=model_path,
            device=device,
            dtype=dtype_torch,
            max_sequence_length=max_sequence_length,
            load_in_4bit=(quantization == "4bit"),
            load_in_8bit=(quantization == "8bit"),
            max_memory=max_memory,
            connectors_path=connectors_path,
            use_connector=use_connector,
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
            from transformers import AutoModelForCausalLM, AutoTokenizer

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
            dtype=self._dtype,
            device_map=device_map,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            max_memory=self._max_memory,
        )

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_id,
            model_max_length=self._max_sequence_length,
        )
        self._tokenizer.padding_side = "left"  # Gemma prefers left padding
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Initialize and load feature extractor from checkpoint
        self._feature_extractor = FeatureExtractorLinear(dtype=self._dtype)
        self._load_connector_weights()

        if self._device_str not in ("cpu", "auto"):
            self._feature_extractor = self._feature_extractor.to(
                device=torch.device(self._device_str)
            )
            if self._embeddings_connector is not None:
                self._embeddings_connector = self._embeddings_connector.to(
                    device=torch.device(self._device_str),
                    dtype=self._dtype,  # Match encoder dtype (bfloat16)
                )

        # Set model to mode without gradients
        self._model.requires_grad_(False)

        self._is_loaded = True
        self._is_offloaded = False
        logger.info(f"Gemma 3 encoder loaded: {self._model.device}")

    def _load_connector_weights(self) -> None:
        """
        Load feature extractor and embeddings connector weights from checkpoint.

        The checkpoint (connectors/diffusion_pytorch_model.safetensors) contains:
        - text_proj_in.weight: [3840, 188160] - feature extractor linear
        - video_connector.*: embeddings connector weights
        - audio_connector.*: audio connector weights (unused for video-only)
        """
        from safetensors import safe_open

        connectors_path = Path(self._connectors_path)
        if not connectors_path.exists():
            logger.warning(
                f"Connectors checkpoint not found at {connectors_path}. "
                "Feature extractor will have random weights (BROKEN OUTPUT)."
            )
            return

        logger.info(f"Loading connector weights from {connectors_path}")

        # Load weights from safetensors
        with safe_open(connectors_path, framework="pt") as f:
            # 1. Load feature extractor weight
            if "text_proj_in.weight" in f.keys():
                fe_weight = f.get_tensor("text_proj_in.weight")
                # Feature extractor: [3840, 188160] -> [3840, 188160]
                self._feature_extractor.aggregate_embed.weight.data = fe_weight.to(
                    dtype=self._dtype
                )
                logger.info(
                    f"Loaded feature extractor weight: {fe_weight.shape}, "
                    f"mean={fe_weight.float().mean():.4f}, std={fe_weight.float().std():.4f}"
                )
            else:
                logger.error(
                    "text_proj_in.weight not found in checkpoint! "
                    "Feature extractor will have random weights."
                )

            # 2. Create and load embeddings connector
            if self._use_connector:
                # Load config for connector parameters
                config_path = connectors_path.parent / "config.json"
                if config_path.exists():
                    with open(config_path) as cfg_file:
                        config = json.load(cfg_file)
                else:
                    # Default config matching LTX-2
                    config = {
                        "video_connector_attention_head_dim": 128,
                        "video_connector_num_attention_heads": 30,
                        "video_connector_num_layers": 2,
                        "video_connector_num_learnable_registers": 128,
                        "rope_type": "split",
                        "rope_theta": 10000.0,
                        "rope_double_precision": True,
                        "connector_rope_base_seq_len": 4096,
                    }

                # Create connector from config
                self._embeddings_connector = Embeddings1DConnector.from_config(config)

                # Load connector weights
                load_connector_weights(
                    self._embeddings_connector,
                    connectors_path,
                    prefix="video_connector.",
                )
                logger.info("Loaded embeddings connector weights")

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
        # Handle device_map="auto" case - model may be spread across devices
        if hasattr(self._model, "device"):
            dev = self._model.device
            if dev is not None:
                return dev
        # For auto device_map with accelerate, get device from first parameter
        if hasattr(self._model, "hf_device_map"):
            # Model is spread across devices, use first available GPU
            try:
                first_param = next(self._model.parameters())
                return first_param.device
            except StopIteration:
                pass
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
        # Note: outputs.hidden_states includes embedding layer (index 0) + decoder layers
        # LTX-2 uses ALL hidden states including embedding layer: 1 + 48 = 49 layers
        hidden_states = outputs.hidden_states  # Include embedding layer

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

        # Ensure feature extractor is on same device as data
        if feature_extractor.aggregate_embed.weight.device != normalized.device:
            feature_extractor = feature_extractor.to(normalized.device)

        # Project to output dimension
        embeddings = feature_extractor(normalized)

        # Run through embeddings connector (2-layer bidirectional transformer)
        if self._embeddings_connector is not None:
            # Convert attention mask to additive format for connector
            # Input mask: [B, T] with 1=valid, 0=padding
            # Connector expects: [B, 1, 1, T] with 0=valid, -10000=padding
            additive_mask = (1.0 - attention_mask.float()) * -10000.0
            additive_mask = additive_mask[:, None, None, :].to(embeddings.dtype)  # Match embedding dtype

            # Ensure connector is on same device
            if next(self._embeddings_connector.parameters()).device != embeddings.device:
                self._embeddings_connector = self._embeddings_connector.to(embeddings.device)

            # Process through connector
            embeddings, _ = self._embeddings_connector(embeddings, additive_mask)
            logger.debug(
                f"Connector output: shape={embeddings.shape}, "
                f"mean={embeddings.float().mean():.4f}, std={embeddings.float().std():.4f}"
            )

        # Apply attention mask (note: after connector, mask may have changed)
        embeddings = embeddings * attention_mask[:, :, None].to(embeddings.dtype)

        # Get sequence lengths for unpadding
        seq_lengths = attention_mask.sum(dim=1).tolist()
        batch_size = len(texts)

        # Build per-sample outputs (EncodingOutput expects List[Tensor])
        embedding_list = [embeddings[i, : int(seq_lengths[i])] for i in range(batch_size)]
        mask_list = [attention_mask[i, : int(seq_lengths[i])].bool() for i in range(batch_size)]

        return EncodingOutput(
            embeddings=embedding_list,
            attention_masks=mask_list,
            padded_embeddings=embeddings if return_padded else None,
            padded_mask=attention_mask if return_padded else None,
            token_counts=[int(s) for s in seq_lengths],
        )

    def encode_multilayer(
        self,
        texts: Union[str, List[str]],
        layer_indices: Optional[List[int]] = None,
        return_projected: bool = True,
        extract_sub_layers: bool = False,
    ) -> dict:
        """
        Encode text and return multi-layer hidden states for routing experiments.

        This method exposes the full layer stack before projection, enabling
        per-token layer routing experiments for LTX-2 research.

        Args:
            texts: Input text(s) to encode.
            layer_indices: Which layers to extract (default: all 49).
                          Example: [10, 20, 30, 40, 48] for 5-layer routing.
            return_projected: Also return the projected embeddings.
            extract_sub_layers: If True, also extract attention/MLP outputs
                               separately via SubLayerExtractor hooks.
                               Adds ~184MB overhead for full extraction.

        Returns:
            Dict with:
            - 'layer_stack': [B, T, 3840, num_layers] - post-MLP layer outputs
            - 'attention_mask': [B, T] - valid token mask
            - 'projected': [B, T, 3840] - after feature extractor (if requested)
            - 'seq_lengths': List[int] - valid sequence lengths
            - 'attention_stack': [B, T, 3840, L] - attention outputs (if extract_sub_layers)
            - 'mlp_stack': [B, T, 3840, L] - MLP outputs (if extract_sub_layers)

        Note:
            For routing experiments, prefer using full/quantized Gemma3 models
            (google/gemma-3-12b-it-qat-q4_0-unquantized), NOT distilled variants.
            Distilled models may have compressed intermediate representations that
            don't represent true layer specialization.
        """
        if not self._is_loaded:
            self._load_model()

        if self._is_offloaded:
            self._model.to(self.device)
            self._is_offloaded = False

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

        # Set up sub-layer extractor if requested
        extractor = None
        if extract_sub_layers:
            extractor = SubLayerExtractor(self._model, layer_indices)
            extractor.register()

        try:
            # Forward pass with all hidden states
            with torch.no_grad():
                outputs = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )

            # Stack hidden states: [B, T, D, L]
            # LTX-2 uses ALL hidden states including embedding layer: 1 + 48 = 49 layers
            hidden_states = outputs.hidden_states  # Include embedding layer
            num_layers = min(len(hidden_states), GEMMA3_NUM_LAYERS)
            hidden_states = hidden_states[:num_layers]
            stacked = torch.stack(hidden_states, dim=-1)  # [B, T, 3840, 49]

            # Select specific layers if requested
            if layer_indices is not None:
                stacked = stacked[..., layer_indices]

            # Compute projection if requested
            projected = None
            if return_projected:
                normalized = _norm_and_concat_layers(stacked, attention_mask)
                # Always compute actual feature dimension from the stacked tensor
                actual_feature_dim = stacked.shape[2] * stacked.shape[3]

                # Use adjusted feature extractor if dimensions don't match default
                if actual_feature_dim != GEMMA3_FEATURE_DIM:
                    fe = FeatureExtractorLinear(
                        input_dim=actual_feature_dim,
                        output_dim=GEMMA3_OUTPUT_DIM,
                        dtype=self._dtype,
                    ).to(normalized.device)
                else:
                    fe = self._feature_extractor
                    # Ensure feature extractor is on same device as data
                    if fe.aggregate_embed.weight.device != normalized.device:
                        fe = fe.to(normalized.device)
                projected = fe(normalized)
                projected = projected * attention_mask[:, :, None].to(projected.dtype)

            seq_lengths = attention_mask.sum(dim=1).tolist()

            result = {
                "layer_stack": stacked,
                "attention_mask": attention_mask,
                "projected": projected,
                "seq_lengths": [int(s) for s in seq_lengths],
            }

            # Add sub-layer outputs if requested
            if extract_sub_layers and extractor is not None:
                sub_outputs = extractor.get_stacked_outputs()
                result["attention_stack"] = sub_outputs["attention"]
                result["mlp_stack"] = sub_outputs["mlp"]
                result["sublayer_indices"] = sub_outputs["layer_indices"]

            return result

        finally:
            # Always clean up hooks
            if extractor is not None:
                extractor.unregister()

    def encode_with_layer_masking(
        self,
        texts: Union[str, List[str]],
        active_layers: List[int],
        masking_mode: LayerMaskingMode = "soft",
        return_packed: bool = True,
    ) -> dict:
        """
        Encode text with layer masking for ablation experiments.

        This masks inactive layers according to the specified mode, allowing
        you to see what a single layer (or subset) contributes in isolation.
        Useful for understanding layer specialization in Gemma3 for LTX-2.

        Args:
            texts: Input text(s) to encode.
            active_layers: List of layer indices to keep active (0-48).
                          Other layers will be masked according to masking_mode.
            masking_mode: How to handle inactive layers:
                - "soft": Replace with per-layer mean (maintains distribution)
                - "zero": Zero out (creates OOD inputs - NOT RECOMMENDED)
                - "weighted": Scale active layers to preserve total norm
            return_packed: If True, return packed embeddings ready for pipeline.
                          If False, return raw hidden states.

        Returns:
            Dict with:
            - 'prompt_embeds': Packed embeddings for pipeline (if return_packed)
            - 'hidden_states': Raw masked hidden states [B, T, D, L]
            - 'attention_mask': Attention mask [B, T]
            - 'seq_lengths': List of sequence lengths

        Example:
            # Test layer 47 in isolation
            result = encoder.encode_with_layer_masking(
                "A cat sleeping on a couch",
                active_layers=[47],
                masking_mode="soft"
            )
            output = pipe(prompt_embeds=result['prompt_embeds'], ...)

        Note:
            - "soft" mode is recommended as it maintains the expected input
              distribution for the projection layer
            - "zero" mode creates out-of-distribution inputs and may produce
              unexpected results
            - "weighted" scales active layers to preserve total signal magnitude
        """
        if not self._is_loaded:
            self._load_model()

        if self._is_offloaded:
            self._model.to(self.device)
            self._is_offloaded = False

        if isinstance(texts, str):
            texts = [texts]

        active_set = set(active_layers)

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

        # Stack hidden states: [B, T, D, L]
        # LTX-2 uses ALL hidden states including embedding layer (index 0)
        # This gives 49 layers: embedding layer + 48 decoder layers = 188160 dims
        hidden_states_list = outputs.hidden_states[:GEMMA3_NUM_LAYERS]
        hidden_states = torch.stack(hidden_states_list, dim=-1)
        num_layers = hidden_states.shape[-1]  # Actual number of layers stacked

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

        seq_lengths = attention_mask.sum(dim=1).tolist()

        result = {
            "hidden_states": hidden_states,
            "attention_mask": attention_mask,
            "seq_lengths": [int(s) for s in seq_lengths],
            "active_layers": active_layers,
            "masking_mode": masking_mode,
        }

        # Pack for pipeline if requested
        # Use _norm_and_concat_layers which matches diffusers' normalization:
        # normalized = 8 * (x - mean) / (max - min)
        # This preserves layer masking information, unlike L2 normalization
        if return_packed:
            packed = _norm_and_concat_layers(
                hidden_states,
                attention_mask,
                padding_side=self._tokenizer.padding_side,
            )
            result["prompt_embeds"] = packed

        return result

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
        generated_ids = outputs[0, input_ids.shape[1] :]
        generated_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

        return generated_text.strip()

    def offload(self) -> None:
        """Offload model to CPU and free GPU memory."""
        if self._model is not None:
            self._model.to("cpu")
        if self._feature_extractor is not None:
            self._feature_extractor.to("cpu")
        if self._embeddings_connector is not None:
            self._embeddings_connector.to("cpu")
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
        if self._embeddings_connector is not None:
            self._embeddings_connector.to(device)
        self._is_offloaded = device.type == "cpu"
        return self


# Convenience alias
LTX2Encoder = Gemma3Encoder
