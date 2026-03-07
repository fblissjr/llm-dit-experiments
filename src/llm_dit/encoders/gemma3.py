"""
Gemma3 Encoder implementation for LTX-2.

Last Updated: 2026-03-02

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

import json
import logging
import math
from pathlib import Path
from typing import List, Literal, Optional, Union

import torch
from PIL import Image
from torch import nn

from llm_dit.encoders.embeddings_connector import (
    Embeddings1DConnector,
    RopeType,
    load_connector_weights,
)
from llm_dit.encoders.protocol import (
    EncoderCapability,
    EncoderInfo,
    EncoderType,
    EncodingOutput,
    GenerativeEncoderProtocol,
    VisionLanguageEncoderProtocol,
)
from llm_dit.encoders.gemma3_feature_extractor_v2 import FeatureExtractorV2
from llm_dit.utils.shuttle import PinnedShuttleMixin

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt enhancement utilities
# ---------------------------------------------------------------------------

# Unicode smart-quote replacements that Gemma tends to insert.
# Matches the official LTX-2 clean_response() in ltx_pipelines/utils/helpers.py.
_UNICODE_REPLACEMENTS = str.maketrans(
    "\u2018\u2019\u201c\u201d\u2014\u2013\u00a0\u2032\u2212",
    "''\"\"-- '-",
)

# System prompt for T2V prompt enhancement.
# Source: official LTX-2 gemma_t2v_system_prompt.txt
LTX2_T2V_SYSTEM_PROMPT = """\
You are a Creative Assistant. Given a user's raw input prompt describing a scene or concept, \
expand it into a detailed video generation prompt with specific visuals and integrated audio \
to guide a text-to-video model.

#### Guidelines
- Strictly follow all aspects of the user's raw input: include every element requested \
(style, visuals, motions, actions, camera movement, audio).
    - If the input is vague, invent concrete details: lighting, textures, materials, scene \
settings, etc.
        - For characters: describe gender, clothing, hair, expressions. DO NOT invent \
unrequested characters.
- Use active language: present-progressive verbs ("is walking," "speaking"). If no action \
specified, describe natural movements.
- Maintain chronological flow: use temporal connectors ("as," "then," "while").
- Audio layer: Describe complete soundscape (background audio, ambient sounds, SFX, \
speech/music when requested). Integrate sounds chronologically alongside actions. Be specific \
(e.g., "soft footsteps on tile"), not vague (e.g., "ambient sound is present").
- Speech (only when requested):
    - For ANY speech-related input (talking, conversation, singing, etc.), ALWAYS include \
exact words in quotes with voice characteristics (e.g., "The man says in an excited voice: \
'You won't believe what I just saw!'").
    - Specify language if not English and accent if relevant.
- Style: Include visual style at the beginning: "Style: <style>, <rest of prompt>." \
Default to cinematic-realistic if unspecified. Omit if unclear.
- Visual and audio only: NO non-visual/auditory senses (smell, taste, touch).
- Restrained language: Avoid dramatic/exaggerated terms. Use mild, natural phrasing.
    - Colors: Use plain terms ("red dress"), not intensified ("vibrant blue," "bright red").
    - Lighting: Use neutral descriptions ("soft overhead light"), not harsh ("blinding light").
    - Facial features: Use delicate modifiers for subtle features (i.e., "subtle freckles").

#### Important notes:
- Analyze the user's raw input carefully. In cases of FPV or POV, exclude the description \
of the subject whose POV is requested.
- Camera motion: DO NOT invent camera motion unless requested by the user.
- Speech: DO NOT modify user-provided character dialogue unless it's a typo.
- No timestamps or cuts: DO NOT use timestamps or describe scene cuts unless explicitly requested.
- Format: DO NOT use phrases like "The scene opens with...". Start directly with Style \
(optional) and chronological scene description.
- Format: DO NOT start your response with special characters.
- DO NOT invent dialogue unless the user mentions speech/talking/singing/conversation.
- If the user's raw input prompt is highly detailed, chronological and in the requested \
format: DO NOT make major edits or introduce new elements. Add/enhance audio descriptions \
if missing.

#### Output Format (Strict):
- Single continuous paragraph in natural language (English).
- NO titles, headings, prefaces, code fences, or Markdown.
- If unsafe/invalid, return original user prompt. Never ask questions or clarifications.

Your output quality is CRITICAL. Generate visually rich, dynamic prompts with integrated \
audio for high-quality video generation."""


def clean_enhanced_prompt(text: str) -> str:
    """Clean Gemma's enhanced prompt output.

    Strips Unicode smart quotes and leading non-letter characters that Gemma
    tends to insert. Matches official LTX-2 ``clean_response()`` behavior.
    """
    text = text.translate(_UNICODE_REPLACEMENTS)

    # Remove leading non-letter characters
    for i, char in enumerate(text):
        if char.isalpha():
            return text[i:]
    return text


# Default paths to LTX-2 model components
# CRITICAL: Tokenizer is in tokenizer/ folder, model weights are in text_encoder/
# Using wrong tokenizer causes completely wrong token IDs -> garbage output
DEFAULT_TOKENIZER_PATH = "models/LTX-2/tokenizer"  # tokenizer.model, tokenizer.json
DEFAULT_TEXT_ENCODER_PATH = "models/LTX-2/text_encoder"  # Gemma model weights + config
DEFAULT_CONNECTOR_WEIGHTS_SHARD = (
    "models/LTX-2/text_encoder/diffusion_pytorch_model-00011-of-00012.safetensors"
)
# Legacy paths (kept for reference, do NOT use)
_LEGACY_CONNECTORS_PATH = "models/LTX-2/connectors/diffusion_pytorch_model.safetensors"
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
    - Use full/quantized models (models/LTX-2/text_encoder), NOT distilled
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
    eps: float = 1e-6,  # Match reference (was 1e-8)
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
    # Match reference: (denom + eps) instead of clamp
    denom = (seq_lengths * d).view(b, 1, 1, 1)
    mean = masked_states.sum(dim=(1, 2), keepdim=True) / (denom + eps)

    # Range per layer (over valid tokens only)
    x_min = hidden_states.masked_fill(~mask_expanded, float("inf")).amin(dim=(1, 2), keepdim=True)
    x_max = hidden_states.masked_fill(~mask_expanded, float("-inf")).amax(dim=(1, 2), keepdim=True)
    range_val = x_max - x_min

    # Normalize: 8 * (x - mean) / (range + eps)
    # Match reference: (range + eps) instead of clamp
    normed = 8.0 * (hidden_states - mean) / (range_val + eps)

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


class Gemma3Encoder(PinnedShuttleMixin):
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
        model_id: str = "models/LTX-2/text_encoder",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        max_sequence_length: int = 256,
        quantization_variant: str = "bf16",
        max_memory: Optional[dict] = None,
        connectors_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        use_connector: bool = True,
        model_version: str = "auto",  # kept for signature compat; always V2.3
    ):
        """
        Initialize Gemma3 encoder for LTX-2.3 (V2.3 only).

        Args:
            model_id: Gemma 3 model ID or path.
            device: Device to load on ("cuda", "cpu", "auto").
            dtype: Model dtype (typically bfloat16).
            max_sequence_length: Maximum sequence length for encoding.
            quantization_variant: Variant metadata for reporting.
            max_memory: Memory limits per device for CPU offloading.
            connectors_path: Path to V2.3 connector weights (safetensors).
            tokenizer_path: Path to tokenizer files.
            use_connector: Whether to use the Embeddings1DConnector.
            model_version: Ignored (always V2.3).
        """
        self._model_id = model_id
        self._device_str = device
        self._dtype = dtype
        self._max_sequence_length = max_sequence_length
        self._quantization_variant = quantization_variant
        self._max_memory = max_memory
        self._connectors_path = connectors_path or DEFAULT_CONNECTOR_WEIGHTS_SHARD
        self._text_encoder_path = model_id
        self._tokenizer_path = tokenizer_path or DEFAULT_TOKENIZER_PATH
        self._use_connector = use_connector

        self._init_shuttle_state()

        # Model components (lazy loaded)
        self._model = None
        self._tokenizer = None
        self._feature_extractor_v2: Optional[FeatureExtractorV2] = None
        self._embeddings_connector: Optional[Embeddings1DConnector] = None
        self._audio_connector: Optional[Embeddings1DConnector] = None
        self._is_loaded = False

    @classmethod
    def from_pretrained(
        cls,
        model_path: str = "models/LTX-2/text_encoder",
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_sequence_length: int = 256,
        quantization: Optional[str] = None,
        max_memory: Optional[dict] = None,
        connectors_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        use_connector: bool = True,
        **kwargs,
    ) -> "Gemma3Encoder":
        """Load Gemma3 encoder from pretrained model (V2.3 only)."""
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
            quantization_variant="q4_0" if quantization in ("4bit", "q4_0") else ("int8" if quantization in ("8bit", "int8") else "bf16"),
            max_memory=max_memory,
            connectors_path=connectors_path,
            tokenizer_path=tokenizer_path,
            use_connector=use_connector,
        )

        encoder._load_model()
        return encoder

    def _load_model(self) -> None:
        """Load Gemma 3 model and tokenizer with manual key remapping from LTX-2 checkpoint.

        The LTX-2 checkpoint stores Gemma weights with 'base_text_encoder.*' prefix,
        but HuggingFace's Gemma3ForConditionalGeneration expects 'model.*'.
        When from_pretrained() encounters mismatched keys, it silently ignores them
        and initializes with random weights - causing signal death.

        Solution: Create architecture from HuggingFace, then manually load and remap
        LTX-2 weights with correct key prefixes.
        """
        if self._is_loaded:
            return

        try:
            # Use Gemma3ForCausalLM (text-only) instead of Gemma3ForConditionalGeneration
            # The multimodal model includes SigLIP vision tower which causes OOM
            # and is not needed for text-to-video generation
            from transformers import AutoTokenizer, Gemma3ForCausalLM
        except ImportError:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Fallback for older transformers
            Gemma3ForCausalLM = AutoModelForCausalLM

        logger.info(f"Loading Gemma 3 text encoder (no vision) from: {self._text_encoder_path}")

        # Quantization is applied post-load by the variant loaders
        # (gemma3_variants.py) via quantize_component(). The _quantization_variant
        # flag is only used for metadata tracking, not model loading.

        # Determine device map
        device_map = None
        if self._device_str == "auto":
            device_map = "auto"
        elif self._device_str != "cpu":
            device_map = {"": self._device_str}

        # Step 1: Create empty model architecture from local config
        # The config.json defines the correct architecture (vocab_size=262208, etc.)
        # We skip automatic weight loading and do it manually with key remapping
        from transformers import Gemma3Config, Gemma3TextConfig

        logger.info(f"Loading Gemma3 text config from: {self._text_encoder_path}")
        full_config = Gemma3Config.from_pretrained(
            self._text_encoder_path,
            local_files_only=True,
        )

        # Extract text config for the causal LM (ignores vision tower)
        text_config = full_config.text_config

        logger.info(
            f"Config loaded: vocab_size={text_config.vocab_size}, "
            f"hidden_size={text_config.hidden_size}, "
            f"num_layers={text_config.num_hidden_layers}"
        )

        # Create text-only model from config on CPU first
        logger.info("Creating Gemma3ForCausalLM (text-only) on CPU from config...")

        # Force CPU creation to avoid GPU OOM during init
        with torch.device("cpu"):
            if hasattr(Gemma3ForCausalLM, "_from_config"):
                self._model = Gemma3ForCausalLM._from_config(text_config)
            else:
                self._model = Gemma3ForCausalLM(text_config)

        # Ensure model is on CPU and correct dtype
        self._model = self._model.to(dtype=self._dtype)

        # Step 2: Load LTX-2 weights with key remapping
        logger.info("Loading LTX-2 Gemma weights with key remapping...")
        state_dict = self._load_ltx2_gemma_weights()

        # Step 3: Load remapped weights into model (on CPU)
        if state_dict:
            missing, unexpected = self._model.load_state_dict(state_dict, strict=False)
            logger.info(
                f"Loaded LTX-2 Gemma weights: {len(state_dict)} keys loaded, "
                f"{len(missing)} missing, {len(unexpected)} unexpected"
            )

            # Separate expected missing keys from unexpected ones
            # LTX-2 uses Gemma as encoder-only (no text generation), so lm_head is not needed
            expected_missing = {"lm_head.weight"}
            unexpected_missing = [k for k in missing if k not in expected_missing]
            expected_found = [k for k in missing if k in expected_missing]

            if expected_found:
                logger.info(
                    f"Expected missing keys (encoder-only mode): {expected_found} - "
                    "will be tied to embed_tokens"
                )
            if unexpected_missing:
                logger.warning(f"Unexpected missing keys (first 10): {unexpected_missing[:10]}")
            if unexpected:
                logger.warning(f"Unexpected keys (first 10): {unexpected[:10]}")

            # Tie weights if lm_head wasn't in checkpoint (standard practice for encoder-only use)
            if "lm_head.weight" in missing and hasattr(self._model, "tie_weights"):
                self._model.tie_weights()
                logger.info("Tied lm_head weights to embed_tokens")
        else:
            logger.error("No LTX-2 Gemma weights loaded - model has random weights!")
            raise RuntimeError("Failed to load LTX-2 Gemma weights")

        # Step 4: Move model to target device(s)
        if device_map == "auto":
            from accelerate import dispatch_model, infer_auto_device_map

            logger.info("Computing device map for auto distribution...")
            device_map_computed = infer_auto_device_map(
                self._model,
                max_memory=self._max_memory,
                dtype=self._dtype,
            )
            logger.info(f"Dispatching model to devices: {set(device_map_computed.values())}")
            self._model = dispatch_model(self._model, device_map_computed)
        elif device_map is not None:
            # Single device
            device_name = (
                list(device_map.values())[0] if isinstance(device_map, dict) else device_map
            )
            logger.info(f"Moving model to {device_name}...")
            self._model = self._model.to(device=device_name)
        # else: keep on CPU

        # Load tokenizer from LOCAL LTX-2 files (AUTHORITATIVE SOURCE)
        # Vocab size analysis (2026-01-20):
        # - Standard Gemma base vocab: 256,000 tokens
        # - LTX-2 model embed_tokens: 262,208 embeddings
        # - The ~6k extra tokens are LTX-2's special tokens:
        #   * "Thinking Tokens" for video conditioning
        #   * BOI (Beginning of Image), EOI (End of Image) markers
        #   * Other special markers added during joint training
        # - HuggingFace tokenizer has DIFFERENT special tokens (vision/multimodal)
        # - Using HF tokenizer strips/mangles LTX-2's conditioning tokens!
        # - MUST use local tokenizer to preserve LTX-2 special token semantics
        logger.info(f"Loading tokenizer from LOCAL path: {self._tokenizer_path}")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._tokenizer_path,
            local_files_only=True,
            model_max_length=self._max_sequence_length,
        )
        self._tokenizer.padding_side = "left"  # Gemma prefers left padding
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Initialize and load V2.3 feature extractor + connectors
        self._load_connector_weights()

        target_device = torch.device(self._device_str) if self._device_str not in ("cpu", "auto") else None
        if target_device is not None:
            if self._feature_extractor_v2 is not None:
                self._feature_extractor_v2 = self._feature_extractor_v2.to(device=target_device)
            if self._embeddings_connector is not None:
                self._embeddings_connector = self._embeddings_connector.to(
                    device=target_device, dtype=self._dtype,
                )
            if self._audio_connector is not None:
                self._audio_connector = self._audio_connector.to(
                    device=target_device, dtype=self._dtype,
                )
        else:
            for mod in [self._embeddings_connector, self._audio_connector]:
                if mod is not None:
                    mod.to(dtype=self._dtype)

        # Set model to mode without gradients
        self._model.requires_grad_(False)

        self._is_loaded = True
        self._is_offloaded = False
        logger.info(f"Gemma 3 encoder loaded: {self._model.device}")

    def _load_connector_weights(self) -> None:
        """Load V2.3 feature extractor and embeddings connector weights.

        The connectors safetensors file contains:
        - text_embedding_projection.video_aggregate_embed.*: FeatureExtractorV2 video projection
        - text_embedding_projection.audio_aggregate_embed.*: FeatureExtractorV2 audio projection
        - model.diffusion_model.video_embeddings_connector.*: 8-block video connector
        - model.diffusion_model.audio_embeddings_connector.*: 8-block audio connector
        """
        from safetensors import safe_open

        connectors_path = Path(self._connectors_path)
        if not connectors_path.exists():
            logger.warning(
                f"Connector weights not found at {connectors_path}. "
                "Feature extractor will have random weights (BROKEN OUTPUT)."
            )
            return

        logger.info(f"Loading V2.3 connector weights from {connectors_path}")

        with safe_open(connectors_path, framework="pt") as f:
            all_keys = set(f.keys())

            # V2.3: FeatureExtractorV2 with dual projections (video + audio)
            self._feature_extractor_v2 = FeatureExtractorV2(dtype=self._dtype)

            # Load video aggregate embed (try both key formats)
            for prefix in ["text_embedding_projection.", ""]:
                vw_key = f"{prefix}video_aggregate_embed.weight"
                vb_key = f"{prefix}video_aggregate_embed.bias"
                if vw_key in all_keys:
                    w = f.get_tensor(vw_key).to(dtype=self._dtype)
                    self._feature_extractor_v2.video_aggregate_embed.weight.data = w
                    logger.info(f"Loaded V2.3 video_aggregate_embed weight: {w.shape}")
                if vb_key in all_keys:
                    b = f.get_tensor(vb_key).to(dtype=self._dtype)
                    self._feature_extractor_v2.video_aggregate_embed.bias.data = b
                if vw_key in all_keys:
                    break

            # Load audio aggregate embed
            for prefix in ["text_embedding_projection.", ""]:
                aw_key = f"{prefix}audio_aggregate_embed.weight"
                ab_key = f"{prefix}audio_aggregate_embed.bias"
                if aw_key in all_keys:
                    w = f.get_tensor(aw_key).to(dtype=self._dtype)
                    if self._feature_extractor_v2.audio_aggregate_embed is not None:
                        self._feature_extractor_v2.audio_aggregate_embed.weight.data = w
                        logger.info(f"Loaded V2.3 audio_aggregate_embed weight: {w.shape}")
                if ab_key in all_keys:
                    b = f.get_tensor(ab_key).to(dtype=self._dtype)
                    if self._feature_extractor_v2.audio_aggregate_embed is not None:
                        self._feature_extractor_v2.audio_aggregate_embed.bias.data = b
                if aw_key in all_keys:
                    break

            # Load embeddings connectors
            if self._use_connector:
                # V2.3 connector config: 8 blocks, 32 heads, gated attention
                video_config = {
                    "video_connector_attention_head_dim": 128,
                    "video_connector_num_attention_heads": 32,
                    "video_connector_num_layers": 8,
                    "video_connector_num_learnable_registers": 128,
                    "rope_type": "interleaved",
                    "rope_theta": 10000.0,
                    "rope_double_precision": False,
                    "connector_positional_embedding_max_pos": [1],
                    "apply_gated_attention": True,
                }
                audio_config = {
                    **video_config,
                    "video_connector_attention_head_dim": 64,  # Audio uses 64 head_dim
                }

                # Video connector
                self._embeddings_connector = Embeddings1DConnector.from_config(video_config)
                load_connector_weights(
                    self._embeddings_connector, connectors_path,
                    prefix="model.diffusion_model.video_embeddings_connector.",
                )
                logger.info("Loaded V2.3 video embeddings connector (8 blocks, gated)")

                # Audio connector
                if any(k.startswith("model.diffusion_model.audio_embeddings_connector.") for k in all_keys):
                    self._audio_connector = Embeddings1DConnector.from_config(audio_config)
                    load_connector_weights(
                        self._audio_connector, connectors_path,
                        prefix="model.diffusion_model.audio_embeddings_connector.",
                    )
                    logger.info("Loaded V2.3 audio embeddings connector (8 blocks, gated)")

    def _load_ltx2_gemma_weights(self) -> dict:
        """Load and remap Gemma weights from checkpoint shards.

        Supports two checkpoint formats:
        1. V1 bundled (LTX-2): diffusion_pytorch_model.safetensors.index.json
           Keys: base_text_encoder.language_model.* -> model.*
        2. Standard HF (Gemma3ForConditionalGeneration): model.safetensors.index.json
           Keys: language_model.model.* -> model.*

        Returns:
            Dict of remapped state_dict keys/tensors ready for load_state_dict()
        """
        from safetensors import safe_open

        enc_path = Path(self._text_encoder_path)

        # Detect checkpoint format
        v1_index = enc_path / "diffusion_pytorch_model.safetensors.index.json"
        hf_index = enc_path / "model.safetensors.index.json"

        if v1_index.exists():
            index_path = v1_index
            prefix = "base_text_encoder.language_model."
        elif hf_index.exists():
            index_path = hf_index
            prefix = "language_model."
        else:
            logger.error(
                f"No index file found in {enc_path}. "
                "Expected diffusion_pytorch_model.safetensors.index.json (V1) "
                "or model.safetensors.index.json (HF)."
            )
            return {}

        logger.info(f"Loading Gemma weights from: {index_path.name} (prefix={prefix!r})")

        with open(index_path) as f:
            index = json.load(f)

        weight_map = index.get("weight_map", {})

        # Find shards containing text model keys
        shards_to_load = set()
        keys_to_load = []
        for key, shard in weight_map.items():
            if key.startswith(prefix):
                shards_to_load.add(shard)
                keys_to_load.append(key)

        logger.info(
            f"Found {len(keys_to_load)} text model keys across "
            f"{len(shards_to_load)} shards"
        )

        # Load weights from each shard and remap keys
        # V1: base_text_encoder.language_model.X -> model.X
        # HF: language_model.model.X -> model.X  (language_model. stripped)
        state_dict = {}
        for shard_name in sorted(shards_to_load):
            shard_path = enc_path / shard_name
            if not shard_path.exists():
                logger.warning(f"Shard not found: {shard_path}")
                continue

            logger.debug(f"Loading shard: {shard_name}")
            with safe_open(shard_path, framework="pt") as f:
                for key in f.keys():
                    if key.startswith(prefix):
                        new_key = key[len(prefix):]
                        # V1 keys become model.* directly after stripping prefix.
                        # HF keys become model.* after stripping "language_model.".
                        tensor = f.get_tensor(key)
                        state_dict[new_key] = tensor.to(self._dtype)

        logger.info(f"Loaded and remapped {len(state_dict)} Gemma weight tensors")

        if state_dict:
            embed_key = "model.embed_tokens.weight"
            if embed_key in state_dict:
                embed = state_dict[embed_key]
                logger.info(
                    f"Embedding layer: shape={embed.shape}, "
                    f"mean={embed.float().mean():.4f}, std={embed.float().std():.4f}"
                )

        return state_dict

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
            quantization=self._quantization_variant if self._quantization_variant != "bf16" else "none",
            device=self.device,
            dtype=self._dtype,
        )

    @property
    def is_v2(self) -> bool:
        """Always True -- V1 support removed, only V2.3 (22B) is supported."""
        return True

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

    def _shuttle_module(self) -> nn.Module | None:
        return self._model

    def _shuttle_extra_modules(self) -> list[nn.Module]:
        return [m for m in [
            self._feature_extractor_v2,
            self._embeddings_connector, self._audio_connector,
        ] if m is not None]

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
            self.to_device(torch.device(self._device_str))

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

        # V2.3: per-token RMSNorm + dual projections
        audio_embeddings_list: list[torch.Tensor] | None = None
        assert self._feature_extractor_v2 is not None, (
            "FeatureExtractorV2 not initialized. Connector weights not loaded?"
        )
        video_embeds, audio_embeds = self._feature_extractor_v2(stacked, attention_mask)
        embeddings = video_embeds
        logger.debug(
            f"[TEXT-ENC] V2.3 video features: shape={list(embeddings.shape)}, "
            f"mean={embeddings.float().mean():.4f}"
        )

        # Run through embeddings connector (2-layer bidirectional transformer)
        def _run_connector(
            embeds: torch.Tensor, connector: Embeddings1DConnector, mask: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            additive_mask = (1.0 - mask.float()) * -10000.0
            additive_mask = additive_mask[:, None, None, :].to(embeds.dtype)
            if next(connector.parameters()).device != embeds.device:
                connector.to(embeds.device)
            out, out_mask = connector(embeds, additive_mask)
            if out_mask is not None:
                mask = (out_mask.squeeze(1).squeeze(1) >= -9000.0).float()
            return out, mask

        if self._embeddings_connector is not None:
            embeddings, attention_mask = _run_connector(
                embeddings, self._embeddings_connector, attention_mask,
            )
            logger.debug(
                f"Video connector output: shape={embeddings.shape}, "
                f"mean={embeddings.float().mean():.4f}"
            )

        # V2: also run audio through its own connector
        if audio_embeds is not None and self._audio_connector is not None:
            audio_embeds, _ = _run_connector(
                audio_embeds, self._audio_connector, attention_mask,
            )
            logger.debug(f"Audio connector output: shape={audio_embeds.shape}")

        # Get sequence lengths for unpadding
        seq_lengths = attention_mask.sum(dim=1).tolist()
        batch_size = len(texts)

        # Build per-sample outputs
        embedding_list = []
        mask_list = []
        if audio_embeds is not None:
            audio_embeddings_list = []
        for i in range(batch_size):
            if self._embeddings_connector is not None:
                embedding_list.append(embeddings[i])
                mask_list.append(attention_mask[i])
            else:
                valid_mask = attention_mask[i].bool()
                embedding_list.append(embeddings[i][valid_mask])
                mask_list.append(valid_mask[valid_mask])

            if audio_embeds is not None and audio_embeddings_list is not None:
                if self._audio_connector is not None:
                    audio_embeddings_list.append(audio_embeds[i])
                else:
                    valid_mask = attention_mask[i].bool()
                    audio_embeddings_list.append(audio_embeds[i][valid_mask])

        return EncodingOutput(
            embeddings=embedding_list,
            attention_masks=mask_list,
            padded_embeddings=embeddings if return_padded else None,
            padded_mask=attention_mask if return_padded else None,
            token_counts=[int(s) for s in seq_lengths],
            audio_embeddings=audio_embeddings_list,
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
            (models/LTX-2/text_encoder), NOT distilled variants.
            Distilled models may have compressed intermediate representations that
            don't represent true layer specialization.
        """
        if not self._is_loaded:
            self._load_model()

        if self._is_offloaded:
            self.to_device(torch.device(self._device_str))

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

                # Always create an appropriately-sized feature extractor for research
                fe = FeatureExtractorLinear(
                    input_dim=actual_feature_dim,
                    output_dim=GEMMA3_OUTPUT_DIM,
                    dtype=self._dtype,
                ).to(normalized.device)
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
            self.to_device(torch.device(self._device_str))

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
            self.to_device(torch.device(self._device_str))

        # Build structured messages for proper Gemma3 chat formatting.
        # The tokenizer ships chat_template.jinja with <start_of_turn>/<end_of_turn>
        # tokens -- naive string concatenation produces worse results.
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        full_prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Use model's actual context window for generation truncation, not
        # _max_sequence_length (which is the DiT embedding output size, e.g. 256).
        model_ctx = getattr(self._model.config, "max_position_embeddings", 8192)
        encoded = self._tokenizer(
            full_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=model_ctx - max_new_tokens,
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

    def to(self, device: torch.device) -> "Gemma3Encoder":
        """Move model to device."""
        if self._model is not None:
            self._model.to(device)
        if self._feature_extractor_v2 is not None:
            self._feature_extractor_v2.to(device)
        if self._embeddings_connector is not None:
            self._embeddings_connector.to(device)
        if self._audio_connector is not None:
            self._audio_connector.to(device)
        self._is_offloaded = device.type == "cpu"
        return self


# Convenience alias
LTX2Encoder = Gemma3Encoder
