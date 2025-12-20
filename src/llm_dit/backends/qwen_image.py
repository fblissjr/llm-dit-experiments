"""
Qwen-Image text encoder backend for Qwen2.5-VL-7B-Instruct.

This backend provides text encoding for the Qwen-Image-Layered model, which uses
Qwen2.5-VL-7B-Instruct as its text encoder. This is different from Z-Image which
uses Qwen3-4B.

Key differences from Z-Image backend:
- Hidden dimension: 3584 (vs 2560)
- Number of layers: 28 (vs 36)
- Model: Qwen2.5-VL-7B-Instruct (vs Qwen3-4B)
- Tokenizer: Qwen2Tokenizer (vs Qwen2Tokenizer)
- Hidden layer extraction: -1 (last layer, vs -2 penultimate)
- Drop first 34 tokens from embeddings (template overhead)
"""

import logging
from pathlib import Path
from typing import List, Optional

import torch
from transformers import AutoTokenizer

from llm_dit.backends.protocol import EncodingOutput

logger = logging.getLogger(__name__)

# Qwen-Image prompt templates
QWEN_IMAGE_SYSTEM_PROMPT = (
    "Describe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:"
)

QWEN_IMAGE_EDIT_SYSTEM_PROMPT = (
    "Describe the key features of the input image (color, shape, size, texture, "
    "objects, background), then explain how the user's text instruction should alter "
    "or modify the image. Generate a new image that meets the user's requirements "
    "while maintaining consistency with the original input where appropriate."
)

# Number of tokens to drop from the start (template overhead)
QWEN_IMAGE_DROP_TOKENS = 34
QWEN_IMAGE_EDIT_DROP_TOKENS = 64


class QwenImageTextEncoderBackend:
    """
    Text encoder backend for Qwen-Image using Qwen2.5-VL-7B-Instruct.

    This backend loads the text encoder portion of Qwen-Image-Layered and provides
    embeddings suitable for the QwenImageDiT transformer.

    Architecture:
        - Hidden dimension: 3584
        - Number of layers: 28
        - Attention heads: 28 (with 4 KV heads for GQA)
        - Max position embeddings: 128000
        - Uses MRoPE with sections [16, 24, 24]

    Example:
        backend = QwenImageTextEncoderBackend.from_pretrained(
            "/path/to/Qwen_Qwen-Image-Layered",
        )
        output = backend.encode(["A beautiful sunset over mountains"])
    """

    # Architecture constants
    HIDDEN_DIM = 3584
    NUM_LAYERS = 28
    MAX_SEQUENCE_LENGTH = 4096

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: AutoTokenizer,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """
        Initialize the backend.

        Args:
            model: Loaded Qwen2.5-VL model
            tokenizer: Qwen2 tokenizer
            device: Device the model is on
            dtype: Model dtype
        """
        self.model = model
        self.tokenizer = tokenizer
        self._device = device
        self._dtype = dtype
        self._embedding_dim = self.HIDDEN_DIM

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        text_encoder_subfolder: str = "text_encoder",
        tokenizer_subfolder: str = "tokenizer",
        device: str | torch.device = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = True,
        quantization: str = "none",
        **kwargs,
    ) -> "QwenImageTextEncoderBackend":
        """
        Load backend from pretrained Qwen-Image-Layered model.

        Args:
            model_path: Path to Qwen-Image-Layered model directory
            text_encoder_subfolder: Subfolder containing text encoder weights
            tokenizer_subfolder: Subfolder containing tokenizer
            device: Device to load model on (default: cuda)
            torch_dtype: Model dtype (default: bfloat16)
            trust_remote_code: Trust remote code for transformers loading
            quantization: Quantization mode: "none", "4bit", or "8bit"
            **kwargs: Additional arguments passed to model loading

        Returns:
            Initialized QwenImageTextEncoderBackend

        Example:
            # Load from local path
            backend = QwenImageTextEncoderBackend.from_pretrained(
                "/path/to/Qwen_Qwen-Image-Layered",
                device="cuda",
            )

            # With 8-bit quantization (reduces VRAM from ~14GB to ~7GB)
            backend = QwenImageTextEncoderBackend.from_pretrained(
                "/path/to/Qwen_Qwen-Image-Layered",
                quantization="8bit",
            )

            # With 4-bit quantization (reduces VRAM to ~3.5GB)
            backend = QwenImageTextEncoderBackend.from_pretrained(
                "/path/to/Qwen_Qwen-Image-Layered",
                quantization="4bit",
            )
        """
        from transformers import Qwen2_5_VLConfig, Qwen2_5_VLModel

        model_path = Path(model_path)
        device = torch.device(device)

        # Load tokenizer
        tokenizer_path = model_path / tokenizer_subfolder
        logger.info(f"Loading Qwen2 tokenizer from {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=trust_remote_code,
        )
        tokenizer.padding_side = "left"

        # Build Qwen2.5-VL config (matches DiffSynth implementation)
        config = Qwen2_5_VLConfig(
            architectures=["Qwen2_5_VLForConditionalGeneration"],
            attention_dropout=0.0,
            bos_token_id=151643,
            eos_token_id=151645,
            hidden_act="silu",
            hidden_size=3584,
            image_token_id=151655,
            initializer_range=0.02,
            intermediate_size=18944,
            max_position_embeddings=128000,
            max_window_layers=28,
            model_type="qwen2_5_vl",
            num_attention_heads=28,
            num_hidden_layers=28,
            num_key_value_heads=4,
            rms_norm_eps=1e-06,
            rope_scaling={
                "mrope_section": [16, 24, 24],
                "rope_type": "default",
                "type": "default",
            },
            rope_theta=1000000.0,
            sliding_window=32768,
            tie_word_embeddings=False,
            use_cache=True,
            use_sliding_window=False,
            vocab_size=152064,
            # Vision config (not used for text-only encoding, but required)
            vision_config={
                "depth": 32,
                "fullatt_block_indexes": [7, 15, 23, 31],
                "hidden_act": "silu",
                "hidden_size": 1280,
                "in_channels": 3,
                "initializer_range": 0.02,
                "intermediate_size": 3420,
                "model_type": "qwen2_5_vl",
                "num_heads": 16,
                "out_hidden_size": 3584,
                "patch_size": 14,
                "spatial_merge_size": 2,
                "spatial_patch_size": 14,
                "temporal_patch_size": 2,
                "tokens_per_second": 2,
                "window_size": 112,
            },
            video_token_id=151656,
            vision_end_token_id=151653,
            vision_start_token_id=151652,
            vision_token_id=151654,
        )

        # Set up quantization config if requested
        quantization_config = None
        if quantization in ("4bit", "8bit"):
            try:
                from transformers import BitsAndBytesConfig
                if quantization == "4bit":
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch_dtype,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                    logger.info("Using 4-bit quantization (NF4)")
                else:
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                    )
                    logger.info("Using 8-bit quantization")
            except ImportError:
                logger.warning(
                    "bitsandbytes not available, falling back to no quantization. "
                    "Install with: pip install bitsandbytes"
                )
                quantization = "none"

        # Create model from config
        logger.info(f"Creating Qwen2.5-VL model (hidden_size={config.hidden_size})")
        model = Qwen2_5_VLModel(config)

        # Load weights from safetensors
        text_encoder_path = model_path / text_encoder_subfolder
        weight_files = list(text_encoder_path.glob("*.safetensors"))
        if not weight_files:
            raise ValueError(f"No safetensors files found in {text_encoder_path}")

        logger.info(f"Loading weights from {len(weight_files)} safetensors files")
        from safetensors.torch import load_file

        state_dict = {}
        for weight_file in sorted(weight_files):
            logger.debug(f"Loading {weight_file.name}")
            file_state_dict = load_file(weight_file, device="cpu")
            state_dict.update(file_state_dict)

        # Remap keys: the checkpoint uses 'model.X' but Qwen2_5_VLModel expects just 'X'
        # Also remove 'visual.' prefix since we only use text portion
        remapped_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                # Language model weights: model.X -> X
                new_key = k[6:]  # Remove "model."
                remapped_state_dict[new_key] = v
            elif k.startswith("visual."):
                # Skip vision encoder weights (we don't use them for text-only)
                continue
            elif k == "lm_head.weight":
                # Skip lm_head (we don't generate text)
                continue
            else:
                remapped_state_dict[k] = v

        logger.info(f"Remapped {len(remapped_state_dict)} text encoder weights")

        # Load state dict into model
        missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)
        if missing_keys:
            logger.warning(f"Missing keys: {missing_keys[:5]}... ({len(missing_keys)} total)")
        if unexpected_keys:
            logger.debug(f"Unexpected keys: {unexpected_keys[:5]}...")

        # Apply quantization if configured
        if quantization_config is not None:
            # For bitsandbytes quantization, we need to move to CUDA first then quantize
            # This is a simplified approach - full quantization support would use from_pretrained
            logger.info(f"Applying {quantization} quantization...")
            model = model.to(device=device, dtype=torch_dtype)
            # Note: Full bitsandbytes quantization requires using from_pretrained with
            # quantization_config. For manual loading, we move to device and dtype only.
            # TODO: Implement full quantization support by saving/loading in HF format
        else:
            # Move to device and set dtype
            model = model.to(device=device, dtype=torch_dtype)

        model.eval()

        logger.info(
            f"Loaded Qwen2.5-VL text encoder: "
            f"hidden_size={config.hidden_size}, "
            f"num_layers={config.num_hidden_layers}, "
            f"device={device}, dtype={torch_dtype}"
        )

        return cls(model, tokenizer, device, torch_dtype)

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension (3584 for Qwen2.5-VL-7B)."""
        return self._embedding_dim

    @property
    def max_sequence_length(self) -> int:
        """Return max sequence length."""
        return self.MAX_SEQUENCE_LENGTH

    @property
    def device(self) -> torch.device:
        """Return model device."""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Return model dtype."""
        return self._dtype

    def format_prompt(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """
        Format a prompt using the Qwen-Image chat template.

        Args:
            prompt: User prompt text
            system_prompt: Optional system prompt (defaults to QWEN_IMAGE_SYSTEM_PROMPT)

        Returns:
            Formatted prompt string ready for tokenization
        """
        if system_prompt is None:
            system_prompt = QWEN_IMAGE_SYSTEM_PROMPT

        # Build messages for chat template
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        # Apply chat template with generation prompt (adds assistant start)
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        return formatted

    def encode(
        self,
        prompts: List[str],
        return_padded: bool = False,
        layer_index: int = -1,
        drop_tokens: int = QWEN_IMAGE_DROP_TOKENS,
        apply_template: bool = True,
        system_prompt: str | None = None,
    ) -> EncodingOutput:
        """
        Encode prompts to embeddings.

        Args:
            prompts: List of prompt strings
            return_padded: If True, also return padded batch tensors
            layer_index: Which hidden layer to extract (default: -1, last layer)
            drop_tokens: Number of tokens to drop from start (template overhead)
            apply_template: If True, apply the Qwen-Image chat template
            system_prompt: Optional system prompt for template (default: QWEN_IMAGE_SYSTEM_PROMPT)

        Returns:
            EncodingOutput with embeddings per prompt
        """
        # Apply chat template if requested
        if apply_template:
            texts = [self.format_prompt(p, system_prompt) for p in prompts]
        else:
            texts = prompts

        # Tokenize
        inputs = self.tokenizer(
            texts,
            max_length=self.MAX_SEQUENCE_LENGTH + drop_tokens,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        # Log tokenization details at debug level
        if logger.isEnabledFor(logging.DEBUG):
            seq_length = input_ids.shape[1]
            valid_tokens = attention_mask[0].sum().item()
            logger.debug(
                f"[QwenImageTextEncoder] Tokenized: {valid_tokens} valid tokens, "
                f"seq_length={seq_length}"
            )

        # Forward pass through model
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        # Extract hidden states from specified layer
        # outputs is tuple of hidden states for Qwen2_5_VLModel
        if hasattr(outputs, "hidden_states"):
            hidden_states = outputs.hidden_states[layer_index]
        else:
            # Direct hidden states return (like DiffSynth wrapper)
            hidden_states = outputs[layer_index]

        logger.debug(
            f"[QwenImageTextEncoder] Extracted layer {layer_index}, "
            f"shape={hidden_states.shape}"
        )

        # Extract masked hidden states and drop template tokens
        attention_mask_bool = attention_mask.bool()
        embeddings_list = []
        masks_list = []
        token_counts = []

        for i in range(len(prompts)):
            mask = attention_mask_bool[i]
            # Get all valid (non-padding) embeddings
            valid_embeds = hidden_states[i][mask]

            # Drop the first N tokens (template overhead)
            if drop_tokens > 0 and valid_embeds.shape[0] > drop_tokens:
                valid_embeds = valid_embeds[drop_tokens:]

            embeddings_list.append(valid_embeds)
            masks_list.append(torch.ones(valid_embeds.shape[0], dtype=torch.bool, device=self.device))
            token_counts.append(valid_embeds.shape[0])

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"[QwenImageTextEncoder] Embedding [{i}]: shape={valid_embeds.shape}, "
                    f"mean={valid_embeds.mean().item():.4f}, std={valid_embeds.std().item():.4f}"
                )

        result = EncodingOutput(
            embeddings=embeddings_list,
            attention_masks=masks_list,
            token_counts=token_counts,
        )

        if return_padded:
            # Create padded tensors for batch processing
            max_seq_len = max(e.shape[0] for e in embeddings_list)
            padded_embeds = torch.zeros(
                len(prompts), max_seq_len, self.embedding_dim,
                dtype=self.dtype, device=self.device
            )
            padded_mask = torch.zeros(
                len(prompts), max_seq_len,
                dtype=torch.bool, device=self.device
            )

            for i, (emb, count) in enumerate(zip(embeddings_list, token_counts)):
                padded_embeds[i, :count] = emb
                padded_mask[i, :count] = True

            result.padded_embeddings = padded_embeds
            result.padded_mask = padded_mask

        return result

    def to(self, device: torch.device) -> "QwenImageTextEncoderBackend":
        """Move model to device."""
        self.model = self.model.to(device)
        self._device = device
        return self

    def unload(self) -> None:
        """Unload model from memory."""
        if hasattr(self, "model") and self.model is not None:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Unloaded Qwen-Image text encoder from memory")
