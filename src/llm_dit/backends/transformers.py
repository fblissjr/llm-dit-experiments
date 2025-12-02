"""
HuggingFace transformers backend for text encoding.

This is the reference implementation, matching the behavior of:
- DiffSynth-Studio z_image.py encode_prompt()
- diffusers ZImagePipeline._encode_prompt()

Key implementation details:
- Uses AutoModel (not ForCausalLM) since we only need embeddings
- Extracts hidden_states[-2] (penultimate layer)
- Filters by attention mask to get variable-length outputs
- Tokenizer padding_side="left" as per Qwen3 convention
"""

import logging
from pathlib import Path
from typing import List

import torch
from transformers import AutoModel, AutoTokenizer

from llm_dit.backends.config import BackendConfig
from llm_dit.backends.protocol import EncodingOutput

logger = logging.getLogger(__name__)


class TransformersBackend:
    """
    HuggingFace transformers backend for Qwen3-4B text encoding.

    This backend loads the text encoder portion of Z-Image and provides
    embeddings suitable for the DiT context refiner.

    Example:
        backend = TransformersBackend.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo",
            subfolder="text_encoder",
        )
        output = backend.encode(["<|im_start|>user\\nA cat<|im_end|>"])
    """

    def __init__(
        self,
        model: AutoModel,
        tokenizer: AutoTokenizer,
        config: BackendConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self._embedding_dim = model.config.hidden_size
        self._device = next(model.parameters()).device
        self._dtype = next(model.parameters()).dtype

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        subfolder: str = "text_encoder",
        config: BackendConfig | None = None,
        **kwargs,
    ) -> "TransformersBackend":
        """
        Load backend from pretrained model.

        Args:
            model_path: Path to model directory or HuggingFace ID
            subfolder: Subfolder containing text encoder (default: "text_encoder")
            config: Optional BackendConfig, created from defaults if not provided
            **kwargs: Additional arguments passed to from_pretrained

        Returns:
            Initialized TransformersBackend

        Example:
            # Load from HuggingFace
            backend = TransformersBackend.from_pretrained("Tongyi-MAI/Z-Image-Turbo")

            # Load from local path
            backend = TransformersBackend.from_pretrained("/path/to/model")
        """
        if config is None:
            config = BackendConfig.for_z_image(model_path, subfolder=subfolder)

        # Merge kwargs with config
        torch_dtype = kwargs.pop("torch_dtype", config.get_torch_dtype())
        device_map = kwargs.pop("device_map", config.device)
        trust_remote_code = kwargs.pop("trust_remote_code", config.trust_remote_code)

        # Determine model path
        model_path_full = model_path
        if subfolder and Path(model_path).exists():
            # Local path with subfolder
            model_path_full = str(Path(model_path) / subfolder)
        elif subfolder:
            # HuggingFace ID with subfolder
            pass  # transformers handles subfolder param

        logger.info(f"Loading tokenizer from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            subfolder=subfolder if not Path(model_path_full).exists() else None,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

        # Qwen3 uses left padding
        tokenizer.padding_side = "left"

        logger.info(f"Loading model from {model_path} (dtype={torch_dtype})")
        model = AutoModel.from_pretrained(
            model_path,
            subfolder=subfolder if not Path(model_path_full).exists() else None,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        model.eval()

        logger.info(
            f"Loaded {model.config.model_type} with "
            f"hidden_size={model.config.hidden_size}, "
            f"num_layers={model.config.num_hidden_layers}"
        )

        return cls(model, tokenizer, config)

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension (2560 for Qwen3-4B)."""
        return self._embedding_dim

    @property
    def max_sequence_length(self) -> int:
        """Return max sequence length from config."""
        return self.config.max_length

    @property
    def device(self) -> torch.device:
        """Return model device."""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Return model dtype."""
        return self._dtype

    def encode(
        self,
        texts: List[str],
        return_padded: bool = False,
    ) -> EncodingOutput:
        """
        Encode pre-formatted text to embeddings.

        This method expects text that already has the chat template applied.
        It does NOT apply any additional formatting.

        Args:
            texts: List of pre-formatted text strings
            return_padded: If True, also return padded batch tensors

        Returns:
            EncodingOutput with variable-length embeddings per input

        Implementation matches DiffSynth z_image.py lines 174-196:
        1. Tokenize with padding="max_length"
        2. Forward through model with output_hidden_states=True
        3. Extract hidden_states[-2] (penultimate layer)
        4. Filter by attention_mask to get valid tokens only
        """
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding="max_length",
            max_length=self.config.max_length,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device).bool()

        # Encode
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        # Extract penultimate layer (hidden_states[-2])
        # This matches Z-Image reference: hidden_states[-2]
        hidden_states = outputs.hidden_states[-2]

        # Filter by attention mask to get variable-length outputs
        # This matches diffusers behavior
        embeddings_list = []
        masks_list = []
        for i in range(len(texts)):
            mask = attention_mask[i]
            valid_embeds = hidden_states[i][mask]
            embeddings_list.append(valid_embeds)
            masks_list.append(mask[mask])  # All True for valid positions

        result = EncodingOutput(
            embeddings=embeddings_list,
            attention_masks=masks_list,
        )

        if return_padded:
            result.padded_embeddings = hidden_states
            result.padded_mask = attention_mask

        return result

    def to(self, device: torch.device) -> "TransformersBackend":
        """Move model to device."""
        self.model = self.model.to(device)
        self._device = device
        return self
