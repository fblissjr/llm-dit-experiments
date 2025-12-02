"""
TextEncoderBackend Protocol definition.

This Protocol defines the interface that all LLM backends must implement.
Using Protocol (duck typing) instead of ABC allows any class that implements
the required methods to be used, without forcing inheritance.

Key design decisions:
- Backends receive PRE-FORMATTED text (chat template already applied)
- Backends return embeddings from hidden_states[-2] (penultimate layer)
- Attention masks are returned for filtering padding
- Variable-length outputs (filtered by mask) match diffusers behavior
"""

from dataclasses import dataclass
from typing import List, Protocol, runtime_checkable

import torch


@dataclass
class EncodingOutput:
    """
    Output from text encoding.

    Attributes:
        embeddings: List of tensors, one per input text.
                   Each tensor has shape [seq_len, hidden_dim] where seq_len
                   is the number of valid (non-padding) tokens.
        attention_masks: List of boolean tensors indicating valid positions.
                        Shape [seq_len] for each input.
        padded_embeddings: Optional padded batch tensor [batch, max_seq_len, hidden_dim].
                          Useful when downstream needs uniform shapes.
        padded_mask: Optional padded attention mask [batch, max_seq_len].
    """

    embeddings: List[torch.Tensor]
    attention_masks: List[torch.Tensor]
    padded_embeddings: torch.Tensor | None = None
    padded_mask: torch.Tensor | None = None

    @property
    def batch_size(self) -> int:
        return len(self.embeddings)

    @property
    def hidden_dim(self) -> int:
        if self.embeddings:
            return self.embeddings[0].shape[-1]
        return 0

    def to(self, device: torch.device) -> "EncodingOutput":
        """Move all tensors to device."""
        return EncodingOutput(
            embeddings=[e.to(device) for e in self.embeddings],
            attention_masks=[m.to(device) for m in self.attention_masks],
            padded_embeddings=(
                self.padded_embeddings.to(device)
                if self.padded_embeddings is not None
                else None
            ),
            padded_mask=(
                self.padded_mask.to(device) if self.padded_mask is not None else None
            ),
        )


@runtime_checkable
class TextEncoderBackend(Protocol):
    """
    Protocol for pluggable LLM text encoder backends.

    Implementations must provide:
    - embedding_dim: The hidden dimension of embeddings (e.g., 2560 for Qwen3-4B)
    - max_sequence_length: Maximum supported sequence length
    - encode(): Encode pre-formatted text to embeddings

    Example usage:
        backend = TransformersBackend.from_pretrained("path/to/model")
        output = backend.encode(["<|im_start|>user\\nHello<|im_end|>"])
        embeddings = output.embeddings[0]  # [seq_len, 2560]
    """

    @property
    def embedding_dim(self) -> int:
        """
        Return the embedding dimension.

        For Qwen3-4B: 2560
        For Qwen2.5-VL 7B: 3584
        """
        ...

    @property
    def max_sequence_length(self) -> int:
        """
        Return maximum supported sequence length.

        Default for Z-Image: 512
        """
        ...

    @property
    def device(self) -> torch.device:
        """Return the device the model is on."""
        ...

    @property
    def dtype(self) -> torch.dtype:
        """Return the model's dtype (e.g., torch.bfloat16)."""
        ...

    def encode(
        self,
        texts: List[str],
        return_padded: bool = False,
    ) -> EncodingOutput:
        """
        Encode pre-formatted text to embeddings.

        IMPORTANT: The input texts should already have chat template applied.
        This method does NOT apply any additional formatting.

        Args:
            texts: List of pre-formatted text strings (with chat template tokens).
            return_padded: If True, also return padded batch tensors.

        Returns:
            EncodingOutput with variable-length embeddings per input.
            Each embedding has shape [valid_seq_len, hidden_dim].

        Note:
            Embeddings are extracted from hidden_states[-2] (penultimate layer),
            matching the Z-Image reference implementation.
        """
        ...

    def to(self, device: torch.device) -> "TextEncoderBackend":
        """Move model to device."""
        ...
