"""
Qwen3-Embedding-4B extractor for Z-Image text conditioning.

This module provides an alternative text encoder using Qwen3-Embedding-4B,
which is specifically trained for embedding quality via contrastive learning.

Architecture Compatibility:
    Qwen3-Embedding-4B uses the same Qwen3ForCausalLM architecture as Qwen3-4B:
    - hidden_size: 2560 (matches Z-Image requirement)
    - num_hidden_layers: 36
    - Same tokenizer family (slightly smaller vocab: 151665 vs 151936)

Key Differences from Qwen3-4B:
    1. Training objective: Contrastive learning for retrieval/similarity
    2. Vocab size: 151665 (vs 151936) - removed ~13K unused tokens
    3. Optimized for semantic representation quality
    4. Supports Matryoshka Representation Learning (MRL) for variable-dim embeddings

Z-Image Compatibility (Based on Experiments 2025-12-14):
    - Use layer -2 (NOT the embedding model's default layer -1)
    - Do NOT use instruction prefixes (they dilute the prompt and shift distribution)
    - Apply 1.15x scaling factor to match Qwen3-4B magnitude distribution
    - Layer -2 produces 99% cosine similarity with Qwen3-4B embeddings
    - Layer -1 is INCOMPATIBLE (only 6% cosine similarity due to contrastive training)

Extraction Modes:
    - "full_sequence": Extract all token hidden states (for Z-Image conditioning)
    - "last_token": Extract only last token's hidden state (for retrieval/similarity)

Reference:
    https://huggingface.co/Qwen/Qwen3-Embedding-4B
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingExtractionResult:
    """Result of embedding extraction."""

    embeddings: torch.Tensor  # (seq_len, hidden_dim) or (hidden_dim,) depending on mode
    num_tokens: int
    hidden_layer: int
    pooling_mode: str  # "full_sequence" or "last_token"
    embedding_dim: int  # Actual embedding dimension (can be reduced via MRL)
    normalized: bool  # Whether L2 normalized
    instruction: str | None  # Instruction prefix used (if any)
    original_text: str
    # Statistics
    mean: float
    std: float


class EmbeddingExtractor:
    """
    Extract embeddings from Qwen3-Embedding-4B for Z-Image conditioning.

    This class provides two extraction modes:
    1. full_sequence: Returns all token hidden states for Z-Image DiT conditioning
    2. last_token: Returns single pooled embedding for retrieval/similarity

    Example (Z-Image conditioning):
        >>> extractor = EmbeddingExtractor.from_pretrained("/path/to/Qwen3-Embedding-4B")
        >>> result = extractor.extract("A cat sleeping in sunlight", mode="full_sequence")
        >>> # Use result.embeddings with Z-Image pipeline

    Example (retrieval embedding):
        >>> result = extractor.extract(
        ...     "What is the capital of France?",
        ...     mode="last_token",
        ...     instruction="Given a query, retrieve relevant documents",
        ...     normalize=True,
        ... )
        >>> # Use result.embeddings for similarity search
    """

    def __init__(
        self,
        model: "AutoModel",
        tokenizer: "AutoTokenizer",
        device: str = "cuda",
    ):
        """
        Initialize with pre-loaded model and tokenizer.

        Use `from_pretrained()` factory method instead of direct initialization.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self._loaded = True
        self._hidden_size = model.config.hidden_size

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "EmbeddingExtractor":
        """
        Load Qwen3-Embedding-4B from path.

        Args:
            model_path: Path to model directory or HuggingFace model ID
            device: Device to load model on (cuda, cpu, auto)
            torch_dtype: Model precision (default: bfloat16)

        Returns:
            EmbeddingExtractor instance
        """
        from transformers import AutoModel, AutoTokenizer

        logger.info(f"Loading Qwen3-Embedding-4B from {model_path}...")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Qwen3 convention: left padding
        tokenizer.padding_side = "left"

        model = AutoModel.from_pretrained(
            model_path,
            dtype=torch_dtype,
            device_map=device,
        )
        model.eval()

        hidden_size = model.config.hidden_size
        logger.info(f"Qwen3-Embedding-4B loaded. Hidden size: {hidden_size}")

        if hidden_size != 2560:
            logger.warning(
                f"Hidden size {hidden_size} != 2560. "
                "This model may not be compatible with Z-Image."
            )

        return cls(model, tokenizer, device)

    def unload(self) -> None:
        """Unload model to free memory."""
        if self._loaded:
            del self.model
            del self.tokenizer
            self._loaded = False
            torch.cuda.empty_cache()
            logger.info("Qwen3-Embedding-4B unloaded")

    def extract(
        self,
        text: str,
        mode: str = "full_sequence",
        hidden_layer: int = -2,
        instruction: str | None = None,
        normalize: bool = False,
        embedding_dim: int | None = None,
    ) -> EmbeddingExtractionResult:
        """
        Extract embeddings from text.

        Args:
            text: Input text to encode
            mode: Extraction mode:
                - "full_sequence": All token hidden states (for Z-Image)
                - "last_token": Last token only (for retrieval)
            hidden_layer: Which hidden layer to extract from (default: -2)
                IMPORTANT: Use -2 for Z-Image compatibility. Layer -1 produces
                only 6% cosine similarity with Qwen3-4B due to contrastive training.
            instruction: Optional instruction prefix for retrieval tasks.
                Format: "Instruct: {instruction}\\nQuery: {text}"
                WARNING: Do NOT use for Z-Image - dilutes prompt and shifts distribution.
            normalize: Whether to L2 normalize embeddings (recommended for retrieval)
            embedding_dim: Optional reduced dimension (MRL support, 32-2560)
                If None, uses full 2560 dimensions

        Returns:
            EmbeddingExtractionResult with embeddings and metadata
        """
        if not self._loaded:
            raise RuntimeError("Model has been unloaded. Create new extractor.")

        if mode not in ("full_sequence", "last_token"):
            raise ValueError(f"Invalid mode: {mode}. Use 'full_sequence' or 'last_token'")

        # Format input with instruction if provided
        if instruction:
            input_text = f"Instruct: {instruction}\nQuery: {text}"
        else:
            input_text = text

        # Tokenize
        inputs = self.tokenizer(
            input_text,
            padding=True,
            truncation=True,
            max_length=8192,
            return_tensors="pt",
        )

        # Move to device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        # Extract hidden states from specified layer
        hidden_states = outputs.hidden_states[hidden_layer]  # (1, seq_len, hidden_dim)

        # Get attention mask for token filtering
        attention_mask = inputs["attention_mask"]
        num_tokens = attention_mask.sum().item()

        if mode == "full_sequence":
            # Extract all non-padding tokens (for Z-Image conditioning)
            # Shape: (seq_len, hidden_dim)
            mask = attention_mask[0].bool()
            embeddings = hidden_states[0, mask]
        else:
            # Last token pooling (for retrieval)
            # Shape: (hidden_dim,)
            embeddings = self._last_token_pool(hidden_states, attention_mask)
            embeddings = embeddings.squeeze(0)

        # Optional: reduce dimension via MRL (truncation)
        actual_dim = self._hidden_size
        if embedding_dim is not None and embedding_dim < self._hidden_size:
            embeddings = embeddings[..., :embedding_dim]
            actual_dim = embedding_dim

        # Optional: L2 normalize
        if normalize:
            if mode == "full_sequence":
                # Normalize each token embedding
                embeddings = F.normalize(embeddings, p=2, dim=-1)
            else:
                # Normalize single embedding
                embeddings = F.normalize(embeddings, p=2, dim=-1)

        return EmbeddingExtractionResult(
            embeddings=embeddings.cpu(),
            num_tokens=num_tokens,
            hidden_layer=hidden_layer,
            pooling_mode=mode,
            embedding_dim=actual_dim,
            normalized=normalize,
            instruction=instruction,
            original_text=text,
            mean=embeddings.mean().item(),
            std=embeddings.std().item(),
        )

    def _last_token_pool(
        self,
        last_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pool by taking the last non-padding token's hidden state.

        This is the official pooling method for Qwen3-Embedding models.
        With left-padding, the last token is always at position -1.

        Args:
            last_hidden_states: Hidden states (batch, seq_len, hidden_dim)
            attention_mask: Attention mask (batch, seq_len)

        Returns:
            Pooled embeddings (batch, hidden_dim)
        """
        # Check if left-padded (all sequences end with real token)
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]

        if left_padding:
            # Simple case: last position is always the last real token
            return last_hidden_states[:, -1]
        else:
            # Right-padded: find last real token for each sequence
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device),
                sequence_lengths,
            ]

    def encode_for_zimage(
        self,
        text: str,
        hidden_layer: int = -2,
        scale_factor: float = 1.15,
    ) -> torch.Tensor:
        """
        Convenience method to extract embeddings for Z-Image conditioning.

        Uses full_sequence mode with layer -2 (penultimate) to match
        the default Qwen3-4B extraction for Z-Image.

        NOTE: This encodes raw text without chat template. For best results
        matching Qwen3-4B output, use encode_for_zimage_formatted() instead.

        IMPORTANT: Based on experiments (2025-12-14):
        - Always use layer -2 (NOT -1, which is incompatible)
        - Do NOT use instruction prefixes (they dilute the prompt)
        - Apply 1.15x scaling to match Qwen3-4B magnitude distribution

        Args:
            text: Prompt text to encode (NO instruction prefix)
            hidden_layer: Hidden layer to extract from (default: -2, DO NOT use -1)
            scale_factor: Magnitude scaling to match Qwen3-4B distribution
                         (default: 1.15 based on std_ratio ~0.87 at layer -2)

        Returns:
            Embeddings tensor of shape (seq_len, 2560)
        """
        if hidden_layer == -1:
            logger.warning(
                "Layer -1 produces only 6% cosine similarity with Qwen3-4B. "
                "Use -2 instead for Z-Image compatibility."
            )

        result = self.extract(
            text,
            mode="full_sequence",
            hidden_layer=hidden_layer,
            normalize=False,
            instruction=None,  # Explicitly no instruction for Z-Image
        )

        embeddings = result.embeddings
        if scale_factor != 1.0:
            embeddings = embeddings * scale_factor

        return embeddings

    def encode_for_zimage_formatted(
        self,
        text: str,
        hidden_layer: int = -2,
        scale_factor: float = 1.0,
    ) -> torch.Tensor:
        """
        Encode with Qwen3 chat template for Z-Image conditioning.

        This method applies the same chat template formatting as Qwen3-4B,
        producing embeddings that should be visually equivalent when used
        with Z-Image DiT.

        Format applied (matches official Z-Image HF Space with enable_thinking=True):
            <|im_start|>user
            {text}<|im_end|>
            <|im_start|>assistant

        IMPORTANT: This does NOT include think blocks because:
        1. Qwen3-Embedding-4B doesn't have <think>/<think> as special tokens
           (they get tokenized as regular text, breaking compatibility)
        2. The official Z-Image HF Space uses enable_thinking=True which
           means NO think block

        Args:
            text: Prompt text to encode
            hidden_layer: Hidden layer to extract from (default: -2)
            scale_factor: Magnitude scaling (default: 1.0)

        Returns:
            Embeddings tensor of shape (seq_len, 2560)
        """
        if hidden_layer == -1:
            logger.warning(
                "Layer -1 produces only 6% cosine similarity with Qwen3-4B. "
                "Use -2 instead for Z-Image compatibility."
            )

        # Build chat template format matching Qwen3-4B (NO think block)
        # This matches the official Z-Image HF Space which uses enable_thinking=True
        formatted = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"

        logger.debug(f"Formatted prompt: {repr(formatted)}")

        result = self.extract(
            formatted,
            mode="full_sequence",
            hidden_layer=hidden_layer,
            normalize=False,
            instruction=None,
        )

        embeddings = result.embeddings
        if scale_factor != 1.0:
            embeddings = embeddings * scale_factor

        return embeddings

    def compare_with_qwen3_4b(
        self,
        text: str,
        qwen3_embeddings: torch.Tensor,
        hidden_layer: int = -2,
    ) -> dict:
        """
        Compare embeddings with Qwen3-4B for the same text.

        Useful for analyzing distribution differences between models.

        Args:
            text: Input text
            qwen3_embeddings: Embeddings from Qwen3-4B for same text
            hidden_layer: Layer to compare

        Returns:
            Dictionary with comparison statistics
        """
        result = self.extract(text, mode="full_sequence", hidden_layer=hidden_layer)
        emb_embeddings = result.embeddings

        # Handle potential length differences
        min_len = min(len(emb_embeddings), len(qwen3_embeddings))
        emb_truncated = emb_embeddings[:min_len]
        qwen3_truncated = qwen3_embeddings[:min_len]

        # Compute statistics
        cosine_sim = F.cosine_similarity(
            emb_truncated.flatten().unsqueeze(0),
            qwen3_truncated.flatten().unsqueeze(0),
        ).item()

        # Per-dimension correlation
        emb_flat = emb_truncated.reshape(-1, 2560)
        qwen3_flat = qwen3_truncated.reshape(-1, 2560)

        emb_mean = emb_flat.mean(dim=0)
        qwen3_mean = qwen3_flat.mean(dim=0)

        # Pearson correlation of per-dim means
        emb_centered = emb_mean - emb_mean.mean()
        qwen3_centered = qwen3_mean - qwen3_mean.mean()
        correlation = (
            (emb_centered * qwen3_centered).sum()
            / (emb_centered.norm() * qwen3_centered.norm())
        ).item()

        return {
            "embedding_tokens": len(emb_embeddings),
            "qwen3_tokens": len(qwen3_embeddings),
            "compared_tokens": min_len,
            "global_cosine_similarity": cosine_sim,
            "per_dim_mean_correlation": correlation,
            "embedding_mean": result.mean,
            "embedding_std": result.std,
            "qwen3_mean": qwen3_embeddings.mean().item(),
            "qwen3_std": qwen3_embeddings.std().item(),
            "std_ratio": result.std / qwen3_embeddings.std().item(),
        }
