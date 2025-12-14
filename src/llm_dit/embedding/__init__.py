"""
Qwen3-Embedding model support for Z-Image.

This module provides the EmbeddingExtractor class for using Qwen3-Embedding-4B
as an alternative text encoder for Z-Image generation.

Key Insight:
    Qwen3-Embedding-4B has hidden_size=2560, matching Z-Image's requirement.
    It's specifically trained for embedding quality via contrastive learning,
    which may produce better semantic representations than the base Qwen3-4B.

Usage:
    >>> from llm_dit.embedding import EmbeddingExtractor
    >>> extractor = EmbeddingExtractor.from_pretrained("/path/to/Qwen3-Embedding-4B")
    >>> result = extractor.extract("A cat sleeping in sunlight")
    >>> print(result.embeddings.shape)  # (seq_len, 2560)
"""

from .qwen3_embedding import EmbeddingExtractor, EmbeddingExtractionResult

__all__ = ["EmbeddingExtractor", "EmbeddingExtractionResult"]
