"""
Embedding cache for text encoder outputs.

Caches encoded embeddings to avoid redundant encoding of identical prompts.
Useful when:
- Generating multiple images with the same prompt (different seeds)
- Iterating on generation parameters without re-encoding
- Web server handling repeated requests

Based on DiffSynth-Studio optimization recommendations.

Usage:
    from llm_dit.utils.embedding_cache import EmbeddingCache

    cache = EmbeddingCache(max_size=100)

    # Check cache first
    cached = cache.get(formatted_prompt)
    if cached is not None:
        return cached

    # Encode and cache
    output = backend.encode([formatted_prompt])
    cache.put(formatted_prompt, output)
"""

import hashlib
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import torch

from llm_dit.backends.protocol import EncodingOutput

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """Statistics for embedding cache performance."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    current_size: int = 0
    max_size: int = 0

    @property
    def hit_rate(self) -> float:
        """Return cache hit rate as a percentage."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return (self.hits / total) * 100


class EmbeddingCache:
    """
    Thread-safe LRU cache for text encoder embeddings.

    Stores embeddings on CPU to avoid GPU memory pressure. Embeddings are
    moved back to the target device when retrieved.

    Args:
        max_size: Maximum number of cached embeddings (default: 100)
        enabled: Whether caching is enabled (default: True)

    Example:
        cache = EmbeddingCache(max_size=100)

        # Try to get from cache
        key = cache.make_key(formatted_prompt, layer_index=-2)
        cached = cache.get(key)

        if cached is not None:
            # Move to target device and use
            return cached.to(device)

        # Cache miss - encode and store
        output = backend.encode([formatted_prompt])
        cache.put(key, output)
    """

    def __init__(self, max_size: int = 100, enabled: bool = True):
        self.max_size = max_size
        self.enabled = enabled
        self._cache: OrderedDict[str, EncodingOutput] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = CacheStats(max_size=max_size)

    @staticmethod
    def make_key(
        text: str,
        layer_index: int = -2,
        return_padded: bool = False,
    ) -> str:
        """
        Create a cache key from encoding parameters.

        Uses SHA256 hash of the text to handle long prompts efficiently.

        Args:
            text: The formatted prompt text
            layer_index: Which hidden layer was used
            return_padded: Whether padded output was requested

        Returns:
            Cache key string
        """
        # Hash the text to handle long prompts
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        return f"{text_hash}:L{layer_index}:P{int(return_padded)}"

    def get(
        self,
        key: str,
        device: Optional[torch.device] = None,
    ) -> Optional[EncodingOutput]:
        """
        Get cached embedding if available.

        Args:
            key: Cache key from make_key()
            device: Optional device to move embeddings to

        Returns:
            Cached EncodingOutput or None if not found
        """
        if not self.enabled:
            return None

        with self._lock:
            if key not in self._cache:
                self._stats.misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._stats.hits += 1

            cached = self._cache[key]
            logger.debug(f"Cache hit for key {key[:8]}... (hit rate: {self._stats.hit_rate:.1f}%)")

            # Clone and optionally move to device
            result = self._clone_output(cached)
            if device is not None:
                result = result.to(device)

            return result

    def put(self, key: str, output: EncodingOutput) -> None:
        """
        Store embedding in cache.

        Embeddings are stored as CPU copies to avoid GPU memory pressure.

        Args:
            key: Cache key from make_key()
            output: EncodingOutput to cache
        """
        if not self.enabled:
            return

        with self._lock:
            # Evict oldest if at capacity
            while len(self._cache) >= self.max_size:
                evicted_key, _ = self._cache.popitem(last=False)
                self._stats.evictions += 1
                logger.debug(f"Evicted cache entry {evicted_key[:8]}...")

            # Store CPU copy
            cpu_output = self._to_cpu(output)
            self._cache[key] = cpu_output
            self._stats.current_size = len(self._cache)

            logger.debug(
                f"Cached embedding {key[:8]}... "
                f"(size: {self._stats.current_size}/{self.max_size})"
            )

    def clear(self) -> None:
        """Clear all cached embeddings."""
        with self._lock:
            self._cache.clear()
            self._stats.current_size = 0
            logger.info("Embedding cache cleared")

    @property
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            self._stats.current_size = len(self._cache)
            return self._stats

    def _to_cpu(self, output: EncodingOutput) -> EncodingOutput:
        """Move all tensors to CPU, clone, and detach from computation graph."""
        return EncodingOutput(
            embeddings=[e.detach().clone().cpu() for e in output.embeddings],
            attention_masks=[m.detach().clone().cpu() for m in output.attention_masks],
            padded_embeddings=(
                output.padded_embeddings.detach().clone().cpu()
                if output.padded_embeddings is not None
                else None
            ),
            padded_mask=(
                output.padded_mask.detach().clone().cpu()
                if output.padded_mask is not None
                else None
            ),
            formatted_prompts=output.formatted_prompts,
            token_counts=output.token_counts,
        )

    def _clone_output(self, output: EncodingOutput) -> EncodingOutput:
        """Clone tensors to avoid modifying cached data."""
        return EncodingOutput(
            embeddings=[e.clone() for e in output.embeddings],
            attention_masks=[m.clone() for m in output.attention_masks],
            padded_embeddings=(
                output.padded_embeddings.clone()
                if output.padded_embeddings is not None
                else None
            ),
            padded_mask=(
                output.padded_mask.clone()
                if output.padded_mask is not None
                else None
            ),
            formatted_prompts=output.formatted_prompts,
            token_counts=output.token_counts,
        )


# Global cache instance (can be replaced or configured)
_global_cache: Optional[EmbeddingCache] = None


def get_embedding_cache() -> EmbeddingCache:
    """Get or create the global embedding cache."""
    global _global_cache
    if _global_cache is None:
        _global_cache = EmbeddingCache()
    return _global_cache


def set_embedding_cache(cache: EmbeddingCache) -> None:
    """Set the global embedding cache."""
    global _global_cache
    _global_cache = cache


def clear_embedding_cache() -> None:
    """Clear the global embedding cache."""
    global _global_cache
    if _global_cache is not None:
        _global_cache.clear()
