"""
Unit tests for embedding cache functionality.

Tests the EmbeddingCache class including:
- Basic put/get operations
- LRU eviction behavior
- Cache statistics
- Thread safety
- Integration with TransformersBackend
"""

import threading
import time

import pytest
import torch

from llm_dit.backends.protocol import EncodingOutput
from llm_dit.utils.embedding_cache import (
    CacheStats,
    EmbeddingCache,
    clear_embedding_cache,
    get_embedding_cache,
    set_embedding_cache,
)


class TestEmbeddingCache:
    """Tests for EmbeddingCache class."""

    def test_init_default(self):
        """Test default initialization."""
        cache = EmbeddingCache()
        assert cache.max_size == 100
        assert cache.enabled is True
        assert len(cache._cache) == 0

    def test_init_custom(self):
        """Test custom initialization."""
        cache = EmbeddingCache(max_size=50, enabled=False)
        assert cache.max_size == 50
        assert cache.enabled is False

    def test_make_key(self):
        """Test cache key generation."""
        key1 = EmbeddingCache.make_key("Hello world", layer_index=-2)
        key2 = EmbeddingCache.make_key("Hello world", layer_index=-2)
        key3 = EmbeddingCache.make_key("Hello world", layer_index=-1)
        key4 = EmbeddingCache.make_key("Different text", layer_index=-2)

        # Same inputs should produce same key
        assert key1 == key2
        # Different layer index should produce different key
        assert key1 != key3
        # Different text should produce different key
        assert key1 != key4

    def test_make_key_with_return_padded(self):
        """Test that return_padded affects key."""
        key1 = EmbeddingCache.make_key("text", layer_index=-2, return_padded=False)
        key2 = EmbeddingCache.make_key("text", layer_index=-2, return_padded=True)
        assert key1 != key2

    def test_put_and_get(self):
        """Test basic put and get operations."""
        cache = EmbeddingCache(max_size=10)

        # Create mock encoding output
        output = EncodingOutput(
            embeddings=[torch.randn(10, 2560)],
            attention_masks=[torch.ones(10, dtype=torch.bool)],
            token_counts=[10],
        )

        key = cache.make_key("test prompt", layer_index=-2)
        cache.put(key, output)

        # Get should return equivalent output
        cached = cache.get(key)
        assert cached is not None
        assert cached.embeddings[0].shape == output.embeddings[0].shape
        assert cached.token_counts == output.token_counts

    def test_get_miss(self):
        """Test cache miss returns None."""
        cache = EmbeddingCache(max_size=10)
        key = cache.make_key("nonexistent", layer_index=-2)
        assert cache.get(key) is None

    def test_get_disabled(self):
        """Test get returns None when cache is disabled."""
        cache = EmbeddingCache(max_size=10, enabled=False)

        output = EncodingOutput(
            embeddings=[torch.randn(10, 2560)],
            attention_masks=[torch.ones(10, dtype=torch.bool)],
        )

        key = cache.make_key("test", layer_index=-2)
        cache.put(key, output)

        # Should return None even if item was "stored"
        assert cache.get(key) is None

    def test_put_disabled(self):
        """Test put is no-op when cache is disabled."""
        cache = EmbeddingCache(max_size=10, enabled=False)

        output = EncodingOutput(
            embeddings=[torch.randn(10, 2560)],
            attention_masks=[torch.ones(10, dtype=torch.bool)],
        )

        key = cache.make_key("test", layer_index=-2)
        cache.put(key, output)

        # Re-enable and check nothing was stored
        cache.enabled = True
        assert cache.get(key) is None

    def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = EmbeddingCache(max_size=3)

        # Add 3 items
        for i in range(3):
            output = EncodingOutput(
                embeddings=[torch.randn(10, 2560)],
                attention_masks=[torch.ones(10, dtype=torch.bool)],
            )
            key = cache.make_key(f"prompt {i}", layer_index=-2)
            cache.put(key, output)

        # Access prompt 0 to make it recently used
        key0 = cache.make_key("prompt 0", layer_index=-2)
        cache.get(key0)

        # Add a 4th item - should evict prompt 1 (oldest unused)
        output = EncodingOutput(
            embeddings=[torch.randn(10, 2560)],
            attention_masks=[torch.ones(10, dtype=torch.bool)],
        )
        key3 = cache.make_key("prompt 3", layer_index=-2)
        cache.put(key3, output)

        # prompt 0 should still be there (was accessed)
        assert cache.get(key0) is not None
        # prompt 1 should be evicted (oldest unused)
        key1 = cache.make_key("prompt 1", layer_index=-2)
        assert cache.get(key1) is None
        # prompt 2 and 3 should be there
        key2 = cache.make_key("prompt 2", layer_index=-2)
        assert cache.get(key2) is not None
        assert cache.get(key3) is not None

    def test_clear(self):
        """Test cache clear."""
        cache = EmbeddingCache(max_size=10)

        # Add items
        for i in range(5):
            output = EncodingOutput(
                embeddings=[torch.randn(10, 2560)],
                attention_masks=[torch.ones(10, dtype=torch.bool)],
            )
            key = cache.make_key(f"prompt {i}", layer_index=-2)
            cache.put(key, output)

        assert len(cache._cache) == 5

        cache.clear()
        assert len(cache._cache) == 0

    def test_get_with_device(self):
        """Test get moves tensors to specified device."""
        cache = EmbeddingCache(max_size=10)

        # Create output on CPU
        output = EncodingOutput(
            embeddings=[torch.randn(10, 2560)],
            attention_masks=[torch.ones(10, dtype=torch.bool)],
        )

        key = cache.make_key("test", layer_index=-2)
        cache.put(key, output)

        # Get with CPU device explicitly
        cached = cache.get(key, device=torch.device("cpu"))
        assert cached is not None
        assert cached.embeddings[0].device.type == "cpu"


class TestCacheStats:
    """Tests for cache statistics."""

    def test_initial_stats(self):
        """Test initial stats are zero."""
        cache = EmbeddingCache(max_size=10)
        stats = cache.stats
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0
        assert stats.current_size == 0

    def test_hit_tracking(self):
        """Test hit counter."""
        cache = EmbeddingCache(max_size=10)

        output = EncodingOutput(
            embeddings=[torch.randn(10, 2560)],
            attention_masks=[torch.ones(10, dtype=torch.bool)],
        )

        key = cache.make_key("test", layer_index=-2)
        cache.put(key, output)

        # Get twice
        cache.get(key)
        cache.get(key)

        stats = cache.stats
        assert stats.hits == 2
        assert stats.misses == 0

    def test_miss_tracking(self):
        """Test miss counter."""
        cache = EmbeddingCache(max_size=10)

        # Try to get non-existent items
        cache.get(cache.make_key("missing1", layer_index=-2))
        cache.get(cache.make_key("missing2", layer_index=-2))

        stats = cache.stats
        assert stats.hits == 0
        assert stats.misses == 2

    def test_eviction_tracking(self):
        """Test eviction counter."""
        cache = EmbeddingCache(max_size=2)

        # Fill cache
        for i in range(2):
            output = EncodingOutput(
                embeddings=[torch.randn(10, 2560)],
                attention_masks=[torch.ones(10, dtype=torch.bool)],
            )
            cache.put(cache.make_key(f"prompt {i}", layer_index=-2), output)

        assert cache.stats.evictions == 0

        # Add one more, triggering eviction
        output = EncodingOutput(
            embeddings=[torch.randn(10, 2560)],
            attention_masks=[torch.ones(10, dtype=torch.bool)],
        )
        cache.put(cache.make_key("prompt 2", layer_index=-2), output)

        assert cache.stats.evictions == 1

    def test_hit_rate(self):
        """Test hit rate calculation."""
        stats = CacheStats(hits=3, misses=7)
        assert stats.hit_rate == 30.0

        stats = CacheStats(hits=0, misses=0)
        assert stats.hit_rate == 0.0


class TestCacheThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_access(self):
        """Test concurrent reads and writes."""
        cache = EmbeddingCache(max_size=100)
        errors = []

        def writer(thread_id: int):
            """Write items to cache."""
            try:
                for i in range(50):
                    output = EncodingOutput(
                        embeddings=[torch.randn(10, 2560)],
                        attention_masks=[torch.ones(10, dtype=torch.bool)],
                    )
                    key = cache.make_key(f"thread{thread_id}_prompt{i}", layer_index=-2)
                    cache.put(key, output)
            except Exception as e:
                errors.append(e)

        def reader(thread_id: int):
            """Read items from cache."""
            try:
                for i in range(50):
                    key = cache.make_key(f"thread{thread_id}_prompt{i}", layer_index=-2)
                    cache.get(key)
            except Exception as e:
                errors.append(e)

        # Start multiple writer and reader threads
        threads = []
        for i in range(4):
            t = threading.Thread(target=writer, args=(i,))
            threads.append(t)
            t = threading.Thread(target=reader, args=(i,))
            threads.append(t)

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # Should complete without errors
        assert len(errors) == 0


class TestGlobalCache:
    """Tests for global cache functions."""

    def test_get_embedding_cache(self):
        """Test getting global cache."""
        # Reset global cache
        set_embedding_cache(None)

        cache1 = get_embedding_cache()
        cache2 = get_embedding_cache()

        # Should return the same instance
        assert cache1 is cache2

    def test_set_embedding_cache(self):
        """Test setting global cache."""
        custom_cache = EmbeddingCache(max_size=50)
        set_embedding_cache(custom_cache)

        retrieved = get_embedding_cache()
        assert retrieved is custom_cache
        assert retrieved.max_size == 50

    def test_clear_embedding_cache(self):
        """Test clearing global cache."""
        cache = get_embedding_cache()

        output = EncodingOutput(
            embeddings=[torch.randn(10, 2560)],
            attention_masks=[torch.ones(10, dtype=torch.bool)],
        )

        key = cache.make_key("test", layer_index=-2)
        cache.put(key, output)

        assert len(cache._cache) > 0

        clear_embedding_cache()
        assert len(cache._cache) == 0


class TestCacheDataIntegrity:
    """Tests for data integrity in cache."""

    def test_cached_tensors_are_clones(self):
        """Test that cached tensors are independent copies."""
        cache = EmbeddingCache(max_size=10)

        original_tensor = torch.randn(10, 2560)
        output = EncodingOutput(
            embeddings=[original_tensor],
            attention_masks=[torch.ones(10, dtype=torch.bool)],
        )

        key = cache.make_key("test", layer_index=-2)
        cache.put(key, output)

        # Modify original tensor
        original_tensor.fill_(0)

        # Cached version should be unchanged
        cached = cache.get(key)
        assert cached is not None
        assert not torch.all(cached.embeddings[0] == 0)

    def test_retrieved_tensors_are_clones(self):
        """Test that retrieved tensors don't affect cache."""
        cache = EmbeddingCache(max_size=10)

        output = EncodingOutput(
            embeddings=[torch.randn(10, 2560)],
            attention_masks=[torch.ones(10, dtype=torch.bool)],
        )

        key = cache.make_key("test", layer_index=-2)
        cache.put(key, output)

        # Get and modify
        cached1 = cache.get(key)
        cached1.embeddings[0].fill_(999)

        # Get again - should be original values
        cached2 = cache.get(key)
        assert not torch.all(cached2.embeddings[0] == 999)

    def test_stores_on_cpu(self):
        """Test that cached data is stored on CPU."""
        cache = EmbeddingCache(max_size=10)

        # Create tensor (on CPU by default)
        output = EncodingOutput(
            embeddings=[torch.randn(10, 2560)],
            attention_masks=[torch.ones(10, dtype=torch.bool)],
        )

        key = cache.make_key("test", layer_index=-2)
        cache.put(key, output)

        # Verify internal storage is on CPU
        stored = cache._cache[key]
        assert stored.embeddings[0].device.type == "cpu"
