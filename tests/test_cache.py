"""Tests for cache management."""
import pytest
import numpy as np
from pathlib import Path
from src.core.cache import CacheKey, CacheManager, PickleCache, JSONCache, NumpyCache


class TestCacheKey:
    """Tests for CacheKey generation."""

    def test_from_string(self):
        """Test cache key from string."""
        key1 = CacheKey.from_string("test")
        key2 = CacheKey.from_string("test")
        key3 = CacheKey.from_string("different")

        assert key1 == key2
        assert key1 != key3
        assert len(key1) == 64  # SHA-256 hash length

    def test_from_file(self, tmp_path):
        """Test cache key from file."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        key1 = CacheKey.from_file(test_file)
        key2 = CacheKey.from_file(test_file)

        assert key1 == key2
        assert len(key1) == 64  # SHA-256 hash length

        # Modify file - key should change
        test_file.write_text("different")
        key3 = CacheKey.from_file(test_file)
        assert key1 != key3

    def test_from_nonexistent_file(self, tmp_path):
        """Test cache key from nonexistent file."""
        nonexistent = tmp_path / "nonexistent.txt"
        key = CacheKey.from_file(nonexistent)
        assert len(key) == 64  # SHA-256 hash length - should still work

    def test_from_files(self, tmp_path):
        """Test cache key from multiple files."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("content1")
        file2.write_text("content2")

        key1 = CacheKey.from_files([file1, file2])
        key2 = CacheKey.from_files([file1, file2])
        key3 = CacheKey.from_files([file2, file1])  # Different order

        assert key1 == key2
        assert key1 == key3  # Should be same (sorted)


class TestPickleCache:
    """Tests for PickleCache."""

    def test_set_get(self, tmp_path):
        """Test setting and getting values."""
        cache = PickleCache(tmp_path / "cache", ttl_days=0)

        # Set value
        test_data = {"key": "value", "number": 42}
        cache.set("test_key", test_data)

        # Get value
        retrieved = cache.get("test_key")
        assert retrieved == test_data

    def test_get_nonexistent(self, tmp_path):
        """Test getting non-existent key."""
        cache = PickleCache(tmp_path / "cache", ttl_days=0)
        assert cache.get("nonexistent") is None

    def test_exists(self, tmp_path):
        """Test exists check."""
        cache = PickleCache(tmp_path / "cache", ttl_days=0)

        assert not cache.exists("test_key")

        cache.set("test_key", {"data": "value"})
        assert cache.exists("test_key")

    def test_clear(self, tmp_path):
        """Test clearing cache."""
        cache = PickleCache(tmp_path / "cache", ttl_days=0)

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        assert cache.exists("key1")
        assert cache.exists("key2")

        cache.clear()

        assert not cache.exists("key1")
        assert not cache.exists("key2")


class TestJSONCache:
    """Tests for JSONCache."""

    def test_set_get(self, tmp_path):
        """Test setting and getting JSON."""
        cache = JSONCache(tmp_path / "cache", ttl_days=0)

        test_data = {
            "classification": "Share",
            "confidence": 0.85,
            "reasoning": "Great photo"
        }

        cache.set("test_key", test_data)
        retrieved = cache.get("test_key")

        assert retrieved == test_data
        assert retrieved["confidence"] == 0.85


class TestNumpyCache:
    """Tests for NumpyCache."""

    def test_set_get(self, tmp_path):
        """Test setting and getting arrays."""
        cache = NumpyCache(tmp_path / "cache", ttl_days=0)

        test_array = np.array([[1, 2, 3], [4, 5, 6]])
        cache.set("test_key", test_array)

        retrieved = cache.get("test_key")
        assert np.array_equal(retrieved, test_array)


class TestCacheManager:
    """Tests for CacheManager."""

    def test_initialization(self, tmp_path):
        """Test cache manager initialization."""
        manager = CacheManager(tmp_path / "cache_root", enabled=True)

        assert manager.enabled
        assert manager.cache_root == tmp_path / "cache_root"
        assert "embeddings" in manager.caches
        assert "quality" in manager.caches
        assert "gemini" in manager.caches

    def test_set_get(self, tmp_path):
        """Test setting and getting from different cache types."""
        manager = CacheManager(tmp_path / "cache_root", enabled=True)

        # Test pickle cache (quality)
        manager.set("quality", "key1", {"score": 0.95})
        assert manager.get("quality", "key1") == {"score": 0.95}

        # Test JSON cache (gemini)
        manager.set("gemini", "key2", {"classification": "Share"})
        assert manager.get("gemini", "key2")["classification"] == "Share"

        # Test numpy cache (embeddings)
        embedding = np.random.rand(512)
        manager.set("embeddings", "key3", embedding)
        retrieved = manager.get("embeddings", "key3")
        assert np.array_equal(retrieved, embedding)

    def test_disabled_cache(self, tmp_path):
        """Test cache manager with caching disabled."""
        manager = CacheManager(tmp_path / "cache_root", enabled=False)

        manager.set("quality", "key1", {"score": 0.95})
        assert manager.get("quality", "key1") is None

    def test_exists(self, tmp_path):
        """Test exists check."""
        manager = CacheManager(tmp_path / "cache_root", enabled=True)

        assert not manager.exists("quality", "key1")

        manager.set("quality", "key1", {"score": 0.95})
        assert manager.exists("quality", "key1")

    def test_clear_specific(self, tmp_path):
        """Test clearing specific cache type."""
        manager = CacheManager(tmp_path / "cache_root", enabled=True)

        manager.set("quality", "key1", {"score": 0.95})
        manager.set("gemini", "key2", {"classification": "Share"})

        manager.clear("quality")

        assert not manager.exists("quality", "key1")
        assert manager.exists("gemini", "key2")

    def test_clear_all(self, tmp_path):
        """Test clearing all caches."""
        manager = CacheManager(tmp_path / "cache_root", enabled=True)

        manager.set("quality", "key1", {"score": 0.95})
        manager.set("gemini", "key2", {"classification": "Share"})

        manager.clear()

        assert not manager.exists("quality", "key1")
        assert not manager.exists("gemini", "key2")

    def test_invalid_cache_type(self, tmp_path):
        """Test using invalid cache type."""
        manager = CacheManager(tmp_path / "cache_root", enabled=True)

        with pytest.raises(ValueError, match="Unknown cache type"):
            manager.get("invalid_type", "key1")

        with pytest.raises(ValueError, match="Unknown cache type"):
            manager.set("invalid_type", "key1", "value")

    def test_stats(self, tmp_path):
        """Test getting cache statistics."""
        manager = CacheManager(tmp_path / "cache_root", enabled=True)

        # Add some data
        manager.set("quality", "key1", {"score": 0.95})
        manager.set("gemini", "key2", {"classification": "Share"})

        stats = manager.get_stats()
        assert "quality" in stats
        assert "gemini" in stats
        assert "total" in stats
        assert stats["total"] > 0
