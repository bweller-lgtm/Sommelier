"""Unified caching system for embeddings, quality scores, and API responses."""
import json
import pickle
import hashlib
from pathlib import Path
from typing import Any, Optional, Dict, Union, List
from datetime import datetime, timedelta
import numpy as np


class CacheKey:
    """Generate consistent cache keys for different data types."""

    @staticmethod
    def from_file(file_path: Path) -> str:
        """
        Generate cache key from file path, size, and modification time.

        Args:
            file_path: Path to the file.

        Returns:
            Cache key string.
        """
        try:
            stat = file_path.stat()
            key_str = f"{file_path}_{stat.st_size}_{stat.st_mtime}"
            return hashlib.sha256(key_str.encode()).hexdigest()
        except (OSError, FileNotFoundError):
            # File doesn't exist or is inaccessible, use path only
            return hashlib.sha256(str(file_path).encode()).hexdigest()

    @staticmethod
    def from_string(text: str) -> str:
        """
        Generate cache key from arbitrary string.

        Args:
            text: Text to hash.

        Returns:
            Cache key string.
        """
        return hashlib.sha256(text.encode()).hexdigest()

    @staticmethod
    def from_files(file_paths: List[Path]) -> str:
        """
        Generate cache key from multiple files.

        Args:
            file_paths: List of file paths.

        Returns:
            Cache key string.
        """
        keys = [CacheKey.from_file(fp) for fp in sorted(file_paths)]
        combined = "_".join(keys)
        return hashlib.sha256(combined.encode()).hexdigest()


class BaseCache:
    """Base class for cache implementations."""

    def __init__(self, cache_dir: Path, ttl_days: int = 365):
        """
        Initialize cache.

        Args:
            cache_dir: Directory to store cache files.
            ttl_days: Time-to-live in days (0 = never expire).
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_days = ttl_days

    def _is_expired(self, cache_file: Path) -> bool:
        """
        Check if cache file has expired.

        Args:
            cache_file: Path to cache file.

        Returns:
            True if expired, False otherwise.
        """
        if self.ttl_days <= 0:
            return False  # Never expire

        try:
            mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
            age = datetime.now() - mtime
            return age > timedelta(days=self.ttl_days)
        except (OSError, FileNotFoundError):
            return True

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        raise NotImplementedError

    def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        raise NotImplementedError

    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        raise NotImplementedError

    def clear(self) -> None:
        """Clear all cache entries."""
        for cache_file in self.cache_dir.iterdir():
            if cache_file.is_file():
                cache_file.unlink()

    def get_size(self) -> int:
        """Get total cache size in bytes."""
        total_size = 0
        for cache_file in self.cache_dir.rglob('*'):
            if cache_file.is_file():
                total_size += cache_file.stat().st_size
        return total_size


class PickleCache(BaseCache):
    """Cache for Python objects using pickle."""

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        cache_file = self.cache_dir / f"{key}.pkl"
        if not cache_file.exists() or self._is_expired(cache_file):
            return None
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except (pickle.PickleError, EOFError, OSError):
            return None

    def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        cache_file = self.cache_dir / f"{key}.pkl"
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
        except (pickle.PickleError, OSError) as e:
            # Log error but don't crash
            print(f"Warning: Failed to cache {key}: {e}")

    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        cache_file = self.cache_dir / f"{key}.pkl"
        return cache_file.exists() and not self._is_expired(cache_file)


class JSONCache(BaseCache):
    """Cache for JSON-serializable data."""

    def get(self, key: str) -> Optional[Dict]:
        """Get value from cache."""
        cache_file = self.cache_dir / f"{key}.json"
        if not cache_file.exists() or self._is_expired(cache_file):
            return None
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def set(self, key: str, value: Dict) -> None:
        """Set value in cache."""
        cache_file = self.cache_dir / f"{key}.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(value, f, indent=2)
        except (TypeError, OSError) as e:
            print(f"Warning: Failed to cache {key}: {e}")

    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        cache_file = self.cache_dir / f"{key}.json"
        return cache_file.exists() and not self._is_expired(cache_file)


class NumpyCache(BaseCache):
    """Cache for NumPy arrays."""

    def get(self, key: str) -> Optional[np.ndarray]:
        """Get array from cache."""
        cache_file = self.cache_dir / f"{key}.npy"
        if not cache_file.exists() or self._is_expired(cache_file):
            return None
        try:
            return np.load(cache_file)
        except (OSError, ValueError):
            return None

    def set(self, key: str, value: np.ndarray) -> None:
        """Set array in cache."""
        cache_file = self.cache_dir / f"{key}.npy"
        try:
            np.save(cache_file, value)
        except (OSError, ValueError) as e:
            print(f"Warning: Failed to cache {key}: {e}")

    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        cache_file = self.cache_dir / f"{key}.npy"
        return cache_file.exists() and not self._is_expired(cache_file)


class CacheManager:
    """
    Unified cache manager for all caching needs.

    Manages different cache types (embeddings, quality, features, etc.)
    with a consistent interface.
    """

    def __init__(
        self,
        cache_root: Path,
        ttl_days: int = 365,
        enabled: bool = True,
        max_size_gb: float = 10.0
    ):
        """
        Initialize cache manager.

        Args:
            cache_root: Root directory for all caches.
            ttl_days: Time-to-live for cache entries in days.
            enabled: Enable/disable caching globally.
            max_size_gb: Maximum cache size in GB (0 = unlimited).
        """
        self.cache_root = Path(cache_root)
        self.enabled = enabled
        self.ttl_days = ttl_days
        self.max_size_gb = max_size_gb

        # Initialize cache directories
        self.cache_dirs = {
            "embeddings": self.cache_root / "emb",
            "quality": self.cache_root / "quality",
            "features": self.cache_root / "features",
            "faces": self.cache_root / "faces",
            "phash": self.cache_root / "phash",
            "gemini": self.cache_root / "labeling_pairwise",
            "checkpoints": self.cache_root / "checkpoints",
        }

        # Create cache instances
        self.caches = {
            "embeddings": NumpyCache(self.cache_dirs["embeddings"], ttl_days),
            "quality": PickleCache(self.cache_dirs["quality"], ttl_days),
            "features": PickleCache(self.cache_dirs["features"], ttl_days),
            "faces": PickleCache(self.cache_dirs["faces"], ttl_days),
            "phash": PickleCache(self.cache_dirs["phash"], ttl_days),
            "gemini": JSONCache(self.cache_dirs["gemini"], ttl_days),
            "checkpoints": JSONCache(self.cache_dirs["checkpoints"], ttl_days),
        }

    def get(self, cache_type: str, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            cache_type: Type of cache (embeddings, quality, features, etc.).
            key: Cache key.

        Returns:
            Cached value or None if not found.
        """
        if not self.enabled:
            return None

        if cache_type not in self.caches:
            raise ValueError(f"Unknown cache type: {cache_type}. Available: {list(self.caches.keys())}")

        return self.caches[cache_type].get(key)

    def set(self, cache_type: str, key: str, value: Any) -> None:
        """
        Set value in cache.

        Args:
            cache_type: Type of cache (embeddings, quality, features, etc.).
            key: Cache key.
            value: Value to cache.
        """
        if not self.enabled:
            return

        if cache_type not in self.caches:
            raise ValueError(f"Unknown cache type: {cache_type}. Available: {list(self.caches.keys())}")

        self.caches[cache_type].set(key, value)

    def exists(self, cache_type: str, key: str) -> bool:
        """
        Check if key exists in cache.

        Args:
            cache_type: Type of cache (embeddings, quality, features, etc.).
            key: Cache key.

        Returns:
            True if key exists, False otherwise.
        """
        if not self.enabled:
            return False

        if cache_type not in self.caches:
            return False

        return self.caches[cache_type].exists(key)

    def clear(self, cache_type: Optional[str] = None) -> None:
        """
        Clear cache.

        Args:
            cache_type: Type of cache to clear. If None, clears all caches.
        """
        if cache_type is None:
            # Clear all caches
            for cache in self.caches.values():
                cache.clear()
        else:
            if cache_type not in self.caches:
                raise ValueError(f"Unknown cache type: {cache_type}")
            self.caches[cache_type].clear()

    def get_total_size(self) -> int:
        """Get total size of all caches in bytes."""
        total = 0
        for cache in self.caches.values():
            total += cache.get_size()
        return total

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        stats = {}
        for cache_type, cache in self.caches.items():
            stats[cache_type] = cache.get_size()
        stats["total"] = sum(stats.values())
        return stats

    def cleanup_expired(self) -> int:
        """
        Remove expired cache entries.

        Returns:
            Number of entries removed.
        """
        removed = 0
        for cache_type, cache in self.caches.items():
            for cache_file in cache.cache_dir.iterdir():
                if cache_file.is_file() and cache._is_expired(cache_file):
                    cache_file.unlink()
                    removed += 1
        return removed

    def enforce_size_limit(self) -> Dict[str, int]:
        """
        Enforce cache size limit by removing oldest entries.

        Returns:
            Dictionary with statistics: {'removed': count, 'size_before_mb': size, 'size_after_mb': size}
        """
        if self.max_size_gb <= 0:
            # Unlimited cache size
            return {"removed": 0, "size_before_mb": 0, "size_after_mb": 0}

        # Get current size
        current_size_bytes = self.get_total_size()
        current_size_gb = current_size_bytes / (1024 ** 3)

        stats = {
            "removed": 0,
            "size_before_mb": int(current_size_bytes / (1024 ** 2)),
            "size_after_mb": 0
        }

        if current_size_gb <= self.max_size_gb:
            # Within limit
            stats["size_after_mb"] = stats["size_before_mb"]
            return stats

        # Exceeded limit - need to cleanup
        print(f"⚠️  Cache size ({current_size_gb:.2f} GB) exceeds limit ({self.max_size_gb} GB)")
        print(f"   Removing oldest entries...")

        # Collect all cache files with their modification times
        all_files = []
        for cache_type, cache in self.caches.items():
            for cache_file in cache.cache_dir.rglob('*'):
                if cache_file.is_file():
                    try:
                        mtime = cache_file.stat().st_mtime
                        size = cache_file.stat().st_size
                        all_files.append((cache_file, mtime, size))
                    except (OSError, FileNotFoundError):
                        pass

        # Sort by modification time (oldest first)
        all_files.sort(key=lambda x: x[1])

        # Remove files until we're under the limit
        target_size_bytes = int(self.max_size_gb * 0.9 * (1024 ** 3))  # Target 90% of limit

        for file_path, mtime, file_size in all_files:
            if current_size_bytes <= target_size_bytes:
                break

            try:
                file_path.unlink()
                current_size_bytes -= file_size
                stats["removed"] += 1
            except (OSError, FileNotFoundError):
                pass

        stats["size_after_mb"] = int(current_size_bytes / (1024 ** 2))
        print(f"   Removed {stats['removed']} files")
        print(f"   Cache size: {stats['size_before_mb']} MB → {stats['size_after_mb']} MB")

        return stats

    def check_and_cleanup(self) -> Dict[str, int]:
        """
        Check cache size and cleanup if necessary.

        Removes expired entries first, then enforces size limit.

        Returns:
            Dictionary with cleanup statistics.
        """
        expired_removed = self.cleanup_expired()
        size_stats = self.enforce_size_limit()

        return {
            "expired_removed": expired_removed,
            **size_stats
        }
