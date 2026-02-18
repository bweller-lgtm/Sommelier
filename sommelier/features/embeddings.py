"""CLIP embedding extraction for images."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict, TYPE_CHECKING
from tqdm import tqdm
from PIL import Image

from ..compat import require

if TYPE_CHECKING:
    import numpy as np
from ..core.cache import CacheManager, CacheKey
from ..core.config import ModelConfig, PerformanceConfig
from ..core.file_utils import ImageUtils


class EmbeddingExtractor:
    """
    Extract CLIP embeddings for images.

    Uses OpenCLIP for efficient embedding extraction with caching.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        performance_config: PerformanceConfig,
        cache_manager: Optional[CacheManager] = None
    ):
        """
        Initialize embedding extractor.

        Args:
            model_config: Model configuration.
            performance_config: Performance configuration.
            cache_manager: Cache manager for caching embeddings.
        """
        self.model_config = model_config
        self.performance_config = performance_config
        self.cache_manager = cache_manager

        # Lazy loading of model
        self._model = None
        self._preprocess = None
        self._device = None

    def _load_model(self):
        """Lazy load CLIP model."""
        if self._model is None:
            torch = require("torch", "ml")
            open_clip = require("open_clip", "ml")
            self._torch = torch
            self._np = require("numpy", "ml")

            print(f"📦 Loading CLIP model: {self.model_config.clip_model}")

            self._device = self.performance_config.device
            if self._device == "cuda" and not torch.cuda.is_available():
                print("⚠️  CUDA not available, falling back to CPU")
                self._device = "cpu"

            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                self.model_config.clip_model,
                pretrained=self.model_config.clip_pretrained,
                device=self._device
            )
            self._model.eval()

            print(f"✅ Model loaded on {self._device}")

    def extract_embedding(
        self,
        image_path: Path,
        use_cache: bool = True
    ) -> Optional[np.ndarray]:
        """
        Extract embedding for single image.

        Args:
            image_path: Path to image.
            use_cache: Whether to use cached embeddings.

        Returns:
            Normalized embedding vector or None if extraction fails.
        """
        # Try cache first
        if use_cache and self.cache_manager is not None:
            cache_key = CacheKey.from_file(image_path)
            cached_embedding = self.cache_manager.get("embeddings", cache_key)
            if cached_embedding is not None:
                return cached_embedding

        # Extract embedding
        embedding = self._extract_embedding_nocache(image_path)

        # Cache result
        if use_cache and self.cache_manager is not None and embedding is not None:
            cache_key = CacheKey.from_file(image_path)
            self.cache_manager.set("embeddings", cache_key, embedding)

        return embedding

    def _extract_embedding_nocache(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Extract embedding without caching.

        Args:
            image_path: Path to image.

        Returns:
            Normalized embedding vector or None if extraction fails.
        """
        try:
            # Ensure model is loaded
            self._load_model()

            # Load and preprocess image
            img = ImageUtils.load_and_fix_orientation(image_path, max_size=512)
            if img is None:
                return None

            img = ImageUtils.ensure_rgb(img)

            # Apply CLIP preprocessing
            img_tensor = self._preprocess(img).unsqueeze(0).to(self._device)

            # Extract embedding
            with self._torch.no_grad():
                embedding = self._model.encode_image(img_tensor)
                # Normalize
                embedding = self._torch.nn.functional.normalize(embedding, dim=-1)
                # Convert to numpy
                embedding = embedding.cpu().numpy()[0]

            return embedding

        except Exception as e:
            print(f"⚠️  Warning: Failed to extract embedding for {image_path}: {e}")
            return None

    def extract_embeddings_batch(
        self,
        image_paths: List[Path],
        use_cache: bool = True,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Extract embeddings for batch of images.

        Args:
            image_paths: List of image paths.
            use_cache: Whether to use cached embeddings.
            show_progress: Whether to show progress bar.

        Returns:
            Array of embeddings, shape [N, D]. Failed extractions get zero vectors.
        """
        # Ensure model is loaded
        self._load_model()

        embeddings = []
        batch_size = self.performance_config.embedding_batch_size

        iterator = range(0, len(image_paths), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Extracting embeddings", total=len(image_paths)//batch_size + 1)

        for batch_start in iterator:
            batch_end = min(batch_start + batch_size, len(image_paths))
            batch_paths = image_paths[batch_start:batch_end]

            batch_embeddings = self._extract_batch(batch_paths, use_cache)
            embeddings.extend(batch_embeddings)

        return self._np.array(embeddings)

    def _extract_batch(
        self,
        batch_paths: List[Path],
        use_cache: bool
    ) -> List[np.ndarray]:
        """
        Extract embeddings for a batch.

        Args:
            batch_paths: List of image paths in batch.
            use_cache: Whether to use cache.

        Returns:
            List of embedding arrays.
        """
        batch_embeddings = []

        # Check cache for all images
        cache_hits = {}
        paths_to_compute = []

        if use_cache and self.cache_manager is not None:
            for path in batch_paths:
                cache_key = CacheKey.from_file(path)
                cached = self.cache_manager.get("embeddings", cache_key)
                if cached is not None:
                    cache_hits[path] = cached
                else:
                    paths_to_compute.append(path)
        else:
            paths_to_compute = batch_paths

        # Compute embeddings for cache misses
        computed_embeddings = {}
        if paths_to_compute:
            computed_embeddings = self._compute_embeddings_batch(paths_to_compute)

            # Cache new embeddings
            if use_cache and self.cache_manager is not None:
                for path, embedding in computed_embeddings.items():
                    cache_key = CacheKey.from_file(path)
                    self.cache_manager.set("embeddings", cache_key, embedding)

        # Combine cached and computed embeddings in original order
        for path in batch_paths:
            if path in cache_hits:
                batch_embeddings.append(cache_hits[path])
            elif path in computed_embeddings:
                batch_embeddings.append(computed_embeddings[path])
            else:
                # Failed - use zero vector
                embedding_dim = 512  # Default for ViT-B-32
                batch_embeddings.append(self._np.zeros(embedding_dim))

        return batch_embeddings

    def _compute_embeddings_batch(
        self,
        image_paths: List[Path]
    ) -> Dict[Path, np.ndarray]:
        """
        Compute embeddings for batch of images (no caching).

        Args:
            image_paths: List of image paths.

        Returns:
            Dictionary mapping paths to embeddings.
        """
        embeddings = {}

        # Load and preprocess images
        images = []
        valid_paths = []

        for path in image_paths:
            img = ImageUtils.load_and_fix_orientation(path, max_size=512)
            if img is not None:
                img = ImageUtils.ensure_rgb(img)
                images.append(img)
                valid_paths.append(path)

        if not images:
            return {}

        try:
            # Preprocess batch
            img_tensors = self._torch.stack([self._preprocess(img) for img in images]).to(self._device)

            # Extract embeddings
            with self._torch.no_grad():
                batch_embeddings = self._model.encode_image(img_tensors)
                # Normalize
                batch_embeddings = self._torch.nn.functional.normalize(batch_embeddings, dim=-1)
                # Convert to numpy
                batch_embeddings = batch_embeddings.cpu().numpy()

            # Map back to paths
            for path, embedding in zip(valid_paths, batch_embeddings):
                embeddings[path] = embedding

        except Exception as e:
            print(f"⚠️  Batch embedding extraction failed: {e}")
            # Fall back to individual extraction
            for path in valid_paths:
                embedding = self._extract_embedding_nocache(path)
                if embedding is not None:
                    embeddings[path] = embedding

        return embeddings

    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding.
            embedding2: Second embedding.

        Returns:
            Cosine similarity (0 to 1).
        """
        import numpy
        return float(numpy.dot(embedding1, embedding2))

    def compute_similarity_matrix(
        self,
        embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute pairwise similarity matrix.

        Args:
            embeddings: Array of embeddings, shape [N, D].

        Returns:
            Similarity matrix, shape [N, N].
        """
        # embeddings are already normalized, so dot product = cosine similarity
        import numpy
        return numpy.dot(embeddings, embeddings.T)
