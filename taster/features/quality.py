"""Quality scoring and face detection for images."""
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

from ..core.cache import CacheManager, CacheKey
from ..core.config import QualityConfig


# Module-level wrapper functions for multiprocessing (Windows compatibility)
def _compute_quality_worker(path: Path, scorer: 'QualityScorer', use_cache: bool) -> Tuple[Path, float]:
    """Worker function for parallel quality computation."""
    return (path, scorer.compute_score(path, use_cache=use_cache))


def _detect_faces_worker(path: Path, detector: 'FaceDetector', use_cache: bool) -> Tuple[Path, Dict[str, float]]:
    """Worker function for parallel face detection."""
    return (path, detector.detect_faces(path, use_cache=use_cache))


class QualityScorer:
    """
    Unified quality scorer for images.

    Computes technical quality metrics including:
    - Sharpness (Laplacian variance)
    - Brightness
    - Overall quality score (weighted combination)
    """

    def __init__(
        self,
        config: QualityConfig,
        cache_manager: Optional[CacheManager] = None
    ):
        """
        Initialize quality scorer.

        Args:
            config: Quality configuration.
            cache_manager: Cache manager for caching scores.
        """
        self.config = config
        self.cache_manager = cache_manager

    def compute_score(self, image_path: Path, use_cache: bool = True) -> float:
        """
        Compute quality score for image.

        Args:
            image_path: Path to image.
            use_cache: Whether to use cached scores.

        Returns:
            Quality score (0.0 to 1.0).
        """
        # Try cache first
        if use_cache and self.cache_manager is not None:
            cache_key = CacheKey.from_file(image_path)
            cached_score = self.cache_manager.get("quality", cache_key)
            if cached_score is not None:
                return float(cached_score)

        # Compute score
        score = self._compute_score_nocache(image_path)

        # Cache result
        if use_cache and self.cache_manager is not None:
            cache_key = CacheKey.from_file(image_path)
            self.cache_manager.set("quality", cache_key, score)

        return score

    def _compute_score_nocache(self, image_path: Path) -> float:
        """
        Compute quality score without caching.

        Args:
            image_path: Path to image.

        Returns:
            Quality score (0.0 to 1.0).
        """
        try:
            # Load image
            with Image.open(image_path) as im:
                im_gray = im.convert("L")
                arr_gray = np.array(im_gray)

            # Compute sharpness (Laplacian variance)
            sharpness = float(cv2.Laplacian(arr_gray, cv2.CV_64F).var())

            # Compute brightness (mean pixel value, normalized)
            brightness = float(arr_gray.mean() / 255.0)

            # Combine into overall score
            # Sharpness: normalize by threshold (200.0), cap at 1.0
            # Brightness: already normalized 0-1
            sharpness_score = min(sharpness / self.config.sharpness_threshold, 1.0)
            score = (
                self.config.sharpness_weight * sharpness_score +
                self.config.brightness_weight * brightness
            )

            return float(score)

        except Exception as e:
            print(f"⚠️  Warning: Failed to compute quality for {image_path}: {e}")
            return 0.0

    def compute_scores_batch(
        self,
        image_paths: List[Path],
        use_multiprocessing: bool = True,
        workers: int = 4,
        use_cache: bool = True
    ) -> Dict[Path, float]:
        """
        Compute quality scores for batch of images.

        Args:
            image_paths: List of image paths.
            use_multiprocessing: Whether to use multiprocessing.
            workers: Number of worker processes.
            use_cache: Whether to use cached scores.

        Returns:
            Dictionary mapping paths to quality scores.
        """
        if not use_multiprocessing or workers <= 1:
            # Serial processing
            results = {}
            for path in tqdm(image_paths, desc="Computing quality scores"):
                results[path] = self.compute_score(path, use_cache=use_cache)
            return results

        # Parallel processing
        try:
            # Use module-level function with partial for Windows compatibility
            worker_func = partial(_compute_quality_worker, scorer=self, use_cache=use_cache)

            with Pool(workers) as pool:
                results = list(tqdm(
                    pool.imap(worker_func, image_paths, chunksize=50),
                    total=len(image_paths),
                    desc=f"Computing quality scores ({workers} workers)"
                ))

            return {path: score for path, score in results}

        except Exception as e:
            print(f"⚠️  Multiprocessing failed ({e}), falling back to serial...")
            results = {}
            for path in tqdm(image_paths, desc="Computing quality scores (serial)"):
                results[path] = self.compute_score(path, use_cache=use_cache)
            return results

    def filter_by_quality(
        self,
        image_paths: List[Path],
        quality_scores: Dict[Path, float],
        threshold: Optional[float] = None
    ) -> Tuple[List[Path], List[Path]]:
        """
        Filter images by quality threshold.

        Args:
            image_paths: List of image paths.
            quality_scores: Dictionary of quality scores.
            threshold: Quality threshold. If None, uses config threshold.

        Returns:
            Tuple of (passing_paths, failing_paths).
        """
        if threshold is None:
            threshold = self.config.quality_filter_threshold

        passing = []
        failing = []

        for path in image_paths:
            score = quality_scores.get(path, 0.0)
            if score >= threshold:
                passing.append(path)
            else:
                failing.append(path)

        return passing, failing


class FaceDetector:
    """
    Face detection and feature extraction.

    Uses OpenCV Haar cascades for fast face detection.
    """

    def __init__(
        self,
        cache_manager: Optional[CacheManager] = None,
        max_faces: int = 10
    ):
        """
        Initialize face detector.

        Args:
            cache_manager: Cache manager for caching results.
            max_faces: Maximum number of faces to report (default: 10).
        """
        self.cache_manager = cache_manager
        self.max_faces = max_faces

        # Load cascades
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml'
            )
            self.smile_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_smile.xml'
            )
        except Exception as e:
            print(f"⚠️  Warning: Failed to load face detection cascades: {e}")
            self.face_cascade = None

    def detect_faces(self, image_path: Path, use_cache: bool = True) -> Dict[str, float]:
        """
        Detect faces and extract features.

        Args:
            image_path: Path to image.
            use_cache: Whether to use cached results.

        Returns:
            Dictionary of face features:
            - num_faces: Number of faces detected
            - face_ratio: Ratio of total face area to image area
            - largest_face_ratio: Ratio of largest face to image area
            - has_face: Binary indicator (0 or 1)
            - avg_face_size: Average face size ratio
            - eyes_detected: Number of faces with both eyes visible
            - smile_detected: Number of faces smiling
        """
        # Try cache first
        if use_cache and self.cache_manager is not None:
            cache_key = CacheKey.from_file(image_path)
            cached_features = self.cache_manager.get("faces", cache_key)
            if cached_features is not None:
                return cached_features

        # Compute features
        features = self._detect_faces_nocache(image_path)

        # Cache result
        if use_cache and self.cache_manager is not None:
            cache_key = CacheKey.from_file(image_path)
            self.cache_manager.set("faces", cache_key, features)

        return features

    def _detect_faces_nocache(self, image_path: Path) -> Dict[str, float]:
        """
        Detect faces without caching.

        Args:
            image_path: Path to image.

        Returns:
            Dictionary of face features.
        """
        # Default features (no faces)
        default_features = {
            "num_faces": 0,
            "face_ratio": 0.0,
            "largest_face_ratio": 0.0,
            "has_face": 0,
            "avg_face_size": 0.0,
            "eyes_detected": 0,
            "smile_detected": 0
        }

        if self.face_cascade is None:
            return default_features

        try:
            # Load image
            img = cv2.imread(str(image_path))
            if img is None:
                return default_features

            height, width = img.shape[:2]
            image_area = width * height

            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                img,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            if len(faces) == 0:
                return default_features

            # Calculate face features
            total_face_area = 0
            largest_face_area = 0
            eyes_count = 0
            smile_count = 0

            for (x, y, w, h) in faces:
                face_area = w * h
                total_face_area += face_area
                largest_face_area = max(largest_face_area, face_area)

                # Detect eyes in face region
                roi_gray = cv2.cvtColor(img[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
                eyes = self.eye_cascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=1.1,
                    minNeighbors=5
                )
                if len(eyes) >= 2:  # Both eyes detected
                    eyes_count += 1

                # Detect smile in lower half of face
                roi_smile = roi_gray[h//2:, :]
                smiles = self.smile_cascade.detectMultiScale(
                    roi_smile,
                    scaleFactor=1.8,
                    minNeighbors=20
                )
                if len(smiles) > 0:
                    smile_count += 1

            features = {
                "num_faces": min(len(faces), self.max_faces),  # Cap at configured limit
                "face_ratio": total_face_area / image_area,
                "largest_face_ratio": largest_face_area / image_area,
                "has_face": 1,
                "avg_face_size": (total_face_area / len(faces)) / image_area,
                "eyes_detected": eyes_count,
                "smile_detected": smile_count
            }

            return features

        except Exception as e:
            print(f"⚠️  Warning: Failed to detect faces in {image_path}: {e}")
            return default_features

    def detect_faces_batch(
        self,
        image_paths: List[Path],
        use_multiprocessing: bool = True,
        workers: int = 4,
        use_cache: bool = True
    ) -> Dict[Path, Dict[str, float]]:
        """
        Detect faces for batch of images.

        Args:
            image_paths: List of image paths.
            use_multiprocessing: Whether to use multiprocessing.
            workers: Number of worker processes.
            use_cache: Whether to use cached results.

        Returns:
            Dictionary mapping paths to face features.
        """
        if not use_multiprocessing or workers <= 1:
            # Serial processing
            results = {}
            for path in tqdm(image_paths, desc="Detecting faces"):
                results[path] = self.detect_faces(path, use_cache=use_cache)
            return results

        # Parallel processing
        try:
            # Use module-level function with partial for Windows compatibility
            worker_func = partial(_detect_faces_worker, detector=self, use_cache=use_cache)

            with Pool(workers) as pool:
                results = list(tqdm(
                    pool.imap(worker_func, image_paths, chunksize=50),
                    total=len(image_paths),
                    desc=f"Detecting faces ({workers} workers)"
                ))

            return {path: features for path, features in results}

        except Exception as e:
            print(f"⚠️  Multiprocessing failed ({e}), falling back to serial...")
            results = {}
            for path in tqdm(image_paths, desc="Detecting faces (serial)"):
                results[path] = self.detect_faces(path, use_cache=use_cache)
            return results
