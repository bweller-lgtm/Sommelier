"""Unified classifier for photos and videos."""
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm

from ..core.models import GeminiClient
from ..core.config import Config
from ..core.cache import CacheManager, CacheKey
from ..core.file_utils import ImageUtils
from .prompt_builder import PromptBuilder


class MediaClassifier:
    """
    Unified classifier for all media types (singletons, bursts, videos).

    Provides consistent classification interface regardless of media type.
    """

    def __init__(
        self,
        config: Config,
        gemini_client: GeminiClient,
        prompt_builder: PromptBuilder,
        cache_manager: Optional[CacheManager] = None
    ):
        """
        Initialize classifier.

        Args:
            config: Configuration object.
            gemini_client: Gemini API client.
            prompt_builder: Prompt builder for generating prompts.
            cache_manager: Optional cache manager for caching responses.
        """
        self.config = config
        self.client = gemini_client
        self.prompt_builder = prompt_builder
        self.cache_manager = cache_manager

    def classify_singleton(
        self,
        photo_path: Path,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Classify a single photo.

        Args:
            photo_path: Path to photo.
            use_cache: Whether to use cached classification.

        Returns:
            Classification result dict with keys:
            - classification: "Share", "Storage", or "Ignore"
            - confidence: 0.0 to 1.0
            - reasoning: explanation string
            - contains_children: bool
            - is_appropriate: bool
        """
        # Check cache
        if use_cache and self.cache_manager is not None:
            cache_key = CacheKey.from_file(photo_path)
            cached = self.cache_manager.get("gemini", cache_key)
            if cached is not None:
                return cached

        # Load image
        img = ImageUtils.load_and_fix_orientation(photo_path, max_size=1024)
        if img is None:
            return self._create_fallback_response("Failed to load image")

        # Build prompt
        prompt = self.prompt_builder.build_singleton_prompt()

        # Call Gemini
        try:
            result = self.client.generate_json(
                prompt=[prompt, img],
                fallback=self._create_fallback_response("API error"),
                generation_config={"max_output_tokens": 512},
                handle_safety_errors=True
            )

            # Validate response
            result = self._validate_singleton_response(result)

            # Cache result
            if use_cache and self.cache_manager is not None:
                cache_key = CacheKey.from_file(photo_path)
                self.cache_manager.set("gemini", cache_key, result)

            return result

        except Exception as e:
            print(f"⚠️  Classification error for {photo_path.name}: {e}")
            return self._create_fallback_response(f"Error: {e}")

    def classify_burst(
        self,
        burst_photos: List[Path],
        use_cache: bool = True,
        enable_chunking: bool = True,
        chunk_size: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Classify a burst of photos together.

        Args:
            burst_photos: List of photo paths in burst.
            use_cache: Whether to use cached classifications.
            enable_chunking: Whether to split large bursts into chunks.
            chunk_size: Maximum photos per chunk.

        Returns:
            List of classification results (one per photo, in order).
        """
        if not burst_photos:
            return []

        # Single photo? Use singleton classification
        if len(burst_photos) == 1:
            return [self.classify_singleton(burst_photos[0], use_cache)]

        # Check if we should chunk
        if enable_chunking and len(burst_photos) > chunk_size:
            return self._classify_burst_chunked(burst_photos, chunk_size, use_cache)

        # Check cache
        if use_cache and self.cache_manager is not None:
            cache_key = CacheKey.from_files(burst_photos)
            cached = self.cache_manager.get("gemini", cache_key)
            if cached is not None and len(cached) == len(burst_photos):
                return cached

        # Load images
        images = []
        for photo_path in burst_photos:
            img = ImageUtils.load_and_fix_orientation(photo_path, max_size=800)
            if img is None:
                # Failed to load - add placeholder
                images.append(None)
            else:
                images.append(img)

        # Build prompt
        prompt = self.prompt_builder.build_burst_prompt(len(burst_photos))

        # Build content list (prompt + images)
        content = [prompt]
        for i, img in enumerate(images):
            if img is not None:
                content.append(img)
            else:
                # Placeholder text for failed images
                content.append(f"[Image {i+1} failed to load]")

        # Call Gemini
        try:
            result = self.client.generate_json(
                prompt=content,
                fallback=[self._create_fallback_response(f"API error") for _ in burst_photos],
                generation_config={"max_output_tokens": 2048},
                handle_safety_errors=True
            )

            # Validate response
            if isinstance(result, list) and len(result) == len(burst_photos):
                result = [self._validate_burst_response(r, i+1) for i, r in enumerate(result)]
            else:
                # Invalid response format
                print(f"⚠️  Invalid burst response format, using fallback")
                result = [
                    self._create_fallback_response(f"Invalid response", rank=i+1)
                    for i in range(len(burst_photos))
                ]

            # Cache result
            if use_cache and self.cache_manager is not None:
                cache_key = CacheKey.from_files(burst_photos)
                self.cache_manager.set("gemini", cache_key, result)

            return result

        except Exception as e:
            print(f"⚠️  Burst classification error: {e}")
            return [
                self._create_fallback_response(f"Error: {e}", rank=i+1)
                for i in range(len(burst_photos))
            ]

    def _classify_burst_chunked(
        self,
        burst_photos: List[Path],
        chunk_size: int,
        use_cache: bool
    ) -> List[Dict[str, Any]]:
        """
        Classify large burst in chunks.

        Args:
            burst_photos: List of photo paths.
            chunk_size: Maximum photos per chunk.
            use_cache: Whether to use cache.

        Returns:
            List of classification results.
        """
        print(f"   📦 Burst has {len(burst_photos)} photos, splitting into chunks of {chunk_size}")

        all_results = []

        for chunk_start in range(0, len(burst_photos), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(burst_photos))
            chunk = burst_photos[chunk_start:chunk_end]

            print(f"   Processing chunk {chunk_start//chunk_size + 1} ({len(chunk)} photos)...")
            chunk_results = self.classify_burst(chunk, use_cache, enable_chunking=False)

            # Adjust ranks to be global
            for result in chunk_results:
                if "rank" in result:
                    result["rank"] += chunk_start

            all_results.extend(chunk_results)

        return all_results

    def classify_video(
        self,
        video_path: Path,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Classify a video.

        Args:
            video_path: Path to video file.
            use_cache: Whether to use cached classification.

        Returns:
            Classification result dict with keys:
            - classification: "Share", "Storage", or "Ignore"
            - confidence: 0.0 to 1.0
            - reasoning: explanation string
            - contains_children: bool
            - is_appropriate: bool
            - audio_quality: "good", "poor", or "silent"
            - highlights: optional key moments
        """
        # Check cache
        if use_cache and self.cache_manager is not None:
            cache_key = CacheKey.from_file(video_path)
            cached = self.cache_manager.get("gemini", cache_key)
            if cached is not None:
                return cached

        # Build prompt
        prompt = self.prompt_builder.build_video_prompt()

        # Call Gemini (Gemini handles video upload automatically)
        try:
            result = self.client.generate_json(
                prompt=[prompt, video_path],  # Path object will be uploaded
                fallback=self._create_fallback_response("API error", is_video=True),
                generation_config={"max_output_tokens": 1024},
                handle_safety_errors=True
            )

            # Validate response
            result = self._validate_video_response(result)

            # Cache result
            if use_cache and self.cache_manager is not None:
                cache_key = CacheKey.from_file(video_path)
                self.cache_manager.set("gemini", cache_key, result)

            return result

        except Exception as e:
            print(f"⚠️  Video classification error for {video_path.name}: {e}")
            return self._create_fallback_response(f"Error: {e}", is_video=True)

    def classify_batch(
        self,
        photos: List[Path],
        show_progress: bool = True,
        use_cache: bool = True
    ) -> Dict[Path, Dict[str, Any]]:
        """
        Classify batch of singleton photos.

        Args:
            photos: List of photo paths.
            show_progress: Whether to show progress bar.
            use_cache: Whether to use cache.

        Returns:
            Dictionary mapping paths to classification results.
        """
        results = {}

        iterator = photos
        if show_progress:
            iterator = tqdm(photos, desc="Classifying photos")

        for photo in iterator:
            results[photo] = self.classify_singleton(photo, use_cache)

        return results

    def _create_fallback_response(
        self,
        reason: str,
        rank: Optional[int] = None,
        is_video: bool = False
    ) -> Dict[str, Any]:
        """
        Create fallback response for errors.

        Args:
            reason: Reason for fallback.
            rank: Optional rank for burst responses.
            is_video: Whether this is a video response.

        Returns:
            Fallback classification dict.
        """
        response = {
            "classification": "Review",
            "confidence": 0.3,
            "reasoning": f"Fallback response: {reason}",
            "contains_children": None,
            "is_appropriate": None
        }

        if rank is not None:
            response["rank"] = rank

        if is_video:
            response["audio_quality"] = "unknown"

        return response

    def _validate_singleton_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and fix singleton response."""
        # Ensure required fields
        if "classification" not in response:
            response["classification"] = "Review"
        if "confidence" not in response:
            response["confidence"] = 0.3
        if "reasoning" not in response:
            response["reasoning"] = "No reasoning provided"

        # Ensure classification is valid
        if response["classification"] not in ["Share", "Storage", "Review", "Ignore"]:
            response["classification"] = "Review"

        # Ensure confidence is in range
        response["confidence"] = max(0.0, min(1.0, float(response.get("confidence", 0.3))))

        # Ensure boolean fields exist
        if "contains_children" not in response:
            response["contains_children"] = None
        if "is_appropriate" not in response:
            response["is_appropriate"] = None

        return response

    def _validate_burst_response(self, response: Dict[str, Any], default_rank: int) -> Dict[str, Any]:
        """Validate and fix burst response."""
        # Validate as singleton first
        response = self._validate_singleton_response(response)

        # Ensure rank exists
        if "rank" not in response:
            response["rank"] = default_rank

        return response

    def _validate_video_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and fix video response."""
        # Validate as singleton first
        response = self._validate_singleton_response(response)

        # Ensure video-specific fields
        if "audio_quality" not in response:
            response["audio_quality"] = "unknown"

        return response
