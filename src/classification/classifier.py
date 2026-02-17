"""Unified classifier for photos, videos, and documents."""
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from tqdm import tqdm

from ..core.models import GeminiClient
from ..core.config import Config
from ..core.cache import CacheManager, CacheKey
from ..core.file_utils import ImageUtils
from ..core.profiles import TasteProfile
from .prompt_builder import PromptBuilder


class MediaClassifier:
    """
    Unified classifier for all media types (singletons, bursts, videos, documents).

    Provides consistent classification interface regardless of media type.
    """

    def __init__(
        self,
        config: Config,
        gemini_client: GeminiClient,
        prompt_builder: PromptBuilder,
        cache_manager: Optional[CacheManager] = None,
        profile: Optional[TasteProfile] = None
    ):
        """
        Initialize classifier.

        Args:
            config: Configuration object.
            gemini_client: Gemini API client.
            prompt_builder: Prompt builder for generating prompts.
            cache_manager: Optional cache manager for caching responses.
            profile: Optional TasteProfile for dynamic category validation.
        """
        self.config = config
        self.client = gemini_client
        self.prompt_builder = prompt_builder
        self.cache_manager = cache_manager
        self.profile = profile
        self.max_output_tokens = config.model.max_output_tokens

    @property
    def _valid_categories(self) -> List[str]:
        """Get valid classification categories."""
        if self.profile:
            return self.profile.category_names + ["Review"]
        return ["Share", "Storage", "Review", "Ignore"]

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
            return self._create_fallback_response("Failed to load image", error_type="load_error")

        # Build prompt
        prompt = self.prompt_builder.build_singleton_prompt()

        # Define the API call as a callable for retry wrapper
        def call_gemini() -> Dict[str, Any]:
            try:
                result = self.client.generate_json(
                    prompt=[prompt, img],
                    fallback=self._create_fallback_response("API error"),
                    generation_config={"max_output_tokens": self.max_output_tokens},
                    handle_safety_errors=True
                )
                return self._validate_singleton_response(result)
            except Exception as e:
                print(f"⚠️  Classification error for {photo_path.name}: {e}")
                return self._create_fallback_response(f"Error: {e}")

        # Call with retry
        result = self._execute_with_retry(call_gemini, f"Singleton {photo_path.name}")

        # Cache result (even error fallbacks, to avoid repeated failures)
        if use_cache and self.cache_manager is not None:
            cache_key = CacheKey.from_file(photo_path)
            self.cache_manager.set("gemini", cache_key, result)

        return result

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

        # Define the API call as a callable for retry wrapper
        def call_gemini_burst() -> Dict[str, Any]:
            try:
                result = self.client.generate_json(
                    prompt=content,
                    fallback=[self._create_fallback_response(f"API error", rank=i+1) for i in range(len(burst_photos))],
                    generation_config={"max_output_tokens": self.max_output_tokens},
                    handle_safety_errors=True
                )

                # Validate response
                if isinstance(result, list) and len(result) == len(burst_photos):
                    result = [self._validate_burst_response(r, i+1) for i, r in enumerate(result)]
                    # Return a wrapper dict for retry logic (check first result for errors)
                    return {"_burst_results": result, "is_error_fallback": False}
                else:
                    # Invalid response format
                    print(f"⚠️  Invalid burst response format, using fallback")
                    fallback_results = [
                        self._create_fallback_response(f"Invalid response format", rank=i+1, error_type="invalid_response")
                        for i in range(len(burst_photos))
                    ]
                    return {"_burst_results": fallback_results, "is_error_fallback": True, "error_type": "invalid_response"}

            except Exception as e:
                print(f"⚠️  Burst classification error: {e}")
                fallback_results = [
                    self._create_fallback_response(f"Error: {e}", rank=i+1)
                    for i in range(len(burst_photos))
                ]
                return {"_burst_results": fallback_results, "is_error_fallback": True, "error_type": "api_error"}

        # Call with retry
        wrapper_result = self._execute_with_retry(call_gemini_burst, f"Burst ({len(burst_photos)} photos)")
        result = wrapper_result.get("_burst_results", [
            self._create_fallback_response("Retry wrapper error", rank=i+1) for i in range(len(burst_photos))
        ])

        # Generate burst ID and update all results
        burst_id = CacheKey.from_files(burst_photos)
        retry_count = wrapper_result.get("retry_count", 0)
        for i, r in enumerate(result):
            r["retry_count"] = retry_count
            r["burst_id"] = burst_id
            r["burst_position"] = i

        # Store burst context for re-processing failed photos later
        if self.cache_manager is not None:
            failed_indices = [i for i, r in enumerate(result) if r.get("is_error_fallback")]
            burst_context = {
                "burst_id": burst_id,
                "photo_paths": [str(p) for p in burst_photos],
                "results": result,
                "failed_indices": failed_indices,
                "has_failures": len(failed_indices) > 0,
            }
            # Always store context (useful for re-processing)
            self.cache_manager.set("burst_context", burst_id, burst_context)

        # Cache gemini result
        if use_cache and self.cache_manager is not None:
            self.cache_manager.set("gemini", burst_id, result)

        return result

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

        # Define the API call as a callable for retry wrapper
        def call_gemini_video() -> Dict[str, Any]:
            try:
                result = self.client.generate_json(
                    prompt=[prompt, video_path],  # Path object will be uploaded
                    fallback=self._create_fallback_response("API error", is_video=True),
                    generation_config={"max_output_tokens": self.max_output_tokens},
                    handle_safety_errors=True
                )
                return self._validate_video_response(result)
            except Exception as e:
                print(f"⚠️  Video classification error for {video_path.name}: {e}")
                return self._create_fallback_response(f"Error: {e}", is_video=True)

        # Call with retry
        result = self._execute_with_retry(call_gemini_video, f"Video {video_path.name}")

        # Cache result
        if use_cache and self.cache_manager is not None:
            cache_key = CacheKey.from_file(video_path)
            self.cache_manager.set("gemini", cache_key, result)

        return result

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

    def classify_document(
        self,
        doc_path: Path,
        text_content: str = "",
        metadata: Optional[Dict] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Classify a single document.

        For PDFs, sends the file directly to Gemini (native PDF support).
        For other formats, sends extracted text + metadata as prompt context.

        Args:
            doc_path: Path to document file.
            text_content: Pre-extracted text content from the document.
            metadata: Optional metadata dict (page count, author, etc.).
            use_cache: Whether to use cached classification.

        Returns:
            Classification result dict.
        """
        # Check cache
        if use_cache and self.cache_manager is not None:
            cache_key = CacheKey.from_file(doc_path)
            cached = self.cache_manager.get("gemini", cache_key)
            if cached is not None:
                return cached

        # Build prompt
        prompt = self.prompt_builder.build_singleton_prompt(media_type="document")

        # Build context with document info
        context_parts = [prompt]

        if metadata:
            meta_str = "\n".join([f"- {k}: {v}" for k, v in metadata.items() if v])
            context_parts.append(f"\n**Document Metadata:**\n{meta_str}")

        # For PDFs, send the file directly (Gemini reads PDFs natively)
        ext = doc_path.suffix.lower()
        if ext == ".pdf":
            context_parts.append(doc_path)
        elif text_content:
            # Truncate text for the prompt
            truncated = text_content[:30000]
            if len(text_content) > 30000:
                truncated += "\n\n[... text truncated ...]"
            context_parts.append(f"\n**Document Content:**\n{truncated}")

        # Define the API call
        def call_gemini() -> Dict[str, Any]:
            try:
                result = self.client.generate_json(
                    prompt=context_parts,
                    fallback=self._create_fallback_response("API error"),
                    generation_config={"max_output_tokens": self.max_output_tokens},
                    handle_safety_errors=True
                )
                return self._validate_document_response(result)
            except Exception as e:
                print(f"Warning: Document classification error for {doc_path.name}: {e}")
                return self._create_fallback_response(f"Error: {e}")

        result = self._execute_with_retry(call_gemini, f"Document {doc_path.name}")

        # Cache result
        if use_cache and self.cache_manager is not None:
            cache_key = CacheKey.from_file(doc_path)
            self.cache_manager.set("gemini", cache_key, result)

        return result

    def classify_document_group(
        self,
        docs: List[Path],
        text_contents: Optional[Dict[Path, str]] = None,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Classify a group of similar documents comparatively.

        Args:
            docs: List of document paths.
            text_contents: Optional mapping of path to extracted text.
            use_cache: Whether to use cache.

        Returns:
            List of classification results (one per document).
        """
        if not docs:
            return []

        if len(docs) == 1:
            text = (text_contents or {}).get(docs[0], "")
            return [self.classify_document(docs[0], text_content=text, use_cache=use_cache)]

        # Check cache
        if use_cache and self.cache_manager is not None:
            cache_key = CacheKey.from_files(docs)
            cached = self.cache_manager.get("gemini", cache_key)
            if cached is not None and len(cached) == len(docs):
                return cached

        # Build prompt
        prompt = self.prompt_builder.build_group_prompt(len(docs), media_type="document")
        content = [prompt]

        for i, doc in enumerate(docs):
            text = (text_contents or {}).get(doc, "")
            ext = doc.suffix.lower()
            if ext == ".pdf":
                content.append(f"\n**Document {i+1}: {doc.name}**")
                content.append(doc)
            else:
                truncated = text[:15000]
                if len(text) > 15000:
                    truncated += "\n[... truncated ...]"
                content.append(f"\n**Document {i+1}: {doc.name}**\n{truncated}")

        def call_gemini_group() -> Dict[str, Any]:
            try:
                result = self.client.generate_json(
                    prompt=content,
                    fallback=[self._create_fallback_response(f"API error", rank=i+1) for i in range(len(docs))],
                    generation_config={"max_output_tokens": self.max_output_tokens},
                    handle_safety_errors=True
                )
                if isinstance(result, list) and len(result) == len(docs):
                    result = [self._validate_document_response(r, default_rank=i+1) for i, r in enumerate(result)]
                    return {"_group_results": result, "is_error_fallback": False}
                else:
                    fallback = [
                        self._create_fallback_response("Invalid response format", rank=i+1, error_type="invalid_response")
                        for i in range(len(docs))
                    ]
                    return {"_group_results": fallback, "is_error_fallback": True, "error_type": "invalid_response"}
            except Exception as e:
                fallback = [self._create_fallback_response(f"Error: {e}", rank=i+1) for i in range(len(docs))]
                return {"_group_results": fallback, "is_error_fallback": True, "error_type": "api_error"}

        wrapper = self._execute_with_retry(call_gemini_group, f"Document group ({len(docs)} docs)")
        result = wrapper.get("_group_results", [
            self._create_fallback_response("Retry wrapper error", rank=i+1) for i in range(len(docs))
        ])

        # Cache
        if use_cache and self.cache_manager is not None:
            cache_key = CacheKey.from_files(docs)
            self.cache_manager.set("gemini", cache_key, result)

        return result

    def _validate_document_response(self, response: Dict[str, Any], default_rank: Optional[int] = None) -> Dict[str, Any]:
        """Validate and fix document classification response."""
        if "classification" not in response:
            response["classification"] = self.profile.default_category if self.profile else "Review"
        if "confidence" not in response:
            response["confidence"] = 0.3
        if "reasoning" not in response:
            response["reasoning"] = "No reasoning provided"

        # Validate category against profile
        valid = self._valid_categories
        if response["classification"] not in valid:
            response["classification"] = self.profile.default_category if self.profile else "Review"

        response["confidence"] = max(0.0, min(1.0, float(response.get("confidence", 0.3))))

        if "content_summary" not in response:
            response["content_summary"] = ""
        if "key_topics" not in response:
            response["key_topics"] = []

        if "is_error_fallback" not in response:
            response["is_error_fallback"] = False
            response["error_type"] = None
            response["error_message"] = None
            response["retry_count"] = 0

        if default_rank is not None and "rank" not in response:
            response["rank"] = default_rank

        return response

    def _create_fallback_response(
        self,
        reason: str,
        rank: Optional[int] = None,
        is_video: bool = False,
        error_type: str = "api_error"
    ) -> Dict[str, Any]:
        """
        Create fallback response for errors.

        Args:
            reason: Reason for fallback.
            rank: Optional rank for burst responses.
            is_video: Whether this is a video response.
            error_type: Type of error that caused fallback. One of:
                - "api_error": General API call failure
                - "load_error": Failed to load image/video file
                - "safety_blocked": Content blocked by safety filters
                - "invalid_response": JSON parsing or format error
                - "timeout": Request timed out
                - "rate_limit": Rate limit exceeded

        Returns:
            Fallback classification dict.
        """
        response = {
            "classification": "Review",
            "confidence": 0.3,
            "reasoning": f"Fallback response: {reason}",
            "contains_children": None,
            "is_appropriate": None,
            # Error tracking fields
            "is_error_fallback": True,
            "error_type": error_type,
            "error_message": reason,
            "retry_count": 0,
        }

        if rank is not None:
            response["rank"] = rank

        if is_video:
            response["audio_quality"] = "unknown"

        # Add improvement fields if enabled (photos only, not videos)
        if self.config.photo_improvement.enabled and not is_video:
            response["improvement_candidate"] = False
            response["improvement_reasons"] = []
            response["contextual_value"] = "low"
            response["contextual_value_reasoning"] = ""

        return response

    def _execute_with_retry(
        self,
        classify_fn: Callable[[], Dict[str, Any]],
        error_context: str = "classification"
    ) -> Dict[str, Any]:
        """
        Execute a classification function with automatic retry on retriable errors.

        Args:
            classify_fn: Zero-argument callable that performs classification and returns result dict.
            error_context: Context string for error messages (e.g., "singleton", "burst", "video").

        Returns:
            Classification result dict with retry_count updated.
        """
        max_retries = self.config.classification.classification_retries
        retry_delay = self.config.classification.retry_delay_seconds
        retriable_errors = set(self.config.classification.retry_on_errors)

        last_result = None

        for attempt in range(max_retries + 1):
            try:
                result = classify_fn()

                # Check if result is a retriable error fallback
                if result.get("is_error_fallback") and result.get("error_type") in retriable_errors:
                    last_result = result
                    if attempt < max_retries:
                        delay = retry_delay * (2 ** attempt)  # Exponential backoff
                        print(f"   ⚠️  {error_context} error (attempt {attempt + 1}/{max_retries + 1}), retrying in {delay:.1f}s...")
                        time.sleep(delay)
                        continue

                # Success or non-retriable error
                result["retry_count"] = attempt
                return result

            except Exception as e:
                # Unexpected exception - create fallback and potentially retry
                last_result = self._create_fallback_response(f"Error: {e}", error_type="api_error")
                if attempt < max_retries:
                    delay = retry_delay * (2 ** attempt)
                    print(f"   ⚠️  {error_context} exception (attempt {attempt + 1}/{max_retries + 1}): {e}, retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    continue

        # All retries exhausted
        if last_result:
            last_result["retry_count"] = max_retries
            return last_result

        return self._create_fallback_response("All retries exhausted", error_type="api_error")

    def _validate_singleton_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and fix singleton response."""
        # Ensure required fields
        if "classification" not in response:
            response["classification"] = "Review"
        if "confidence" not in response:
            response["confidence"] = 0.3
        if "reasoning" not in response:
            response["reasoning"] = "No reasoning provided"

        # Ensure classification is valid (dynamic categories from profile)
        valid = self._valid_categories
        if response["classification"] not in valid:
            response["classification"] = self.profile.default_category if self.profile else "Review"

        # Ensure confidence is in range
        response["confidence"] = max(0.0, min(1.0, float(response.get("confidence", 0.3))))

        # Ensure boolean fields exist
        if "contains_children" not in response:
            response["contains_children"] = None
        if "is_appropriate" not in response:
            response["is_appropriate"] = None

        # Mark as successful (non-error) response
        if "is_error_fallback" not in response:
            response["is_error_fallback"] = False
            response["error_type"] = None
            response["error_message"] = None
            response["retry_count"] = 0

        # Validate improvement fields if enabled
        if self.config.photo_improvement.enabled:
            response = self._validate_improvement_fields(response)

        return response

    def _validate_improvement_fields(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and fix improvement-related fields."""
        # Valid improvement reasons
        valid_reasons = {
            "motion_blur", "focus_blur", "noise", "underexposed",
            "overexposed", "white_balance", "low_resolution", "composition"
        }

        # Ensure improvement_candidate is boolean
        if "improvement_candidate" not in response:
            response["improvement_candidate"] = False
        else:
            response["improvement_candidate"] = bool(response["improvement_candidate"])

        # Ensure improvement_reasons is a list of valid reasons
        if "improvement_reasons" not in response:
            response["improvement_reasons"] = []
        else:
            reasons = response["improvement_reasons"]
            if not isinstance(reasons, list):
                reasons = [reasons] if reasons else []
            # Filter to only valid reasons
            response["improvement_reasons"] = [
                r for r in reasons if r in valid_reasons
            ]

        # Ensure contextual_value is valid
        if "contextual_value" not in response:
            response["contextual_value"] = "low"
        elif response["contextual_value"] not in ["high", "medium", "low"]:
            response["contextual_value"] = "low"

        # Ensure contextual_value_reasoning exists
        if "contextual_value_reasoning" not in response:
            response["contextual_value_reasoning"] = ""
        else:
            response["contextual_value_reasoning"] = str(response["contextual_value_reasoning"])

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
