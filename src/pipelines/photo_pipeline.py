"""Photo/video classification pipeline.

Extracts the core logic from taste_classify.py into a reusable pipeline class.
"""
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm

from ..core.config import Config
from ..core.cache import CacheManager
from ..core.models import GeminiClient
from ..core.file_utils import FileTypeRegistry
from ..core.profiles import TasteProfile
from ..features.embeddings import EmbeddingExtractor
from ..features.burst_detector import BurstDetector
from ..classification.prompt_builder import PromptBuilder
from ..classification.classifier import MediaClassifier
from ..classification.routing import Router
from .base import ClassificationPipeline, ClassificationResult


class PhotoPipeline(ClassificationPipeline):
    """Pipeline for classifying photos and videos."""

    def __init__(
        self,
        config: Config,
        profile: Optional[TasteProfile] = None,
        cache_manager: Optional[CacheManager] = None,
        gemini_client: Optional[GeminiClient] = None,
    ):
        super().__init__(config, profile)
        self.cache_manager = cache_manager
        self.gemini_client = gemini_client

    def collect_files(self, folder: Path) -> List[Path]:
        """Collect images from the folder."""
        return FileTypeRegistry.list_images(folder, recursive=False)

    def collect_videos(self, folder: Path) -> List[Path]:
        """Collect videos from the folder."""
        return FileTypeRegistry.list_videos(folder, recursive=False)

    def extract_features(self, files: List[Path]) -> Dict[Path, Any]:
        """Extract CLIP embeddings for burst detection."""
        if not files:
            return {}

        print(f"\nStep 1: Extracting CLIP embeddings for {len(files)} images...")
        extractor = EmbeddingExtractor(
            self.config.model, self.config.performance, self.cache_manager
        )
        embeddings = extractor.extract_embeddings_batch(
            files, use_cache=True, show_progress=True
        )
        return {"embeddings": embeddings, "images": files}

    def group_files(self, files: List[Path], features: Dict) -> List[List[Path]]:
        """Detect bursts using temporal + visual similarity."""
        if not files:
            return []

        print("\nStep 2: Detecting photo bursts...")
        embeddings = features.get("embeddings")
        detector = BurstDetector(self.config.burst_detection)
        return detector.detect_bursts(files, embeddings)

    def classify(self, groups: List[List[Path]], features: Dict) -> List[Dict[str, Any]]:
        """Classify all photos using Gemini."""
        if not groups:
            return []

        prompt_builder = PromptBuilder(
            self.config, training_examples={}, profile=self.profile
        )
        classifier = MediaClassifier(
            self.config, self.gemini_client, prompt_builder,
            self.cache_manager, profile=self.profile
        )
        embeddings = features.get("embeddings")
        images = features.get("images", [])

        print(f"\nStep 3: Classifying photos with Gemini...")
        results = []

        for burst in tqdm(groups, desc="Processing bursts/singletons"):
            if len(burst) == 1:
                photo = burst[0]
                classification = classifier.classify_singleton(photo, use_cache=True)
                results.append({
                    "path": photo,
                    "burst_size": 1,
                    "burst_index": -1,
                    "classification": classification,
                })
            else:
                classifications = classifier.classify_burst(burst, use_cache=True)
                for i, (photo, classification) in enumerate(zip(burst, classifications)):
                    results.append({
                        "path": photo,
                        "burst_size": len(burst),
                        "burst_index": i,
                        "classification": classification,
                    })

        return results

    def route(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Route classified photos to destinations."""
        router = Router(self.config, self.gemini_client, profile=self.profile)

        for result in results:
            classification = result["classification"]
            if result.get("burst_size", 1) > 1:
                # For burst photos, use singleton routing per-photo
                # (burst routing is done at the group level in classify())
                result["destination"] = router.route_singleton(classification)
            else:
                result["destination"] = router.route_singleton(classification)

        return results

    def run(
        self,
        input_folder: Path,
        output_folder: Path,
        dry_run: bool = False,
        classify_videos: bool = True,
    ) -> ClassificationResult:
        """Execute the full photo/video pipeline."""
        # Process images
        images = self.collect_files(input_folder)
        image_result = ClassificationResult()

        if images:
            print(f"\nProcessing {len(images)} images...")
            features = self.extract_features(images)
            groups = self.group_files(images, features)
            results = self.classify(groups, features)
            routed = self.route(results)
            if not dry_run:
                self.move_files(routed, output_folder)
            image_result = ClassificationResult(
                results=routed,
                stats=self.compute_stats(routed),
            )

        # Process videos
        videos = self.collect_videos(input_folder)
        video_result = ClassificationResult()

        if videos:
            print(f"\nProcessing {len(videos)} videos...")
            if classify_videos:
                video_result = self._classify_videos(videos, output_folder, dry_run)
            else:
                video_result = self._copy_videos(videos, output_folder, dry_run)

        return image_result.merge(video_result)

    def _classify_videos(
        self, videos: List[Path], output_folder: Path, dry_run: bool
    ) -> ClassificationResult:
        """Classify videos with Gemini."""
        prompt_builder = PromptBuilder(
            self.config, training_examples={}, profile=self.profile
        )
        classifier = MediaClassifier(
            self.config, self.gemini_client, prompt_builder,
            self.cache_manager, profile=self.profile
        )
        router = Router(self.config, self.gemini_client, profile=self.profile)

        results = []
        for video in tqdm(videos, desc="Classifying videos"):
            classification = classifier.classify_video(video, use_cache=True)
            destination = router.route_video(classification)
            results.append({
                "path": video,
                "classification": classification,
                "destination": destination,
            })

        if not dry_run:
            self.move_files(results, output_folder)

        return ClassificationResult(
            results=results,
            stats=self.compute_stats(results),
        )

    def _copy_videos(
        self, videos: List[Path], output_folder: Path, dry_run: bool
    ) -> ClassificationResult:
        """Copy videos without classification."""
        results = []
        for video in videos:
            results.append({
                "path": video,
                "classification": {"classification": "Videos", "confidence": None},
                "destination": "Videos",
            })

        if not dry_run:
            self.move_files(results, output_folder)

        return ClassificationResult(
            results=results,
            stats=self.compute_stats(results),
        )
