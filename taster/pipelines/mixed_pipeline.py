"""Mixed-media classification pipeline.

Auto-detects media types in a folder and dispatches to the appropriate sub-pipeline.
"""
from pathlib import Path
from typing import Dict, List, Any, Optional

from ..core.config import Config
from ..core.cache import CacheManager
from ..core.ai_client import AIClient
from ..core.file_utils import FileTypeRegistry
from ..core.profiles import TasteProfile
from .base import ClassificationPipeline, ClassificationResult
from .photo_pipeline import PhotoPipeline
from .document_pipeline import DocumentPipeline


class MixedPipeline(ClassificationPipeline):
    """Pipeline that auto-detects media types and dispatches to sub-pipelines."""

    def __init__(
        self,
        config: Config,
        profile: Optional[TasteProfile] = None,
        cache_manager: Optional[CacheManager] = None,
        gemini_client: Optional[AIClient] = None,
    ):
        super().__init__(config, profile)
        self.cache_manager = cache_manager
        self.gemini_client = gemini_client

    def collect_files(self, folder: Path) -> List[Path]:
        """Collect all media files (delegated to sub-pipelines)."""
        media = FileTypeRegistry.list_all_media(folder)
        return media["images"] + media["videos"] + media["documents"]

    def extract_features(self, files: List[Path]) -> Dict[Path, Any]:
        """Not used directly - delegated to sub-pipelines."""
        return {}

    def group_files(self, files: List[Path], features: Dict) -> List[List[Path]]:
        """Not used directly - delegated to sub-pipelines."""
        return [[f] for f in files]

    def classify(self, groups: List[List[Path]], features: Dict) -> List[Dict[str, Any]]:
        """Not used directly - delegated to sub-pipelines."""
        return []

    def route(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Not used directly - delegated to sub-pipelines."""
        return results

    def run(
        self,
        input_folder: Path,
        output_folder: Path,
        dry_run: bool = False,
        classify_videos: bool = True,
    ) -> ClassificationResult:
        """Execute the mixed pipeline by dispatching to sub-pipelines."""
        media = FileTypeRegistry.list_all_media(input_folder)
        combined = ClassificationResult()

        has_images = len(media["images"]) > 0 or len(media["videos"]) > 0
        has_documents = len(media["documents"]) > 0

        if has_images:
            print(f"\nFound {len(media['images'])} images, {len(media['videos'])} videos")
            photo_pipeline = PhotoPipeline(
                self.config, self.profile, self.cache_manager, self.gemini_client
            )
            photo_result = photo_pipeline.run(
                input_folder, output_folder, dry_run, classify_videos
            )
            combined = combined.merge(photo_result)

        if has_documents:
            print(f"\nFound {len(media['documents'])} documents")
            doc_pipeline = DocumentPipeline(
                self.config, self.profile, self.cache_manager, self.gemini_client
            )
            doc_result = doc_pipeline.run(input_folder, output_folder, dry_run)
            combined = combined.merge(doc_result)

        if not has_images and not has_documents:
            print("\nNo media files found to process.")

        return combined
