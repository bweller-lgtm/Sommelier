"""Document classification pipeline."""
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm

from ..core.config import Config
from ..core.cache import CacheManager
from ..core.ai_client import AIClient
from ..core.file_utils import FileTypeRegistry
from ..core.profiles import TasteProfile
from ..features.document_features import DocumentFeatureExtractor, DocumentFeatures, DocumentGrouper
from ..classification.prompt_builder import PromptBuilder
from ..classification.classifier import MediaClassifier
from ..classification.routing import Router
from .base import ClassificationPipeline, ClassificationResult


class DocumentPipeline(ClassificationPipeline):
    """Pipeline for classifying documents."""

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
        self.feature_extractor = DocumentFeatureExtractor(
            max_chars=config.document.text_extraction_max_chars,
            max_pages=config.document.max_pages_per_document,
        )

    def collect_files(self, folder: Path) -> List[Path]:
        """Collect documents from the folder."""
        return FileTypeRegistry.list_documents(folder, recursive=False)

    def extract_features(self, files: List[Path]) -> Dict[Path, Any]:
        """Extract text, metadata, and embeddings from documents."""
        if not files:
            return {}

        print(f"\nExtracting features from {len(files)} documents...")
        features = {}
        for doc_path in tqdm(files, desc="Extracting document features"):
            feat = self.feature_extractor.extract_features(doc_path)
            # Compute embedding if enabled
            if self.config.document.enable_text_embeddings and feat.text_content:
                feat.embedding = self.feature_extractor.compute_text_embedding(feat.text_content)
            features[doc_path] = feat

        return features

    def group_files(self, files: List[Path], features: Dict) -> List[List[Path]]:
        """Group similar documents using text embeddings."""
        if not files:
            return []

        doc_settings = None
        if self.profile and self.profile.document_settings:
            doc_settings = self.profile.document_settings

        enable_grouping = doc_settings.enable_similarity_grouping if doc_settings else True
        threshold = doc_settings.similarity_threshold if doc_settings else self.config.document.similarity_threshold

        if not enable_grouping:
            return [[f] for f in files]

        print("\nGrouping similar documents...")
        grouper = DocumentGrouper(similarity_threshold=threshold)
        return grouper.group_documents(files, features)

    def classify(self, groups: List[List[Path]], features: Dict) -> List[Dict[str, Any]]:
        """Classify documents using Gemini."""
        if not groups:
            return []

        prompt_builder = PromptBuilder(
            self.config, training_examples={}, profile=self.profile
        )
        classifier = MediaClassifier(
            self.config, self.gemini_client, prompt_builder,
            self.cache_manager, profile=self.profile
        )

        print(f"\nClassifying documents with Gemini...")
        results = []

        for group in tqdm(groups, desc="Classifying documents"):
            text_contents = {}
            for doc in group:
                feat = features.get(doc)
                if feat:
                    text_contents[doc] = feat.text_content

            if len(group) == 1:
                doc = group[0]
                feat = features.get(doc)
                metadata = feat.metadata if feat else None
                classification = classifier.classify_document(
                    doc,
                    text_content=text_contents.get(doc, ""),
                    metadata=metadata,
                    use_cache=True,
                )
                results.append({
                    "path": doc,
                    "group_size": 1,
                    "classification": classification,
                })
            else:
                classifications = classifier.classify_document_group(
                    group, text_contents=text_contents, use_cache=True
                )
                for i, (doc, classification) in enumerate(zip(group, classifications)):
                    results.append({
                        "path": doc,
                        "group_size": len(group),
                        "group_index": i,
                        "classification": classification,
                    })

        return results

    def route(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Route classified documents to destinations."""
        router = Router(self.config, self.gemini_client, profile=self.profile)

        for result in results:
            result["destination"] = router.route_document(result["classification"])

        return results
