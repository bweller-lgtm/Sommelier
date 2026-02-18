"""Feature extraction modules."""
from .quality import QualityScorer, FaceDetector
from .burst_detector import BurstDetector
from .embeddings import EmbeddingExtractor
from .document_features import DocumentFeatureExtractor, DocumentFeatures, DocumentGrouper

__all__ = [
    "QualityScorer",
    "FaceDetector",
    "BurstDetector",
    "EmbeddingExtractor",
    "DocumentFeatureExtractor",
    "DocumentFeatures",
    "DocumentGrouper",
]
