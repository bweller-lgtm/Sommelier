"""Tests for features: quality scoring, face detection, embeddings, burst detection, document features."""
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from PIL import Image

from sommelier.core.config import load_config, QualityConfig, BurstDetectionConfig, ModelConfig, PerformanceConfig
from sommelier.features.quality import QualityScorer, FaceDetector
from sommelier.features.burst_detector import BurstDetector
from sommelier.features.document_features import DocumentFeatures, DocumentFeatureExtractor, DocumentGrouper


# ── QualityScorer ───────────────────────────────────────────────────


class TestQualityScorer:

    @pytest.fixture
    def config(self):
        return QualityConfig()

    @pytest.fixture
    def scorer(self, config):
        return QualityScorer(config)

    def test_compute_score_real_image(self, scorer, tmp_path):
        """Quality score on a simple generated image."""
        img = Image.new("RGB", (200, 200), color=(128, 128, 128))
        path = tmp_path / "test.jpg"
        img.save(path)
        score = scorer.compute_score(path, use_cache=False)
        assert 0.0 <= score <= 1.0

    def test_compute_score_bright_image(self, scorer, tmp_path):
        """Bright image should have higher brightness component."""
        img = Image.new("RGB", (200, 200), color=(250, 250, 250))
        path = tmp_path / "bright.jpg"
        img.save(path)
        score = scorer.compute_score(path, use_cache=False)
        assert score > 0.0

    def test_compute_score_dark_image(self, scorer, tmp_path):
        """Dark image should have lower brightness component."""
        img = Image.new("RGB", (200, 200), color=(10, 10, 10))
        path = tmp_path / "dark.jpg"
        img.save(path)
        score = scorer.compute_score(path, use_cache=False)
        assert score < 0.5

    def test_compute_score_invalid_path(self, scorer, tmp_path):
        """Non-existent file returns 0.0."""
        score = scorer.compute_score(tmp_path / "nonexistent.jpg", use_cache=False)
        assert score == 0.0

    def test_compute_score_with_cache(self, config, tmp_path):
        """Score gets cached and retrieved."""
        mock_cache = MagicMock()
        mock_cache.get.return_value = 0.75
        scorer = QualityScorer(config, cache_manager=mock_cache)

        score = scorer.compute_score(tmp_path / "test.jpg", use_cache=True)
        assert score == 0.75
        mock_cache.get.assert_called_once()

    def test_compute_score_cache_miss(self, config, tmp_path):
        """Cache miss triggers computation and caching."""
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        scorer = QualityScorer(config, cache_manager=mock_cache)

        img = Image.new("RGB", (200, 200), color=(128, 128, 128))
        path = tmp_path / "test.jpg"
        img.save(path)

        score = scorer.compute_score(path, use_cache=True)
        assert 0.0 <= score <= 1.0
        mock_cache.set.assert_called_once()

    def test_compute_scores_batch_serial(self, scorer, tmp_path):
        """Batch computation in serial mode."""
        paths = []
        for i in range(3):
            img = Image.new("RGB", (100, 100), color=(i * 80, i * 80, i * 80))
            path = tmp_path / f"img{i}.jpg"
            img.save(path)
            paths.append(path)

        scores = scorer.compute_scores_batch(paths, use_multiprocessing=False, use_cache=False)
        assert len(scores) == 3
        for path in paths:
            assert 0.0 <= scores[path] <= 1.0

    def test_filter_by_quality(self, scorer):
        p1, p2, p3 = Path("a.jpg"), Path("b.jpg"), Path("c.jpg")
        scores = {p1: 0.9, p2: 0.3, p3: 0.6}
        passing, failing = scorer.filter_by_quality([p1, p2, p3], scores, threshold=0.5)
        assert p1 in passing
        assert p3 in passing
        assert p2 in failing

    def test_filter_by_quality_missing_score(self, scorer):
        """Missing score defaults to 0.0."""
        p1 = Path("a.jpg")
        scores = {}
        passing, failing = scorer.filter_by_quality([p1], scores, threshold=0.5)
        assert p1 in failing

    def test_filter_by_quality_config_threshold(self, config):
        config.quality_filter_threshold = 0.7
        scorer = QualityScorer(config)
        p1 = Path("a.jpg")
        scores = {p1: 0.65}
        passing, failing = scorer.filter_by_quality([p1], scores)
        assert p1 in failing


# ── FaceDetector ────────────────────────────────────────────────────


class TestFaceDetector:

    def test_init(self):
        detector = FaceDetector()
        assert detector.max_faces == 10
        assert detector.face_cascade is not None

    def test_detect_no_faces_blank(self, tmp_path):
        """Blank image should detect no faces."""
        detector = FaceDetector()
        img = Image.new("RGB", (200, 200), color=(255, 255, 255))
        path = tmp_path / "blank.jpg"
        img.save(path)
        features = detector.detect_faces(path, use_cache=False)
        assert features["num_faces"] == 0
        assert features["has_face"] == 0

    def test_detect_nonexistent_image(self, tmp_path):
        """Non-existent file returns default features."""
        detector = FaceDetector()
        features = detector.detect_faces(tmp_path / "nope.jpg", use_cache=False)
        assert features["num_faces"] == 0

    def test_detect_with_cache_hit(self):
        mock_cache = MagicMock()
        cached = {"num_faces": 2, "has_face": 1}
        mock_cache.get.return_value = cached
        detector = FaceDetector(cache_manager=mock_cache)
        features = detector.detect_faces(Path("test.jpg"), use_cache=True)
        assert features["num_faces"] == 2

    def test_detect_no_cascade(self):
        """If cascade fails to load, returns default."""
        detector = FaceDetector()
        detector.face_cascade = None
        features = detector._detect_faces_nocache(Path("test.jpg"))
        assert features["num_faces"] == 0

    def test_detect_faces_batch_serial(self, tmp_path):
        detector = FaceDetector()
        img = Image.new("RGB", (200, 200), color=(200, 200, 200))
        p1 = tmp_path / "a.jpg"
        p2 = tmp_path / "b.jpg"
        img.save(p1)
        img.save(p2)
        results = detector.detect_faces_batch([p1, p2], use_multiprocessing=False, use_cache=False)
        assert len(results) == 2


# ── BurstDetector ───────────────────────────────────────────────────


class TestBurstDetector:

    @pytest.fixture
    def config(self):
        return BurstDetectionConfig()

    def test_empty_photos(self, config):
        detector = BurstDetector(config)
        assert detector.detect_bursts([], np.array([])) == []

    def test_disabled(self, config):
        config.enabled = False
        detector = BurstDetector(config)
        photos = [Path("a.jpg"), Path("b.jpg")]
        embeddings = np.random.rand(2, 512)
        bursts = detector.detect_bursts(photos, embeddings)
        assert len(bursts) == 2
        assert all(len(b) == 1 for b in bursts)

    def test_mismatch_raises(self, config):
        detector = BurstDetector(config)
        with pytest.raises(ValueError, match="does not match"):
            detector.detect_bursts([Path("a.jpg")], np.random.rand(2, 512))

    def test_temporal_groups_creation(self, config):
        from datetime import datetime, timedelta
        config.time_window_seconds = 5
        config.min_burst_size = 2
        detector = BurstDetector(config)

        now = datetime.now()
        photo_times = [
            (0, Path("a.jpg"), now),
            (1, Path("b.jpg"), now + timedelta(seconds=2)),
            (2, Path("c.jpg"), now + timedelta(seconds=60)),
        ]
        groups = detector._create_temporal_groups(photo_times)
        assert len(groups) == 1
        assert len(groups[0]) == 2

    def test_temporal_groups_empty(self, config):
        detector = BurstDetector(config)
        assert detector._create_temporal_groups([]) == []

    def test_get_photo_timestamp_fallback(self, config, tmp_path):
        """Non-image file falls back to mtime."""
        detector = BurstDetector(config)
        path = tmp_path / "test.txt"
        path.write_text("hello")
        ts = detector._get_photo_timestamp(path)
        assert ts is not None

    def test_get_photo_timestamp_whatsapp(self, config, tmp_path):
        """WhatsApp filename format should be parsed."""
        detector = BurstDetector(config)
        path = tmp_path / "IMG-20240115-WA0001.jpg"
        path.write_text("fake")
        ts = detector._get_photo_timestamp(path)
        assert ts is not None
        assert ts.year == 2024
        assert ts.month == 1
        assert ts.day == 15

    def test_visual_only_clustering(self, config):
        config.embedding_similarity_threshold = 0.9
        detector = BurstDetector(config)
        # Two identical embeddings, one different
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.999, 0.01, 0.0],
            [0.0, 1.0, 0.0],
        ])
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        photos = [Path("a.jpg"), Path("b.jpg"), Path("c.jpg")]
        bursts = detector._detect_visual_only(photos, embeddings)
        assert len(bursts) >= 2  # At least 2 clusters (pair + singleton)

    def test_print_statistics(self, config, capsys):
        detector = BurstDetector(config)
        detector._print_statistics(
            [[Path("a.jpg"), Path("b.jpg")]],
            [Path("c.jpg")],
        )
        captured = capsys.readouterr()
        assert "Burst" in captured.out
        assert "Singletons: 1" in captured.out


# ── DocumentFeatures dataclass ──────────────────────────────────────


class TestDocumentFeatures:

    def test_defaults(self):
        f = DocumentFeatures()
        assert f.text_content == ""
        assert f.text_length == 0
        assert f.page_count is None
        assert f.word_count == 0
        assert f.embedding is None


# ── DocumentFeatureExtractor ────────────────────────────────────────


class TestDocumentFeatureExtractor:

    @pytest.fixture
    def extractor(self):
        return DocumentFeatureExtractor()

    def test_extract_plain_text(self, extractor, tmp_path):
        path = tmp_path / "test.txt"
        path.write_text("Hello world! This is a test document.", encoding="utf-8")
        features = extractor.extract_features(path)
        assert features.text_length > 0
        assert features.word_count > 0
        assert features.file_format == "txt"
        assert features.file_size_bytes > 0

    def test_extract_markdown(self, extractor, tmp_path):
        path = tmp_path / "test.md"
        path.write_text("# Title\nSome content", encoding="utf-8")
        features = extractor.extract_features(path)
        assert "Title" in features.text_content

    def test_extract_csv(self, extractor, tmp_path):
        path = tmp_path / "test.csv"
        path.write_text("name,age\nAlice,30\nBob,25", encoding="utf-8")
        features = extractor.extract_features(path)
        assert features.text_length > 0

    def test_extract_text_dispatch(self, extractor, tmp_path):
        path = tmp_path / "test.rtf"
        path.write_text("RTF content", encoding="utf-8")
        text = extractor.extract_text(path)
        assert "RTF content" in text

    def test_extract_metadata_plain(self, extractor, tmp_path):
        path = tmp_path / "test.txt"
        path.write_text("content", encoding="utf-8")
        meta = extractor.extract_metadata(path)
        assert meta["filename"] == "test.txt"
        assert meta["extension"] == ".txt"
        assert meta["file_size_bytes"] > 0

    def test_compute_text_embedding_empty(self, extractor):
        result = extractor.compute_text_embedding("")
        assert result is None

    def test_compute_text_embedding_whitespace(self, extractor):
        result = extractor.compute_text_embedding("   ")
        assert result is None

    def test_compute_similarity_no_embeddings(self, extractor):
        a = DocumentFeatures(embedding=None)
        b = DocumentFeatures(embedding=None)
        assert extractor.compute_similarity(a, b) == 0.0

    def test_compute_similarity_with_embeddings(self, extractor):
        a = DocumentFeatures(embedding=np.array([1.0, 0.0, 0.0]))
        b = DocumentFeatures(embedding=np.array([1.0, 0.0, 0.0]))
        assert extractor.compute_similarity(a, b) == pytest.approx(1.0)

    def test_compute_similarity_orthogonal(self, extractor):
        a = DocumentFeatures(embedding=np.array([1.0, 0.0]))
        b = DocumentFeatures(embedding=np.array([0.0, 1.0]))
        assert extractor.compute_similarity(a, b) == pytest.approx(0.0)

    def test_plain_text_encoding_fallback(self, extractor, tmp_path):
        """File with latin-1 encoding should be read via fallback."""
        path = tmp_path / "latin.txt"
        path.write_bytes("caf\xe9".encode("latin-1"))
        text = extractor._extract_plain_text(path)
        assert "caf" in text

    def test_extract_features_stat_error(self, extractor, tmp_path):
        """Non-existent file should raise."""
        with pytest.raises(Exception):
            extractor.extract_features(tmp_path / "nonexistent.txt")

    def test_html_extraction(self, extractor, tmp_path):
        path = tmp_path / "test.html"
        path.write_text("<html><body><p>Hello world</p></body></html>", encoding="utf-8")
        text = extractor.extract_text(path)
        assert "Hello world" in text


# ── DocumentGrouper ─────────────────────────────────────────────────


class TestDocumentGrouper:

    def test_empty_documents(self):
        grouper = DocumentGrouper()
        assert grouper.group_documents([], {}) == []

    def test_no_embeddings_all_singletons(self):
        grouper = DocumentGrouper()
        docs = [Path("a.pdf"), Path("b.pdf")]
        features = {
            Path("a.pdf"): DocumentFeatures(embedding=None),
            Path("b.pdf"): DocumentFeatures(embedding=None),
        }
        groups = grouper.group_documents(docs, features)
        assert len(groups) == 2
        assert all(len(g) == 1 for g in groups)

    def test_similar_documents_grouped(self):
        grouper = DocumentGrouper(similarity_threshold=0.9)
        docs = [Path("a.pdf"), Path("b.pdf"), Path("c.pdf")]
        # a and b are very similar, c is different
        features = {
            Path("a.pdf"): DocumentFeatures(embedding=np.array([1.0, 0.0, 0.0])),
            Path("b.pdf"): DocumentFeatures(embedding=np.array([0.99, 0.01, 0.0])),
            Path("c.pdf"): DocumentFeatures(embedding=np.array([0.0, 1.0, 0.0])),
        }
        # Normalize embeddings
        for f in features.values():
            f.embedding = f.embedding / np.linalg.norm(f.embedding)

        groups = grouper.group_documents(docs, features)
        # a and b should be in one group, c alone
        group_sizes = sorted([len(g) for g in groups])
        assert group_sizes == [1, 2]

    def test_all_different_documents(self):
        grouper = DocumentGrouper(similarity_threshold=0.99)
        docs = [Path("a.pdf"), Path("b.pdf")]
        features = {
            Path("a.pdf"): DocumentFeatures(embedding=np.array([1.0, 0.0])),
            Path("b.pdf"): DocumentFeatures(embedding=np.array([0.0, 1.0])),
        }
        groups = grouper.group_documents(docs, features)
        assert len(groups) == 2

    def test_mixed_with_and_without_embeddings(self):
        grouper = DocumentGrouper(similarity_threshold=0.5)
        docs = [Path("a.pdf"), Path("b.pdf"), Path("c.pdf")]
        features = {
            Path("a.pdf"): DocumentFeatures(embedding=np.array([1.0, 0.0])),
            Path("b.pdf"): DocumentFeatures(embedding=None),
            Path("c.pdf"): DocumentFeatures(embedding=np.array([0.9, 0.1])),
        }
        for f in features.values():
            if f.embedding is not None:
                f.embedding = f.embedding / np.linalg.norm(f.embedding)

        groups = grouper.group_documents(docs, features)
        # b should be a singleton (no embedding)
        flat = [doc for g in groups for doc in g]
        assert Path("b.pdf") in flat


# ── EmbeddingExtractor ──────────────────────────────────────────────


class TestEmbeddingExtractor:

    def test_compute_similarity(self):
        from sommelier.features.embeddings import EmbeddingExtractor
        model_config = MagicMock()
        model_config.clip_model = "ViT-B-32"
        model_config.clip_pretrained = "laion2b_s34b_b79k"
        perf_config = MagicMock()
        perf_config.device = "cpu"

        extractor = EmbeddingExtractor(model_config, perf_config)
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        assert extractor.compute_similarity(a, b) == pytest.approx(1.0)

    def test_compute_similarity_orthogonal(self):
        from sommelier.features.embeddings import EmbeddingExtractor
        extractor = EmbeddingExtractor(MagicMock(), MagicMock())
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert extractor.compute_similarity(a, b) == pytest.approx(0.0)

    def test_compute_similarity_matrix(self):
        from sommelier.features.embeddings import EmbeddingExtractor
        extractor = EmbeddingExtractor(MagicMock(), MagicMock())
        embeddings = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ])
        matrix = extractor.compute_similarity_matrix(embeddings)
        assert matrix.shape == (3, 3)
        assert matrix[0, 0] == pytest.approx(1.0)
        assert matrix[0, 1] == pytest.approx(0.0)
        assert matrix[0, 2] == pytest.approx(1.0)

    def test_lazy_model_not_loaded(self):
        from sommelier.features.embeddings import EmbeddingExtractor
        extractor = EmbeddingExtractor(MagicMock(), MagicMock())
        assert extractor._model is None
        assert extractor._preprocess is None

    def test_extract_embedding_cache_hit(self):
        from sommelier.features.embeddings import EmbeddingExtractor
        mock_cache = MagicMock()
        mock_cache.get.return_value = np.array([0.5, 0.5])
        extractor = EmbeddingExtractor(MagicMock(), MagicMock(), cache_manager=mock_cache)
        result = extractor.extract_embedding(Path("test.jpg"))
        assert result is not None
        np.testing.assert_array_equal(result, [0.5, 0.5])
