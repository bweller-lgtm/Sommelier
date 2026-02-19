"""Tests for pipeline base, photo, document, and mixed pipelines."""
import pytest
import shutil
from pathlib import Path
from collections import defaultdict
from unittest.mock import MagicMock, patch

from taster.core.config import load_config
from taster.core.profiles import TasteProfile, CategoryDefinition
from taster.pipelines.base import ClassificationResult, ClassificationPipeline
from taster.pipelines.photo_pipeline import PhotoPipeline
from taster.pipelines.document_pipeline import DocumentPipeline
from taster.pipelines.mixed_pipeline import MixedPipeline


@pytest.fixture
def config():
    return load_config(Path("config.yaml"))


# ── ClassificationResult ─────────────────────────────────────────────


class TestClassificationResult:

    def test_empty_result(self):
        r = ClassificationResult()
        assert r.results == []
        assert r.stats == {}

    def test_print_summary(self, capsys):
        r = ClassificationResult(
            results=[],
            stats={"Share": 5, "Storage": 10},
        )
        r.print_summary("photos")
        captured = capsys.readouterr()
        assert "Share" in captured.out
        assert "Storage" in captured.out
        assert "TOTAL" in captured.out

    def test_print_summary_empty(self, capsys):
        r = ClassificationResult()
        r.print_summary()
        captured = capsys.readouterr()
        assert "No items processed" in captured.out

    def test_merge(self):
        r1 = ClassificationResult(
            results=[{"path": "a.jpg", "destination": "Share"}],
            stats={"Share": 1},
        )
        r2 = ClassificationResult(
            results=[{"path": "b.jpg", "destination": "Storage"}],
            stats={"Storage": 1, "Share": 2},
        )
        merged = r1.merge(r2)
        assert len(merged.results) == 2
        assert merged.stats["Share"] == 3
        assert merged.stats["Storage"] == 1


# ── Base pipeline ────────────────────────────────────────────────────


class TestBasePipeline:

    def test_move_files(self, tmp_path, config):
        # Create source files
        src = tmp_path / "src"
        src.mkdir()
        (src / "photo1.jpg").write_text("img1")
        (src / "photo2.jpg").write_text("img2")

        output = tmp_path / "output"

        results = [
            {"path": src / "photo1.jpg", "destination": "Share"},
            {"path": src / "photo2.jpg", "destination": "Storage"},
        ]

        # Use PhotoPipeline which inherits move_files
        pipeline = PhotoPipeline(config)
        pipeline.move_files(results, output)

        assert (output / "Share" / "photo1.jpg").exists()
        assert (output / "Storage" / "photo2.jpg").exists()

    def test_compute_stats(self, config):
        pipeline = PhotoPipeline(config)
        results = [
            {"destination": "Share"},
            {"destination": "Share"},
            {"destination": "Storage"},
            {"destination": "Review"},
        ]
        stats = pipeline.compute_stats(results)
        assert stats == {"Share": 2, "Storage": 1, "Review": 1}

    def test_run_empty_folder(self, tmp_path, config):
        """Pipeline run on empty folder returns empty result."""
        empty = tmp_path / "empty"
        empty.mkdir()
        output = tmp_path / "output"

        pipeline = PhotoPipeline(config)
        result = pipeline.run(empty, output)
        assert result.results == []
        assert result.stats == {}


# ── Photo pipeline ───────────────────────────────────────────────────


class TestPhotoPipeline:

    def test_collect_files(self, tmp_path, config):
        (tmp_path / "photo.jpg").write_text("img")
        (tmp_path / "photo.png").write_text("img")
        (tmp_path / "doc.pdf").write_text("doc")

        pipeline = PhotoPipeline(config)
        files = pipeline.collect_files(tmp_path)
        extensions = {f.suffix.lower() for f in files}
        assert ".jpg" in extensions or ".png" in extensions
        assert ".pdf" not in extensions

    def test_collect_videos(self, tmp_path, config):
        (tmp_path / "video.mp4").write_text("vid")
        (tmp_path / "photo.jpg").write_text("img")

        pipeline = PhotoPipeline(config)
        videos = pipeline.collect_videos(tmp_path)
        assert all(v.suffix.lower() in (".mp4",) for v in videos)

    def test_route_singletons(self, config):
        mock_client = MagicMock()
        pipeline = PhotoPipeline(config, gemini_client=mock_client)

        results = [
            {
                "path": Path("a.jpg"),
                "burst_size": 1,
                "classification": {
                    "classification": "Share", "score": 5,
                    "contains_children": True, "is_appropriate": True,
                },
            },
            {
                "path": Path("b.jpg"),
                "burst_size": 1,
                "classification": {
                    "classification": "Storage", "score": 2,
                    "contains_children": True, "is_appropriate": True,
                },
            },
        ]
        routed = pipeline.route(results)
        assert routed[0]["destination"] == "Share"
        assert routed[1]["destination"] == "Storage"

    def test_route_burst_photos(self, config):
        mock_client = MagicMock()
        pipeline = PhotoPipeline(config, gemini_client=mock_client)

        results = [
            {
                "path": Path("a.jpg"),
                "burst_size": 3,
                "burst_index": 0,
                "classification": {
                    "classification": "Share", "score": 5,
                    "contains_children": True, "is_appropriate": True,
                },
            },
        ]
        routed = pipeline.route(results)
        assert routed[0]["destination"] == "Share"


# ── Document pipeline ────────────────────────────────────────────────


class TestDocumentPipeline:

    def test_collect_files(self, tmp_path, config):
        (tmp_path / "doc.pdf").write_text("doc")
        (tmp_path / "doc.docx").write_text("doc")
        (tmp_path / "photo.jpg").write_text("img")
        (tmp_path / "code.py").write_text("code")

        pipeline = DocumentPipeline(config)
        files = pipeline.collect_files(tmp_path)
        extensions = {f.suffix.lower() for f in files}
        assert ".pdf" in extensions
        assert ".py" in extensions  # Code files are documents
        assert ".jpg" not in extensions

    def test_route_with_profile(self, config):
        profile = TasteProfile(
            name="test", description="test", media_types=["document"],
            categories=[
                CategoryDefinition("Strong", "good"),
                CategoryDefinition("Weak", "bad"),
            ],
            default_category="Weak",
            thresholds={"Strong": 4},
        )
        pipeline = DocumentPipeline(config, profile=profile)

        results = [
            {
                "path": Path("a.pdf"),
                "classification": {"classification": "Strong", "score": 4},
            },
            {
                "path": Path("b.pdf"),
                "classification": {"classification": "Strong", "score": 3},
            },
        ]
        routed = pipeline.route(results)
        assert routed[0]["destination"] == "Strong"
        assert routed[1]["destination"] == "Weak"  # Below threshold

    def test_group_files_disabled(self, config):
        from taster.core.profiles import DocumentProfileSettings
        profile = TasteProfile(
            name="test", description="test", media_types=["document"],
            categories=[CategoryDefinition("A", "a")],
            document_settings=DocumentProfileSettings(enable_similarity_grouping=False),
        )
        pipeline = DocumentPipeline(config, profile=profile)
        files = [Path("a.pdf"), Path("b.pdf")]
        groups = pipeline.group_files(files, {})
        assert groups == [[Path("a.pdf")], [Path("b.pdf")]]

    def test_group_files_empty(self, config):
        pipeline = DocumentPipeline(config)
        assert pipeline.group_files([], {}) == []


# ── Mixed pipeline ───────────────────────────────────────────────────


class TestMixedPipeline:

    def test_collect_all_media(self, tmp_path, config):
        (tmp_path / "photo.jpg").write_text("img")
        (tmp_path / "video.mp4").write_text("vid")
        (tmp_path / "doc.pdf").write_text("doc")

        pipeline = MixedPipeline(config)
        files = pipeline.collect_files(tmp_path)
        assert len(files) == 3

    def test_stub_methods(self, config):
        pipeline = MixedPipeline(config)
        assert pipeline.extract_features([]) == {}
        assert pipeline.group_files([Path("a")], {}) == [[Path("a")]]
        assert pipeline.classify([], {}) == []
        assert pipeline.route([{"destination": "X"}]) == [{"destination": "X"}]

    def test_run_empty(self, tmp_path, config, capsys):
        empty = tmp_path / "empty"
        empty.mkdir()
        output = tmp_path / "output"

        pipeline = MixedPipeline(config)
        result = pipeline.run(empty, output)
        captured = capsys.readouterr()
        assert "No media files found" in captured.out

    def test_dispatches_to_sub_pipelines(self, tmp_path, config):
        """Verify mixed pipeline creates and runs sub-pipelines."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "photo.jpg").write_text("img")

        mock_client = MagicMock()
        pipeline = MixedPipeline(config, gemini_client=mock_client)

        # Mock the sub-pipeline's run method
        mock_photo_result = ClassificationResult(
            results=[{"path": Path("photo.jpg"), "destination": "Share"}],
            stats={"Share": 1},
        )
        with patch.object(PhotoPipeline, "run", return_value=mock_photo_result):
            result = pipeline.run(src, tmp_path / "output")
            assert result.stats.get("Share", 0) == 1

    def test_copy_videos_no_classify(self, tmp_path, config):
        """Test _copy_videos path when classify_videos=False."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "video.mp4").write_bytes(b"fake video")

        pipeline = PhotoPipeline(config)
        result = pipeline._copy_videos(
            [src / "video.mp4"], tmp_path / "output", dry_run=True
        )
        assert result.stats == {"Videos": 1}
        assert result.results[0]["destination"] == "Videos"
