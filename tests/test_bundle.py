"""Tests for bundle (multi-file) classification."""

import os
import json
import shutil
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from taster.core.file_utils import FileTypeRegistry
from taster.core.config import Config
from taster.core.ai_client import AIClient, AIResponse
from taster.cli import _build_parser


# ── Bundle discovery ─────────────────────────────────────────────────


class TestListBundles:
    """Tests for FileTypeRegistry.list_bundles()."""

    def test_discovers_subfolders(self, tmp_path):
        (tmp_path / "alice").mkdir()
        (tmp_path / "alice" / "resume.txt").write_text("resume")
        (tmp_path / "bob").mkdir()
        (tmp_path / "bob" / "cover.txt").write_text("cover")
        bundles = FileTypeRegistry.list_bundles(tmp_path)
        assert len(bundles) == 2
        names = [b["name"] for b in bundles]
        assert "alice" in names
        assert "bob" in names

    def test_skips_empty_subdirs(self, tmp_path):
        (tmp_path / "alice").mkdir()
        (tmp_path / "alice" / "resume.txt").write_text("resume")
        (tmp_path / "empty_dir").mkdir()
        bundles = FileTypeRegistry.list_bundles(tmp_path)
        assert len(bundles) == 1
        assert bundles[0]["name"] == "alice"

    def test_ignores_loose_files(self, tmp_path):
        (tmp_path / "loose.txt").write_text("not a bundle")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "doc.txt").write_text("in a bundle")
        bundles = FileTypeRegistry.list_bundles(tmp_path)
        assert len(bundles) == 1
        assert bundles[0]["name"] == "sub"

    def test_returns_correct_structure(self, tmp_path):
        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / "readme.txt").write_text("hello")
        (tmp_path / "pkg" / "notes.md").write_text("notes")
        bundles = FileTypeRegistry.list_bundles(tmp_path)
        assert len(bundles) == 1
        b = bundles[0]
        assert b["name"] == "pkg"
        assert b["path"] == tmp_path / "pkg"
        assert "documents" in b["files"]
        assert len(b["files"]["documents"]) == 2

    def test_nonexistent_dir(self, tmp_path):
        bundles = FileTypeRegistry.list_bundles(tmp_path / "nope")
        assert bundles == []

    def test_sorted_by_name(self, tmp_path):
        for name in ["charlie", "alice", "bob"]:
            (tmp_path / name).mkdir()
            (tmp_path / name / "doc.txt").write_text("x")
        bundles = FileTypeRegistry.list_bundles(tmp_path)
        names = [b["name"] for b in bundles]
        assert names == ["alice", "bob", "charlie"]

    def test_mixed_media_types(self, tmp_path):
        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / "photo.jpg").write_bytes(b"\xff\xd8\xff")
        (tmp_path / "pkg" / "doc.txt").write_text("text")
        bundles = FileTypeRegistry.list_bundles(tmp_path)
        b = bundles[0]
        assert len(b["files"]["images"]) == 1
        assert len(b["files"]["documents"]) == 1


# ── Bundle prompt ────────────────────────────────────────────────────


class TestBundlePrompt:
    """Tests for PromptBuilder.build_bundle_prompt()."""

    def test_includes_file_labels(self):
        from taster.classification.prompt_builder import PromptBuilder
        config = Config()
        pb = PromptBuilder(config)
        prompt = pb.build_bundle_prompt(["resume.pdf", "cover.docx", "portfolio.pdf"])
        assert "resume.pdf" in prompt
        assert "cover.docx" in prompt
        assert "portfolio.pdf" in prompt

    def test_includes_holistic_instruction(self):
        from taster.classification.prompt_builder import PromptBuilder
        config = Config()
        pb = PromptBuilder(config)
        prompt = pb.build_bundle_prompt(["file1.txt"])
        assert "holistically" in prompt.lower() or "COMPLETE package" in prompt

    def test_includes_file_count(self):
        from taster.classification.prompt_builder import PromptBuilder
        config = Config()
        pb = PromptBuilder(config)
        prompt = pb.build_bundle_prompt(["a.txt", "b.txt", "c.txt"])
        assert "3 FILES" in prompt

    def test_includes_json_format(self):
        from taster.classification.prompt_builder import PromptBuilder
        config = Config()
        pb = PromptBuilder(config)
        prompt = pb.build_bundle_prompt(["a.txt"])
        assert "files_reviewed" in prompt
        assert "classification" in prompt
        assert "score" in prompt

    def test_uses_profile_categories(self):
        from taster.classification.prompt_builder import PromptBuilder
        from taster.core.profiles import TasteProfile, CategoryDefinition
        profile = TasteProfile(
            name="test",
            description="test",
            media_types=["document"],
            categories=[
                CategoryDefinition(name="Advance", description="Move forward"),
                CategoryDefinition(name="Reject", description="Pass"),
            ],
        )
        config = Config()
        pb = PromptBuilder(config, profile=profile)
        prompt = pb.build_bundle_prompt(["a.txt"])
        assert "Advance" in prompt
        assert "Reject" in prompt


# ── Bundle classifier ────────────────────────────────────────────────


class TestClassifyBundle:
    """Tests for MediaClassifier.classify_bundle()."""

    @pytest.fixture
    def mock_client(self):
        client = MagicMock(spec=AIClient)
        client.provider_name = "mock"
        client.generate_json.return_value = {
            "classification": "Share",
            "score": 4,
            "reasoning": "Strong package overall",
            "content_summary": "Resume and cover letter",
            "key_topics": ["experience", "skills"],
            "files_reviewed": ["resume.txt", "cover.txt"],
        }
        return client

    def test_returns_single_classification(self, mock_client, tmp_path):
        from taster.classification.prompt_builder import PromptBuilder
        from taster.classification.classifier import MediaClassifier
        config = Config()
        pb = PromptBuilder(config)
        classifier = MediaClassifier(config, mock_client, pb)

        (tmp_path / "resume.txt").write_text("I am qualified")
        (tmp_path / "cover.txt").write_text("Dear hiring manager")
        files = [tmp_path / "resume.txt", tmp_path / "cover.txt"]

        result = classifier.classify_bundle("alice", files, use_cache=False)
        assert result["classification"] == "Share"
        assert result["score"] == 4
        assert "reasoning" in result

    def test_calls_generate_json_once(self, mock_client, tmp_path):
        from taster.classification.prompt_builder import PromptBuilder
        from taster.classification.classifier import MediaClassifier
        config = Config()
        pb = PromptBuilder(config)
        classifier = MediaClassifier(config, mock_client, pb)

        (tmp_path / "doc.txt").write_text("content")
        files = [tmp_path / "doc.txt"]

        classifier.classify_bundle("test", files, use_cache=False)
        assert mock_client.generate_json.call_count == 1

    def test_empty_bundle_returns_fallback(self):
        from taster.classification.prompt_builder import PromptBuilder
        from taster.classification.classifier import MediaClassifier
        config = Config()
        client = MagicMock(spec=AIClient)
        client.provider_name = "mock"
        pb = PromptBuilder(config)
        classifier = MediaClassifier(config, client, pb)

        result = classifier.classify_bundle("empty", [], use_cache=False)
        assert result["is_error_fallback"] is True

    def test_caches_with_from_files_key(self, mock_client, tmp_path):
        from taster.classification.prompt_builder import PromptBuilder
        from taster.classification.classifier import MediaClassifier
        from taster.core.cache import CacheManager

        config = Config()
        pb = PromptBuilder(config)
        cache = CacheManager(tmp_path / "cache", enabled=True)
        classifier = MediaClassifier(config, mock_client, pb, cache_manager=cache)

        (tmp_path / "doc.txt").write_text("content")
        files = [tmp_path / "doc.txt"]

        # First call
        classifier.classify_bundle("test", files, use_cache=True)
        # Second call should use cache
        classifier.classify_bundle("test", files, use_cache=True)
        assert mock_client.generate_json.call_count == 1


# ── Bundle move_files ────────────────────────────────────────────────


class TestBundleMoveFiles:
    """Tests for base pipeline move_files with bundle results."""

    def test_copies_all_bundle_files(self, tmp_path):
        from taster.pipelines.base import ClassificationPipeline

        # Create a concrete subclass for testing
        class DummyPipeline(ClassificationPipeline):
            def collect_files(self, folder): return []
            def extract_features(self, files): return {}
            def group_files(self, files, features): return []
            def classify(self, groups, features): return []
            def route(self, results): return results

        config = Config()
        pipeline = DummyPipeline(config)

        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "resume.txt").write_text("resume")
        (src_dir / "cover.txt").write_text("cover")

        out_dir = tmp_path / "out"
        results = [{
            "bundle": True,
            "bundle_name": "alice",
            "files": [src_dir / "resume.txt", src_dir / "cover.txt"],
            "destination": "Advance",
        }]

        pipeline.move_files(results, out_dir)

        assert (out_dir / "Advance" / "alice" / "resume.txt").exists()
        assert (out_dir / "Advance" / "alice" / "cover.txt").exists()


# ── CLI flag ─────────────────────────────────────────────────────────


class TestBundleCLI:
    """Tests for --bundles CLI flag."""

    def test_bundles_flag_parses(self):
        parser = _build_parser()
        args = parser.parse_args(["classify", "/f", "--bundles"])
        assert args.bundles is True

    def test_bundles_flag_defaults_false(self):
        parser = _build_parser()
        args = parser.parse_args(["classify", "/f"])
        assert args.bundles is False
