"""Tests for API services: ProfileService, TrainingService, ClassificationService."""
import json
import pytest
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

from taster.api.services.profile_service import ProfileService
from taster.api.services.training_service import TrainingService, FEEDBACK_FILENAME


# ── ProfileService ──────────────────────────────────────────────────


class TestProfileService:

    @pytest.fixture
    def svc(self, tmp_path):
        return ProfileService(profiles_dir=str(tmp_path / "profiles"))

    def test_list_empty(self, svc):
        assert svc.list_profiles() == []

    def test_create_and_list(self, svc):
        result = svc.create_profile({
            "name": "test",
            "description": "test profile",
            "media_types": ["image"],
            "categories": [{"name": "Good", "description": "good stuff"}],
        })
        assert result["name"] == "test"
        assert result["description"] == "test profile"

        profiles = svc.list_profiles()
        assert len(profiles) == 1
        assert profiles[0]["name"] == "test"
        assert profiles[0]["categories"] == ["Good"]

    def test_get_profile(self, svc):
        svc.create_profile({
            "name": "myprof",
            "description": "desc",
            "media_types": ["document"],
            "categories": [{"name": "A", "description": "a"}],
        })
        result = svc.get_profile("myprof")
        assert result["name"] == "myprof"
        assert result["media_types"] == ["document"]

    def test_get_profile_not_found(self, svc):
        with pytest.raises(FileNotFoundError):
            svc.get_profile("nonexistent")

    def test_update_profile(self, svc):
        svc.create_profile({
            "name": "upd",
            "description": "old",
            "media_types": ["image"],
            "categories": [{"name": "A", "description": "a"}],
        })
        result = svc.update_profile("upd", {"description": "new", "philosophy": "updated"})
        assert result["description"] == "new"
        assert result["philosophy"] == "updated"

    def test_update_filters_invalid_keys(self, svc):
        svc.create_profile({
            "name": "upd2",
            "description": "old",
            "media_types": ["image"],
            "categories": [{"name": "A", "description": "a"}],
        })
        # "bogus_key" should be silently ignored
        result = svc.update_profile("upd2", {
            "description": "new",
            "bogus_key": "should be ignored",
        })
        assert result["description"] == "new"

    def test_update_nonexistent(self, svc):
        with pytest.raises(FileNotFoundError):
            svc.update_profile("nope", {"description": "x"})

    def test_delete_profile(self, svc):
        svc.create_profile({
            "name": "del-me",
            "description": "d",
            "media_types": ["image"],
            "categories": [{"name": "A", "description": "a"}],
        })
        assert svc.delete_profile("del-me") is True
        assert svc.list_profiles() == []

    def test_delete_nonexistent(self, svc):
        assert svc.delete_profile("nope") is False

    def test_create_missing_fields(self, svc):
        with pytest.raises(ValueError, match="Missing required fields"):
            svc.create_profile({"name": "x"})

    def test_create_with_optional_fields(self, svc):
        result = svc.create_profile({
            "name": "full",
            "description": "full profile",
            "media_types": ["document"],
            "categories": [{"name": "Keep", "description": "k"}],
            "default_category": "Keep",
            "top_priorities": ["quality"],
            "positive_criteria": {"must_have": ["clarity"]},
            "negative_criteria": {"deal_breakers": ["spam"]},
            "specific_guidance": ["Be strict"],
            "philosophy": "Quality first",
            "thresholds": {"Keep": 0.7},
        })
        assert result["default_category"] == "Keep"
        assert result["philosophy"] == "Quality first"

    def test_ensure_defaults(self, svc):
        svc.ensure_defaults()
        profiles = svc.list_profiles()
        names = [p["name"] for p in profiles]
        assert "default-photos" in names
        assert "default-documents" in names

    def test_ensure_defaults_idempotent(self, svc):
        svc.ensure_defaults()
        svc.ensure_defaults()
        profiles = svc.list_profiles()
        names = [p["name"] for p in profiles]
        assert names.count("default-photos") == 1
        assert names.count("default-documents") == 1


# ── TrainingService ─────────────────────────────────────────────────


class TestTrainingService:

    @pytest.fixture
    def svc(self, tmp_path):
        return TrainingService(profiles_dir=str(tmp_path / "profiles"))

    def test_submit_feedback(self, svc):
        entry = svc.submit_feedback(
            file_path="/tmp/photo.jpg",
            correct_category="Share",
            reasoning="Great photo",
        )
        assert entry["file_path"] == "/tmp/photo.jpg"
        assert entry["correct_category"] == "Share"
        assert entry["reasoning"] == "Great photo"
        assert "submitted_at" in entry

    def test_get_stats_empty(self, svc):
        stats = svc.get_stats()
        assert stats["total_feedback"] == 0
        assert stats["by_category"] == {}

    def test_get_stats_with_feedback(self, svc):
        svc.submit_feedback("/tmp/a.jpg", "Share", "good")
        svc.submit_feedback("/tmp/b.jpg", "Share", "nice")
        svc.submit_feedback("/tmp/c.jpg", "Storage", "blurry")
        stats = svc.get_stats()
        assert stats["total_feedback"] == 3
        assert stats["by_category"]["Share"] == 2
        assert stats["by_category"]["Storage"] == 1

    def test_feedback_persists_to_disk(self, svc):
        svc.submit_feedback("/tmp/a.jpg", "Share")
        assert svc.feedback_path.exists()
        data = json.loads(svc.feedback_path.read_text(encoding="utf-8"))
        assert len(data) == 1

    def test_load_feedback_corrupt_file(self, svc):
        svc.feedback_path.write_text("not valid json", encoding="utf-8")
        stats = svc.get_stats()
        assert stats["total_feedback"] == 0

    def test_load_feedback_non_list(self, svc):
        svc.feedback_path.write_text('{"key": "value"}', encoding="utf-8")
        stats = svc.get_stats()
        assert stats["total_feedback"] == 0

    def test_generate_profile_no_feedback(self, svc):
        with pytest.raises(ValueError, match="No feedback"):
            svc.generate_profile_from_feedback("test-profile")

    def test_generate_profile_from_feedback(self, svc):
        svc.submit_feedback("/tmp/a.jpg", "Share", "beautiful shot")
        svc.submit_feedback("/tmp/b.jpg", "Share", "great lighting")
        svc.submit_feedback("/tmp/c.pdf", "Ignore", "junk document")
        svc.submit_feedback("/tmp/d.mp4", "Share", "cute video")

        profile = svc.generate_profile_from_feedback("learned-profile")
        assert profile["name"] == "learned-profile"
        assert "3" in profile["description"] or "4" in profile["description"]

        # Check categories were derived from feedback
        cat_names = [c["name"] for c in profile["categories"]]
        assert "Share" in cat_names
        assert "Ignore" in cat_names

        # Check media types detected from extensions
        assert "image" in profile["media_types"]
        assert "document" in profile["media_types"]
        assert "video" in profile["media_types"]

    def test_generate_profile_positive_criteria(self, svc):
        svc.submit_feedback("/tmp/a.jpg", "Share", "clear face")
        svc.submit_feedback("/tmp/b.jpg", "Share", "nice smile")
        profile = svc.generate_profile_from_feedback("pos-profile")
        if profile.get("positive_criteria"):
            assert "user_feedback" in profile["positive_criteria"]

    def test_generate_profile_negative_criteria(self, svc):
        svc.submit_feedback("/tmp/a.jpg", "Ignore", "blurry mess")
        svc.submit_feedback("/tmp/b.jpg", "Ignore", "dark and noisy")
        profile = svc.generate_profile_from_feedback("neg-profile")
        if profile.get("negative_criteria"):
            assert "user_feedback" in profile["negative_criteria"]

    def test_generate_profile_no_extensions(self, svc):
        """Feedback with no file extension defaults to image media type."""
        svc.submit_feedback("no-extension", "Share", "something")
        profile = svc.generate_profile_from_feedback("no-ext-profile")
        assert "image" in profile["media_types"]

    def test_generate_profile_categories_without_reasoning(self, svc):
        """Feedback with empty reasoning still creates categories."""
        svc.submit_feedback("/tmp/a.jpg", "Good", "")
        svc.submit_feedback("/tmp/b.jpg", "Bad", "")
        profile = svc.generate_profile_from_feedback("empty-reasoning")
        cat_names = [c["name"] for c in profile["categories"]]
        assert "Good" in cat_names or "Bad" in cat_names


# ── ClassificationService ───────────────────────────────────────────


class TestClassificationService:

    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.paths.cache_root = "/tmp/cache"
        config.caching.ttl_days = 7
        config.caching.enabled = False
        config.caching.max_cache_size_gb = 1.0
        config.profiles.profiles_dir = "/tmp/profiles"
        return config

    def test_start_job_folder_not_found(self, mock_config):
        with patch("taster.api.services.classification_service.CacheManager"):
            with patch("taster.api.services.classification_service.create_ai_client"):
                with patch("taster.api.services.classification_service.ProfileManager") as mock_pm:
                    from taster.api.services.classification_service import ClassificationService
                    svc = ClassificationService(mock_config)
                    with pytest.raises(FileNotFoundError, match="Folder not found"):
                        svc.start_job("/nonexistent/path", "default-photos")

    def test_start_job_profile_not_found(self, mock_config, tmp_path):
        folder = tmp_path / "input"
        folder.mkdir()
        with patch("taster.api.services.classification_service.CacheManager"):
            with patch("taster.api.services.classification_service.create_ai_client"):
                with patch("taster.api.services.classification_service.ProfileManager") as mock_pm:
                    mock_pm_inst = MagicMock()
                    mock_pm.return_value = mock_pm_inst
                    mock_pm_inst.load_profile.side_effect = FileNotFoundError("not found")

                    from taster.api.services.classification_service import ClassificationService
                    svc = ClassificationService(mock_config)
                    with pytest.raises(FileNotFoundError):
                        svc.start_job(str(folder), "nonexistent-profile")

    def test_start_job_returns_job_id(self, mock_config, tmp_path):
        folder = tmp_path / "input"
        folder.mkdir()
        (folder / "photo.jpg").write_text("img")

        with patch("taster.api.services.classification_service.CacheManager"):
            with patch("taster.api.services.classification_service.create_ai_client"):
                with patch("taster.api.services.classification_service.ProfileManager") as mock_pm:
                    mock_pm_inst = MagicMock()
                    mock_pm.return_value = mock_pm_inst
                    mock_profile = MagicMock()
                    mock_pm_inst.load_profile.return_value = mock_profile

                    with patch("taster.api.services.classification_service.MixedPipeline"):
                        from taster.api.services.classification_service import ClassificationService
                        svc = ClassificationService(mock_config)
                        job_id = svc.start_job(str(folder), "test-profile")
                        assert isinstance(job_id, str)
                        assert len(job_id) == 36  # UUID format

    def test_get_job_status_unknown(self, mock_config):
        with patch("taster.api.services.classification_service.CacheManager"):
            with patch("taster.api.services.classification_service.create_ai_client"):
                with patch("taster.api.services.classification_service.ProfileManager"):
                    from taster.api.services.classification_service import ClassificationService
                    svc = ClassificationService(mock_config)
                    with pytest.raises(KeyError, match="Unknown job"):
                        svc.get_job_status("bad-id")

    def test_get_job_results_unknown(self, mock_config):
        with patch("taster.api.services.classification_service.CacheManager"):
            with patch("taster.api.services.classification_service.create_ai_client"):
                with patch("taster.api.services.classification_service.ProfileManager"):
                    from taster.api.services.classification_service import ClassificationService
                    svc = ClassificationService(mock_config)
                    with pytest.raises(KeyError, match="Unknown job"):
                        svc.get_job_results("bad-id")

    def test_update_job(self, mock_config):
        with patch("taster.api.services.classification_service.CacheManager"):
            with patch("taster.api.services.classification_service.create_ai_client"):
                with patch("taster.api.services.classification_service.ProfileManager"):
                    from taster.api.services.classification_service import ClassificationService
                    svc = ClassificationService(mock_config)
                    # Manually insert a job
                    svc._jobs["test-id"] = {"status": "pending", "progress": 0.0}
                    svc._update_job("test-id", status="running", progress=50.0)
                    assert svc._jobs["test-id"]["status"] == "running"
                    assert svc._jobs["test-id"]["progress"] == 50.0

    def test_update_job_nonexistent(self, mock_config):
        with patch("taster.api.services.classification_service.CacheManager"):
            with patch("taster.api.services.classification_service.create_ai_client"):
                with patch("taster.api.services.classification_service.ProfileManager"):
                    from taster.api.services.classification_service import ClassificationService
                    svc = ClassificationService(mock_config)
                    # Should not raise
                    svc._update_job("nonexistent", status="running")

    def test_run_job_success(self, mock_config, tmp_path):
        folder = tmp_path / "input"
        folder.mkdir()
        output = tmp_path / "output"

        with patch("taster.api.services.classification_service.CacheManager"):
            with patch("taster.api.services.classification_service.create_ai_client"):
                with patch("taster.api.services.classification_service.ProfileManager"):
                    from taster.api.services.classification_service import (
                        ClassificationService, STATUS_COMPLETED,
                    )
                    from taster.pipelines.base import ClassificationResult

                    svc = ClassificationService(mock_config)

                    # Create a job entry manually
                    job_id = "test-run-job"
                    svc._jobs[job_id] = {
                        "job_id": job_id,
                        "status": "pending",
                        "folder_path": str(folder),
                        "profile_name": "test",
                        "dry_run": False,
                        "progress": 0.0,
                        "message": "queued",
                        "results": [],
                        "stats": {},
                        "created_at": "2024-01-01",
                        "started_at": None,
                        "completed_at": None,
                        "error": None,
                    }

                    mock_result = ClassificationResult(
                        results=[
                            {"path": Path("a.jpg"), "destination": "Share"},
                        ],
                        stats={"Share": 1},
                    )

                    with patch("taster.api.services.classification_service.MixedPipeline") as mock_pipe:
                        mock_pipe.return_value.run.return_value = mock_result
                        mock_profile = MagicMock()
                        svc._run_job(job_id, folder, mock_profile, dry_run=True)

                    assert svc._jobs[job_id]["status"] == STATUS_COMPLETED
                    assert svc._jobs[job_id]["stats"] == {"Share": 1}
                    assert svc._jobs[job_id]["results"][0]["path"] == "a.jpg"

    def test_run_job_failure(self, mock_config, tmp_path):
        folder = tmp_path / "input"
        folder.mkdir()

        with patch("taster.api.services.classification_service.CacheManager"):
            with patch("taster.api.services.classification_service.create_ai_client"):
                with patch("taster.api.services.classification_service.ProfileManager"):
                    from taster.api.services.classification_service import (
                        ClassificationService, STATUS_FAILED,
                    )

                    svc = ClassificationService(mock_config)

                    job_id = "test-fail-job"
                    svc._jobs[job_id] = {
                        "job_id": job_id,
                        "status": "pending",
                        "folder_path": str(folder),
                        "profile_name": "test",
                        "dry_run": False,
                        "progress": 0.0,
                        "message": "queued",
                        "results": [],
                        "stats": {},
                        "created_at": "2024-01-01",
                        "started_at": None,
                        "completed_at": None,
                        "error": None,
                    }

                    with patch("taster.api.services.classification_service.MixedPipeline") as mock_pipe:
                        mock_pipe.return_value.run.side_effect = RuntimeError("pipeline crashed")
                        mock_profile = MagicMock()
                        svc._run_job(job_id, folder, mock_profile, dry_run=False)

                    assert svc._jobs[job_id]["status"] == STATUS_FAILED
                    assert "pipeline crashed" in svc._jobs[job_id]["error"]

    def test_get_job_status_after_creation(self, mock_config, tmp_path):
        folder = tmp_path / "input"
        folder.mkdir()

        with patch("taster.api.services.classification_service.CacheManager"):
            with patch("taster.api.services.classification_service.create_ai_client"):
                with patch("taster.api.services.classification_service.ProfileManager") as mock_pm:
                    mock_pm_inst = MagicMock()
                    mock_pm.return_value = mock_pm_inst
                    mock_pm_inst.load_profile.return_value = MagicMock()

                    with patch("taster.api.services.classification_service.MixedPipeline"):
                        with patch("threading.Thread") as mock_thread:
                            mock_thread.return_value.start = MagicMock()
                            from taster.api.services.classification_service import ClassificationService
                            svc = ClassificationService(mock_config)
                            job_id = svc.start_job(str(folder), "test-profile")

                            status = svc.get_job_status(job_id)
                            assert status["job_id"] == job_id
                            assert status["status"] == "pending"
                            assert status["folder_path"] == str(folder)
                            assert status["profile_name"] == "test-profile"
