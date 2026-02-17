"""Tests for MCP pairwise training tool handlers."""

import json
import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.core.profiles import ProfileManager
from src.training.session import TrainingSession


@pytest.fixture
def profiles_dir(tmp_path):
    """Create a temporary profiles directory."""
    d = tmp_path / "profiles"
    d.mkdir()
    return d


@pytest.fixture
def pm(profiles_dir):
    """Create a ProfileManager with temp directory."""
    return ProfileManager(profiles_dir)


@pytest.fixture
def sample_folder(tmp_path):
    """Create a folder with sample image files."""
    folder = tmp_path / "photos"
    folder.mkdir()
    for i in range(10):
        (folder / f"photo_{i:03d}.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
    return folder


@pytest.fixture
def active_session(profiles_dir):
    """Create an active training session with comparisons ready."""
    bursts = [
        ["/a.jpg", "/b.jpg", "/c.jpg"],
        ["/d.jpg", "/e.jpg"],
    ]
    singletons = ["/f.jpg", "/g.jpg", "/h.jpg"]
    session = TrainingSession.create(
        profile_name="test-profile",
        folder_path="/photos",
        bursts=bursts,
        singletons=singletons,
    )
    session.current_comparison = {
        "type": "pairwise_within",
        "photo_a": "/a.jpg",
        "photo_b": "/b.jpg",
        "comparison_type": "within_burst",
        "context": "Same burst",
    }
    session.comparisons_served = 1
    session.save(profiles_dir)
    return session


@pytest.fixture
def gallery_session(profiles_dir):
    """Create a session with a gallery comparison ready."""
    bursts = [[f"/{i}.jpg" for i in range(10)]]
    session = TrainingSession.create(
        profile_name="gallery-test",
        folder_path="/photos",
        bursts=bursts,
        singletons=[],
    )
    session.current_comparison = {
        "type": "gallery",
        "photos": [f"/{i}.jpg" for i in range(10)],
        "context": "Burst of 10 photos",
    }
    session.comparisons_served = 1
    session.save(profiles_dir)
    return session


class TestHandleStartTraining:
    def test_start_training_with_skip_embeddings(self, pm, sample_folder):
        from src.mcp.server import _handle_start_training

        with patch("src.mcp.server._get_config"):
            with patch("src.mcp.server._get_profile_manager", return_value=pm):
                result = _handle_start_training(pm, {
                    "folder_path": str(sample_folder),
                    "profile_name": "test-train",
                    "skip_embeddings": True,
                })

        assert "session_id" in result
        assert result["total_photos"] == 10
        assert "first_comparison" in result

    def test_start_training_nonexistent_folder(self, pm):
        from src.mcp.server import _handle_start_training

        result = _handle_start_training(pm, {
            "folder_path": "/nonexistent",
            "profile_name": "test",
        })
        assert "error" in result

    def test_start_training_empty_folder(self, pm, tmp_path):
        from src.mcp.server import _handle_start_training

        empty = tmp_path / "empty"
        empty.mkdir()

        result = _handle_start_training(pm, {
            "folder_path": str(empty),
            "profile_name": "test",
            "skip_embeddings": True,
        })
        assert "error" in result


class TestHandleGetComparison:
    def test_get_comparison(self, pm, active_session):
        from src.mcp.server import _handle_get_comparison

        result = _handle_get_comparison(pm, {
            "session_id": active_session.session_id,
        })
        assert "type" in result or "status" in result
        assert "stats" in result

    def test_get_comparison_invalid_session(self, pm):
        from src.mcp.server import _handle_get_comparison

        with pytest.raises(FileNotFoundError):
            _handle_get_comparison(pm, {"session_id": "nonexistent"})

    def test_get_comparison_completed_session(self, pm, profiles_dir):
        from src.mcp.server import _handle_get_comparison

        session = TrainingSession.create("test", "/photos", [], ["/a.jpg"])
        session.status = "completed"
        session.save(profiles_dir)

        result = _handle_get_comparison(pm, {"session_id": session.session_id})
        assert "error" in result


class TestHandleSubmitComparison:
    def test_submit_comparison(self, pm, active_session):
        from src.mcp.server import _handle_submit_comparison

        result = _handle_submit_comparison(pm, {
            "session_id": active_session.session_id,
            "choice": "left",
            "reason": "Photo A is sharper and has better expressions",
        })

        assert result["status"] == "recorded"
        assert "stats" in result
        assert result["stats"]["pairwise_count"] == 1

    def test_submit_comparison_records_reason(self, pm, active_session, profiles_dir):
        from src.mcp.server import _handle_submit_comparison

        _handle_submit_comparison(pm, {
            "session_id": active_session.session_id,
            "choice": "right",
            "reason": "Better lighting",
        })

        loaded = TrainingSession.load(active_session.session_id, profiles_dir)
        assert loaded.pairwise[0].reason == "Better lighting"
        assert loaded.pairwise[0].choice == "right"

    def test_submit_without_active_comparison(self, pm, profiles_dir):
        from src.mcp.server import _handle_submit_comparison

        session = TrainingSession.create("test", "/photos", [], ["/a.jpg", "/b.jpg"])
        session.current_comparison = None
        session.save(profiles_dir)

        result = _handle_submit_comparison(pm, {
            "session_id": session.session_id,
            "choice": "left",
            "reason": "test",
        })
        assert "error" in result

    def test_piggybacks_next_comparison(self, pm, active_session):
        from src.mcp.server import _handle_submit_comparison

        result = _handle_submit_comparison(pm, {
            "session_id": active_session.session_id,
            "choice": "both",
            "reason": "Both are great",
        })

        # Should include next_comparison in response
        assert "next_comparison" in result


class TestHandleSubmitGallery:
    def test_submit_gallery(self, pm, gallery_session):
        from src.mcp.server import _handle_submit_gallery

        result = _handle_submit_gallery(pm, {
            "session_id": gallery_session.session_id,
            "selected_indices": [0, 3, 7],
            "reason": "Best expressions and focus",
        })

        assert result["status"] == "recorded"
        assert "stats" in result

    def test_submit_gallery_records_selections(self, pm, gallery_session, profiles_dir):
        from src.mcp.server import _handle_submit_gallery

        _handle_submit_gallery(pm, {
            "session_id": gallery_session.session_id,
            "selected_indices": [1, 4],
            "reason": "Sharp and well-composed",
        })

        loaded = TrainingSession.load(gallery_session.session_id, profiles_dir)
        assert len(loaded.gallery) == 1
        assert loaded.gallery[0].selected_indices == [1, 4]
        assert loaded.gallery[0].reason == "Sharp and well-composed"

    def test_submit_gallery_without_active_gallery(self, pm, active_session):
        from src.mcp.server import _handle_submit_gallery

        # active_session has a pairwise comparison, not gallery
        result = _handle_submit_gallery(pm, {
            "session_id": active_session.session_id,
            "selected_indices": [0],
            "reason": "test",
        })
        assert "error" in result


class TestHandleTrainingStatus:
    def test_status_specific_session(self, pm, active_session):
        from src.mcp.server import _handle_training_status

        result = _handle_training_status(pm, {
            "session_id": active_session.session_id,
        })

        assert result["session_id"] == active_session.session_id
        assert result["total_photos"] == 8
        assert result["pairwise_count"] == 0

    def test_status_list_all(self, pm, active_session, profiles_dir):
        from src.mcp.server import _handle_training_status

        result = _handle_training_status(pm, {})
        assert "sessions" in result
        assert len(result["sessions"]) >= 1

    def test_status_no_sessions(self, pm):
        from src.mcp.server import _handle_training_status

        result = _handle_training_status(pm, {})
        assert result["sessions"] == []
        assert "message" in result


class TestHandleSynthesizeProfile:
    def test_synthesize_not_enough_data(self, pm, active_session):
        from src.mcp.server import _handle_synthesize_profile

        result = _handle_synthesize_profile(pm, {
            "session_id": active_session.session_id,
        })
        assert "error" in result
        assert "15" in result["error"]

    @patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key"})
    def test_synthesize_with_enough_data(self, pm, profiles_dir):
        from src.mcp.server import _handle_synthesize_profile

        # Create session with 15+ labels
        session = TrainingSession.create(
            "synth-test", "/photos",
            bursts=[], singletons=[f"/{i}.jpg" for i in range(30)]
        )
        for i in range(15):
            session.add_pairwise(
                f"/{i*2}.jpg", f"/{i*2+1}.jpg", "left", f"reason {i}", "between_burst"
            )
        session.save(profiles_dir)

        mock_client = MagicMock()
        mock_client.generate_json.return_value = {
            "description": "Test profile",
            "media_types": ["image"],
            "categories": [
                {"name": "Share", "description": "Worth sharing"},
                {"name": "Storage", "description": "Keep in storage"},
            ],
            "default_category": "Storage",
            "top_priorities": ["Expression quality"],
            "positive_criteria": {"must_have": ["Clear faces"]},
            "negative_criteria": {"deal_breakers": ["Blurry"]},
            "specific_guidance": ["Check focus"],
            "philosophy": "Share the best moments",
        }
        mock_client.generate.return_value = MagicMock(
            parse_json=MagicMock(return_value=None)
        )

        with patch("src.core.provider_factory.create_ai_client", return_value=mock_client):
            with patch("src.mcp.server._get_config"):
                result = _handle_synthesize_profile(pm, {
                    "session_id": session.session_id,
                })

        assert result["status"] == "created"
        assert "profile" in result
        assert "training_summary" in result


class TestHandleRefineProfile:
    @patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key"})
    def test_refine_profile(self, pm):
        from src.mcp.server import _handle_refine_profile

        # Create a profile to refine
        pm.create_profile(
            name="refine-test",
            description="Test profile",
            media_types=["image"],
            categories=[
                {"name": "Share", "description": "Worth sharing"},
                {"name": "Storage", "description": "Storage"},
            ],
            top_priorities=["Quality"],
            positive_criteria={"must_have": ["Clear"]},
            negative_criteria={"deal_breakers": ["Blurry"]},
        )

        mock_client = MagicMock()
        mock_client.generate_json.return_value = {
            "top_priorities": ["Expression quality", "Composition"],
            "positive_criteria": {"must_have": ["Clear faces", "Good expressions"]},
            "negative_criteria": {"deal_breakers": ["Blurry", "Eyes closed"]},
            "specific_guidance": ["Pay attention to expressions"],
            "changes_made": ["Added expression-related criteria"],
        }

        with patch("src.core.provider_factory.create_ai_client", return_value=mock_client):
            with patch("src.mcp.server._get_config"):
                result = _handle_refine_profile(pm, {
                    "profile_name": "refine-test",
                    "corrections": [
                        {
                            "file_path": "/photo1.jpg",
                            "original_category": "Storage",
                            "correct_category": "Share",
                            "reason": "Great expressions, should be shared",
                        },
                    ],
                })

        assert result["status"] == "refined"
        assert result["corrections_applied"] == 1

    def test_refine_nonexistent_profile(self, pm):
        from src.mcp.server import _handle_refine_profile

        with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key"}):
            result = _handle_refine_profile(pm, {
                "profile_name": "nonexistent",
                "corrections": [{"file_path": "/a.jpg", "original_category": "A", "correct_category": "B"}],
            })
        assert "error" in result

    def test_refine_no_corrections(self, pm):
        from src.mcp.server import _handle_refine_profile

        pm.create_profile(
            name="empty-corrections",
            description="Test",
            media_types=["image"],
            categories=[{"name": "A", "description": "A"}],
        )

        with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key"}):
            result = _handle_refine_profile(pm, {
                "profile_name": "empty-corrections",
                "corrections": [],
            })
        assert "error" in result
