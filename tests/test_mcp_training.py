"""Tests for MCP training-related tool handlers (refine_profile).

The pairwise training tools (start_training, get_comparison, submit_comparison,
submit_gallery, training_status, synthesize_profile) were removed from the MCP
server and replaced by the standalone Gradio trainer (taste_trainer.py).
"""

import os
import pytest
from unittest.mock import patch, MagicMock

from sommelier.core.profiles import ProfileManager


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


class TestHandleRefineProfile:
    @patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key"})
    def test_refine_profile(self, pm):
        from sommelier.mcp.server import _handle_refine_profile

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

        with patch("sommelier.core.provider_factory.create_ai_client", return_value=mock_client):
            with patch("sommelier.mcp.server._get_config"):
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
        from sommelier.mcp.server import _handle_refine_profile

        with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key"}):
            result = _handle_refine_profile(pm, {
                "profile_name": "nonexistent",
                "corrections": [{"file_path": "/a.jpg", "original_category": "A", "correct_category": "B"}],
            })
        assert "error" in result

    def test_refine_no_corrections(self, pm):
        from sommelier.mcp.server import _handle_refine_profile

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
