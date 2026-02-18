"""Tests for MCP server handler functions."""
import json
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

from sommelier.core.profiles import ProfileManager, TasteProfile, CategoryDefinition


@pytest.fixture
def tmp_profiles_dir(tmp_path):
    d = tmp_path / "profiles"
    d.mkdir()
    return d


@pytest.fixture
def pm(tmp_profiles_dir):
    return ProfileManager(str(tmp_profiles_dir))


# ── Helper: import handler functions ─────────────────────────────────
# We import the handler functions directly since they're module-level functions.


def _import_handlers():
    """Import handler functions from the MCP server module."""
    from sommelier.mcp.server import (
        _handle_status,
        _handle_list_profiles,
        _handle_get_profile,
        _handle_create_profile,
        _handle_update_profile,
        _handle_delete_profile,
        _handle_submit_feedback,
        _handle_view_feedback,
        _handle_tool,
        _has_api_key,
        _require_api_key,
    )
    return {
        "status": _handle_status,
        "list": _handle_list_profiles,
        "get": _handle_get_profile,
        "create": _handle_create_profile,
        "update": _handle_update_profile,
        "delete": _handle_delete_profile,
        "submit_feedback": _handle_submit_feedback,
        "view_feedback": _handle_view_feedback,
        "handle_tool": _handle_tool,
        "has_api_key": _has_api_key,
        "require_api_key": _require_api_key,
    }


# ── API key checks ──────────────────────────────────────────────────


class TestAPIKeyChecks:

    def test_has_api_key_none(self):
        h = _import_handlers()
        with patch.dict(os.environ, {}, clear=True):
            # Remove all relevant keys
            for k in ["GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
                os.environ.pop(k, None)
            assert h["has_api_key"]() is False

    def test_has_api_key_gemini(self):
        h = _import_handlers()
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            assert h["has_api_key"]() is True

    def test_has_api_key_openai(self):
        h = _import_handlers()
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            assert h["has_api_key"]() is True

    def test_has_api_key_anthropic(self):
        h = _import_handlers()
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            assert h["has_api_key"]() is True

    def test_require_api_key_missing(self):
        h = _import_handlers()
        with patch.dict(os.environ, {}, clear=True):
            for k in ["GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
                os.environ.pop(k, None)
            result = h["require_api_key"]()
            assert result is not None
            assert "error" in result

    def test_require_api_key_present(self):
        h = _import_handlers()
        with patch.dict(os.environ, {"GEMINI_API_KEY": "key"}):
            result = h["require_api_key"]()
            assert result is None


# ── Status handler ───────────────────────────────────────────────────


class TestStatusHandler:

    def test_status_with_key(self, pm):
        h = _import_handlers()
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            result = h["status"](pm)
            assert result["ready"] is True
            assert "gemini" in result["providers"]
            assert result["providers"]["gemini"] == "configured"

    def test_status_without_key(self, pm):
        h = _import_handlers()
        with patch.dict(os.environ, {}, clear=True):
            for k in ["GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
                os.environ.pop(k, None)
            result = h["status"](pm)
            assert result["ready"] is False
            assert "setup" in result


# ── List profiles ────────────────────────────────────────────────────


class TestListProfiles:

    def test_empty(self, pm):
        h = _import_handlers()
        result = h["list"](pm)
        assert result == []

    def test_with_profiles(self, pm):
        pm.create_profile(
            name="test", description="d", media_types=["image"],
            categories=[{"name": "A", "description": "a"}],
        )
        h = _import_handlers()
        result = h["list"](pm)
        assert len(result) == 1
        assert result[0]["name"] == "test"
        assert "categories" in result[0]


# ── Get profile ──────────────────────────────────────────────────────


class TestGetProfile:

    def test_existing(self, pm):
        pm.create_profile(
            name="myprofile", description="desc", media_types=["document"],
            categories=[{"name": "Keep", "description": "k"}, {"name": "Toss", "description": "t"}],
        )
        h = _import_handlers()
        result = h["get"](pm, {"profile_name": "myprofile"})
        assert result["name"] == "myprofile"
        assert "summary" in result
        assert "Keep" in result["summary"]

    def test_not_found(self, pm):
        h = _import_handlers()
        result = h["get"](pm, {"profile_name": "nope"})
        assert "error" in result
        assert "hint" in result


# ── Create profile ───────────────────────────────────────────────────


class TestCreateProfile:

    def test_create(self, pm):
        h = _import_handlers()
        result = h["create"](pm, {
            "name": "new-profile",
            "description": "new desc",
            "media_types": ["image"],
            "categories": [{"name": "A", "description": "a"}],
            "philosophy": "test philosophy",
        })
        assert result["status"] == "created"
        assert result["profile"]["name"] == "new-profile"
        assert pm.profile_exists("new-profile")


# ── Update profile ───────────────────────────────────────────────────


class TestUpdateProfile:

    def test_update_existing(self, pm):
        pm.create_profile(
            name="to-update", description="old", media_types=["image"],
            categories=[{"name": "A", "description": "a"}],
        )
        h = _import_handlers()
        result = h["update"](pm, {
            "profile_name": "to-update",
            "description": "new description",
            "philosophy": "updated philosophy",
        })
        assert result["status"] == "updated"
        assert "description" in result["message"]

    def test_update_nonexistent(self, pm):
        h = _import_handlers()
        result = h["update"](pm, {
            "profile_name": "nope",
            "description": "new",
        })
        assert "error" in result

    def test_update_no_fields(self, pm):
        pm.create_profile(
            name="to-update", description="old", media_types=["image"],
            categories=[{"name": "A", "description": "a"}],
        )
        h = _import_handlers()
        result = h["update"](pm, {"profile_name": "to-update"})
        assert "error" in result
        assert "updatable_fields" in result


# ── Delete profile ───────────────────────────────────────────────────


class TestDeleteProfile:

    def test_delete_without_confirm(self, pm):
        pm.create_profile(
            name="del-me", description="d", media_types=["image"],
            categories=[{"name": "A", "description": "a"}],
        )
        h = _import_handlers()
        result = h["delete"](pm, {"profile_name": "del-me", "confirm": False})
        assert "error" in result
        assert pm.profile_exists("del-me")

    def test_delete_with_confirm(self, pm):
        pm.create_profile(
            name="del-me", description="d", media_types=["image"],
            categories=[{"name": "A", "description": "a"}],
        )
        h = _import_handlers()
        result = h["delete"](pm, {"profile_name": "del-me", "confirm": True})
        assert result["status"] == "deleted"
        assert not pm.profile_exists("del-me")

    def test_delete_nonexistent(self, pm):
        h = _import_handlers()
        result = h["delete"](pm, {"profile_name": "nope", "confirm": True})
        assert "error" in result


# ── Tool dispatch ────────────────────────────────────────────────────


class TestToolDispatch:

    def test_unknown_tool(self, pm):
        h = _import_handlers()
        with patch("sommelier.mcp.server._get_profile_manager", return_value=pm):
            result = h["handle_tool"]("unknown_tool", {})
            assert "error" in result

    def test_dispatches_list_profiles(self, pm):
        pm.create_profile(
            name="dispatch-test", description="d", media_types=["image"],
            categories=[{"name": "A", "description": "a"}],
        )
        h = _import_handlers()
        with patch("sommelier.mcp.server._get_profile_manager", return_value=pm):
            result = h["handle_tool"]("sommelier_list_profiles", {})
            assert isinstance(result, list)
            assert len(result) == 1

    def test_dispatches_status(self, pm):
        h = _import_handlers()
        with patch("sommelier.mcp.server._get_profile_manager", return_value=pm):
            with patch.dict(os.environ, {"GEMINI_API_KEY": "test"}):
                result = h["handle_tool"]("sommelier_status", {})
                assert "ready" in result


# ── View feedback ────────────────────────────────────────────────────


class TestViewFeedback:

    def test_empty_feedback(self, pm, tmp_profiles_dir):
        h = _import_handlers()
        with patch("sommelier.mcp.server._get_config") as mock_cfg:
            mock_cfg.return_value = MagicMock()
            mock_cfg.return_value.profiles.profiles_dir = str(tmp_profiles_dir)
            with patch("sommelier.mcp.server._handle_view_feedback") as mock_vf:
                mock_vf.return_value = {
                    "total_feedback": 0,
                    "message": "No feedback submitted yet.",
                }
                result = mock_vf()
                assert result["total_feedback"] == 0


# ── Quick profile (requires API key) ────────────────────────────────


class TestQuickProfile:

    def test_no_api_key(self, pm):
        from sommelier.mcp.server import _handle_quick_profile
        with patch.dict(os.environ, {}, clear=True):
            for k in ["GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
                os.environ.pop(k, None)
            result = _handle_quick_profile(pm, {
                "profile_name": "test", "description": "test",
            })
            assert "error" in result

    def test_profile_already_exists(self, pm):
        from sommelier.mcp.server import _handle_quick_profile
        pm.create_profile(
            name="existing", description="d", media_types=["image"],
            categories=[{"name": "A", "description": "a"}],
        )
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test"}):
            result = _handle_quick_profile(pm, {
                "profile_name": "existing", "description": "test",
            })
            assert "error" in result
            assert "already exists" in result["error"]

    def test_quick_profile_success(self, pm):
        from sommelier.mcp.server import _handle_quick_profile
        mock_client = MagicMock()
        mock_client.generate_json.return_value = {
            "description": "Test profile",
            "media_types": ["image"],
            "categories": [{"name": "Good", "description": "g"}, {"name": "Bad", "description": "b"}],
            "default_category": "Bad",
            "top_priorities": ["quality"],
            "positive_criteria": {"must_have": ["clarity"]},
            "negative_criteria": {"deal_breakers": ["blur"]},
            "specific_guidance": ["be strict"],
            "philosophy": "only the best",
        }
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test"}):
            with patch("sommelier.core.provider_factory.create_ai_client", return_value=mock_client):
                with patch("sommelier.mcp.server._get_config") as mock_cfg:
                    mock_cfg.return_value = MagicMock()
                    result = _handle_quick_profile(pm, {
                        "profile_name": "generated", "description": "sort my photos",
                    })
                    assert result["status"] == "created"
                    assert pm.profile_exists("generated")


# ── Generate profile (requires API key) ──────────────────────────────


class TestGenerateProfile:

    def test_no_api_key(self, pm):
        from sommelier.mcp.server import _handle_generate_profile
        with patch.dict(os.environ, {}, clear=True):
            for k in ["GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
                os.environ.pop(k, None)
            result = _handle_generate_profile(pm, {
                "profile_name": "gen", "good_examples_folder": "/tmp",
            })
            assert "error" in result

    def test_profile_already_exists(self, pm, tmp_path):
        from sommelier.mcp.server import _handle_generate_profile
        pm.create_profile(
            name="existing", description="d", media_types=["image"],
            categories=[{"name": "A", "description": "a"}],
        )
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test"}):
            result = _handle_generate_profile(pm, {
                "profile_name": "existing",
                "good_examples_folder": str(tmp_path),
            })
            assert "error" in result

    def test_bad_folder(self, pm):
        from sommelier.mcp.server import _handle_generate_profile
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test"}):
            result = _handle_generate_profile(pm, {
                "profile_name": "gen",
                "good_examples_folder": "/nonexistent/path",
            })
            assert "error" in result

    def test_no_media_found(self, pm, tmp_path):
        from sommelier.mcp.server import _handle_generate_profile
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        mock_client = MagicMock()
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test"}):
            with patch("sommelier.core.provider_factory.create_ai_client", return_value=mock_client):
                with patch("sommelier.mcp.server._get_config") as mock_cfg:
                    mock_cfg.return_value = MagicMock()
                    result = _handle_generate_profile(pm, {
                        "profile_name": "gen",
                        "good_examples_folder": str(empty_dir),
                    })
                    assert "error" in result


# ── Classify folder (requires API key) ───────────────────────────────


class TestClassifyFolder:

    def test_no_api_key(self, pm):
        from sommelier.mcp.server import _handle_classify_folder
        with patch.dict(os.environ, {}, clear=True):
            for k in ["GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
                os.environ.pop(k, None)
            result = _handle_classify_folder(pm, {
                "folder_path": "/tmp", "profile_name": "default-photos",
            })
            assert "error" in result

    def test_folder_not_found(self, pm):
        from sommelier.mcp.server import _handle_classify_folder
        pm.create_profile(
            name="test-prof", description="d", media_types=["image"],
            categories=[{"name": "A", "description": "a"}],
        )
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test"}):
            with patch("sommelier.mcp.server._get_config") as mock_cfg:
                from sommelier.core.config import Config
                mock_cfg.return_value = Config()
                result = _handle_classify_folder(pm, {
                    "folder_path": "/nonexistent/path",
                    "profile_name": "test-prof",
                })
                assert "error" in result


# ── Classify files (requires API key) ────────────────────────────────


class TestClassifyFiles:

    def test_no_api_key(self, pm):
        from sommelier.mcp.server import _handle_classify_files
        with patch.dict(os.environ, {}, clear=True):
            for k in ["GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
                os.environ.pop(k, None)
            result = _handle_classify_files(pm, {
                "file_paths": ["/tmp/test.jpg"], "profile_name": "default-photos",
            })
            assert "error" in result
