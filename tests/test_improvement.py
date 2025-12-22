"""Tests for photo improvement module (gray zone detection and enhancement)."""
import pytest
import tempfile
import csv
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import yaml

from src.core.config import Config, PhotoImprovementConfig, load_config
from src.improvement import PhotoImprover, launch_review_ui
from src.classification import Router


class TestPhotoImprovementConfig:
    """Tests for PhotoImprovementConfig."""

    def test_defaults(self):
        """Test default values."""
        config = PhotoImprovementConfig()
        assert config.enabled is True
        assert config.contextual_value_threshold == "high"
        assert config.min_issues_for_candidate == 1
        assert config.cost_per_image == 0.134  # Gemini 3 Pro Image pricing
        assert config.parallel_workers == 5
        assert config.max_retries == 2
        assert config.model_name == "gemini-3-pro-image-preview"
        assert config.max_output_tokens == 8192
        assert config.review_after_sort is True

    def test_custom_values(self):
        """Test custom values."""
        config = PhotoImprovementConfig(
            enabled=False,
            contextual_value_threshold="medium",
            min_issues_for_candidate=2,
            cost_per_image=0.05,
            parallel_workers=10
        )
        assert config.enabled is False
        assert config.contextual_value_threshold == "medium"
        assert config.min_issues_for_candidate == 2
        assert config.cost_per_image == 0.05
        assert config.parallel_workers == 10

    def test_contextual_value_validation(self):
        """Test contextual_value_threshold validation."""
        # Valid values
        config = PhotoImprovementConfig(contextual_value_threshold="high")
        assert config.contextual_value_threshold == "high"

        config = PhotoImprovementConfig(contextual_value_threshold="medium")
        assert config.contextual_value_threshold == "medium"

        # Invalid value
        with pytest.raises(ValueError, match="contextual_value_threshold"):
            PhotoImprovementConfig(contextual_value_threshold="invalid")

    def test_min_issues_validation(self):
        """Test min_issues_for_candidate validation."""
        with pytest.raises(ValueError, match="min_issues_for_candidate"):
            PhotoImprovementConfig(min_issues_for_candidate=0)

    def test_parallel_workers_validation(self):
        """Test parallel_workers validation."""
        with pytest.raises(ValueError, match="parallel_workers"):
            PhotoImprovementConfig(parallel_workers=0)


class TestConfigWithImprovement:
    """Tests for Config with photo_improvement section."""

    def test_config_has_photo_improvement(self):
        """Test that Config has photo_improvement attribute."""
        config = Config()
        assert hasattr(config, 'photo_improvement')
        assert isinstance(config.photo_improvement, PhotoImprovementConfig)

    def test_load_config_with_improvement(self, tmp_path):
        """Test loading config with photo_improvement section."""
        config_data = {
            "model": {"name": "gemini-3-flash-preview"},
            "photo_improvement": {
                "enabled": True,
                "contextual_value_threshold": "medium",
                "cost_per_image": 0.05,
            },
        }

        config_file = tmp_path / "test_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(config_file)
        assert config.photo_improvement.enabled is True
        assert config.photo_improvement.contextual_value_threshold == "medium"
        assert config.photo_improvement.cost_per_image == 0.05

    def test_from_dict_with_improvement(self):
        """Test creating config from dict with photo_improvement."""
        data = {
            "model": {"name": "test-model"},
            "photo_improvement": {
                "enabled": False,
                "min_issues_for_candidate": 2,
            },
        }

        config = Config.from_dict(data)
        assert config.photo_improvement.enabled is False
        assert config.photo_improvement.min_issues_for_candidate == 2


class TestRouterImprovement:
    """Tests for Router improvement routing."""

    @pytest.fixture
    def router(self):
        """Create router instance."""
        config = load_config(Path("config.yaml"))
        return Router(config)

    def test_should_route_to_improvement_true(self, router):
        """Test photo routes to improvement candidates."""
        classification = {
            "classification": "Storage",
            "confidence": 0.6,
            "contains_children": True,
            "is_appropriate": True,
            "improvement_candidate": True,
            "improvement_reasons": ["motion_blur", "focus_blur"],
            "contextual_value": "high",
            "contextual_value_reasoning": "Rare family moment",
        }

        result = router.should_route_to_improvement(classification)
        assert result is True

    def test_should_route_to_improvement_disabled(self, router):
        """Test routing disabled when config disabled."""
        router.config.photo_improvement.enabled = False

        classification = {
            "improvement_candidate": True,
            "improvement_reasons": ["motion_blur"],
            "contextual_value": "high",
        }

        result = router.should_route_to_improvement(classification)
        assert result is False

    def test_should_route_to_improvement_not_candidate(self, router):
        """Test routing when not marked as candidate."""
        classification = {
            "improvement_candidate": False,
            "improvement_reasons": [],
            "contextual_value": "low",
        }

        result = router.should_route_to_improvement(classification)
        assert result is False

    def test_should_route_to_improvement_low_contextual_value(self, router):
        """Test routing when contextual value below threshold."""
        router.config.photo_improvement.contextual_value_threshold = "high"

        classification = {
            "improvement_candidate": True,
            "improvement_reasons": ["motion_blur"],
            "contextual_value": "medium",  # Below "high" threshold
        }

        result = router.should_route_to_improvement(classification)
        assert result is False

    def test_should_route_to_improvement_medium_threshold(self, router):
        """Test routing with medium contextual value threshold."""
        router.config.photo_improvement.contextual_value_threshold = "medium"

        classification = {
            "improvement_candidate": True,
            "improvement_reasons": ["motion_blur"],
            "contextual_value": "medium",
        }

        result = router.should_route_to_improvement(classification)
        assert result is True

    def test_should_route_to_improvement_insufficient_issues(self, router):
        """Test routing when insufficient technical issues."""
        router.config.photo_improvement.min_issues_for_candidate = 2

        classification = {
            "improvement_candidate": True,
            "improvement_reasons": ["motion_blur"],  # Only 1 issue
            "contextual_value": "high",
        }

        result = router.should_route_to_improvement(classification)
        assert result is False

    def test_route_singleton_with_improvement(self, router):
        """Test route_singleton routes to ImprovementCandidates when applicable."""
        classification = {
            "classification": "Storage",
            "confidence": 0.6,
            "contains_children": True,
            "is_appropriate": True,
            "improvement_candidate": True,
            "improvement_reasons": ["motion_blur"],
            "contextual_value": "high",
        }

        destination = router.route_singleton(classification)
        # Should route to ImprovementCandidates when photo qualifies
        assert destination == "ImprovementCandidates"

    def test_route_singleton_normal_without_improvement(self, router):
        """Test route_singleton routes normally when not improvement candidate."""
        classification = {
            "classification": "Storage",
            "confidence": 0.6,
            "contains_children": True,
            "is_appropriate": True,
            "improvement_candidate": False,
            "contextual_value": "low",
        }

        destination = router.route_singleton(classification)
        # Should route to normal destination
        assert destination in ["Share", "Storage", "Review", "Ignore"]


class TestPhotoImprover:
    """Tests for PhotoImprover."""

    @pytest.fixture
    def config(self):
        """Create config instance."""
        return load_config(Path("config.yaml"))

    @pytest.fixture
    def improver(self, config):
        """Create improver instance."""
        return PhotoImprover(config)

    def test_initialization(self, improver, config):
        """Test improver initialization."""
        assert improver.config == config
        assert improver.parallel_workers == config.photo_improvement.parallel_workers
        assert improver.max_retries == config.photo_improvement.max_retries
        assert improver.cost_per_image == config.photo_improvement.cost_per_image

    def test_estimate_cost(self, improver):
        """Test cost estimation."""
        candidates = [
            {"filename": "photo1.jpg"},
            {"filename": "photo2.jpg"},
            {"filename": "photo3.jpg"},
        ]

        cost = improver.estimate_cost(candidates)
        expected = 3 * improver.cost_per_image
        assert cost == expected

    def test_estimate_cost_empty(self, improver):
        """Test cost estimation for empty list."""
        cost = improver.estimate_cost([])
        assert cost == 0

    def test_craft_improvement_prompt_motion_blur(self, improver):
        """Test prompt crafting for motion blur."""
        candidate = {
            "filename": "test.jpg",
            "improvement_reasons": "motion_blur",
            "contextual_reasoning": "Parent catching child mid-jump",
        }

        prompt = improver.craft_improvement_prompt(candidate)

        assert "motion blur" in prompt.lower() or "motion" in prompt.lower()
        assert "catching child" in prompt
        assert "family photo" in prompt.lower()
        assert "same people" in prompt.lower()  # Identity preservation

    def test_craft_improvement_prompt_multiple_issues(self, improver):
        """Test prompt crafting for multiple issues."""
        candidate = {
            "filename": "test.jpg",
            "improvement_reasons": "motion_blur, noise, underexposed",
            "contextual_reasoning": "Rare four-generation photo",
        }

        prompt = improver.craft_improvement_prompt(candidate)

        # Should mention multiple improvements
        assert len(prompt) > 100  # Substantial prompt
        assert "four-generation" in prompt

    def test_craft_improvement_prompt_unknown_issues(self, improver):
        """Test prompt crafting with unknown issues."""
        candidate = {
            "filename": "test.jpg",
            "improvement_reasons": "unknown_issue",
            "contextual_reasoning": "Some context",
        }

        prompt = improver.craft_improvement_prompt(candidate)

        # Should fall back to generic enhancement
        assert "enhance" in prompt.lower()

    def test_load_candidates_empty(self, improver, tmp_path):
        """Test loading candidates from non-existent folder."""
        candidates = improver.load_candidates(tmp_path / "nonexistent")
        assert candidates == []

    def test_load_candidates_no_csv(self, improver, tmp_path):
        """Test loading candidates when no CSV exists."""
        candidates_folder = tmp_path / "ImprovementCandidates"
        candidates_folder.mkdir()

        candidates = improver.load_candidates(candidates_folder)
        assert candidates == []

    def test_load_candidates_with_approved(self, improver, tmp_path):
        """Test loading only approved candidates."""
        candidates_folder = tmp_path / "ImprovementCandidates"
        candidates_folder.mkdir()

        csv_path = candidates_folder / "improvement_candidates.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "approved"])
            writer.writeheader()
            writer.writerow({"filename": "photo1.jpg", "approved": "Y"})
            writer.writerow({"filename": "photo2.jpg", "approved": "N"})
            writer.writerow({"filename": "photo3.jpg", "approved": "Y"})
            writer.writerow({"filename": "photo4.jpg", "approved": ""})

        candidates = improver.load_candidates(candidates_folder)

        assert len(candidates) == 2
        filenames = [c["filename"] for c in candidates]
        assert "photo1.jpg" in filenames
        assert "photo3.jpg" in filenames
        assert "photo2.jpg" not in filenames
        assert "photo4.jpg" not in filenames

    def test_improve_photo_no_client(self, improver, tmp_path):
        """Test improve_photo fails gracefully without client."""
        # Create a test file so the file check passes
        test_file = tmp_path / "test.jpg"
        test_file.write_bytes(b"fake image data")

        candidate = {"filename": "test.jpg"}

        success, status, path = improver.improve_photo(
            candidate, tmp_path, tmp_path / "Improved"
        )

        assert success is False
        assert "not initialized" in status

    def test_improve_photo_missing_file(self, improver, tmp_path):
        """Test improve_photo handles missing source file."""
        improver.client = Mock()
        candidate = {"filename": "nonexistent.jpg"}

        success, status, path = improver.improve_photo(
            candidate, tmp_path, tmp_path / "Improved"
        )

        assert success is False
        assert "not found" in status


class TestReviewUI:
    """Tests for review UI functions."""

    def test_load_candidates_from_csv(self, tmp_path):
        """Test loading candidates from CSV."""
        from src.improvement.review_ui import load_candidates_from_csv

        candidates_folder = tmp_path / "ImprovementCandidates"
        candidates_folder.mkdir()

        csv_path = candidates_folder / "improvement_candidates.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["filename", "contextual_value", "approved"]
            )
            writer.writeheader()
            writer.writerow({
                "filename": "photo1.jpg",
                "contextual_value": "high",
                "approved": "",
            })
            writer.writerow({
                "filename": "photo2.jpg",
                "contextual_value": "medium",
                "approved": "Y",
            })

        candidates = load_candidates_from_csv(candidates_folder)

        assert len(candidates) == 2
        assert candidates[0]["filename"] == "photo1.jpg"
        assert candidates[1]["contextual_value"] == "medium"

    def test_load_candidates_no_csv(self, tmp_path):
        """Test loading candidates when no CSV exists."""
        from src.improvement.review_ui import load_candidates_from_csv

        candidates = load_candidates_from_csv(tmp_path)
        assert candidates == []

    def test_save_candidates_to_csv(self, tmp_path):
        """Test saving candidates to CSV."""
        from src.improvement.review_ui import save_candidates_to_csv

        candidates_folder = tmp_path / "ImprovementCandidates"
        candidates_folder.mkdir()

        candidates = [
            {"filename": "photo1.jpg", "approved": "Y"},
            {"filename": "photo2.jpg", "approved": "N"},
        ]

        save_candidates_to_csv(candidates, candidates_folder)

        csv_path = candidates_folder / "improvement_candidates.csv"
        assert csv_path.exists()

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["approved"] == "Y"
        assert rows[1]["approved"] == "N"

    def test_save_candidates_empty(self, tmp_path):
        """Test saving empty candidates list."""
        from src.improvement.review_ui import save_candidates_to_csv

        candidates_folder = tmp_path / "ImprovementCandidates"
        candidates_folder.mkdir()

        save_candidates_to_csv([], candidates_folder)

        csv_path = candidates_folder / "improvement_candidates.csv"
        # Should not create file for empty list
        assert not csv_path.exists()


class TestGeminiImageClient:
    """Tests for GeminiImageClient."""

    def test_initialization_no_api_key(self):
        """Test initialization fails without API key."""
        from src.core.models import GeminiImageClient

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="GEMINI_API_KEY"):
                GeminiImageClient(api_key=None)

    def test_initialization_with_api_key(self):
        """Test initialization with API key."""
        from src.core.models import GeminiImageClient

        with patch("google.genai.Client"):
            client = GeminiImageClient(api_key="test_key")
            assert client.api_key == "test_key"
            assert client.model_name == "gemini-3-pro-image-preview"
            assert client.max_output_tokens == 8192

    def test_initialization_custom_model(self):
        """Test initialization with custom model."""
        from src.core.models import GeminiImageClient

        with patch("google.genai.Client"):
            client = GeminiImageClient(
                api_key="test_key",
                model_name="custom-model",
                max_output_tokens=4096
            )
            assert client.model_name == "custom-model"
            assert client.max_output_tokens == 4096


class TestPromptBuilderGrayZone:
    """Tests for PromptBuilder gray zone detection."""

    @pytest.fixture
    def prompt_builder(self):
        """Create prompt builder instance."""
        from src.classification import PromptBuilder
        config = load_config(Path("config.yaml"))
        return PromptBuilder(config)

    def test_singleton_prompt_includes_gray_zone(self, prompt_builder):
        """Test singleton prompt includes gray zone section when enabled."""
        prompt_builder.config.photo_improvement.enabled = True
        prompt = prompt_builder.build_singleton_prompt()

        assert "improvement_candidate" in prompt.lower() or "gray zone" in prompt.lower() or "contextual_value" in prompt.lower()

    def test_singleton_prompt_excludes_gray_zone(self, prompt_builder):
        """Test singleton prompt excludes gray zone when disabled."""
        prompt_builder.config.photo_improvement.enabled = False
        prompt = prompt_builder.build_singleton_prompt()

        # Should not have improvement-specific fields when disabled
        assert "improvement_candidate" not in prompt.lower() or "IMPROVEMENT" not in prompt


class TestImprovementIntegration:
    """Integration tests for improvement workflow."""

    def test_full_workflow_candidates_csv(self, tmp_path):
        """Test creating and reading improvement candidates CSV."""
        candidates_folder = tmp_path / "ImprovementCandidates"
        candidates_folder.mkdir()

        # Simulate creating candidates during classification
        candidates = [
            {
                "filename": "family_moment.jpg",
                "original_path": "/photos/family_moment.jpg",
                "classification": "Storage",
                "contextual_value": "high",
                "contextual_value_reasoning": "Rare three-generation moment",
                "improvement_candidate": "True",
                "improvement_reasons": "motion_blur, noise",
                "estimated_cost": "$0.039",
                "approved": "",
            }
        ]

        csv_path = candidates_folder / "improvement_candidates.csv"
        fieldnames = list(candidates[0].keys())

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(candidates)

        # Simulate approving
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        rows[0]["approved"] = "Y"

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        # Verify improver can load approved candidates
        config = load_config(Path("config.yaml"))
        improver = PhotoImprover(config)
        approved = improver.load_candidates(candidates_folder)

        assert len(approved) == 1
        assert approved[0]["filename"] == "family_moment.jpg"
        assert approved[0]["contextual_value"] == "high"


def test_improvement_module_imports():
    """Test that all improvement module components can be imported."""
    from src.improvement import PhotoImprover, launch_review_ui
    from src.improvement.improver import PhotoImprover as PI
    from src.improvement.review_ui import (
        load_candidates_from_csv,
        save_candidates_to_csv,
        create_review_interface,
    )

    assert PhotoImprover is not None
    assert launch_review_ui is not None
    assert PI is not None
    assert load_candidates_from_csv is not None
    assert save_candidates_to_csv is not None
    assert create_review_interface is not None


def test_gemini_image_client_import():
    """Test that GeminiImageClient can be imported."""
    from src.core.models import GeminiImageClient

    assert GeminiImageClient is not None
