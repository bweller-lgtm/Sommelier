"""Tests for classification pipeline components."""
import pytest
from pathlib import Path

from taster.core import load_config, CacheManager, GeminiClient
from taster.classification import PromptBuilder, MediaClassifier, Router


class TestPromptBuilder:
    """Tests for PromptBuilder."""

    @pytest.fixture
    def prompt_builder(self):
        """Create prompt builder instance."""
        config = load_config(Path("config.yaml"))
        return PromptBuilder(config)

    def test_initialization(self, prompt_builder):
        """Test prompt builder initialization."""
        assert prompt_builder is not None
        assert prompt_builder.config is not None

    def test_singleton_prompt(self, prompt_builder):
        """Test singleton prompt generation."""
        prompt = prompt_builder.build_singleton_prompt()

        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "share" in prompt.lower()
        assert "storage" in prompt.lower()
        assert "ignore" in prompt.lower()
        assert "young children" in prompt.lower()

    def test_burst_prompt(self, prompt_builder):
        """Test burst prompt generation."""
        prompt = prompt_builder.build_burst_prompt(burst_size=5)

        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "BURST" in prompt
        assert "5 photos" in prompt
        assert "rank" in prompt.lower()

    def test_video_prompt(self, prompt_builder):
        """Test video prompt generation."""
        prompt = prompt_builder.build_video_prompt()

        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "VIDEO" in prompt
        assert "audio" in prompt.lower()


class TestRouter:
    """Tests for Router."""

    @pytest.fixture
    def router(self):
        """Create router instance."""
        config = load_config(Path("config.yaml"))
        return Router(config)

    def test_initialization(self, router):
        """Test router initialization."""
        assert router is not None
        assert router.config is not None

    def test_route_singleton_share(self, router):
        """Test routing high-score Share photo."""
        classification = {
            "classification": "Share",
            "score": 4,
            "contains_children": True,
            "is_appropriate": True
        }

        destination = router.route_singleton(classification)
        assert destination == "Share"

    def test_route_singleton_low_score(self, router):
        """Test routing mid-score Share photo goes to Review."""
        classification = {
            "classification": "Share",
            "score": 3,
            "contains_children": True,
            "is_appropriate": True
        }

        destination = router.route_singleton(classification)
        assert destination == "Review"

    def test_route_singleton_no_children(self, router):
        """Test routing photo without children."""
        classification = {
            "classification": "Share",
            "score": 5,
            "contains_children": False,
            "is_appropriate": True
        }

        destination = router.route_singleton(classification)
        assert destination == "Ignore"

    def test_route_singleton_inappropriate(self, router):
        """Test routing inappropriate photo."""
        classification = {
            "classification": "Share",
            "score": 5,
            "contains_children": True,
            "is_appropriate": False
        }

        destination = router.route_singleton(classification)
        assert destination == "Review"

    def test_route_singleton_storage(self, router):
        """Test routing Storage photo."""
        classification = {
            "classification": "Storage",
            "score": 2,
            "contains_children": True,
            "is_appropriate": True
        }

        destination = router.route_singleton(classification)
        assert destination == "Storage"


def test_component_imports():
    """Test that all classification components can be imported."""
    from taster.classification import PromptBuilder, MediaClassifier, Router

    assert PromptBuilder is not None
    assert MediaClassifier is not None
    assert Router is not None


# ── Custom profile tests (document/resume screening) ────────────────


class TestCustomProfileClassification:
    """Tests for classification with custom profiles (e.g., resume screening)."""

    @pytest.fixture
    def resume_profile(self):
        """Create a resume screening taste profile."""
        from taster.core.profiles import TasteProfile, CategoryDefinition
        return TasteProfile(
            name="resume-screener",
            description="Screen resumes for engineering candidates",
            media_types=["document"],
            categories=[
                CategoryDefinition(name="Strong", description="Excellent candidates to interview"),
                CategoryDefinition(name="Maybe", description="Worth a second look"),
                CategoryDefinition(name="Pass", description="Not a match for this role"),
            ],
            default_category="Maybe",
            top_priorities=[
                "Relevant engineering experience",
                "Track record of increasing responsibility",
            ],
            positive_criteria={
                "must_have": ["Relevant technical experience"],
                "highly_valued": ["Leadership experience", "Open source contributions"],
            },
            negative_criteria={
                "deal_breakers": ["No relevant experience"],
            },
            philosophy="Focus on demonstrated impact over credentials.",
            thresholds={"Strong": 4, "Maybe": 2},
        )

    @pytest.fixture
    def config(self):
        return load_config(Path("config.yaml"))

    def test_prompt_uses_custom_categories(self, config, resume_profile):
        """PromptBuilder should use profile's categories, not defaults."""
        pb = PromptBuilder(config, profile=resume_profile)
        prompt = pb.build_singleton_prompt(media_type="document")

        assert "strong" in prompt.lower()
        assert "maybe" in prompt.lower()
        assert "pass" in prompt.lower()
        # Should NOT contain default photo categories
        assert "share" not in prompt.lower() or "sharepoint" in prompt.lower()

    def test_prompt_includes_criteria(self, config, resume_profile):
        """PromptBuilder should include profile criteria in document prompts."""
        pb = PromptBuilder(config, profile=resume_profile)
        prompt = pb.build_singleton_prompt(media_type="document")

        assert "relevant" in prompt.lower()
        assert "impact" in prompt.lower() or "philosophy" in prompt.lower() or "experience" in prompt.lower()

    def test_classifier_valid_categories(self, config, resume_profile):
        """Classifier should accept profile's categories as valid."""
        from unittest.mock import MagicMock
        mock_client = MagicMock()
        pb = PromptBuilder(config, profile=resume_profile)
        classifier = MediaClassifier(config, mock_client, pb, profile=resume_profile)

        assert "Strong" in classifier._valid_categories
        assert "Maybe" in classifier._valid_categories
        assert "Pass" in classifier._valid_categories
        # "Review" should NOT be in valid categories (not a profile category)
        assert "Review" not in classifier._valid_categories

    def test_classifier_default_category(self, config, resume_profile):
        """Classifier should use profile's default_category, not 'Review'."""
        from unittest.mock import MagicMock
        mock_client = MagicMock()
        pb = PromptBuilder(config, profile=resume_profile)
        classifier = MediaClassifier(config, mock_client, pb, profile=resume_profile)

        assert classifier._default_category == "Maybe"

    def test_fallback_uses_profile_default(self, config, resume_profile):
        """Fallback response should use profile's default_category."""
        from unittest.mock import MagicMock
        mock_client = MagicMock()
        pb = PromptBuilder(config, profile=resume_profile)
        classifier = MediaClassifier(config, mock_client, pb, profile=resume_profile)

        fallback = classifier._create_fallback_response("API error")
        assert fallback["classification"] == "Maybe"
        assert fallback["is_error_fallback"] is True

    def test_validation_remaps_invalid_category(self, config, resume_profile):
        """Validator should remap invalid categories to profile default."""
        from unittest.mock import MagicMock
        mock_client = MagicMock()
        pb = PromptBuilder(config, profile=resume_profile)
        classifier = MediaClassifier(config, mock_client, pb, profile=resume_profile)

        response = {"classification": "Share", "score": 4, "reasoning": "test"}
        validated = classifier._validate_singleton_response(response)
        assert validated["classification"] == "Maybe"  # remapped to default

    def test_validation_keeps_valid_category(self, config, resume_profile):
        """Validator should keep valid profile categories."""
        from unittest.mock import MagicMock
        mock_client = MagicMock()
        pb = PromptBuilder(config, profile=resume_profile)
        classifier = MediaClassifier(config, mock_client, pb, profile=resume_profile)

        response = {"classification": "Strong", "score": 5, "reasoning": "great candidate"}
        validated = classifier._validate_singleton_response(response)
        assert validated["classification"] == "Strong"

    def test_document_validation_remaps_invalid(self, config, resume_profile):
        """Document validator should remap invalid categories to profile default."""
        from unittest.mock import MagicMock
        mock_client = MagicMock()
        pb = PromptBuilder(config, profile=resume_profile)
        classifier = MediaClassifier(config, mock_client, pb, profile=resume_profile)

        response = {"classification": "Exemplary", "score": 4, "reasoning": "test"}
        validated = classifier._validate_document_response(response)
        assert validated["classification"] == "Maybe"

    def test_router_respects_profile_categories(self, config, resume_profile):
        """Router should accept profile categories for documents."""
        router = Router(config, profile=resume_profile)

        classification = {"classification": "Strong", "score": 4}
        dest = router.route_document(classification)
        assert dest == "Strong"

    def test_router_applies_profile_thresholds(self, config, resume_profile):
        """Router should apply profile thresholds to documents."""
        router = Router(config, profile=resume_profile)

        # Below "Strong" threshold of 4
        classification = {"classification": "Strong", "score": 3}
        dest = router.route_document(classification)
        assert dest == "Maybe"  # falls to default

    def test_router_invalid_category_falls_to_default(self, config, resume_profile):
        """Router should remap invalid categories to profile default."""
        router = Router(config, profile=resume_profile)

        classification = {"classification": "Share", "score": 5}
        dest = router.route_document(classification)
        assert dest == "Maybe"
