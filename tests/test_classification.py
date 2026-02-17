"""Tests for classification pipeline components."""
import pytest
from pathlib import Path

from src.core import load_config, CacheManager, GeminiClient
from src.classification import PromptBuilder, MediaClassifier, Router


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
        """Test routing high-confidence Share photo."""
        classification = {
            "classification": "Share",
            "confidence": 0.85,
            "contains_children": True,
            "is_appropriate": True
        }

        destination = router.route_singleton(classification)
        assert destination == "Share"

    def test_route_singleton_low_confidence(self, router):
        """Test routing low-confidence Share photo."""
        classification = {
            "classification": "Share",
            "confidence": 0.45,
            "contains_children": True,
            "is_appropriate": True
        }

        destination = router.route_singleton(classification)
        assert destination == "Review"

    def test_route_singleton_no_children(self, router):
        """Test routing photo without children."""
        classification = {
            "classification": "Share",
            "confidence": 0.90,
            "contains_children": False,
            "is_appropriate": True
        }

        destination = router.route_singleton(classification)
        assert destination == "Ignore"

    def test_route_singleton_inappropriate(self, router):
        """Test routing inappropriate photo."""
        classification = {
            "classification": "Share",
            "confidence": 0.90,
            "contains_children": True,
            "is_appropriate": False
        }

        destination = router.route_singleton(classification)
        assert destination == "Review"

    def test_route_singleton_storage(self, router):
        """Test routing Storage photo."""
        classification = {
            "classification": "Storage",
            "confidence": 0.50,
            "contains_children": True,
            "is_appropriate": True
        }

        destination = router.route_singleton(classification)
        assert destination == "Storage"


def test_component_imports():
    """Test that all classification components can be imported."""
    from src.classification import PromptBuilder, MediaClassifier, Router

    assert PromptBuilder is not None
    assert MediaClassifier is not None
    assert Router is not None
