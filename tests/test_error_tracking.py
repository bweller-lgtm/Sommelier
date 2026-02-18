"""Tests for error tracking and retry functionality."""
import pytest
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from sommelier.core import load_config, CacheManager
from sommelier.classification import PromptBuilder, MediaClassifier


class TestFallbackResponse:
    """Tests for _create_fallback_response with error tracking."""

    @pytest.fixture
    def classifier(self, tmp_path):
        """Create classifier instance with mocked dependencies."""
        config = load_config(Path("config.yaml"))
        cache_manager = CacheManager(tmp_path / ".cache", enabled=True)

        # Mock Gemini client
        mock_client = Mock()
        mock_client.generate_json = Mock(return_value={
            "classification": "Share",
            "confidence": 0.8,
            "reasoning": "Test"
        })

        prompt_builder = PromptBuilder(config)

        return MediaClassifier(config, mock_client, prompt_builder, cache_manager)

    def test_fallback_has_error_fields(self, classifier):
        """Test that fallback response contains error tracking fields."""
        result = classifier._create_fallback_response("Test error")

        assert "is_error_fallback" in result
        assert "error_type" in result
        assert "error_message" in result
        assert "retry_count" in result

    def test_fallback_defaults(self, classifier):
        """Test default values in fallback response."""
        result = classifier._create_fallback_response("Test error")

        assert result["is_error_fallback"] == True
        assert result["error_type"] == "api_error"
        assert result["error_message"] == "Test error"
        assert result["retry_count"] == 0
        assert result["classification"] == "Review"
        assert result["confidence"] == 0.3

    def test_fallback_with_error_type(self, classifier):
        """Test fallback with custom error type."""
        result = classifier._create_fallback_response(
            "Image load failed",
            error_type="load_error"
        )

        assert result["error_type"] == "load_error"
        assert result["error_message"] == "Image load failed"

    def test_fallback_with_rank(self, classifier):
        """Test fallback response with rank for burst photos."""
        result = classifier._create_fallback_response(
            "Burst error",
            rank=3,
            error_type="invalid_response"
        )

        assert result["rank"] == 3
        assert result["error_type"] == "invalid_response"

    def test_fallback_video_response(self, classifier):
        """Test fallback response for video."""
        result = classifier._create_fallback_response(
            "Video error",
            is_video=True
        )

        assert result["audio_quality"] == "unknown"
        assert result["is_error_fallback"] == True


class TestValidationWithErrorFields:
    """Tests for response validation with error tracking fields."""

    @pytest.fixture
    def classifier(self, tmp_path):
        """Create classifier instance."""
        config = load_config(Path("config.yaml"))
        cache_manager = CacheManager(tmp_path / ".cache", enabled=True)
        mock_client = Mock()
        prompt_builder = PromptBuilder(config)
        return MediaClassifier(config, mock_client, prompt_builder, cache_manager)

    def test_successful_response_marked_not_error(self, classifier):
        """Test that successful responses have is_error_fallback = False."""
        response = {
            "classification": "Share",
            "confidence": 0.85,
            "reasoning": "Good photo"
        }

        validated = classifier._validate_singleton_response(response)

        assert validated["is_error_fallback"] == False
        assert validated["error_type"] is None
        assert validated["error_message"] is None
        assert validated["retry_count"] == 0

    def test_error_fallback_preserved(self, classifier):
        """Test that error fallback fields are preserved through validation."""
        response = {
            "classification": "Review",
            "confidence": 0.3,
            "reasoning": "Fallback",
            "is_error_fallback": True,
            "error_type": "api_error",
            "error_message": "Connection failed",
            "retry_count": 2
        }

        validated = classifier._validate_singleton_response(response)

        assert validated["is_error_fallback"] == True
        assert validated["error_type"] == "api_error"
        assert validated["error_message"] == "Connection failed"
        assert validated["retry_count"] == 2


class TestRetryLogic:
    """Tests for _execute_with_retry wrapper."""

    @pytest.fixture
    def classifier(self, tmp_path):
        """Create classifier with specific retry config."""
        config = load_config(Path("config.yaml"))
        # Set fast retry for testing
        config.classification.classification_retries = 2
        config.classification.retry_delay_seconds = 0.01  # Very short for tests

        cache_manager = CacheManager(tmp_path / ".cache", enabled=True)
        mock_client = Mock()
        prompt_builder = PromptBuilder(config)

        return MediaClassifier(config, mock_client, prompt_builder, cache_manager)

    def test_successful_first_try(self, classifier):
        """Test successful classification on first try."""
        def success_fn():
            return {
                "classification": "Share",
                "confidence": 0.85,
                "is_error_fallback": False
            }

        result = classifier._execute_with_retry(success_fn, "Test")

        assert result["classification"] == "Share"
        assert result["retry_count"] == 0

    def test_retry_on_retriable_error(self, classifier):
        """Test that retriable errors trigger retry."""
        call_count = [0]

        def failing_then_success():
            call_count[0] += 1
            if call_count[0] < 2:
                return {
                    "classification": "Review",
                    "confidence": 0.3,
                    "is_error_fallback": True,
                    "error_type": "api_error"
                }
            return {
                "classification": "Share",
                "confidence": 0.85,
                "is_error_fallback": False
            }

        result = classifier._execute_with_retry(failing_then_success, "Test")

        assert call_count[0] == 2
        assert result["classification"] == "Share"
        assert result["retry_count"] == 1

    def test_no_retry_on_non_retriable_error(self, classifier):
        """Test that non-retriable errors don't trigger retry."""
        call_count = [0]

        def load_error():
            call_count[0] += 1
            return {
                "classification": "Review",
                "confidence": 0.3,
                "is_error_fallback": True,
                "error_type": "load_error"  # Not in retry_on_errors
            }

        result = classifier._execute_with_retry(load_error, "Test")

        assert call_count[0] == 1  # No retry
        assert result["error_type"] == "load_error"

    def test_max_retries_exhausted(self, classifier):
        """Test behavior when all retries are exhausted."""
        call_count = [0]

        def always_fail():
            call_count[0] += 1
            return {
                "classification": "Review",
                "confidence": 0.3,
                "is_error_fallback": True,
                "error_type": "api_error"
            }

        result = classifier._execute_with_retry(always_fail, "Test")

        assert call_count[0] == 3  # Initial + 2 retries
        assert result["is_error_fallback"] == True
        assert result["retry_count"] == 2  # Max retries

    def test_exception_handling_with_retry(self, classifier):
        """Test that exceptions are caught and retried."""
        call_count = [0]

        def raises_then_success():
            call_count[0] += 1
            if call_count[0] < 2:
                raise Exception("Connection timeout")
            return {
                "classification": "Share",
                "confidence": 0.85,
                "is_error_fallback": False
            }

        result = classifier._execute_with_retry(raises_then_success, "Test")

        assert call_count[0] == 2
        assert result["classification"] == "Share"


class TestBurstContextStorage:
    """Tests for burst context storage."""

    @pytest.fixture
    def cache_manager(self, tmp_path):
        """Create cache manager."""
        return CacheManager(tmp_path / ".cache", enabled=True)

    def test_burst_context_cache_exists(self, cache_manager):
        """Test that burst_context cache type exists."""
        assert "burst_context" in cache_manager.caches

    def test_burst_context_storage(self, cache_manager):
        """Test storing and retrieving burst context."""
        context = {
            "burst_id": "test123",
            "photo_paths": ["/path/1.jpg", "/path/2.jpg"],
            "results": [
                {"classification": "Share", "confidence": 0.8},
                {"classification": "Storage", "confidence": 0.5}
            ],
            "failed_indices": [],
            "has_failures": False
        }

        cache_manager.set("burst_context", "test123", context)
        retrieved = cache_manager.get("burst_context", "test123")

        assert retrieved is not None
        assert retrieved["burst_id"] == "test123"
        assert len(retrieved["photo_paths"]) == 2
        assert len(retrieved["results"]) == 2


class TestPromptBuilderWithBurstContext:
    """Tests for context-aware prompt building."""

    @pytest.fixture
    def prompt_builder(self):
        """Create prompt builder."""
        config = load_config(Path("config.yaml"))
        return PromptBuilder(config)

    def test_singleton_with_burst_context(self, prompt_builder):
        """Test singleton prompt with burst context."""
        prompt = prompt_builder.build_singleton_with_burst_context(
            burst_size=5,
            sibling_classifications=["Share", "Storage", "Storage"],
            original_position=2
        )

        assert "BURST CONTEXT" in prompt
        assert "5 photos" in prompt
        assert "#3" in prompt  # Position 2 is photo #3
        assert "Share" in prompt
        assert "Storage" in prompt

    def test_singleton_with_empty_siblings(self, prompt_builder):
        """Test context prompt with no sibling info."""
        prompt = prompt_builder.build_singleton_with_burst_context(
            burst_size=3,
            sibling_classifications=[],
            original_position=0
        )

        assert "BURST CONTEXT" in prompt
        assert "3 photos" in prompt
        assert "#1" in prompt

    def test_context_truncation_for_large_bursts(self, prompt_builder):
        """Test that large sibling lists are truncated."""
        many_siblings = ["Share"] * 10

        prompt = prompt_builder.build_singleton_with_burst_context(
            burst_size=12,
            sibling_classifications=many_siblings,
            original_position=5
        )

        assert "+5 more" in prompt  # Shows truncation indicator


class TestConfigRetrySettings:
    """Tests for retry configuration."""

    def test_retry_config_defaults(self):
        """Test default retry configuration values."""
        config = load_config(Path("config.yaml"))

        assert hasattr(config.classification, "classification_retries")
        assert hasattr(config.classification, "retry_delay_seconds")
        assert hasattr(config.classification, "retry_on_errors")

    def test_retry_config_values(self):
        """Test retry configuration values from config file."""
        config = load_config(Path("config.yaml"))

        assert config.classification.classification_retries >= 0
        assert config.classification.retry_delay_seconds > 0
        assert "api_error" in config.classification.retry_on_errors


class TestErrorTypes:
    """Tests for error type taxonomy."""

    @pytest.fixture
    def classifier(self, tmp_path):
        """Create classifier."""
        config = load_config(Path("config.yaml"))
        cache_manager = CacheManager(tmp_path / ".cache", enabled=True)
        mock_client = Mock()
        prompt_builder = PromptBuilder(config)
        return MediaClassifier(config, mock_client, prompt_builder, cache_manager)

    def test_all_error_types_valid(self, classifier):
        """Test all defined error types produce valid fallback responses."""
        error_types = [
            "api_error",
            "load_error",
            "safety_blocked",
            "invalid_response",
            "timeout",
            "rate_limit"
        ]

        for error_type in error_types:
            result = classifier._create_fallback_response(
                f"Test {error_type}",
                error_type=error_type
            )

            assert result["is_error_fallback"] == True
            assert result["error_type"] == error_type
            assert result["classification"] == "Review"
