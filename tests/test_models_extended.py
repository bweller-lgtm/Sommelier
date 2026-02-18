"""Extended tests for GeminiClient and GeminiImageClient in models.py."""
import pytest
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

from taster.core.ai_client import AIResponse


# ── AIResponse ──────────────────────────────────────────────────────


class TestAIResponse:

    def test_basic_response(self):
        r = AIResponse(text="hello", raw_response=None, blocked=False, finish_reason=1, provider="gemini")
        assert r.text == "hello"
        assert r.blocked is False
        assert r.provider == "gemini"

    def test_parse_json_clean(self):
        r = AIResponse(text='{"key": "value"}', raw_response=None, blocked=False, finish_reason=1, provider="gemini")
        result = r.parse_json()
        assert result == {"key": "value"}

    def test_parse_json_with_markdown(self):
        r = AIResponse(
            text='```json\n{"key": "value"}\n```',
            raw_response=None, blocked=False, finish_reason=1, provider="gemini",
        )
        result = r.parse_json()
        assert result == {"key": "value"}

    def test_parse_json_fallback(self):
        r = AIResponse(text="not json", raw_response=None, blocked=False, finish_reason=1, provider="gemini")
        result = r.parse_json(fallback={"default": True})
        assert result == {"default": True}

    def test_parse_json_no_fallback_raises(self):
        r = AIResponse(text="not json", raw_response=None, blocked=False, finish_reason=1, provider="gemini")
        with pytest.raises(ValueError):
            r.parse_json()

    def test_blocked_response(self):
        r = AIResponse(text="", raw_response=None, blocked=True, finish_reason=3, provider="gemini")
        assert r.blocked is True


# ── GeminiClient ────────────────────────────────────────────────────


class TestGeminiClient:

    def test_init_no_api_key(self):
        from taster.core.models import GeminiClient
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GEMINI_API_KEY", None)
            with pytest.raises(ValueError, match="GEMINI_API_KEY not set"):
                GeminiClient()

    def test_init_with_explicit_key(self):
        from taster.core.models import GeminiClient
        with patch("google.generativeai.configure"):
            client = GeminiClient(api_key="test-key-123")
            assert client.api_key == "test-key-123"
            assert client.provider_name == "gemini"

    def test_supports_video_and_pdf(self):
        from taster.core.models import GeminiClient
        with patch("google.generativeai.configure"):
            client = GeminiClient(api_key="test")
            assert client.supports_video() is True
            assert client.supports_pdf() is True

    def test_validate_response_no_candidates(self):
        from taster.core.models import GeminiClient, SafetyFilterError
        with patch("google.generativeai.configure"):
            client = GeminiClient(api_key="test")
            mock_response = MagicMock()
            mock_response.candidates = []
            with pytest.raises(SafetyFilterError, match="blocked"):
                client._validate_response(mock_response)

    def test_validate_response_safety_blocked(self):
        from taster.core.models import GeminiClient, SafetyFilterError
        with patch("google.generativeai.configure"):
            client = GeminiClient(api_key="test")
            mock_response = MagicMock()
            mock_candidate = MagicMock()
            mock_candidate.finish_reason = 3  # SAFETY
            mock_response.candidates = [mock_candidate]
            with pytest.raises(SafetyFilterError):
                client._validate_response(mock_response)

    def test_validate_response_max_tokens(self):
        from taster.core.models import GeminiClient
        with patch("google.generativeai.configure"):
            client = GeminiClient(api_key="test")
            mock_response = MagicMock()
            mock_candidate = MagicMock()
            mock_candidate.finish_reason = 2  # MAX_TOKENS
            mock_response.candidates = [mock_candidate]
            mock_response.text = "some text"
            result = client._validate_response(mock_response)
            assert result.text == "some text"

    def test_validate_response_normal(self):
        from taster.core.models import GeminiClient
        with patch("google.generativeai.configure"):
            client = GeminiClient(api_key="test")
            mock_response = MagicMock()
            mock_candidate = MagicMock()
            mock_candidate.finish_reason = 1  # STOP
            mock_response.candidates = [mock_candidate]
            mock_response.text = "result text"
            result = client._validate_response(mock_response)
            assert result.text == "result text"
            assert result.provider == "gemini"
            assert result.blocked is False

    def test_generate_safety_filter_handled(self):
        from taster.core.models import GeminiClient, SafetyFilterError
        with patch("google.generativeai.configure"):
            client = GeminiClient(api_key="test", max_retries=1)
            mock_model = MagicMock()
            mock_response = MagicMock()
            mock_response.candidates = []
            mock_model.generate_content.return_value = mock_response

            with patch.object(client, "_create_model", return_value=mock_model):
                with patch("time.sleep"):
                    result = client.generate("test prompt", handle_safety_errors=True)
                    assert result.blocked is True

    def test_generate_safety_filter_raised(self):
        from taster.core.models import GeminiClient, SafetyFilterError
        with patch("google.generativeai.configure"):
            client = GeminiClient(api_key="test", max_retries=1)
            mock_model = MagicMock()
            mock_response = MagicMock()
            mock_response.candidates = []
            mock_model.generate_content.return_value = mock_response

            with patch.object(client, "_create_model", return_value=mock_model):
                with patch("time.sleep"):
                    with pytest.raises(SafetyFilterError):
                        client.generate("test prompt", handle_safety_errors=False)

    def test_generate_json_success(self):
        from taster.core.models import GeminiClient
        with patch("google.generativeai.configure"):
            client = GeminiClient(api_key="test")
            mock_model = MagicMock()
            mock_response = MagicMock()
            mock_candidate = MagicMock()
            mock_candidate.finish_reason = 1
            mock_response.candidates = [mock_candidate]
            mock_response.text = '{"result": "success"}'
            mock_model.generate_content.return_value = mock_response

            with patch.object(client, "_create_model", return_value=mock_model):
                with patch("time.sleep"):
                    result = client.generate_json("test")
                    assert result == {"result": "success"}

    def test_generate_json_fallback(self):
        from taster.core.models import GeminiClient
        with patch("google.generativeai.configure"):
            client = GeminiClient(api_key="test")
            mock_model = MagicMock()
            mock_response = MagicMock()
            mock_candidate = MagicMock()
            mock_candidate.finish_reason = 1
            mock_response.candidates = [mock_candidate]
            mock_response.text = "not json at all"
            mock_model.generate_content.return_value = mock_response

            with patch.object(client, "_create_model", return_value=mock_model):
                with patch("time.sleep"):
                    result = client.generate_json("test", fallback={"default": True})
                    assert result == {"default": True}

    def test_generate_rate_limit_retry(self):
        from taster.core.models import GeminiClient, GeminiError
        with patch("google.generativeai.configure"):
            client = GeminiClient(api_key="test", max_retries=2, retry_delay=0.01)
            mock_model = MagicMock()
            mock_model.generate_content.side_effect = Exception("rate limit exceeded")

            with patch.object(client, "_create_model", return_value=mock_model):
                with patch("time.sleep"):
                    with pytest.raises(GeminiError, match="Failed after"):
                        client.generate("test", rate_limit_delay=0)

    def test_generate_unknown_error_fails_fast(self):
        from taster.core.models import GeminiClient, GeminiError
        with patch("google.generativeai.configure"):
            client = GeminiClient(api_key="test", max_retries=3)
            mock_model = MagicMock()
            mock_model.generate_content.side_effect = Exception("unknown bizarre error")

            with patch.object(client, "_create_model", return_value=mock_model):
                with patch("time.sleep"):
                    with pytest.raises(GeminiError, match="Gemini API error"):
                        client.generate("test", rate_limit_delay=0)

    def test_generate_timeout_retry(self):
        from taster.core.models import GeminiClient, GeminiError
        with patch("google.generativeai.configure"):
            client = GeminiClient(api_key="test", max_retries=2, retry_delay=0.01)
            mock_model = MagicMock()
            mock_model.generate_content.side_effect = Exception("request timed out")

            with patch.object(client, "_create_model", return_value=mock_model):
                with patch("time.sleep"):
                    with pytest.raises(GeminiError, match="Failed after"):
                        client.generate("test", rate_limit_delay=0)

    def test_generate_string_prompt(self):
        from taster.core.models import GeminiClient
        with patch("google.generativeai.configure"):
            client = GeminiClient(api_key="test")
            mock_model = MagicMock()
            mock_response = MagicMock()
            mock_candidate = MagicMock()
            mock_candidate.finish_reason = 1
            mock_response.candidates = [mock_candidate]
            mock_response.text = "response"
            mock_model.generate_content.return_value = mock_response

            with patch.object(client, "_create_model", return_value=mock_model):
                with patch("time.sleep"):
                    result = client.generate("simple string prompt", rate_limit_delay=0)
                    assert result.text == "response"

    def test_load_images_in_prompt_with_image_path(self):
        from taster.core.models import GeminiClient
        with patch("google.generativeai.configure"):
            client = GeminiClient(api_key="test")
            mock_img = MagicMock()

            with patch("taster.core.file_utils.ImageUtils.load_and_fix_orientation", return_value=mock_img):
                with patch("taster.core.file_utils.FileTypeRegistry.is_video", return_value=False):
                    result = client._load_images_in_prompt(["text", Path("photo.jpg")])
                    assert result[0] == "text"
                    assert result[1] == mock_img

    def test_load_images_in_prompt_image_load_fails(self):
        from taster.core.models import GeminiClient
        with patch("google.generativeai.configure"):
            client = GeminiClient(api_key="test")
            with patch("taster.core.file_utils.ImageUtils.load_and_fix_orientation", return_value=None):
                with patch("taster.core.file_utils.FileTypeRegistry.is_video", return_value=False):
                    result = client._load_images_in_prompt([Path("bad.jpg")])
                    assert len(result) == 0


# ── GeminiImageClient ──────────────────────────────────────────────


class TestGeminiImageClient:

    def test_init_no_api_key(self):
        from taster.core.models import GeminiImageClient
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GEMINI_API_KEY", None)
            with pytest.raises(ValueError, match="GEMINI_API_KEY not set"):
                GeminiImageClient()

    def test_get_closest_aspect_ratio(self):
        from taster.core.models import GeminiImageClient
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test"}):
            try:
                client = GeminiImageClient(api_key="test")
                # Square
                assert client._get_closest_aspect_ratio(100, 100) == "1:1"
                # Landscape
                ratio = client._get_closest_aspect_ratio(1920, 1080)
                assert ratio == "16:9"
                # Portrait
                ratio = client._get_closest_aspect_ratio(1080, 1920)
                assert ratio == "9:16"
            except ImportError:
                pytest.skip("google-genai not installed")

    def test_save_image(self, tmp_path):
        from taster.core.models import GeminiImageClient
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test"}):
            try:
                client = GeminiImageClient(api_key="test")
                out = tmp_path / "output.jpg"
                assert client.save_image(b"fake image data", out) is True
                assert out.read_bytes() == b"fake image data"
            except ImportError:
                pytest.skip("google-genai not installed")

    def test_save_image_error(self, tmp_path):
        from taster.core.models import GeminiImageClient
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test"}):
            try:
                client = GeminiImageClient(api_key="test")
                # Try to save to non-existent directory
                out = tmp_path / "nonexistent_dir" / "output.jpg"
                assert client.save_image(b"data", out) is False
            except ImportError:
                pytest.skip("google-genai not installed")


# ── initialize_gemini ───────────────────────────────────────────────


class TestInitializeGemini:

    def test_returns_client(self):
        from taster.core.models import initialize_gemini
        with patch("google.generativeai.configure"):
            client = initialize_gemini(api_key="test-key")
            assert client.api_key == "test-key"


# ── Backward compatibility ──────────────────────────────────────────


class TestBackwardCompatibility:

    def test_gemini_response_alias(self):
        from taster.core.models import GeminiResponse
        assert GeminiResponse is AIResponse

    def test_error_classes(self):
        from taster.core.models import GeminiError, SafetyFilterError, TokenLimitError, RateLimitError
        assert issubclass(SafetyFilterError, GeminiError)
        assert issubclass(TokenLimitError, GeminiError)
        assert issubclass(RateLimitError, GeminiError)
