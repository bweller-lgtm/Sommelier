"""Model management and Gemini API client with unified error handling."""
import os
import re
import json
import time
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
import google.generativeai as genai
from PIL import Image


class GeminiError(Exception):
    """Base exception for Gemini API errors."""
    pass


class SafetyFilterError(GeminiError):
    """Exception raised when content is blocked by safety filters."""
    pass


class TokenLimitError(GeminiError):
    """Exception raised when token limit is exceeded."""
    pass


class RateLimitError(GeminiError):
    """Exception raised when rate limit is hit."""
    pass


class GeminiResponse:
    """Parsed response from Gemini API."""

    def __init__(
        self,
        text: str,
        raw_response: Any,
        blocked: bool = False,
        finish_reason: Optional[int] = None
    ):
        """
        Initialize response.

        Args:
            text: Response text.
            raw_response: Raw API response object.
            blocked: Whether response was blocked.
            finish_reason: Finish reason code.
        """
        self.text = text
        self.raw_response = raw_response
        self.blocked = blocked
        self.finish_reason = finish_reason

    def parse_json(self, fallback: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Parse JSON from response text.

        Handles markdown code blocks and other formatting.

        Args:
            fallback: Fallback dict if parsing fails.

        Returns:
            Parsed JSON dict or fallback.
        """
        # Remove markdown code blocks
        text_cleaned = re.sub(r'^```json\s*\n', '', self.text, flags=re.MULTILINE)
        text_cleaned = re.sub(r'^```\s*\n', '', text_cleaned, flags=re.MULTILINE)
        text_cleaned = re.sub(r'\n```\s*$', '', text_cleaned, flags=re.MULTILINE)
        text_cleaned = text_cleaned.strip()

        # Try to parse entire text
        try:
            return json.loads(text_cleaned)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in text
        json_match = re.search(r'\{[^{}]*\}', text_cleaned, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        # Try to find nested JSON
        json_match = re.search(r'\{.*\}', text_cleaned, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        # Give up and return fallback
        if fallback is not None:
            return fallback

        raise ValueError(f"Could not parse JSON from response: {self.text[:200]}")


class GeminiClient:
    """
    Unified Gemini API client with proper error handling and retries.

    Handles:
    - Safety filters
    - Rate limits
    - Token limits
    - JSON parsing
    - Retries with exponential backoff
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
        max_retries: int = 3,
        retry_delay: float = 2.0,
        timeout: float = 120.0,
    ):
        """
        Initialize Gemini client.

        Args:
            api_key: Gemini API key. If None, reads from GEMINI_API_KEY env var.
            model_name: Model to use.
            max_retries: Maximum number of retries for API calls.
            retry_delay: Initial retry delay in seconds (doubles each retry).
            timeout: Timeout for API calls in seconds (default: 120s).
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set. Set environment variable or pass to constructor.")

        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout

        # Configure API
        genai.configure(api_key=self.api_key)

    def _create_model(self, **model_kwargs) -> genai.GenerativeModel:
        """Create model instance with optional kwargs."""
        return genai.GenerativeModel(self.model_name, **model_kwargs)

    def _validate_response(self, response: Any) -> GeminiResponse:
        """
        Validate and parse raw API response.

        Args:
            response: Raw response from Gemini API.

        Returns:
            GeminiResponse object.

        Raises:
            SafetyFilterError: If blocked by safety filters.
            GeminiError: If response is invalid.
        """
        # Check if response has candidates
        if not response.candidates:
            raise SafetyFilterError("Response blocked by safety filters")

        candidate = response.candidates[0]

        # Check finish reason
        # 1 = STOP (normal completion)
        # 2 = MAX_TOKENS
        # 3 = SAFETY
        # 4 = RECITATION
        # 5 = OTHER
        if candidate.finish_reason == 3:
            raise SafetyFilterError("Content blocked by safety filters")
        elif candidate.finish_reason == 2:
            # Max tokens - not necessarily an error, but log it
            print(f"⚠️  Warning: Response reached max tokens")
        elif candidate.finish_reason not in [1, 2]:
            print(f"⚠️  Warning: Unusual finish reason: {candidate.finish_reason}")

        # Extract text - handle truncated responses gracefully
        try:
            text = response.text.strip()
        except Exception as e:
            # If MAX_TOKENS and text extraction fails, raise retriable error
            if candidate.finish_reason == 2:
                raise GeminiError(f"Response truncated (MAX_TOKENS) and could not extract text: {e}")
            raise GeminiError(f"Could not extract text from response: {e}")

        return GeminiResponse(
            text=text,
            raw_response=response,
            blocked=False,
            finish_reason=candidate.finish_reason
        )

    def generate(
        self,
        prompt: Union[str, List[Union[str, Image.Image, Path]]],
        generation_config: Optional[Dict[str, Any]] = None,
        handle_safety_errors: bool = True,
        rate_limit_delay: float = 0.5,
    ) -> GeminiResponse:
        """
        Generate content with retries and error handling.

        Args:
            prompt: Text prompt or list of [text, images, videos].
            generation_config: Generation configuration dict.
            handle_safety_errors: If True, return fallback instead of raising.
            rate_limit_delay: Delay between requests to avoid rate limits.

        Returns:
            GeminiResponse object.

        Raises:
            GeminiError: If generation fails after all retries.
        """
        # Convert single string to list
        if isinstance(prompt, str):
            prompt = [prompt]
        else:
            # Load images from paths
            prompt = self._load_images_in_prompt(prompt)

        # Default generation config
        if generation_config is None:
            generation_config = {"max_output_tokens": 4096}

        # Rate limiting
        if rate_limit_delay > 0:
            time.sleep(rate_limit_delay)

        # Retry loop
        last_error = None
        for attempt in range(self.max_retries):
            try:
                model = self._create_model()

                # Note: Gemini API doesn't directly support timeout parameter in generate_content
                # We rely on the underlying HTTP client's timeout (typically set via environment)
                # For production use, consider wrapping in threading.Timer or asyncio.timeout
                response = model.generate_content(
                    prompt,
                    generation_config=generation_config,
                    request_options={"timeout": self.timeout}  # Passes to underlying HTTP client
                )
                return self._validate_response(response)

            except SafetyFilterError as e:
                if handle_safety_errors:
                    # Return fallback response
                    print(f"⚠️  Safety filter triggered: {e}")
                    return GeminiResponse(
                        text='{"classification": "Review", "confidence": 0.3, "reasoning": "Blocked by safety filters"}',
                        raw_response=None,
                        blocked=True,
                        finish_reason=3
                    )
                else:
                    raise

            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                # Check if timeout error
                if "timeout" in error_str or "timed out" in error_str:
                    delay = self.retry_delay * (2 ** attempt)
                    print(f"⚠️  Request timeout ({self.timeout}s), retrying in {delay}s...")
                    time.sleep(delay)
                    continue

                # Check if rate limit error
                if "rate" in error_str or "quota" in error_str:
                    delay = self.retry_delay * (2 ** attempt)
                    print(f"⚠️  Rate limit hit, retrying in {delay}s...")
                    time.sleep(delay)
                    continue

                # Check if temporary error
                if "unavailable" in error_str or "connection" in error_str:
                    delay = self.retry_delay * (2 ** attempt)
                    print(f"⚠️  Temporary error, retrying in {delay}s...")
                    time.sleep(delay)
                    continue

                # Unknown error - fail fast
                raise GeminiError(f"Gemini API error: {e}") from e

        # All retries failed
        raise GeminiError(f"Failed after {self.max_retries} retries: {last_error}")

    def generate_json(
        self,
        prompt: Union[str, List[Union[str, Image.Image, Path]]],
        fallback: Optional[Dict] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate content and parse as JSON.

        Args:
            prompt: Text prompt or list of [text, images, videos].
            fallback: Fallback dict if JSON parsing fails.
            generation_config: Generation configuration dict.
            **kwargs: Additional arguments passed to generate().

        Returns:
            Parsed JSON dict or fallback.
        """
        response = self.generate(
            prompt=prompt,
            generation_config=generation_config,
            **kwargs
        )

        try:
            return response.parse_json(fallback=fallback)
        except ValueError as e:
            if fallback is not None:
                print(f"⚠️  JSON parsing failed, using fallback: {e}")
                return fallback
            raise

    def _load_images_in_prompt(self, prompt: List) -> List:
        """
        Load images from Path objects in prompt.

        Args:
            prompt: List potentially containing Path objects.

        Returns:
            List with Paths replaced by PIL Images.
        """
        result = []
        for item in prompt:
            if isinstance(item, Path):
                # Load image
                try:
                    from .file_utils import ImageUtils
                    img = ImageUtils.load_and_fix_orientation(item)
                    if img is not None:
                        result.append(img)
                    else:
                        print(f"⚠️  Warning: Could not load image {item}")
                except Exception as e:
                    print(f"⚠️  Warning: Error loading image {item}: {e}")
            else:
                result.append(item)
        return result

    def count_tokens(self, prompt: Union[str, List]) -> int:
        """
        Estimate token count for prompt.

        Args:
            prompt: Text prompt or list of [text, images].

        Returns:
            Estimated token count.
        """
        try:
            model = self._create_model()
            if isinstance(prompt, str):
                prompt = [prompt]
            return model.count_tokens(prompt).total_tokens
        except Exception:
            # Fallback to rough estimate
            if isinstance(prompt, str):
                return len(prompt) // 4  # Rough estimate: 4 chars per token
            return 0


def initialize_gemini(api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash") -> GeminiClient:
    """
    Initialize Gemini client with API key.

    Args:
        api_key: API key. If None, reads from environment.
        model_name: Model name to use.

    Returns:
        GeminiClient instance.
    """
    return GeminiClient(api_key=api_key, model_name=model_name)
