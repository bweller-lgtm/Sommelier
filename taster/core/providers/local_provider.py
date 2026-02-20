"""Local LLM provider via any OpenAI-compatible endpoint (Ollama, LM Studio, vLLM, llama.cpp)."""
import os
from typing import Optional

from .openai_provider import OpenAIProvider

_DEFAULT_BASE_URL = "http://localhost:11434/v1"
_DEFAULT_MODEL = "llama3.2"
_DEFAULT_TIMEOUT = 300.0


class LocalProvider(OpenAIProvider):
    """Local LLM provider using any OpenAI-compatible endpoint.

    Resolution for ``base_url``:
      1. Explicit *base_url* parameter
      2. ``LOCAL_LLM_URL`` environment variable
      3. Default: ``http://localhost:11434/v1`` (Ollama)
    """

    provider_name = "local"

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        resolved_url = (
            base_url
            or os.environ.get("LOCAL_LLM_URL")
            or _DEFAULT_BASE_URL
        )
        super().__init__(
            base_url=resolved_url,
            model_name=model_name or _DEFAULT_MODEL,
            timeout=timeout if timeout is not None else _DEFAULT_TIMEOUT,
            **kwargs,
        )
