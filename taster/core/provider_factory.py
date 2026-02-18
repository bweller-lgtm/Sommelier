"""Factory for creating AI clients with auto-detection of available providers."""
import os
from typing import Dict, Optional

from .ai_client import AIClient
from .config import Config


# Provider preference order for auto-detection (cheapest first)
_PROVIDER_ORDER = ["gemini", "openai", "anthropic"]

_ENV_KEYS = {
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


def detect_available_providers() -> Dict[str, bool]:
    """Check which AI provider API keys are configured."""
    return {name: bool(os.environ.get(key)) for name, key in _ENV_KEYS.items()}


def create_ai_client(
    config: Config,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
) -> AIClient:
    """Create an AI client using the best available provider.

    Resolution order:
      1. Explicit *provider* argument
      2. config.model.provider
      3. Auto-detect from environment (gemini > openai > anthropic)

    Args:
        config: Application configuration.
        provider: Force a specific provider name.
        api_key: Override API key (otherwise read from env).

    Returns:
        An AIClient implementation.

    Raises:
        ValueError: If no suitable provider is found.
    """
    chosen = provider or config.model.provider

    if chosen is None:
        # Auto-detect
        available = detect_available_providers()
        for name in _PROVIDER_ORDER:
            if available.get(name):
                chosen = name
                break

    if chosen is None:
        raise ValueError(
            "No AI provider configured. Set one of these environment variables:\n"
            "  GEMINI_API_KEY   (recommended — cheapest, native video/PDF)\n"
            "  OPENAI_API_KEY   (GPT-4o / GPT-4.1)\n"
            "  ANTHROPIC_API_KEY (Claude)\n"
        )

    return _build_client(chosen, config, api_key)


def _build_client(
    provider: str, config: Config, api_key: Optional[str]
) -> AIClient:
    """Instantiate the correct provider class."""
    common = dict(
        max_retries=config.system.max_retries,
        retry_delay=config.system.retry_delay_seconds,
    )

    if provider == "gemini":
        from .models import GeminiClient

        return GeminiClient(
            api_key=api_key,
            model_name=config.model.name,
            **common,
        )

    if provider == "openai":
        from .providers.openai_provider import OpenAIProvider

        return OpenAIProvider(
            api_key=api_key,
            model_name=config.model.openai_model,
            video_frame_count=config.model.video_frame_count,
            pdf_render_dpi=config.model.pdf_render_dpi,
            **common,
        )

    if provider == "anthropic":
        from .providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(
            api_key=api_key,
            model_name=config.model.anthropic_model,
            video_frame_count=config.model.video_frame_count,
            **common,
        )

    raise ValueError(
        f"Unknown provider '{provider}'. Choose from: gemini, openai, anthropic"
    )
