"""Gemini AI provider — delegates to the existing GeminiClient."""
from ..models import GeminiClient

# GeminiClient already extends AIClient, so just re-export it.
GeminiProvider = GeminiClient
