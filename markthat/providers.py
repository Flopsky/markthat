"""Provider client abstraction layer.

This module collapses the responsibilities previously spread across
`providers_clients.py` into a more concise, type-hinted interface.

A single public helper – :func:`get_client` – returns a *ready-to-use* client
for the requested provider.  Provider-specific classes hide third-party SDK
initialisation details and expose a minimal attribute ``raw`` that contains
that SDK instance so callers can invoke provider-specific APIs when needed.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Final, Mapping, MutableMapping, Type

from .exceptions import ProviderInitializationError
from .logging_config import get_logger

logger = get_logger(__name__)


class BaseProvider(ABC):
    """Abstract base-class for provider wrappers."""

    #: Sub-classes should override this with the canonical environment variable
    #: expected to contain the API key.
    ENV_VAR_NAME: Final[str]

    def __init__(self, api_key: str | None = None):
        self._api_key: str | None = api_key or os.environ.get(self.ENV_VAR_NAME)
        if not self._api_key:
            raise ProviderInitializationError(
                f"API key missing.  Provide `api_key` or set env var {self.ENV_VAR_NAME}."
            )
        self._client: Any | None = None

    @property
    def raw(self) -> Any:
        """Return the lazily-initialised raw SDK client."""
        if self._client is None:
            self._client = self._create()
        return self._client

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    @abstractmethod
    def _create(self) -> Any:  # pragma: no cover – implemented by subclasses
        """Return a *new* instance of the underlying SDK client."""


class GeminiProvider(BaseProvider):
    ENV_VAR_NAME = "GEMINI_API_KEY"

    def _create(self) -> Any:  # noqa: D401 – simple return
        try:
            import google.generativeai as genai

            genai.configure(api_key=self._api_key)
            return genai
        except ImportError as exc:  # pragma: no cover – runtime dependency
            raise ProviderInitializationError(
                "Missing dependency for Google Generative AI: `pip install google-generativeai`."
            ) from exc


class OpenAIProvider(BaseProvider):
    ENV_VAR_NAME = "OPENAI_API_KEY"

    def _create(self) -> Any:  # noqa: D401 – simple return
        try:
            from openai import OpenAI

            return OpenAI(api_key=self._api_key)
        except ImportError as exc:  # pragma: no cover
            raise ProviderInitializationError(
                "Missing dependency for OpenAI: `pip install openai`."
            ) from exc


class AnthropicProvider(BaseProvider):
    ENV_VAR_NAME = "ANTHROPIC_API_KEY"

    def _create(self) -> Any:  # noqa: D401 – simple return
        try:
            import anthropic

            return anthropic.Anthropic(api_key=self._api_key)
        except ImportError as exc:  # pragma: no cover
            raise ProviderInitializationError(
                "Missing dependency for Anthropic: `pip install anthropic`."
            ) from exc


class MistralProvider(BaseProvider):
    ENV_VAR_NAME = "MISTRAL_API_KEY"

    def _create(self) -> Any:  # noqa: D401 – simple return
        try:
            from mistralai import Mistral

            return Mistral(api_key=self._api_key)
        except ImportError as exc:  # pragma: no cover
            raise ProviderInitializationError(
                "Missing dependency for Mistral: `pip install mistralai>=1.7.0`."
            ) from exc


class OpenRouterProvider(BaseProvider):
    ENV_VAR_NAME = "OPENROUTER_API_KEY"

    def _create(self) -> Any:  # noqa: D401 – simple return
        try:
            from openai import OpenAI

            return OpenAI(api_key=self._api_key, base_url="https://openrouter.ai/api/v1")
        except ImportError as exc:  # pragma: no cover
            raise ProviderInitializationError(
                "Missing dependency for OpenRouter: `pip install openai`."
            ) from exc


# -------------------------------------------------------------------------
# Public factory helpers
# -------------------------------------------------------------------------

_PROVIDER_MAP: Mapping[str, Type[BaseProvider]] = {
    "gemini": GeminiProvider,
    "gpt": OpenAIProvider,
    "claude": AnthropicProvider,
    "mistral": MistralProvider,
    "openrouter": OpenRouterProvider,
}

# Cache previously initialised providers to avoid duplicate SDK configuration
_INSTANCE_CACHE: MutableMapping[tuple[str, str | None], BaseProvider] = {}


def get_client(provider_key: str, *, api_key: str | None = None) -> Any:
    """Return *raw* SDK client for *provider_key*.

    Parameters
    ----------
    provider_key:
        One of ``gemini``, ``gpt``, ``claude``, ``mistral`` or ``openrouter``.
    api_key:
        Optional override for the provider API key.
    """

    provider_key_lower = provider_key.lower()
    provider_cls = _PROVIDER_MAP.get(provider_key_lower)
    if provider_cls is None:
        raise ProviderInitializationError(f"Unsupported provider: {provider_key}.")

    cache_key = (provider_key_lower, api_key)
    if cache_key not in _INSTANCE_CACHE:
        logger.debug("Creating new provider client for %s", provider_key_lower)
        _INSTANCE_CACHE[cache_key] = provider_cls(api_key=api_key)

    return _INSTANCE_CACHE[cache_key].raw
