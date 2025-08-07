"""LangChain-based provider abstraction layer.

This module provides a unified interface for LLM providers using LangChain,
replacing the custom provider implementations with standardized LangChain 
chat models. This enables better consistency, error handling, and future 
extensibility.
"""

from __future__ import annotations

import base64
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI

from .exceptions import ProviderInitializationError
from .logging_config import get_logger

logger = get_logger(__name__)


class LangChainProvider(ABC):
    """Abstract base class for LangChain-based providers."""

    ENV_VAR_NAME: str

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        self._api_key = api_key or os.environ.get(self.ENV_VAR_NAME)
        if not self._api_key:
            raise ProviderInitializationError(
                f"API key missing. Provide `api_key` or set env var {self.ENV_VAR_NAME}."
            )
        self._client: Optional[BaseChatModel] = None
        self._kwargs = kwargs

    @property
    def client(self) -> BaseChatModel:
        """Return the lazily-initialized LangChain chat model."""
        if self._client is None:
            self._client = self._create_client()
        return self._client

    @abstractmethod
    def _create_client(self) -> BaseChatModel:
        """Create and return a LangChain chat model instance."""
        pass

    def invoke(
        self,
        system_prompt: str,
        user_prompt: str,
        image_bytes: Optional[bytes] = None,
        mime_type: str = "image/jpeg",
        **kwargs,
    ) -> str:
        """Unified method to invoke the LLM with text and optional image."""
        messages = self._build_messages(system_prompt, user_prompt, image_bytes, mime_type)

        try:
            response = self.client.invoke(messages, **kwargs)
            return response.content
        except Exception as e:
            logger.error(f"Error invoking {self.__class__.__name__}: {e}")
            raise

    @abstractmethod
    def _build_messages(
        self,
        system_prompt: str,
        user_prompt: str,
        image_bytes: Optional[bytes] = None,
        mime_type: str = "image/jpeg",
    ) -> List[BaseMessage]:
        """Build the message list for the specific provider."""
        pass


class LangChainOpenAIProvider(LangChainProvider):
    """OpenAI provider using LangChain."""

    ENV_VAR_NAME = "OPENAI_API_KEY"

    def _create_client(self) -> BaseChatModel:
        try:
            return ChatOpenAI(api_key=self._api_key, **self._kwargs)
        except ImportError as e:
            raise ProviderInitializationError(
                "Missing dependency: `pip install langchain-openai`"
            ) from e

    def _build_messages(
        self,
        system_prompt: str,
        user_prompt: str,
        image_bytes: Optional[bytes] = None,
        mime_type: str = "image/jpeg",
    ) -> List[BaseMessage]:
        messages = [SystemMessage(content=system_prompt)]

        if image_bytes:
            b64_image = base64.b64encode(image_bytes).decode()
            content = [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_image}"}},
            ]
            messages.append(HumanMessage(content=content))
        else:
            messages.append(HumanMessage(content=user_prompt))

        return messages


class LangChainAnthropicProvider(LangChainProvider):
    """Anthropic provider using LangChain."""

    ENV_VAR_NAME = "ANTHROPIC_API_KEY"

    def _create_client(self) -> BaseChatModel:
        try:
            return ChatAnthropic(api_key=self._api_key, max_tokens=4000, **self._kwargs)
        except ImportError as e:
            raise ProviderInitializationError(
                "Missing dependency: `pip install langchain-anthropic`"
            ) from e

    def _build_messages(
        self,
        system_prompt: str,
        user_prompt: str,
        image_bytes: Optional[bytes] = None,
        mime_type: str = "image/jpeg",
    ) -> List[BaseMessage]:
        messages = [SystemMessage(content=system_prompt)]

        if image_bytes:
            b64_image = base64.b64encode(image_bytes).decode()
            content = [
                {"type": "text", "text": user_prompt},
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": mime_type, "data": b64_image},
                },
            ]
            messages.append(HumanMessage(content=content))
        else:
            messages.append(HumanMessage(content=user_prompt))

        return messages


class LangChainGeminiProvider(LangChainProvider):
    """Google Gemini provider using LangChain."""

    ENV_VAR_NAME = "GEMINI_API_KEY"

    def _create_client(self) -> BaseChatModel:
        try:
            return ChatGoogleGenerativeAI(google_api_key=self._api_key, **self._kwargs)
        except ImportError as e:
            raise ProviderInitializationError(
                "Missing dependency: `pip install langchain-google-genai`"
            ) from e

    def _build_messages(
        self,
        system_prompt: str,
        user_prompt: str,
        image_bytes: Optional[bytes] = None,
        mime_type: str = "image/jpeg",
    ) -> List[BaseMessage]:
        # Gemini combines system and user prompts
        combined_prompt = f"{system_prompt}\n\n{user_prompt}"

        if image_bytes:
            content = [
                {"type": "text", "text": combined_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode()}"
                    },
                },
            ]
            return [HumanMessage(content=content)]
        else:
            return [HumanMessage(content=combined_prompt)]


class LangChainMistralProvider(LangChainProvider):
    """Mistral provider using LangChain."""

    ENV_VAR_NAME = "MISTRAL_API_KEY"

    def _create_client(self) -> BaseChatModel:
        try:
            return ChatMistralAI(api_key=self._api_key, **self._kwargs)
        except ImportError as e:
            raise ProviderInitializationError(
                "Missing dependency: `pip install langchain-mistralai`"
            ) from e

    def _build_messages(
        self,
        system_prompt: str,
        user_prompt: str,
        image_bytes: Optional[bytes] = None,
        mime_type: str = "image/jpeg",
    ) -> List[BaseMessage]:
        # Mistral doesn't support images in the current implementation
        messages = [SystemMessage(content=system_prompt)]
        messages.append(HumanMessage(content=user_prompt))
        return messages


class LangChainOpenRouterProvider(LangChainProvider):
    """OpenRouter provider using LangChain (OpenAI-compatible)."""

    ENV_VAR_NAME = "OPENROUTER_API_KEY"

    def _create_client(self) -> BaseChatModel:
        try:
            return ChatOpenAI(
                api_key=self._api_key, base_url="https://openrouter.ai/api/v1", **self._kwargs
            )
        except ImportError as e:
            raise ProviderInitializationError(
                "Missing dependency: `pip install langchain-openai`"
            ) from e

    def _build_messages(
        self,
        system_prompt: str,
        user_prompt: str,
        image_bytes: Optional[bytes] = None,
        mime_type: str = "image/jpeg",
    ) -> List[BaseMessage]:
        messages = [SystemMessage(content=system_prompt)]

        if image_bytes:
            b64_image = base64.b64encode(image_bytes).decode()
            content = [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_image}"}},
            ]
            messages.append(HumanMessage(content=content))
        else:
            messages.append(HumanMessage(content=user_prompt))

        return messages


# Provider registry
LANGCHAIN_PROVIDER_MAP: Dict[str, type[LangChainProvider]] = {
    "gemini": LangChainGeminiProvider,
    "openai": LangChainOpenAIProvider,
    "claude": LangChainAnthropicProvider,
    "mistral": LangChainMistralProvider,
    "openrouter": LangChainOpenRouterProvider,
}

# Cache for provider instances
_LANGCHAIN_CACHE: Dict[tuple[str, Optional[str]], LangChainProvider] = {}


def get_langchain_provider(
    provider_key: str, model_name: Optional[str] = None, api_key: Optional[str] = None, **kwargs
) -> LangChainProvider:
    """Get a LangChain provider instance.

    Args:
        provider_key: One of 'gemini', 'openai', 'claude', 'mistral', 'openrouter'
        model_name: Optional model name to pass to the provider
        api_key: Optional API key override
        **kwargs: Additional arguments to pass to the provider

    Returns:
        LangChainProvider instance
    """
    provider_key_lower = provider_key.lower()
    provider_cls = LANGCHAIN_PROVIDER_MAP.get(provider_key_lower)

    if provider_cls is None:
        raise ProviderInitializationError(f"Unsupported provider: {provider_key}")

    cache_key = (provider_key_lower, api_key)

    if cache_key not in _LANGCHAIN_CACHE:
        logger.debug("Creating new LangChain provider for %s", provider_key_lower)

        # Add model name to kwargs if provided
        if model_name:
            if provider_key_lower == "gemini":
                kwargs["model"] = model_name
            elif provider_key_lower in ["openai", "openrouter"]:
                kwargs["model"] = model_name
            elif provider_key_lower == "claude":
                kwargs["model"] = model_name
            elif provider_key_lower == "mistral":
                kwargs["model"] = model_name

        _LANGCHAIN_CACHE[cache_key] = provider_cls(api_key=api_key, **kwargs)

    return _LANGCHAIN_CACHE[cache_key]


def unified_langchain_call(
    model: str,
    system_prompt: str,
    user_prompt: str,
    image_bytes: Optional[bytes] = None,
    mime_type: str = "image/jpeg",
    api_key: Optional[str] = None,
    **kwargs,
) -> str:
    """Unified API call using LangChain providers.

    Args:
        model: Model name to use
        system_prompt: System prompt
        user_prompt: User prompt
        image_bytes: Optional image data
        mime_type: MIME type for image
        api_key: Optional API key override
        **kwargs: Additional arguments

    Returns:
        Generated text response
    """
    from .client import _infer_provider_from_model

    provider_key = _infer_provider_from_model(model)
    provider = get_langchain_provider(
        provider_key=provider_key, model_name=model, api_key=api_key, **kwargs
    )

    return provider.invoke(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        image_bytes=image_bytes,
        mime_type=mime_type,
        **kwargs,
    )
