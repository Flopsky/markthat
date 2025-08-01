"""
Provider clients for different multimodal LLM services.
This module handles the initialization and management of clients for various LLM providers.
"""

import os
from typing import Dict, Optional, Any


class ProviderClient:
    """Base class for provider clients."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self._client = None
    
    def get_client(self):
        """Return the initialized client."""
        if not self._client:
            self._initialize_client()
        return self._client
    
    def _initialize_client(self):
        """Initialize the client. To be implemented by subclasses."""
        raise NotImplementedError


class GeminiClient(ProviderClient):
    """Client for Google's Gemini models."""
    
    def _initialize_client(self):
        try:
            import google.generativeai as genai
            
            api_key = self.api_key or os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("Gemini API key is required")
            
            genai.configure(api_key=api_key)
            self._client = genai
        except ImportError:
            raise ImportError("Please install the required package: pip install google-generativeai")


class OpenAIClient(ProviderClient):
    """Client for OpenAI's GPT models."""
    
    def _initialize_client(self):
        try:
            from openai import OpenAI
            
            api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key is required")
            
            self._client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("Please install the required package: pip install openai")


class AnthropicClient(ProviderClient):
    """Client for Anthropic's Claude models."""
    
    def _initialize_client(self):
        try:
            import anthropic
            
            api_key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("Anthropic API key is required")
            
            self._client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("Please install the required package: pip install anthropic")


class MistralClient(ProviderClient):
    """Client for Mistral AI models."""
    
    def _initialize_client(self):
        try:
            from mistralai import Mistral
            
            api_key = self.api_key or os.environ.get("MISTRAL_API_KEY")
            if not api_key:
                raise ValueError("Mistral API key is required")
            
            self._client = Mistral(api_key=api_key)
        except ImportError:
            raise ImportError("Please install the required package: pip install mistralai>=1.7.0")


class OpenRouterClient(ProviderClient):
    """Client for OpenRouter - unified access to multiple LLM providers."""
    
    def _initialize_client(self):
        try:
            from openai import OpenAI
            
            api_key = self.api_key or os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OpenRouter API key is required")
            
            # OpenRouter uses OpenAI-compatible API with custom base URL
            self._client = OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1"
            )
        except ImportError:
            raise ImportError("Please install the required package: pip install openai")


class ProviderClientFactory:
    """Factory class to create appropriate provider clients."""
    
    # Provider mapping (model identifier to client class)
    PROVIDER_MAP = {
        "gemini": GeminiClient,
        "gpt": OpenAIClient,
        "claude": AnthropicClient,
        "mistral": MistralClient,
        "openrouter": OpenRouterClient,
    }
    
    @classmethod
    def create_client(cls, provider: str, api_key: Optional[str] = None) -> ProviderClient:
        """
        Create a provider client based on the provider name.
        
        Args:
            provider: Name of the provider (e.g., 'openai', 'anthropic', 'google', 'mistral')
            api_key: API key for the provider
            
        Returns:
            An initialized provider client
        """
        provider_lower = provider.lower()
        
        # Map provider names to our internal provider keys
        provider_key_map = {
            "openai": "gpt",
            "anthropic": "claude", 
            "google": "gemini",
            "mistral": "mistral",
            "openrouter": "openrouter"
        }
        
        # Get the provider key
        provider_key = provider_key_map.get(provider_lower)
        if not provider_key:
            # Try direct mapping for backward compatibility
            provider_key = provider_lower
        
        # Get the appropriate provider class
        provider_class = cls.PROVIDER_MAP.get(provider_key)
        
        if not provider_class:
            raise ValueError(f"Unsupported provider: {provider}. Supported providers: {list(provider_key_map.keys())}")
        
        return provider_class(api_key=api_key)


def get_client(model_name: str, provider: str, api_key: Optional[str] = None) -> Any:
    """
    Get the appropriate client for the given model and provider.
    
    Args:
        model_name: Name of the model (used for logging and context)
        provider: Name of the provider (e.g., 'openai', 'anthropic', 'google', 'mistral')
        api_key: API key for the provider
        
    Returns:
        Initialized client for the specified provider
    """
    client = ProviderClientFactory.create_client(provider, api_key)
    return client.get_client()
