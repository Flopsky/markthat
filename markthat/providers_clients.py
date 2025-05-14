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
            from mistralai.client import MistralClient as MistralSDK
            
            api_key = self.api_key or os.environ.get("MISTRAL_API_KEY")
            if not api_key:
                raise ValueError("Mistral API key is required")
            
            self._client = MistralSDK(api_key=api_key)
        except ImportError:
            raise ImportError("Please install the required package: pip install mistralai")


class ProviderClientFactory:
    """Factory class to create appropriate provider clients."""
    
    # Provider mapping (model identifier to client class)
    PROVIDER_MAP = {
        "gemini": GeminiClient,
        "gpt": OpenAIClient,
        "claude": AnthropicClient,
        "mistral": MistralClient,
    }
    
    @classmethod
    def create_client(cls, provider_name: str, api_key: Optional[str] = None) -> ProviderClient:
        """
        Create a provider client based on the model name.
        
        Args:
            provider_name: Name of the model/provider
            api_key: API key for the provider
            
        Returns:
            An initialized provider client
        """
        provider_class = None
        provider_name_lower = provider_name.lower()
        
        # Determine which provider this is based on the model name
        if "gemini" in provider_name_lower:
            provider = "gemini"
        elif "gpt" in provider_name_lower:
            provider = "gpt"
        elif "claude" in provider_name_lower:
            provider = "claude"
        elif "mistral" in provider_name_lower:
            provider = "mistral"
        else:
            # Default to a generic provider
            provider = None
        
        # Get the appropriate provider class
        provider_class = cls.PROVIDER_MAP.get(provider)
        
        if not provider_class:
            raise ValueError(f"Unsupported provider: {provider_name}")
        
        return provider_class(api_key=api_key)


def get_client(model_name: str, api_key: Optional[str] = None) -> Any:
    """
    Get the appropriate client for the given model.
    
    Args:
        model_name: Name of the model
        api_key: API key for the provider
        
    Returns:
        Initialized client for the specified model
    """
    client = ProviderClientFactory.create_client(model_name, api_key)
    return client.get_client()
