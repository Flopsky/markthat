"""
Main client module for MarkThat.
This module provides the MarkThat class which serves as the main entry point for image to markdown conversion.
"""

from typing import Dict, List, Optional, Any, Union
import os
import time
from .providers_clients import get_client
from .prompts import get_prompt_for_model
import logging
import base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetryPolicy:
    """Configuration for retry behavior."""
    
    def __init__(
        self, 
        max_attempts: int = 3, 
        timeout: int = 30, 
        backoff_factor: float = 1.0
    ):
        """
        Initialize retry policy.
        
        Args:
            max_attempts: Maximum number of retry attempts
            timeout: Timeout in seconds for each attempt
            backoff_factor: Factor to increase wait time between retries
        """
        self.max_attempts = max_attempts
        self.timeout = timeout
        self.backoff_factor = backoff_factor


class MarkThat:
    """Main class for converting images to markdown using multimodal LLMs."""
    
    def __init__(
        self,
        primary_model: str,
        fallback_models: Optional[List[str]] = None,
        retry_policy: Optional[RetryPolicy] = None,
        api_key: Optional[str] = None,
        max_retry: int = 3
    ):
        """
        Initialize the MarkThat converter.
        
        Args:
            primary_model: Name of the primary model to use
            fallback_models: List of fallback models to try if primary fails
            retry_policy: Custom retry policy configuration
            api_key: API key for the provider
            max_retry: Maximum number of retries per model
        """
        self.primary_model = primary_model
        self.fallback_models = fallback_models or []
        self.retry_policy = retry_policy or RetryPolicy(max_attempts=max_retry)
        self.api_key = api_key
        self.max_retry = max_retry
        
        # Initialize the primary client
        self._primary_client = None
        self._fallback_clients = {}
        
        logger.info(f"MarkThat initialized with primary model: {primary_model}")
        if fallback_models:
            logger.info(f"Fallback models: {', '.join(fallback_models)}")
        logger.info(f"Max retry attempts: {self.retry_policy.max_attempts}")
    
    def _get_primary_client(self):
        """Get the primary client, initializing if necessary."""
        if not self._primary_client:
            logger.debug(f"Initializing primary client for model: {self.primary_model}")
            self._primary_client = get_client(self.primary_model, self.api_key)
        return self._primary_client
    
    def _get_fallback_client(self, model_name: str):
        """Get a fallback client by name, initializing if necessary."""
        if model_name not in self._fallback_clients:
            logger.debug(f"Initializing fallback client for model: {model_name}")
            self._fallback_clients[model_name] = get_client(model_name, self.api_key)
        return self._fallback_clients[model_name]
    
    def _process_file(self, file_path: str) -> List[Any]:
        """
        Process a file based on its type.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of file contents (for PDFs, each page; for images, just one item)
        """
        logger.info(f"Processing file: {file_path}")
        
        # Simple check based on file extension
        if file_path.lower().endswith('.pdf'):
            try:
                import fitz  # PyMuPDF
                
                doc = fitz.open(file_path)
                page_count = len(doc)
                logger.info(f"PDF loaded successfully with {page_count} pages")
                return [page for page in doc]
            except ImportError:
                logger.error("PDF support requires PyMuPDF. Install with: pip install pymupdf")
                raise ImportError("PDF support requires PyMuPDF. Install with: pip install pymupdf")
            except Exception as e:
                logger.error(f"Failed to process PDF: {str(e)}")
                raise
        else:
            # Assume it's an image
            try:
                with open(file_path, 'rb') as f:
                    content = f.read()
                    logger.info(f"Image loaded successfully, size: {len(content)} bytes")
                    return [content]
            except Exception as e:
                logger.error(f"Failed to read image file: {str(e)}")
                raise
    
    def _convert_with_model(self, model_name: str, content: bytes, format_options: Optional[Dict[str, Any]] = None, additional_instructions: Optional[str] = None):
        """
        Try to convert content using the specified model with retries.
        
        Args:
            model_name: Name of the model to use
            content: File content bytes
            format_options: Options for formatting the output
            additional_instructions: Additional instructions for the prompt
            
        Returns:
            Markdown string or None if all attempts fail
        """
        logger.info(f"Attempting conversion with model: {model_name}")
        if format_options:
            logger.debug(f"Format options: {format_options}")
        if additional_instructions:
            logger.debug(f"Additional instructions provided: {len(additional_instructions)} characters")
        
        client = get_client(model_name, self.api_key)
        
        # Get prompts for this model
        prompts = get_prompt_for_model(
            model_name, 
            format_options=format_options,
            additional_instructions=additional_instructions
        )
        system_prompt = prompts["system_prompt"]
        user_prompt = prompts["user_prompt"]
        
        logger.debug(f"System prompt length: {len(system_prompt)} characters")
        logger.debug(f"User prompt length: {len(user_prompt)} characters")
        
        # Encode the image as base64 for APIs that need it
        base64_image = base64.b64encode(content).decode('utf-8')
        
        for attempt in range(self.retry_policy.max_attempts):
            logger.info(f"Attempt {attempt+1}/{self.retry_policy.max_attempts} with {model_name}")
            
            try:
                # Handle different client APIs
                if "gemini" in model_name.lower():
                    logger.debug("Using Gemini API")
                    model = client.GenerativeModel(model_name)
                    
                    # Combine prompts since Gemini may not support separate system prompts
                    combined_prompt = f"{system_prompt}\n\n{user_prompt}"
                    
                    # Create the generation request
                    response = model.generate_content(
                        contents=[
                            combined_prompt,
                            {"mime_type": "image/jpeg", "data": content}
                        ]
                    )
                    
                    logger.info(f"Gemini generation successful")
                    return response.text
                
                elif "gpt" in model_name.lower():
                    logger.debug("Using OpenAI API")
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": [
                                {"type": "text", "text": user_prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                            ]}
                        ]
                    )
                    logger.info(f"OpenAI generation successful")
                    return response.choices[0].message.content
                
                elif "claude" in model_name.lower():
                    logger.debug("Using Anthropic API")
                    response = client.messages.create(
                        model=model_name,
                        max_tokens=4000,
                        system=system_prompt,
                        messages=[
                            {"role": "user", "content": [
                                {"type": "text", "text": user_prompt},
                                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64_image}}
                            ]}
                        ]
                    )
                    logger.info(f"Claude generation successful")
                    return response.content[0].text
                
                elif "mistral" in model_name.lower():
                    logger.debug("Using Mistral API")
                    response = client.chat(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"{user_prompt}\n\n[Image: data:image/jpeg;base64,{base64_image}]"}
                        ]
                    )
                    logger.info(f"Mistral generation successful")
                    return response.choices[0].message.content
                
                else:
                    error_msg = f"Unsupported model: {model_name}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
            
            except Exception as e:
                logger.error(f"Attempt {attempt+1} with {model_name} failed: {str(e)}")
                logger.error(f"Error type: {type(e).__name__}")
                
                # For detailed debugging
                if hasattr(e, 'response'):
                    try:
                        error_details = e.response.json()
                        logger.error(f"API error details: {error_details}")
                    except:
                        pass
                
                if attempt < self.retry_policy.max_attempts - 1:
                    # Exponential backoff
                    wait_time = 1
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All {self.retry_policy.max_attempts} attempts with {model_name} failed")
                    return None
    
    def convert(
        self, 
        file_path: str, 
        format_options: Optional[Dict[str, Any]] = None,
        max_retry: Optional[int] = None,
        additional_instructions: Optional[str] = None,
    ) -> Union[str, List[str]]:
        """
        Convert an image or PDF to markdown.
        
        Args:
            file_path: Path to the image or PDF file
            format_options: Options for formatting the output
            max_retry: Override the default max retries
            additional_instructions: Additional instructions for the prompt
            
        Returns:
            For a single image: markdown string
            For a PDF: list of markdown strings, one per page
        """
        logger.info(f"Starting conversion of {file_path}")
        
        if max_retry is not None:
            self.retry_policy.max_attempts = max_retry
            logger.info(f"Max retry attempts updated to {max_retry}")
        
        # Process the file
        contents = self._process_file(file_path)
        results = []
        
        for i, content in enumerate(contents):
            logger.info(f"Processing content {i+1}/{len(contents)}")
            
            # Try with primary model
            logger.info(f"Attempting conversion with primary model: {self.primary_model}")
            result = self._convert_with_model(
                self.primary_model, 
                content, 
                format_options=format_options,
                additional_instructions=additional_instructions,
            )
            
            # If primary model failed and we have fallbacks
            if result is None and self.fallback_models:
                logger.info(f"Primary model failed, trying {len(self.fallback_models)} fallback models")
                for fallback_model in self.fallback_models:
                    logger.info(f"Attempting conversion with fallback model: {fallback_model}")
                    result = self._convert_with_model(
                        fallback_model, 
                        content, 
                        format_options=format_options,
                        additional_instructions=additional_instructions,
                    )
                    if result:
                        logger.info(f"Fallback model {fallback_model} succeeded")
                        break
                    else:
                        logger.info(f"Fallback model {fallback_model} failed")
            
            if result:
                logger.info(f"Conversion successful for content {i+1}")
                logger.debug(f"Result length: {len(result)} characters")
            else:
                logger.error(f"Conversion failed for content {i+1} with all models")
            
            results.append(result or "Conversion failed with all models")
        
        # Return single result for images, list for PDFs
        logger.info(f"Conversion complete, returning {len(results)} results")
        return results if len(results) > 1 else results[0]
