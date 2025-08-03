"""
Main client module for MarkThat.
This module provides the MarkThat class which serves as the main entry point for image to markdown conversion.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
import os
import time
import re
from .providers_clients import get_client
from .prompts import get_prompt_for_model
import logging
import base64
import io
import asyncio
import concurrent.futures

import fitz  # PyMuPDF
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import uuid
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Regular expressions for validation
START_MARKER = r"\[START COPY TEXT\]"
END_MARKER = r"\[END COPY TEXT\]"
MARKERS_PATTERN = re.compile(f"{START_MARKER}(.*?){END_MARKER}", re.DOTALL)
MARKDOWN_FENCE_PATTERN = re.compile(r"```markdown\s*\n(.*?)```", re.DOTALL)


def is_valid_markdown(markdown: str) -> bool:
    """
    Check if the provided string is valid markdown.
    This is a basic validation that checks for common markdown syntax errors.
    
    Args:
        markdown: The markdown string to validate
        
    Returns:
        True if valid, False otherwise
    """
    # Check for basic structural validity
    try:
        # Check for unclosed code blocks
        code_blocks = re.findall(r"```[^\n]*\n", markdown)
        closing_blocks = re.findall(r"\n```", markdown)
        if len(code_blocks) != len(closing_blocks):
            logger.warning("Markdown validation failed: Unclosed code blocks")
            return False
            
        # Check for unclosed inline code
        if markdown.count('`') % 2 != 0:
            logger.warning("Markdown validation failed: Unclosed inline code")
            return False
            
        # Check for unclosed links
        if markdown.count('[') != markdown.count(']'):
            logger.warning("Markdown validation failed: Unclosed square brackets")
            return False
            
        # Check for unclosed parentheses in links
        if markdown.count('(') != markdown.count(')'):
            logger.warning("Markdown validation failed: Unclosed parentheses")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Error validating markdown: {str(e)}")
        return False


def has_copy_markers(markdown: str) -> bool:
    """
    Check if the markdown contains both START COPY TEXT and END COPY TEXT markers.
    
    Args:
        markdown: The markdown string to check
        
    Returns:
        True if both markers are present, False otherwise
    """
    has_start = re.search(START_MARKER, markdown) is not None
    has_end = re.search(END_MARKER, markdown) is not None
    
    if not has_start:
        logger.warning("Markdown missing [START COPY TEXT] marker")
    if not has_end:
        logger.warning("Markdown missing [END COPY TEXT] marker")
        
    return has_start and has_end


def extract_content_between_markers(markdown: str) -> str:
    """
    Extract the content between START COPY TEXT and END COPY TEXT markers.
    
    Args:
        markdown: The markdown string with markers
        
    Returns:
        The content between the markers, or the original string if markers not found
    """
    match = MARKERS_PATTERN.search(markdown)
    if match:
        return match.group(1).strip()
    else:
        logger.warning("Could not extract content between markers")
        return markdown


def remove_markdown_and_markers(markdown: str) -> str:
    """
    Remove both markdown code fence (```) and START/END COPY TEXT markers.
    First extracts content between START/END COPY TEXT markers, then removes
    the first and last occurrences of ```.
    
    Args:
        markdown: The original markdown string with potential fences and markers
        
    Returns:
        Clean markdown content without fences or markers
    """
    logger.debug(f"Original markdown before cleaning: {markdown[:100]}...")
    
    # Step 1: First extract content between START/END COPY TEXT markers
    marker_match = MARKERS_PATTERN.search(markdown)
    if marker_match:
        # Found markers, extract the content
        markdown = marker_match.group(1).strip()
        logger.debug("Removed START/END COPY TEXT markers")
    else:
        # If no pattern match, try direct string replacement for markers
        if '[START COPY TEXT]' in markdown and '[END COPY TEXT]' in markdown:
            start_idx = markdown.find('[START COPY TEXT]') + len('[START COPY TEXT]')
            end_idx = markdown.find('[END COPY TEXT]')
            if end_idx > start_idx:
                markdown = markdown[start_idx:end_idx].strip()
                logger.debug("Extracted content between markers using string indices")
            else:
                # Direct replacement as last resort
                markdown = markdown.replace('[START COPY TEXT]', '').replace('[END COPY TEXT]', '')
                markdown = markdown.strip()
                logger.debug("Removed markers using direct replacement")
        else:
            logger.debug("No START/END COPY TEXT markers found for removal")
    
    # Step 2: Remove first and last occurrences of ```
    if '```' in markdown:
        # Count occurrences
        count = markdown.count('```')
        logger.debug(f"Found {count} code fence markers")
        
        if count >= 2:
            # Find first occurrence
            first_idx = markdown.find('```')
            # Find index after first ``` (including the ```)
            after_first = first_idx + 3
            
            # Find last occurrence
            last_idx = markdown.rfind('```')
            
            # Only proceed if they're different occurrences
            if last_idx > after_first:
                # Get the content between the first occurrence (after the ```) and before the last occurrence
                # +3 to skip past the first ```
                # We don't include the last ```
                markdown = markdown[after_first:last_idx].strip()
                logger.debug("Removed first and last ``` markers")
            else:
                logger.debug("Cannot remove first/last ```, they are the same occurrence or invalid positions")
        else:
            # Remove the single ``` if there's only one
            markdown = markdown.replace('```', '').strip()
            logger.debug("Removed single ``` marker")
    else:
        logger.debug("No ``` markers found for removal")
    
    # Final cleanup - remove any markdown language specifier if present
    #markdown = re.sub(r'^markdown\s*\n', '', markdown)
    
    return markdown.strip()


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


class FailureTracker:
    """Tracks failures and generates feedback for prompts."""
    
    def __init__(self):
        self.failures = []
        
    def add_failure(self, attempt_number: int, error_type: str, error_message: str, model_output: Optional[str] = None):
        """
        Add a failed attempt with error information.
        
        Args:
            attempt_number: The attempt number that failed
            error_type: Type of error (validation, API, etc.)
            error_message: Error message describing the failure
            model_output: The model's output that caused the failure (if any)
        """
        failure_info = {
            "attempt": attempt_number,
            "error_type": error_type,
            "error_message": error_message,
            "output_snippet": model_output[:200] + "..." if model_output and len(model_output) > 200 else model_output
        }
        self.failures.append(failure_info)
        logger.info(f"Added failure #{attempt_number}: {error_type} - {error_message}")
    
    def get_feedback_for_prompt(self) -> str:
        """
        Generate feedback text to append to the prompt for the next attempt.
        
        Returns:
            Formatted feedback text
        """
        if not self.failures:
            return ""
            
        feedback_lines = ["Previous attempts failed for the following reasons:"]
        
        for failure in self.failures:
            attempt = failure["attempt"]
            error_type = failure["error_type"]
            message = failure["error_message"]
            snippet = failure["output_snippet"]
            
            feedback_lines.append(f"- Attempt #{attempt}: {error_type} error - {message}")
            if snippet:
                feedback_lines.append(f"  Output sample: \"{snippet}\"")
        
        feedback_lines.append("\nPlease avoid these issues in your response.")
        
        return "\n".join(feedback_lines)
    
    def clear(self):
        """Clear all tracked failures."""
        self.failures = []


class MarkThat:
    """Main class for converting images to markdown using multimodal LLMs."""
    
    def __init__(
        self,
        model: str,
        provider: Optional[str] = None,
        fallback_models: Optional[List[str]] = None,
        retry_policy: Optional[RetryPolicy] = None,
        api_key: Optional[str] = None,
        max_retry: int = 3
    ):
        """
        Initialize the MarkThat converter.
        
        Args:
            model: Name of the primary model to use
            provider: Provider name (e.g., 'openai', 'anthropic', 'google', 'mistral'). 
                     If not provided, will be inferred from model name for backward compatibility.
            fallback_models: List of fallback models to try if primary fails
            retry_policy: Custom retry policy configuration
            api_key: API key for the provider
            max_retry: Maximum number of retries per model
        """
        self.model = model
        self.provider = provider or self._infer_provider_from_model(model)
        self.fallback_models = fallback_models or []
        self.retry_policy = retry_policy or RetryPolicy(max_attempts=max_retry)
        self.api_key = api_key
        self.max_retry = max_retry
        
        # Initialize the primary client
        self._primary_client = None
        self._fallback_clients = {}
        
        logger.info(f"MarkThat initialized with primary model: {model}")
        logger.info(f"Provider: {self.provider}")
        if fallback_models:
            logger.info(f"Fallback models: {', '.join(fallback_models)}")
        logger.info(f"Max retry attempts: {self.retry_policy.max_attempts}")
    
    def _infer_provider_from_model(self, model_name: str) -> str:
        """
        Infer the provider from the model name for backward compatibility.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Inferred provider name
        """
        model_name_lower = model_name.lower()
        
        # Check for OpenRouter format first (provider/model)
        if "/" in model_name_lower:
            return "openrouter"
        elif "gemini" in model_name_lower:
            return "google"
        elif "gpt" in model_name_lower:
            return "openai"
        elif "claude" in model_name_lower:
            return "anthropic"
        elif "mistral" in model_name_lower:
            return "mistral"
        else:
            raise ValueError(f"Cannot infer provider from model name: {model_name}. Please specify provider explicitly.")
    
    def _get_primary_client(self):
        """Get the primary client, initializing if necessary."""
        if not self._primary_client:
            logger.debug(f"Initializing primary client for model: {self.model}")
            self._primary_client = get_client(self.model, self.provider, self.api_key)
        return self._primary_client
    
    def _get_fallback_client(self, model_name: str):
        """Get a fallback client by name, initializing if necessary."""
        if model_name not in self._fallback_clients:
            logger.debug(f"Initializing fallback client for model: {model_name}")
            # For fallback models, infer provider from model name
            fallback_provider = self._infer_provider_from_model(model_name)
            self._fallback_clients[model_name] = get_client(model_name, fallback_provider, self.api_key)
        return self._fallback_clients[model_name]
    
    def _convert_page_to_image(self, page) -> bytes:
        """
        Convert a PyMuPDF Page object to a JPEG image in bytes.
        
        Args:
            page: PyMuPDF Page object
        
        Returns:
            JPEG image as bytes
        """
        try:
            # Render the page to a pixmap
            # zoom = 2.0  # Higher resolution
            # Create transformation matrix for zoom
            # For compatibility with different PyMuPDF versions
            try:
                # Newer API attempt
                # matrix = page.rotation_matrix @ page.scale_matrix(zoom, zoom)
                mat = page.get_pixmap()
            except AttributeError:
                # Fallback for older PyMuPDF versions
                logger.info("Using fallback method for older PyMuPDF version")
            except Exception as e:
                # Last resort fallback
                logger.warning(f"Using basic pixmap rendering due to: {str(e)}")
                mat = page.get_pixmap(dpi=150)  # Use DPI instead of matrix
            
            # Convert pixmap to JPEG bytes using BytesIO to handle different PyMuPDF versions
            buffer = io.BytesIO()
            try:
                # Try the direct tobytes method first
                img_bytes = mat.tobytes("jpeg")
                logger.debug("Used tobytes method for image conversion")
            except (AttributeError, TypeError):
                # Fallback to pil_save method if available
                logger.info("Using PIL fallback for image conversion")
                try:
                    mat.pil_save(buffer, format="JPEG")
                    img_bytes = buffer.getvalue()
                except AttributeError:
                    # Last resort: convert to PIL image manually and save
                    from PIL import Image
                    img = Image.frombytes("RGB", [mat.width, mat.height], mat.samples)
                    img.save(buffer, format="JPEG")
                    img_bytes = buffer.getvalue()
            
            logger.info(f"Converted PDF page to JPEG image, size: {len(img_bytes)} bytes")
            return img_bytes
        except Exception as e:
            logger.error(f"Error converting PDF page to image: {str(e)}")
            raise
    
    def _process_file(self, file_path: str) -> List[bytes]:
        """
        Process a file based on its type.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of image content as bytes (for PDFs, one item per page; for images, just one item)
        """
        logger.info(f"Processing file: {file_path}")
        
        # Simple check based on file extension
        if file_path.lower().endswith('.pdf'):
            try:
                doc = fitz.open(file_path)
                page_count = len(doc)
                logger.info(f"PDF loaded successfully with {page_count} pages")
                
                # Convert each page to an image
                image_bytes_list = []
                for i, page in enumerate(doc):
                    logger.info(f"Converting PDF page {i+1}/{page_count} to image")
                    page_image = self._convert_page_to_image(page)
                    image_bytes_list.append(page_image)
                
                return image_bytes_list
                
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
    
    def _convert_with_model(self, model_name: str, content: bytes, format_options: Optional[Dict[str, Any]] = None, additional_instructions: Optional[str] = None, description_mode: bool = False, extract_figure = False, coordinate_model: str = "gemini-2.0-flash-001", parsing_model: str = "gemini-2.5-flash-lite"):
        """
        Try to convert content using the specified model with retries.
        
        Args:
            model_name: Name of the model to use
            content: File content bytes
            format_options: Options for formatting the output
            additional_instructions: Additional instructions for the prompt
            description_mode: If True, generate a description instead of markdown
            extract_figure: If True, extract figures and save them as separate images
            coordinate_model: Model to use for figure coordinate extraction
            parsing_model: Model to use for coordinate parsing
            
        Returns:
            Markdown string or None if all attempts fail
        """
        logger.info(f"Attempting conversion with model: {model_name} (Description mode: {description_mode})")
        if format_options:
            logger.debug(f"Format options: {format_options}")
        if additional_instructions:
            logger.debug(f"Additional instructions provided: {len(additional_instructions)} characters")
        
        # Get the provider for this model
        if model_name == self.model:
            provider = self.provider
        else:
            # For fallback models, infer provider from model name
            provider = self._infer_provider_from_model(model_name)
        
        client = get_client(model_name, provider, self.api_key)
        
        # Create a failure tracker for this conversion
        failure_tracker = FailureTracker()
        
        for attempt in range(self.retry_policy.max_attempts):
            logger.info(f"Attempt {attempt+1}/{self.retry_policy.max_attempts} with {model_name}")
            
            # Get prompts for this model, including feedback from previous failures
            failure_feedback = failure_tracker.get_feedback_for_prompt()
            enhanced_instructions = additional_instructions or ""
            if failure_feedback and attempt > 0:  # Only add failure feedback after first attempt
                if enhanced_instructions:
                    enhanced_instructions = f"{enhanced_instructions}\n\n{failure_feedback}"
                else:
                    enhanced_instructions = failure_feedback
                logger.info(f"Added failure feedback to prompt for attempt {attempt+1}")
            
            prompts = get_prompt_for_model(
                model_name, 
                format_options=format_options,
                additional_instructions=enhanced_instructions,
                description_mode=description_mode
            )
            system_prompt = prompts["system_prompt"]
            user_prompt = prompts["user_prompt"]
            
            logger.debug(f"System prompt length: {len(system_prompt)} characters")
            logger.debug(f"User prompt length: {len(user_prompt)} characters")
            
            # Encode the image as base64 for APIs that need it
            base64_image = base64.b64encode(content).decode('utf-8')
            
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
                    
                    result = response.text
                    
                    # Validate the result
                    is_valid, validation_msg = self.validate_markdown(result, description_mode=description_mode)
                    if not is_valid:
                        logger.warning(f"Generated output validation failed: {validation_msg}")
                        # If it's missing markers but otherwise valid, we'll add them
                        if validation_msg == "Missing required START/END COPY TEXT markers" and (description_mode or is_valid_markdown(result)):
                            logger.info("Adding missing markers to otherwise valid output")
                            result = f"[START COPY TEXT]\n{result}\n[END COPY TEXT]"
                            is_valid = True
                    
                    if is_valid:
                        logger.info(f"Gemini generation successful")
                        # Apply figure extraction if enabled
                        if extract_figure:
                            result = self._extract_and_save_figures(result, content, coordinate_model, parsing_model)
                        return result
                    else:
                        error_msg = f"Generated output validation failed: {validation_msg}"
                        logger.error(f"Invalid output generated, will retry")
                        # Track this failure
                        failure_tracker.add_failure(
                            attempt_number=attempt+1,
                            error_type="Validation",
                            error_message=validation_msg,
                            model_output=result
                        )
                        raise ValueError(error_msg)
                
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
                    
                    result = response.choices[0].message.content
                    
                    # Validate the result
                    is_valid, validation_msg = self.validate_markdown(result, description_mode=description_mode)
                    if not is_valid:
                        logger.warning(f"Generated output validation failed: {validation_msg}")
                        # If it's missing markers but otherwise valid, we'll add them
                        if validation_msg == "Missing required START/END COPY TEXT markers" and (description_mode or is_valid_markdown(result)):
                            logger.info("Adding missing markers to otherwise valid output")
                            result = f"[START COPY TEXT]\n{result}\n[END COPY TEXT]"
                            is_valid = True
                    
                    if is_valid:
                        logger.info(f"OpenAI generation successful")
                        # Apply figure extraction if enabled
                        if extract_figure:
                            result = self._extract_and_save_figures(result, content, coordinate_model, parsing_model)
                        return result
                    else:
                        error_msg = f"Generated output validation failed: {validation_msg}"
                        logger.error(f"Invalid output generated, will retry")
                        # Track this failure
                        failure_tracker.add_failure(
                            attempt_number=attempt+1,
                            error_type="Validation",
                            error_message=validation_msg,
                            model_output=result
                        )
                        raise ValueError(error_msg)
                
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
                    
                    result = response.content[0].text
                    
                    # Validate the result
                    is_valid, validation_msg = self.validate_markdown(result, description_mode=description_mode)
                    if not is_valid:
                        logger.warning(f"Generated output validation failed: {validation_msg}")
                        # If it's missing markers but otherwise valid, we'll add them
                        if validation_msg == "Missing required START/END COPY TEXT markers" and (description_mode or is_valid_markdown(result)):
                            logger.info("Adding missing markers to otherwise valid output")
                            result = f"[START COPY TEXT]\n{result}\n[END COPY TEXT]"
                            is_valid = True
                    
                    if is_valid:
                        logger.info(f"Claude generation successful")
                        # Apply figure extraction if enabled
                        if extract_figure:
                            result = self._extract_and_save_figures(result, content, coordinate_model, parsing_model)
                        return result
                    else:
                        error_msg = f"Generated output validation failed: {validation_msg}"
                        logger.error(f"Invalid output generated, will retry")
                        # Track this failure
                        failure_tracker.add_failure(
                            attempt_number=attempt+1,
                            error_type="Validation",
                            error_message=validation_msg,
                            model_output=result
                        )
                        raise ValueError(error_msg)
                
                elif "mistral" in model_name.lower():
                    logger.debug("Using Mistral API (new)")
                    # Removed check for ChatMessage is None
                    # if ChatMessage is None: # Check if import failed, though setup.py should handle install
                    #     raise ImportError("Mistral\'s ChatMessage could not be imported. Ensure mistralai package is installed correctly.")

                    # Prepare messages for Mistral API v1.x
                    # System prompt can be a simple string
                    # User prompt combines text and image data
                    user_content_list = [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                    
                    messages_payload = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content_list}
                    ]
                    
                    # Ensure the model is vision-capable, e.g., "mistral-large-latest"
                    # The user is responsible for providing a correct model_name.
                    # Forcing a specific model here might be too restrictive.
                    # model_to_call = "mistral-large-latest" if "large" not in model_name else model_name

                    response = client.chat.complete( # Changed from client.chat
                        model=model_name, # Use the provided model_name
                        messages=messages_payload
                    )
                    
                    result = response.choices[0].message.content
                    
                    # Validate the result
                    is_valid, validation_msg = self.validate_markdown(result, description_mode=description_mode)
                    if not is_valid:
                        logger.warning(f"Generated output validation failed: {validation_msg}")
                        # If it's missing markers but otherwise valid, we'll add them
                        if validation_msg == "Missing required START/END COPY TEXT markers" and (description_mode or is_valid_markdown(result)):
                            logger.info("Adding missing markers to otherwise valid output")
                            result = f"[START COPY TEXT]\n{result}\n[END COPY TEXT]"
                            is_valid = True
                    
                    if is_valid:
                        logger.info(f"Mistral generation successful")
                        # Apply figure extraction if enabled
                        if extract_figure:
                            result = self._extract_and_save_figures(result, content, coordinate_model, parsing_model)
                        return result
                    else:
                        error_msg = f"Generated output validation failed: {validation_msg}"
                        logger.error(f"Invalid output generated, will retry")
                        # Track this failure
                        failure_tracker.add_failure(
                            attempt_number=attempt+1,
                            error_type="Validation",
                            error_message=validation_msg,
                            model_output=result
                        )
                        raise ValueError(error_msg)
                
                elif provider == "openrouter" or "/" in model_name:
                    logger.debug("Using OpenRouter API")
                    
                    # Check if this is a PDF file based on MIME type detection
                    # For now, assume image format - PDF support would need file extension detection
                    
                    # Prepare messages for OpenRouter (OpenAI-compatible API)
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": [
                            {"type": "text", "text": user_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]}
                    ]
                    
                    # For OpenRouter, we can add optional plugins for enhanced features
                    # This is particularly useful for PDF processing
                    payload = {
                        "model": model_name,
                        "messages": messages
                    }
                    
                    # Optional: Add plugins for file processing capabilities
                    # payload["plugins"] = [{"id": "file-parser", "pdf": {"engine": "pdf-text"}}]
                    
                    response = client.chat.completions.create(**payload)
                    
                    result = response.choices[0].message.content
                    
                    # Validate the result
                    is_valid, validation_msg = self.validate_markdown(result, description_mode=description_mode)
                    if not is_valid:
                        logger.warning(f"Generated output validation failed: {validation_msg}")
                        # If it's missing markers but otherwise valid, we'll add them
                        if validation_msg == "Missing required START/END COPY TEXT markers" and (description_mode or is_valid_markdown(result)):
                            logger.info("Adding missing markers to otherwise valid output")
                            result = f"[START COPY TEXT]\n{result}\n[END COPY TEXT]"
                            is_valid = True
                    
                    if is_valid:
                        logger.info(f"OpenRouter generation successful")
                        # Apply figure extraction if enabled
                        if extract_figure:
                            result = self._extract_and_save_figures(result, content, coordinate_model, parsing_model)
                        return result
                    else:
                        error_msg = f"Generated output validation failed: {validation_msg}"
                        logger.error(f"Invalid output generated, will retry")
                        # Track this failure
                        failure_tracker.add_failure(
                            attempt_number=attempt+1,
                            error_type="Validation",
                            error_message=validation_msg,
                            model_output=result
                        )
                        raise ValueError(error_msg)
                
                else:
                    error_msg = f"Unsupported model: {model_name}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
            
            except Exception as e:
                logger.error(f"Attempt {attempt+1} with {model_name} failed: {str(e)}")
                logger.error(f"Error type: {type(e).__name__}")
                
                # Track the failure with appropriate information
                error_type = type(e).__name__
                error_message = str(e)
                model_output = None
                
                # Try to extract error details if available
                if hasattr(e, 'response'):
                    try:
                        error_details = e.response.json()
                        error_message = f"{error_message} - API Error: {error_details}"
                        logger.error(f"API error details: {error_details}")
                    except:
                        pass
                
                # Add failure to tracker
                failure_tracker.add_failure(
                    attempt_number=attempt+1,
                    error_type=error_type,
                    error_message=error_message,
                    model_output=model_output
                )
                
                if attempt < self.retry_policy.max_attempts - 1:
                    # Exponential backoff
                    wait_time = 1
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All {self.retry_policy.max_attempts} attempts with {model_name} failed")
                    return None
    
    def _add_coordinate_grid_to_image(self, image_content: bytes) -> bytes:
        """
        Add a coordinate grid to an image and return the modified image as bytes.
        
        Args:
            image_content: Original image content as bytes
            
        Returns:
            Modified image with coordinate grid as bytes
        """
        # Load the image from bytes
        img = Image.open(io.BytesIO(image_content))
        img_array = np.array(img)
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Display the image without deformation in its original orientation
        ax.imshow(img_array)
        
        # Set up coordinate system overlay
        height, width = img_array.shape[:2]
        
        # Create a coordinate system overlay that preserves image proportions
        # Use the image's actual dimensions but scale the coordinate display
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)  # Invert y-axis to have origin at bottom-left
        
        # Create custom tick locations and labels for 0-30 scale
        # Calculate tick positions based on image dimensions
        x_tick_positions = np.linspace(0, width, 31)  # 31 points for 0-30
        y_tick_positions = np.linspace(0, height, 31)
        x_tick_labels = np.arange(0, 31)
        y_tick_labels = np.arange(30, -1, -1)  # Reverse labels to have 0 at bottom
        
        # Set ticks with custom positions and labels
        ax.set_xticks(x_tick_positions)
        ax.set_xticklabels(x_tick_labels)
        ax.set_yticks(y_tick_positions)
        ax.set_yticklabels(y_tick_labels)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Add axis labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # Add title
        ax.set_title('Image with Coordinate System (Origin at 0,0)')
        
        # Save the figure to bytes
        plt.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)
        
        return buffer.getvalue()
    
    def _get_figure_coordinates_from_model(self, image_with_grid: bytes, figure_description: str, model_name: str) -> str:
        """
        Call specified model to get coordinates of a figure in the image.
        
        Args:
            image_with_grid: Image with coordinate grid as bytes
            figure_description: Description of the figure to find
            model_name: Name of the model to use for coordinate extraction
            
        Returns:
            String response from model with coordinate description
        """
        try:
            from .providers_clients import get_client
            from .prompts.base_prompts import load_prompt_template

            logger.info(f"Getting coordinates for figure: {figure_description} using model: {model_name}")
            
            # Get provider for this model
            provider = self._infer_provider_from_model(model_name)
            client = get_client(model_name, provider, self.api_key)
            
            # Load prompts
            system_template = load_prompt_template("system_prompt_extract_figure.j2")
            user_template = load_prompt_template("user_prompt_extract_figure.j2")
            system_prompt = system_template.render()
            user_prompt = user_template.render(prompt=figure_description)
            
            # Encode image as base64 for APIs that need it
            base64_image = base64.b64encode(image_with_grid).decode('utf-8')
            
            # Handle different client APIs
            if "gemini" in model_name.lower():
                logger.debug("Using Gemini API for coordinate extraction")
                model = client.GenerativeModel(model_name)
                combined_prompt = f"{system_prompt}\n\n{user_prompt}"
                response = model.generate_content(
                    contents=[
                        combined_prompt,
                        {"mime_type": "image/png", "data": image_with_grid}
                    ]
                )
                result = response.text
                
            elif "gpt" in model_name.lower():
                logger.debug("Using OpenAI API for coordinate extraction")
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": [
                            {"type": "text", "text": user_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                        ]}
                    ]
                )
                result = response.choices[0].message.content
                
            elif "claude" in model_name.lower():
                logger.debug("Using Anthropic API for coordinate extraction")
                response = client.messages.create(
                    model=model_name,
                    max_tokens=4000,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": [
                            {"type": "text", "text": user_prompt},
                            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64_image}}
                        ]}
                    ]
                )
                result = response.content[0].text
                
            else:
                # OpenRouter or other providers
                logger.debug("Using OpenRouter API for coordinate extraction")
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": [
                            {"type": "text", "text": user_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                        ]}
                    ]
                )
                result = response.choices[0].message.content
            
            logger.info(f"Response from {model_name} for figure {figure_description}: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to get coordinates from {model_name}: {str(e)}")
            raise e
    
    def _parse_coordinates_from_model(self, coordinate_description: str, model_name: str) -> dict:
        """
        Call specified model to parse coordinates from description into JSON format.
        
        Args:
            coordinate_description: String description of coordinates from first model call
            model_name: Name of the model to use for coordinate parsing
            
        Returns:
            Dictionary with A, B, C, D coordinates
        """
        try:
            from .providers_clients import get_client
            from .prompts.base_prompts import load_prompt_template
            
            logger.info(f"Parsing coordinates using model: {model_name}")
            
            # Get provider for this model
            provider = self._infer_provider_from_model(model_name)
            client = get_client(model_name, provider, self.api_key)
            
            # Load prompts
            system_template = load_prompt_template("system_prompt_get_coordinates.j2")
            user_template = load_prompt_template("user_prompt_get_coordinates.j2")
            system_prompt = system_template.render()
            user_prompt = user_template.render(model_response=coordinate_description)
            
            # Handle different client APIs
            if "gemini" in model_name.lower():
                logger.debug("Using Gemini API for coordinate parsing")
                model = client.GenerativeModel(model_name)
                combined_prompt = f"{system_prompt}\n\n{user_prompt}"
                response = model.generate_content(
                    contents=[combined_prompt]
                )
                result = response.text
                
            elif "gpt" in model_name.lower():
                logger.debug("Using OpenAI API for coordinate parsing")
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                result = response.choices[0].message.content
                
            elif "claude" in model_name.lower():
                logger.debug("Using Anthropic API for coordinate parsing")
                response = client.messages.create(
                    model=model_name,
                    max_tokens=4000,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ]
                )
                result = response.content[0].text
                
            else:
                # OpenRouter or other providers
                logger.debug("Using OpenRouter API for coordinate parsing")
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                result = response.choices[0].message.content
            
            # Parse JSON response
            json_str = result.replace("```json", "").replace("```", "").strip()
            return json.loads(json_str)
            
        except Exception as e:
            logger.error(f"Failed to parse coordinates from {model_name}: {str(e)}")
            return None
    
    def _crop_image_with_coordinates(self, image_content: bytes, coordinates: dict) -> bytes:
        """
        Crop an image using coordinates in 0-30 scale.
        
        Args:
            image_content: Original image content as bytes
            coordinates: Dictionary with A, B, C, D coordinates
            
        Returns:
            Cropped image as bytes
        """
        # Load the original image
        img = Image.open(io.BytesIO(image_content))
        width, height = img.size
        
        # Extract all coordinate points
        points = [coordinates["A"], coordinates["B"], coordinates["C"], coordinates["D"]]
        
        # Convert from 0-30 scale to pixel coordinates
        pixel_points = []
        for point in points:
            pixel_x = int(point[0] * width / 30)
            pixel_y = int((30 - point[1]) * height / 30)  # Invert Y since image origin is top-left
            pixel_points.append([pixel_x, pixel_y])
        
        # Find bounding box (min/max x and y coordinates)
        x_coords = [p[0] for p in pixel_points]
        y_coords = [p[1] for p in pixel_points]
        
        min_x = max(0, min(x_coords))
        max_x = min(width, max(x_coords))
        min_y = max(0, min(y_coords))
        max_y = min(height, max(y_coords))
        
        # Crop the image
        cropped_img = img.crop((min_x, min_y, max_x, max_y))
        
        # Save to bytes
        buffer = io.BytesIO()
        cropped_img.save(buffer, format='PNG')
        buffer.seek(0)
        
        return buffer.getvalue()
    
    def _extract_and_save_figures(self, result: str, original_image_content: bytes, coordinate_model: str, parsing_model: str) -> str:
        """
        Extract figures from OCR result and save them as separate images.
        
        Args:
            result: OCR result text
            original_image_content: Original image content as bytes
            coordinate_model: Model to use for figure coordinate extraction
            parsing_model: Model to use for coordinate parsing
            
        Returns:
            Modified result text with figure paths added
        """
        # Check if extract_figure is enabled and result contains figure references
        logger.info(f"Extracting figures from result: {result}")
        figure_pattern = re.compile(r'^\s*(?:\*\*)?(?:##\s*)?Figure\s*(\d+)(?:\*\*)?:?\s*', re.IGNORECASE | re.MULTILINE)
        matches = figure_pattern.finditer(result)

        matches = [match.group(0).strip() for match in matches]
        
        if not matches:
            logger.info(f"No figure references found in result: {result}")
            return result
        
        logger.info(f"Found {len(matches)} figure references: {matches}")
        
        # Ensure images directory exists
        images_dir = "images"
        os.makedirs(images_dir, exist_ok=True)
        
        # Collect all absolute paths for final insertion
        figure_paths = []
        
        for figure_number in matches:
            try:
                logger.info(f"Processing Figure {figure_number}")
                
                # Add coordinate grid to the image
                image_with_grid = self._add_coordinate_grid_to_image(original_image_content)
                
                # Get coordinates from model
                figure_description = f"{figure_number}"
                coordinate_description = self._get_figure_coordinates_from_model(
                    image_with_grid, figure_description, coordinate_model
                )
                
                if not coordinate_description:
                    logger.error(f"Failed to get coordinates for Figure {figure_number}")
                    continue
                
                # Parse coordinates
                coordinates = self._parse_coordinates_from_model(coordinate_description, parsing_model)
                
                if not coordinates:
                    logger.error(f"Failed to parse coordinates for Figure {figure_number}")
                    continue
                
                logger.info(f"Extracted coordinates for Figure {figure_number}: {coordinates}")
                
                # Crop the image
                cropped_image = self._crop_image_with_coordinates(original_image_content, coordinates)
                
                # Generate unique filename
                unique_id = str(uuid.uuid4())[:8]
                filename = f"figure_{figure_number.replace('*', '').replace(':', '').replace('#', '').strip()[-1]}_{unique_id}.png"
                filepath = os.path.join(images_dir, filename)
                
                # Save the cropped image
                with open(filepath, 'wb') as f:
                    f.write(cropped_image)
                
                logger.info(f"Saved Figure {figure_number} to {filepath}")
                
                # Get absolute path and store for later insertion
                absolute_path = os.path.abspath(filepath)
                figure_paths.append(f"The path of {figure_number} is {absolute_path}")
                
            except Exception as e:
                logger.error(f"Failed to process Figure {figure_number}: {str(e)}")
                continue
        
        # Insert all figure paths just before [END COPY TEXT] marker
        modified_result = result
        if figure_paths:
            paths_text = "\n\n" + "\n".join(figure_paths)
            if "[END COPY TEXT]" in result:
                modified_result = result.replace("[END COPY TEXT]", f"{paths_text}\n[END COPY TEXT]")
            else:
                # If no END marker, append at the end
                modified_result = result + paths_text
        else:
            modified_result = result
        
        return modified_result
    
    def convert(
        self, 
        file_path: str, 
        format_options: Optional[Dict[str, Any]] = None,
        max_retry: Optional[int] = None,
        additional_instructions: Optional[str] = None,
        clean_output: bool = True,
        description_mode: bool = False, 
        extract_figure = False,
        coordinate_model: str = "google/gemini-2.0-flash", 
        parsing_model: str = "google/gemini-2.5-flash-lite"
    ) -> List[str]:
        """
        Convert an image or PDF to markdown, or describe it.
        
        Args:
            file_path: Path to the image or PDF file
            format_options: Options for formatting the output
            max_retry: Override the default max retries
            additional_instructions: Additional instructions for the prompt
            clean_output: If True, removes markdown fences and START/END COPY TEXT markers
            description_mode: If True, generate a description instead of markdown
            extract_figure: If True, extract figures and save them as separate images
            coordinate_model: Model to use for figure coordinate extraction
            parsing_model: Model to use for coordinate parsing
            
        Returns:
            For a single image: markdown string or description string
            For a PDF: list of markdown strings or description strings, one per page
        """
        logger.info(f"Starting conversion of {file_path} (Description mode: {description_mode})")
        
        if max_retry is not None:
            self.retry_policy.max_attempts = max_retry
            logger.info(f"Max retry attempts updated to {max_retry}")
        
        # Process the file
        contents = self._process_file(file_path)
        results = []
        
        for i, content in enumerate(contents):
            logger.info(f"Processing content {i+1}/{len(contents)}")
            
            # Try with primary model
            logger.info(f"Attempting conversion with primary model: {self.model}")
            result = self._convert_with_model(
                self.model, 
                content, 
                format_options=format_options,
                additional_instructions=additional_instructions,
                description_mode=description_mode,
                extract_figure=extract_figure,
                coordinate_model=coordinate_model,
                parsing_model=parsing_model
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
                        description_mode=description_mode,
                        extract_figure=extract_figure,
                        coordinate_model=coordinate_model,
                        parsing_model=parsing_model
                    )
                    if result:
                        logger.info(f"Fallback model {fallback_model} succeeded")
                        break
                    else:
                        logger.info(f"Fallback model {fallback_model} failed")
            
            if result:
                logger.info(f"Conversion successful for content {i+1}")
                logger.debug(f"Result length: {len(result)} characters")
                
                # Clean the output if requested
                if clean_output and result != "Conversion failed with all models":
                    original_length = len(result)
                    logger.info(f"Original result: {result}")
                    logger.info(f"Original result length: {original_length} characters")
                    result = remove_markdown_and_markers(result)
                    new_length = len(result)
                    logger.info(f"Cleaned output by removing markdown fences and markers (reduced from {original_length} to {new_length} characters)")
            else:
                logger.error(f"Conversion failed for content {i+1} with all models")
            
            results.append(result or "Conversion failed with all models")
        
        # Return single result for images, list for PDFs
        logger.info(f"Conversion complete, returning {len(results)} results")
        return results
    
    async def async_convert(
        self, 
        file_path: str, 
        format_options: Optional[Dict[str, Any]] = None,
        max_retry: Optional[int] = None,
        additional_instructions: Optional[str] = None,
        clean_output: bool = True,
        description_mode: bool = False,
        extract_figure = False,
        coordinate_model: str = "gemini-2.0-flash-001", 
        parsing_model: str = "gemini-2.5-flash-lite"
    ) -> List[str]:
        """
        Async version of convert that processes multiple content items concurrently.
        Convert an image or PDF to markdown, or describe it.
        
        Args:
            file_path: Path to the image or PDF file
            format_options: Options for formatting the output
            max_retry: Override the default max retries
            additional_instructions: Additional instructions for the prompt
            clean_output: If True, removes markdown fences and START/END COPY TEXT markers
            description_mode: If True, generate a description instead of markdown
            extract_figure: If True, extract figures and save them as separate images
            coordinate_model: Model to use for figure coordinate extraction
            parsing_model: Model to use for coordinate parsing
            
        Returns:
            For a single image: markdown string or description string
            For a PDF: list of markdown strings or description strings, one per page
        """
        logger.info(f"Starting async conversion of {file_path} (Description mode: {description_mode})")
        
        if max_retry is not None:
            self.retry_policy.max_attempts = max_retry
            logger.info(f"Max retry attempts updated to {max_retry}")
        
        # Process the file (synchronous as requested)
        contents = self._process_file(file_path)
        
        async def process_single_content(i: int, content: bytes) -> str:
            """Process a single content item asynchronously."""
            logger.info(f"Processing content {i+1}/{len(contents)}")
            
            # Run the synchronous _convert_with_model in a thread pool
            loop = asyncio.get_event_loop()
            
            # Try with primary model
            logger.info(f"Attempting conversion with primary model: {self.model}")
            with concurrent.futures.ThreadPoolExecutor() as executor:
                result = await loop.run_in_executor(
                    executor,
                    self._convert_with_model,
                    self.model,
                    content,
                    format_options,
                    additional_instructions,
                    description_mode,
                    extract_figure,
                    coordinate_model,
                    parsing_model
                )
            
            # If primary model failed and we have fallbacks
            if result is None and self.fallback_models:
                logger.info(f"Primary model failed, trying {len(self.fallback_models)} fallback models")
                for fallback_model in self.fallback_models:
                    logger.info(f"Attempting conversion with fallback model: {fallback_model}")
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        result = await loop.run_in_executor(
                            executor,
                            self._convert_with_model,
                            fallback_model,
                            content,
                            format_options,
                            additional_instructions,
                            description_mode,
                            extract_figure,
                            coordinate_model,
                            parsing_model
                        )
                    if result:
                        logger.info(f"Fallback model {fallback_model} succeeded")
                        break
                    else:
                        logger.info(f"Fallback model {fallback_model} failed")
            
            if result:
                logger.info(f"Conversion successful for content {i+1}")
                logger.debug(f"Result length: {len(result)} characters")
                
                # Clean the output if requested
                if clean_output and result != "Conversion failed with all models":
                    original_length = len(result)
                    logger.info(f"Original result: {result}")
                    logger.info(f"Original result length: {original_length} characters")
                    result = remove_markdown_and_markers(result)
                    new_length = len(result)
                    logger.info(f"Cleaned output by removing markdown fences and markers (reduced from {original_length} to {new_length} characters)")
            else:
                logger.error(f"Conversion failed for content {i+1} with all models")
            
            return result or "Conversion failed with all models"
        
        # Process all content items concurrently
        tasks = [process_single_content(i, content) for i, content in enumerate(contents)]
        results = await asyncio.gather(*tasks)
        
        # Return results
        logger.info(f"Async conversion complete, returning {len(results)} results")
        return results
    
    def get_clean_markdown(self, markdown: str) -> str:
        """
        Extract the content between START COPY TEXT and END COPY TEXT markers.
        
        Args:
            markdown: The markdown string with markers
            
        Returns:
            The content between the markers, or the original string if markers not found
        """
        return extract_content_between_markers(markdown)
    
    def get_clean_content(self, markdown: str) -> str:
        """
        Remove both markdown fence tags and START/END COPY TEXT markers.
        
        Args:
            markdown: The original markdown string
            
        Returns:
            Clean markdown content without fences or markers
        """
        return remove_markdown_and_markers(markdown)
    
    def validate_markdown(self, markdown: str, description_mode: bool = False, extract_figure = False) -> Tuple[bool, str]:
        """
        Validate the generated markdown (or description) for structure and markers.
        
        Args:
            markdown: Generated markdown string or description
            
        Returns:
            Tuple of (is_valid, validation_message)
        """
        # Skip validation if generation failed
        if markdown == "Conversion failed with all models":
            logger.error(f"Generation failed: {markdown}")
            return False, "Generation failed"
        
        if (not is_valid_markdown(markdown)) and (not has_copy_markers(markdown)):
            if not is_valid_markdown(markdown):
                logger.error(f"Invalid markdown structure is_valid_markdown : {markdown}")
                return False, "Invalid markdown structure"
                
            # Marker validation (applies to both modes)
            if not has_copy_markers(markdown):
                logger.error(f"Missing required START/END COPY TEXT markers: {markdown}")
                return False, "Missing required START/END COPY TEXT markers"
            
        return True, "Validation successful"
