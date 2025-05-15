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
    
    def _validate_markdown(self, markdown: str) -> Tuple[bool, str]:
        """
        Validate the generated markdown for structure and markers.
        
        Args:
            markdown: Generated markdown string
            
        Returns:
            Tuple of (is_valid, validation_message)
        """
        # Skip validation if generation failed
        if markdown == "Conversion failed with all models":
            logger.error(f"Generation failed: {markdown}")
            return False, "Generation failed"
            
        if not is_valid_markdown(markdown):
            logger.error(f"Invalid markdown structure is_valid_markdown : {markdown}")
            return False, "Invalid markdown structure"
            
        if not has_copy_markers(markdown):
            logger.error(f"Missing required START/END COPY TEXT markers: {markdown}")
            return False, "Missing required START/END COPY TEXT markers"
            
        return True, "Validation successful"
    
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
                    
                    result = response.text
                    
                    # Validate the result
                    is_valid, validation_msg = self._validate_markdown(result)
                    if not is_valid:
                        logger.warning(f"Generated markdown validation failed: {validation_msg}")
                        # If it's missing markers but otherwise valid, we'll add them
                        if validation_msg == "Missing required START/END COPY TEXT markers" and is_valid_markdown(result):
                            logger.info("Adding missing markers to otherwise valid markdown")
                            result = f"[START COPY TEXT]\n{result}\n[END COPY TEXT]"
                            is_valid = True
                    
                    if is_valid:
                        logger.info(f"Gemini generation successful")
                        return result
                    else:
                        logger.error(f"Invalid markdown generated, will retry")
                        raise ValueError(f"Generated markdown validation failed: {validation_msg}")
                
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
                    is_valid, validation_msg = self._validate_markdown(result)
                    if not is_valid:
                        logger.warning(f"Generated markdown validation failed: {validation_msg}")
                        # If it's missing markers but otherwise valid, we'll add them
                        if validation_msg == "Missing required START/END COPY TEXT markers" and is_valid_markdown(result):
                            logger.info("Adding missing markers to otherwise valid markdown")
                            result = f"[START COPY TEXT]\n{result}\n[END COPY TEXT]"
                            is_valid = True
                    
                    if is_valid:
                        logger.info(f"OpenAI generation successful")
                        return result
                    else:
                        logger.error(f"Invalid markdown generated, will retry")
                        raise ValueError(f"Generated markdown validation failed: {validation_msg}")
                
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
                    is_valid, validation_msg = self._validate_markdown(result)
                    if not is_valid:
                        logger.warning(f"Generated markdown validation failed: {validation_msg}")
                        # If it's missing markers but otherwise valid, we'll add them
                        if validation_msg == "Missing required START/END COPY TEXT markers" and is_valid_markdown(result):
                            logger.info("Adding missing markers to otherwise valid markdown")
                            result = f"[START COPY TEXT]\n{result}\n[END COPY TEXT]"
                            is_valid = True
                    
                    if is_valid:
                        logger.info(f"Claude generation successful")
                        return result
                    else:
                        logger.error(f"Invalid markdown generated, will retry")
                        raise ValueError(f"Generated markdown validation failed: {validation_msg}")
                
                elif "mistral" in model_name.lower():
                    logger.debug("Using Mistral API")
                    response = client.chat(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"{user_prompt}\n\n[Image: data:image/jpeg;base64,{base64_image}]"}
                        ]
                    )
                    
                    result = response.choices[0].message.content
                    
                    # Validate the result
                    is_valid, validation_msg = self._validate_markdown(result)
                    if not is_valid:
                        logger.warning(f"Generated markdown validation failed: {validation_msg}")
                        # If it's missing markers but otherwise valid, we'll add them
                        if validation_msg == "Missing required START/END COPY TEXT markers" and is_valid_markdown(result):
                            logger.info("Adding missing markers to otherwise valid markdown")
                            result = f"[START COPY TEXT]\n{result}\n[END COPY TEXT]"
                            is_valid = True
                    
                    if is_valid:
                        logger.info(f"Mistral generation successful")
                        return result
                    else:
                        logger.error(f"Invalid markdown generated, will retry")
                        raise ValueError(f"Generated markdown validation failed: {validation_msg}")
                
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
        clean_output: bool = True
    ) -> Union[str, List[str]]:
        """
        Convert an image or PDF to markdown.
        
        Args:
            file_path: Path to the image or PDF file
            format_options: Options for formatting the output
            max_retry: Override the default max retries
            additional_instructions: Additional instructions for the prompt
            clean_output: If True, removes markdown fences and START/END COPY TEXT markers
            
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
        return results if len(results) > 1 else results[0]
    
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
    
    def validate_markdown(self, markdown: str) -> Tuple[bool, str]:
        """
        Validate the markdown for structure and required markers.
        
        Args:
            markdown: The markdown string to validate
            
        Returns:
            Tuple of (is_valid, validation_message)
        """
        return self._validate_markdown(markdown)
