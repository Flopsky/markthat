# MarkThat

A Python module for converting images to Markdown or generating image descriptions using state-of-the-art multimodal LLMs.

## Overview

MarkThat is a robust tool that leverages various multimodal LLMs to convert visual content into Markdown format or generate rich textual descriptions. It includes an intelligent retry mechanism to ensure high-quality outputs.

## Features

- Support for multiple multimodal LLMs:
  - Google Gemini 2.0 Flash
  - OpenAI GPT-4.1
  - Anthropic Claude Sonnet
  - Mistral Medium
  - **OpenRouter** (unified access to 300+ models)
  - And more...
- **Dual Mode**: Convert images to Markdown or generate detailed image descriptions.
- Automatic retry with different models if conversion/description fails.
- Configurable retry policies and failure feedback to guide retries.
- Lightweight and easy to integrate.

## Installation

```bash
pip install markthat
```

## Quick Start

```python
from markthat import MarkThat

# Initialize with your preferred model and API key
converter = MarkThat(model="gemini-2.0-flash", api_key="YOUR_API_KEY")

# Convert an image to markdown (default mode)
markdown_output = converter.convert("path/to/image.jpg", max_retry=4)
print(markdown_output)

# Generate a description of an image
description_output = converter.convert("path/to/image.jpg", description_mode=True, max_retry=4)
print(description_output)

# Convert a full PDF to markdown (each page processed separately)
pdf_markdown_pages = converter.convert("path/to/pdf.pdf", max_retry=4)
for page_md in pdf_markdown_pages:
    print(page_md)

# Generate descriptions for each page of a PDF
pdf_description_pages = converter.convert("path/to/pdf.pdf", description_mode=True, max_retry=4)
for page_desc in pdf_description_pages:
    print(page_desc)

# Using with fallbacks when each model fails all their retries
converter_with_fallback = MarkThat(
    model="gpt-4.1",
    max_retry=4,
    fallback_models=["claude-sonnet", "mistral-medium"],
    api_key="YOUR_API_KEY"
)
complex_image_md = converter_with_fallback.convert("path/to/complex_image.png")
print(complex_image_md)
```

## Async Usage

For better performance when processing multiple content items (like multi-page PDFs), use the async version:

```python
import asyncio
from markthat import MarkThat

async def main():
    # Initialize converter
    converter = MarkThat(
        model="gemini-2.0-flash", 
        api_key="YOUR_API_KEY"
    )
    
    # Convert a multi-page PDF asynchronously (pages processed concurrently)
    pdf_results = await converter.async_convert("path/to/document.pdf")
    for i, page_markdown in enumerate(pdf_results):
        print(f"Page {i+1}:\n{page_markdown}\n")
    
    # Generate descriptions for PDF pages concurrently
    pdf_descriptions = await converter.async_convert(
        "path/to/document.pdf", 
        description_mode=True
    )
    for i, description in enumerate(pdf_descriptions):
        print(f"Page {i+1} description:\n{description}\n")
    
    # Single image (still works but no concurrency benefit)
    image_result = await converter.async_convert("path/to/image.jpg")
    print(image_result[0])  # async_convert always returns a list

# Run the async function
asyncio.run(main())
```

**Benefits of Async Mode:**
- **Concurrent Processing**: Multi-page PDFs are processed simultaneously instead of sequentially
- **Better Performance**: Significantly faster for documents with multiple pages
- **Same API**: Identical parameters and behavior as the synchronous `convert` method
- **Thread-Safe**: Uses thread pools for the actual API calls while maintaining async benefits

## Provider Examples

### Direct Provider Access

Use specific providers directly with their native APIs:

```python
from markthat import MarkThat

# OpenAI GPT-4o
openai_converter = MarkThat(
    model="gpt-4o",
    provider="openai",
    api_key="YOUR_OPENAI_API_KEY"
)

# Anthropic Claude
claude_converter = MarkThat(
    model="claude-3-5-sonnet-20241022",
    provider="anthropic", 
    api_key="YOUR_ANTHROPIC_API_KEY"
)

# Google Gemini
gemini_converter = MarkThat(
    model="gemini-2.0-flash-exp",
    provider="google",
    api_key="YOUR_GEMINI_API_KEY"
)

# Mistral
mistral_converter = MarkThat(
    model="mistral-large-latest",
    provider="mistral",
    api_key="YOUR_MISTRAL_API_KEY"
)

# Convert image with any provider
result = openai_converter.convert("path/to/image.jpg")
print(result)
```

### OpenRouter - Unified Provider Access

Access 300+ models from multiple providers through a single API:

```python
from markthat import MarkThat

# OpenRouter automatically detected for models with "/" format
openrouter_converter = MarkThat(
    model="anthropic/claude-3.5-sonnet",  # Auto-detects OpenRouter
    api_key="YOUR_OPENROUTER_API_KEY"
)

# Or explicitly specify OpenRouter provider
explicit_converter = MarkThat(
    model="openai/gpt-4o",
    provider="openrouter",
    api_key="YOUR_OPENROUTER_API_KEY"
)

# Popular OpenRouter models
models = [
    "openai/gpt-4o",                    # OpenAI GPT-4o
    "anthropic/claude-3.5-sonnet",      # Anthropic Claude 3.5 Sonnet
    "google/gemini-pro-vision",         # Google Gemini Pro Vision
    "meta-llama/llama-3.2-90b-vision", # Meta Llama Vision
    "qwen/qwen-2-vl-72b-instruct"      # Qwen Vision
]

# Multi-provider fallbacks through OpenRouter
multi_provider_converter = MarkThat(
    model="anthropic/claude-3.5-sonnet",
    fallback_models=["openai/gpt-4o", "google/gemini-pro-vision"],
    api_key="YOUR_OPENROUTER_API_KEY"
)

# Convert image
result = openrouter_converter.convert("path/to/image.jpg")
print(result)
```

## Advanced Usage

```python
from markthat import MarkThat, RetryPolicy

# Configure custom retry policy
policy = RetryPolicy(
    max_attempts=5,
    timeout=30,
    backoff_factor=1.5
)

converter = MarkThat(
    model="gemini-2.0-flash",
    fallback_models=["gpt-4.1", "claude-sonnet"],
    retry_policy=policy,
    api_key="YOUR_API_KEY"
)

# Convert with additional options (e.g., for Markdown output)
markdown_with_options = converter.convert(
    "path/to/image.jpg",
    format_options={"include_tables": True, "code_syntax_highlighting": True},
    additional_instructions="Please ensure all tables are well-formatted."
)
print(markdown_with_options)

# Generate description with additional instructions
description_with_instructions = converter.convert(
    "path/to/image.jpg",
    description_mode=True,
    additional_instructions="Focus on the artistic style of the image."
)
print(description_with_instructions)

# Get raw output and clean it manually
raw_output = converter.convert("path/to/image.jpg", clean_output=False)
# ... inspect raw_output ...
clean_content = converter.get_clean_content(raw_output) # Removes markers and markdown fences
print(clean_content)

# Validate output
is_valid, message = converter.validate_markdown(raw_output) # Checks for markers and markdown structure (if not description_mode)
print(f"Is valid: {is_valid}, Message: {message}")
```

## Environment Variables

Set your API keys as environment variables for automatic detection:

```bash
# Direct providers
export OPENAI_API_KEY="your_openai_api_key"
export ANTHROPIC_API_KEY="your_anthropic_api_key"
export GEMINI_API_KEY="your_gemini_api_key"
export MISTRAL_API_KEY="your_mistral_api_key"

# OpenRouter (unified access)
export OPENROUTER_API_KEY="your_openrouter_api_key"
```

Then use without specifying API keys:

```python
from markthat import MarkThat

# Uses environment variables automatically
converter = MarkThat(model="anthropic/claude-3.5-sonnet")
result = converter.convert("image.jpg")
```

## OpenRouter Benefits

- **Unified API**: Access models from OpenAI, Anthropic, Google, and more through one interface
- **Cost Optimization**: Compare costs across providers and choose the most economical option
- **Model Availability**: Automatic fallback when your primary model is unavailable
- **300+ Models**: Access to the largest collection of multimodal LLMs
- **Enhanced Features**: Support for advanced features like PDF processing with OCR

## TODO 

- [x] Init the first feature: convert images to markdown with only images
- [x] Add support for pdfs
- [x] Add support for fallback on other models
- [x] Add support for custom retry policies
- [x] Add support for more models
- [x] Add OpenRouter support for unified provider access
- [ ] Add support for more file formats (e.g., TIFF, WEBP)
- [ ] Add support for more options (e.g., custom prompt templates per call)
- [ ] Implement `description_mode` for image descriptions
- [x] Add detailed logging for failures and retries
- [x] Implement `[START COPY TEXT]` and `[END COPY TEXT]` marker handling
- [ ] Add OpenRouter PDF processing with OCR engines
- [ ] Add cost tracking and optimization features


## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 