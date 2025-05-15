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
converter = MarkThat(primary_model="gemini-2.0-flash", api_key="YOUR_API_KEY")

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
    primary_model="gpt-4.1",
    max_retry=4,
    fallback_models=["claude-sonnet", "mistral-medium"],
    api_key="YOUR_API_KEY"
)
complex_image_md = converter_with_fallback.convert("path/to/complex_image.png")
print(complex_image_md)
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
    primary_model="gemini-2.0-flash",
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

## TODO 

- [x] Init the first feature: convert images to markdown with only images
- [x] Add support for pdfs
- [x] Add support for fallback on other models
- [x] Add support for custom retry policies
- [x] Add support for more models
- [ ] Add support for more file formats (e.g., TIFF, WEBP)
- [ ] Add support for more options (e.g., custom prompt templates per call)
- [ ] Implement `description_mode` for image descriptions
- [x] Add detailed logging for failures and retries
- [x] Implement `[START COPY TEXT]` and `[END COPY TEXT]` marker handling


## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 