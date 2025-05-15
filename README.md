# MarkThat

A Python module for converting images to Markdown using state-of-the-art multimodal LLMs.

## Overview

MarkThat is a robust tool that leverages various multimodal LLMs to convert visual content into Markdown format. It includes an intelligent retry mechanism to ensure high-quality conversions.

## Features

- Support for multiple multimodal LLMs:
  - Google Gemini 2.0 Flash
  - OpenAI GPT-4.1
  - Anthropic Claude Sonnet
  - Mistral Medium
  - And more...
- Automatic retry with different models if conversion fails
- Configurable retry policies
- Lightweight and easy to integrate

## Installation

```bash
pip install markthat
```

## Quick Start

```python
from markthat import MarkThat

# Initialize with your preferred model
converter = MarkThat(primary_model="gemini-2.0-flash", api_key="YOUR_API_KEY")

# Convert an image to markdown
markdown = converter.convert("path/to/image.jpg", max_retry=4)
print(markdown)

#convert a full pdf to markdown
markdown = converter.convert("path/to/pdf.pdf", max_retry=4)
for page in markdown:
    print(page)

# Using with fallbacks when each model fails all their retries
converter = MarkThat(
    primary_model="gpt-4.1",
    max_retry=4,
    fallback_models=["claude-sonnet", "mistral-medium"],
    api_key="YOUR_API_KEY"
)
markdown = converter.convert("path/to/complex_image.png")
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

# Convert with additional options
markdown = converter.convert(
    "path/to/image.jpg",
    format_options={"include_tables": True, "code_syntax_highlighting": True},
    additional_instructions="Please ensure all tables are well-formatted."
)
```

## TODO 

- [x] Init the first feature: convert images to markdown with only images
- [ ] Add support for pdfs
- [ ] Add retry mechanism to detect when the model is looping on a value
- [ ] Add support for image description with custum prompts.
- [ ] Add supprot for fallback on other models (include litellm ?)
- [ ] Add support for custom retry policies
- [ ] Add support for more models
- [ ] Add support for more file formats
- [ ] Add support for more options


## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 