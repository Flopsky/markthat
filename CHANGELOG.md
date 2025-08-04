# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Code refactor with modular architecture
- Type hints throughout the codebase
- Comprehensive test suite with pytest
- Pre-commit hooks for code quality
- Sphinx-ready documentation
- Custom exception hierarchy
- Structured logging configuration

### Changed
- Reorganized codebase into logical modules
- Improved error handling and validation
- Enhanced provider abstraction layer
- Streamlined retry logic with failure tracking

## [0.1.0] - 2024-01-XX

### Added
- Initial release of MarkThat
- Support for multiple LLM providers (OpenAI, Anthropic, Google, Mistral)
- OpenRouter integration for unified model access
- PDF and image processing capabilities
- Advanced figure extraction pipeline
- Async processing support
- Configurable retry policies
- Environment variable support for API keys

### Features
- **Multi-Provider Support**: Access to 300+ models through various providers
- **Figure Extraction**: Automated detection and extraction of figures from PDFs
- **Dual Mode Operation**: Convert to Markdown or generate descriptions
- **Async Processing**: Concurrent page processing for improved performance
- **Robust Retry Logic**: Intelligent retry with fallback models
- **Type Safety**: Full type annotations with mypy validation

### Dependencies
- openai >= 1.0.0
- anthropic >= 0.60.0
- google-generativeai >= 0.8.0
- mistralai >= 1.7.0
- pymupdf >= 1.26.0
- numpy >= 1.25.0
- matplotlib >= 3.7.0
- Pillow >= 10.0.0
- Jinja2 >= 3.1.0

## [0.0.1] - 2024-01-XX

### Added
- Initial prototype implementation
- Basic image to markdown conversion
- Support for OpenAI GPT models
- Simple retry mechanism

---

## Types of Changes

- **Added** for new features
- **Changed** for changes in existing functionality
- **Deprecated** for soon-to-be removed features
- **Removed** for now removed features
- **Fixed** for any bug fixes
- **Security** in case of vulnerabilities