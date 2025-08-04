# Contributing to MarkThat

Thank you for your interest in contributing to MarkThat! This document provides guidelines and information for contributors.

## 🚀 Getting Started

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/your-username/markthat.git
   cd markthat
   ```

2. **Set up development environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install in development mode
   pip install -e .[dev]
   
   # Set up pre-commit hooks
   pre-commit install
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys for testing
   ```

## 🔧 Development Workflow

### Code Quality Standards

This project maintains high code quality through automated tools:

- **Type Checking**: mypy for static type analysis
- **Code Formatting**: Black for consistent code style
- **Linting**: Ruff for fast, comprehensive linting
- **Import Sorting**: isort for organized imports
- **Pre-commit Hooks**: Automated quality checks on every commit

### Running Quality Checks

```bash
# Run all quality checks
pre-commit run --all-files

# Individual tools
black .                    # Format code
ruff check .              # Lint code
ruff check . --fix        # Fix auto-fixable issues
isort .                   # Sort imports
mypy markthat             # Type checking
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=markthat --cov-report=html

# Run specific test files
pytest tests/test_validation.py
pytest tests/test_providers.py

# Run tests with verbose output
pytest -v
```

## 📝 Code Guidelines

### Python Style

- **Python Version**: Minimum Python 3.9
- **Line Length**: 100 characters (configured in pyproject.toml)
- **Type Hints**: Required for all public functions and methods
- **Docstrings**: Required for all public modules, classes, and functions

### Code Structure

```python
"""Module docstring describing the module's purpose."""

from __future__ import annotations

import standard_library_imports
import third_party_imports

from .internal_imports import something

# Constants
CONSTANT_VALUE = "value"

class ExampleClass:
    """Class docstring describing the class."""
    
    def __init__(self, param: str) -> None:
        """Initialize with proper type hints."""
        self.param = param
    
    def public_method(self, arg: int) -> str:
        """Public method with full documentation."""
        return str(arg)
    
    def _private_method(self) -> None:
        """Private method (internal use)."""
        pass
```

### Docstring Format

Use Google-style docstrings:

```python
def function_example(param1: str, param2: int = 0) -> bool:
    """One-line summary of the function.
    
    Longer description if needed. Explain the function's behavior,
    any side effects, and important details.
    
    Args:
        param1: Description of the first parameter.
        param2: Description of the second parameter with default.
        
    Returns:
        Description of the return value.
        
    Raises:
        ValueError: When param1 is empty.
        ConnectionError: When unable to connect to service.
        
    Example:
        >>> result = function_example("test", 42)
        >>> print(result)
        True
    """
```

## 🧪 Testing Guidelines

### Test Structure

```python
import pytest
from markthat import MarkThat
from markthat.exceptions import ConversionError

class TestMarkThatBasic:
    """Test basic MarkThat functionality."""
    
    def test_initialization(self):
        """Test proper initialization."""
        converter = MarkThat(model="test-model")
        assert converter.model == "test-model"
    
    def test_invalid_model_raises_error(self):
        """Test that invalid models raise appropriate errors."""
        with pytest.raises(ConversionError):
            MarkThat(model="invalid-model")
    
    @pytest.mark.asyncio
    async def test_async_conversion(self):
        """Test async conversion functionality."""
        converter = MarkThat(model="test-model")
        # Test implementation
```

### Test Categories

- **Unit Tests**: Test individual functions and methods
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete workflows
- **Performance Tests**: Test performance characteristics

### Mocking External Services

```python
import pytest
from unittest.mock import Mock, patch

@patch('markthat.providers.get_client')
def test_conversion_with_mock(mock_get_client):
    """Test conversion with mocked provider."""
    mock_client = Mock()
    mock_client.generate.return_value = "mocked response"
    mock_get_client.return_value = mock_client
    
    converter = MarkThat(model="test-model")
    result = converter.convert("test.jpg")
    
    assert result[0] == "mocked response"
```

## 🏗️ Architecture Guidelines

### Module Organization

- **`client.py`**: Main API class and orchestration
- **`providers.py`**: LLM provider abstractions
- **`file_processor.py`**: File loading and processing
- **`image_processing.py`**: Image manipulation utilities
- **`figure_extraction.py`**: Figure detection and extraction
- **`prompts/`**: Prompt templates and rendering
- **`utils/`**: Shared utilities and helpers
- **`exceptions.py`**: Custom exception hierarchy

### Adding New Features

1. **Plan the feature**: Discuss in an issue first
2. **Design the API**: Consider backwards compatibility
3. **Implement core logic**: Add to appropriate module
4. **Add comprehensive tests**: Unit and integration tests
5. **Update documentation**: Docstrings and README
6. **Add example usage**: Update examples/

### Adding New Providers

1. **Create provider class** in `providers.py`:
   ```python
   class NewProvider(BaseProvider):
       ENV_VAR_NAME = "NEW_PROVIDER_API_KEY"
       
       def _create(self) -> Any:
           # Implementation
   ```

2. **Add to provider map**:
   ```python
   _PROVIDER_MAP = {
       # ...existing providers...
       "new_provider": NewProvider,
   }
   ```

3. **Add unified API call support** in `client.py`
4. **Add comprehensive tests**
5. **Update documentation**

## 📋 Pull Request Process

### Before Submitting

1. **Create an issue** (for significant changes)
2. **Fork the repository** and create a feature branch
3. **Make your changes** following the guidelines
4. **Add/update tests** for your changes
5. **Run the full test suite** and quality checks
6. **Update documentation** as needed

### PR Requirements

- [ ] All tests pass
- [ ] Code coverage maintained or improved
- [ ] Quality checks pass (pre-commit hooks)
- [ ] Documentation updated
- [ ] CHANGELOG.md updated (for user-facing changes)
- [ ] Examples updated (if applicable)

### PR Template

```markdown
## Description
Brief description of the changes.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes

## Checklist
- [ ] My code follows the style guidelines of this project
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
```

## 🐛 Issue Reporting

### Bug Reports

Please include:

- **Python version** and operating system
- **MarkThat version** 
- **Complete error traceback**
- **Minimal code example** to reproduce the issue
- **Expected vs. actual behavior**

### Feature Requests

Please include:

- **Clear description** of the proposed feature
- **Use case** and motivation
- **Proposed API** (if applicable)
- **Implementation considerations**

## 🎯 Areas for Contribution

### High Priority
- Additional LLM provider support
- Performance optimizations
- More comprehensive test coverage
- Documentation improvements

### Medium Priority
- Additional file format support
- Cost tracking features
- Batch processing capabilities
- Custom prompt template system

### Good First Issues
- Documentation improvements
- Example additions
- Test coverage improvements
- Bug fixes with clear reproduction steps

## 📞 Getting Help

- **Questions**: Use [GitHub Discussions](https://github.com/your-repo/markthat/discussions)
- **Bugs**: Create an [issue](https://github.com/your-repo/markthat/issues)
- **Features**: Discuss in [GitHub Discussions](https://github.com/your-repo/markthat/discussions) first

## 📜 Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). By participating, you are expected to uphold this code.

## 🏆 Recognition

Contributors will be recognized in:
- README.md contributors section
- CHANGELOG.md for their contributions
- GitHub releases

Thank you for contributing to MarkThat! 🚀