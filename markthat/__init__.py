"""
MarkThat - A Python module for converting images to markdown using multimodal LLMs.
"""

from .client import MarkThat, RetryPolicy

__version__ = "0.1.0"
__all__ = ["MarkThat", "RetryPolicy"]
