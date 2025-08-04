"""Custom exception hierarchy for *markthat*.

Having a dedicated exception tree allows callers to catch library-specific
errors without relying on brittle string comparisons.
"""

from __future__ import annotations


class MarkThatError(Exception):
    """Base-class for all *markthat* exceptions."""


class ProviderInitializationError(MarkThatError):
    """Raised when a provider client cannot be initialised (missing deps, bad key)."""


class ValidationError(MarkThatError):
    """Raised when generated Markdown or description fails structural validation."""


class ConversionError(MarkThatError):
    """Raised when the conversion pipeline exhausts all retries and still fails."""
