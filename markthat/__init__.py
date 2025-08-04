"""Top-level package for markthat.

This package contains a refactored, production–ready version of the original
`markthat` project.  End-users should only need to interact with the
`MarkThat` class which is re-exported at package level.
"""

from importlib import metadata as _metadata

# Version is taken from the installed package metadata when available.  Fallback
# to a hard-coded string for local development.
try:
    __version__: str = _metadata.version(__name__)  # type: ignore[arg-type]
except _metadata.PackageNotFoundError:  # pragma: no cover – during local dev
    __version__ = "0.0.0.dev0"

from .client import MarkThat  # noqa: E402  (import after version determination)

__all__ = [
    "MarkThat",
    "__version__",
]
