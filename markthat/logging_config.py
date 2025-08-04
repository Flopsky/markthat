"""Central logging configuration for *markthat*.

Importing this module **configures** a sensible default logging setup that
writes INFO-level logs of the library to stderr without affecting the root
logger configuration used by the host application.

Applications can call :func:`configure` again to override the defaults or
provide their own handlers.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

_LOGGER_NAME = "markthat"
_DEFAULT_LEVEL = logging.INFO


def configure(
    level: int = _DEFAULT_LEVEL, handlers: Optional[Iterable[logging.Handler]] = None
) -> None:
    """Configure *markthat* library logging.

    The function is intentionally *idempotent* – calling it multiple times will
    not add duplicate handlers.

    Parameters
    ----------
    level:
        Logging level to set for the library logger, defaults to :pydata:`logging.INFO`.
    handlers:
        Optional iterable of custom handlers.  If omitted, a standard
        :class:`logging.StreamHandler` pointing to *stderr* is installed.
    """

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level)

    if logger.handlers:
        # Logger already configured – avoid duplicate handlers.
        return

    if handlers is None:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(
            logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")
        )
        handlers = (stream_handler,)

    for h in handlers:
        logger.addHandler(h)


# Configure on first import with defaults.
configure()


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child logger scoped under *markthat* namespace."""

    return logging.getLogger(f"{_LOGGER_NAME}{'.' if name else ''}{name or ''}")
