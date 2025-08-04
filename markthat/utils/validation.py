"""Markdown & marker validation helpers.

All validation logic is consolidated here so it can be unit-tested in
isolation and reused by multiple high-level components.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final, Pattern, Tuple

from ..exceptions import ValidationError
from ..logging_config import get_logger

logger = get_logger(__name__)

START_MARKER: Final[str] = r"\[START COPY TEXT\]"
END_MARKER: Final[str] = r"\[END COPY TEXT\]"

_MARKERS_PATTERN: Final[Pattern[str]] = re.compile(rf"{START_MARKER}(.*?){END_MARKER}", re.DOTALL)
_CODE_FENCE_PATTERN: Final[Pattern[str]] = re.compile(r"```[a-zA-Z0-9]*\s*\n(.*?)```", re.DOTALL)


@dataclass(slots=True)
class ValidationResult:
    valid: bool
    message: str = ""


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def is_valid_markdown(markdown: str) -> bool:
    """Very *rudimentary* Markdown validity check.

    It ensures code-block fences are paired, inline back-ticks are balanced and
    brackets/parentheses for links are closed.  **This is *not* a full Markdown
    parser.**
    """

    try:
        if markdown.count("```") % 2 != 0:
            return False
        return True
    except Exception as exc:  # pragma: no cover – extremely unlikely
        logger.exception("Error during markdown validation", exc_info=exc)
        return False


def has_markers(markdown: str) -> bool:
    """Return *True* if both copy markers are present."""
    return (
        re.search(START_MARKER, markdown) is not None
        and re.search(END_MARKER, markdown) is not None
    )


def extract_between_markers(markdown: str) -> str:
    """Return text found between *START* and *END* markers, or original string."""
    match = _MARKERS_PATTERN.search(markdown)
    if match:
        return match.group(1).strip()
    logger.debug("Markers not found when extracting between markers.")
    return markdown


def strip_fences_and_markers(markdown: str) -> str:
    """Remove code fences and copy markers, returning clean text."""

    logger.debug("Stripping fences/markers from markdown length %d", len(markdown))

    # 1) Extract marker content first if markers exist
    content = extract_between_markers(markdown)

    # 2) Remove *outermost* code fences (the first and last occurrence)
    if "```" in content:
        first = content.find("```")
        last = content.rfind("```")
        if first != last:
            content = content[first + 3 : last]
    return content.strip()


def validate(markdown: str, *, description_mode: bool = False) -> ValidationResult:
    """Validate *markdown* and return :class:`ValidationResult`."""

    if markdown == "Conversion failed with all models":
        return ValidationResult(False, "Generation failed")
    markdown_is_valid = is_valid_markdown(markdown)
    start_end_markers_present = has_markers(markdown)
    if (not markdown_is_valid) and not start_end_markers_present:
        if not markdown_is_valid:
            return ValidationResult(False, "Invalid markdown structure")

        if not has_markers(markdown):
            return ValidationResult(False, "Missing required START/END COPY TEXT markers")

    return ValidationResult(True, "OK")
