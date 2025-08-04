"""Load image or PDF files into in-memory *bytes* representations.

The module abstracts away PyMuPDF / PIL details so that the rest of the
library only deals with raw bytes.
"""

from __future__ import annotations

import io
import os
from typing import List

import fitz  # PyMuPDF
from PIL import Image

from .logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------


def load_file(file_path: str) -> List[bytes]:
    """Return a list of image bytes for *file_path*.

    * For a single image file, the list contains one element.
    * For a PDF, each page is rendered to a separate JPEG at 300 DPI.
    """

    if not os.path.isfile(file_path):
        raise FileNotFoundError(file_path)

    path_lower = file_path.lower()
    if path_lower.endswith(".pdf"):
        return _load_pdf(file_path)

    return [_load_image(file_path)]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_image(path: str) -> bytes:
    logger.info("Loading image %s", path)
    with open(path, "rb") as handle:
        data = handle.read()
    logger.debug("Image size: %d bytes", len(data))
    return data


def _load_pdf(path: str) -> List[bytes]:
    logger.info("Loading PDF %s", path)

    doc = fitz.open(path)
    pages: List[bytes] = []

    for page_index, page in enumerate(doc):
        logger.debug("Rendering page %d", page_index + 1)
        # High dpi for quality
        pix = page.get_pixmap(dpi=300)
        buffer = io.BytesIO()
        pix.pil_save(buffer, format="JPEG", quality=95, optimize=True)  # type: ignore[attr-defined]
        pages.append(buffer.getvalue())

    logger.info("Loaded %d pages from PDF", len(pages))
    doc.close()
    return pages
