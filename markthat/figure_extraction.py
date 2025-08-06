"""High-level helper functions for figure detection & extraction.

The implementation mirrors the behaviour of the original *markthat* code but
is split into smaller, easier-to-test helpers.  All external calls to
LLM providers go through :pymod:`markthat.providers` and Jinja2 templates
from :pymod:`markthat.prompts`.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .image_processing import add_coordinate_grid, crop_with_coordinates
from .langchain_providers import unified_langchain_call
from .logging_config import get_logger
from .prompts import get_prompt_for_model, load_template
from .providers import get_client
from .utils.validation import strip_fences_and_markers

logger = get_logger(__name__)

IMAGES_DIR = Path("images")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_figures(
    paginated_results: List[str],
    *,
    model: str,
    api_key: str | None = None,
) -> List[Dict[str, Any]]:
    """Analyse *paginated_results* and return list with figure metadata."""

    if not paginated_results:
        return []

    logger.info(f"Analyzing {len(paginated_results)} pages for figure detection using {model}")

    # Format paginated content for analysis - EXACTLY like original
    paginated_content = "\n".join(
        [f"START_PAGE{i}\n{page}\n[END_PAGE{i}]" for i, page in enumerate(paginated_results)]
    )

    logger.debug(f"Formatted content length: {len(paginated_content)} characters")

    try:
        provider_key = _infer_provider_from_model(model)
        get_client(provider_key, api_key=api_key)

        # Use EXACT same prompts as original
        system_prompt = """You are an expert at analyzing document content to identify pages that contain figure illustrations (not just figure references or captions).

Your task is to identify pages that actually contain visual figures/charts/diagrams/images that are illustrated or embedded in the document, not just pages that mention or reference figures in text.

Return a JSON array of objects with this structure:
[
  {
    "page_number": 0,
    "figure_name": "Figure 1",
    "figure_description": "Description of what the figure shows based on the text context"
  }
]

Only include pages that actually contain the visual figure illustration itself. If no figures are found, return an empty array []."""

        user_prompt = f"""Analyze the following paginated document content and identify which pages contain actual figure illustrations:

{paginated_content}

Return only the JSON array as specified in the system prompt."""

        response = unified_langchain_call(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            api_key=api_key,
        )

        # Parse JSON response - EXACTLY like original
        json_str = response.replace("```json", "").replace("```", "").strip()
        figure_list = json.loads(json_str)
        logger.info(f"Figure list: {figure_list}")
        logger.info(
            f"Detected {len(figure_list)} figures across pages: {[f['page_number'] for f in figure_list]}"
        )
        return figure_list

    except Exception as e:
        logger.error(f"Failed to detect figures using {model}: {str(e)}")
        return []


def extract_single_figure(
    *,
    figure_info: Dict[str, Any],
    page_markdown: str,
    page_image_bytes: bytes,
    coordinate_model: str,
    parsing_model: str,
    api_key_coordinate: str | None = None,
    api_key_parse: str | None = None,
) -> str | None:
    """Extract the figure specified by *figure_info* and return its file path."""

    figure_name = figure_info["figure_name"]
    figure_description = figure_info["figure_description"]
    page_number = figure_info["page_number"]

    logger.info(f"Extracting {figure_name} from page {page_number}: {figure_description}")

    # 1. Overlay coordinate grid on the original image
    image_with_grid = add_coordinate_grid(page_image_bytes)

    # 2. Ask model for coordinates
    coord_desc = _get_coordinates_llm(
        image_with_grid,
        prompt=f"{figure_name}: {figure_description}",
        model=coordinate_model,
        api_key=api_key_coordinate,
    )
    if not coord_desc:
        return None

    # 3. Parse coordinates into JSON
    coords = _parse_coordinates_llm(coord_desc, model=parsing_model, api_key=api_key_parse)
    if coords is None:
        return None

    # 4. Crop
    cropped_bytes = crop_with_coordinates(page_image_bytes, coords)

    # 5. Save - EXACTLY like original

    # Ensure images directory exists - EXACTLY like original
    images_dir = "images"
    os.makedirs(images_dir, exist_ok=True)

    # Generate unique filename - EXACTLY like original
    unique_id = str(uuid.uuid4())[:8]
    # Clean figure name for filename
    clean_name = re.sub(r"[^\w\s-]", "", figure_name.lower())
    clean_name = re.sub(r"[-\s]+", "_", clean_name)
    filename = f"{clean_name}_page{page_number}_{unique_id}.png"
    filepath = os.path.join(images_dir, filename)

    # Save the cropped image - EXACTLY like original
    with open(filepath, "wb") as f:
        f.write(cropped_bytes)

    logger.info(f"Saved {figure_name} to {filepath}")

    # Get absolute path - EXACTLY like original
    absolute_path = os.path.abspath(filepath)
    return f"The path of {figure_name} is {absolute_path}"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


# Legacy unified call functions have been replaced with LangChain implementation
# in langchain_providers.py - unified_langchain_call()


def _infer_provider_from_model(model_name: str) -> str:
    lower = model_name.lower()
    if "/" in lower:
        return "openrouter"
    if "gemini" in lower:
        return "gemini"
    if "gpt" in lower:
        return "gpt"
    if "claude" in lower:
        return "claude"
    if "mistral" in lower:
        return "mistral"
    raise ValueError(f"Cannot infer provider from {model_name}")


# ------------------------------------------------------------------
# Detection prompt helpers
# ------------------------------------------------------------------

_SYSTEM_PROMPT_DETECT = """You are an expert at analyzing document content to identify pages that contain figure illustrations (not just figure references or captions).

Your task is to identify pages that actually contain visual figures/charts/diagrams/images that are illustrated or embedded in the document, not just pages that mention or reference figures in text.

Return a JSON array of objects with this structure:
[
  {
    \"page_number\": 0,
    \"figure_name\": \"Figure 1\",
    \"figure_description\": \"Description of what the figure shows based on the text context\"
  }
]

Only include pages that actually contain the visual figure illustration itself. If no figures are found, return an empty array []."""


def _build_user_prompt_detect(pages: List[str]) -> str:
    parts = []
    for i, page in enumerate(pages):
        parts.append(f"START_PAGE{i}\n{page}\nEND_PAGE{i}")
    return "\n".join(parts)


# ------------------------------------------------------------------
# Coordinate helpers
# ------------------------------------------------------------------


def _get_coordinates_llm(
    image_with_grid: bytes, *, prompt: str, model: str, api_key: str | None
) -> str | None:
    try:
        logger.info(f"Getting coordinates for figure: {prompt} using model: {model}")

        provider = _infer_provider_from_model(model)
        get_client(provider, api_key=api_key)

        sys_template = load_template("system_prompt_extract_figure.j2")
        user_template = load_template("user_prompt_extract_figure.j2")

        system_prompt = sys_template.render()
        user_prompt = user_template.render(prompt=prompt)

        result = unified_langchain_call(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_bytes=image_with_grid,
            mime_type="image/png",
            api_key=api_key,
        )

        logger.info(f"Response from {model} for figure {prompt}: {result}")
        return result

    except Exception as e:
        logger.error(f"Failed to get coordinates from {model}: {str(e)}")
        raise e


def _parse_coordinates_llm(
    description: str, *, model: str, api_key: str | None
) -> Dict[str, Tuple[int, int]] | None:
    try:
        logger.info(f"Parsing coordinates using model: {model}")

        provider = _infer_provider_from_model(model)
        get_client(provider, api_key=api_key)

        sys_template = load_template("system_prompt_get_coordinates.j2")
        user_template = load_template("user_prompt_get_coordinates.j2")

        system_prompt = sys_template.render()
        user_prompt = user_template.render(model_response=description)

        response = unified_langchain_call(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            api_key=api_key,
        )

        # Parse JSON response - EXACTLY like original
        json_str = response.replace("```json", "").replace("```", "").strip()
        return json.loads(json_str)

    except Exception as e:
        logger.error(f"Failed to parse coordinates from {model}: {str(e)}")
        return None
