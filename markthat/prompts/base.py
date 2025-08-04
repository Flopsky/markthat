"""Core prompt-handling helpers.

Prompts are stored as Jinja2 templates inside the *templates* directory that
lives adjacent to this module.  High-level components render the templates via
:func:`get_prompt_for_model`.
"""

from __future__ import annotations

import importlib.resources as pkg_resources
import textwrap
from pathlib import Path
from typing import Any, Dict, Optional

import jinja2

from ..logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Jinja environment & template loading
# ---------------------------------------------------------------------------

PROMPT_TEMPLATES_DIR: Path = Path(__file__).with_suffix("").parent / "templates"

_jinja_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(PROMPT_TEMPLATES_DIR)),
    trim_blocks=True,
    lstrip_blocks=True,
)


def load_template(name: str) -> jinja2.Template:  # noqa: D401 – simple return
    """Return the requested Jinja2 template from the *templates* directory."""
    try:
        return _jinja_env.get_template(name)
    except jinja2.TemplateNotFound as exc:
        raise FileNotFoundError(f"Prompt template '{name}' not found.") from exc


# ---------------------------------------------------------------------------
# Public helper – get_prompt_for_model
# ---------------------------------------------------------------------------

def get_prompt_for_model(
    model_name: str,
    *,
    format_options: Optional[Dict[str, Any]] = None,
    additional_instructions: Optional[str] = None,
    description_mode: bool = False,
) -> Dict[str, str]:
    """Return a dict with **system_prompt** and **user_prompt** strings.

    The function mirrors the behaviour of the original utility but is now
    contained in a single, well-documented place.
    """

    if description_mode:
        system_template_name = "system_prompt_describe.j2"
        user_template_name = "user_prompt_describe.j2"
    else:
        system_template_name = "system_prompt.j2"
        user_template_name = "user_prompt.j2"

    system_template = load_template(system_template_name)
    user_template = load_template(user_template_name)

    system_prompt = system_template.render(format_options=format_options or {})

    # Jinja2 treats *None* as an empty string when rendering, but we pass the
    # variable explicitly for clarity.
    user_prompt = user_template.render(additional_instructions=additional_instructions or "")

    logger.debug(
        "Rendered prompts for model %s — system=%d chars, user=%d chars",
        model_name,
        len(system_prompt),
        len(user_prompt),
    )

    return {
        "system_prompt": textwrap.dedent(system_prompt).strip(),
        "user_prompt": textwrap.dedent(user_prompt).strip(),
    }
