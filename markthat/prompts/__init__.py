"""Prompt utilities for *markthat*.

This sub-package exposes two helpers:

* :func:`load_template` – return a :pyclass:`jinja2.Template` from the internal
  *templates* directory.
* :func:`get_prompt_for_model` – render system & user prompts with optional
  parameters (same signature as the original helper).
"""

from __future__ import annotations

from .base import PROMPT_TEMPLATES_DIR, get_prompt_for_model, load_template

__all__ = [
    "load_template",
    "get_prompt_for_model",
    "PROMPT_TEMPLATES_DIR",
]
