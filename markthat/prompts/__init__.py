"""
Prompt templates for different LLM providers using Jinja2.
"""

from .base_prompts import load_prompt_template, get_prompt_for_model
 
__all__ = [
    "load_prompt_template", 
    "get_prompt_for_model"
] 