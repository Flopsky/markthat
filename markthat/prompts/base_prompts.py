"""
Base prompt utilities for loading and managing model prompts.
"""

import os
import json
from typing import Dict, Any, Optional
import jinja2


# Set up Jinja environment
_template_dir = os.path.join(os.path.dirname(__file__), 'templates')
_jinja_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(_template_dir),
    trim_blocks=True,
    lstrip_blocks=True
)


def load_prompt_template(template_name: str) -> jinja2.Template:
    """
    Load a Jinja2 template by name.
    
    Args:
        template_name: Name of the template file (e.g., 'system_prompt.j2')
    
    Returns:
        The loaded Jinja2 template
    """
    return _jinja_env.get_template(template_name)


def get_prompt_for_model(
    model_name: str, 
    format_options: Optional[Dict[str, Any]] = None,
    additional_instructions: Optional[str] = None
) -> Dict[str, str]:
    """
    Get the appropriate prompt configuration for a given model.
    
    Args:
        model_name: Name of the model
        format_options: Optional formatting options to include in the prompt
        additional_instructions: Additional instructions to include in the user prompt
    
    Returns:
        A dictionary with system and user prompts
    """
    # Load templates
    system_template = load_prompt_template("system_prompt.j2")
    user_template = load_prompt_template("user_prompt.j2")
    
    # Render templates with options
    system_prompt = system_template.render(format_options=format_options)
    user_prompt = user_template.render(additional_instructions=additional_instructions)
    
    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt
    } 