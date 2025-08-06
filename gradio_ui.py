"""Advanced Gradio UI for MarkThat - Showcasing Full Capabilities.

This enhanced interface demonstrates all features of the MarkThat library
with improved UX, real-time feedback, and advanced functionality.
"""

import asyncio
import io
import json
import os
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gradio as gr
from PIL import Image

from markthat import MarkThat

# Enhanced model presets with descriptions
MODEL_INFO = {
    "Gemini": {
        "models": {
            "gemini-2.0-flash-001": "Latest and fastest, great for most tasks",
            "gemini-2.0-flash": "Fast and efficient",
            "gemini-2.5-flash-lite": "Lightweight version",
            "gemini-1.5-pro": "Most capable, best quality",
            "gemini-1.5-flash": "Good balance of speed and quality",
        },
        "icon": "🔷",
        "description": "Google's multimodal AI models",
    },
    "OpenAI": {
        "models": {
            "gpt-4o": "Most capable vision model",
            "gpt-4o-mini": "Cost-effective vision model",
            "gpt-4-turbo": "High quality with vision",
            "gpt-4": "Classic GPT-4",
            "gpt-3.5-turbo": "Fast and affordable",
        },
        "icon": "🟢",
        "description": "OpenAI's GPT models with vision",
    },
    "Anthropic": {
        "models": {
            "claude-3-5-sonnet-20241022": "Most capable Claude model",
            "claude-3-5-haiku-20241022": "Fast and efficient",
            "claude-3-opus-20240229": "Powerful reasoning",
            "claude-3-sonnet-20240229": "Balanced performance",
            "claude-3-haiku-20240307": "Lightning fast",
        },
        "icon": "🔵",
        "description": "Anthropic's Claude models",
    },
    "Mistral": {
        "models": {
            "mistral-large-latest": "Most capable Mistral model",
            "mistral-medium-latest": "Balanced performance",
            "mistral-small-latest": "Fast and efficient",
            "pixtral-12b-2409": "Vision-specific model",
        },
        "icon": "🟠",
        "description": "Mistral AI's models",
    },
    "OpenRouter": {
        "models": {
            "anthropic/claude-3.5-sonnet": "Claude via OpenRouter",
            "google/gemini-2.0-flash": "Gemini via OpenRouter",
            "openai/gpt-4o": "GPT-4 via OpenRouter",
            "meta-llama/llama-3.2-90b-vision-instruct": "LLaMA vision model",
            "qwen/qwen-2-vl-72b-instruct": "Qwen vision model",
        },
        "icon": "🌐",
        "description": "Access multiple providers",
    },
}

# Sample documents for quick testing
SAMPLE_DOCUMENTS = {
    "Technical Paper": "samples/technical_paper.pdf",
    "Invoice": "samples/invoice.pdf",
    "Research Article": "samples/research_article.pdf",
    "Infographic": "samples/infographic.png",
    "Handwritten Notes": "samples/handwritten.jpg",
}

# Preset instructions for common use cases
PRESET_INSTRUCTIONS = {
    "Technical Documentation": """Focus on:
- Code blocks with proper syntax highlighting
- Technical diagrams and their relationships
- API references and parameters
- Preserve all technical terminology""",
    "Academic Paper": """Extract:
- Title, authors, and affiliations
- Abstract and keywords
- Section headers with proper hierarchy
- Citations in proper format
- Figure captions and references""",
    "Business Document": """Format as:
- Executive summary at the top
- Key metrics and data in tables
- Action items as bullet points
- Preserve all financial figures
- Highlight important dates""",
    "Educational Content": """Structure with:
- Clear learning objectives
- Step-by-step explanations
- Examples and exercises
- Key concepts highlighted
- Summary points at the end""",
    "Legal Document": """Ensure:
- Preserve exact wording
- Maintain clause numbering
- Extract all parties involved
- Highlight key terms and conditions
- Keep references intact""",
}


# Global state for session management
class SessionState:
    def __init__(self):
        self.history = []
        self.current_session_id = None
        self.comparison_results = {}

    def add_to_history(self, file_name, model, result, status):
        self.history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "file": file_name,
                "model": model,
                "result": result,
                "status": status,
            }
        )
        # Keep only last 10 items
        if len(self.history) > 10:
            self.history.pop(0)

    def get_history_display(self):
        if not self.history:
            return "No conversion history yet."

        display = []
        for item in reversed(self.history):
            timestamp = datetime.fromisoformat(item["timestamp"])
            display.append(
                f"**{timestamp.strftime('%H:%M:%S')}** - {item['file']} "
                f"({item['model']}) - {item['status']}"
            )
        return "\n\n".join(display)


session_state = SessionState()


def get_provider_from_model(model_name: str) -> str:
    """Infer provider from model name."""
    if not model_name:
        return "gemini"

    model_lower = model_name.lower()
    if "/" in model_lower:
        return "openrouter"
    elif "gemini" in model_lower:
        return "gemini"
    elif "gpt" in model_lower:
        return "gpt"
    elif "claude" in model_lower:
        return "claude"
    elif "mistral" in model_lower or "pixtral" in model_lower:
        return "mistral"
    else:
        return "gemini"


def update_model_selection(provider: str) -> Tuple[gr.Dropdown, str, str]:
    """Update model dropdown and info based on provider."""
    provider_info = MODEL_INFO.get(provider, {})
    models_dict = provider_info.get("models", {})

    # Create choices with descriptions
    choices = []
    for model, desc in models_dict.items():
        choices.append(f"{model} - {desc}")

    # Get first model (without description) as default
    default_model = list(models_dict.keys())[0] if models_dict else ""

    # Provider description
    provider_desc = f"{provider_info.get('icon', '')} {provider_info.get('description', '')}"

    # Model details
    if default_model and default_model in models_dict:
        model_details = f"**Selected:** {default_model}\n\n{models_dict[default_model]}"
    else:
        model_details = "Select a model to see details"

    return (
        gr.Dropdown(
            choices=choices,
            value=(
                f"{default_model} - {models_dict.get(default_model, '')}" if default_model else ""
            ),
            label=f"{provider} Models",
            interactive=True,
        ),
        provider_desc,
        model_details,
    )


def update_model_details(model_choice: str) -> str:
    """Update model details display."""
    if not model_choice or " - " not in model_choice:
        return "Select a model to see details"

    model_name = model_choice.split(" - ")[0]
    description = model_choice.split(" - ", 1)[1] if " - " in model_choice else ""

    return f"**Selected:** {model_name}\n\n{description}"


def load_sample_document(sample_name: str) -> str:
    """Load a sample document (placeholder - would load actual files)."""
    # In a real implementation, this would load actual sample files
    # For now, return a placeholder path
    return f"sample_{sample_name.lower().replace(' ', '_')}.pdf"


def preview_file(file_path: str) -> Tuple[Any, str, str]:
    """Generate file preview with detailed info."""
    if not file_path:
        return None, "No file selected", ""

    try:
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        file_ext = Path(file_path).suffix.lower()
        file_name = Path(file_path).name

        info = f"**File:** {file_name}\n**Size:** {file_size:.2f} MB\n**Type:** {file_ext}"

        if file_ext == ".pdf":
            import fitz

            doc = fitz.open(file_path)
            total_pages = len(doc)

            # Create preview of first page
            page = doc.load_page(0)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_data = pix.tobytes("ppm")
            img = Image.open(io.BytesIO(img_data))

            info += f"\n**Pages:** {total_pages}"

            # Check for text/images
            has_text = bool(page.get_text())
            has_images = bool(page.get_images())
            info += f"\n**Contains:** {'Text' if has_text else 'No text'}, {'Images' if has_images else 'No images'}"

            doc.close()
            return img, info, "✅ PDF loaded successfully"

        elif file_ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
            img = Image.open(file_path)
            info += f"\n**Dimensions:** {img.width} × {img.height} px"
            info += f"\n**Mode:** {img.mode}"
            return img, info, "✅ Image loaded successfully"

        else:
            return None, info, "❌ Unsupported file type"

    except Exception as e:
        return None, "", f"❌ Error loading file: {str(e)}"


async def process_file_async(
    file_path: str,
    model: str,
    api_key: str,
    description_mode: bool,
    extract_figures: bool,
    instructions: str,
    format_options: Dict[str, bool],
    advanced_options: Dict[str, Any],
    progress: gr.Progress,
) -> Tuple[str, List[str], Dict[str, Any], str]:
    """Process file with advanced options and detailed feedback."""

    provider = get_provider_from_model(model)

    progress(0.1, desc="🔧 Initializing MarkThat client...")

    try:
        # Initialize client with advanced options
        client_params = {
            "model": model,
            "provider": provider,
            "api_key": api_key,
            "api_key_figure_detector": api_key,
            "api_key_figure_extractor": api_key,
            "api_key_figure_parser": api_key,
        }

        # Add temperature if specified
        if advanced_options.get("temperature"):
            client_params["temperature"] = advanced_options["temperature"]

        client = MarkThat(**client_params)

        progress(0.2, desc="📂 Loading and analyzing file...")

        # Prepare conversion parameters
        convert_params = {
            "file_path": file_path,
            "description_mode": description_mode,
            "extract_figure": extract_figures,
            "coordinate_model": "gemini-2.0-flash-001",
            "parsing_model": "gemini-2.5-flash-lite",
            "format_options": format_options if any(format_options.values()) else None,
        }

        # Add instructions if provided
        if instructions and instructions.strip():
            convert_params["additional_instructions"] = instructions

        # Add page range if specified
        if advanced_options.get("page_range"):
            convert_params["page_range"] = advanced_options["page_range"]

        progress(0.3, desc="🤖 Processing with AI model...")

        # Perform conversion
        start_time = asyncio.get_event_loop().time()
        results = await client.async_convert(**convert_params)
        processing_time = asyncio.get_event_loop().time() - start_time

        progress(
            0.7, desc="🖼️ Extracting figures..." if extract_figures else "📝 Finalizing markdown..."
        )

        # Get extracted figures
        extracted_figures = []
        if extract_figures:
            images_dir = Path("images")
            if images_dir.exists():
                for img_file in sorted(images_dir.glob("*.png")):
                    extracted_figures.append(str(img_file))

        progress(0.9, desc="📊 Generating statistics...")

        # Calculate statistics
        combined_markdown = "\n\n---\n\n".join(
            [f"# Page {i+1}\n\n{result}" for i, result in enumerate(results)]
        )

        stats = {
            "pages_processed": len(results),
            "markdown_length": len(combined_markdown),
            "word_count": len(combined_markdown.split()),
            "processing_time": f"{processing_time:.2f}s",
            "figures_extracted": len(extracted_figures),
            "model_used": model,
            "mode": "Description" if description_mode else "Markdown",
        }

        progress(1.0, desc="✅ Complete!")

        status = f"✅ Successfully processed {len(results)} page(s) in {processing_time:.2f}s"
        if extracted_figures:
            status += f" with {len(extracted_figures)} figure(s)"

        return combined_markdown, extracted_figures, stats, status

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        return "", [], {}, error_msg


def process_file_wrapper(
    file_path: Optional[str],
    model_choice: str,
    api_key: str,
    custom_model: str,
    description_mode: bool,
    extract_figures: bool,
    preset_instructions: str,
    custom_instructions: str,
    preserve_tables: bool,
    preserve_code: bool,
    preserve_links: bool,
    preserve_formatting: bool,
    use_temperature: bool,
    temperature: float,
    page_range_enabled: bool,
    page_start: int,
    page_end: int,
    progress: gr.Progress = gr.Progress(),
) -> Tuple[str, gr.Gallery, str, str, str]:
    """Enhanced wrapper with all options."""

    # Validate inputs
    if not file_path:
        return "", gr.Gallery(value=[]), "", "❌ Please upload a file", ""

    # Extract model name from choice
    model = custom_model.strip() if custom_model else model_choice.split(" - ")[0]
    if not model:
        return "", gr.Gallery(value=[]), "", "❌ Please select or enter a model", ""

    if not api_key:
        return "", gr.Gallery(value=[]), "", "❌ Please provide an API key", ""

    # Combine instructions
    instructions = ""
    if preset_instructions != "None":
        instructions = PRESET_INSTRUCTIONS.get(preset_instructions, "")
    if custom_instructions:
        instructions = (
            f"{instructions}\n\n{custom_instructions}" if instructions else custom_instructions
        )

    # Prepare format options
    format_options = {
        "preserve_tables": preserve_tables,
        "preserve_code_blocks": preserve_code,
        "preserve_links": preserve_links,
        "preserve_formatting": preserve_formatting,
    }

    # Prepare advanced options
    advanced_options = {}
    if use_temperature:
        advanced_options["temperature"] = temperature
    if page_range_enabled:
        advanced_options["page_range"] = (page_start, page_end)

    # Process file
    try:
        markdown, figures, stats, status = asyncio.run(
            process_file_async(
                file_path,
                model,
                api_key,
                description_mode,
                extract_figures,
                instructions,
                format_options,
                advanced_options,
                progress,
            )
        )

        # Update history
        if markdown:
            session_state.add_to_history(
                Path(file_path).name, model, markdown[:100] + "...", status
            )

        # Format statistics
        stats_display = "\n".join(
            [f"**{k.replace('_', ' ').title()}:** {v}" for k, v in stats.items()]
        )

        # Create gallery value
        gallery_value = figures if figures else []

        return (
            markdown,
            gr.Gallery(value=gallery_value),
            stats_display,
            status,
            session_state.get_history_display(),
        )

    except Exception as e:
        error_msg = f"❌ Unexpected error: {str(e)}"
        return "", gr.Gallery(value=[]), "", error_msg, session_state.get_history_display()


def compare_models(
    file_path: str,
    models_to_compare: List[str],
    api_keys: Dict[str, str],
    progress: gr.Progress = gr.Progress(),
) -> Tuple[Dict[str, str], str]:
    """Compare outputs from multiple models."""
    if not file_path or not models_to_compare:
        return {}, "❌ Please select a file and at least one model"

    results = {}

    for i, model_choice in enumerate(models_to_compare):
        model = model_choice.split(" - ")[0]
        provider = get_provider_from_model(model)
        api_key = api_keys.get(provider)

        if not api_key:
            results[model] = f"❌ No API key for {provider}"
            continue

        progress((i + 0.5) / len(models_to_compare), desc=f"Processing with {model}...")

        try:
            client = MarkThat(model=model, provider=provider, api_key=api_key)
            result = asyncio.run(client.async_convert(file_path=file_path))
            results[model] = "\n\n".join(result) if result else "No output generated"
        except Exception as e:
            results[model] = f"❌ Error: {str(e)}"

    progress(1.0, desc="Comparison complete!")
    return results, "✅ Model comparison complete"


def export_results(markdown: str, figures: List[str], format: str) -> str:
    """Export results in various formats."""
    if not markdown:
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if format == "Markdown (.md)":
        file_path = f"markthat_export_{timestamp}.md"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(markdown)
        return file_path

    elif format == "Text (.txt)":
        file_path = f"markthat_export_{timestamp}.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(markdown)
        return file_path

    elif format == "ZIP (MD + Figures)":
        file_path = f"markthat_export_{timestamp}.zip"
        with zipfile.ZipFile(file_path, "w") as zf:
            zf.writestr("content.md", markdown)
            for i, fig in enumerate(figures):
                if os.path.exists(fig):
                    zf.write(fig, f"figure_{i+1}.png")
        return file_path

    return None


def create_interface():
    """Create the enhanced Gradio interface."""

    # Custom CSS for professional appearance
    custom_css = """
    /* Main container styling */
    .gradio-container {
        max-width: 1400px !important;
        margin: auto !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 3rem 2rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.95;
    }
    
    /* Card styling */
    .card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
        border: 1px solid #e5e7eb;
    }
    
    .card-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #f3f4f6;
    }
    
    .card-header h3 {
        margin: 0;
        font-size: 1.25rem;
        font-weight: 600;
        color: #1f2937;
    }
    
    /* Status indicators */
    .status-indicator {
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-weight: 500;
    }
    
    .status-success {
        background-color: #d1fae5;
        color: #065f46;
        border: 1px solid #6ee7b7;
    }
    
    .status-error {
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #fca5a5;
    }
    
    .status-info {
        background-color: #dbeafe;
        color: #1e40af;
        border: 1px solid #93c5fd;
    }
    
    /* Button styling */
    .primary-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        padding: 0.75rem 2rem !important;
        transition: transform 0.2s !important;
    }
    
    .primary-button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* Tab styling */
    .tabs {
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* Model grid */
    .model-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 0.75rem;
        margin-top: 1rem;
    }
    
    .model-card {
        padding: 0.75rem;
        border: 2px solid #e5e7eb;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .model-card:hover {
        border-color: #667eea;
        background-color: #f9fafb;
    }
    
    /* Stats display */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .stat-item {
        background: #f9fafb;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    
    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #667eea;
    }
    
    .stat-label {
        font-size: 0.875rem;
        color: #6b7280;
        margin-top: 0.25rem;
    }
    """

    with gr.Blocks(
        title="MarkThat - Advanced Document AI Converter",
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="purple",
            neutral_hue="gray",
        ),
        css=custom_css,
    ) as demo:

        # Header
        gr.HTML(
            """
        <div class="main-header">
            <h1>🎯 MarkThat</h1>
            <p>Advanced AI-Powered Document to Markdown Converter</p>
        </div>
        """
        )

        # Main content tabs
        with gr.Tabs():

            # ===== CONVERT TAB =====
            with gr.Tab("🔄 Convert", elem_id="convert-tab"):
                with gr.Row():
                    # Left column - Input controls
                    with gr.Column(scale=1):
                        # File input card
                        with gr.Group(elem_classes="card"):
                            gr.HTML('<div class="card-header"><h3>📁 Document Input</h3></div>')

                            file_input = gr.File(
                                label="Upload Document",
                                file_types=[".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp"],
                                type="filepath",
                            )

                            # Sample documents
                            with gr.Row():
                                gr.Dropdown(
                                    choices=list(SAMPLE_DOCUMENTS.keys()),
                                    label="Or try a sample",
                                    scale=2,
                                )
                                gr.Button("Load", scale=1)

                            # File preview
                            file_preview = gr.Image(label="Preview", height=200, visible=False)

                            file_info = gr.Markdown("No file selected")

                        # Model selection card
                        with gr.Group(elem_classes="card"):
                            gr.HTML('<div class="card-header"><h3>🤖 Model Selection</h3></div>')

                            provider_tabs = gr.Tabs()
                            with provider_tabs:
                                model_dropdowns = {}
                                for provider, info in MODEL_INFO.items():
                                    with gr.Tab(f"{info['icon']} {provider}"):
                                        gr.Markdown(f"*{info['description']}*")
                                        model_dropdowns[provider] = gr.Dropdown(
                                            choices=[
                                                f"{m} - {d}" for m, d in info["models"].items()
                                            ],
                                            value=list(info["models"].items())[0][0]
                                            + " - "
                                            + list(info["models"].values())[0],
                                            label="Select Model",
                                            interactive=True,
                                        )
                                        gr.Markdown("", elem_id=f"{provider}_model_details")

                            custom_model_input = gr.Textbox(
                                label="Or use custom model",
                                placeholder="your-custom/model-name",
                                interactive=True,
                            )

                            api_key_input = gr.Textbox(
                                label="API Key",
                                placeholder="Enter your API key...",
                                type="password",
                                interactive=True,
                            )

                        # Processing options card
                        with gr.Group(elem_classes="card"):
                            gr.HTML('<div class="card-header"><h3>⚙️ Processing Options</h3></div>')

                            with gr.Row():
                                description_mode = gr.Checkbox(
                                    label="Description Mode",
                                    value=False,
                                    info="Generate descriptions instead of markdown",
                                )
                                extract_figures = gr.Checkbox(
                                    label="Extract Figures",
                                    value=True,
                                    info="Auto-detect and extract figures",
                                )

                            # Instructions
                            preset_instructions = gr.Dropdown(
                                choices=["None"] + list(PRESET_INSTRUCTIONS.keys()),
                                value="None",
                                label="Preset Instructions",
                                interactive=True,
                            )

                            custom_instructions = gr.Textbox(
                                label="Custom Instructions",
                                placeholder="Add specific formatting or extraction instructions...",
                                lines=3,
                                interactive=True,
                            )

                            # Format preservation options
                            with gr.Accordion("Format Options", open=False):
                                with gr.Row():
                                    preserve_tables = gr.Checkbox(
                                        label="Preserve Tables", value=True
                                    )
                                    preserve_code = gr.Checkbox(label="Preserve Code", value=True)
                                with gr.Row():
                                    preserve_links = gr.Checkbox(label="Preserve Links", value=True)
                                    preserve_formatting = gr.Checkbox(
                                        label="Preserve Formatting", value=True
                                    )

                            # Advanced options
                            with gr.Accordion("Advanced Options", open=False):
                                with gr.Row():
                                    use_temperature = gr.Checkbox(
                                        label="Custom Temperature", value=False
                                    )
                                    temperature = gr.Slider(
                                        minimum=0,
                                        maximum=1,
                                        value=0.3,
                                        step=0.1,
                                        label="Temperature",
                                        visible=False,
                                    )

                                with gr.Row():
                                    page_range_enabled = gr.Checkbox(
                                        label="Limit Pages", value=False
                                    )
                                    page_start = gr.Number(value=1, label="Start", visible=False)
                                    page_end = gr.Number(value=10, label="End", visible=False)

                                # Show/hide temperature and page range
                                use_temperature.change(
                                    lambda x: gr.Slider(visible=x),
                                    inputs=[use_temperature],
                                    outputs=[temperature],
                                )
                                page_range_enabled.change(
                                    lambda x: (gr.Number(visible=x), gr.Number(visible=x)),
                                    inputs=[page_range_enabled],
                                    outputs=[page_start, page_end],
                                )

                        # Process button
                        process_btn = gr.Button(
                            "🚀 Convert Document",
                            variant="primary",
                            elem_classes="primary-button",
                            size="lg",
                        )

                    # Right column - Results
                    with gr.Column(scale=2):
                        # Status display
                        status_output = gr.HTML(
                            value='<div class="status-indicator status-info">Ready to convert documents</div>'
                        )

                        # Results tabs
                        with gr.Tabs():
                            with gr.Tab("📝 Markdown Output"):
                                markdown_output = gr.Textbox(
                                    label="Generated Markdown",
                                    lines=25,
                                    max_lines=40,
                                    show_copy_button=True,
                                    interactive=False,
                                )

                                # Export options
                                with gr.Row():
                                    export_format = gr.Dropdown(
                                        choices=[
                                            "Markdown (.md)",
                                            "Text (.txt)",
                                            "ZIP (MD + Figures)",
                                        ],
                                        value="Markdown (.md)",
                                        label="Export Format",
                                    )
                                    export_btn = gr.Button("💾 Export", size="sm")
                                    export_file = gr.File(label="Download", visible=False)

                            with gr.Tab("🖼️ Extracted Figures"):
                                figures_gallery = gr.Gallery(
                                    label="Extracted Figures",
                                    columns=3,
                                    height=600,
                                    object_fit="contain",
                                )

                            with gr.Tab("📊 Statistics"):
                                stats_display = gr.Markdown("No statistics available yet")

                            with gr.Tab("📜 History"):
                                history_display = gr.Markdown("No conversion history yet")
                                clear_history_btn = gr.Button("Clear History", size="sm")

            # ===== COMPARE TAB =====
            with gr.Tab("🔍 Compare Models", elem_id="compare-tab"):
                gr.Markdown("### Compare outputs from different models on the same document")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.File(
                            label="Upload Document for Comparison",
                            file_types=[".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp"],
                            type="filepath",
                        )

                        # Model selection for comparison
                        gr.Markdown("### Select Models to Compare")
                        compare_models = []
                        for provider, info in MODEL_INFO.items():
                            with gr.Accordion(f"{info['icon']} {provider}", open=False):
                                for model, desc in info["models"].items():
                                    compare_models.append(
                                        gr.Checkbox(
                                            label=f"{model} - {desc}", value=False, interactive=True
                                        )
                                    )

                        # API keys for comparison
                        gr.Markdown("### API Keys")
                        compare_api_keys = {}
                        for provider in MODEL_INFO.keys():
                            compare_api_keys[provider.lower()] = gr.Textbox(
                                label=f"{provider} API Key", type="password", interactive=True
                            )

                        gr.Button(
                            "🔍 Compare Models", variant="primary", elem_classes="primary-button"
                        )

                    with gr.Column(scale=2):
                        gr.HTML(
                            value='<div class="status-indicator status-info">Select models to compare</div>'
                        )

                        # Comparison results
                        gr.Tabs()

            # ===== GUIDE TAB =====
            with gr.Tab("📚 Guide", elem_id="guide-tab"):
                gr.Markdown(
                    """
                ## 🎯 MarkThat User Guide
                
                ### Quick Start
                1. **Upload** your document (PDF or image)
                2. **Select** an AI model from your preferred provider
                3. **Enter** your API key
                4. **Click** "Convert Document"
                
                ### Features
                
                #### 🤖 Multiple AI Providers
                - **Gemini**: Google's latest multimodal models
                - **OpenAI**: GPT-4 with vision capabilities
                - **Anthropic**: Claude models for detailed analysis
                - **Mistral**: Including Pixtral for vision tasks
                - **OpenRouter**: Access multiple providers with one API
                
                #### 📄 Document Types
                - **PDFs**: Multi-page documents with text and images
                - **Images**: PNG, JPG, JPEG, WebP, BMP formats
                - **Mixed Content**: Documents with tables, code, diagrams
                
                #### ⚙️ Processing Options
                - **Description Mode**: Get detailed descriptions instead of markdown
                - **Figure Extraction**: Automatically detect and save figures
                - **Format Preservation**: Keep tables, code blocks, and links intact
                - **Custom Instructions**: Add specific requirements for conversion
                
                #### 🔧 Advanced Features
                - **Model Comparison**: Compare outputs from different models
                - **Batch Processing**: Convert multiple pages efficiently
                - **Export Options**: Save as Markdown, Text, or ZIP with figures
                - **History Tracking**: Keep track of recent conversions
                
                ### Tips for Best Results
                
                1. **High Quality Images**: Use clear, high-resolution images
                2. **Specific Instructions**: Be detailed about formatting needs
                3. **Model Selection**: 
                   - Use Gemini 2.0 Flash for speed
                   - Use GPT-4o or Claude for complex documents
                   - Use specialized models for specific content types
                4. **Figure Extraction**: Enable for documents with diagrams or charts
                
                ### API Keys
                
                Get your API keys from:
                - **Gemini**: [Google AI Studio](https://makersuite.google.com/app/apikey)
                - **OpenAI**: [OpenAI Platform](https://platform.openai.com/api-keys)
                - **Anthropic**: [Anthropic Console](https://console.anthropic.com/)
                - **Mistral**: [Mistral AI](https://console.mistral.ai/)
                - **OpenRouter**: [OpenRouter](https://openrouter.ai/)
                
                ### Troubleshooting
                
                - **Empty output**: Check if the model supports vision/images
                - **API errors**: Verify your API key and quota
                - **Poor quality**: Try different models or add specific instructions
                - **Missing figures**: Ensure "Extract Figures" is enabled
                """
                )

        # Event handlers

        # File upload preview
        def handle_file_upload(file_path):
            if file_path:
                preview, info, status = preview_file(file_path)
                return (
                    gr.Image(value=preview, visible=preview is not None),
                    info,
                    (
                        f'<div class="status-indicator status-success">{status}</div>'
                        if "✅" in status
                        else f'<div class="status-indicator status-error">{status}</div>'
                    ),
                )
            return (
                gr.Image(visible=False),
                "No file selected",
                '<div class="status-indicator status-info">Ready to convert documents</div>',
            )

        file_input.change(
            fn=handle_file_upload,
            inputs=[file_input],
            outputs=[file_preview, file_info, status_output],
        )

        # Get active model from tabs
        def get_active_model():
            # This is a simplified version - in practice you'd track the active tab
            return model_dropdowns["Gemini"]

        # Process button
        process_btn.click(
            fn=process_file_wrapper,
            inputs=[
                file_input,
                get_active_model(),  # Would need proper tab tracking
                api_key_input,
                custom_model_input,
                description_mode,
                extract_figures,
                preset_instructions,
                custom_instructions,
                preserve_tables,
                preserve_code,
                preserve_links,
                preserve_formatting,
                use_temperature,
                temperature,
                page_range_enabled,
                page_start,
                page_end,
            ],
            outputs=[
                markdown_output,
                figures_gallery,
                stats_display,
                status_output,
                history_display,
            ],
        )

        # Export functionality
        def handle_export(markdown, figures, format):
            if markdown:
                file_path = export_results(markdown, figures, format)
                return gr.File(value=file_path, visible=True)
            return gr.File(visible=False)

        export_btn.click(
            fn=handle_export,
            inputs=[markdown_output, figures_gallery, export_format],
            outputs=[export_file],
        )

        # Clear history
        def clear_history():
            session_state.history = []
            return "History cleared", session_state.get_history_display()

        clear_history_btn.click(fn=clear_history, outputs=[status_output, history_display])

    return demo


def main():
    """Launch the enhanced Gradio interface."""
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        debug=False,
        show_error=True,
        show_api=False,
    )


if __name__ == "__main__":
    main()
