Quick Start Guide
=================

This guide will help you get started with MarkThat quickly.

Installation
------------

Install MarkThat using pip:

.. code-block:: bash

   pip install markthat

Environment Setup
-----------------

Set up your API keys as environment variables:

.. code-block:: bash

   export OPENAI_API_KEY="your_openai_api_key"
   export ANTHROPIC_API_KEY="your_anthropic_api_key"
   export GEMINI_API_KEY="your_gemini_api_key"
   export MISTRAL_API_KEY="your_mistral_api_key"
   export OPENROUTER_API_KEY="your_openrouter_api_key"

Note
~~~~

For figure extraction you may pass dedicated keys via the `MarkThat` constructor:
`api_key_figure_detector`, `api_key_figure_extractor`, and `api_key_figure_parser`.
If omitted, they default to the main `api_key`.

Basic Usage
-----------

.. code-block:: python

   from markthat import MarkThat

   # Initialize with your preferred model
   converter = MarkThat(
       model="gemini-2.0-flash-001",
       provider="gemini",
       api_key="YOUR_API_KEY"
   )

   # Convert image to markdown
   result = converter.convert("path/to/image.jpg")
   print(result[0])

   # Generate image description
   description = converter.convert(
       "path/to/image.jpg", 
       description_mode=True
   )
   print(description[0])

Examples from basic_usage.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from markthat import MarkThat
   from dotenv import load_dotenv
   import os
   import asyncio

   load_dotenv()

   def test_markthat_with_figure_extraction():
       """Test MarkThat with advanced figure extraction capabilities."""
       try:
           client = MarkThat(
               provider="gemini",
               model="gemini-2.0-flash-001",
               api_key=os.getenv("GEMINI_API_KEY"),
               api_key_figure_detector=os.getenv("GEMINI_API_KEY"),
               api_key_figure_extractor=os.getenv("GEMINI_API_KEY"),
               api_key_figure_parser=os.getenv("GEMINI_API_KEY"),
           )

           result = asyncio.run(
               client.async_convert(
                   "path/to/document.pdf",
                   extract_figure=True,
                   coordinate_model="gemini-2.0-flash-001",
                   parsing_model="gemini-2.5-flash-lite",
               )
           )
           return result
       except Exception as e:
           print("Figure extraction failed:", e)
           return None

   def test_markthat_without_figure_extraction():
       """Test standard MarkThat conversion without figure extraction."""
       try:
           client = MarkThat(
               provider="gemini",
               model="gemini-2.0-flash-001",
               api_key=os.getenv("GEMINI_API_KEY"),
           )

           result = asyncio.run(
               client.async_convert(
                   "path/to/document.pdf",
                   extract_figure=False,
               )
           )
           return result
       except Exception as e:
           print("Standard conversion failed:", e)
           return None

   if __name__ == "__main__":
       # Test both approaches
       with_figures = test_markthat_with_figure_extraction()
       without_figures = test_markthat_without_figure_extraction()
       
       print("With figure extraction:", with_figures)
       print("Without figure extraction:", without_figures)