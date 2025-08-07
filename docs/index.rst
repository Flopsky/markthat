MarkThat Documentation
======================

A Python library for converting images and PDFs to Markdown or generating rich image descriptions using state-of-the-art multimodal LLMs.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart
   ui
   api
   examples

Quick Start
===========

Installation
------------

.. code-block:: bash

   pip install markthat

Basic Usage
-----------

.. code-block:: python

   from markthat import MarkThat

   # Initialize with your preferred model
   converter = MarkThat(
       model="gemini-2.0-flash-001",
       provider="google",
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

Features
========

- **Multiple Provider Support**: OpenAI, Anthropic, Google Gemini, Mistral, and OpenRouter
- **Dual Mode Operation**: Convert to Markdown or generate detailed descriptions
- **Advanced Figure Extraction**: Automatically detect, extract, and process figures from PDFs
- **Robust Retry Logic**: Intelligent retry with fallback models and failure feedback
- **Async Support**: Concurrent processing for improved performance
- **Easy Integration**: Simple API with comprehensive configuration options

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`