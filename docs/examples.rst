Examples
========

Basic Usage Examples
--------------------

Simple Image Conversion
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from markthat import MarkThat

   converter = MarkThat(model="gemini-2.0-flash-001")
   result = converter.convert("screenshot.png")
   print(result[0])

Advanced Configuration
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from markthat import MarkThat, RetryPolicy

   # Custom retry policy
   retry_policy = RetryPolicy(
       max_attempts=5,
       timeout_seconds=30,
       backoff_factor=1.5
   )

   # Multi-provider setup with fallbacks
   converter = MarkThat(
       model="gpt-4o",
       provider="openai",
       fallback_models=["claude-3-5-sonnet-20241022", "gemini-2.0-flash-001"],
       retry_policy=retry_policy,
       api_key="YOUR_OPENAI_KEY"
   )

   result = converter.convert("complex_image.png")

Figure Extraction Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   converter = MarkThat(
       model="gemini-2.0-flash-001",
       api_key_figure_detector="DETECTOR_KEY",
       api_key_figure_extractor="EXTRACTOR_KEY", 
       api_key_figure_parser="PARSER_KEY"
   )

   results = await converter.async_convert(
       "path/to/document.pdf",
       extract_figure=True,
       figure_detector_model="gemini-2.0-flash",
       coordinate_model="gemini-2.0-flash-001",
       parsing_model="gemini-2.5-flash-lite"
   )

OpenRouter Integration
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # OpenRouter automatically detected for models with "/" format
   openrouter_converter = MarkThat(
       model="anthropic/claude-3.5-sonnet",  # Auto-detects OpenRouter
       api_key="YOUR_OPENROUTER_API_KEY"
   )

   # Or explicitly specify OpenRouter provider
   explicit_converter = MarkThat(
       model="openai/gpt-4o",
       provider="openrouter",
       api_key="YOUR_OPENROUTER_API_KEY"
   )

   # Multi-provider fallbacks through OpenRouter
   multi_provider_converter = MarkThat(
       model="anthropic/claude-3.5-sonnet",
       fallback_models=["openai/gpt-4o", "google/gemini-2.0-flash"],
       api_key="YOUR_OPENROUTER_API_KEY"
   )

   # Convert image
   result = openrouter_converter.convert("path/to/image.jpg")
   print(result)

Error Handling
~~~~~~~~~~~~~~

.. code-block:: python

   from markthat import MarkThat
   from markthat.exceptions import ConversionError, ProviderInitializationError

   try:
       converter = MarkThat(model="gemini-2.0-flash-001")
       result = converter.convert("image.jpg")
   except ProviderInitializationError as e:
       print(f"Provider setup failed: {e}")
   except ConversionError as e:
       print(f"Conversion failed: {e}")