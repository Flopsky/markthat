"""High-level client API for *markthat*.

Compared to the original *markthat.client* the new implementation focuses on
*clarity* and *testability*.  Heavy-lifting helpers live in dedicated modules:

* provider initialisation → :pymod:`markthat.providers`
* prompt rendering        → :pymod:`markthat.prompts`
* file loading            → :pymod:`markthat.file_processor`
* image manipulation      → :pymod:`markthat.image_processing`
* figure extraction       → :pymod:`markthat.figure_extraction`
* markdown validation     → :pymod:`markthat.utils.validation`

Only orchestration logic and retry/back-off reside here.
"""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .file_processor import load_file
from .langchain_providers import unified_langchain_call
from .logging_config import get_logger
from .prompts import get_prompt_for_model
from .providers import get_client
from .utils.validation import ValidationResult, strip_fences_and_markers, validate

logger = get_logger(__name__)

__all__ = ["RetryPolicy", "MarkThat"]


# ---------------------------------------------------------------------------
# Retry helpers
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RetryPolicy:
    max_attempts: int = 3
    timeout_seconds: int = 30
    backoff_factor: float = 1.0  # exponential back-off coefficient


class FailureTracker:
    """Keeps context between retry attempts for richer prompts."""

    def __init__(self) -> None:
        self._failures: List[str] = []

    # ------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------

    def add(self, message: str) -> None:  # noqa: D401 – simple helper
        logger.debug("Failure: %s", message)
        self._failures.append(message)

    def feedback(self) -> str:
        if not self._failures:
            return ""
        lines = ["Previous attempts failed:"] + [f"- {msg}" for msg in self._failures]
        lines.append("Please avoid these issues in your next answer.")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------


class MarkThat:
    """Public entry-point class mirroring the original API surface."""

    def __init__(
        self,
        model: str,
        *,
        provider: Optional[str] = None,
        fallback_models: Optional[Sequence[str]] = None,
        retry_policy: Optional[RetryPolicy] = None,
        api_key: Optional[str] = None,
        api_key_figure_detector: Optional[str] = None,
        api_key_figure_extractor: Optional[str] = None,
        api_key_figure_parser: Optional[str] = None,
        max_retry: int = 3,
    ) -> None:
        self.model = model
        self.provider = (provider or _infer_provider_from_model(model)).lower()
        self.fallback_models = list(fallback_models or [])
        self.retry_policy = retry_policy or RetryPolicy(max_attempts=max_retry)
        self.max_retry = max_retry
        self.api_key = api_key
        self.api_key_figure_detector = api_key_figure_detector or api_key
        self.api_key_figure_extractor = api_key_figure_extractor or api_key
        self.api_key_figure_parser = api_key_figure_parser or api_key

        logger.info(
            "MarkThat initialised – primary model: %s (provider=%s)", self.model, self.provider
        )
        if self.fallback_models:
            logger.info("Fallback models: %s", ", ".join(self.fallback_models))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def convert(
        self,
        file_path: str,
        *,
        format_options: Optional[Dict[str, Any]] = None,
        additional_instructions: Optional[str] = None,
        description_mode: bool = False,
        extract_figure: bool = False,
        figure_detector_model: str = "gemini-2.0-flash",
        coordinate_model: str = "gemini-2.0-flash",
        parsing_model: str = "gemini-2.5-flash-lite",
        max_retry: Optional[int] = None,
        clean_output: bool = True,
    ) -> List[str]:
        """Synchronously convert *file_path* to markdown descriptions."""

        if max_retry is not None:
            self.retry_policy.max_attempts = max_retry
            logger.info("Max retry attempts updated to %d", max_retry)

        images = load_file(file_path)
        results: List[str] = []
        for idx, image_bytes in enumerate(images):
            logger.info("Converting page %d/%d", idx + 1, len(images))
            result = self._convert_single(
                image_bytes, format_options, additional_instructions, description_mode
            )

            # Clean the output if requested
            if clean_output and result != "Conversion failed with all models":
                result = strip_fences_and_markers(result)

            results.append(result)

        # Optional figure extraction
        if extract_figure:
            from .figure_extraction import detect_figures, extract_single_figure

            figures = detect_figures(
                results,
                model=figure_detector_model,
                api_key=self.api_key_figure_detector,
            )
            if figures:
                logger.info("Detected %d figures across pages", len(figures))
                for fig in figures:
                    page_num = fig.get("page_number", -1)
                    if 0 <= page_num < len(images):
                        path = extract_single_figure(
                            figure_info=fig,
                            page_markdown=results[page_num],
                            page_image_bytes=images[page_num],
                            coordinate_model=coordinate_model,
                            parsing_model=parsing_model,
                            api_key_coordinate=self.api_key_figure_extractor,
                            api_key_parse=self.api_key_figure_parser,
                        )
                        if path:
                            results[page_num] += f"\n\n{path}"
            else:
                logger.info("No figures detected for extraction")

        return results

    async def async_convert(
        self,
        file_path: str,
        *,
        format_options: Optional[Dict[str, Any]] = None,
        additional_instructions: Optional[str] = None,
        description_mode: bool = False,
        extract_figure: bool = False,
        figure_detector_model: str = "gemini-2.0-flash",
        coordinate_model: str = "gemini-2.0-flash",
        parsing_model: str = "gemini-2.5-flash-lite",
        max_retry: Optional[int] = None,
        clean_output: bool = True,
    ) -> List[str]:
        """Async counterpart processing pages concurrently."""

        if max_retry is not None:
            self.retry_policy.max_attempts = max_retry
            logger.info("Max retry attempts updated to %d", max_retry)

        images = load_file(file_path)
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            tasks = [
                loop.run_in_executor(
                    pool,
                    self._convert_single,
                    img,
                    format_options,
                    additional_instructions,
                    description_mode,
                )
                for img in images
            ]
            results: List[str] = await asyncio.gather(*tasks)

        # Clean output if requested
        if clean_output:
            results = [
                strip_fences_and_markers(r) if r != "Conversion failed with all models" else r
                for r in results
            ]

        # Optional figure extraction (run sync in thread pool for simplicity)
        if extract_figure and results:
            from .figure_extraction import detect_figures, extract_single_figure

            loop = asyncio.get_event_loop()
            figures = detect_figures(
                results,
                model=figure_detector_model,
                api_key=self.api_key_figure_detector,
            )

            if figures:

                async def process_figure(fig):
                    page_num = fig.get("page_number", -1)
                    if 0 <= page_num < len(images):
                        with concurrent.futures.ThreadPoolExecutor() as p:
                            path = await loop.run_in_executor(
                                p,
                                lambda: extract_single_figure(
                                    figure_info=fig,
                                    page_markdown=results[page_num],
                                    page_image_bytes=images[page_num],
                                    coordinate_model=coordinate_model,
                                    parsing_model=parsing_model,
                                    api_key_coordinate=self.api_key_figure_extractor,
                                    api_key_parse=self.api_key_figure_parser,
                                ),
                            )
                        if path:
                            results[page_num] += f"\n\n{path}"

                await asyncio.gather(*(process_figure(f) for f in figures))

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _convert_single(
        self,
        image_bytes: bytes,
        format_options: Optional[Dict[str, Any]],
        additional_instructions: Optional[str],
        description_mode: bool,
    ) -> str:
        # 1. primary model
        result = self._attempt_convert(
            self.model,
            image_bytes,
            format_options,
            additional_instructions,
            description_mode,
        )
        if result is not None:
            return result

        # 2. fallbacks
        for fb in self.fallback_models:
            result = self._attempt_convert(
                fb,
                image_bytes,
                format_options,
                additional_instructions,
                description_mode,
            )
            if result is not None:
                return result

        logger.error("All models failed – returning placeholder failure text")
        return "Conversion failed with all models"

    # ------------------------------------------------------------
    # Core conversion attempt with retry loop
    # ------------------------------------------------------------

    def _attempt_convert(
        self,
        model_name: str,
        image_bytes: bytes,
        format_options: Optional[Dict[str, Any]],
        additional_instructions: Optional[str],
        description_mode: bool,
    ) -> Optional[str]:
        provider_key = (
            _infer_provider_from_model(model_name) if model_name != self.model else self.provider
        )
        get_client(provider_key, api_key=self.api_key)

        failure_tracker = FailureTracker()
        for attempt in range(1, self.retry_policy.max_attempts + 1):
            logger.info("%s – attempt %d/%d", model_name, attempt, self.retry_policy.max_attempts)
            prompts = get_prompt_for_model(
                model_name,
                format_options=format_options,
                additional_instructions=_merge_instructions(
                    additional_instructions, failure_tracker.feedback()
                ),
                description_mode=description_mode,
            )

            try:
                output = unified_langchain_call(
                    model=model_name,
                    system_prompt=prompts["system_prompt"],
                    user_prompt=prompts["user_prompt"],
                    image_bytes=image_bytes,
                    api_key=self.api_key,
                )
                val: ValidationResult = validate(output, description_mode=description_mode)
                if val.valid:
                    cleaned = strip_fences_and_markers(output)
                    return cleaned
                failure_tracker.add(val.message)
                raise ValueError(val.message)
            except Exception as exc:
                failure_tracker.add(str(exc))
                if attempt < self.retry_policy.max_attempts:
                    wait = self.retry_policy.backoff_factor * (2 ** (attempt - 1))
                    logger.info("Retrying in %.1fs", wait)
                    time.sleep(wait)
                else:
                    logger.error("Model %s exhausted retries", model_name)
        return None

    # ------------------------------------------------------------------
    # Helper methods for backward compatibility
    # ------------------------------------------------------------------

    def get_clean_markdown(self, markdown: str) -> str:
        """Extract content between START COPY TEXT and END COPY TEXT markers."""
        from .utils.validation import extract_between_markers

        return extract_between_markers(markdown)

    def get_clean_content(self, markdown: str) -> str:
        """Remove both markdown fence tags and START/END COPY TEXT markers."""
        return strip_fences_and_markers(markdown)

    def validate_markdown(self, markdown: str, description_mode: bool = False) -> Tuple[bool, str]:
        """Validate the generated markdown for structure and markers."""
        result = validate(markdown, description_mode=description_mode)
        return result.valid, result.message

    def figure_extraction(
        self, paginated_results: List[str], model_name: str = "gemini-2.0-flash"
    ) -> List[Dict[str, Any]]:
        """Analyze paginated OCR results to identify pages containing figure illustrations."""
        from .figure_extraction import detect_figures

        return detect_figures(
            paginated_results, model=model_name, api_key=self.api_key_figure_detector
        )


# ---------------------------------------------------------------------------
# Provider inference & unified API call
# ---------------------------------------------------------------------------


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


# Legacy unified call function has been replaced with LangChain implementation
# in langchain_providers.py - unified_langchain_call()


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _merge_instructions(base: Optional[str], feedback: str) -> str:
    if base and feedback:
        return f"{base}\n\n{feedback}"
    return base or feedback or ""
