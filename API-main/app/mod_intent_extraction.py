import asyncio
import json
import re
from datetime import datetime, timezone
from typing import Any

import google.auth
from google import genai
from google.api_core import exceptions as google_exceptions
from google.genai.types import GenerateContentConfig
from pydantic import ValidationError

from app.config import (
    ENV,
    LOCATION,
    MODEL_VERSION,
    PROJECT_ID,
    SERVICE_NAME,
    SERVICE_VERSION,
)
from app.exceptions import (
    LLMError,
    LLMInvalidRequestError,
    LLMLocationError,
    LLMRateLimitError,
    LLMServiceError,
    LLMTimeoutError,
    TemporalIndexUnavailableError,
    _is_location_model_error,
)
from app.mod_requests import (
    IntentExtractionResponse,
    IntentExtractionResponseV2,
    QueryExpansionOutput,
    RepresentativeTermsOutput,
)
from app.prompts.prompt_intent_extraction import INTENT_EXTRACTION_PROMPT
from app.prompts.prompt_query_expansion import QUERY_EXPANSION_PROMPT
from app.prompts.v2.contextual_environment import CONTEXTUAL_ENVIRONMENT_PROMPT_V2
from app.prompts.v2.intent_extraction import INTENT_EXTRACTION_PROMPT_V2
from app.prompts.v2.query_expansion import QUERY_EXPANSION_PROMPT_V2
from app.prompts.v2.representative_terms import REPRESENTATIVE_TERMS_PROMPT_V2
from app.prompts.v3.contextual_environment import (
    build_contextual_environment_prompt_v3,
)
from app.utils.cadence_memory import CadenceMemory, window_label
from app.utils.concept_vocab import ConceptVocab
from app.utils.generation_config import build_generation_config, extract_usage_metadata
from app.utils.retrieval_signals import (
    ContextualEnvironmentOutput,
    ContextualEnvironmentOutputCanonical,
    ContextualEnvironmentOutputCanonicalSchema,
    TemporalExtractionOutput,
    assemble_v2_intents,
)
from app.utils.temporal_canonical import CanonicalTemporalResolver
from app.utils.temporal_index import CANONICAL_UNITS, TemporalIndex
from app.utils.temporal_vocab import (
    NullTemporalResolver,
    TemporalResolver,
    TemporalVocab,
)

# Temporal modes for the contextual-environment step.
TEMPORAL_MODE_VOCAB_LIST = "vocab_list"  # legacy: vocabulary pasted, model picks ids
TEMPORAL_MODE_CANONICAL = "canonical"  # /v3: model emits a window, code resolves it
TEMPORAL_MODES = (TEMPORAL_MODE_VOCAB_LIST, TEMPORAL_MODE_CANONICAL)

# LLM call timeout in seconds
LLM_TIMEOUT_SECONDS = 300

# Fixed seed for best-effort reproducibility across identical calls (temperature=0
# alone does not guarantee identical output run-to-run for "thinking" models).
LLM_SEED = 0


class ContextualIntentPipeline:
    """
    This class is responsible for extracting the intents from the query.
    It uses the LLM to extract the intents and then validates the response.
    It uses the prompts to expand the query and extract the intents.
    It uses the prompts to validate the response.
    """

    def __init__(
        self,
        project: str = PROJECT_ID,
        location: str = LOCATION,
        model: str = MODEL_VERSION,
        logger=None,
        tracer=None,
        temporal_vocab: TemporalVocab | None = None,
        temporal_resolver: TemporalResolver | None = None,
        record_type_vocab: ConceptVocab | None = None,
        temporal_index: TemporalIndex | None = None,
        cadence_memory: CadenceMemory | None = None,
        temporal_settings: dict[str, Any] | None = None,
    ):
        self._project = project
        self._default_location = location
        self._credentials, self._project_id = google.auth.default()
        self._clients: dict[str, genai.Client] = {}
        self.client = self._get_client(location)
        self.model_name = model
        self.lg = logger
        self.tracer = tracer

        # v2 retrieval signals: FOLDED temporal matching happens inline in
        # build_context when a vocab is set; otherwise the resolver path runs.
        self._temporal_vocab = temporal_vocab
        self._temporal_resolver = temporal_resolver or NullTemporalResolver()

        # v2 retrieval signals: when a record-type vocab is set, its labels are
        # injected into the contextual environment prompt so the model matches
        # against real vocabulary entries instead of inventing free-text labels.
        self._record_type_vocab = record_type_vocab

        # Canonical temporal mode (/v3): the vocabulary index resolves the
        # model's structured windows; the cadence memory supplies examples the
        # service learned from its own traffic. All limits come from settings.
        self._temporal_index = temporal_index
        self._cadence_memory = cadence_memory
        self._temporal_settings: dict[str, Any] = {
            "max_codes": 2,
            "min_similarity": 0.3,
            "menu_max_values": 12,
            "menu_units": list(CANONICAL_UNITS),
            "shortlist_top_n": 8,
            "shortlist_min_score": 0.3,
            "memory_examples_per_concept": 3,
            "memory_max_concepts": 10,
            "memory_min_similarity": 0.6,
            **(temporal_settings or {}),
        }
        self._canonical_resolver = (
            CanonicalTemporalResolver(
                temporal_index,
                max_codes=self._temporal_settings["max_codes"],
                min_similarity=self._temporal_settings["min_similarity"],
            )
            if temporal_index is not None
            else None
        )

        # Base log structure
        self.base_log = {
            "service_name": SERVICE_NAME,
            "version": SERVICE_VERSION,
            "environment": ENV,
        }

        if self.lg:
            self.lg.log_struct(
                message="Intent Extraction client initialized successfully",
                structured_data={
                    **self.base_log,
                    "project": project,
                    "location": location,
                    "model": model,
                },
                severity="INFO",
            )

    def _get_client(self, location: str | None = None) -> genai.Client:
        """Return a cached Vertex AI client for the given location, building
        one lazily on first use. Only VALID_LOCATIONS values are expected
        per request, so the cache stays small."""
        loc = location or self._default_location
        client = self._clients.get(loc)
        if client is None:
            client = genai.Client(
                vertexai=True,
                project=self._project_id or self._project,
                credentials=self._credentials,
                location=loc,
            )
            self._clients[loc] = client
        return client

    # -----------------------
    # Model Invocation
    # -----------------------
    def _call_model(
        self,
        prompt: str,
        model_name: str | None = None,
        generation_config: dict[str, Any] | None = None,
        location: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 65536,
        timeout: int = LLM_TIMEOUT_SECONDS,
        step_name: str = "llm_call",
        response_schema: type | None = None,
    ) -> tuple[str, dict[str, int]]:
        """
        Calls the model to extract the intents from the query.
        Args:
            prompt: The prompt to use for the intent extraction.
            model_name: Optional model name override (defaults to self.model_name).
            generation_config: Optional generation config from request.
            location: Optional GCP region override (defaults to the pipeline's default location).
            temperature: The temperature to use for the intent extraction.
            max_tokens: The max tokens to use for the intent extraction.
            timeout: Timeout in seconds for the LLM call (default: 180).
            step_name: Name of the step (e.g., "query_expansion", "intent_extraction") for logging.
            response_schema: Optional Pydantic model for Gemini structured JSON output (v2).
        Returns:
            Tuple of (response_text, usage_metadata)
        Raises:
            LLMTimeoutError: If the LLM call times out.
            LLMRateLimitError: If rate limit is exceeded.
            LLMServiceError: If LLM service is unavailable.
            LLMInvalidRequestError: If request is invalid.
            LLMError: For other LLM-related errors.
        """
        # Use provided model or fall back to instance default
        model = model_name or self.model_name
        client = self._get_client(location)

        try:
            # Span: llm_call
            if self.tracer:
                with self.tracer.start_as_current_span("llm_call") as span:
                    span.set_attribute("model", model)
                    span.set_attribute("prompt_length", len(prompt))
                    span.set_attribute("step", step_name)

                    # Build base config with defaults
                    base_cfg = {
                        "temperature": temperature,
                        "max_output_tokens": max_tokens,
                        "top_p": 1.0,
                        "top_k": 1,
                        "seed": LLM_SEED,
                    }

                    # Merge user config with base config
                    cfg_kwargs = build_generation_config(
                        generation_config,
                        base_cfg,
                        max_output_limit=max_tokens,
                    )
                    if response_schema is not None:
                        cfg_kwargs["response_mime_type"] = "application/json"
                        cfg_kwargs["response_schema"] = response_schema

                    config = GenerateContentConfig(**cfg_kwargs)

                    if not isinstance(prompt, list):
                        prompt = [prompt]

                    response = client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=config,
                    )

                    # Extract and log usage metadata for audit
                    usage_metadata = extract_usage_metadata(response)

                    # Update span attributes
                    response_text_for_span = None
                    if hasattr(response, "text") and response.text:
                        response_text_for_span = response.text.strip()
                    elif hasattr(response, "candidates") and response.candidates:
                        candidate = response.candidates[0]
                        if (
                            hasattr(candidate, "content")
                            and candidate.content
                            and hasattr(candidate.content, "parts")
                            and candidate.content.parts
                        ):
                            part = candidate.content.parts[0]
                            if hasattr(part, "text") and part.text:
                                response_text_for_span = part.text.strip()

                    span.set_attribute(
                        "response_length",
                        len(response_text_for_span) if response_text_for_span else 0,
                    )
                    span.set_attribute("success", True)
                    span.set_attribute(
                        "usage_metadata.prompt_token_count",
                        usage_metadata.get("prompt_token_count", 0),
                    )
                    span.set_attribute(
                        "usage_metadata.candidates_token_count",
                        usage_metadata.get("candidates_token_count", 0),
                    )
                    span.set_attribute(
                        "usage_metadata.total_token_count",
                        usage_metadata.get("total_token_count", 0),
                    )
                    span.set_attribute(
                        "usage_metadata.thinking_token_count",
                        usage_metadata.get("thinking_token_count", 0),
                    )
            else:
                # No tracing - process normally
                # Build base config with defaults
                base_cfg = {
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                    "top_p": 1.0,
                    "top_k": 1,
                    "seed": LLM_SEED,
                }

                # Merge user config with base config
                cfg_kwargs = build_generation_config(
                    generation_config,
                    base_cfg,
                    max_output_limit=max_tokens,
                )
                if response_schema is not None:
                    cfg_kwargs["response_mime_type"] = "application/json"
                    cfg_kwargs["response_schema"] = response_schema

                config = GenerateContentConfig(**cfg_kwargs)

                if not isinstance(prompt, list):
                    prompt = [prompt]

                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=config,
                )

                # Extract and log usage metadata for audit
                usage_metadata = extract_usage_metadata(response)

            # Logging (outside span)
            if self.lg:
                self.lg.log_struct(
                    message=f"LLM call completed - {step_name}",
                    structured_data={
                        **self.base_log,
                        "model": model,
                        "step": step_name,
                        "usage_metadata": usage_metadata,
                    },
                    severity="INFO",
                )

            # Extract text from response
            response_text = None
            if hasattr(response, "text") and response.text:
                response_text = response.text.strip()
            elif hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if (
                    hasattr(candidate, "content")
                    and candidate.content
                    and hasattr(candidate.content, "parts")
                    and candidate.content.parts
                ):
                    part = candidate.content.parts[0]
                    if hasattr(part, "text") and part.text:
                        response_text = part.text.strip()

            if response_text:
                return response_text, usage_metadata

            # Fallback: try to convert response to string
            response_str = str(response)
            if response_str and response_str != "{}":
                return response_str.strip(), usage_metadata

            # If we can't find text, log and return empty
            if self.lg:
                self.lg.log_struct(
                    message="Could not extract text from llm response",
                    structured_data={
                        **self.base_log,
                        "response_type": str(type(response)),
                        "response_attrs": [
                            a for a in dir(response) if not a.startswith("_")
                        ],
                    },
                    severity="WARNING",
                )
            return "{}", usage_metadata

        except ValueError as e:
            # Handle invalid generation_config errors
            if self.lg:
                self.lg.log_struct(
                    message="Invalid generation config",
                    structured_data={
                        **self.base_log,
                        "error": str(e),
                        "error_type": "ValueError",
                    },
                    severity="ERROR",
                )
            empty_metadata = {
                "prompt_token_count": 0,
                "candidates_token_count": 0,
                "total_token_count": 0,
                "thinking_token_count": 0,
            }
            return "{}", empty_metadata

        except google_exceptions.DeadlineExceeded as e:
            if self.lg:
                self.lg.log_struct(
                    message="LLM call timed out",
                    structured_data={
                        **self.base_log,
                        "error": str(e),
                        "timeout_seconds": timeout,
                    },
                    severity="ERROR",
                )
            raise LLMTimeoutError(
                f"LLM call timed out after {timeout} seconds",
                details={"timeout_seconds": timeout, "original_error": str(e)},
            )

        except google_exceptions.ResourceExhausted as e:
            if self.lg:
                self.lg.log_struct(
                    message="LLM rate limit exceeded",
                    structured_data={
                        **self.base_log,
                        "error": str(e),
                    },
                    severity="ERROR",
                )
            raise LLMRateLimitError(
                "Rate limit exceeded, please retry later",
                details={"original_error": str(e)},
            )

        except google_exceptions.ServiceUnavailable as e:
            if self.lg:
                self.lg.log_struct(
                    message="LLM service unavailable",
                    structured_data={
                        **self.base_log,
                        "error": str(e),
                    },
                    severity="ERROR",
                )
            raise LLMServiceError(
                "LLM service is temporarily unavailable",
                details={"original_error": str(e)},
            )

        except (
            google_exceptions.FailedPrecondition,
            google_exceptions.NotFound,
            google_exceptions.InvalidArgument,
        ) as e:
            if _is_location_model_error(e):
                if self.lg:
                    self.lg.log_struct(
                        message="LLM call failed - unsupported model/location",
                        structured_data={
                            **self.base_log,
                            "model": model,
                            "location": location or self._default_location,
                            "error": str(e),
                        },
                        severity="ERROR",
                    )
                raise LLMLocationError(
                    f"Requested model/location is not available: {e!s}",
                    details={
                        "original_error": str(e),
                        "model": model,
                        "location": location or self._default_location,
                    },
                )

            if self.lg:
                self.lg.log_struct(
                    message="Invalid LLM request",
                    structured_data={
                        **self.base_log,
                        "error": str(e),
                    },
                    severity="ERROR",
                )
            raise LLMInvalidRequestError(
                f"Invalid request to LLM: {e!s}",
                details={"original_error": str(e)},
            )

        except Exception as e:
            if _is_location_model_error(e):
                if self.lg:
                    self.lg.log_struct(
                        message="LLM call failed - unsupported model/location",
                        structured_data={
                            **self.base_log,
                            "model": model,
                            "location": location or self._default_location,
                            "error": str(e),
                            "error_type": type(e).__name__,
                        },
                        severity="ERROR",
                    )
                raise LLMLocationError(
                    f"Requested model/location is not available: {e!s}",
                    details={
                        "error_type": type(e).__name__,
                        "original_error": str(e),
                        "model": model,
                        "location": location or self._default_location,
                    },
                )

            if self.lg:
                self.lg.log_struct(
                    message="LLM call failed",
                    structured_data={
                        **self.base_log,
                        "error": str(e),
                        "error_type": str(type(e).__name__),
                    },
                    severity="ERROR",
                )
            raise LLMError(
                f"LLM call failed: {e!s}",
                details={"error_type": type(e).__name__, "original_error": str(e)},
            )

    async def _call_model_async(
        self,
        prompt: str,
        model_name: str | None = None,
        generation_config: dict[str, Any] | None = None,
        location: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 65536,
        timeout: int = LLM_TIMEOUT_SECONDS,
        step_name: str = "llm_call",
        response_schema: type | None = None,
    ) -> tuple[str, dict[str, int]]:
        """
        Calls the model to extract the intents from the query.
        Args:
            prompt: The prompt to use for the intent extraction.
            model_name: Optional model name override (defaults to self.model_name).
            generation_config: Optional generation config from request.
            location: Optional GCP region override (defaults to the pipeline's default location).
            temperature: The temperature to use for the intent extraction.
            max_tokens: The max tokens to use for the intent extraction.
            timeout: Timeout in seconds for the LLM call (default: 180).
            step_name: Name of the step (e.g., "query_expansion", "intent_extraction") for logging.
            response_schema: Optional Pydantic model for Gemini structured JSON output (v2).
        Returns:
            Tuple of (response_text, usage_metadata)
        Raises:
            LLMTimeoutError: If the LLM call times out.
            LLMRateLimitError: If rate limit is exceeded.
            LLMServiceError: If LLM service is unavailable.
            LLMInvalidRequestError: If request is invalid.
            LLMError: For other LLM-related errors.
        """
        # Use provided model or fall back to instance default
        model = model_name or self.model_name
        client = self._get_client(location)

        try:
            # Span: llm_call
            if self.tracer:
                with self.tracer.start_as_current_span("llm_call") as span:
                    span.set_attribute("model", model)
                    span.set_attribute("prompt_length", len(prompt))
                    span.set_attribute("step", step_name)

                    # Build base config with defaults
                    base_cfg = {
                        "temperature": temperature,
                        "max_output_tokens": max_tokens,
                        "top_p": 1.0,
                        "top_k": 1,
                        "seed": LLM_SEED,
                    }

                    # Merge user config with base config
                    cfg_kwargs = build_generation_config(
                        generation_config,
                        base_cfg,
                        max_output_limit=max_tokens,
                    )
                    if response_schema is not None:
                        cfg_kwargs["response_mime_type"] = "application/json"
                        cfg_kwargs["response_schema"] = response_schema

                    config = GenerateContentConfig(**cfg_kwargs)

                    if not isinstance(prompt, list):
                        prompt = [prompt]

                    response = await client.aio.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=config,
                    )

                    # Extract and log usage metadata for audit
                    usage_metadata = extract_usage_metadata(response)

                    # Update span attributes
                    response_text_for_span = None
                    if hasattr(response, "text") and response.text:
                        response_text_for_span = response.text.strip()
                    elif hasattr(response, "candidates") and response.candidates:
                        candidate = response.candidates[0]
                        if (
                            hasattr(candidate, "content")
                            and candidate.content
                            and hasattr(candidate.content, "parts")
                            and candidate.content.parts
                        ):
                            part = candidate.content.parts[0]
                            if hasattr(part, "text") and part.text:
                                response_text_for_span = part.text.strip()

                    span.set_attribute(
                        "response_length",
                        len(response_text_for_span) if response_text_for_span else 0,
                    )
                    span.set_attribute("success", True)
                    span.set_attribute(
                        "usage_metadata.prompt_token_count",
                        usage_metadata.get("prompt_token_count", 0),
                    )
                    span.set_attribute(
                        "usage_metadata.candidates_token_count",
                        usage_metadata.get("candidates_token_count", 0),
                    )
                    span.set_attribute(
                        "usage_metadata.total_token_count",
                        usage_metadata.get("total_token_count", 0),
                    )
                    span.set_attribute(
                        "usage_metadata.thinking_token_count",
                        usage_metadata.get("thinking_token_count", 0),
                    )
            else:
                # No tracing - process normally
                # Build base config with defaults
                base_cfg = {
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                    "top_p": 1.0,
                    "top_k": 1,
                    "seed": LLM_SEED,
                }

                # Merge user config with base config
                cfg_kwargs = build_generation_config(
                    generation_config,
                    base_cfg,
                    max_output_limit=max_tokens,
                )
                if response_schema is not None:
                    cfg_kwargs["response_mime_type"] = "application/json"
                    cfg_kwargs["response_schema"] = response_schema

                config = GenerateContentConfig(**cfg_kwargs)

                if not isinstance(prompt, list):
                    prompt = [prompt]

                response = await client.aio.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=config,
                )

                # Extract and log usage metadata for audit
                usage_metadata = extract_usage_metadata(response)

            # Logging (outside span)
            if self.lg:
                self.lg.log_struct(
                    message=f"LLM call completed - {step_name}",
                    structured_data={
                        **self.base_log,
                        "model": model,
                        "step": step_name,
                        "usage_metadata": usage_metadata,
                    },
                    severity="INFO",
                )

            # Extract text from response
            response_text = None
            if hasattr(response, "text") and response.text:
                response_text = response.text.strip()
            elif hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if (
                    hasattr(candidate, "content")
                    and candidate.content
                    and hasattr(candidate.content, "parts")
                    and candidate.content.parts
                ):
                    part = candidate.content.parts[0]
                    if hasattr(part, "text") and part.text:
                        response_text = part.text.strip()

            if response_text:
                return response_text, usage_metadata

            # Fallback: try to convert response to string
            response_str = str(response)
            if response_str and response_str != "{}":
                return response_str.strip(), usage_metadata

            # If we can't find text, log and return empty
            if self.lg:
                self.lg.log_struct(
                    message="Could not extract text from llm response",
                    structured_data={
                        **self.base_log,
                        "response_type": str(type(response)),
                        "response_attrs": [
                            a for a in dir(response) if not a.startswith("_")
                        ],
                    },
                    severity="WARNING",
                )
            return "{}", usage_metadata

        except ValueError as e:
            # Handle invalid generation_config errors
            if self.lg:
                self.lg.log_struct(
                    message="Invalid generation config",
                    structured_data={
                        **self.base_log,
                        "error": str(e),
                        "error_type": "ValueError",
                    },
                    severity="ERROR",
                )
            empty_metadata = {
                "prompt_token_count": 0,
                "candidates_token_count": 0,
                "total_token_count": 0,
                "thinking_token_count": 0,
            }
            return "{}", empty_metadata

        except google_exceptions.DeadlineExceeded as e:
            if self.lg:
                self.lg.log_struct(
                    message="LLM call timed out",
                    structured_data={
                        **self.base_log,
                        "error": str(e),
                        "timeout_seconds": timeout,
                    },
                    severity="ERROR",
                )
            raise LLMTimeoutError(
                f"LLM call timed out after {timeout} seconds",
                details={"timeout_seconds": timeout, "original_error": str(e)},
            )

        except google_exceptions.ResourceExhausted as e:
            if self.lg:
                self.lg.log_struct(
                    message="LLM rate limit exceeded",
                    structured_data={
                        **self.base_log,
                        "error": str(e),
                    },
                    severity="ERROR",
                )
            raise LLMRateLimitError(
                "Rate limit exceeded, please retry later",
                details={"original_error": str(e)},
            )

        except google_exceptions.ServiceUnavailable as e:
            if self.lg:
                self.lg.log_struct(
                    message="LLM service unavailable",
                    structured_data={
                        **self.base_log,
                        "error": str(e),
                    },
                    severity="ERROR",
                )
            raise LLMServiceError(
                "LLM service is temporarily unavailable",
                details={"original_error": str(e)},
            )

        except (
            google_exceptions.FailedPrecondition,
            google_exceptions.NotFound,
            google_exceptions.InvalidArgument,
        ) as e:
            if _is_location_model_error(e):
                if self.lg:
                    self.lg.log_struct(
                        message="LLM call failed - unsupported model/location",
                        structured_data={
                            **self.base_log,
                            "model": model,
                            "location": location or self._default_location,
                            "error": str(e),
                        },
                        severity="ERROR",
                    )
                raise LLMLocationError(
                    f"Requested model/location is not available: {e!s}",
                    details={
                        "original_error": str(e),
                        "model": model,
                        "location": location or self._default_location,
                    },
                )

            if self.lg:
                self.lg.log_struct(
                    message="Invalid LLM request",
                    structured_data={
                        **self.base_log,
                        "error": str(e),
                    },
                    severity="ERROR",
                )
            raise LLMInvalidRequestError(
                f"Invalid request to LLM: {e!s}",
                details={"original_error": str(e)},
            )

        except Exception as e:
            if _is_location_model_error(e):
                if self.lg:
                    self.lg.log_struct(
                        message="LLM call failed - unsupported model/location",
                        structured_data={
                            **self.base_log,
                            "model": model,
                            "location": location or self._default_location,
                            "error": str(e),
                            "error_type": type(e).__name__,
                        },
                        severity="ERROR",
                    )
                raise LLMLocationError(
                    f"Requested model/location is not available: {e!s}",
                    details={
                        "error_type": type(e).__name__,
                        "original_error": str(e),
                        "model": model,
                        "location": location or self._default_location,
                    },
                )

            if self.lg:
                self.lg.log_struct(
                    message="LLM call failed",
                    structured_data={
                        **self.base_log,
                        "error": str(e),
                        "error_type": str(type(e).__name__),
                    },
                    severity="ERROR",
                )
            raise LLMError(
                f"LLM call failed: {e!s}",
                details={"error_type": type(e).__name__, "original_error": str(e)},
            )

    # -----------------------
    # Safe JSON Parsing
    # -----------------------
    def _safe_json(self, text: str) -> dict[str, Any]:
        """Parses the text into a dictionary.
        Args:
            text: The text to parse.
        Returns:
            The dictionary from the text, or error dict if parsing fails.
        """
        # Log full response length before parsing
        original_length = len(text) if text else 0
        if self.lg:
            self.lg.log_struct(
                message="Parsing LLM response",
                structured_data={
                    **self.base_log,
                    "response_length": original_length,
                },
                severity="DEBUG",
            )

        # Clean up markdown code blocks
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*", "", text)
        text = text.strip()

        # Detect potential truncation (object "}" or array "]" are both complete)
        is_truncated = False
        if text and not text.rstrip().endswith(("}", "]")):
            is_truncated = True
            if self.lg:
                self.lg.log_struct(
                    message="Response appears truncated - does not end with '}' or ']'",
                    structured_data={
                        **self.base_log,
                        "response_length": len(text),
                        "last_50_chars": text[-50:] if len(text) > 50 else text,
                    },
                    severity="WARNING",
                )

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            # Log actual JSONDecodeError message
            if self.lg:
                self.lg.log_struct(
                    message="JSON decode error",
                    structured_data={
                        **self.base_log,
                        "error_message": str(e),
                        "error_position": e.pos,
                        "error_line": e.lineno,
                        "error_column": e.colno,
                        "response_length": len(text),
                        "is_truncated": is_truncated,
                    },
                    severity="ERROR",
                )

            # Try regex fallback
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError as e2:
                    if self.lg:
                        self.lg.log_struct(
                            message="Regex fallback JSON parse also failed",
                            structured_data={
                                **self.base_log,
                                "error_message": str(e2),
                                "matched_length": len(match.group(0)),
                            },
                            severity="ERROR",
                        )

            # Return meaningful error dict instead of empty {}
            if self.lg:
                self.lg.log_struct(
                    message="Failed to parse JSON from model response",
                    structured_data={
                        **self.base_log,
                        "response_snippet": text[:300],
                        "response_length": len(text),
                        "is_truncated": is_truncated,
                    },
                    severity="ERROR",
                )

            # Return error dict with details
            return {
                "_parsing_error": True,
                "_error_message": f"JSON parsing failed: {e!s}",
                "_is_truncated": is_truncated,
                "_response_length": len(text),
                "_response_snippet": text[:500] if text else "",
            }

    def _validate_json_structure(
        self, data: dict[str, Any], required_keys: list[str]
    ) -> bool:
        return all(key in data for key in required_keys)

    def _validate_llm_response(self, data: dict[str, Any]) -> IntentExtractionResponse:
        """
        Validate LLM response against Pydantic model.
        Raises ValidationError if validation fails.

        Args:
            data: Parsed JSON data from LLM response

        Returns:
            IntentExtractionResponse: Validated response model

        Raises:
            ValidationError: If response doesn't match expected format
            ValueError: If JSON parsing failed (indicated by _parsing_error key)
        """
        # Handle parsing error dict from _safe_json()
        if data.get("_parsing_error"):
            error_msg = data.get("_error_message", "Unknown parsing error")
            is_truncated = data.get("_is_truncated", False)
            response_length = data.get("_response_length", 0)

            if self.lg:
                self.lg.log_struct(
                    message="JSON parsing error - cannot validate response",
                    structured_data={
                        **self.base_log,
                        "error_message": error_msg,
                        "is_truncated": is_truncated,
                        "response_length": response_length,
                    },
                    severity="ERROR",
                )

            # Raise ValueError with meaningful message
            if is_truncated:
                raise ValueError(
                    f"LLM response was truncated (length: {response_length}). "
                    "The response exceeded token limits and JSON is incomplete."
                )
            else:
                raise ValueError(f"Failed to parse LLM response: {error_msg}")

        # Handle empty dict case
        if not data or data == {}:
            if self.lg:
                self.lg.log_struct(
                    message="Empty or invalid JSON response from LLM",
                    structured_data={**self.base_log},
                    severity="ERROR",
                )
            # Let Pydantic validate empty dict to get proper error
            data = {}

        try:
            return IntentExtractionResponse(**data)
        except ValidationError as e:
            # Log detailed validation errors
            if self.lg:
                self.lg.log_struct(
                    message="LLM response validation failed",
                    structured_data={
                        **self.base_log,
                        "validation_errors": [err["msg"] for err in e.errors()],
                        "error_count": len(e.errors()),
                        "error_details": e.errors(),
                    },
                    severity="ERROR",
                )
            raise

    def _validate_llm_response_v2(self, data: Any) -> IntentExtractionResponseV2:
        """Validate v2 intent LLM response (no clinical fields)."""
        if isinstance(data, list):
            data = {"intents": data}

        if isinstance(data, dict) and data.get("_parsing_error"):
            error_msg = data.get("_error_message", "Unknown parsing error")
            is_truncated = data.get("_is_truncated", False)
            response_length = data.get("_response_length", 0)
            if self.lg:
                self.lg.log_struct(
                    message="JSON parsing error - cannot validate v2 response",
                    structured_data={
                        **self.base_log,
                        "error_message": error_msg,
                        "is_truncated": is_truncated,
                        "response_length": response_length,
                    },
                    severity="ERROR",
                )
            if is_truncated:
                raise ValueError(
                    f"LLM response was truncated (length: {response_length}). "
                    "The response exceeded token limits and JSON is incomplete."
                )
            raise ValueError(f"Failed to parse LLM response: {error_msg}")

        if not data or data == {}:
            data = {}

        try:
            return IntentExtractionResponseV2(**data)
        except ValidationError as e:
            if self.lg:
                self.lg.log_struct(
                    message="LLM v2 response validation failed",
                    structured_data={
                        **self.base_log,
                        "validation_errors": [err["msg"] for err in e.errors()],
                        "error_count": len(e.errors()),
                        "error_details": e.errors(),
                    },
                    severity="ERROR",
                )
            raise

    # -----------------------
    # STEP 1: Query Expansion
    # -----------------------
    def expand_query(
        self,
        query: str,
        model_name: str | None = None,
        generation_config: dict[str, Any] | None = None,
        location: str | None = None,
    ) -> dict[str, Any]:
        """Expands the query using the LLM.
        Args:
            query: The query to expand.
            model_name: Optional model name override.
            generation_config: Optional generation config from request.
            location: Optional GCP region override.
        Returns:
            Dict with expanded_query, abbreviations_expanded, and usage_metadata.
        Raises:
            LLMError: If the LLM call fails (timeout, rate limit, etc.)
        """
        # Span: query_expansion
        if self.tracer:
            with self.tracer.start_as_current_span("query_expansion") as span:
                span.set_attribute("query_length", len(query))

                prompt = QUERY_EXPANSION_PROMPT.format(query=query)

                try:
                    raw_response, usage_metadata = self._call_model(
                        prompt,
                        model_name=model_name,
                        generation_config=generation_config,
                        location=location,
                        step_name="query_expansion",
                    )
                except LLMError as e:
                    # Log the LLM error and fall back to original query
                    span.set_attribute("success", False)
                    span.set_attribute("error", str(e))
                    if self.lg:
                        self.lg.log_struct(
                            message="Query expansion LLM call failed, falling back to original query",
                            structured_data={
                                **self.base_log,
                                "query_length": len(query),
                                "error_type": e.error_type,
                                "error_message": str(e),
                                "fallback": True,
                            },
                            severity="ERROR",
                        )
                    return {
                        "expanded_query": query,
                        "abbreviations_expanded": [],
                        "expansion_failed": True,
                        "expansion_error": str(e),
                        "expansion_error_type": e.error_type,
                        "usage_metadata": {
                            "prompt_token_count": 0,
                            "candidates_token_count": 0,
                            "total_token_count": 0,
                            "thinking_token_count": 0,
                        },
                    }

                data = self._safe_json(raw_response)

                # Handle parsing error from _safe_json()
                if data.get("_parsing_error"):
                    span.set_attribute("success", False)
                    span.set_attribute("parsing_error", True)
                    if self.lg:
                        self.lg.log_struct(
                            message="Query expansion JSON parsing failed, falling back to original query",
                            structured_data={
                                **self.base_log,
                                "query_length": len(query),
                                "error_message": data.get("_error_message"),
                                "is_truncated": data.get("_is_truncated"),
                                "fallback": True,
                            },
                            severity="WARNING",
                        )
                    return {
                        "expanded_query": query,
                        "abbreviations_expanded": [],
                        "expansion_failed": True,
                        "expansion_error": data.get("_error_message"),
                        "usage_metadata": usage_metadata,
                    }

                if not self._validate_json_structure(data, ["expanded_query"]):
                    if self.lg:
                        self.lg.log_struct(
                            message="Query expansion failed, falling back to original query",
                            structured_data={
                                **self.base_log,
                                "query": query,
                            },
                            severity="WARNING",
                        )
                    expanded_query = query
                    span.set_attribute("expanded_length", len(expanded_query))
                    span.set_attribute("success", False)
                    return {
                        "expanded_query": expanded_query,
                        "abbreviations_expanded": [],
                        "expansion_failed": True,
                        "expansion_error": "Response missing required fields",
                        "usage_metadata": usage_metadata,
                    }

                expanded_query = data.get("expanded_query", query)
                span.set_attribute("expanded_length", len(expanded_query))
                span.set_attribute("success", True)

                return {
                    "expanded_query": expanded_query,
                    "abbreviations_expanded": data.get("abbreviations_expanded", []),
                    "usage_metadata": usage_metadata,
                }
        else:
            # No tracing - process normally
            prompt = QUERY_EXPANSION_PROMPT.format(query=query)

            try:
                raw_response, usage_metadata = self._call_model(
                    prompt,
                    model_name=model_name,
                    generation_config=generation_config,
                    location=location,
                    step_name="query_expansion",
                )
            except LLMError as e:
                # Log the LLM error and fall back to original query
                if self.lg:
                    self.lg.log_struct(
                        message="Query expansion LLM call failed, falling back to original query",
                        structured_data={
                            **self.base_log,
                            "query_length": len(query),
                            "error_type": e.error_type,
                            "error_message": str(e),
                            "fallback": True,
                        },
                        severity="ERROR",
                    )
                return {
                    "expanded_query": query,
                    "abbreviations_expanded": [],
                    "expansion_failed": True,
                    "expansion_error": str(e),
                    "expansion_error_type": e.error_type,
                    "usage_metadata": {
                        "prompt_token_count": 0,
                        "candidates_token_count": 0,
                        "total_token_count": 0,
                        "thinking_token_count": 0,
                    },
                }

            data = self._safe_json(raw_response)

            # Handle parsing error from _safe_json()
            if data.get("_parsing_error"):
                if self.lg:
                    self.lg.log_struct(
                        message="Query expansion JSON parsing failed, falling back to original query",
                        structured_data={
                            **self.base_log,
                            "query_length": len(query),
                            "error_message": data.get("_error_message"),
                            "is_truncated": data.get("_is_truncated"),
                            "fallback": True,
                        },
                        severity="WARNING",
                    )
                return {
                    "expanded_query": query,
                    "abbreviations_expanded": [],
                    "expansion_failed": True,
                    "expansion_error": data.get("_error_message"),
                    "usage_metadata": usage_metadata,
                }

            if not self._validate_json_structure(data, ["expanded_query"]):
                if self.lg:
                    self.lg.log_struct(
                        message="Query expansion failed, falling back to original query",
                        structured_data={
                            **self.base_log,
                            "query": query,
                        },
                        severity="WARNING",
                    )
                return {
                    "expanded_query": query,
                    "abbreviations_expanded": [],
                    "expansion_failed": True,
                    "expansion_error": "Response missing required fields",
                    "usage_metadata": usage_metadata,
                }

            return {
                "expanded_query": data.get("expanded_query", query),
                "abbreviations_expanded": data.get("abbreviations_expanded", []),
                "usage_metadata": usage_metadata,
            }

    # -----------------------
    # STEP 2: Intent Extraction
    # -----------------------
    def extract_intents(
        self,
        original_query: str,
        expanded_query: str,
        model_name: str | None = None,
        generation_config: dict[str, Any] | None = None,
        location: str | None = None,
    ) -> dict[str, Any]:
        """Extracts the intents from the query.
        Args:
            original_query: The original query.
            expanded_query: The expanded query.
            model_name: Optional model name override.
            generation_config: Optional generation config from request.
            location: Optional GCP region override.
        Returns:
            The intents from the query.
        Raises:
            Exception: If the intent extraction fails.
        """
        # Span: intent_extraction
        if self.tracer:
            with self.tracer.start_as_current_span("intent_extraction") as span:
                prompt = INTENT_EXTRACTION_PROMPT.format(
                    original_query=original_query,
                    expanded_query=expanded_query,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )

                raw_response, usage_metadata = self._call_model(
                    prompt,
                    model_name=model_name,
                    generation_config=generation_config,
                    location=location,
                    max_tokens=65536,
                    step_name="intent_extraction",
                )
                data = self._safe_json(raw_response)

                # Handle parsing error early - skip non-clinical field handling
                if data.get("_parsing_error"):
                    # Will be handled in _validate_llm_response()
                    pass
                # For non-clinical responses, ensure all required fields are present
                elif not data.get("is_clinical", True):
                    if "original_query" not in data:
                        data["original_query"] = original_query
                    if "expanded_query" not in data:
                        data["expanded_query"] = expanded_query
                    if "total_intents_detected" not in data:
                        data["total_intents_detected"] = 0

                # Validate LLM response using Pydantic models
                try:
                    validated_response = self._validate_llm_response(data)
                    span.set_attribute("is_clinical", validated_response.is_clinical)
                    span.set_attribute("intent_count", len(validated_response.intents))
                    span.set_attribute("validation_success", True)

                    if self.lg:
                        self.lg.log_struct(
                            message="LLM response validation successful",
                            structured_data={
                                **self.base_log,
                                "intent_count": len(validated_response.intents),
                                "is_clinical": validated_response.is_clinical,
                            },
                            severity="INFO",
                        )
                except ValueError as e:
                    # JSON parsing failed - return meaningful error
                    span.set_attribute("validation_success", False)
                    span.set_attribute("parsing_error", True)
                    if self.lg:
                        self.lg.log_struct(
                            message="LLM response parsing failed - returning error",
                            structured_data={
                                **self.base_log,
                                "error_message": str(e),
                            },
                            severity="ERROR",
                        )
                    return {
                        "intents": [],
                        "total_intents_detected": 0,
                        "is_clinical": True,
                        "error": "LLM response parsing failed",
                        "error_details": str(e),
                        "original_query": original_query,
                        "expanded_query": expanded_query,
                        "usage_metadata": usage_metadata,
                    }
                except ValidationError as e:
                    # Validation failed - return error
                    span.set_attribute("validation_success", False)
                    span.set_attribute("error_count", len(e.errors()))
                    if self.lg:
                        self.lg.log_struct(
                            message="LLM response validation failed - returning error",
                            structured_data={
                                **self.base_log,
                                "validation_error_count": len(e.errors()),
                            },
                            severity="ERROR",
                        )
                    return {
                        "intents": [],
                        "total_intents_detected": 0,
                        "is_clinical": True,
                        "error": "LLM response validation failed",
                        "validation_errors": [err["msg"] for err in e.errors()],
                        "original_query": original_query,
                        "expanded_query": expanded_query,
                        "usage_metadata": usage_metadata,
                    }

                # Handle non-clinical responses
                if not validated_response.is_clinical:
                    if self.lg:
                        self.lg.log_struct(
                            message="Non-clinical query rejected",
                            structured_data={
                                **self.base_log,
                                "reason": validated_response.reason,
                            },
                            severity="INFO",
                        )
                    return {
                        "intents": [],
                        "total_intents_detected": 0,
                        "is_clinical": False,
                        "rejected_reason": validated_response.reason
                        or "Query is not clinical",
                        "original_query": validated_response.original_query,
                        "expanded_query": validated_response.expanded_query,
                        "usage_metadata": usage_metadata,
                    }

                # Convert validated Pydantic models back to dict for return
                validated_intents = []
                for intent in validated_response.intents:
                    validated_intents.append(
                        {
                            "intent_title": intent.intent_title,
                            "description": intent.description,
                            "nature": intent.nature,
                            "sub_natures": [
                                {
                                    "category_path": sub_nature.category_path,
                                    "atomic_concepts": sub_nature.atomic_concepts,
                                }
                                for sub_nature in intent.sub_natures
                            ],
                            "final_queries": intent.final_queries,
                        }
                    )

                if self.lg:
                    self.lg.log_struct(
                        message="Intents extracted successfully",
                        structured_data={
                            **self.base_log,
                            "intent_count": len(validated_intents),
                        },
                        severity="INFO",
                    )

                return {
                    "intents": validated_intents,
                    "total_intents_detected": validated_response.total_intents_detected,
                    "is_clinical": True,
                    "original_query": validated_response.original_query,
                    "expanded_query": validated_response.expanded_query,
                    "usage_metadata": usage_metadata,
                }
        else:
            # No tracing - process normally
            prompt = INTENT_EXTRACTION_PROMPT.format(
                original_query=original_query,
                expanded_query=expanded_query,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

            raw_response, usage_metadata = self._call_model(
                prompt,
                model_name=model_name,
                generation_config=generation_config,
                location=location,
                max_tokens=65536,
                step_name="intent_extraction",
            )
            data = self._safe_json(raw_response)

            # Handle parsing error early - skip non-clinical field handling
            if data.get("_parsing_error"):
                # Will be handled in _validate_llm_response()
                pass
            # For non-clinical responses, ensure all required fields are present
            elif not data.get("is_clinical", True):
                if "original_query" not in data:
                    data["original_query"] = original_query
                if "expanded_query" not in data:
                    data["expanded_query"] = expanded_query
                if "total_intents_detected" not in data:
                    data["total_intents_detected"] = 0

            # Validate LLM response using Pydantic models
            try:
                validated_response = self._validate_llm_response(data)
                if self.lg:
                    self.lg.log_struct(
                        message="LLM response validation successful",
                        structured_data={
                            **self.base_log,
                            "intent_count": len(validated_response.intents),
                            "is_clinical": validated_response.is_clinical,
                        },
                        severity="INFO",
                    )
            except ValueError as e:
                # JSON parsing failed - return meaningful error
                if self.lg:
                    self.lg.log_struct(
                        message="LLM response parsing failed - returning error",
                        structured_data={
                            **self.base_log,
                            "error_message": str(e),
                        },
                        severity="ERROR",
                    )
                return {
                    "intents": [],
                    "total_intents_detected": 0,
                    "is_clinical": True,
                    "error": "LLM response parsing failed",
                    "error_details": str(e),
                    "original_query": original_query,
                    "expanded_query": expanded_query,
                    "usage_metadata": usage_metadata,
                }
            except ValidationError as e:
                # Validation failed - return error
                if self.lg:
                    self.lg.log_struct(
                        message="LLM response validation failed - returning error",
                        structured_data={
                            **self.base_log,
                            "validation_error_count": len(e.errors()),
                        },
                        severity="ERROR",
                    )
                return {
                    "intents": [],
                    "total_intents_detected": 0,
                    "is_clinical": True,
                    "error": "LLM response validation failed",
                    "validation_errors": [err["msg"] for err in e.errors()],
                    "original_query": original_query,
                    "expanded_query": expanded_query,
                    "usage_metadata": usage_metadata,
                }

            # Handle non-clinical responses
            if not validated_response.is_clinical:
                if self.lg:
                    self.lg.log_struct(
                        message="Non-clinical query rejected",
                        structured_data={
                            **self.base_log,
                            "reason": validated_response.reason,
                        },
                        severity="INFO",
                    )
                return {
                    "intents": [],
                    "total_intents_detected": 0,
                    "is_clinical": False,
                    "rejected_reason": validated_response.reason
                    or "Query is not clinical",
                    "original_query": validated_response.original_query,
                    "expanded_query": validated_response.expanded_query,
                    "usage_metadata": usage_metadata,
                }

            # Convert validated Pydantic models back to dict for return
            validated_intents = []
            for intent in validated_response.intents:
                validated_intents.append(
                    {
                        "intent_title": intent.intent_title,
                        "description": intent.description,
                        "nature": intent.nature,
                        "sub_natures": [
                            {
                                "category_path": sub_nature.category_path,
                                "atomic_concepts": sub_nature.atomic_concepts,
                            }
                            for sub_nature in intent.sub_natures
                        ],
                        "final_queries": intent.final_queries,
                    }
                )

            if self.lg:
                self.lg.log_struct(
                    message="Intents extracted successfully",
                    structured_data={
                        **self.base_log,
                        "intent_count": len(validated_intents),
                    },
                    severity="INFO",
                )

            return {
                "intents": validated_intents,
                "total_intents_detected": validated_response.total_intents_detected,
                "is_clinical": True,
                "original_query": validated_response.original_query,
                "expanded_query": validated_response.expanded_query,
                "usage_metadata": usage_metadata,
            }

    # -----------------------
    # STEP 1 (v2): Query Expansion
    # -----------------------
    def expand_query_v2(
        self,
        query: str,
        model_name: str | None = None,
        generation_config: dict[str, Any] | None = None,
        location: str | None = None,
    ) -> dict[str, Any]:
        """Expands the query using the v2 prompt. Mirrors expand_query() but
        uses QUERY_EXPANSION_PROMPT_V2 (v1 method left untouched)."""
        if self.tracer:
            with self.tracer.start_as_current_span("query_expansion_v2") as span:
                span.set_attribute("query_length", len(query))

                prompt = QUERY_EXPANSION_PROMPT_V2.format(query=query)

                try:
                    raw_response, usage_metadata = self._call_model(
                        prompt,
                        model_name=model_name,
                        generation_config=generation_config,
                        location=location,
                        step_name="query_expansion_v2",
                        response_schema=QueryExpansionOutput,
                    )
                except LLMError as e:
                    span.set_attribute("success", False)
                    span.set_attribute("error", str(e))
                    if self.lg:
                        self.lg.log_struct(
                            message="Query expansion (v2) LLM call failed, falling back to original query",
                            structured_data={
                                **self.base_log,
                                "query_length": len(query),
                                "error_type": e.error_type,
                                "error_message": str(e),
                                "fallback": True,
                            },
                            severity="ERROR",
                        )
                    return {
                        "expanded_query": query,
                        "abbreviations_expanded": [],
                        "expansion_failed": True,
                        "expansion_error": str(e),
                        "expansion_error_type": e.error_type,
                        "usage_metadata": {
                            "prompt_token_count": 0,
                            "candidates_token_count": 0,
                            "total_token_count": 0,
                            "thinking_token_count": 0,
                        },
                    }

                data = self._safe_json(raw_response)

                if data.get("_parsing_error"):
                    span.set_attribute("success", False)
                    span.set_attribute("parsing_error", True)
                    if self.lg:
                        self.lg.log_struct(
                            message="Query expansion (v2) JSON parsing failed, falling back to original query",
                            structured_data={
                                **self.base_log,
                                "query_length": len(query),
                                "error_message": data.get("_error_message"),
                                "is_truncated": data.get("_is_truncated"),
                                "fallback": True,
                            },
                            severity="WARNING",
                        )
                    return {
                        "expanded_query": query,
                        "abbreviations_expanded": [],
                        "expansion_failed": True,
                        "expansion_error": data.get("_error_message"),
                        "usage_metadata": usage_metadata,
                    }

                if not self._validate_json_structure(data, ["expanded_query"]):
                    if self.lg:
                        self.lg.log_struct(
                            message="Query expansion (v2) failed, falling back to original query",
                            structured_data={
                                **self.base_log,
                                "query": query,
                            },
                            severity="WARNING",
                        )
                    expanded_query = query
                    span.set_attribute("expanded_length", len(expanded_query))
                    span.set_attribute("success", False)
                    return {
                        "expanded_query": expanded_query,
                        "abbreviations_expanded": [],
                        "expansion_failed": True,
                        "expansion_error": "Response missing required fields",
                        "usage_metadata": usage_metadata,
                    }

                expanded_query = data.get("expanded_query", query)
                span.set_attribute("expanded_length", len(expanded_query))
                span.set_attribute("success", True)

                return {
                    "expanded_query": expanded_query,
                    "abbreviations_expanded": data.get("abbreviations_expanded", []),
                    "usage_metadata": usage_metadata,
                }
        else:
            prompt = QUERY_EXPANSION_PROMPT_V2.format(query=query)

            try:
                raw_response, usage_metadata = self._call_model(
                    prompt,
                    model_name=model_name,
                    generation_config=generation_config,
                    location=location,
                    step_name="query_expansion_v2",
                    response_schema=QueryExpansionOutput,
                )
            except LLMError as e:
                if self.lg:
                    self.lg.log_struct(
                        message="Query expansion (v2) LLM call failed, falling back to original query",
                        structured_data={
                            **self.base_log,
                            "query_length": len(query),
                            "error_type": e.error_type,
                            "error_message": str(e),
                            "fallback": True,
                        },
                        severity="ERROR",
                    )
                return {
                    "expanded_query": query,
                    "abbreviations_expanded": [],
                    "expansion_failed": True,
                    "expansion_error": str(e),
                    "expansion_error_type": e.error_type,
                    "usage_metadata": {
                        "prompt_token_count": 0,
                        "candidates_token_count": 0,
                        "total_token_count": 0,
                        "thinking_token_count": 0,
                    },
                }

            data = self._safe_json(raw_response)

            if data.get("_parsing_error"):
                if self.lg:
                    self.lg.log_struct(
                        message="Query expansion (v2) JSON parsing failed, falling back to original query",
                        structured_data={
                            **self.base_log,
                            "query_length": len(query),
                            "error_message": data.get("_error_message"),
                            "is_truncated": data.get("_is_truncated"),
                            "fallback": True,
                        },
                        severity="WARNING",
                    )
                return {
                    "expanded_query": query,
                    "abbreviations_expanded": [],
                    "expansion_failed": True,
                    "expansion_error": data.get("_error_message"),
                    "usage_metadata": usage_metadata,
                }

            if not self._validate_json_structure(data, ["expanded_query"]):
                if self.lg:
                    self.lg.log_struct(
                        message="Query expansion (v2) failed, falling back to original query",
                        structured_data={
                            **self.base_log,
                            "query": query,
                        },
                        severity="WARNING",
                    )
                return {
                    "expanded_query": query,
                    "abbreviations_expanded": [],
                    "expansion_failed": True,
                    "expansion_error": "Response missing required fields",
                    "usage_metadata": usage_metadata,
                }

            return {
                "expanded_query": data.get("expanded_query", query),
                "abbreviations_expanded": data.get("abbreviations_expanded", []),
                "usage_metadata": usage_metadata,
            }

    # -----------------------
    # STEP 2 (v2): Intent Extraction
    # -----------------------
    def extract_intents_v2(
        self,
        original_query: str,
        expanded_query: str,
        model_name: str | None = None,
        generation_config: dict[str, Any] | None = None,
        location: str | None = None,
    ) -> dict[str, Any]:
        """Extracts intents using INTENT_EXTRACTION_PROMPT_V2.

        No clinical validation — schema matches the reference
        (total_intents_detected + intents only). v1 extract_intents() is untouched.
        """
        span = None
        span_ctx = (
            self.tracer.start_as_current_span("intent_extraction_v2")
            if self.tracer
            else None
        )
        if span_ctx is not None:
            span = span_ctx.__enter__()

        try:
            prompt = INTENT_EXTRACTION_PROMPT_V2.format(
                expanded_query=expanded_query,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

            raw_response, usage_metadata = self._call_model(
                prompt,
                model_name=model_name,
                generation_config=generation_config,
                location=location,
                max_tokens=65536,
                step_name="intent_extraction_v2",
                response_schema=IntentExtractionResponseV2,
            )
            data = self._safe_json(raw_response)
            if isinstance(data, list):
                data = {"intents": data}

            try:
                validated_response = self._validate_llm_response_v2(data)
                if span is not None:
                    span.set_attribute("intent_count", len(validated_response.intents))
                    span.set_attribute("validation_success", True)
                if self.lg:
                    self.lg.log_struct(
                        message="LLM v2 response validation successful",
                        structured_data={
                            **self.base_log,
                            "intent_count": len(validated_response.intents),
                        },
                        severity="INFO",
                    )
            except ValueError as e:
                if span is not None:
                    span.set_attribute("validation_success", False)
                    span.set_attribute("parsing_error", True)
                if self.lg:
                    self.lg.log_struct(
                        message="LLM v2 response parsing failed - returning error",
                        structured_data={
                            **self.base_log,
                            "error_message": str(e),
                        },
                        severity="ERROR",
                    )
                return {
                    "intents": [],
                    "total_intents_detected": 0,
                    "error": "LLM response parsing failed",
                    "error_details": str(e),
                    "original_query": original_query,
                    "expanded_query": expanded_query,
                    "usage_metadata": usage_metadata,
                }
            except ValidationError as e:
                if span is not None:
                    span.set_attribute("validation_success", False)
                    span.set_attribute("error_count", len(e.errors()))
                if self.lg:
                    self.lg.log_struct(
                        message="LLM v2 response validation failed - returning error",
                        structured_data={
                            **self.base_log,
                            "validation_error_count": len(e.errors()),
                        },
                        severity="ERROR",
                    )
                return {
                    "intents": [],
                    "total_intents_detected": 0,
                    "error": "LLM response validation failed",
                    "validation_errors": [err["msg"] for err in e.errors()],
                    "original_query": original_query,
                    "expanded_query": expanded_query,
                    "usage_metadata": usage_metadata,
                }

            validated_intents = []
            for intent in validated_response.intents:
                validated_intents.append(
                    {
                        "intent_title": intent.intent_title,
                        "description": intent.description,
                        "nature": intent.nature,
                        "sub_natures": [
                            {
                                "category_path": sub_nature.category_path,
                                "atomic_concepts": sub_nature.atomic_concepts,
                            }
                            for sub_nature in intent.sub_natures
                        ],
                        "final_queries": intent.final_queries,
                    }
                )

            if self.lg:
                self.lg.log_struct(
                    message="Intents extracted successfully (v2)",
                    structured_data={
                        **self.base_log,
                        "intent_count": len(validated_intents),
                    },
                    severity="INFO",
                )

            return {
                "intents": validated_intents,
                "total_intents_detected": validated_response.total_intents_detected,
                "original_query": original_query,
                "expanded_query": expanded_query,
                "usage_metadata": usage_metadata,
            }
        finally:
            if span_ctx is not None:
                span_ctx.__exit__(None, None, None)

    # -----------------------
    # STEP 3 (v2): Representative Terms
    # -----------------------
    def extract_representative_terms(
        self,
        expanded_query: str,
        model_name: str | None = None,
        generation_config: dict[str, Any] | None = None,
        location: str | None = None,
    ) -> dict[str, Any]:
        """Distill the expanded query into canonical representative terms.
        New v2 LLM step; schema is {representative_terms: []}."""
        empty_metadata = {
            "prompt_token_count": 0,
            "candidates_token_count": 0,
            "total_token_count": 0,
            "thinking_token_count": 0,
        }
        prompt = REPRESENTATIVE_TERMS_PROMPT_V2.format(expanded_query=expanded_query)

        try:
            raw_response, usage_metadata = self._call_model(
                prompt,
                model_name=model_name,
                generation_config=generation_config,
                location=location,
                max_tokens=2048,
                step_name="representative_terms",
                response_schema=RepresentativeTermsOutput,
            )
        except LLMError as e:
            if self.lg:
                self.lg.log_struct(
                    message="Representative terms LLM call failed, returning empty list",
                    structured_data={
                        **self.base_log,
                        "error_type": e.error_type,
                        "error_message": str(e),
                        "fallback": True,
                    },
                    severity="ERROR",
                )
            return {"representative_terms": [], "usage_metadata": empty_metadata}

        data = self._safe_json(raw_response)

        if data.get("_parsing_error") or not isinstance(
            data.get("representative_terms"), list
        ):
            if self.lg:
                self.lg.log_struct(
                    message="Representative terms parsing failed, returning empty list",
                    structured_data={
                        **self.base_log,
                        "error_message": data.get("_error_message"),
                        "fallback": True,
                    },
                    severity="WARNING",
                )
            return {
                "representative_terms": [],
                "usage_metadata": usage_metadata,
            }

        return {
            "representative_terms": data.get("representative_terms", []),
            "usage_metadata": usage_metadata,
        }

    # -----------------------
    # STEP 4 (v2): Contextual Environment (retrieval signals source)
    # -----------------------
    def build_context(
        self,
        query: str,
        expanded_query: str,
        intent_result: dict[str, Any],
        model_name: str | None = None,
        generation_config: dict[str, Any] | None = None,
        location: str | None = None,
        include_record_type_matching: bool = True,
    ) -> dict[str, Any]:
        """Map each atomic concept's documentation facets AND produce lifecycle
        qualifiers per intent/candidate in one LLM call. Returns parsed
        ContextualEnvironmentOutput + derived TemporalExtractionOutput.

        include_record_type_matching: when False the KNOWN RECORD TYPES vocabulary
        block is not injected and no record_type_matches are requested — used by
        /v2/retrieval-signals, which codes record types through the document
        cluster instead of the local JSON vocabulary."""
        empty_metadata = {
            "prompt_token_count": 0,
            "candidates_token_count": 0,
            "total_token_count": 0,
            "thinking_token_count": 0,
        }
        intents = intent_result.get("intents", []) or []
        if not intents:
            return {"context": None, "temporal": None, "usage_metadata": empty_metadata}

        # Flat list of unique concepts (Task 1 target list)
        concepts: list[dict[str, str]] = []
        seen: set = set()
        for i in intents:
            intent_title = i.get("intent_title", "")
            for sn in i.get("sub_natures") or []:
                for c in sn.get("atomic_concepts") or []:
                    k = c.strip().lower()
                    if k and k not in seen:
                        concepts.append(
                            {"atomic_concept": c.strip(), "intent_title": intent_title}
                        )
                        seen.add(k)
        if not concepts:
            return {"context": None, "temporal": None, "usage_metadata": empty_metadata}

        # Rich per-intent structure (for Task 2 lifecycle reasoning)
        intents_payload: list[dict[str, Any]] = []
        for i in intents:
            cand_entries: list[str] = []
            seen_c: set = set()
            for sn in i.get("sub_natures") or []:
                for c in sn.get("atomic_concepts") or []:
                    k = c.strip().lower()
                    if k and k not in seen_c:
                        cand_entries.append(c.strip())
                        seen_c.add(k)
            intents_payload.append(
                {
                    "intent_title": i.get("intent_title", ""),
                    "description": i.get("description", ""),
                    "nature": i.get("nature", ""),
                    "sub_natures": [
                        {
                            "category_path": sn.get("category_path", ""),
                            "atomic_concepts": list(sn.get("atomic_concepts") or []),
                        }
                        for sn in i.get("sub_natures") or []
                    ],
                    "candidates": cand_entries,
                }
            )

        # Record-type matching guidance (parallel to temporal matching semantics):
        # allow one or more canonical record-type matches while keeping Task 1
        # output shape unchanged (record_types remains a List[str]).
        record_type_matching_block = (
            "\nRECORD TYPE MATCHING (Task 1, for EVERY concept):\n"
            "When producing `record_types`, normalize each item to canonical clinical document "
            "type labels and allow ONE OR MORE matches when appropriate. Keep strict relevance "
            "ordering: most definitive source first, less direct sources later.\n"
            "Output only the normalized record-type names (strings) in `record_types` — do not "
            "output ids or reasoning fields.\n"
            "If two labels are both valid and commonly used, include both. If uncertain, prefer "
            "specific clinical document names over generic placeholders.\n"
            "Deduplicate near-synonyms; keep the canonical form with the broadest interoperability "
            "(e.g., prefer 'progress_note' over variations).\n"
        )

        # FOLDED matching (parallel to temporal): when a vocab is set, inject the
        # known record-type labels so the model chooses from real vocabulary
        # entries. Matching is the model's job — it reasons over the concept in
        # context; downstream lookup is an exact/case-insensitive resolution only.
        if include_record_type_matching and self._record_type_vocab is not None:
            record_type_names = self._record_type_vocab.names()
            if record_type_names:
                record_type_matching_block += (
                    "\nKNOWN RECORD TYPES (MANDATORY VOCABULARY):\n"
                    "The list below is the authoritative vocabulary of known record-type labels. "
                    "For EVERY record type you emit, compare the concept against the ENTIRE list "
                    "before deciding whether a known label matches.\n"
                    "\n"
                    "IMPORTANT: these are TWO separate outputs:\n"
                    "1. `record_types`: canonical snake_case document types describing the "
                    "underlying document.\n"
                    "2. `record_type_matches`: maps each canonical record type to ALL compatible "
                    "raw vocabulary labels that refer to that same document.\n"
                    "The vocabulary labels are evidence for matching and do not need to use the "
                    "same wording as the canonical record type.\n"
                    "\n"
                    "STEP 1 — Name the document, not the data.\n"
                    "Ask: in which physical document does a clinician WRITE this concept down? "
                    "The answer is a document, never the finding, the value, the procedure, or "
                    "the person. Common mistakes to avoid:\n"
                    "  - a data element recorded inside a document is NOT the document "
                    "(discharge disposition is a field; the document is the discharge summary)\n"
                    "  - a clinical act is NOT the document "
                    "(prescribing is an act; the document is the prescription or medication order)\n"
                    "  - a specialty, role or setting is NOT a document "
                    "(cardiology, pathologist, inpatient are none of them)\n"
                    "\n"
                    "STEP 2 — Match by MEANING against the list.\n"
                    "Read the ENTIRE list and semantically compare it with the concept, intent, "
                    "description, original query, expanded query, workflow, and care setting. "
                    "Pick labels that name THAT document, even when the wording differs from "
                    "the canonical type. Case, spacing and underscores do not "
                    "matter. Expand abbreviations before judging (AVS = after visit summary, "
                    "IP = inpatient, MAR = medication administration record, H&P = history and "
                    "physical, Rx = prescription, D/C = discharge). Output the list label "
                    "VERBATIM — do not reword, shorten, pluralize or re-case it.\n"
                    "\n"
                    "STEP 3 — Respect the setting.\n"
                    "When a list label carries a care setting, it must AGREE with the concept: "
                    "do not pick an outpatient label for an inpatient-only source, or the "
                    "reverse. When the concept implies no particular setting, prefer the "
                    "setting-neutral label.\n"
                    "\n"
                    "STEP 4 — Rank and keep only the top document types.\n"
                    "After semantic matching, rank the distinct matched document types by "
                    "relevance: direct source first, then interpretation/management, then "
                    "summary/incidental sources. Emit only the highest-value matches needed to "
                    "cover the concept, normally 1-3 document types and never more than 5. "
                    "Do NOT emit low-value matches just to fill the list. Preserve this ranking "
                    "in `record_types` and use the same order in `record_type_matches`. Do NOT "
                    "merge two documents into one label or use a broader label as a stand-in for "
                    "two narrower ones.\n"
                    "\n"
                    "STEP 5 — Canonical record type format.\n"
                    "Each emitted `record_type` must be a concise snake_case name for the "
                    "underlying document, such as `radiology_report`, `discharge_summary`, "
                    "`medication_order`, or `consultation_report`. The canonical name does not "
                    "need to literally occur in the vocabulary; it groups synonymous vocabulary "
                    "labels for the same document.\n"
                    "Never emit a bare generic word.\n"
                    "A label such as 'report', 'note', 'list', 'summary' or 'record' alone names no "
                    "document and cannot be coded. Emit the full document name "
                    "('radiology_report', not 'report').\n"
                    "\n"
                    "STEP 6 — SECOND FULL SEMANTIC PASS IF NOTHING MATCHES.\n"
                    "If the first full-list semantic pass produces no valid match, do not stop. "
                    "Re-read the atomic concept, intent title, description, original query, and "
                    "expanded query; expand abbreviations and identify the actual document "
                    "generated by the workflow. Scan the ENTIRE list a second time for synonyms, "
                    "abbreviations, setting-qualified labels, and document-vs-action distinctions. "
                    "Then rank and emit the best valid matches. Only after this second semantic "
                    "check may `selected_names` be empty.\n"
                    "If the second semantic check still finds no label in the list, emit your own "
                    "snake_case label with an empty `selected_names` list. A label outside the "
                    "list is acceptable only after both semantic checks; a forced wrong match is "
                    "not.\n"
                    "\n"
                    "=== REQUIRED TOP-LEVEL OUTPUT: `record_type_matches` ===\n"
                    "Your response has THREE top-level keys: concepts_with_context, "
                    "temporal_by_intent, AND record_type_matches. Omitting record_type_matches "
                    "or returning it empty is INVALID output whenever you emitted any "
                    "record_types.\n"
                    "\n"
                    "COUNT RULE: one object per DISTINCT record type emitted anywhere in Task 1. "
                    "If Task 1 emitted 6 distinct record types across all concepts, "
                    "record_type_matches contains exactly 6 objects. Count them before "
                    "returning.\n"
                    "\n"
                    "Each object is:\n"
                    '  {{"record_type": "<canonical record_type exactly as emitted>", '
                    '"selected_names": ["<list label>", ...], '
                    '"selected_reasoning": "<one short sentence>"}}\n'
                    "\n"
                    "MULTI-MATCH IS THE NORM, NOT THE EXCEPTION. `selected_names` holds EVERY "
                    "list label naming that same document. The list carries a base concept "
                    "PLUS setting- and role-qualified variants of it, and they are SEPARATE "
                    "entries. Returning only the base concept, or only one label, is the most "
                    "common mistake — scan the WHOLE list for every entry naming that document "
                    "before you stop.\n"
                    "\n"
                    'Worked example. List contains: ["Consultation Note", '
                    '"Consultation Note | Hospital", "Consultation Note | Outpatient", '
                    '"Confirmatory Consultation Note", "Progress Note"].\n'
                    "  Task 1 emitted: consultation_report\n"
                    '  CORRECT: {{"record_type": "consultation_report", "selected_names": '
                    '["Consultation Note", "Consultation Note | Hospital", '
                    '"Consultation Note | Outpatient", "Confirmatory Consultation Note"], '
                    '"selected_reasoning": "base concept and its setting variants all name the '
                    'consultation note document"}}\n'
                    '  WRONG:   selected_names ["Consultation Note"]  (stopped at the first hit)\n'
                    '  WRONG:   selected_names ["Progress Note"]      (different document)\n'
                    "\n"
                    "Include a variant only when its setting is compatible with the concept: an "
                    "inpatient-only source must not take an outpatient-qualified variant.\n"
                    "Copy each label from the list VERBATIM — a label not in the list resolves "
                    "to nothing. Use an EMPTY selected_names ONLY when no list label names that "
                    "document; never invent a label.\n"
                    f"{json.dumps(record_type_names, ensure_ascii=False)}\n"
                )

        # FOLDED matching: when a vocab is set, inject the concept names so the
        # model picks a match for each temporal_signal inline. Otherwise leave
        # the block empty (the temporal_resolver handles matching downstream).
        if self._temporal_vocab is not None:
            transactions = self._temporal_vocab.transactions()
            temporal_matching_block = (
                "\nTEMPORAL CONCEPT MATCHING (do this for EVERY temporal_signal entry):\n"
                "For each temporal_signal entry, besides `signal`, identify ONE OR MORE temporal "
                "concepts from the TEMPORAL CONCEPTS list that are closely aligned by meaning. "
                "If there is one best concept, set `selected_id` and `selected_name` as before. "
                "If multiple concepts are valid, set `selected_ids` to an array of ids and "
                "`selected_names` to the aligned concept names in the same order; keep `selected_id`/"
                "`selected_name` null in that case. Set `selected_reasoning` to a short justification.\n"
                "Match by MEANING, not surface spelling: a lifecycle/state phrase maps to the "
                "concept for the position in time it occupies (an ongoing state -> a "
                "'present'/'current' concept, a planned one -> a 'future' concept); a duration maps "
                "to the matching duration concept.\n"
                "Copy `selected_name` / `selected_names` from the list VERBATIM — a name not in "
                "the list resolves to no code.\n\n"
                "EVERY ENTRY IS MATCHED, INCLUDING TYPE DEFAULTS. `signal_basis` has no bearing "
                "here: an entry produced by the type-default rule (no explicit time wording in "
                "the query) is matched exactly like an explicit one. A concept inferred from the "
                "candidate's clinical nature is still a real temporal concept and still has a "
                "code — skipping the match for it because the query said nothing about time is "
                "the single most common failure of this task.\n\n"
                "THE LIST MAY NOT CONTAIN THE QUALIFIER WORD ITSELF. Do not assume a bare "
                "lifecycle or ordering word ('current', 'past', 'recent', 'historical', 'most "
                "recent', 'planned') appears in the list — read the list and check. When it does "
                "not, the qualifier is NOT unmatchable: it is codeable as the RETRIEVAL WINDOW "
                "it implies for that candidate. Return the NEXT POSSIBLE MATCH from the list "
                "rather than nothing — the span that qualifier means for THIS concept's "
                "documentation cadence — and state that substitution in `selected_reasoning`.\n\n"
                "EVERY ENTRY RESOLVES TO A CONCEPT. There is no unmatched outcome: an entry whose "
                "selected fields are left empty is DISCARDED downstream and the candidate loses "
                "its temporal signal entirely. Work down this ladder and stop at the first rung "
                "that yields a concept:\n"
                "  1. the signal's own wording, in any casing or spacing\n"
                "  2. the same meaning worded differently (an abbreviation, a spelled-out "
                "number, a singular/plural unit, a leading 'past'/'last'/'in the')\n"
                "  3. the same span expressed in another unit (a span in days vs the same span "
                "in weeks or months)\n"
                "  4. for a lifecycle or ordering qualifier absent from the list, the window it "
                "implies for this candidate (see above)\n"
                "  5. the nearest concept BROADER than the signal — a wider window that still "
                "contains it is always a valid match\n"
                "Rung 5 can always be satisfied: a broader window never contradicts the signal, "
                "it only retrieves more. Never return null because no entry is exact.\n"
                "Two constraints on the choice: ignore list entries that are questionnaire or "
                "survey item text (they mention a span inside a sentence about something else "
                "and do not name the window itself), and never substitute a NARROWER window or "
                "an unrelated clinical concept — widen, do not narrow.\n\n"
                "Examples:\n"
                "  signal 'historical', concepts [[id_1,'Historical'],[id_2,'Histological type'],[id_3,'history']]\n"
                "    -> selected_id id_1, selected_name 'Historical' (closest alignment, best choice)\n"
                "  signal 'past 6 months', concepts [[id_4,'Last six months'],[id_5,'Within 6 months']]\n"
                "    -> selected_ids ['id_4','id_5'], selected_names ['Last six months','Within 6 months']\n"
                "  signal 'current' for an active medication, list has no bare 'current' but has "
                "[[id_6,'Past 30 days'],[id_7,'Past Year'],[id_8,'Gestation period, 30 weeks']]\n"
                "    -> selected_id id_6, selected_name 'Past 30 days' (no 'current' entry exists; "
                "an active medication is current as of its most recent 30-day documentation "
                "window — next possible match, not null)\n"
                "  signal 'most recent' for an encounter, concepts [[id_1,'Most recent treponemal "
                "test type'],[id_2,'most recent pregnancy'],[id_3,'Past Year']]\n"
                "    -> selected_id id_3, selected_name 'Past Year' (id_1 and id_2 are narrower "
                "clinical concepts, not windows; the broader window is the valid match)\n"
                "  signal 'ongoing', concepts include [id_k,'Present']\n"
                "    -> selected_id id_k, selected_name 'Present' (an ongoing state maps by meaning to the present)\n\n"
                "TEMPORAL CONCEPTS (id, name):\n"
                f"{json.dumps(transactions, ensure_ascii=False)}\n"
            )
        else:
            temporal_matching_block = ""

        prompt = CONTEXTUAL_ENVIRONMENT_PROMPT_V2.format(
            original_query=query,
            expanded_query=expanded_query,
            concepts_json=json.dumps(concepts, indent=2, ensure_ascii=False),
            concept_count=len(concepts),
            intents_json=json.dumps(intents_payload, indent=2, ensure_ascii=False),
            record_type_matching_block=record_type_matching_block,
            temporal_matching_block=temporal_matching_block,
        )

        raw_response, usage_metadata = self._call_model(
            prompt,
            model_name=model_name,
            generation_config=generation_config,
            location=location,
            # Budget covers thinking + answer; 16384 let reasoning starve the JSON body.
            max_tokens=65536,
            step_name="contextual_environment",
            response_schema=ContextualEnvironmentOutput,
        )
        data = self._safe_json(raw_response)

        if data.get("_parsing_error"):
            if self.lg:
                self.lg.log_struct(
                    message="Contextual environment parsing failed, returning empty context",
                    structured_data={
                        **self.base_log,
                        "error_message": data.get("_error_message"),
                        "is_truncated": data.get("_is_truncated"),
                        "fallback": True,
                    },
                    severity="WARNING",
                )
            return {
                "context": None,
                "temporal": None,
                "record_type_matches": [],
                "usage_metadata": usage_metadata,
            }

        try:
            context = ContextualEnvironmentOutput.model_validate(data)
        except ValidationError as e:
            if self.lg:
                self.lg.log_struct(
                    message="Contextual environment validation failed, returning empty context",
                    structured_data={
                        **self.base_log,
                        "validation_error_count": len(e.errors()),
                    },
                    severity="WARNING",
                )
            return {
                "context": None,
                "temporal": None,
                "record_type_matches": [],
                "usage_metadata": usage_metadata,
            }

        temporal = TemporalExtractionOutput(
            intents=list(context.temporal_by_intent or [])
        )

        return {
            "context": context,
            "temporal": temporal,
            # Carried alongside (not inside) the facets so record_types stays a
            # plain List[str] at every layer, exactly as temporal_by_intent is
            # carried alongside the temporal_signal strings.
            "record_type_matches": [
                m.model_dump() for m in (context.record_type_matches or [])
            ],
            "usage_metadata": usage_metadata,
        }

    # -----------------------
    # Pipeline Runner
    # -----------------------
    def run(
        self,
        query: str,
        model_name: str | None = None,
        generation_config: dict[str, Any] | None = None,
        location: str | None = None,
    ) -> dict[str, Any]:
        """Runs the intent extraction pipeline.
        Args:
            query: The query to extract the intents from.
            model_name: Optional model name override.
            generation_config: Optional generation config from request.
            location: Optional GCP region override.
        Returns:
            The intents from the query.
        Raises:
            Exception: If the intent extraction pipeline fails.
        """
        start_time = datetime.now(timezone.utc)

        expansion = self.expand_query(
            query,
            model_name=model_name,
            generation_config=generation_config,
            location=location,
        )
        expanded_query = expansion["expanded_query"]

        intent_result = self.extract_intents(
            query,
            expanded_query,
            model_name=model_name,
            generation_config=generation_config,
            location=location,
        )

        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()

        # Keep usage metadata separate for query expansion and intent extraction
        expansion_metadata = expansion.get("usage_metadata", {})
        intent_metadata = intent_result.get("usage_metadata", {})

        if not intent_result.get("is_clinical", True):
            return {
                "original_query": query,
                "expanded_query": expanded_query,
                "intents": [],
                "is_clinical": False,
                "rejected_reason": intent_result.get("rejected_reason"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "processing_time_seconds": processing_time,
                "usage_metadata": {
                    "query_expansion": expansion_metadata,
                    "intent_extraction": intent_metadata,
                },
            }

        return {
            "original_query": query,
            "expanded_query": expanded_query,
            "abbreviations_expanded": expansion.get("abbreviations_expanded", []),
            "is_clinical": True,
            "intents": intent_result.get("intents", []),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "processing_time_seconds": processing_time,
            "usage_metadata": {
                "query_expansion": expansion_metadata,
                "intent_extraction": intent_metadata,
            },
        }

    # -----------------------
    # v2 async step methods (retrieval-signals async path)
    # -----------------------

    async def expand_query_v2_async(
        self,
        query: str,
        model_name: str | None = None,
        generation_config: dict[str, Any] | None = None,
        location: str | None = None,
    ) -> dict[str, Any]:
        """Expands the query using the v2 prompt. Mirrors expand_query() but
        uses QUERY_EXPANSION_PROMPT_V2 (v1 method left untouched)."""
        if self.tracer:
            with self.tracer.start_as_current_span("query_expansion_v2") as span:
                span.set_attribute("query_length", len(query))

                prompt = QUERY_EXPANSION_PROMPT_V2.format(query=query)

                try:
                    raw_response, usage_metadata = await self._call_model_async(
                        prompt,
                        model_name=model_name,
                        generation_config=generation_config,
                        location=location,
                        step_name="query_expansion_v2",
                        response_schema=QueryExpansionOutput,
                    )
                except LLMError as e:
                    span.set_attribute("success", False)
                    span.set_attribute("error", str(e))
                    if self.lg:
                        self.lg.log_struct(
                            message="Query expansion (v2) LLM call failed, falling back to original query",
                            structured_data={
                                **self.base_log,
                                "query_length": len(query),
                                "error_type": e.error_type,
                                "error_message": str(e),
                                "fallback": True,
                            },
                            severity="ERROR",
                        )
                    return {
                        "expanded_query": query,
                        "abbreviations_expanded": [],
                        "expansion_failed": True,
                        "expansion_error": str(e),
                        "expansion_error_type": e.error_type,
                        "usage_metadata": {
                            "prompt_token_count": 0,
                            "candidates_token_count": 0,
                            "total_token_count": 0,
                            "thinking_token_count": 0,
                        },
                    }

                data = self._safe_json(raw_response)

                if data.get("_parsing_error"):
                    span.set_attribute("success", False)
                    span.set_attribute("parsing_error", True)
                    if self.lg:
                        self.lg.log_struct(
                            message="Query expansion (v2) JSON parsing failed, falling back to original query",
                            structured_data={
                                **self.base_log,
                                "query_length": len(query),
                                "error_message": data.get("_error_message"),
                                "is_truncated": data.get("_is_truncated"),
                                "fallback": True,
                            },
                            severity="WARNING",
                        )
                    return {
                        "expanded_query": query,
                        "abbreviations_expanded": [],
                        "expansion_failed": True,
                        "expansion_error": data.get("_error_message"),
                        "usage_metadata": usage_metadata,
                    }

                if not self._validate_json_structure(data, ["expanded_query"]):
                    if self.lg:
                        self.lg.log_struct(
                            message="Query expansion (v2) failed, falling back to original query",
                            structured_data={
                                **self.base_log,
                                "query": query,
                            },
                            severity="WARNING",
                        )
                    expanded_query = query
                    span.set_attribute("expanded_length", len(expanded_query))
                    span.set_attribute("success", False)
                    return {
                        "expanded_query": expanded_query,
                        "abbreviations_expanded": [],
                        "expansion_failed": True,
                        "expansion_error": "Response missing required fields",
                        "usage_metadata": usage_metadata,
                    }

                expanded_query = data.get("expanded_query", query)
                span.set_attribute("expanded_length", len(expanded_query))
                span.set_attribute("success", True)

                return {
                    "expanded_query": expanded_query,
                    "abbreviations_expanded": data.get("abbreviations_expanded", []),
                    "usage_metadata": usage_metadata,
                }
        else:
            prompt = QUERY_EXPANSION_PROMPT_V2.format(query=query)

            try:
                raw_response, usage_metadata = await self._call_model_async(
                    prompt,
                    model_name=model_name,
                    generation_config=generation_config,
                    location=location,
                    step_name="query_expansion_v2",
                    response_schema=QueryExpansionOutput,
                )
            except LLMError as e:
                if self.lg:
                    self.lg.log_struct(
                        message="Query expansion (v2) LLM call failed, falling back to original query",
                        structured_data={
                            **self.base_log,
                            "query_length": len(query),
                            "error_type": e.error_type,
                            "error_message": str(e),
                            "fallback": True,
                        },
                        severity="ERROR",
                    )
                return {
                    "expanded_query": query,
                    "abbreviations_expanded": [],
                    "expansion_failed": True,
                    "expansion_error": str(e),
                    "expansion_error_type": e.error_type,
                    "usage_metadata": {
                        "prompt_token_count": 0,
                        "candidates_token_count": 0,
                        "total_token_count": 0,
                        "thinking_token_count": 0,
                    },
                }

            data = self._safe_json(raw_response)

            if data.get("_parsing_error"):
                if self.lg:
                    self.lg.log_struct(
                        message="Query expansion (v2) JSON parsing failed, falling back to original query",
                        structured_data={
                            **self.base_log,
                            "query_length": len(query),
                            "error_message": data.get("_error_message"),
                            "is_truncated": data.get("_is_truncated"),
                            "fallback": True,
                        },
                        severity="WARNING",
                    )
                return {
                    "expanded_query": query,
                    "abbreviations_expanded": [],
                    "expansion_failed": True,
                    "expansion_error": data.get("_error_message"),
                    "usage_metadata": usage_metadata,
                }

            if not self._validate_json_structure(data, ["expanded_query"]):
                if self.lg:
                    self.lg.log_struct(
                        message="Query expansion (v2) failed, falling back to original query",
                        structured_data={
                            **self.base_log,
                            "query": query,
                        },
                        severity="WARNING",
                    )
                return {
                    "expanded_query": query,
                    "abbreviations_expanded": [],
                    "expansion_failed": True,
                    "expansion_error": "Response missing required fields",
                    "usage_metadata": usage_metadata,
                }

            return {
                "expanded_query": data.get("expanded_query", query),
                "abbreviations_expanded": data.get("abbreviations_expanded", []),
                "usage_metadata": usage_metadata,
            }

    # -----------------------
    # STEP 2 (v2): Intent Extraction
    # -----------------------

    async def extract_intents_v2_async(
        self,
        original_query: str,
        expanded_query: str,
        model_name: str | None = None,
        generation_config: dict[str, Any] | None = None,
        location: str | None = None,
    ) -> dict[str, Any]:
        """Extracts intents using INTENT_EXTRACTION_PROMPT_V2.

        No clinical validation — schema matches the reference
        (total_intents_detected + intents only). v1 extract_intents() is untouched.
        """
        span = None
        span_ctx = (
            self.tracer.start_as_current_span("intent_extraction_v2")
            if self.tracer
            else None
        )
        if span_ctx is not None:
            span = span_ctx.__enter__()

        try:
            prompt = INTENT_EXTRACTION_PROMPT_V2.format(
                expanded_query=expanded_query,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

            raw_response, usage_metadata = await self._call_model_async(
                prompt,
                model_name=model_name,
                generation_config=generation_config,
                location=location,
                max_tokens=65536,
                step_name="intent_extraction_v2",
                response_schema=IntentExtractionResponseV2,
            )
            data = self._safe_json(raw_response)
            if isinstance(data, list):
                data = {"intents": data}

            try:
                validated_response = self._validate_llm_response_v2(data)
                if span is not None:
                    span.set_attribute("intent_count", len(validated_response.intents))
                    span.set_attribute("validation_success", True)
                if self.lg:
                    self.lg.log_struct(
                        message="LLM v2 response validation successful",
                        structured_data={
                            **self.base_log,
                            "intent_count": len(validated_response.intents),
                        },
                        severity="INFO",
                    )
            except ValueError as e:
                if span is not None:
                    span.set_attribute("validation_success", False)
                    span.set_attribute("parsing_error", True)
                if self.lg:
                    self.lg.log_struct(
                        message="LLM v2 response parsing failed - returning error",
                        structured_data={
                            **self.base_log,
                            "error_message": str(e),
                        },
                        severity="ERROR",
                    )
                return {
                    "intents": [],
                    "total_intents_detected": 0,
                    "error": "LLM response parsing failed",
                    "error_details": str(e),
                    "original_query": original_query,
                    "expanded_query": expanded_query,
                    "usage_metadata": usage_metadata,
                }
            except ValidationError as e:
                if span is not None:
                    span.set_attribute("validation_success", False)
                    span.set_attribute("error_count", len(e.errors()))
                if self.lg:
                    self.lg.log_struct(
                        message="LLM v2 response validation failed - returning error",
                        structured_data={
                            **self.base_log,
                            "validation_error_count": len(e.errors()),
                        },
                        severity="ERROR",
                    )
                return {
                    "intents": [],
                    "total_intents_detected": 0,
                    "error": "LLM response validation failed",
                    "validation_errors": [err["msg"] for err in e.errors()],
                    "original_query": original_query,
                    "expanded_query": expanded_query,
                    "usage_metadata": usage_metadata,
                }

            validated_intents = []
            for intent in validated_response.intents:
                validated_intents.append(
                    {
                        "intent_title": intent.intent_title,
                        "description": intent.description,
                        "nature": intent.nature,
                        "sub_natures": [
                            {
                                "category_path": sub_nature.category_path,
                                "atomic_concepts": sub_nature.atomic_concepts,
                            }
                            for sub_nature in intent.sub_natures
                        ],
                        "final_queries": intent.final_queries,
                    }
                )

            if self.lg:
                self.lg.log_struct(
                    message="Intents extracted successfully (v2)",
                    structured_data={
                        **self.base_log,
                        "intent_count": len(validated_intents),
                    },
                    severity="INFO",
                )

            return {
                "intents": validated_intents,
                "total_intents_detected": validated_response.total_intents_detected,
                "original_query": original_query,
                "expanded_query": expanded_query,
                "usage_metadata": usage_metadata,
            }
        finally:
            if span_ctx is not None:
                span_ctx.__exit__(None, None, None)

    # -----------------------
    # STEP 3 (v2): Representative Terms
    # -----------------------

    async def extract_representative_terms_async(
        self,
        expanded_query: str,
        model_name: str | None = None,
        generation_config: dict[str, Any] | None = None,
        location: str | None = None,
    ) -> dict[str, Any]:
        """Distill the expanded query into canonical representative terms.
        New v2 LLM step; schema is {representative_terms: []}."""
        empty_metadata = {
            "prompt_token_count": 0,
            "candidates_token_count": 0,
            "total_token_count": 0,
            "thinking_token_count": 0,
        }
        prompt = REPRESENTATIVE_TERMS_PROMPT_V2.format(expanded_query=expanded_query)

        try:
            raw_response, usage_metadata = await self._call_model_async(
                prompt,
                model_name=model_name,
                generation_config=generation_config,
                location=location,
                max_tokens=2048,
                step_name="representative_terms",
                response_schema=RepresentativeTermsOutput,
            )
        except LLMError as e:
            if self.lg:
                self.lg.log_struct(
                    message="Representative terms LLM call failed, returning empty list",
                    structured_data={
                        **self.base_log,
                        "error_type": e.error_type,
                        "error_message": str(e),
                        "fallback": True,
                    },
                    severity="ERROR",
                )
            return {"representative_terms": [], "usage_metadata": empty_metadata}

        data = self._safe_json(raw_response)

        if data.get("_parsing_error") or not isinstance(
            data.get("representative_terms"), list
        ):
            if self.lg:
                self.lg.log_struct(
                    message="Representative terms parsing failed, returning empty list",
                    structured_data={
                        **self.base_log,
                        "error_message": data.get("_error_message"),
                        "fallback": True,
                    },
                    severity="WARNING",
                )
            return {
                "representative_terms": [],
                "usage_metadata": usage_metadata,
            }

        return {
            "representative_terms": data.get("representative_terms", []),
            "usage_metadata": usage_metadata,
        }

    # -----------------------
    # STEP 4 (v2): Contextual Environment (retrieval signals source)
    # -----------------------

    async def build_context_async(
        self,
        query: str,
        expanded_query: str,
        intent_result: dict[str, Any],
        model_name: str | None = None,
        generation_config: dict[str, Any] | None = None,
        location: str | None = None,
        include_record_type_matching: bool = True,
        temporal_mode: str = TEMPORAL_MODE_VOCAB_LIST,
    ) -> dict[str, Any]:
        """Map each atomic concept's documentation facets AND produce lifecycle
        qualifiers per intent/candidate in one LLM call. Returns parsed
        ContextualEnvironmentOutput + derived TemporalExtractionOutput.

        include_record_type_matching: when False the KNOWN RECORD TYPES vocabulary
        block is not injected and no record_type_matches are requested — used by
        /v2/retrieval-signals, which codes record types through the document
        cluster instead of the local JSON vocabulary."""
        empty_metadata = {
            "prompt_token_count": 0,
            "candidates_token_count": 0,
            "total_token_count": 0,
            "thinking_token_count": 0,
        }
        intents = intent_result.get("intents", []) or []
        if not intents:
            return {"context": None, "temporal": None, "usage_metadata": empty_metadata}

        # Flat list of unique concepts (Task 1 target list)
        concepts: list[dict[str, str]] = []
        seen: set = set()
        for i in intents:
            intent_title = i.get("intent_title", "")
            for sn in i.get("sub_natures") or []:
                for c in sn.get("atomic_concepts") or []:
                    k = c.strip().lower()
                    if k and k not in seen:
                        concepts.append(
                            {"atomic_concept": c.strip(), "intent_title": intent_title}
                        )
                        seen.add(k)
        if not concepts:
            return {"context": None, "temporal": None, "usage_metadata": empty_metadata}

        # Rich per-intent structure (for Task 2 lifecycle reasoning)
        intents_payload: list[dict[str, Any]] = []
        for i in intents:
            cand_entries: list[str] = []
            seen_c: set = set()
            for sn in i.get("sub_natures") or []:
                for c in sn.get("atomic_concepts") or []:
                    k = c.strip().lower()
                    if k and k not in seen_c:
                        cand_entries.append(c.strip())
                        seen_c.add(k)
            intents_payload.append(
                {
                    "intent_title": i.get("intent_title", ""),
                    "description": i.get("description", ""),
                    "nature": i.get("nature", ""),
                    "sub_natures": [
                        {
                            "category_path": sn.get("category_path", ""),
                            "atomic_concepts": list(sn.get("atomic_concepts") or []),
                        }
                        for sn in i.get("sub_natures") or []
                    ],
                    "candidates": cand_entries,
                }
            )

        # Record-type matching guidance (parallel to temporal matching semantics):
        # allow one or more canonical record-type matches while keeping Task 1
        # output shape unchanged (record_types remains a List[str]).
        record_type_matching_block = (
            "\nRECORD TYPE MATCHING (Task 1, for EVERY concept):\n"
            "When producing `record_types`, normalize each item to canonical clinical document "
            "type labels and allow ONE OR MORE matches when appropriate. Keep strict relevance "
            "ordering: most definitive source first, less direct sources later.\n"
            "Output only the normalized record-type names (strings) in `record_types` — do not "
            "output ids or reasoning fields.\n"
            "If two labels are both valid and commonly used, include both. If uncertain, prefer "
            "specific clinical document names over generic placeholders.\n"
            "Deduplicate near-synonyms; keep the canonical form with the broadest interoperability "
            "(e.g., prefer 'progress_note' over variations).\n"
        )

        # FOLDED matching (parallel to temporal): when a vocab is set, inject the
        # known record-type labels so the model chooses from real vocabulary
        # entries. Matching is the model's job — it reasons over the concept in
        # context; downstream lookup is an exact/case-insensitive resolution only.
        if include_record_type_matching and self._record_type_vocab is not None:
            record_type_names = self._record_type_vocab.names()
            if record_type_names:
                record_type_matching_block += (
                    "\nKNOWN RECORD TYPES (MANDATORY VOCABULARY):\n"
                    "The list below is the authoritative vocabulary of known record-type labels. "
                    "For EVERY record type you emit, compare the concept against the ENTIRE list "
                    "before deciding whether a known label matches.\n"
                    "\n"
                    "IMPORTANT: these are TWO separate outputs:\n"
                    "1. `record_types`: canonical snake_case document types describing the "
                    "underlying document.\n"
                    "2. `record_type_matches`: maps each canonical record type to ALL compatible "
                    "raw vocabulary labels that refer to that same document.\n"
                    "The vocabulary labels are evidence for matching and do not need to use the "
                    "same wording as the canonical record type.\n"
                    "\n"
                    "STEP 1 — Name the document, not the data.\n"
                    "Ask: in which physical document does a clinician WRITE this concept down? "
                    "The answer is a document, never the finding, the value, the procedure, or "
                    "the person. Common mistakes to avoid:\n"
                    "  - a data element recorded inside a document is NOT the document "
                    "(discharge disposition is a field; the document is the discharge summary)\n"
                    "  - a clinical act is NOT the document "
                    "(prescribing is an act; the document is the prescription or medication order)\n"
                    "  - a specialty, role or setting is NOT a document "
                    "(cardiology, pathologist, inpatient are none of them)\n"
                    "\n"
                    "STEP 2 — Match by MEANING against the list.\n"
                    "Read the ENTIRE list and semantically compare it with the concept, intent, "
                    "description, original query, expanded query, workflow, and care setting. "
                    "Pick labels that name THAT document, even when the wording differs from "
                    "the canonical type. Case, spacing and underscores do not "
                    "matter. Expand abbreviations before judging (AVS = after visit summary, "
                    "IP = inpatient, MAR = medication administration record, H&P = history and "
                    "physical, Rx = prescription, D/C = discharge). Output the list label "
                    "VERBATIM — do not reword, shorten, pluralize or re-case it.\n"
                    "\n"
                    "STEP 3 — Respect the setting.\n"
                    "When a list label carries a care setting, it must AGREE with the concept: "
                    "do not pick an outpatient label for an inpatient-only source, or the "
                    "reverse. When the concept implies no particular setting, prefer the "
                    "setting-neutral label.\n"
                    "\n"
                    "STEP 4 — Rank and keep only the top document types.\n"
                    "After semantic matching, rank the distinct matched document types by "
                    "relevance: direct source first, then interpretation/management, then "
                    "summary/incidental sources. Emit only the highest-value matches needed to "
                    "cover the concept, normally 1-3 document types and never more than 5. "
                    "Do NOT emit low-value matches just to fill the list. Preserve this ranking "
                    "in `record_types` and use the same order in `record_type_matches`. Do NOT "
                    "merge two documents into one label or use a broader label as a stand-in for "
                    "two narrower ones.\n"
                    "\n"
                    "STEP 5 — Canonical record type format.\n"
                    "Each emitted `record_type` must be a concise snake_case name for the "
                    "underlying document, such as `radiology_report`, `discharge_summary`, "
                    "`medication_order`, or `consultation_report`. The canonical name does not "
                    "need to literally occur in the vocabulary; it groups synonymous vocabulary "
                    "labels for the same document.\n"
                    "Never emit a bare generic word.\n"
                    "A label such as 'report', 'note', 'list', 'summary' or 'record' alone names no "
                    "document and cannot be coded. Emit the full document name "
                    "('radiology_report', not 'report').\n"
                    "\n"
                    "STEP 6 — SECOND FULL SEMANTIC PASS IF NOTHING MATCHES.\n"
                    "If the first full-list semantic pass produces no valid match, do not stop. "
                    "Re-read the atomic concept, intent title, description, original query, and "
                    "expanded query; expand abbreviations and identify the actual document "
                    "generated by the workflow. Scan the ENTIRE list a second time for synonyms, "
                    "abbreviations, setting-qualified labels, and document-vs-action distinctions. "
                    "Then rank and emit the best valid matches. Only after this second semantic "
                    "check may `selected_names` be empty.\n"
                    "If the second semantic check still finds no label in the list, emit your own "
                    "snake_case label with an empty `selected_names` list. A label outside the "
                    "list is acceptable only after both semantic checks; a forced wrong match is "
                    "not.\n"
                    "\n"
                    "=== REQUIRED TOP-LEVEL OUTPUT: `record_type_matches` ===\n"
                    "Your response has THREE top-level keys: concepts_with_context, "
                    "temporal_by_intent, AND record_type_matches. Omitting record_type_matches "
                    "or returning it empty is INVALID output whenever you emitted any "
                    "record_types.\n"
                    "\n"
                    "COUNT RULE: one object per DISTINCT record type emitted anywhere in Task 1. "
                    "If Task 1 emitted 6 distinct record types across all concepts, "
                    "record_type_matches contains exactly 6 objects. Count them before "
                    "returning.\n"
                    "\n"
                    "Each object is:\n"
                    '  {{"record_type": "<canonical record_type exactly as emitted>", '
                    '"selected_names": ["<list label>", ...], '
                    '"selected_reasoning": "<one short sentence>"}}\n'
                    "\n"
                    "MULTI-MATCH IS THE NORM, NOT THE EXCEPTION. `selected_names` holds EVERY "
                    "list label naming that same document. The list carries a base concept "
                    "PLUS setting- and role-qualified variants of it, and they are SEPARATE "
                    "entries. Returning only the base concept, or only one label, is the most "
                    "common mistake — scan the WHOLE list for every entry naming that document "
                    "before you stop.\n"
                    "\n"
                    'Worked example. List contains: ["Consultation Note", '
                    '"Consultation Note | Hospital", "Consultation Note | Outpatient", '
                    '"Confirmatory Consultation Note", "Progress Note"].\n'
                    "  Task 1 emitted: consultation_report\n"
                    '  CORRECT: {{"record_type": "consultation_report", "selected_names": '
                    '["Consultation Note", "Consultation Note | Hospital", '
                    '"Consultation Note | Outpatient", "Confirmatory Consultation Note"], '
                    '"selected_reasoning": "base concept and its setting variants all name the '
                    'consultation note document"}}\n'
                    '  WRONG:   selected_names ["Consultation Note"]  (stopped at the first hit)\n'
                    '  WRONG:   selected_names ["Progress Note"]      (different document)\n'
                    "\n"
                    "Include a variant only when its setting is compatible with the concept: an "
                    "inpatient-only source must not take an outpatient-qualified variant.\n"
                    "Copy each label from the list VERBATIM — a label not in the list resolves "
                    "to nothing. Use an EMPTY selected_names ONLY when no list label names that "
                    "document; never invent a label.\n"
                    f"{json.dumps(record_type_names, ensure_ascii=False)}\n"
                )

        # Canonical mode (/v3) renders the frozen v3 template (app/prompts/v3):
        # no vocabulary list; the model emits a structured window per entry and
        # the index resolves it after the call. Only a small data-derived menu,
        # a query shortlist and learned examples are injected.
        canonical = temporal_mode == TEMPORAL_MODE_CANONICAL
        if canonical:
            temporal_matching_block = ""
        # FOLDED matching: when a vocab is set, inject the concept names so the
        # model picks a match for each temporal_signal inline. Otherwise leave
        # the block empty (the temporal_resolver handles matching downstream).
        elif self._temporal_vocab is not None:
            transactions = self._temporal_vocab.transactions()
            temporal_matching_block = (
                "\nTEMPORAL CONCEPT MATCHING (do this for EVERY temporal_signal entry):\n"
                "For each temporal_signal entry, besides `signal`, identify ONE OR MORE temporal "
                "concepts from the TEMPORAL CONCEPTS list that are closely aligned by meaning. "
                "If there is one best concept, set `selected_id` and `selected_name` as before. "
                "If multiple concepts are valid, set `selected_ids` to an array of ids and "
                "`selected_names` to the aligned concept names in the same order; keep `selected_id`/"
                "`selected_name` null in that case. Set `selected_reasoning` to a short justification.\n"
                "Match by MEANING, not surface spelling: a lifecycle/state phrase maps to the "
                "concept for the position in time it occupies (an ongoing state -> a "
                "'present'/'current' concept, a planned one -> a 'future' concept); a duration maps "
                "to the matching duration concept.\n"
                "Copy `selected_name` / `selected_names` from the list VERBATIM — a name not in "
                "the list resolves to no code.\n\n"
                "EVERY ENTRY IS MATCHED, INCLUDING TYPE DEFAULTS. `signal_basis` has no bearing "
                "here: an entry produced by the type-default rule (no explicit time wording in "
                "the query) is matched exactly like an explicit one. A concept inferred from the "
                "candidate's clinical nature is still a real temporal concept and still has a "
                "code — skipping the match for it because the query said nothing about time is "
                "the single most common failure of this task.\n\n"
                "THE LIST MAY NOT CONTAIN THE QUALIFIER WORD ITSELF. Do not assume a bare "
                "lifecycle or ordering word ('current', 'past', 'recent', 'historical', 'most "
                "recent', 'planned') appears in the list — read the list and check. When it does "
                "not, the qualifier is NOT unmatchable: it is codeable as the RETRIEVAL WINDOW "
                "it implies for that candidate. Return the NEXT POSSIBLE MATCH from the list "
                "rather than nothing — the span that qualifier means for THIS concept's "
                "documentation cadence — and state that substitution in `selected_reasoning`.\n\n"
                "EVERY ENTRY RESOLVES TO A CONCEPT. There is no unmatched outcome: an entry whose "
                "selected fields are left empty is DISCARDED downstream and the candidate loses "
                "its temporal signal entirely. Work down this ladder and stop at the first rung "
                "that yields a concept:\n"
                "  1. the signal's own wording, in any casing or spacing\n"
                "  2. the same meaning worded differently (an abbreviation, a spelled-out "
                "number, a singular/plural unit, a leading 'past'/'last'/'in the')\n"
                "  3. the same span expressed in another unit (a span in days vs the same span "
                "in weeks or months)\n"
                "  4. for a lifecycle or ordering qualifier absent from the list, the window it "
                "implies for this candidate (see above)\n"
                "  5. the nearest concept BROADER than the signal — a wider window that still "
                "contains it is always a valid match\n"
                "Rung 5 can always be satisfied: a broader window never contradicts the signal, "
                "it only retrieves more. Never return null because no entry is exact.\n"
                "Two constraints on the choice: ignore list entries that are questionnaire or "
                "survey item text (they mention a span inside a sentence about something else "
                "and do not name the window itself), and never substitute a NARROWER window or "
                "an unrelated clinical concept — widen, do not narrow.\n\n"
                "Examples:\n"
                "  signal 'historical', concepts [[id_1,'Historical'],[id_2,'Histological type'],[id_3,'history']]\n"
                "    -> selected_id id_1, selected_name 'Historical' (closest alignment, best choice)\n"
                "  signal 'past 6 months', concepts [[id_4,'Last six months'],[id_5,'Within 6 months']]\n"
                "    -> selected_ids ['id_4','id_5'], selected_names ['Last six months','Within 6 months']\n"
                "  signal 'current' for an active medication, list has no bare 'current' but has "
                "[[id_6,'Past 30 days'],[id_7,'Past Year'],[id_8,'Gestation period, 30 weeks']]\n"
                "    -> selected_id id_6, selected_name 'Past 30 days' (no 'current' entry exists; "
                "an active medication is current as of its most recent 30-day documentation "
                "window — next possible match, not null)\n"
                "  signal 'most recent' for an encounter, concepts [[id_1,'Most recent treponemal "
                "test type'],[id_2,'most recent pregnancy'],[id_3,'Past Year']]\n"
                "    -> selected_id id_3, selected_name 'Past Year' (id_1 and id_2 are narrower "
                "clinical concepts, not windows; the broader window is the valid match)\n"
                "  signal 'ongoing', concepts include [id_k,'Present']\n"
                "    -> selected_id id_k, selected_name 'Present' (an ongoing state maps by meaning to the present)\n\n"
                "TEMPORAL CONCEPTS (id, name):\n"
                f"{json.dumps(transactions, ensure_ascii=False)}\n"
            )
        else:
            temporal_matching_block = ""

        if canonical:
            prompt = build_contextual_environment_prompt_v3(
                original_query=query,
                expanded_query=expanded_query,
                concepts_json=json.dumps(concepts, indent=2, ensure_ascii=False),
                concept_count=len(concepts),
                intents_json=json.dumps(intents_payload, indent=2, ensure_ascii=False),
                record_type_matching_block=record_type_matching_block,
                **self._canonical_temporal_slots(query, expanded_query, intents),
            )
        else:
            prompt = CONTEXTUAL_ENVIRONMENT_PROMPT_V2.format(
                original_query=query,
                expanded_query=expanded_query,
                concepts_json=json.dumps(concepts, indent=2, ensure_ascii=False),
                concept_count=len(concepts),
                intents_json=json.dumps(intents_payload, indent=2, ensure_ascii=False),
                record_type_matching_block=record_type_matching_block,
                temporal_matching_block=temporal_matching_block,
            )

        # Canonical mode asks the model for a strict schema (window / basis /
        # rationale required on every entry) and parses with the tolerant one.
        output_model = (
            ContextualEnvironmentOutputCanonical
            if canonical
            else ContextualEnvironmentOutput
        )
        schema_model = (
            ContextualEnvironmentOutputCanonicalSchema
            if canonical
            else ContextualEnvironmentOutput
        )
        raw_response, usage_metadata = await self._call_model_async(
            prompt,
            model_name=model_name,
            generation_config=generation_config,
            location=location,
            # Budget covers thinking + answer; 16384 let reasoning starve the JSON body.
            max_tokens=65536,
            step_name="contextual_environment",
            response_schema=schema_model,
        )
        data = self._safe_json(raw_response)

        if data.get("_parsing_error"):
            if self.lg:
                self.lg.log_struct(
                    message="Contextual environment parsing failed, returning empty context",
                    structured_data={
                        **self.base_log,
                        "error_message": data.get("_error_message"),
                        "is_truncated": data.get("_is_truncated"),
                        "fallback": True,
                    },
                    severity="WARNING",
                )
            return {
                "context": None,
                "temporal": None,
                "record_type_matches": [],
                "usage_metadata": usage_metadata,
            }

        try:
            context = output_model.model_validate(data)
        except ValidationError as e:
            if self.lg:
                self.lg.log_struct(
                    message="Contextual environment validation failed, returning empty context",
                    structured_data={
                        **self.base_log,
                        "validation_error_count": len(e.errors()),
                    },
                    severity="WARNING",
                )
            return {
                "context": None,
                "temporal": None,
                "record_type_matches": [],
                "usage_metadata": usage_metadata,
            }

        temporal = TemporalExtractionOutput(
            intents=list(context.temporal_by_intent or [])
        )

        return {
            "context": context,
            "temporal": temporal,
            # Carried alongside (not inside) the facets so record_types stays a
            # plain List[str] at every layer, exactly as temporal_by_intent is
            # carried alongside the temporal_signal strings.
            "record_type_matches": [
                m.model_dump() for m in (context.record_type_matches or [])
            ],
            "usage_metadata": usage_metadata,
        }

    # -----------------------
    # Pipeline Runner (v2)
    # -----------------------
    async def run_v2_async(
        self,
        query: str,
        model_name: str | None = None,
        generation_config: dict[str, Any] | None = None,
        enable_retrieval_signals: bool = False,
        location: str | None = None,
        include_record_type_matching: bool = True,
        temporal_mode: str = TEMPORAL_MODE_VOCAB_LIST,
        temporal_shadow: bool = False,
    ) -> dict[str, Any]:
        """Runs the v2 pipeline: v2 query expansion, then parallel intent
        extraction + representative terms. When enable_retrieval_signals is
        True, also runs contextual environment (step 4) and assembles
        final_candidates + retrieval_signals. v2 prompts always run regardless
        of the flag; the flag only gates step 4. No clinical validation.
        """
        start_time = datetime.now(timezone.utc)

        # Step 1 — query expansion (v2 prompt)
        expansion = await self.expand_query_v2_async(
            query,
            model_name=model_name,
            generation_config=generation_config,
            location=location,
        )
        expanded_query = expansion["expanded_query"]

        # Steps 2 + 3 — both depend only on the expanded query, run in parallel.
        intent_result, rep_terms_result = await asyncio.gather(
            self.extract_intents_v2_async(
                query,
                expanded_query,
                model_name,
                generation_config,
                location,
            ),
            self.extract_representative_terms_async(
                expanded_query,
                model_name,
                generation_config,
                location,
            ),
        )

        expansion_metadata = expansion.get("usage_metadata", {})
        intent_metadata = intent_result.get("usage_metadata", {})
        rep_metadata = rep_terms_result.get("usage_metadata", {})

        # Step 4 + assembly — only when retrieval signals are requested.
        # No clinical short-circuit in v2 (prompt/schema have no is_clinical).
        if enable_retrieval_signals:
            canonical = temporal_mode == TEMPORAL_MODE_CANONICAL
            if canonical and self._canonical_resolver is None:
                raise TemporalIndexUnavailableError(
                    "Canonical temporal mode requested but no temporal index is loaded",
                    {"temporal_mode": temporal_mode},
                )
            context_result = await self.build_context_async(
                query,
                expanded_query,
                intent_result,
                model_name=model_name,
                generation_config=generation_config,
                location=location,
                include_record_type_matching=include_record_type_matching,
                temporal_mode=temporal_mode,
            )
            context_metadata = context_result.get("usage_metadata", {})
            record_type_matches = context_result.get("record_type_matches", []) or []
            v2_intents = assemble_v2_intents(
                intents=intent_result.get("intents", []),
                context=context_result.get("context"),
                temporal=context_result.get("temporal"),
                vocab=self._temporal_vocab,
                resolver=self._temporal_resolver,
                canonical_resolver=self._canonical_resolver if canonical else None,
            )
            shadow_metadata: dict[str, Any] | None = None
            if canonical:
                self._log_temporal_inferences(query, v2_intents, model_name)
                if temporal_shadow:
                    # Shadow: also run the legacy list path and log where the
                    # two disagree. The legacy result is never returned.
                    shadow_metadata = await self._temporal_shadow_compare(
                        query,
                        expanded_query,
                        intent_result,
                        v2_intents,
                        model_name=model_name,
                        generation_config=generation_config,
                        location=location,
                        include_record_type_matching=include_record_type_matching,
                    )
        else:
            context_metadata = {}
            shadow_metadata = None
            record_type_matches = []
            # v2 base shape: intents from step 2 only (no final_candidates).
            v2_intents = intent_result.get("intents", [])

        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()

        usage_metadata = {
            "query_expansion": expansion_metadata,
            "intent_extraction": intent_metadata,
            "representative_terms": rep_metadata,
        }
        if enable_retrieval_signals:
            usage_metadata["contextual_environment"] = context_metadata
        if shadow_metadata is not None and self.lg:
            # Shadow cost is an operational number: logged, never part of the
            # response envelope (the envelope is the /v1 contract).
            self.lg.log_struct(
                message="Temporal shadow usage",
                structured_data={
                    **self.base_log,
                    "query": query[:200],
                    "shadow_contextual_environment": shadow_metadata,
                },
                severity="INFO",
            )

        return {
            "original_query": query,
            "expanded_query": expanded_query,
            # "abbreviations_expanded": expansion.get("abbreviations_expanded", []),
            "representative_terms": rep_terms_result.get("representative_terms", []),
            "total_intents_detected": len(v2_intents),
            "intents": v2_intents,
            "record_type_matches": record_type_matches,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "processing_time_seconds": processing_time,
            "usage_metadata": usage_metadata,
        }

    def run_v2(
        self,
        query: str,
        model_name: str | None = None,
        generation_config: dict[str, Any] | None = None,
        enable_retrieval_signals: bool = False,
        location: str | None = None,
        include_record_type_matching: bool = True,
        temporal_mode: str = TEMPORAL_MODE_VOCAB_LIST,
        temporal_shadow: bool = False,
    ) -> dict[str, Any]:
        """Sync wrapper for callers that do not run inside an event loop."""
        return asyncio.run(
            self.run_v2_async(
                query,
                model_name=model_name,
                generation_config=generation_config,
                enable_retrieval_signals=enable_retrieval_signals,
                location=location,
                include_record_type_matching=include_record_type_matching,
                temporal_mode=temporal_mode,
                temporal_shadow=temporal_shadow,
            )
        )

    # -----------------------
    # Canonical temporal mode helpers (/v3)
    # -----------------------
    def _canonical_temporal_slots(
        self, query: str, expanded_query: str, intents: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Data for the v3 template's CODEABLE WINDOWS section: codeable-window
        menu (from the index), query shortlist (index similarity) and learned
        examples (cadence memory). Every part is derived from data and bounded
        by settings; the vocabulary list itself is never injected."""
        st = self._temporal_settings
        index = self._temporal_index
        menu = (
            index.menu(st["menu_max_values"], units=st["menu_units"]) if index else {}
        )

        shortlist: list[str] = []
        if index is not None and st["shortlist_top_n"] > 0:
            probe = f"{query} {expanded_query}".strip()
            if index.windows_from_text(probe):
                seen: set[str] = set()
                for m in index.nearest(
                    probe, k=st["shortlist_top_n"], min_score=st["shortlist_min_score"]
                ):
                    if m.entry.name not in seen:
                        seen.add(m.entry.name)
                        shortlist.append(m.entry.name)

        examples: list[str] = []
        if self._cadence_memory is not None and len(self._cadence_memory):
            concepts: list[str] = []
            for i in intents:
                for sn in i.get("sub_natures") or []:
                    for c in sn.get("atomic_concepts") or []:
                        if isinstance(c, str) and c.strip():
                            concepts.append(c.strip())
            examples = self._cadence_memory.examples_for(
                concepts,
                per_concept=st["memory_examples_per_concept"],
                max_concepts=st["memory_max_concepts"],
                min_similarity=st["memory_min_similarity"],
            )

        return {
            "units": list(CANONICAL_UNITS),
            "menu": menu,
            "shortlist": shortlist,
            "examples": examples,
        }

    def _log_temporal_inferences(
        self, query: str, v2_intents: list[dict[str, Any]], model_name: str | None
    ) -> None:
        """One structured event per resolved candidate window. These events are
        the only input of the cadence-memory job and of the temporal evaluation
        replays; they carry everything needed to audit an inference."""
        if not self.lg:
            return
        candidates = 0
        defaulted: list[str] = []
        for intent in v2_intents:
            for fc in intent.get("final_candidates") or []:
                temporal = (fc.get("retrieval_signals") or {}).get("temporal") or []
                candidates += 1
                if temporal and all(t.get("resolution") == "default" for t in temporal):
                    defaulted.append(str(fc.get("candidate")))
                for t in temporal:
                    window = t.get("window")
                    self.lg.log_struct(
                        message="Temporal inference",
                        structured_data={
                            **self.base_log,
                            "temporal_mode": TEMPORAL_MODE_CANONICAL,
                            "model": model_name or self.model_name,
                            "query": query[:200],
                            "intent_title": intent.get("intent_title"),
                            "candidate": fc.get("candidate"),
                            "nature": intent.get("nature"),
                            "basis": t.get("basis"),
                            "rationale": t.get("rationale"),
                            "window": window,
                            "window_label": window_label(window),
                            "resolution": t.get("resolution"),
                            "time_window": t.get("time_window"),
                            "cui": t.get("codes"),
                            "formula": t.get("formula"),
                            "index_version": (
                                self._temporal_index.vocab_hash
                                if self._temporal_index
                                else None
                            ),
                            "memory_version": (
                                self._cadence_memory.version
                                if self._cadence_memory
                                else None
                            ),
                        },
                        severity="INFO",
                    )
        if defaulted:
            # Why a candidate is on the default is diagnosable from the
            # index coverage and the entry events above; the summary makes
            # the condition visible without reading every event.
            coverage = self._temporal_index.coverage() if self._temporal_index else {}
            self.lg.log_struct(
                message="Temporal default applied",
                structured_data={
                    **self.base_log,
                    "query": query[:200],
                    "candidates": candidates,
                    "defaulted": defaulted,
                    "all_defaulted": len(defaulted) == candidates,
                    "index_keyed_windows": coverage.get("distinct_windows"),
                    "index_unkeyed_entries": coverage.get("unkeyed"),
                },
                severity="WARNING" if len(defaulted) == candidates else "INFO",
            )

    async def _temporal_shadow_compare(
        self,
        query: str,
        expanded_query: str,
        intent_result: dict[str, Any],
        canonical_intents: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        """Run the legacy vocab-list path for the same query and log, per
        candidate, whether the two paths agree on the coded window. Returns the
        shadow call's usage metadata (logged as "Temporal shadow usage") or
        None if it failed."""
        try:
            legacy_ctx = await self.build_context_async(
                query,
                expanded_query,
                intent_result,
                temporal_mode=TEMPORAL_MODE_VOCAB_LIST,
                **kwargs,
            )
        except LLMError as e:
            if self.lg:
                self.lg.log_struct(
                    message="Temporal shadow comparison skipped - legacy call failed",
                    structured_data={**self.base_log, "error": str(e)},
                    severity="WARNING",
                )
            return None
        legacy_intents = assemble_v2_intents(
            intents=intent_result.get("intents", []),
            context=legacy_ctx.get("context"),
            temporal=legacy_ctx.get("temporal"),
            vocab=self._temporal_vocab,
            resolver=self._temporal_resolver,
        )

        def _by_candidate(intents: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
            out: dict[str, dict[str, Any]] = {}
            for intent in intents:
                for fc in intent.get("final_candidates") or []:
                    temporal = (fc.get("retrieval_signals") or {}).get("temporal") or []
                    out[str(fc.get("candidate", "")).lower()] = {
                        "cuis": sorted(
                            {t.get("codes") for t in temporal if t.get("codes")}
                        ),
                        "formulas": sorted(
                            {tuple(t.get("formula") or []) for t in temporal}
                        ),
                        "names": [t.get("time_window") for t in temporal],
                    }
            return out

        a, b = _by_candidate(canonical_intents), _by_candidate(legacy_intents)
        if self.lg:
            for cand in sorted(set(a) | set(b)):
                ca, cb = a.get(cand, {}), b.get(cand, {})
                same_cui = bool(set(ca.get("cuis", [])) & set(cb.get("cuis", [])))
                same_formula = bool(
                    set(ca.get("formulas", [])) & set(cb.get("formulas", []))
                )
                self.lg.log_struct(
                    message="Temporal shadow comparison",
                    structured_data={
                        **self.base_log,
                        "query": query[:200],
                        "candidate": cand,
                        "canonical": ca,
                        "legacy": cb,
                        "agree_cui": same_cui,
                        "agree_formula": same_formula,
                        "agreement": "cui"
                        if same_cui
                        else ("formula" if same_formula else "none"),
                    },
                    severity="INFO",
                )
        return legacy_ctx.get("usage_metadata", {}) or {}
