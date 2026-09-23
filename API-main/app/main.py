import asyncio
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from aie_logging import GCPLogger, Severity
from fastapi import FastAPI, HTTPException
from starlette.middleware import Middleware

from app import config
from app.action_finder import ActionFinder
from app.exceptions import ClusterServiceError, LLMError, TemporalIndexUnavailableError
from app.middleware.logging_middleware import StructuredLoggingMiddleware
from app.mod_intent_extraction import (
    TEMPORAL_MODE_CANONICAL,
    TEMPORAL_MODE_VOCAB_LIST,
    ContextualIntentPipeline,
)
from app.mod_requests import (
    ExtractIntentsResponse,
    IntentRequest,
    RetrievalSignalsRequest,
    RetrievalSignalsRequestV2,
    RetrievalSignalsResponse,
)
from app.utils import context_lonic_document_cluster as document_cluster
from app.utils.action_gate import (
    ACTION_TYPES,
    build_skip_block,
    evaluate_gate_async,
    invalid_actions,
    normalize_actions,
)
from app.utils.cadence_memory import CadenceMemory
from app.utils.concept_vocab import ConceptVocab
from app.utils.dtree_signals import (
    merge_hints,
    project_primary_temporal,
    project_query_signals,
    project_record_types_cluster,
    project_temporal_by_candidate,
)
from app.utils.location import MODEL_LOCATION_DEFAULTS, resolve_model_location
from app.utils.temporal_index import TemporalIndex
from app.utils.temporal_vocab import TemporalVocab
from app.utils.tracing import (
    FlushTracingMiddleware,
    get_tracer,
    instrument_fastapi,
    setup_tracing,
)


def validate_input_texts(texts: list[str], logger, base_log: dict) -> None:
    """
    Validate input texts for empty or whitespace-only values.

    Args:
        texts: List of input texts to validate.
        logger: Logger instance for logging validation errors.
        base_log: Base log metadata.

    Raises:
        HTTPException: 422 if texts list is empty or any text is empty/whitespace-only.
    """
    # Check for empty list
    if not texts:
        logger.log_struct(
            message="Input validation failed - no input texts provided",
            structured_data={
                **base_log,
            },
            severity="WARNING",
        )
        raise HTTPException(
            status_code=422,
            detail={
                "error": "validation_error",
                "message": "No input text provided",
            },
        )

    # Check for empty/whitespace individual texts
    for idx, text in enumerate(texts):
        if not text or not text.strip():
            logger.log_struct(
                message="Input validation failed - empty or whitespace text",
                structured_data={
                    **base_log,
                    "input_index": idx,
                    "text_value": repr(text),
                },
                severity="WARNING",
            )
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "validation_error",
                    "message": f"Input at index {idx} is empty or contains only whitespace",
                    "input_index": idx,
                },
            )


def handle_llm_exception(e: LLMError, logger, base_log: dict) -> None:
    """
    Handle LLM exceptions and raise appropriate HTTPException.

    Args:
        e: The LLM exception that was raised.
        logger: Logger instance for logging.
        base_log: Base log metadata.

    Raises:
        HTTPException: With appropriate status code based on exception type.
    """
    logger.log_struct(
        message=f"LLM error occurred: {e.error_type}",
        structured_data={
            **base_log,
            "error_type": e.error_type,
            "error_message": str(e),
            "details": e.details,
        },
        severity="ERROR",
    )

    raise HTTPException(
        status_code=e.status_code,
        detail={
            "error": e.error_type,
            "message": str(e),
            "details": e.details,
        },
    )


# Base structured metadata for every log
base_log = {
    "service_name": config.SERVICE_NAME,
    "version": config.SERVICE_VERSION,
    "environment": config.ENV,
}

# Initialize GCPLogger with base_log_metadata
lg = GCPLogger(
    gcp_project=config.PROJECT_ID,
    log_name=config.SERVICE_NAME,
    console_logging=True,
    base_log_metadata=base_log,
)

# Add middleware to app
middleware = [Middleware(StructuredLoggingMiddleware, logger=lg, severity=Severity)]

# Serve docs/openapi under the same prefix as the routes so they're reachable
# through the ILB (which does not strip API_PREFIX before forwarding).
app = FastAPI(
    middleware=middleware,
    docs_url=f"{config.API_PREFIX}/docs",
    openapi_url=f"{config.API_PREFIX}/openapi.json",
)

# Initialize tracing
tracer = None
tracing_result = setup_tracing()
if tracing_result:
    tracer, provider = tracing_result
    instrument_fastapi(app, tracer_provider=provider)
    app.add_middleware(FlushTracingMiddleware, tracer_provider=provider)

# Get tracer for custom spans
custom_tracer = get_tracer() if tracer else None

# Load temporal vocab for v2 retrieval signals (optional).
# Prefer GCS when bucket is set; on GCS failure (or when bucket unset),
# fall back to a local file. Failure of both leaves vocab=None so v1 is
# unaffected and v2 uses temporal defaults.
temporal_vocab = None
try:
    if config.TEMPORAL_GCS_BUCKET and config.TEMPORAL_GCS_PATH:
        try:
            temporal_vocab = TemporalVocab.from_gcs(
                config.PROJECT_ID,
                config.TEMPORAL_GCS_BUCKET,
                config.TEMPORAL_GCS_PATH,
            )
            lg.log_struct(
                message="Temporal vocabulary loaded from GCS",
                structured_data={
                    **base_log,
                    "bucket": config.TEMPORAL_GCS_BUCKET,
                    "path": config.TEMPORAL_GCS_PATH,
                    "term_count": len(temporal_vocab._names),
                },
                severity="INFO",
            )
        except Exception as gcs_err:
            lg.log_struct(
                message="Failed to load temporal vocabulary from GCS; trying local file",
                structured_data={
                    **base_log,
                    "bucket": config.TEMPORAL_GCS_BUCKET,
                    "path": config.TEMPORAL_GCS_PATH,
                    "error": str(gcs_err),
                    "error_type": type(gcs_err).__name__,
                    "local_path": config.TEMPORAL_LOCAL_PATH or None,
                },
                severity="WARNING",
            )
            temporal_vocab = None

    if temporal_vocab is None and config.TEMPORAL_LOCAL_PATH:
        temporal_vocab = TemporalVocab.from_file(config.TEMPORAL_LOCAL_PATH)
        lg.log_struct(
            message="Temporal vocabulary loaded from local file",
            structured_data={
                **base_log,
                "path": config.TEMPORAL_LOCAL_PATH,
                "term_count": len(temporal_vocab._names),
            },
            severity="INFO",
        )
except Exception as e:
    lg.log_struct(
        message="Failed to load temporal vocabulary; v2 will use fallbacks",
        structured_data={
            **base_log,
            "bucket": config.TEMPORAL_GCS_BUCKET or None,
            "gcs_path": config.TEMPORAL_GCS_PATH or None,
            "local_path": config.TEMPORAL_LOCAL_PATH or None,
            "error": str(e),
            "error_type": type(e).__name__,
        },
        severity="ERROR",
    )
    temporal_vocab = None

# Load record-type vocab for /v1/retrieval-signals (optional).
# Same GCS-first / local-file fallback pattern as the temporal vocabulary.
# Failure leaves vocab=None; record types then return names with empty coding.
record_type_vocab = None
try:
    if config.RECORD_TYPE_GCS_BUCKET and config.RECORD_TYPE_GCS_PATH:
        try:
            record_type_vocab = ConceptVocab.from_gcs(
                config.PROJECT_ID,
                config.RECORD_TYPE_GCS_BUCKET,
                config.RECORD_TYPE_GCS_PATH,
            )
            lg.log_struct(
                message="Record-type vocabulary loaded from GCS",
                structured_data={
                    **base_log,
                    "bucket": config.RECORD_TYPE_GCS_BUCKET,
                    "path": config.RECORD_TYPE_GCS_PATH,
                    "term_count": len(record_type_vocab),
                },
                severity="INFO",
            )
        except Exception as gcs_err:
            lg.log_struct(
                message="Failed to load record-type vocabulary from GCS; trying local file",
                structured_data={
                    **base_log,
                    "bucket": config.RECORD_TYPE_GCS_BUCKET,
                    "path": config.RECORD_TYPE_GCS_PATH,
                    "error": str(gcs_err),
                    "error_type": type(gcs_err).__name__,
                    "local_path": config.RECORD_TYPE_LOCAL_PATH or None,
                },
                severity="WARNING",
            )
            record_type_vocab = None

    if record_type_vocab is None and config.RECORD_TYPE_LOCAL_PATH:
        record_type_vocab = ConceptVocab.from_file(config.RECORD_TYPE_LOCAL_PATH)
        lg.log_struct(
            message="Record-type vocabulary loaded from local file",
            structured_data={
                **base_log,
                "path": config.RECORD_TYPE_LOCAL_PATH,
                "term_count": len(record_type_vocab),
            },
            severity="INFO",
        )
except Exception as e:
    lg.log_struct(
        message="Record-type vocabulary unavailable; record types will have empty coding",
        structured_data={
            **base_log,
            "error": str(e),
            "error_type": type(e).__name__,
        },
        severity="ERROR",
    )
    record_type_vocab = None

# A file that parses but yields no usable entries (wrong shape, e.g. values that
# are not lists of {"cui": ...}) would otherwise load "successfully" and leave
# every record type uncoded for the life of the process. Treat it as a failure.
if record_type_vocab is not None:
    if len(record_type_vocab) == 0:
        lg.log_struct(
            message="Record-type vocabulary loaded but contains no usable entries; "
            'check file shape { name: [ {"cui": ...} ] }',
            structured_data={
                **base_log,
                "gcs_path": config.RECORD_TYPE_GCS_PATH or None,
                "local_path": config.RECORD_TYPE_LOCAL_PATH or None,
                "skipped_names": record_type_vocab.skipped_names,
            },
            severity="ERROR",
        )
        record_type_vocab = None
    elif record_type_vocab.skipped_names or record_type_vocab.skipped_codes:
        lg.log_struct(
            message="Record-type vocabulary loaded with malformed entries dropped",
            structured_data={
                **base_log,
                "term_count": len(record_type_vocab),
                "skipped_names": record_type_vocab.skipped_names,
                "skipped_codes": record_type_vocab.skipped_codes,
            },
            severity="WARNING",
        )


# /v3/retrieval-signals — temporal index, built from the loaded temporal
# vocabulary (formula index + name similarity). Nothing is pasted into prompts
# any more; a failure here leaves /v3 unavailable (503) and /v1, /v2 untouched.
temporal_index = None
if temporal_vocab is not None:
    try:
        temporal_index = TemporalIndex.from_vocab(
            temporal_vocab,
            similarity_provider=config.TEMPORAL_INDEX_SIMILARITY_PROVIDER,
            embedding_model=config.TEMPORAL_EMBEDDING_MODEL,
            project=config.PROJECT_ID,
            location=config.LOCATION,
            cache_dir=config.TEMPORAL_INDEX_CACHE_DIR or None,
        )
        lg.log_struct(
            message="Temporal index built for /v3/retrieval-signals",
            structured_data={**base_log, **temporal_index.coverage()},
            severity="INFO",
        )
    except Exception as e:
        lg.log_struct(
            message="Temporal index unavailable; /v3/retrieval-signals will answer 503",
            structured_data={
                **base_log,
                "error": str(e),
                "error_type": type(e).__name__,
            },
            severity="ERROR",
        )
        temporal_index = None
else:
    lg.log_struct(
        message="No temporal vocabulary loaded; /v3/retrieval-signals will answer 503",
        structured_data={**base_log},
        severity="WARNING",
    )

# /v3 — cadence memory: the service's own inferred windows, aggregated from
# its logs by scripts/build_cadence_memory.py. Optional; GCS first, then local.
cadence_memory = CadenceMemory.empty()
try:
    loaded_from = None
    if config.CADENCE_MEMORY_GCS_BUCKET and config.CADENCE_MEMORY_GCS_PATH:
        try:
            cadence_memory = CadenceMemory.from_gcs(
                config.PROJECT_ID,
                config.CADENCE_MEMORY_GCS_BUCKET,
                config.CADENCE_MEMORY_GCS_PATH,
            )
            loaded_from = f"gs://{config.CADENCE_MEMORY_GCS_BUCKET}/{config.CADENCE_MEMORY_GCS_PATH}"
        except Exception as gcs_err:
            lg.log_struct(
                message="Failed to load cadence memory from GCS; trying local file",
                structured_data={**base_log, "error": str(gcs_err)},
                severity="WARNING",
            )
    if loaded_from is None and config.CADENCE_MEMORY_LOCAL_PATH:
        import os as _os

        if _os.path.exists(config.CADENCE_MEMORY_LOCAL_PATH):
            cadence_memory = CadenceMemory.from_file(config.CADENCE_MEMORY_LOCAL_PATH)
            loaded_from = config.CADENCE_MEMORY_LOCAL_PATH
    lg.log_struct(
        message="Cadence memory loaded"
        if loaded_from
        else "No cadence memory file; /v3 runs without learned examples",
        structured_data={
            **base_log,
            "source": loaded_from,
            "version": cadence_memory.version,
            "concepts": len(cadence_memory),
        },
        severity="INFO",
    )
except Exception as e:
    lg.log_struct(
        message="Cadence memory unavailable; /v3 runs without learned examples",
        structured_data={**base_log, "error": str(e), "error_type": type(e).__name__},
        severity="WARNING",
    )
    cadence_memory = CadenceMemory.empty()

pipeline = ContextualIntentPipeline(
    project=config.PROJECT_ID,
    location=config.LOCATION,
    model=config.MODEL_VERSION,
    logger=lg,
    tracer=custom_tracer,
    temporal_vocab=temporal_vocab,
    record_type_vocab=record_type_vocab,
    temporal_index=temporal_index,
    cadence_memory=cadence_memory,
    temporal_settings={
        "max_codes": config.TEMPORAL_MAX_CODES_PER_WINDOW,
        "min_similarity": config.TEMPORAL_MIN_SIMILARITY,
        "menu_max_values": config.TEMPORAL_MENU_MAX_VALUES,
        "menu_units": config.TEMPORAL_MENU_UNITS,
        "shortlist_top_n": config.TEMPORAL_SHORTLIST_TOP_N,
        "shortlist_min_score": config.TEMPORAL_SHORTLIST_MIN_SCORE,
        "memory_examples_per_concept": config.CADENCE_MEMORY_EXAMPLES_PER_CONCEPT,
        "memory_max_concepts": config.CADENCE_MEMORY_MAX_CONCEPTS,
        "memory_min_similarity": config.CADENCE_MEMORY_MIN_SIMILARITY,
    },
)

# /v2/retrieval-signals — Action Finder gate. Runs through the pipeline's model
# client (same allow-list, location routing, credentials and error mapping) as
# a direct in-process call: never an HTTP hop to a sibling endpoint.
action_finder = ActionFinder(pipeline, logger=lg)

# /v2/retrieval-signals — document-cluster record-type resolution. Nothing is
# loaded at startup (lookups are per request); only the configuration is
# checked so a missing URL/dataset is visible in the logs before the first 503.
if document_cluster.is_configured():
    lg.log_struct(
        message="Document-cluster record-type resolution configured for /v2/retrieval-signals",
        structured_data={
            **base_log,
            "cluster_selection_url": config.CLUSTER_SELECTION_URL,
            "cluster_set": config.CLUSTER_SET,
            "bq_project": config.CLUSTER_BQ_PROJECT,
            "bq_dataset": config.CLUSTER_BQ_DATASET,
        },
        severity="INFO",
    )
else:
    lg.log_struct(
        message="Document-cluster service not configured; /v2/retrieval-signals will answer 503",
        structured_data={
            **base_log,
            "missing_settings": document_cluster.missing_configuration(),
        },
        severity="WARNING",
    )


def _aggregate_usage_metadata(
    all_usage_metadata: list[dict[str, Any]], step_keys: list[str]
) -> dict[str, dict[str, int]]:
    """Sum token counts per pipeline step across inputs."""
    aggregated: dict[str, dict[str, int]] = {}
    for step in step_keys:
        aggregated[step] = {
            "prompt_token_count": sum(
                m.get(step, {}).get("prompt_token_count", 0) for m in all_usage_metadata
            ),
            "candidates_token_count": sum(
                m.get(step, {}).get("candidates_token_count", 0)
                for m in all_usage_metadata
            ),
            "total_token_count": sum(
                m.get(step, {}).get("total_token_count", 0) for m in all_usage_metadata
            ),
            "thinking_token_count": sum(
                m.get(step, {}).get("thinking_token_count", 0)
                for m in all_usage_metadata
            ),
        }
    return aggregated


def _process_nature_breakdown(
    request: IntentRequest,
    run_fn: Callable[..., dict[str, Any]],
    usage_step_keys: list[str],
    completed_inputs_key: str = "is_clinical",
) -> dict[str, Any]:
    """Shared nature-breakdown loop: validate, run pipeline, aggregate envelope."""
    texts: list[str] = request.texts

    # Validate input texts (raises 422 if invalid)
    validate_input_texts(texts, lg, base_log)

    # Get model and generation config from request (or use defaults)
    model_name = request.model_name or config.MODEL_VERSION
    generation_config = (
        request.generation_config.model_dump(exclude_none=True)
        if request.generation_config
        else None
    )

    # Resolve location: explicit request value > model-family map > configured
    # GCP_LOCATION > DEFAULT_LOCATION fallback.
    _, resolved_location = resolve_model_location(
        model=model_name,
        location=request.location,
        config_map=MODEL_LOCATION_DEFAULTS,
        config_location=config.LOCATION,
        logger=lg,
    )
    print("----------------------------------------------")
    print(f"Resolved model location: {resolved_location} for model: {model_name}")

    lg.log_struct(
        message="API input received",
        structured_data={
            **base_log,
            "model_name": model_name,
            "generation_config": generation_config,
            "location": resolved_location,
            "input_count": len(texts),
            "source": getattr(request, "source", None),
            "timestamp": getattr(request, "timestamp", None),
        },
        severity="INFO",
    )

    results = []
    processing_times = []
    all_usage_metadata = []

    try:
        # Span: process_inputs
        if custom_tracer:
            with custom_tracer.start_as_current_span("process_inputs") as span:
                span.set_attribute("total_inputs", len(texts))
                span.set_attribute("model_used", model_name)

                for idx, text in enumerate(texts):
                    # Span: process_single_input
                    with custom_tracer.start_as_current_span(
                        "process_single_input"
                    ) as input_span:
                        input_span.set_attribute("input_index", idx)
                        input_span.set_attribute("text_length", len(text))
                        input_span.set_attribute(
                            "text_snippet", text[:100] if len(text) > 100 else text
                        )

                        lg.log_struct(
                            message="Processing input text",
                            structured_data={
                                **base_log,
                                "input_index": idx,
                                "text_snippet": text[:100] if len(text) > 100 else text,
                            },
                            severity="INFO",
                        )

                        output = run_fn(
                            text,
                            model_name=model_name,
                            generation_config=generation_config,
                            location=resolved_location,
                        )
                        processing_time = output.pop("processing_time_seconds", 0)
                        processing_times.append(processing_time)

                        # Extract usage_metadata before removing timestamp
                        usage_metadata = output.pop("usage_metadata", {})
                        if usage_metadata:
                            all_usage_metadata.append(usage_metadata)

                        output.pop("timestamp", None)

                        results.append({"input_index": idx, **output})
        else:
            # No tracing - process normally
            for idx, text in enumerate(texts):
                lg.log_struct(
                    message="Processing input text",
                    structured_data={
                        **base_log,
                        "input_index": idx,
                        "text_snippet": text[:100] if len(text) > 100 else text,
                    },
                    severity="INFO",
                )

                output = run_fn(
                    text,
                    model_name=model_name,
                    generation_config=generation_config,
                    location=resolved_location,
                )
                processing_time = output.pop("processing_time_seconds", 0)
                processing_times.append(processing_time)

                # Extract usage_metadata before removing timestamp
                usage_metadata = output.pop("usage_metadata", {})
                if usage_metadata:
                    all_usage_metadata.append(usage_metadata)

                output.pop("timestamp", None)

                results.append({"input_index": idx, **output})
    except LLMError as e:
        handle_llm_exception(e, lg, base_log)

    if completed_inputs_key == "is_clinical":
        completed_count = sum(1 for r in results if r.get("is_clinical"))
        completed_log_key = "clinical_inputs"
    else:
        completed_count = sum(1 for r in results if r.get("intents"))
        completed_log_key = "inputs_with_intents"

    lg.log_struct(
        message="API request completed",
        structured_data={
            **base_log,
            "total_inputs": len(results),
            completed_log_key: completed_count,
        },
        severity="INFO",
    )

    # Aggregate timing from all outputs
    total_llm_seconds = sum(processing_times)
    max_llm_seconds = max(processing_times) if processing_times else 0
    max_llm_seconds_input_index = (
        processing_times.index(max_llm_seconds) if processing_times else None
    )

    # Aggregate usage metadata separately for query expansion and intent extraction
    aggregated_usage_metadata = _aggregate_usage_metadata(
        all_usage_metadata, usage_step_keys
    )

    # Check if any output has an error
    has_error = any("error" in r for r in results)
    status = 0 if has_error else 1

    return {
        "status": status,
        "output": {
            "total_inputs": len(results),
            "results": results,
        },
        "details": {
            "timing": {
                "total_llm_seconds": total_llm_seconds,
                "max_llm_seconds": max_llm_seconds,
                "max_llm_seconds_input_index": max_llm_seconds_input_index,
            },
            "usage_metadata": aggregated_usage_metadata,
            "version": config.SERVICE_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": getattr(request, "source", None),
            "model": model_name,
            "location": resolved_location,
        },
        "service": config.SERVICE_ID,
    }


@app.post(
    f"{config.API_PREFIX}/v1/nature-breakdown",
    summary="Extract clinical intents with full nature breakdown",
    description="Accepts one or multiple text inputs and returns structured clinical intents with nature, sub_natures, and final_queries.",
    response_model=ExtractIntentsResponse,
)
def nature_breakdown(request: IntentRequest):
    """Extracts the intents from the query (v1 pipeline)."""
    return _process_nature_breakdown(
        request,
        run_fn=pipeline.run,
        usage_step_keys=["query_expansion", "intent_extraction"],
        completed_inputs_key="is_clinical",
    )


@app.post(
    f"{config.API_PREFIX}/v2/nature-breakdown",
    summary="Extract clinical intents with retrieval signals (v2)",
    description=(
        "Accepts one or multiple text inputs and returns structured clinical intents "
        "with nature breakdown, representative_terms, and optional final_candidates / "
        "retrieval_signals when enable_retrieval_signals is true."
    ),
    response_model=ExtractIntentsResponse,
)
def nature_breakdown_v2(request: IntentRequest):
    """v2 nature breakdown — always uses v2 prompts; signals gated by request flag."""
    include_signals = request.enable_retrieval_signals is True

    def run_v2_fn(
        text: str,
        model_name: str | None = None,
        generation_config: dict[str, Any] | None = None,
        location: str | None = None,
    ) -> dict[str, Any]:
        return pipeline.run_v2(
            text,
            model_name=model_name,
            generation_config=generation_config,
            enable_retrieval_signals=include_signals,
            location=location,
        )

    usage_keys = [
        "query_expansion",
        "intent_extraction",
        "representative_terms",
    ]
    if include_signals:
        usage_keys.append("contextual_environment")

    return _process_nature_breakdown(
        request,
        run_fn=run_v2_fn,
        usage_step_keys=usage_keys,
        completed_inputs_key="intents",
    )


@app.post(
    f"{config.API_PREFIX}/v1/extract-intents",
    summary="Extract clinical intents from input text(s)",
    description="Accepts one or multiple text inputs and returns structured clinical intents.",
    response_model=ExtractIntentsResponse,
)
def extract_intents(request: IntentRequest):
    """Extracts the intents from the query.
    Args:
        request: The request containing the text to extract the intents from.
    Returns:
        The intents from the query.
    Raises:
        HTTPException: 422 if input validation fails.
        HTTPException: 502/503/504 if LLM call fails.
    """
    texts: list[str] = request.texts

    # Validate input texts (raises 422 if invalid)
    validate_input_texts(texts, lg, base_log)

    # Get model and generation config from request (or use defaults)
    model_name = request.model_name or config.MODEL_VERSION
    generation_config = (
        request.generation_config.model_dump(exclude_none=True)
        if request.generation_config
        else None
    )

    # Resolve location: explicit request value > model-family map > configured
    # GCP_LOCATION > DEFAULT_LOCATION fallback.
    _, resolved_location = resolve_model_location(
        model=model_name,
        location=request.location,
        config_map=MODEL_LOCATION_DEFAULTS,
        config_location=config.LOCATION,
        logger=lg,
    )

    lg.log_struct(
        message="API input received",
        structured_data={
            **base_log,
            "model_name": model_name,
            "generation_config": generation_config,
            "location": resolved_location,
            "input_count": len(texts),
            "source": getattr(request, "source", None),
            "timestamp": getattr(request, "timestamp", None),
        },
        severity="INFO",
    )

    results = []
    processing_times = []
    all_usage_metadata = []

    try:
        # Span: process_inputs
        if custom_tracer:
            with custom_tracer.start_as_current_span("process_inputs") as span:
                span.set_attribute("total_inputs", len(texts))
                span.set_attribute("model_used", model_name)

                for idx, text in enumerate(texts):
                    # Span: process_single_input
                    with custom_tracer.start_as_current_span(
                        "process_single_input"
                    ) as input_span:
                        input_span.set_attribute("input_index", idx)
                        input_span.set_attribute("text_length", len(text))
                        input_span.set_attribute(
                            "text_snippet", text[:100] if len(text) > 100 else text
                        )

                        lg.log_struct(
                            message="Processing input text",
                            structured_data={
                                **base_log,
                                "input_index": idx,
                                "text_snippet": text[:100] if len(text) > 100 else text,
                            },
                            severity="INFO",
                        )

                        output = pipeline.run(
                            text,
                            model_name=model_name,
                            generation_config=generation_config,
                            location=resolved_location,
                        )

                        processing_time = output.pop("processing_time_seconds", 0)
                        processing_times.append(processing_time)

                        # Extract usage_metadata before removing timestamp
                        usage_metadata = output.pop("usage_metadata", {})
                        if usage_metadata:
                            all_usage_metadata.append(usage_metadata)

                        output.pop("timestamp", None)

                        # Transform to simplified format
                        simple_output = {
                            "original_query": output.get("original_query", text),
                            "intents": [
                                {
                                    "intent": intent.get("intent_title"),
                                    "intent_description": intent.get("description"),
                                }
                                for intent in output.get("intents", [])
                            ],
                        }

                        # Preserve non-clinical fields if present
                        if not output.get("is_clinical", True):
                            simple_output["is_clinical"] = False
                            if "rejected_reason" in output:
                                simple_output["rejected_reason"] = output[
                                    "rejected_reason"
                                ]
                            if "expanded_query" in output:
                                simple_output["expanded_query"] = output[
                                    "expanded_query"
                                ]

                        result = {
                            "input_index": idx,
                            **simple_output,
                        }
                        results.append(result)
        else:
            # No tracing - process normally
            for idx, text in enumerate(texts):
                lg.log_struct(
                    message="Processing input text",
                    structured_data={
                        **base_log,
                        "input_index": idx,
                        "text_snippet": text[:100] if len(text) > 100 else text,
                    },
                    severity="INFO",
                )

                output = pipeline.run(
                    text,
                    model_name=model_name,
                    generation_config=generation_config,
                    location=resolved_location,
                )

                processing_time = output.pop("processing_time_seconds", 0)
                processing_times.append(processing_time)

                # Extract usage_metadata before removing timestamp
                usage_metadata = output.pop("usage_metadata", {})
                if usage_metadata:
                    all_usage_metadata.append(usage_metadata)

                output.pop("timestamp", None)

                # Transform to simplified format
                simple_output = {
                    "original_query": output.get("original_query", text),
                    "intents": [
                        {
                            "intent": intent.get("intent_title"),
                            "intent_description": intent.get("description"),
                        }
                        for intent in output.get("intents", [])
                    ],
                }

                # Preserve non-clinical fields if present
                if not output.get("is_clinical", True):
                    simple_output["is_clinical"] = False
                    if "rejected_reason" in output:
                        simple_output["rejected_reason"] = output["rejected_reason"]
                    if "expanded_query" in output:
                        simple_output["expanded_query"] = output["expanded_query"]

                result = {
                    "input_index": idx,
                    **simple_output,
                }
                results.append(result)
    except LLMError as e:
        handle_llm_exception(e, lg, base_log)

    lg.log_struct(
        message="API request completed",
        structured_data={
            **base_log,
            "total_inputs": len(results),
            "clinical_inputs": sum(1 for r in results if r.get("intents")),
        },
        severity="INFO",
    )

    # Aggregate timing from all outputs
    total_llm_seconds = sum(processing_times)
    max_llm_seconds = max(processing_times) if processing_times else 0
    max_llm_seconds_input_index = (
        processing_times.index(max_llm_seconds) if processing_times else None
    )

    # Aggregate usage metadata separately for query expansion and intent extraction
    aggregated_usage_metadata = {
        "query_expansion": {
            "prompt_token_count": sum(
                m.get("query_expansion", {}).get("prompt_token_count", 0)
                for m in all_usage_metadata
            ),
            "candidates_token_count": sum(
                m.get("query_expansion", {}).get("candidates_token_count", 0)
                for m in all_usage_metadata
            ),
            "total_token_count": sum(
                m.get("query_expansion", {}).get("total_token_count", 0)
                for m in all_usage_metadata
            ),
            "thinking_token_count": sum(
                m.get("query_expansion", {}).get("thinking_token_count", 0)
                for m in all_usage_metadata
            ),
        },
        "intent_extraction": {
            "prompt_token_count": sum(
                m.get("intent_extraction", {}).get("prompt_token_count", 0)
                for m in all_usage_metadata
            ),
            "candidates_token_count": sum(
                m.get("intent_extraction", {}).get("candidates_token_count", 0)
                for m in all_usage_metadata
            ),
            "total_token_count": sum(
                m.get("intent_extraction", {}).get("total_token_count", 0)
                for m in all_usage_metadata
            ),
            "thinking_token_count": sum(
                m.get("intent_extraction", {}).get("thinking_token_count", 0)
                for m in all_usage_metadata
            ),
        },
    }

    # Check if any output has an error
    has_error = any("error" in r for r in results)
    status = 0 if has_error else 1

    return {
        "status": status,
        "output": {
            "total_inputs": len(results),
            "results": results,
        },
        "details": {
            "timing": {
                "total_llm_seconds": total_llm_seconds,
                "max_llm_seconds": max_llm_seconds,
                "max_llm_seconds_input_index": max_llm_seconds_input_index,
            },
            "usage_metadata": aggregated_usage_metadata,
            "version": config.SERVICE_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": getattr(request, "source", None),
            "model": model_name,
            "location": resolved_location,
        },
        "service": config.SERVICE_ID,
    }


def validate_queries(queries, logger, base_log: dict) -> None:
    """
    Validate query objects for empty/whitespace text.
    Mirrors validate_input_texts: logs a WARNING and raises 422 with the same
    detail shape ({"error": "validation_error", "message": ...}).

    Raises:
        HTTPException: 422 if any query text is empty/whitespace.
    """
    for idx, q in enumerate(queries):
        if not q.text or not q.text.strip():
            logger.log_struct(
                message="Input validation failed - empty or whitespace query text",
                structured_data={
                    **base_log,
                    "query_index": idx,
                    "query_id": q.id,
                },
                severity="WARNING",
            )
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "validation_error",
                    "message": f"Query '{q.id}' (index {idx}) has empty or whitespace-only text",
                    "query_index": idx,
                },
            )


def normalize_retrieval_queries(queries):
    """Generate query ids while preserving each array item as one query."""
    normalized = []
    generated_index = 1
    for query in queries:
        for part in query.text:
            part = part.strip()
            query_id = f"q{generated_index}"
            normalized.append(query.model_copy(update={"id": query_id, "text": part}))
            generated_index += 1
    return normalized


@app.post(
    f"{config.API_PREFIX}/v1/retrieval-signals",
    summary="Extract normalized retrieval signals for one or more clinical queries",
    description=(
        "Accepts one or more clinical query strings and returns normalized "
        "record types, a temporal validity window, and representative tags — "
        "each with UMLS codes — per query, keyed by the client-supplied id. "
        "record_types, temporal, and tags may be supplied as optional hints: "
        "a hinted temporal overrides the derived window; hinted record_types "
        "and tags are merged ahead of derived values."
    ),
    response_model=RetrievalSignalsResponse,
)
async def dtree_retrieval_signals(request: RetrievalSignalsRequest):
    """Retrieval-signals endpoint: runs the v2 pipeline per query and projects
    the result into the dtree schema. Follows the same flow as the other
    endpoints: handler-level validation (422), LLM errors via
    handle_llm_exception, and the standard response envelope.
    Args:
        request: The request containing one or more query objects.
    Returns:
        Envelope with output.queries — one retrieval-signals block per id.
    Raises:
        HTTPException: 422 if input validation fails.
        HTTPException: 429/502/503/504 if an LLM call fails (via handle_llm_exception).
    """
    # Generate ids and normalize each array item before validation.
    queries = normalize_retrieval_queries(request.queries)

    # Validate query objects (raises 422 if invalid)
    validate_queries(queries, lg, base_log)

    # Get model and generation config from request (or use defaults)
    model_name = request.model_name or config.MODEL_VERSION
    generation_config = (
        request.generation_config.model_dump(exclude_none=True)
        if request.generation_config
        else None
    )

    # Resolve location: explicit request value > model-family map > configured
    # GCP_LOCATION > DEFAULT_LOCATION fallback.
    _, resolved_location = resolve_model_location(
        model=model_name,
        location=request.location,
        config_map=MODEL_LOCATION_DEFAULTS,
        config_location=config.LOCATION,
        logger=lg,
    )

    lg.log_struct(
        message="API input received",
        structured_data={
            **base_log,
            "model_name": model_name,
            "generation_config": generation_config,
            "location": resolved_location,
            "input_count": len(queries),
            "source": getattr(request, "source", None),
            "timestamp": getattr(request, "timestamp", None),
        },
        severity="INFO",
    )

    query_results = []
    processing_times = []
    all_usage_metadata = []

    async def _process_query(q) -> None:
        """Run the v2 pipeline for one query and append its projected block."""
        v2_result = await pipeline.run_v2_async(
            q.text,
            model_name=model_name,
            generation_config=generation_config,
            enable_retrieval_signals=True,
            location=resolved_location,
        )

        processing_time = v2_result.pop("processing_time_seconds", 0)
        processing_times.append(processing_time)

        # Extract usage_metadata before removing timestamp
        usage_metadata = v2_result.pop("usage_metadata", {})
        if usage_metadata:
            all_usage_metadata.append(usage_metadata)

        v2_result.pop("timestamp", None)

        block = project_query_signals(
            query_id=q.id,
            text=q.text,
            v2_result=v2_result,
            record_type_vocab=record_type_vocab,
            temporal_vocab=temporal_vocab,
            tag_vocab=None,  # no tag vocabulary provisioned yet
            hint_record_types=(
                [rt.model_dump() for rt in q.record_types] if q.record_types else None
            ),
            hint_temporal=q.temporal.model_dump() if q.temporal else None,
            hint_tags=[t.model_dump() for t in q.tags] if q.tags else None,
        )

        # Vocabulary coverage. A record type with empty coding is a label the
        # vocabulary could not resolve; without this the only symptom is a
        # missing code in the response and there is no way to learn which
        # labels to add. Logged per query so coverage can be tracked over time
        # and the vocabulary grown from real traffic rather than guesswork.
        # Skipped when no vocabulary is loaded — that failure is already logged
        # at startup and every name would trivially be unresolved.
        if record_type_vocab is not None or temporal_vocab is not None:
            record_types = block.get("record_types") or []
            unresolved = [
                rt.get("name", "") for rt in record_types if not rt.get("coding")
            ]
            temporal_block = block.get("temporal") or {}
            temporal_name = temporal_block.get("name")
            lg.log_struct(
                message="Vocabulary coverage",
                structured_data={
                    **base_log,
                    "query_id": q.id,
                    "record_types_total": len(record_types),
                    "record_types_coded": len(record_types) - len(unresolved),
                    "unresolved_record_types": unresolved,
                    "record_type_vocab_terms": (
                        len(record_type_vocab)
                        if record_type_vocab is not None
                        else None
                    ),
                    # Temporal now emits the model's own signal uncoded when the
                    # vocabulary cannot code it, instead of a fabricated window.
                    # An uncoded name here is the term to add to the temporal
                    # vocabulary; a null name means no signal was produced.
                    "temporal_name": temporal_name,
                    "temporal_coded": bool(temporal_block.get("coding")),
                    "unresolved_temporal": (
                        temporal_name
                        if temporal_name and not temporal_block.get("coding")
                        else None
                    ),
                },
                severity="INFO",
            )

        query_results.append(block)

    try:
        # Span: process_inputs
        if custom_tracer:
            with custom_tracer.start_as_current_span("process_inputs") as span:
                span.set_attribute("total_inputs", len(queries))
                span.set_attribute("model_used", model_name)

                for idx, q in enumerate(queries):
                    # Span: process_single_input
                    with custom_tracer.start_as_current_span(
                        "process_single_input"
                    ) as input_span:
                        input_span.set_attribute("input_index", idx)
                        input_span.set_attribute("query_id", q.id)
                        input_span.set_attribute("text_length", len(q.text))
                        input_span.set_attribute(
                            "text_snippet",
                            q.text[:100] if len(q.text) > 100 else q.text,
                        )

                        lg.log_struct(
                            message="Processing input text",
                            structured_data={
                                **base_log,
                                "input_index": idx,
                                "query_id": q.id,
                                "text_snippet": q.text[:100]
                                if len(q.text) > 100
                                else q.text,
                            },
                            severity="INFO",
                        )

                        await _process_query(q)
        else:
            # No tracing - process normally
            for idx, q in enumerate(queries):
                lg.log_struct(
                    message="Processing input text",
                    structured_data={
                        **base_log,
                        "input_index": idx,
                        "query_id": q.id,
                        "text_snippet": q.text[:100] if len(q.text) > 100 else q.text,
                    },
                    severity="INFO",
                )

                await _process_query(q)
    except LLMError as e:
        handle_llm_exception(e, lg, base_log)

    lg.log_struct(
        message="API request completed",
        structured_data={
            **base_log,
            "total_inputs": len(query_results),
            "queries_with_signals": sum(
                1
                for r in query_results
                if r.get("record_types") or r.get("tags") or r.get("temporal")
            ),
        },
        severity="INFO",
    )

    # Aggregate timing from all outputs
    total_llm_seconds = sum(processing_times)
    max_llm_seconds = max(processing_times) if processing_times else 0
    max_llm_seconds_input_index = (
        processing_times.index(max_llm_seconds) if processing_times else None
    )

    # Aggregate usage metadata for all v2 pipeline steps
    aggregated_usage_metadata = _aggregate_usage_metadata(
        all_usage_metadata,
        [
            "query_expansion",
            "intent_extraction",
            "representative_terms",
            "contextual_environment",
        ],
    )

    # Check if any output has an error
    has_error = any("error" in r for r in query_results)
    status = 0 if has_error else 1

    return {
        "status": status,
        "output": {
            "total_queries": len(query_results),
            "queries": query_results,
        },
        "details": {
            "timing": {
                "total_llm_seconds": total_llm_seconds,
                "max_llm_seconds": max_llm_seconds,
                "max_llm_seconds_input_index": max_llm_seconds_input_index,
            },
            "usage_metadata": aggregated_usage_metadata,
            "version": config.SERVICE_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": getattr(request, "source", None),
            "model": model_name,
            "location": resolved_location,
        },
        "service": config.SERVICE_ID,
    }


# ---------------------------------------------------------------------------
# /v2/retrieval-signals
#
# Same request/response contract as /v1 with two additions:
#   1. Action Finder gate: optional `actions` (+ `enable_action_finder`). Each
#      query is classified in-process before any pipeline work; non-matching
#      queries get a skip block instead of signals.
#   2. Record types are coded through the document-cluster service (cluster
#      selection + BigQuery), not the record_type_name_to_cui.json vocabulary.
#      The contextual-environment prompt therefore runs without the KNOWN
#      RECORD TYPES block (include_record_type_matching=False).
# /v1 is untouched.
# ---------------------------------------------------------------------------
def validate_actions(actions, logger, base_log: dict) -> None:
    """Reject action names outside the Action Finder taxonomy (422).

    A typo would otherwise never match and every query would be skipped, which
    looks exactly like a correct gate.
    """
    bad = invalid_actions(actions)
    if bad:
        logger.log_struct(
            message="Input validation failed - unknown action type(s)",
            structured_data={**base_log, "invalid_actions": bad},
            severity="WARNING",
        )
        raise HTTPException(
            status_code=422,
            detail={
                "error": "validation_error",
                "message": f"Unknown action type(s): {bad}",
                "invalid_actions": bad,
            },
        )


@app.get(
    f"{config.API_PREFIX}/v2/retrieval-signals/action-types",
    summary="List selectable Action Finder action types",
    description=(
        "Returns the action types that can be passed in `actions` to "
        "POST /v2/retrieval-signals to gate queries through the Action Finder."
    ),
)
def retrieval_signal_action_types():
    """Return the action types clients can pass when Action Finder is enabled."""
    return {
        "enable_action_finder": True,
        "actions": sorted(ACTION_TYPES),
    }


@app.post(
    f"{config.API_PREFIX}/v2/retrieval-signals",
    summary="Extract retrieval signals with Action Finder gating and document-cluster record-type CUIs",
    description=(
        "Same contract as /v1/retrieval-signals: one or more clinical query strings "
        "in, normalized record types, a temporal validity window and representative "
        "tags — each with UMLS codes — out, per query. Two differences: (1) optional "
        "`actions` gate each query through the Action Finder first; queries whose "
        "identified actions do not overlap the list (or do not include "
        "information_retrieval) return a skip block instead of signals. (2) Record "
        "types are resolved to CUIs through the document-cluster service instead of "
        "the local record-type vocabulary. record_types, temporal, and tags may be "
        "supplied as optional hints exactly as in /v1."
    ),
    response_model=RetrievalSignalsResponse,
)
async def dtree_retrieval_signals_v2(request: RetrievalSignalsRequestV2):
    """Retrieval-signals v2 endpoint. Mirrors /v1 end to end (handler-level
    validation, LLM errors via handle_llm_exception, standard envelope) and adds
    the Action Finder gate and document-cluster record-type coding.
    Args:
        request: The request containing one or more query objects, optionally
            with `actions` / `enable_action_finder`.
    Returns:
        Envelope with output.queries — one retrieval-signals block (or skip
        block) per id.
    Raises:
        HTTPException: 422 if input validation fails (including unknown actions).
        HTTPException: 429/502/503/504 if an LLM call fails (via handle_llm_exception).
        HTTPException: 503 cluster_service_unavailable if the document-cluster
            dependency is not configured or fails.
    """
    return await _run_retrieval_signals(request, temporal_mode=TEMPORAL_MODE_VOCAB_LIST)


@app.post(
    f"{config.API_PREFIX}/v3/retrieval-signals",
    summary="Extract retrieval signals with canonical temporal windows (indexed vocabulary, per-candidate windows)",
    description=(
        "Same contract as /v2/retrieval-signals (Action Finder gate, document-cluster "
        "record types, hints, envelope) with the temporal step in canonical mode: the "
        "temporal vocabulary is never sent to the model; the model emits each window as "
        "a structured span that is resolved against the in-process vocabulary index. "
        "`temporal` stays the single primary window (the query's explicit span, else the "
        "widest inferred span); `temporal_by_candidate` adds one window per candidate "
        "concept with `basis` (explicit | inferred | default) and `rationale`. "
        "`details.temporal_mode` reports the path."
    ),
    response_model=RetrievalSignalsResponse,
)
async def dtree_retrieval_signals_v3(request: RetrievalSignalsRequestV2):
    """Retrieval-signals v3: /v2 plus canonical temporal mode.
    Raises:
        HTTPException: as /v2, plus 503 temporal_index_unavailable when the
            temporal vocabulary could not be indexed at startup.
    """
    if pipeline._canonical_resolver is None:
        handle_llm_exception(
            TemporalIndexUnavailableError(
                "Temporal index is not available; /v3 cannot run canonical temporal mode",
                {"temporal_mode": TEMPORAL_MODE_CANONICAL},
            ),
            lg,
            base_log,
        )
    return await _run_retrieval_signals(
        request,
        temporal_mode=TEMPORAL_MODE_CANONICAL,
        temporal_shadow=config.TEMPORAL_SHADOW_MODE,
    )


async def _run_retrieval_signals(
    request: RetrievalSignalsRequestV2,
    temporal_mode: str,
    temporal_shadow: bool = False,
):
    """Shared /v2 and /v3 handler. /v2 = vocab_list temporal mode (response
    byte-for-byte as before); /v3 = canonical mode (+ temporal_by_candidate,
    details.temporal_mode)."""
    canonical = temporal_mode == TEMPORAL_MODE_CANONICAL
    # Generate ids and normalize each array item before validation.
    queries = normalize_retrieval_queries(request.queries)

    # Validate query objects (raises 422 if invalid)
    validate_queries(queries, lg, base_log)

    # Optional action gate. Omitted / empty list == not provided (no gating).
    expected_actions = normalize_actions(request.actions)
    action_gate_enabled = request.enable_action_finder and bool(expected_actions)
    if action_gate_enabled:
        validate_actions(request.actions, lg, base_log)

    # Get model and generation config from request (or use defaults)
    model_name = request.model_name or config.MODEL_VERSION
    generation_config = (
        request.generation_config.model_dump(exclude_none=True)
        if request.generation_config
        else None
    )

    # Resolve location: explicit request value > model-family map > configured
    # GCP_LOCATION > DEFAULT_LOCATION fallback.
    _, resolved_location = resolve_model_location(
        model=model_name,
        location=request.location,
        config_map=MODEL_LOCATION_DEFAULTS,
        config_location=config.LOCATION,
        logger=lg,
    )

    lg.log_struct(
        message="API input received",
        structured_data={
            **base_log,
            "model_name": model_name,
            "generation_config": generation_config,
            "location": resolved_location,
            "input_count": len(queries),
            "expected_actions": expected_actions or None,
            "enable_action_finder": request.enable_action_finder,
            "action_gate_enabled": action_gate_enabled,
            "record_type_source": "document_cluster",
            "temporal_mode": temporal_mode,
            "source": getattr(request, "source", None),
            "timestamp": getattr(request, "timestamp", None),
        },
        severity="INFO",
    )

    # Fail fast (503) before any LLM spend when the cluster dependency is absent.
    if not document_cluster.is_configured():
        handle_llm_exception(
            ClusterServiceError(
                "Document-cluster service is not configured",
                {"missing_settings": document_cluster.missing_configuration()},
            ),
            lg,
            base_log,
        )

    query_results = []
    processing_times = []
    all_usage_metadata = []

    async def _classify(text: str) -> dict[str, Any]:
        return await action_finder.find_actions_async(
            text, model_name=model_name, location=resolved_location
        )

    async def _process_query(q) -> None:
        """Gate, run the v2 pipeline for one query and append its projected block."""
        # Gate BEFORE any pipeline work. Direct in-process call — never an HTTP
        # request to a sibling endpoint. This is one serial LLM call ahead of
        # the pipeline's own.
        if action_gate_enabled:
            decision = await evaluate_gate_async(q.text, expected_actions, _classify)
            lg.log_struct(
                message="Action gate decision",
                structured_data={
                    **base_log,
                    "query_id": q.id,
                    "matched": decision.matched,
                    "expected_actions": decision.expected_actions,
                    "identified_actions": decision.identified_actions,
                    "is_processable": decision.is_processable,
                    "action_finder_error": decision.error,
                    "failed_open": decision.failed_open,
                },
                severity="WARNING" if decision.error else "INFO",
            )
            if decision.usage_metadata:
                all_usage_metadata.append({"action_finder": decision.usage_metadata})
            if not decision.matched:
                query_results.append(build_skip_block(q.id, q.text, decision))
                return

        v2_result = await pipeline.run_v2_async(
            q.text,
            model_name=model_name,
            generation_config=generation_config,
            enable_retrieval_signals=True,
            location=resolved_location,
            # No KNOWN RECORD TYPES block / record_type_matches: record types
            # are coded through the document cluster below.
            include_record_type_matching=False,
            temporal_mode=temporal_mode,
            temporal_shadow=temporal_shadow,
        )

        processing_time = v2_result.pop("processing_time_seconds", 0)
        processing_times.append(processing_time)

        # Extract usage_metadata before removing timestamp
        usage_metadata = v2_result.pop("usage_metadata", {})
        if usage_metadata:
            all_usage_metadata.append(usage_metadata)

        v2_result.pop("timestamp", None)

        # Derive temporal + tags exactly as /v1 (no record-type vocabulary),
        # then code the record types through the document cluster. The lookup
        # is blocking I/O (HTTP + BigQuery), so it runs off the event loop.
        block = project_query_signals(
            query_id=q.id,
            text=q.text,
            v2_result=v2_result,
            record_type_vocab=None,
            temporal_vocab=temporal_vocab,
            tag_vocab=None,  # no tag vocabulary provisioned yet
        )
        if canonical:
            # /v3: explicit span first, else the widest inferred span; plus the
            # additive per-candidate list with basis / rationale.
            block["temporal"] = project_primary_temporal(v2_result, temporal_vocab)
            block["temporal_by_candidate"] = project_temporal_by_candidate(
                v2_result, config.TEMPORAL_BY_CANDIDATE_MAX
            )
        try:
            block["record_types"] = await asyncio.to_thread(
                project_record_types_cluster, v2_result, block["tags"]
            )
        except ClusterServiceError:
            raise
        except Exception as e:
            raise ClusterServiceError(
                "Document-cluster record-type resolution failed",
                {"error_type": type(e).__name__, "error": str(e)},
            ) from e

        # Client hints: same precedence as /v1 (hints first, temporal hint wins).
        block = merge_hints(
            block,
            [rt.model_dump() for rt in q.record_types] if q.record_types else None,
            q.temporal.model_dump() if q.temporal else None,
            [t.model_dump() for t in q.tags] if q.tags else None,
        )

        # Coverage. A record type with empty coding is a label the document
        # cluster could not resolve; logged per query so cluster coverage can be
        # tracked over time from real traffic. Temporal mirrors the /v1 log.
        record_types = block.get("record_types") or []
        unresolved = [rt.get("name", "") for rt in record_types if not rt.get("coding")]
        temporal_block = block.get("temporal") or {}
        temporal_name = temporal_block.get("name")
        lg.log_struct(
            message="Vocabulary coverage",
            structured_data={
                **base_log,
                "query_id": q.id,
                "temporal_mode": temporal_mode,
                "temporal_by_candidate_total": len(
                    block.get("temporal_by_candidate") or []
                ),
                "record_type_source": "document_cluster",
                "record_types_total": len(record_types),
                "record_types_coded": len(record_types) - len(unresolved),
                "unresolved_record_types": unresolved,
                "temporal_name": temporal_name,
                "temporal_coded": bool(temporal_block.get("coding")),
                "unresolved_temporal": (
                    temporal_name
                    if temporal_name and not temporal_block.get("coding")
                    else None
                ),
            },
            severity="INFO",
        )

        query_results.append(block)

    try:
        # Span: process_inputs
        if custom_tracer:
            with custom_tracer.start_as_current_span("process_inputs") as span:
                span.set_attribute("total_inputs", len(queries))
                span.set_attribute("model_used", model_name)
                span.set_attribute("action_gate_enabled", action_gate_enabled)

                for idx, q in enumerate(queries):
                    # Span: process_single_input
                    with custom_tracer.start_as_current_span(
                        "process_single_input"
                    ) as input_span:
                        input_span.set_attribute("input_index", idx)
                        input_span.set_attribute("query_id", q.id)
                        input_span.set_attribute("text_length", len(q.text))
                        input_span.set_attribute(
                            "text_snippet",
                            q.text[:100] if len(q.text) > 100 else q.text,
                        )

                        lg.log_struct(
                            message="Processing input text",
                            structured_data={
                                **base_log,
                                "input_index": idx,
                                "query_id": q.id,
                                "text_snippet": q.text[:100]
                                if len(q.text) > 100
                                else q.text,
                            },
                            severity="INFO",
                        )

                        await _process_query(q)
        else:
            # No tracing - process normally
            for idx, q in enumerate(queries):
                lg.log_struct(
                    message="Processing input text",
                    structured_data={
                        **base_log,
                        "input_index": idx,
                        "query_id": q.id,
                        "text_snippet": q.text[:100] if len(q.text) > 100 else q.text,
                    },
                    severity="INFO",
                )

                await _process_query(q)
    except LLMError as e:
        # ClusterServiceError is an LLMError: same handler, 503 envelope.
        handle_llm_exception(e, lg, base_log)

    lg.log_struct(
        message="API request completed",
        structured_data={
            **base_log,
            "total_inputs": len(query_results),
            "queries_with_signals": sum(
                1
                for r in query_results
                if r.get("record_types") or r.get("tags") or r.get("temporal")
            ),
            "queries_skipped": sum(1 for r in query_results if r.get("skipped")),
        },
        severity="INFO",
    )

    # Aggregate timing from all outputs
    total_llm_seconds = sum(processing_times)
    max_llm_seconds = max(processing_times) if processing_times else 0
    max_llm_seconds_input_index = (
        processing_times.index(max_llm_seconds) if processing_times else None
    )

    # Aggregate usage metadata for all v2 pipeline steps (+ the gate)
    usage_steps = [
        "action_finder",
        "query_expansion",
        "intent_extraction",
        "representative_terms",
        "contextual_environment",
    ]
    if canonical and temporal_shadow:
        usage_steps.append("shadow_contextual_environment")
    aggregated_usage_metadata = _aggregate_usage_metadata(
        all_usage_metadata, usage_steps
    )

    # Check if any output has an error
    has_error = any("error" in r for r in query_results)
    status = 0 if has_error else 1

    details = {
        "timing": {
            "total_llm_seconds": total_llm_seconds,
            "max_llm_seconds": max_llm_seconds,
            "max_llm_seconds_input_index": max_llm_seconds_input_index,
        },
        "usage_metadata": aggregated_usage_metadata,
        "version": config.SERVICE_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": getattr(request, "source", None),
        "model": model_name,
        "location": resolved_location,
    }
    if canonical:
        details["temporal_mode"] = temporal_mode
        details["temporal_shadow"] = temporal_shadow

    return {
        "status": status,
        "output": {
            "total_queries": len(query_results),
            "queries": query_results,
        },
        "details": details,
        "service": config.SERVICE_ID,
    }
