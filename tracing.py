import logging
import os

from aie_logging import GCPLogger, Severity
from fastapi import Request
from opentelemetry import trace
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import ALWAYS_ON, ParentBased
from opentelemetry.trace import Span
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class ServiceNameSpanProcessor(SpanProcessor):
    """Span processor that adds service.name and service.version to all spans."""

    def __init__(self, service_name: str, service_version: str):
        self.service_name = service_name
        self.service_version = service_version

    def on_start(self, span: Span, parent_context=None):
        """Called when a span starts - add service attributes."""
        span.set_attribute("service.name", self.service_name)
        span.set_attribute("service.version", self.service_version)


def setup_tracing():
    """Initialize OpenTelemetry tracing with GCP Cloud Trace exporter."""
    # Import config inside function to avoid circular import
    from app.config import (
        ENABLE_TRACING,
        PROJECT_ID,
        SERVICE_NAME,
        SERVICE_VERSION,
    )

    # Initialize GCPLogger for tracing module
    project_id = PROJECT_ID
    base_log_metadata = {
        "service_name": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "module": "tracing",
    }
    tracing_logger = GCPLogger(
        gcp_project=project_id,
        console_logging=True,
        base_log_metadata=base_log_metadata,
    )

    # Debug: Check what value is actually being read
    env_value = os.getenv("ENABLE_TRACING", "not set")
    tracing_logger.log_struct(
        message="Tracing configuration check",
        structured_data={
            "service_name": SERVICE_NAME,
            "version": SERVICE_VERSION,
            "module": "tracing",
            "ENABLE_TRACING_env_var": env_value,
            "ENABLE_TRACING_config_value": str(ENABLE_TRACING),
            "ENABLE_TRACING_type": str(type(ENABLE_TRACING)),
        },
        severity="INFO",
    )

    # Check if tracing is explicitly disabled
    if not ENABLE_TRACING:
        tracing_logger.log_struct(
            message="OpenTelemetry tracing disabled",
            structured_data={
                "service_name": SERVICE_NAME,
                "version": SERVICE_VERSION,
                "module": "tracing",
                "reason": "ENABLE_TRACING is False",
            },
            severity="INFO",
        )
        return None

    try:
        # Create Resource with service name and version
        resource = Resource.create(
            {
                "service.name": SERVICE_NAME,
                "service.version": SERVICE_VERSION,
            }
        )

        # Create TracerProvider with Resource and Parent-Based Sampling
        sampler = ParentBased(
            ALWAYS_ON
        )  # Sampler that always samples spans, regardless of the parent span's sampling decision.
        provider = TracerProvider(resource=resource, sampler=sampler)

        # Add custom span processor to set service.name on all spans
        service_name_processor = ServiceNameSpanProcessor(SERVICE_NAME, SERVICE_VERSION)
        provider.add_span_processor(service_name_processor)

        # Configure Cloud Trace exporter
        cloud_trace_exporter = CloudTraceSpanExporter(project_id=project_id)

        # Add BatchSpanProcessor for exporting
        provider.add_span_processor(BatchSpanProcessor(cloud_trace_exporter))

        # Set the global tracer provider
        trace.set_tracer_provider(provider)

        # Instrument requests library for outgoing HTTP request tracing
        RequestsInstrumentor().instrument(tracer_provider=provider)
        # Get tracer
        tracer = trace.get_tracer(SERVICE_NAME, SERVICE_VERSION)

        # Verify the sampler was set correctly
        current_provider = trace.get_tracer_provider()
        if hasattr(current_provider, "sampler"):
            sampler_type = type(current_provider.sampler).__name__
            tracing_logger.log_struct(
                message="OpenTelemetry tracing initialized with ParentBased sampler",
                structured_data={
                    "service_name": SERVICE_NAME,
                    "version": SERVICE_VERSION,
                    "module": "tracing",
                    "project_id": project_id,
                    "sampler_type": sampler_type,
                    "sampler_config": "ParentBased(ALWAYS_ON)",
                },
                severity="INFO",
            )
        else:
            tracing_logger.log_struct(
                message="OpenTelemetry tracing initialized",
                structured_data={
                    "service_name": SERVICE_NAME,
                    "version": SERVICE_VERSION,
                    "module": "tracing",
                    "project_id": project_id,
                    "warning": "Could not verify sampler configuration",
                },
                severity="INFO",
            )

        # Return both tracer and provider so FastAPIInstrumentor can use the provider explicitly
        return tracer, provider
    except Exception as e:
        tracing_logger.log_text(
            f"Failed to initialize OpenTelemetry tracing: {e}",
            severity=Severity.ERROR,
        )
        return None


def get_tracer():
    """Get the tracer instance for creating spans.

    Uses SERVICE_NAME and SERVICE_VERSION from config.
    This function avoids circular imports by importing config inside the function.
    """
    # Import inside function to avoid circular import issues
    from app.config import SERVICE_NAME, SERVICE_VERSION

    return trace.get_tracer(SERVICE_NAME, SERVICE_VERSION)


def instrument_fastapi(app, tracer_provider=None):
    """Instrument FastAPI application for automatic tracing.

    Args:
        app: FastAPI application instance
        tracer_provider: Optional TracerProvider to use. If None, uses the global tracer provider.
    """
    # Explicitly pass tracer_provider to ensure FastAPIInstrumentor uses our configured provider
    # with ParentBased(ALWAYS_ON) sampler instead of creating its own
    if tracer_provider is None:
        tracer_provider = trace.get_tracer_provider()

    FastAPIInstrumentor.instrument_app(app, tracer_provider=tracer_provider)


class FlushTracingMiddleware(BaseHTTPMiddleware):
    """Flush OpenTelemetry spans at the end of each request."""

    def __init__(self, app, tracer_provider: TracerProvider):
        super().__init__(app)
        self.tracer_provider = tracer_provider

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        try:
            if self.tracer_provider:
                self.tracer_provider.force_flush(timeout_millis=5000)
        except Exception:
            logger.warning("Failed to flush tracer provider", exc_info=True)
        return response


def shutdown_tracing():
    """Shutdown OpenTelemetry tracing and flush all pending spans."""
    try:
        provider = trace.get_tracer_provider()
        if isinstance(provider, TracerProvider):
            provider.force_flush(timeout_millis=5000)
            provider.shutdown()
    except Exception:
        logger.warning("Failed to shut down tracer provider", exc_info=True)
