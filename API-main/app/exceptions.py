"""Custom exceptions for the Intent Nature Breakdown API."""

from google.api_core.exceptions import FailedPrecondition, InvalidArgument, NotFound


class LLMError(Exception):
    """Base exception for LLM-related errors."""

    status_code = 502
    error_type = "llm_error"

    def __init__(self, message: str, details: dict | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class LLMTimeoutError(LLMError):
    """Raised when LLM call times out."""

    status_code = 504
    error_type = "llm_timeout"

    def __init__(
        self, message: str = "LLM call timed out", details: dict | None = None
    ):
        super().__init__(message, details)


class LLMRateLimitError(LLMError):
    """Raised when rate limit is exceeded."""

    status_code = 429
    error_type = "rate_limit_exceeded"

    def __init__(
        self,
        message: str = "Rate limit exceeded, please retry later",
        details: dict | None = None,
    ):
        super().__init__(message, details)


class LLMServiceError(LLMError):
    """Raised when LLM service is unavailable."""

    status_code = 503
    error_type = "service_unavailable"

    def __init__(
        self,
        message: str = "LLM service is temporarily unavailable",
        details: dict | None = None,
    ):
        super().__init__(message, details)


class LLMInvalidRequestError(LLMError):
    """Raised when request to LLM is invalid."""

    status_code = 400
    error_type = "invalid_llm_request"

    def __init__(
        self, message: str = "Invalid request to LLM", details: dict | None = None
    ):
        super().__init__(message, details)


class ClusterServiceError(LLMError):
    """Document-cluster dependency (cluster-selection service / BigQuery lookup)
    is unavailable, misconfigured, or returned invalid data.

    Subclasses LLMError so /v2/retrieval-signals maps it through the same
    handler and envelope as every other upstream failure (503).
    """

    status_code = 503
    error_type = "cluster_service_unavailable"

    def __init__(
        self,
        message: str = "Document-cluster service is unavailable",
        details: dict | None = None,
    ):
        super().__init__(message, details)


class TemporalIndexUnavailableError(LLMError):
    """/v3 needs the in-process temporal index (built from the temporal
    vocabulary at startup); without it canonical temporal mode cannot run."""

    status_code = 503
    error_type = "temporal_index_unavailable"

    def __init__(
        self,
        message: str = "Temporal index is not available",
        details: dict | None = None,
    ):
        super().__init__(message, details)


class LLMLocationError(LLMError):
    """Raised when the requested model/location combination is unavailable."""

    status_code = 404
    error_type = "llm_location_error"

    def __init__(
        self,
        message: str = "Requested model or location is not available",
        details: dict | None = None,
    ):
        super().__init__(message, details)


def _is_location_model_error(exc: Exception) -> bool:
    """Classify whether an exception represents an unsupported model/location
    combination downstream, as opposed to a generic invalid-request or
    transient error."""
    if isinstance(exc, FailedPrecondition):
        return True

    msg = str(exc).lower()

    location_markers = (
        "invalid hostname",
        "not available in",
        "does not exist",
        "not supported",
        "was not found",
        "not_found",
    )

    if isinstance(exc, (NotFound, InvalidArgument)):
        return (
            any(m in msg for m in location_markers)
            or "location" in msg
            or "region" in msg
        )

    # classify_api_exception wraps the original error — check the message directly
    return any(m in msg for m in location_markers)
