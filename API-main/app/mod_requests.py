from typing import Annotated, Any, Optional

from pydantic import BaseModel, Field, model_validator, validator

import app.config as config
from app.utils.action_gate import ACTION_TYPES
from app.utils.location import VALID_LOCATIONS

# Action Finder taxonomy surfaced as an enum in the OpenAPI schema.
ActionType = Annotated[str, Field(json_schema_extra={"enum": sorted(ACTION_TYPES)})]

_MODEL_NAME_FIELD_DESC = (
    "Optional model override (defaults to config.MODEL_VERSION). "
    f"Must be one of: {', '.join(config.ALLOWED_GEN_MODELS)}."
)


class ModelNameValidationMixin:
    """Reject unknown model_name values before any Vertex call."""

    @model_validator(mode="after")
    def validate_model_name_allowed(self):
        resolved = self.model_name or config.MODEL_VERSION
        if resolved not in config.ALLOWED_GEN_MODELS:
            raise ValueError(
                f"Invalid 'model_name'. Allowed models are: "
                f"{', '.join(config.ALLOWED_GEN_MODELS)}"
            )
        return self


class ThinkingConfig(BaseModel):
    """Thinking configuration for Gemini thinking mode."""

    thinking_budget: int = Field(
        ..., ge=0, description="Token budget for thinking (0 to disable)"
    )


class GenerationConfig(BaseModel):
    """Generation configuration for LLM calls."""

    temperature: float | None = Field(
        None, ge=0.0, le=2.0, description="Sampling temperature (0.0-2.0)"
    )
    top_p: float | None = Field(
        None, ge=0.0, le=1.0, description="Nucleus sampling parameter"
    )
    top_k: int | None = Field(None, ge=0, description="Top-k sampling parameter")
    max_output_tokens: int | None = Field(
        None, ge=1, description="Maximum output tokens"
    )
    thinking_config: ThinkingConfig | None = Field(
        None, description="Thinking mode configuration (requires thinking_budget)"
    )


class IntentRequest(ModelNameValidationMixin, BaseModel):
    """Validates intent request structure from client request."""

    texts: str | list[str] = Field(
        ...,
        description="Single text, comma-separated texts, or list of texts",
        example="Patient with DM, SOB on exertion",
    )

    model_name: str | None = Field(
        None,
        description=_MODEL_NAME_FIELD_DESC,
        example="gemini-2.5-flash",
    )

    generation_config: GenerationConfig | None = Field(
        None,
        description="Generation configuration (temperature, top_p, top_k, max_output_tokens, thinking_config, etc.)",
    )

    enable_retrieval_signals: bool | None = Field(
        False,
        description=(
            "Only used by /v2/nature-breakdown. When true, runs contextual environment "
            "and attaches retrieval_signals. Ignored by /v1/nature-breakdown."
        ),
    )

    location: str | None = Field(
        default=None,
        description=(
            "GCP region for routing the LLM call. Allowed values: 'us', 'us-central1'. "
            "When not provided, defaults to 'us-central1' for existing models, or 'us' "
            "for newer model families (see MODEL_LOCATION_DEFAULTS)."
        ),
    )

    @validator("texts", pre=True)
    def normalize_texts(cls, value):
        """
        Normalize texts into List[str]
        """
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []

        if isinstance(value, list):
            return [t.strip() for t in value if isinstance(t, str) and t.strip()]

        raise ValueError("texts must be a string or list of strings")

    @model_validator(mode="after")
    def validate_location(self):
        # Falsy (None or "") means "not provided" and is resolved later via
        # resolve_model_location(); only a non-empty, explicitly-set value is validated here.
        if self.location and self.location not in VALID_LOCATIONS:
            raise ValueError(
                f"Invalid 'location'. Allowed values are: {', '.join(sorted(VALID_LOCATIONS))}"
            )
        return self


class ExtractIntentsResponse(BaseModel):
    """Validates extract intents response structure from LLM response."""

    status: int
    output: dict[str, Any]  # Contains total_inputs and results
    details: dict[str, Any]
    service: str


# LLM Response Validation Models
class SubNature(BaseModel):
    """Validates sub_nature structure from LLM response."""

    category_path: str = Field(
        ..., min_length=1, description="Hierarchical path separated by ' >> '"
    )
    atomic_concepts: list[str] = Field(
        ..., min_items=1, description="List of atomic concepts"
    )


class Intent(BaseModel):
    """Validates intent structure from LLM response."""

    intent_title: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    nature: str = Field(..., min_length=1)
    sub_natures: list[SubNature] = Field(
        ..., min_items=1, description="List of sub-natures"
    )
    final_queries: list[str] = Field(
        ..., min_items=1, description="Atomic queries for retrieval"
    )


class IntentExtractionResponse(BaseModel):
    """Validates complete LLM response structure matching prompt output format."""

    is_clinical: bool
    reason: str = Field(default="", description="Reason if non-clinical")
    original_query: str
    expanded_query: str
    total_intents_detected: int = Field(
        ..., ge=0, description="Total number of intents detected"
    )
    intents: list[Intent] = Field(
        default_factory=list, description="List of extracted intents"
    )

    @model_validator(mode="after")
    def validate_clinical_logic(self):
        """Validate business logic: clinical queries must have intents, non-clinical must not."""
        if self.is_clinical and len(self.intents) == 0:
            raise ValueError("Clinical queries must have at least one intent")
        if not self.is_clinical and len(self.intents) > 0:
            raise ValueError("Non-clinical queries should have empty intents list")
        return self

    @model_validator(mode="after")
    def validate_intent_count_match(self):
        """Validate that total_intents_detected matches actual intents count."""
        if self.total_intents_detected != len(self.intents):
            raise ValueError(
                f"total_intents_detected ({self.total_intents_detected}) must match "
                f"actual intents count ({len(self.intents)})"
            )
        return self


class QueryExpansionOutput(BaseModel):
    """v2 query-expansion LLM schema (Gemini response_schema)."""

    expanded_query: str
    abbreviations_expanded: list[str] = Field(default_factory=list)


class RepresentativeTermsOutput(BaseModel):
    """v2 representative-terms LLM schema (Gemini response_schema)."""

    representative_terms: list[str] = Field(default_factory=list)


class IntentExtractionResponseV2(BaseModel):
    """v2 intent extraction schema — no clinical validation (matches reference)."""

    total_intents_detected: int = Field(default=0, ge=0)
    intents: list[Intent] = Field(default_factory=list)

    @model_validator(mode="after")
    def fix_intent_count(self):
        """Align count with list length (reference behavior)."""
        self.total_intents_detected = len(self.intents)
        return self


# ---------------------------------------------------------------------------
# v2 API response shape (OpenAPI documentation; runtime envelope stays Dict)
# ---------------------------------------------------------------------------


class TemporalEntry(BaseModel):
    """CUI-resolved temporal window on retrieval_signals."""

    time_window: str | None = None
    codes: str | None = None
    formula: list[str] | None = None

    @validator("formula", pre=True)
    def coerce_formula(cls, value):
        if isinstance(value, str):
            return [value]
        return value


class RetrievalSignals(BaseModel):
    """Structured retrieval signals after post-processing."""

    record_types: list[str] = Field(default_factory=list)
    temporal: list[TemporalEntry] = Field(default_factory=list)
    authors: list[str] = Field(default_factory=list)
    longitudinal_scope: list[str] = Field(default_factory=list)
    content_signals: list[str] = Field(default_factory=list)
    clinical_setting: list[str] = Field(default_factory=list)


class FinalCandidate(BaseModel):
    """One atomic-concept candidate with optional retrieval signals."""

    candidate_id: str
    intent_title: str
    nature: str
    sub_nature: str
    candidate: str
    retrieval_signals: RetrievalSignals | None = None


class IntentV2(BaseModel):
    """v2 intent block: keeps final_queries; adds candidates + signals when enabled."""

    intent_title: str
    description: str
    nature: str
    sub_natures: list[SubNature] = Field(default_factory=list)
    final_queries: list[str] = Field(default_factory=list)
    final_candidates: list[FinalCandidate] = Field(default_factory=list)
    retrieval_signals: RetrievalSignals | None = None


class RecordTypeSignal(BaseModel):
    """Normalized record type with codes."""

    name: str = Field(..., min_length=1, example="Radiology Report")
    coding: list[dict[str, str]] = Field(
        default_factory=list, example=[{"code": "C0034571"}]
    )

    @validator("coding", pre=True)
    def normalize_coding(cls, value):
        # Record-type codings carry `code` only — no `system`, by contract.
        # Bare CUI strings are accepted and wrapped; a `system` supplied by a
        # client is dropped rather than echoed back.
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("coding must be a list")
        out: list[dict[str, str]] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append({"code": item.strip()})
            elif isinstance(item, dict):
                code = item.get("code")
                if isinstance(code, str) and code.strip():
                    out.append({"code": code.strip()})
        return out


class TemporalSignal(BaseModel):
    """Normalized temporal validity window with codes."""

    name: str | None = Field(None, example="Last One Year")
    formula: list[str] | None = Field(None, example=["REF_POINT - 1Y"])
    coding: list[dict[str, str]] = Field(default_factory=list)

    @validator("coding", pre=True)
    def normalize_coding(cls, value):
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("coding must be a list")
        out: list[dict[str, str]] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append({"system": "UMLS", "code": item.strip()})
            elif isinstance(item, dict):
                code = item.get("code")
                if isinstance(code, str) and code.strip():
                    system = item.get("system")
                    if not isinstance(system, str) or not system.strip():
                        system = "UMLS"
                    out.append({"system": system.strip(), "code": code.strip()})
        return out

    @validator("formula", pre=True)
    def coerce_formula(cls, value):
        if isinstance(value, str):
            return [value]
        return value


class TagSignal(BaseModel):
    """Representative term / tag with codes and topic labels."""

    name: str = Field(..., min_length=1, example="Chest X-ray")
    coding: list[dict[str, str]] = Field(
        default_factory=list, example=[{"system": "UMLS", "code": "C0039985"}]
    )
    topics: list[str] = Field(default_factory=list, example=["imaging", "radiology"])

    @validator("coding", pre=True)
    def normalize_coding(cls, value):
        # Same shape as TemporalSignal.coding: a bare CUI is promoted to a
        # {system, code} object so one response never carries two coding shapes.
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("coding must be a list")
        out: list[dict[str, str]] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append({"system": "UMLS", "code": item.strip()})
            elif isinstance(item, dict):
                code = item.get("code")
                if isinstance(code, str) and code.strip():
                    system = item.get("system")
                    if not isinstance(system, str) or not system.strip():
                        system = "UMLS"
                    out.append({"system": system.strip(), "code": code.strip()})
        return out


class RetrievalSignalsQuery(BaseModel):
    """One input query. record_types / temporal / tags are optional client hints."""

    text: list[str] = Field(
        ...,
        description="List of complete clinical query strings.",
        example=["chest x-ray reports from the last year, including follow-up"],
    )
    record_types: list[RecordTypeSignal] | None = Field(
        None, description="Optional record-type hints; merged ahead of derived values"
    )
    temporal: TemporalSignal | None = Field(
        None, description="Optional temporal hint; overrides the derived window"
    )
    tags: list[TagSignal] | None = Field(
        None, description="Optional tag hints; merged ahead of derived values"
    )


class RetrievalSignalsRequest(ModelNameValidationMixin, BaseModel):
    """Request body for POST /v1/retrieval-signals."""

    queries: list[RetrievalSignalsQuery] = Field(
        ..., min_length=1, description="One or more query objects"
    )
    model_name: str | None = Field(None, description=_MODEL_NAME_FIELD_DESC)
    generation_config: Optional["GenerationConfig"] = Field(
        None, description="Generation configuration override"
    )

    location: str | None = Field(
        default=None,
        description=(
            "GCP region for routing the LLM call. Allowed values: 'us', 'us-central1'. "
            "When not provided, defaults to 'us-central1' for existing models, or 'us' "
            "for newer model families (see MODEL_LOCATION_DEFAULTS)."
        ),
    )

    @model_validator(mode="after")
    def validate_location(self):
        # Falsy (None or "") means "not provided" and is resolved later via
        # resolve_model_location(); only a non-empty, explicitly-set value is validated here.
        if self.location and self.location not in VALID_LOCATIONS:
            raise ValueError(
                f"Invalid 'location'. Allowed values are: {', '.join(sorted(VALID_LOCATIONS))}"
            )
        return self


class RetrievalSignalsRequestV2(RetrievalSignalsRequest):
    """Request body for POST /v2/retrieval-signals.

    Same body as /v1 plus the optional Action Finder gate. Omitting `actions`
    (or sending an empty list) keeps the ungated behaviour.
    """

    actions: list[ActionType] | None = Field(
        None,
        description=(
            "Optional expected activity types. When supplied, each query is first "
            "classified by the Action Finder (in-process); retrieval signals are "
            "computed only for queries whose identified actions overlap this list "
            "and include information_retrieval. Non-matching queries return a "
            "skip block instead of signals. Omitted or empty: no gating."
        ),
        json_schema_extra={"example": ["information_retrieval"]},
    )
    enable_action_finder: bool = Field(
        True,
        description=(
            "Enable Action Finder gating when actions are supplied. Set false to "
            "compute retrieval signals for every query regardless of actions."
        ),
    )


class QueryRetrievalSignals(BaseModel):
    """Per-query retrieval-signals block, keyed by the request's id."""

    id: str
    text: str
    record_types: list[RecordTypeSignal] = Field(default_factory=list)
    temporal: TemporalSignal | None = None
    tags: list[TagSignal] = Field(default_factory=list)


class RetrievalSignalsResponse(BaseModel):
    """Service envelope for /v1/retrieval-signals (matches other endpoints)."""

    status: int
    output: (
        dict  # {"total_queries": int, "queries": [QueryRetrievalSignals | error block]}
    )
    details: dict
    service: str
