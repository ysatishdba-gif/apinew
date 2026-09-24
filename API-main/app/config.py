import os

#  For local development purposes, load environment variables from .env files
# Comment or remove these lines two while pushing to repository
# from dotenv import load_dotenv

# load_dotenv(
#     dotenv_path=os.path.abspath(
#         os.path.join(os.path.dirname(__file__), "..", "config", ".env.dev")
#     )
# )


ALLOWED_GEN_MODELS = [
    "gemini-3.8-flash",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.5-flash-lite",
]
DEFAULT_GEN_MODEL = ALLOWED_GEN_MODELS[4]

ENV = os.getenv("ENV", "dev").strip()
PROJECT_ID = os.getenv("PROJECT_ID").strip()
LOCATION = os.getenv("GCP_LOCATION", "us-central1")
MODEL_VERSION = os.getenv("MODEL_VERSION", DEFAULT_GEN_MODEL).strip()

if MODEL_VERSION not in ALLOWED_GEN_MODELS:
    raise ValueError(
        f"MODEL_VERSION {MODEL_VERSION!r} is not in ALLOWED_GEN_MODELS: "
        f"{', '.join(ALLOWED_GEN_MODELS)}"
    )
SERVICE_NAME = "intent-nature-breakdown-api"
SERVICE_VERSION = "1.5.8"
SERVICE_ID = SERVICE_NAME + "-" + SERVICE_VERSION
# Path prefix the app serves under; matches the ILB/URL-map route. Set API_PREFIX=""
# if the load balancer strips the prefix before forwarding.
API_PREFIX = os.getenv("API_PREFIX", "/intent-nature-breakdown").rstrip("/")
os.environ.setdefault("OTEL_SERVICE_NAME", SERVICE_NAME)
# Enabled by default. Set ENABLE_TRACING=false in environment to disable (e.g., for local development)
ENABLE_TRACING = os.getenv("ENABLE_TRACING", "true").lower().strip() not in ("false")
TEST_VARIABLE = os.getenv("TEST_VARIABLE", "default_value").strip()

# Temporal vocabulary for v2 retrieval signals.
# Prefer GCS when TEMPORAL_GCS_BUCKET is set; otherwise load from local file.
TEMPORAL_GCS_BUCKET = os.getenv("TEMPORAL_GCS_BUCKET", "").strip()
TEMPORAL_GCS_PATH = os.getenv(
    "TEMPORAL_GCS_PATH", "Nature_breakdown/temporal_name_to_cui.json"
).strip()

_DEFAULT_TEMPORAL_LOCAL = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "app",
        "temporal_name_to_cui.json",
    )
)
TEMPORAL_LOCAL_PATH = os.getenv("TEMPORAL_LOCAL_PATH", _DEFAULT_TEMPORAL_LOCAL).strip()

# Record-type vocabulary for /v1/retrieval-signals.
# Same GCS-first / local-fallback convention as the temporal vocabulary.
RECORD_TYPE_GCS_BUCKET = os.getenv("RECORD_TYPE_GCS_BUCKET", "").strip()
RECORD_TYPE_GCS_PATH = os.getenv(
    "RECORD_TYPE_GCS_PATH", "Nature_breakdown/record_type_name_to_cui.json"
).strip()

_DEFAULT_RECORD_TYPE_LOCAL = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "app",
        "record_type_name_to_cui.json",
    )
)
RECORD_TYPE_LOCAL_PATH = os.getenv(
    "RECORD_TYPE_LOCAL_PATH", _DEFAULT_RECORD_TYPE_LOCAL
).strip()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    """An integer setting; a blank or malformed value (an `X=` placeholder
    left in an env file) means the default rather than a startup failure."""
    raw = os.getenv(name)
    try:
        return int(float(raw.strip())) if raw and raw.strip() else default
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    try:
        return float(raw.strip()) if raw and raw.strip() else default
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# /v2/retrieval-signals — Action Finder gate.
# Optional model override for the in-process Action Finder classification.
# Empty (default) means: use the request's model_name, else MODEL_VERSION.
# ---------------------------------------------------------------------------
ACTION_FINDER_MODEL = os.getenv("ACTION_FINDER_MODEL", "").strip()
if ACTION_FINDER_MODEL and ACTION_FINDER_MODEL not in ALLOWED_GEN_MODELS:
    raise ValueError(
        f"ACTION_FINDER_MODEL {ACTION_FINDER_MODEL!r} is not in ALLOWED_GEN_MODELS: "
        f"{', '.join(ALLOWED_GEN_MODELS)}"
    )

# ===========================================================================
# >>> DEPLOYMENT SETTINGS — UPDATE THESE PER ENVIRONMENT (dev / stage / prod) <<<
#
# /v2 and /v3 code record types through the document cluster. Two values
# below have NO usable default and MUST be set (in config/.env.<env> or the
# Cloud Run environment); without them /v2 and /v3 answer
# 503 cluster_service_unavailable and log the missing names at startup:
#
#   CLUSTER_SELECTION_URL   the cluster-selection service endpoint, e.g.
#                           https://<service>-<hash>-<region>.a.run.app/select
#                           (also the identity-token audience)
#   CLUSTER_BQ_DATASET      the BigQuery dataset holding the cluster tables
#
# Usually also set per environment:
#
#   CLUSTER_BQ_PROJECT      BigQuery project of that dataset — defaults to
#                           PROJECT_ID, set it when the tables live elsewhere
#   CLUSTER_BQ_LOCATION     BigQuery location of the dataset (default US)
#   CLUSTER_MEMBER_TABLE / CLUSTER_MAP_TABLE / CLUSTER_CLUSTER_TABLE
#                           table names (defaults SCENARIO_6_*)
#   CLUSTER_SET             cluster set name sent to the service
#
# The service account running the API needs: invoker on the cluster-selection
# service (identity token) and BigQuery Data Viewer + Job User on the dataset.
# ===========================================================================
CLUSTER_SELECTION_URL = os.getenv("CLUSTER_SELECTION_URL", "").strip()  # REQUIRED
CLUSTER_SET = os.getenv("CLUSTER_SET", "loinc_document_v001").strip()
CLUSTER_TOP_K = max(1, _env_int("CLUSTER_TOP_K", 10))
CLUSTER_TOP_P = max(1, _env_int("CLUSTER_TOP_P", 20))
CLUSTER_COMBINED = _env_bool("CLUSTER_COMBINED", True)
CLUSTER_REQUEST_TIMEOUT_SECONDS = max(
    1, _env_int("CLUSTER_REQUEST_TIMEOUT_SECONDS", 120)
)
# BigQuery tables backing the cluster -> member -> CUI lookup (see the
# DEPLOYMENT SETTINGS block above).
CLUSTER_BQ_PROJECT = os.getenv("CLUSTER_BQ_PROJECT", "").strip() or PROJECT_ID
CLUSTER_BQ_DATASET = os.getenv("CLUSTER_BQ_DATASET", "").strip()  # REQUIRED
CLUSTER_BQ_LOCATION = os.getenv("CLUSTER_BQ_LOCATION", "US").strip()
CLUSTER_MEMBER_TABLE = os.getenv("CLUSTER_MEMBER_TABLE", "SCENARIO_6_MEMBER").strip()
CLUSTER_MAP_TABLE = os.getenv("CLUSTER_MAP_TABLE", "SCENARIO_6_MAP").strip()
CLUSTER_CLUSTER_TABLE = os.getenv("CLUSTER_CLUSTER_TABLE", "SCENARIO_6_CLUSTER").strip()
# Developer setups only: allow `gcloud auth print-identity-token` as the
# identity-token source when no application default credentials exist.
ALLOW_LOCAL_GCLOUD_AUTH = _env_bool("ALLOW_LOCAL_GCLOUD_AUTH", False)
# Field name(s) carrying member document names in the cluster-selection
# response (comma-separated; searched recursively).
CLUSTER_MEMBER_NAME_KEYS = os.getenv("CLUSTER_MEMBER_NAME_KEYS", "member_name").strip()
# /v2 and /v3: a record type the document cluster could not code keeps the
# CUIs the /v1 vocabulary match found for it (deterministic fallback).
RECORD_TYPE_JSON_FALLBACK = _env_bool("RECORD_TYPE_JSON_FALLBACK", True)
# Document catalog: the MEMBER -> MAP -> CLUSTER join is loaded once and kept
# in memory for this long (0 = disabled: one BigQuery query per request).
# Member names from the cluster service are matched against it by normalised
# form, folded form and character n-gram similarity (floor below).
CLUSTER_DOCUMENT_CATALOG_TTL_SECONDS = _env_float(
    "CLUSTER_DOCUMENT_CATALOG_TTL_SECONDS", 3600
)
CLUSTER_MEMBER_MATCH_MIN_SIMILARITY = _env_float(
    "CLUSTER_MEMBER_MATCH_MIN_SIMILARITY", 0.75
)
# Abbreviation-aware token match floor (matched tokens / tokens of the longer
# name; every token of the shorter name must match). 0 disables the stage.
CLUSTER_MEMBER_MATCH_MIN_TOKEN_SCORE = _env_float(
    "CLUSTER_MEMBER_MATCH_MIN_TOKEN_SCORE", 0.6
)
# Extra search texts per record type sent to the cluster service: the
# vocabulary labels the model matched the record type to (0 = name only).
CLUSTER_ALIAS_TEXTS_MAX = max(0, _env_int("CLUSTER_ALIAS_TEXTS_MAX", 2))
# Record types still uncoded after the cluster, the catalog and the /v1
# vocabulary are matched by the model against the cluster's document classes
# (one call per request for all of them). Off -> such record types stay uncoded.
RECORD_TYPE_LLM_FALLBACK = _env_bool("RECORD_TYPE_LLM_FALLBACK", True)
RECORD_TYPE_LLM_FALLBACK_MODEL = os.getenv("RECORD_TYPE_LLM_FALLBACK_MODEL", "").strip()
RECORD_TYPE_LLM_FALLBACK_MAX_LABELS = max(
    1, _env_int("RECORD_TYPE_LLM_FALLBACK_MAX_LABELS", 400)
)
# Concurrent cluster-selection calls per query (one per record type).
CLUSTER_SELECTION_MAX_WORKERS = max(1, _env_int("CLUSTER_SELECTION_MAX_WORKERS", 8))

# Tag vocabulary (optional): name -> CUI file in the same shape as the
# record-type vocabulary, used by /v2 and /v3 to code `tags[].coding`. Not
# configured -> tags keep names and topics with an empty coding, as in /v1.
TAG_GCS_BUCKET = os.getenv("TAG_GCS_BUCKET", "").strip()
TAG_GCS_PATH = os.getenv(
    "TAG_GCS_PATH", "Nature_breakdown/tag_name_to_cui.json"
).strip()
TAG_LOCAL_PATH = os.getenv("TAG_LOCAL_PATH", "").strip()

# ---------------------------------------------------------------------------
# /v3/retrieval-signals — canonical temporal mode.
# The temporal vocabulary is indexed in-process at startup (formula index +
# name similarity) and never pasted into the prompt; the model emits a
# structured window that the index resolves. Every limit below is a setting;
# no vocabulary content or clinical mapping lives in code.
# ---------------------------------------------------------------------------
# Name-similarity provider for the index: "lexical" (in-process character
# n-gram TF-IDF, no network) or "vertex" (Vertex text embeddings; vectors are
# cached on disk keyed by model + vocabulary hash).
TEMPORAL_INDEX_SIMILARITY_PROVIDER = (
    os.getenv("TEMPORAL_INDEX_SIMILARITY_PROVIDER", "lexical").strip().lower()
)
TEMPORAL_EMBEDDING_MODEL = os.getenv("TEMPORAL_EMBEDDING_MODEL", "").strip()
TEMPORAL_INDEX_CACHE_DIR = os.getenv("TEMPORAL_INDEX_CACHE_DIR", "").strip()
# Resolution: distinct CUIs kept per window (ranked by closeness to the
# model's wording) and the similarity floor below which extra entries drop.
TEMPORAL_MAX_CODES_PER_WINDOW = max(1, _env_int("TEMPORAL_MAX_CODES_PER_WINDOW", 2))
TEMPORAL_MIN_SIMILARITY = _env_float("TEMPORAL_MIN_SIMILARITY", 0.3)
# Prompt: codeable-window menu (values per unit, most represented first) and
# the query-time shortlist of vocabulary names close to the query wording.
TEMPORAL_MENU_MAX_VALUES = max(0, _env_int("TEMPORAL_MENU_MAX_VALUES", 12))
TEMPORAL_MENU_UNITS = [
    u.strip().lower()
    for u in os.getenv("TEMPORAL_MENU_UNITS", "day,week,month,year").split(",")
    if u.strip()
]
TEMPORAL_SHORTLIST_TOP_N = max(0, _env_int("TEMPORAL_SHORTLIST_TOP_N", 8))
TEMPORAL_SHORTLIST_MIN_SCORE = _env_float("TEMPORAL_SHORTLIST_MIN_SCORE", 0.3)
# Shadow mode: /v3 also runs the legacy vocabulary-list path and logs
# per-candidate agreement ("Temporal shadow comparison"). Doubles the
# contextual-environment call; for rollout only.
TEMPORAL_SHADOW_MODE = _env_bool("TEMPORAL_SHADOW_MODE", False)

# Cadence memory: the service's own inferred windows aggregated from its logs
# by scripts/build_cadence_memory.py (GCS first, local fallback, optional).
CADENCE_MEMORY_GCS_BUCKET = os.getenv("CADENCE_MEMORY_GCS_BUCKET", "").strip()
CADENCE_MEMORY_GCS_PATH = os.getenv(
    "CADENCE_MEMORY_GCS_PATH", "Nature_breakdown/cadence_memory.json"
).strip()
_DEFAULT_CADENCE_MEMORY_LOCAL = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "app", "cadence_memory.json")
)
CADENCE_MEMORY_LOCAL_PATH = os.getenv(
    "CADENCE_MEMORY_LOCAL_PATH", _DEFAULT_CADENCE_MEMORY_LOCAL
).strip()
CADENCE_MEMORY_EXAMPLES_PER_CONCEPT = max(
    1, _env_int("CADENCE_MEMORY_EXAMPLES_PER_CONCEPT", 3)
)
CADENCE_MEMORY_MAX_CONCEPTS = max(0, _env_int("CADENCE_MEMORY_MAX_CONCEPTS", 10))
CADENCE_MEMORY_MIN_SIMILARITY = _env_float("CADENCE_MEMORY_MIN_SIMILARITY", 0.6)
