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
SERVICE_VERSION = "1.5.0"
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

# ---------------------------------------------------------------------------
# /v2/retrieval-signals — document-cluster record-type resolution.
# Record types are resolved to CUIs through the cluster-selection service and
# the LOINC document-cluster tables in BigQuery instead of the local
# record_type_name_to_cui.json vocabulary (which /v1 still uses).
# CLUSTER_SELECTION_URL and CLUSTER_BQ_DATASET must be set for /v2 to work;
# when either is missing /v2 answers 503 cluster_service_unavailable.
# ---------------------------------------------------------------------------
CLUSTER_SELECTION_URL = os.getenv("CLUSTER_SELECTION_URL", "").strip()
CLUSTER_SET = os.getenv("CLUSTER_SET", "loinc_document_v001").strip()
CLUSTER_TOP_K = max(1, int(os.getenv("CLUSTER_TOP_K", "10")))
CLUSTER_TOP_P = max(1, int(os.getenv("CLUSTER_TOP_P", "20")))
CLUSTER_COMBINED = _env_bool("CLUSTER_COMBINED", True)
CLUSTER_REQUEST_TIMEOUT_SECONDS = max(
    1, int(os.getenv("CLUSTER_REQUEST_TIMEOUT_SECONDS", "120"))
)
# BigQuery tables backing the cluster -> member -> CUI lookup.
CLUSTER_BQ_PROJECT = os.getenv("CLUSTER_BQ_PROJECT", "").strip() or PROJECT_ID
CLUSTER_BQ_DATASET = os.getenv("CLUSTER_BQ_DATASET", "").strip()
CLUSTER_BQ_LOCATION = os.getenv("CLUSTER_BQ_LOCATION", "US").strip()
CLUSTER_MEMBER_TABLE = os.getenv("CLUSTER_MEMBER_TABLE", "SCENARIO_6_MEMBER").strip()
CLUSTER_MAP_TABLE = os.getenv("CLUSTER_MAP_TABLE", "SCENARIO_6_MAP").strip()
CLUSTER_CLUSTER_TABLE = os.getenv("CLUSTER_CLUSTER_TABLE", "SCENARIO_6_CLUSTER").strip()
# Developer setups only: allow `gcloud auth print-identity-token` as the
# identity-token source when no application default credentials exist.
ALLOW_LOCAL_GCLOUD_AUTH = _env_bool("ALLOW_LOCAL_GCLOUD_AUTH", False)

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
TEMPORAL_MAX_CODES_PER_WINDOW = max(
    1, int(os.getenv("TEMPORAL_MAX_CODES_PER_WINDOW", "2"))
)
TEMPORAL_MIN_SIMILARITY = float(os.getenv("TEMPORAL_MIN_SIMILARITY", "0"))
# Prompt: codeable-window menu (values per unit, most represented first) and
# the query-time shortlist of vocabulary names close to the query wording.
TEMPORAL_MENU_MAX_VALUES = max(0, int(os.getenv("TEMPORAL_MENU_MAX_VALUES", "12")))
TEMPORAL_MENU_UNITS = [
    u.strip().lower()
    for u in os.getenv("TEMPORAL_MENU_UNITS", "day,week,month,year").split(",")
    if u.strip()
]
TEMPORAL_SHORTLIST_TOP_N = max(0, int(os.getenv("TEMPORAL_SHORTLIST_TOP_N", "8")))
TEMPORAL_SHORTLIST_MIN_SCORE = float(os.getenv("TEMPORAL_SHORTLIST_MIN_SCORE", "0.3"))
# Response: cap on temporal_by_candidate entries per query.
TEMPORAL_BY_CANDIDATE_MAX = max(1, int(os.getenv("TEMPORAL_BY_CANDIDATE_MAX", "8")))
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
    1, int(os.getenv("CADENCE_MEMORY_EXAMPLES_PER_CONCEPT", "3"))
)
CADENCE_MEMORY_MAX_CONCEPTS = max(
    0, int(os.getenv("CADENCE_MEMORY_MAX_CONCEPTS", "10"))
)
CADENCE_MEMORY_MIN_SIMILARITY = float(os.getenv("CADENCE_MEMORY_MIN_SIMILARITY", "0.6"))
