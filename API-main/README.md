# Nature Breakdown API

## Overview

Nature Breakdown API extracts structured, ontology-ready clinical intents from unstructured medical text using a Vertex AI (Gemini) LLM pipeline.

**Service version:** `1.1.0`

**Key Features:**
- Accepts single or multiple clinical text inputs
- Expands medical abbreviations and implicit context
- Detects whether input is clinical (v1)
- Returns validated, structured clinical intents
- **v2:** representative terms and optional retrieval signals for downstream document retrieval

## Architecture

```
Client
  │
FastAPI
  ├─ /v1/extract-intents     (Simplified)
  ├─ /v1/nature-breakdown    (Full v1 pipeline)
  ├─ /v2/nature-breakdown    (v2 pipeline + optional retrieval signals)
  └─ /v1/retrieval-signals (Standalone retrieval signals)
  │
ContextualIntentPipeline
  ├─ Query Expansion (LLM)
  ├─ Intent Extraction (LLM)
  ├─ Representative Terms (LLM, v2)
  ├─ Contextual Environment (LLM, v2 when enable_retrieval_signals=true)
  ├─ Pydantic Validation / signal assembly
  │
Structured JSON Response
```

**Version routing:** Pipeline version is selected by URL path only. `/v1/*` stays on the legacy pipeline. `/v2/nature-breakdown` always uses v2 prompts; `enable_retrieval_signals` only gates the contextual-environment / retrieval-signals step.

| Endpoint | Pipeline | Notes |
|---|---|---|
| `/v1/extract-intents` | v1 | Simplified intents |
| `/v1/nature-breakdown` | v1 | Full nature breakdown; ignores signals flag |
| `/v2/nature-breakdown` | v2 | Representative terms; optional `final_candidates` + `retrieval_signals` |
| `/v1/retrieval-signals` | v2 | Standalone retrieval signals; output ids are generated |

---

## Getting Started

Follow these steps to get the project running on your local system.

### Installation

1. Ensure Python 3.12+ is installed on your system.

2. Clone the repository:
   ```bash
   git clone <repo-url>
   cd <repo-folder>
   ```

3. Create a virtual environment and activate it:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

Start the service locally:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Running Tests

Execute the unit test cases using pytest:

```bash
pytest
```

For more detailed test output:

```bash
pytest -v -s
```

### API References

After running the service locally, access the interactive API documentation at:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Temporal vocabulary (v2 retrieval signals)

CUI-resolved temporal windows need a name→CUI vocabulary loaded **once at process startup** (not per request).

| Variable | Required | Description |
|---|---|---|
| `TEMPORAL_GCS_BUCKET` | No | GCS bucket. When set (with path), vocab is loaded from GCS. |
| `TEMPORAL_GCS_PATH` | No | Object path in the bucket (default blob path used when bucket is set). |
| `TEMPORAL_LOCAL_PATH` | No | Local JSON file used when bucket is not set, **or when GCS load fails**. Default: `app/temporal_name_to_cui.json`. |

**Behavior:**
- Prefer GCS when `TEMPORAL_GCS_BUCKET` is set; on GCS failure (or when bucket is unset), load from `TEMPORAL_LOCAL_PATH`.
- If both fail, v2 still runs but temporal falls back to `Recent` / `C0332185`.
- Updating the JSON in GCS or on disk requires a process restart (or new Cloud Run instance) to take effect.
- For Cloud Run: set the GCS env vars, **or** ship the JSON in the image (local backup runs automatically if GCS fails).

### Record-type vocabulary (`/v1/retrieval-signals`)

Record-type CUIs need a name→CUI vocabulary, loaded **once at process startup** using the same GCS-first / local-fallback convention as the temporal vocabulary.

| Variable | Required | Description |
|---|---|---|
| `RECORD_TYPE_GCS_BUCKET` | No | GCS bucket for the record-type vocabulary. Default `""`. |
| `RECORD_TYPE_GCS_PATH` | No | Blob path. Default: `Nature_breakdown/record_type_name_to_cui.json`. |
| `RECORD_TYPE_LOCAL_PATH` | No | Local JSON file used when bucket is not set, **or when GCS load fails**. Default: `app/record_type_name_to_cui.json`. |

**Behavior:**
- Vocabulary file shape (identical to the temporal vocab): `{ "Radiology Report": [{"cui": "C0034571"}], ... }`. Lookup is case- and underscore-insensitive (`radiology_report` matches `Radiology Report`). Keys starting with `_` are ignored.
- The loaded labels are injected into the contextual-environment prompt so the model matches against real vocabulary entries instead of inventing free-text labels.
- If both sources fail, the endpoint still responds but record types carry an empty `coding` list.

### Document-cluster record types + Action Finder (`/v2/retrieval-signals`)

`/v2/retrieval-signals` does **not** use the record-type JSON vocabulary. Record-type names still come from the pipeline, but their CUIs are resolved **per request** through the LOINC document cluster: the name (with the query's tag names as context) is sent to the cluster-selection service, and the returned member document names are mapped to document classes and CUIs in BigQuery. Nothing is loaded at startup; a missing setting makes `/v2` answer `503 cluster_service_unavailable` (logged once at startup as a warning).

| Variable | Required | Description |
|---|---|---|
| `CLUSTER_SELECTION_URL` | **Yes (for /v2)** | Cluster-selection service endpoint. Identity token audience. |
| `CLUSTER_BQ_DATASET` | **Yes (for /v2)** | BigQuery dataset holding the document-cluster tables. |
| `CLUSTER_BQ_PROJECT` | No | BigQuery project. Default: `PROJECT_ID`. |
| `CLUSTER_BQ_LOCATION` | No | BigQuery location. Default `US`. |
| `CLUSTER_MEMBER_TABLE` / `CLUSTER_MAP_TABLE` / `CLUSTER_CLUSTER_TABLE` | No | Table names. Defaults `SCENARIO_6_MEMBER` / `SCENARIO_6_MAP` / `SCENARIO_6_CLUSTER`. |
| `CLUSTER_SET` | No | Cluster set sent to the service. Default `loinc_document_v001`. |
| `CLUSTER_TOP_K` / `CLUSTER_TOP_P` / `CLUSTER_COMBINED` | No | Cluster-selection parameters. Defaults `10` / `20` / `true`. |
| `CLUSTER_REQUEST_TIMEOUT_SECONDS` | No | HTTP timeout for the cluster-selection call. Default `120`. |
| `ALLOW_LOCAL_GCLOUD_AUTH` | No | Developer setups only: use `gcloud auth print-identity-token` when no application default credentials exist. Default `false`. |
| `ACTION_FINDER_MODEL` | No | Model used for the Action Finder classification. Default: the request's `model_name`, else `MODEL_VERSION`. Must be in the allowed models list. |

**Behavior:**
- Authentication: on Cloud Run the identity token comes from the attached service account (metadata server); the token is cached and refreshed before expiry, and refreshed once more if the service answers `401/403`. Without any token the service is not called and record types carry an empty `coding` list.
- Cluster-selection calls for the record types of one query run concurrently; one BigQuery query serves all of them. The result per record type is identical to resolving that name alone.
- The contextual-environment prompt runs **without** the `KNOWN RECORD TYPES` vocabulary block for `/v2` (no `record_type_matches` are requested), so the pipeline's record-type names are free-text canonical labels coded externally.
- The Action Finder is an in-process LLM classification (never an HTTP hop) that runs through the same model client as the pipeline. Its token usage is reported under `usage_metadata.action_finder`.

### Canonical temporal index + cadence memory (`/v3/retrieval-signals`)

`/v3/retrieval-signals` never sends the temporal vocabulary to the model. At startup the loaded vocabulary (same `TEMPORAL_*` sources as above) is indexed in-process: every entry's formula is parsed into a canonical window (`last 6 months`, `between 6 and 12 months`, …), unit codes are learned from the file, and entry names get a similarity index. At request time the model writes each window as a structured span and code resolves it against the index. Nothing about the vocabulary, its defaults or any clinical cadence is written in code; if the vocabulary could not be indexed, `/v3` answers `503 temporal_index_unavailable` while `/v1` and `/v2` keep working.

| Variable | Required | Description |
|---|---|---|
| `TEMPORAL_INDEX_SIMILARITY_PROVIDER` | No | `lexical` (in-process character n-gram TF-IDF, no network) or `vertex` (Vertex text embeddings, cached on disk; falls back to lexical on failure). Default `lexical`. |
| `TEMPORAL_EMBEDDING_MODEL` | No | Embedding model for the `vertex` provider. |
| `TEMPORAL_INDEX_CACHE_DIR` | No | Directory for the embedding cache (keyed by model + vocabulary hash). |
| `TEMPORAL_MAX_CODES_PER_WINDOW` | No | Distinct CUIs kept per resolved window, ranked by closeness to the model's wording. Default `2`. |
| `TEMPORAL_MIN_SIMILARITY` | No | Similarity floor below which additional entries for the same window are dropped. Default `0`. |
| `TEMPORAL_MENU_MAX_VALUES` / `TEMPORAL_MENU_UNITS` | No | Codeable-window menu shown to the model: values per unit (most represented first) and which units. Defaults `12` / `day,week,month,year`. |
| `TEMPORAL_SHORTLIST_TOP_N` / `TEMPORAL_SHORTLIST_MIN_SCORE` | No | Query-time shortlist of vocabulary names close to the query wording (only when the query states a span). Defaults `8` / `0.3`. |
| `TEMPORAL_BY_CANDIDATE_MAX` | No | Cap on `temporal_by_candidate` entries per query. Default `8`. |
| `TEMPORAL_SHADOW_MODE` | No | `true` makes `/v3` also run the legacy vocabulary-list path and log per-candidate agreement (`Temporal shadow comparison`). Doubles the contextual-environment call; rollout only. Default `false`. |
| `CADENCE_MEMORY_GCS_BUCKET` / `CADENCE_MEMORY_GCS_PATH` | No | Cadence-memory file in GCS (default path `Nature_breakdown/cadence_memory.json`). |
| `CADENCE_MEMORY_LOCAL_PATH` | No | Local fallback. Default `app/cadence_memory.json` (absent → memory disabled, `/v3` still runs). |
| `CADENCE_MEMORY_EXAMPLES_PER_CONCEPT` / `CADENCE_MEMORY_MAX_CONCEPTS` / `CADENCE_MEMORY_MIN_SIMILARITY` | No | How many previously inferred windows are shown per concept, for how many of the query's concepts, and how close a concept name must be to a remembered one. Defaults `3` / `10` / `0.6`. |

**Behavior:**
- `/v3` uses its own prompt, `app/prompts/v3/contextual_environment.py` (a frozen copy of the v2 template with the vocabulary-list slot replaced by the `CANONICAL WINDOW` section); `/v1` and `/v2` keep `app/prompts/v2/contextual_environment.py` unchanged. Its temporal section contains only data derived at request time: the codeable-window menu (from the index), the shortlist (from the query) and, when a cadence-memory file is loaded, the windows the service itself inferred before for the query's concepts (`WINDOWS THIS SERVICE HAS INFERRED BEFORE …`). Measured on the bundled vocabulary the contextual-environment prompt is ≈17.5K characters instead of ≈82.8K (≈4.4K vs ≈20.7K tokens, −79%).
- Every resolved window is logged as one `Temporal inference` event (query, candidate, basis, rationale, window, formula, CUI, index and memory versions). These events are the only input of the cadence memory and of the evaluation scripts.
- Cadence memory is built, never authored: `python scripts/build_cadence_memory.py --input <exported log JSONL> --output app/cadence_memory.json [--previous <current file>] [--upload gs://bucket/path]`. The job prints a drift report (top window changed / distribution shift per concept) and exits `3` when any concept moved more than `--max-shift`, so a scheduled run can refuse to publish a memory that moved.
- Evaluation before switching traffic: `python scripts/replay_retrieval_signals.py --base-url http://host/intent-nature-breakdown --queries queries.txt --versions v2 v3 --output replay.jsonl`, then `python scripts/evaluate_temporal.py --replay replay.jsonl --baseline v2 --candidate v3 [--min-agreement 0.995] [--judge-model gemini-2.5-flash --project <id>]`. The report gives CUI/formula agreement, unresolved rates, mean prompt tokens and latency per version and, with a judge model, an accept rate over inferred windows with only the rejections listed for a human to sample. Exit code `2` below the agreement threshold for CI.

---

## Troubleshooting

### Installing `aie_logging_utility==0.1.0` fails

If you encounter errors while installing the `aie_logging_utility` package, follow these steps:

1. **Update pip** - Ensure you are using the latest version:
   ```bash
   python -m pip install --upgrade pip
   ```

2. **Install required keyrings** - Some private packages require authentication via keyring:
   ```bash
   pip install keyring artifacts-keyring
   ```

3. **Configure pip to use the private package index** - Create a `pip.ini` file (Windows) or `pip.conf` file (Mac/Linux) in your virtual environment with the following content:
   ```ini
   [global]
   index-url = https://pkgs.dev.azure.com/mclm/_packaging/KMD/pypi/simple/
   ```

4. **Install the package again**:
   ```bash
   pip install aie_logging_utility==0.1.0
   ```

---

## Endpoints

The API provides six endpoints:

1. **`/v1/extract-intents`** — Simplified response (intent title and description only)
2. **`/v1/nature-breakdown`** — Full v1 response (nature, sub_natures, `final_queries`)
3. **`/v2/nature-breakdown`** — v2 response (representative terms; optional retrieval signals)
4. **`/v1/retrieval-signals`** — standalone retrieval signals; accepts comma-separated text and generates output ids
5. **`/v2/retrieval-signals`** — same contract as `/v1/retrieval-signals` plus optional Action Finder gating (`actions`) and record-type CUIs resolved through the document cluster instead of the JSON vocabulary. `GET /v2/retrieval-signals/action-types` lists the selectable actions.
6. **`/v3/retrieval-signals`** — the `/v2` contract with the temporal step in **canonical mode**: the temporal vocabulary is indexed in-process instead of being pasted into the prompt, every candidate gets a window with `basis` and `rationale` (`temporal_by_candidate`), and queries with no time expression get a window inferred from the candidate's clinical nature.

v1 and v2 nature-breakdown accept the same base request shape. `enable_retrieval_signals` is only used by `/v2/nature-breakdown`.

### Request Schema

```json
{
  "texts": [
    "Patient with diabetes mellitus and shortness of breath on exertion",
    "current medication"
  ],
  "model_name": "gemini-2.5-flash",
  "generation_config": {
    "temperature": 0.2,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 64536,
    "thinking_config": {
      "thinking_budget": 1000
    }
  },
  "enable_retrieval_signals": false,
  "location": "us-central1"
}
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `texts` | string \| string[] | Yes | Single text, or list of texts to process |
| `model_name` | string | No | LLM model override (default: `MODEL_VERSION` env, fallback `gemini-3.5-flash`). Must be one of the [allowed models](#allowed-generation-models). |
| `generation_config` | object | No | LLM generation settings (see below) |
| `enable_retrieval_signals` | boolean | No | **v2 only.** When `true`, runs contextual environment and attaches `final_candidates` / `retrieval_signals`. Default `false`. Ignored by `/v1/*`. |
| `location` | string | No | GCP region for routing the LLM call. Allowed values: `us`, `us-central1`. Any other value is rejected with `422`. When omitted, see [Location Routing](#location-routing) for how it's resolved. Applies to all four endpoints. |

**Generation Config Options:**

| Option | Type | Range | Description |
|--------|------|-------|-------------|
| `temperature` | float | 0.0-2.0 | Sampling temperature (higher = more creative) |
| `top_p` | float | 0.0-1.0 | Nucleus sampling parameter |
| `top_k` | int | ≥0 | Top-k sampling parameter |
| `max_output_tokens` | int | ≥1 | Maximum output tokens |
| `thinking_config` | object | - | Thinking mode configuration |

**Thinking Config:**

| Option | Type | Description |
|--------|------|-------------|
| `thinking_budget` | int | Token budget for thinking/reasoning (0 to disable) |

### Allowed Generation Models

`model_name` must be one of the values in `ALLOWED_GEN_MODELS` (`app/config.py`). Unknown models are rejected with **422** before any Vertex call.

| Model |
|-------|
| `gemini-2.5-flash` |
| `gemini-2.5-flash-lite` |
| `gemini-3.1-flash-lite` |
| `gemini-3.5-flash` |
| `gemini-3.6-flash` |
| `gemini-3.7-flash` |
| `gemini-3.8-flash` |
| `gemini-3.5-flash-lite` |

When `model_name` is omitted, the service uses the `MODEL_VERSION` environment variable (must also be on the allowlist).

### Location Routing

`location` pins the LLM call to a specific GCP region instead of the service default. Allowed values: `us`, `us-central1` — any other value is rejected at request validation with `422`.

Resolution order when `location` is not provided (highest priority first):

1. **Model-family default** — `MODEL_LOCATION_DEFAULTS` in `app/utils/location.py`, matched by longest model-name prefix. Empty today, so this tier is currently inert; fill it in once a model family needs a non-default region.
2. **Configured `GCP_LOCATION`** environment variable.
3. **Fallback** — `us-central1` (`DEFAULT_LOCATION`).

Applies uniformly to `/v1/extract-intents`, `/v1/nature-breakdown`, and `/v2/nature-breakdown`. The pipeline caches one Vertex AI client per distinct location it sees, so switching locations across requests doesn't re-authenticate each time.

### Error Responses

LLM-call failures surface as an HTTP error with a structured body:

```json
{
  "detail": {
    "error": "llm_location_error",
    "message": "Requested model/location is not available: 404 Publisher Model ... was not found",
    "details": {
      "original_error": "...",
      "model": "gemini-2.5-flash",
      "location": "us"
    }
  }
}
```

| Status | `error` | Cause |
|---|---|---|
| 422 | `validation_error` | Empty `texts`, whitespace-only text, an invalid `location`, an invalid `model_name`, or (`/v2/retrieval-signals`) an unknown action in `actions` |
| 400 | `invalid_llm_request` | Malformed request to the LLM (e.g. bad `generation_config`) that isn't a location/model issue |
| 404 | `llm_location_error` | The requested model/location combination is unavailable downstream (model not offered in that region, etc.) |
| 429 | `rate_limit_exceeded` | LLM provider rate limit hit — retry later |
| 502 | `llm_error` | Unclassified LLM call failure |
| 503 | `service_unavailable` | LLM service temporarily unavailable (e.g. regional outage) — the service does **not** fail over to another region |
| 503 | `cluster_service_unavailable` | `/v2/retrieval-signals` only: the document-cluster dependency is not configured (`details.missing_settings`) or the cluster-selection / BigQuery lookup failed |
| 504 | `llm_timeout` | LLM call exceeded the 300s timeout |

For a batch (`texts` with multiple entries), the first failure aborts the whole request — errors are not accumulated per-item.

---

### [POST] `/v1/extract-intents`

Extract clinical intents with simplified response. Returns only intent title and description for each intent.

**Use Case:** When you only need basic intent information without detailed breakdown.

#### Response Structure

```json
{
  "status": 1,
  "output": {
    "total_inputs": 1,
    "results": [
      {
        "input_index": 0,
        "original_query": "Patient with DM and SOB",
        "intents": [
          {
            "intent": "Diabetes Mellitus",
            "intent_description": "Patient has diabetes mellitus condition"
          },
          {
            "intent": "Shortness of Breath",
            "intent_description": "Patient experiencing shortness of breath"
          }
        ]
      }
    ]
  },
  "details": {
    "timing": {
      "total_llm_seconds": 15.867464,
      "max_llm_seconds": 15.867464,
      "max_llm_seconds_input_index": 0
    },
    "usage_metadata": {
      "query_expansion": {
        "prompt_token_count": 150,
        "candidates_token_count": 50,
        "total_token_count": 200,
        "thinking_token_count": 0
      },
      "intent_extraction": {
        "prompt_token_count": 500,
        "candidates_token_count": 300,
        "total_token_count": 800,
        "thinking_token_count": 0
      }
    },
    "version": "1.0.0",
    "timestamp": "2026-01-14T19:40:20.712694",
    "source": null
  },
  "service": "intent-nature-breakdown-api-1.0.0"
}
```

**Response Fields:**
- `status` (int): `1` for success, `0` for errors (e.g., validation failures)
- `output` (object): Contains results and count:
  - `total_inputs` (int): Number of input texts processed
  - `results` (array): Array of results, each containing:
    - `input_index` (int): Index of the input text
    - `original_query` (string): Original query text
    - `intents` (array): Array of extracted intents
    - For clinical queries: Also contains `is_clinical: true`, `expanded_query`, `abbreviations_expanded`, etc.
    - For non-clinical queries: Contains `intents: []`, `is_clinical: false`, `rejected_reason`, `expanded_query`
    - For errors: Contains `error` field with error details
- `details` (object): Metadata and timing information:
  - `timing` (object): Processing time metrics
    - `total_llm_seconds` (float): Sum of processing times for all inputs
    - `max_llm_seconds` (float): Maximum processing time across all inputs
    - `max_llm_seconds_input_index` (int): Index of the input that took the longest
    - `usage_metadata` (object): Token usage for audit/cost tracking
    - `query_expansion` (object): Tokens used in query expansion step
    - `intent_extraction` (object): Tokens used in intent extraction step
    - Each contains: `prompt_token_count`, `candidates_token_count`, `total_token_count`, `thinking_token_count`
  - `version` (string): API version
  - `timestamp` (string): Response timestamp (ISO 8601)
  - `source` (string|null): Source identifier if provided in request
- `service` (string): Service identifier

**Note:** `timestamp` and `processing_time_seconds` are no longer included in individual result objects. They are aggregated in `details.timing` and `details.timestamp`.

---

### [POST] `/v1/nature-breakdown`

Extract clinical intents with full nature breakdown. Returns complete structured intents including nature, sub_natures, and final_queries.

**Use Case:** When you need detailed intent analysis with hierarchical categorization and atomic queries. This endpoint is unchanged in `1.1.0` (legacy v1 pipeline).

#### Response Structure

The API returns structured clinical intents with the following format:

```
Intent
 ├── intent_title
 ├── description
 ├── nature                    ← primary informational role
 ├── sub_natures[]             ← constituent conceptual components and sub set of Nature
 │    ├── category_path        ← flattened, ordered hierarchy (canonical vocabulary) which is a subset of nature/sub nature
 │    ├── atomic_concepts[]    ← smallest ontology-mappable units in all levels
 └── final_queries[]           ← derived from all the above, especially from atomic concepts
```

---

### [POST] `/v2/nature-breakdown`

v2 nature breakdown with representative terms and optional retrieval signals.

**Always runs the v2 pipeline** (query expansion → intent extraction → representative terms). When `enable_retrieval_signals` is `true`, also runs contextual environment and assembles retrieval metadata for document search.

**Use Case:** Downstream retrieval that needs record types, authors, longitudinal scope, content signals, clinical setting, and CUI-resolved temporal windows.

#### Response fields (additive vs v1)

| Field | When present | Description |
|---|---|---|
| `representative_terms` | Always (v2) | Canonical clinical entities distilled from the expanded query |
| `final_queries` | Always | Kept for compatibility (string list per intent) |
| `final_candidates` | Signals on | One candidate object per atomic concept, with `candidate_id`, `sub_nature`, `candidate`, and `retrieval_signals` |
| `retrieval_signals` | Signals on | On each candidate **and** each intent. Intent `temporal` is the deduped union (by CUI) of its candidates' resolved temporal entries |

Each `retrieval_signals` block includes:

- `record_types`, `authors`, `longitudinal_scope`, `content_signals`, `clinical_setting`
- `temporal`: `[{ "time_window", "codes", "formula" }, ...]`

#### Example (signals on)

```python
import requests

payload = {
    "texts": "elevated psa in last 3 years",
    "enable_retrieval_signals": True,
}

response = requests.post(
    "http://localhost:8000/v2/nature-breakdown",
    headers={"Content-Type": "application/json"},
    json=payload,
)
print(response.json())
```

**Truncated response shape:**

```json
{
  "status": 1,
  "output": {
    "total_inputs": 1,
    "results": [
      {
        "input_index": 0,
        "original_query": "elevated psa in last 3 years",
        "expanded_query": "...",
        "representative_terms": ["PSA", "prostate disease"],
        "total_intents_detected": 2,
        "intents": [
          {
            "intent_title": "Elevated PSA Laboratory Values",
            "description": "...",
            "nature": "Laboratory / Result",
            "sub_natures": [],
            "final_queries": ["elevated PSA", "PSA laboratory values"],
            "final_candidates": [
              {
                "candidate_id": "fc_001",
                "intent_title": "Elevated PSA Laboratory Values",
                "nature": "Laboratory / Result",
                "sub_nature": "Laboratory Test / Biomarker",
                "candidate": "prostate-specific antigen",
                "retrieval_signals": {
                  "record_types": ["laboratory_report", "progress_note"],
                  "temporal": [
                    {
                      "time_window": "In past 3 years",
                      "codes": "C3843792",
                      "formula": null
                    }
                  ],
                  "authors": ["pathologist", "urologist"],
                  "longitudinal_scope": ["screening", "diagnostic_workup"],
                  "content_signals": ["loinc_lab_code", "measurement_value"],
                  "clinical_setting": ["outpatient"]
                }
              }
            ],
            "retrieval_signals": {
              "record_types": ["laboratory_report", "progress_note"],
              "temporal": [
                {
                  "time_window": "In past 3 years",
                  "codes": "C3843792",
                  "formula": null
                }
              ],
              "authors": ["pathologist", "urologist"],
              "longitudinal_scope": ["screening", "diagnostic_workup"],
              "content_signals": ["loinc_lab_code", "measurement_value"],
              "clinical_setting": ["outpatient"]
            }
          }
        ]
      }
    ]
  },
  "details": {
    "usage_metadata": {
      "query_expansion": {},
      "intent_extraction": {},
      "representative_terms": {},
      "contextual_environment": {}
    },
    "version": "1.1.0"
  },
  "service": "intent-nature-breakdown-api-1.1.0"
}
```

With `enable_retrieval_signals: false`, v2 still returns `representative_terms` and intents with `final_queries`, but omits `final_candidates` / `retrieval_signals` (and does not call contextual environment). See Swagger for the full OpenAPI schema.

---

### [POST] `/v1/retrieval-signals`

Returns normalized record types, temporal validity windows, and representative tags for each input query. Query IDs are generated in the response; callers do not provide them.

Provide `text` as a JSON array. Each array item is processed as one complete query and receives its own generated output ID (`q1`, `q2`, and so on):

#### Request

```json
{
  "queries": [
    {
      "text": [
        "chest x-ray reports from the last year",
        "tell me a joke"
      ]
    }
  ]
}
```

The endpoint also accepts optional `record_types`, `temporal`, and `tags` hints on each query. Record-type and tag hints are merged ahead of derived values. A temporal hint overrides the derived temporal result.

`model_name`, `generation_config`, and `location` are also accepted at the top level and behave exactly as on the other endpoints — see [Location Routing](#location-routing).

#### Response

```json
{
  "queries": [
    {
      "id": "q1",
      "text": "chest x-ray reports from the last year",
      "record_types": [
        {
          "name": "Clinical Note",
          "coding": [
            {
              "code": "C0332185"
            },
            {
              "code": "C0236123"
            }
          ]
        },
        {
          "name": "Radiology Report",
          "coding": [
            {
              "code": "C0034571"
            }
          ]
        }
      ],
      "temporal": {
        "name": "Last One Year",
        "formula": ["REF_POINT", "REF_POINT - 1Y"],
        "coding": [
          {
            "system": "UMLS",
            "code": "C0332185"
          }
        ]
      },
      "tags": [
        {
          "name": "Chest X-ray",
          "coding": [],
          "topics": [
            "imaging",
            "radiology",
            "chest"
          ]
        }
      ]
    }
  ]
}
```

`queries` must contain at least one item, and each item must contain non-empty `text`. Empty or whitespace-only text returns `422`.

---

### [POST] `/v2/retrieval-signals`

Same request body, response envelope and per-query block as `/v1/retrieval-signals`, with two additions:

1. **Action Finder gate** — optional top-level `actions` (plus `enable_action_finder`, default `true`). When `actions` is supplied, each query is first classified in-process by the Action Finder. Retrieval signals are computed only for queries whose identified actions overlap `actions` **and** include `information_retrieval`; every other query returns a *skip block* instead of signals (HTTP 200 — "not serviceable by retrieval" is a result, not a client error). Omitting `actions` (or sending `[]`, or `enable_action_finder: false`) keeps the ungated behaviour. An action name outside the taxonomy returns `422` with `invalid_actions`. If the Action Finder itself fails, the gate fails **open** (the query is processed) and the failure is logged.
2. **Document-cluster record types** — `record_types` CUIs come from the cluster-selection service + BigQuery lookup (see [Document-cluster record types](#document-cluster-record-types--action-finder-v2retrieval-signals)); the `record_type_name_to_cui.json` vocabulary is not consulted. When that dependency is not configured or fails the endpoint returns `503` with `error: "cluster_service_unavailable"`.

Selectable actions (`GET /v2/retrieval-signals/action-types`): `admin_response`, `conversational_response`, `information_retrieval`, `knowledge_fact_search`, `out_of_scope`. Matching is case-, space- and hyphen-insensitive (`"Information Retrieval"` == `information_retrieval`).

#### Request

```json
{
  "actions": ["information_retrieval"],
  "queries": [
    {
      "text": [
        "chest x-ray reports from the last year",
        "book me a cardiology appointment Tuesday"
      ]
    }
  ]
}
```

`model_name`, `generation_config`, `location` and the per-query `record_types` / `temporal` / `tags` hints behave exactly as on `/v1/retrieval-signals`.

#### Response

```json
{
  "status": 1,
  "output": {
    "total_queries": 2,
    "queries": [
      {
        "id": "q1",
        "text": "chest x-ray reports from the last year",
        "record_types": [
          {
            "name": "Radiology Report",
            "coding": [
              { "code": "C0034571" },
              { "code": "C0011923" }
            ]
          }
        ],
        "temporal": {
          "name": "Last One Year",
          "formula": ["REF_POINT", "REF_POINT - 1Y"],
          "coding": [
            { "system": "UMLS", "code": "C0332185" }
          ]
        },
        "tags": [
          {
            "name": "Chest X-ray",
            "coding": [],
            "topics": ["imaging", "radiology", "chest"]
          }
        ]
      },
      {
        "id": "q2",
        "text": "book me a cardiology appointment Tuesday",
        "skipped": true,
        "skip_reason": ["admin_response"],
        "skip_reason_detail": "Action Finder classified this query as admin_response",
        "record_types": [],
        "temporal": null,
        "tags": []
      }
    ]
  },
  "details": {
    "timing": { "total_llm_seconds": 6.2, "max_llm_seconds": 6.2, "max_llm_seconds_input_index": 0 },
    "usage_metadata": {
      "action_finder": { "prompt_token_count": 1800, "candidates_token_count": 20, "total_token_count": 1820, "thinking_token_count": 0 },
      "query_expansion": { "...": "..." },
      "intent_extraction": { "...": "..." },
      "representative_terms": { "...": "..." },
      "contextual_environment": { "...": "..." }
    },
    "version": "1.5.0",
    "timestamp": "2026-09-23T14:02:11.120334+00:00",
    "source": null,
    "model": "gemini-3.5-flash",
    "location": "us"
  },
  "service": "intent-nature-breakdown-api-1.5.0"
}
```

Skip block contract: `skipped: true`, `skip_reason` is the machine-readable list of actions the Action Finder identified, `skip_reason_detail` is the human-readable explanation, and the signal fields keep the same shape as a computed block (`record_types: []`, `temporal: null`, `tags: []`) so every entry parses the same way. `usage_metadata.action_finder` sums the gate's token usage across queries (zeros when no gating ran).

---

### [POST] `/v3/retrieval-signals`

Same request body, gate, record-type resolution, envelope and per-query block as `/v2/retrieval-signals`. What changes is how `temporal` is produced and what is added next to it:

1. **Canonical temporal mode** — the model no longer picks an id from the pasted vocabulary; it writes each window as `window {relation: last|within|range, value, unit[, to_value, to_unit]}` with `basis` (`explicit` when the span is stated in the query, `inferred` when derived from a qualifier such as "current"/"history of" or from the candidate's clinical nature) and a one-clause `rationale`. Code turns the window into the `REF_POINT` formula and attaches the vocabulary CUIs for that formula. A window the vocabulary cannot code is **kept** with its formula and an empty `coding` (never silently dropped or replaced by the default).
2. **Queries with no time expression** — every candidate still gets a window, inferred from its clinical nature and, when a cadence memory is loaded, kept consistent with what the service inferred before for the same concept. When nothing can be inferred, the vocabulary's declared default applies (`basis: default`).
3. **`temporal`** — the query's explicit window when one was stated, else the **widest** inferred window (a too-narrow primary loses documents). Same object shape as `/v1`/`/v2`.
4. **`temporal_by_candidate`** (new, additive) — one entry per candidate concept and window: `candidate`, `intent_title`, `name`, `formula`, `coding`, `basis`, `rationale`, `window`. Entries for the same candidate and formula are merged (CUIs unioned); capped by `TEMPORAL_BY_CANDIDATE_MAX`.
5. **`details.temporal_mode`** (`canonical`) and **`details.temporal_shadow`**; in shadow mode `usage_metadata.shadow_contextual_environment` is added.

Failure specific to `/v3`: `503` with `error: "temporal_index_unavailable"` when the temporal vocabulary could not be indexed at startup.

#### Request

```json
{
  "actions": ["information_retrieval"],
  "queries": [ { "text": ["patient with diabetes"] } ]
}
```

#### Response

```json
{
  "status": 1,
  "output": {
    "total_queries": 1,
    "queries": [
      {
        "id": "q1",
        "text": "patient with diabetes",
        "record_types": [ { "name": "Progress Note", "coding": [ { "system": "UMLS", "code": "C0747978" } ] } ],
        "temporal": {
          "name": "Past 2 years",
          "formula": ["REF_POINT", "REF_POINT - 2Y"],
          "coding": [ { "system": "UMLS", "code": "C…" } ]
        },
        "temporal_by_candidate": [
          {
            "candidate": "diabetes mellitus",
            "intent_title": "Diabetes documentation",
            "name": "Past 2 years",
            "formula": ["REF_POINT", "REF_POINT - 2Y"],
            "coding": [ { "system": "UMLS", "code": "C…" } ],
            "basis": "inferred",
            "rationale": "chronic condition documented across encounters over years",
            "window": { "relation": "last", "value": 2, "unit": "year" }
          },
          {
            "candidate": "HbA1c",
            "intent_title": "Diabetes documentation",
            "name": "Past 6 Months",
            "formula": ["REF_POINT", "REF_POINT - 6M"],
            "coding": [ { "system": "UMLS", "code": "C…" } ],
            "basis": "inferred",
            "rationale": "lab repeated on a monitoring cadence",
            "window": { "relation": "last", "value": 6, "unit": "month" }
          }
        ],
        "tags": []
      }
    ]
  },
  "details": {
    "timing": { "total_llm_seconds": 5.1, "max_llm_seconds": 5.1, "max_llm_seconds_input_index": 0 },
    "usage_metadata": {
      "action_finder": { "...": "..." },
      "query_expansion": { "...": "..." },
      "intent_extraction": { "...": "..." },
      "representative_terms": { "...": "..." },
      "contextual_environment": { "...": "..." }
    },
    "version": "1.5.0",
    "timestamp": "2026-09-23T14:02:11.120334+00:00",
    "source": null,
    "model": "gemini-3.5-flash",
    "location": "us",
    "temporal_mode": "canonical",
    "temporal_shadow": false
  },
  "service": "intent-nature-breakdown-api-1.5.0"
}
```

The windows, names and CUIs above are illustrative: which span a candidate gets is decided by the model per request (and kept consistent by the cadence memory), and which name/CUI a span carries comes from the vocabulary index — none of it is fixed in code.

---

### Additional examples (v1)

**Example 1: Simplified endpoint (`/v1/extract-intents`)**

```python
import requests

payload = {"texts": "Patient with DM, SOB on exertion"}

response = requests.post(
    "http://localhost:8000/v1/extract-intents",
    headers={"Content-Type": "application/json"},
    json=payload,
)

print(response.json())
```

**Response:**
```json
{
  "status": 1,
  "output": {
    "total_inputs": 1,
    "results": [
      {
        "input_index": 0,
        "original_query": "Patient with DM, SOB on exertion",
        "intents": [
          {
            "intent": "Diabetes Mellitus",
            "intent_description": "Patient has diabetes mellitus condition"
          },
          {
            "intent": "Shortness of Breath",
            "intent_description": "Patient experiencing shortness of breath on exertion"
          }
        ]
      }
    ]
  },
  "details": {
    "timing": {
      "total_llm_seconds": 15.867464,
      "max_llm_seconds": 15.867464,
      "max_llm_seconds_input_index": 0
    },
    "usage_metadata": {
      "query_expansion": {
        "prompt_token_count": 120,
        "candidates_token_count": 45,
        "total_token_count": 165,
        "thinking_token_count": 0
      },
      "intent_extraction": {
        "prompt_token_count": 450,
        "candidates_token_count": 280,
        "total_token_count": 730,
        "thinking_token_count": 0
      }
    },
    "version": "1.0.0",
    "timestamp": "2026-01-14T19:40:20.712694",
    "source": null
  },
  "service": "intent-nature-breakdown-api-1.0.0"
}
```

**Example 2: Full breakdown endpoint (`/v1/nature-breakdown`)**

```python
payload = {
    "texts": [
        "Patient with diabetes mellitus",
        "Current medication: metformin 500mg twice daily",
    ]
}

response = requests.post(
    "http://localhost:8000/v1/nature-breakdown",
    headers={"Content-Type": "application/json"},
    json=payload,
)

print(response.json())
```

**Response:**
```json
{
  "status": 1,
  "output": {
    "total_inputs": 2,
    "results": [
      {
        "input_index": 0,
        "original_query": "Patient with diabetes mellitus",
        "expanded_query": "Patient with diabetes mellitus",
        "abbreviations_expanded": [],
        "is_clinical": true,
        "intents": [
          {
            "intent_title": "Diabetes Mellitus",
            "description": "This intent captures information related to the chronic metabolic condition, Diabetes Mellitus, which affects how the body uses blood sugar.",
            "nature": "Condition / Diagnosis",
            "sub_natures": [
              {
                "category_path": "Condition >> Type",
                "atomic_concepts": [
                  "Diabetes Mellitus"
                ]
              }
            ],
            "final_queries": [
              "Diabetes Mellitus"
            ]
          }
        ]
      },
      {
        "input_index": 1,
        "original_query": "Current medication: metformin 500mg twice daily",
        "expanded_query": "Current medication: metformin 500mg twice daily",
        "abbreviations_expanded": [],
        "is_clinical": true,
        "intents": [
          {
            "intent_title": "Metformin Medication",
            "description": "This intent captures information related to the medication metformin, including dosage and frequency.",
            "nature": "Medication / Treatment",
            "sub_natures": [
              {
                "category_path": "Medication >> Dosage",
                "atomic_concepts": [
                  "metformin",
                  "500mg"
                ]
              },
              {
                "category_path": "Medication >> Frequency",
                "atomic_concepts": [
                  "twice daily"
                ]
              }
            ],
            "final_queries": [
              "metformin 500mg",
              "twice daily"
            ]
          }
        ]
      }
    ]
  },
  "details": {
    "timing": {
      "total_llm_seconds": 15.425826,
      "max_llm_seconds": 7.712913,
      "max_llm_seconds_input_index": 0
    },
    "usage_metadata": {
      "query_expansion": {
        "prompt_token_count": 240,
        "candidates_token_count": 90,
        "total_token_count": 330,
        "thinking_token_count": 0
      },
      "intent_extraction": {
        "prompt_token_count": 900,
        "candidates_token_count": 560,
        "total_token_count": 1460,
        "thinking_token_count": 0
      }
    },
    "version": "1.0.0",
    "timestamp": "2026-01-16T17:55:06.018337",
    "source": null
  },
  "service": "intent-nature-breakdown-api-1.0.0"
}
```

**Example 3: Non-clinical query (both endpoints)**

```python
payload = {
    "texts": "PLEASE JOIN THE MEETING TO LEARN MORE ABOUT HOW THIS CAN HELP IMPROVE YOUR WORKFLOW AND PRODUCTIVITY."
}

response = requests.post(
    "http://localhost:8000/v1/nature-breakdown",
    headers={"Content-Type": "application/json"},
    json=payload,
)

print(response.json())
```

**Response:**
```json
{
  "status": 1,
  "output": {
    "total_inputs": 1,
    "results": [
      {
        "input_index": 0,
        "original_query": "PLEASE JOIN THE MEETING TO LEARN MORE ABOUT HOW THIS CAN HELP IMPROVE YOUR WORKFLOW AND PRODUCTIVITY.",
        "expanded_query": "PLEASE JOIN THE MEETING TO LEARN MORE ABOUT HOW THIS CAN HELP IMPROVE YOUR WORKFLOW AND PRODUCTIVITY.",
        "intents": [],
        "is_clinical": false,
        "rejected_reason": "Query is not clinical in nature"
      }
    ]
  },
  "details": {
    "timing": {
      "total_llm_seconds": 1.552227,
      "max_llm_seconds": 1.552227,
      "max_llm_seconds_input_index": 0
    },
    "usage_metadata": {
      "query_expansion": {
        "prompt_token_count": 100,
        "candidates_token_count": 40,
        "total_token_count": 140,
        "thinking_token_count": 0
      },
      "intent_extraction": {
        "prompt_token_count": 200,
        "candidates_token_count": 60,
        "total_token_count": 260,
        "thinking_token_count": 0
      }
    },
    "version": "1.0.0",
    "timestamp": "2026-01-20T15:58:02.197652",
    "source": null
  },
  "service": "intent-nature-breakdown-api-1.0.0"
}
```

**Example 4: Invalid request format (error handling)**

```python
# Incorrect: Using "text" instead of "texts"
payload = {"text": "I am a care team member looking to identify the 4% AHI value"}

response = requests.post(
    "http://localhost:8000/v1/extract-intents",
    headers={"Content-Type": "application/json"},
    json=payload,
)

print(response.status_code)  # 422
print(response.json())
```

**Error Response:**
```json
{
  "detail": [
    {
      "type": "missing",
      "loc": [
        "body",
        "texts"
      ],
      "msg": "Field required",
      "input": {
        "text": "I am a care team member looking to identify the 4% AHI value (Apnea-Hypopnea Index) from a patient's outside documents. I am interested in this value if it comes from a source other than a home sleep apnea test."
      }
    }
  ]
}
```

**Common Request Errors:**
- Missing `texts` field: Returns 422 with validation error
- Empty `texts` array: Returns 400 with "No input text provided" error
- Invalid JSON: Returns 422 with JSON parsing error
- Invalid `location` (not `us` or `us-central1`): Returns 422 with validation error
- Downstream model/location unavailable: Returns 404 (see [Error Responses](#error-responses) for the full status-code table)

**Example 5: Custom model and generation config**

```python
payload = {
    "texts": "Patient with severe chest pain radiating to left arm",
    "model_name": "gemini-2.5-flash",
    "generation_config": {"temperature": 0.1, "top_p": 0.9, "max_output_tokens": 4096},
}

response = requests.post(
    "http://localhost:8000/v1/extract-intents",
    headers={"Content-Type": "application/json"},
    json=payload,
)
```

**Example 7: Multiple texts with paragraphs**

```python
payload = {
    "texts": [
        "Patient with DM and SOB",
        """45-year-old male presents with acute onset abdominal pain, 
worse with movement. No prior episodes. 
Concern for appendicitis.""",
        """Current medications include:
- Metformin 500mg twice daily for diabetes management
- Lisinopril 10mg daily for hypertension""",
    ]
}

response = requests.post(
    "http://localhost:8000/v1/nature-breakdown",
    headers={"Content-Type": "application/json"},
    json=payload,
)
```

**Note:** For multiple inputs:
- `details.timing.total_llm_seconds` = sum of all processing times
- `details.timing.max_llm_seconds` = maximum processing time across all inputs
- `details.timing.max_llm_seconds_input_index` = index of the input that took longest
- `details.usage_metadata` = aggregated token counts across all inputs
