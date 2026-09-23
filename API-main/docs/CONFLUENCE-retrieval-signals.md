# Retrieval Signals API — POST /v1/retrieval-signals · POST /v2/retrieval-signals · POST /v3/retrieval-signals

**Service:** intent-nature-breakdown-api · **Status:** /v1 PROD · /v2 DEV · /v3 DEV (shadow rollout)

> `/v2/retrieval-signals` (service 1.4.0) keeps the `/v1` contract byte-for-byte and adds two things: an **Action Finder gate** (optional `actions`) and **record-type CUIs from the document cluster** instead of the JSON vocabulary. See the **/v2** section further down. `/v3/retrieval-signals` (service 1.5.0) is `/v2` with the temporal step in **canonical mode** — the temporal vocabulary is indexed in-process instead of being pasted into the prompt, every candidate gets its own window with `basis` and `rationale`, and queries with no time expression get a window inferred from the candidate's clinical nature. See the **/v3** section at the end. Everything up to the /v2 section describes `/v1` and applies unchanged to `/v2` and `/v3` unless stated there.

## Purpose

A dedicated endpoint that accepts one or more clinical query strings and returns only the normalized retrieval signals — record types, temporal validity window, and representative tags, each with UMLS codes — without the full Nature Breakdown response. Intended for downstream retrieval/routing components (dtree) that need signals, not intents.

## How it works internally

Each query independently runs the existing v2 pipeline with retrieval signals forced on (query expansion → intent extraction ∥ representative terms → contextual environment with folded temporal-CUI matching), and the result is projected into the dtree schema:

| Output field | Source | Code resolution |
|---|---|---|
| `record_types[]` | Union of intent-level `retrieval_signals.record_types` (relevance-ordered by the contextual-environment prompt) | `record_type_name_to_cui.json` vocabulary (GCS first, local fallback). Vocab miss → normalized name with empty `coding`. |
| `temporal` | First CUI-resolved intent-level temporal entry (`time_window` / `codes` / `formula`) | Existing `temporal_name_to_cui.json` vocabulary via folded matching in the contextual-environment step. |
| `tags[]` | v2 `representative_terms` | No tag vocabulary is provisioned yet → `coding: []`. `topics` are heuristically derived from intent nature / category-path tokens. |

## Request

```
POST /v1/retrieval-signals
Content-Type: application/json
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `queries[]` | array | yes | min 1 item |
| `queries[].id` | string | yes | Unique within the request; echoed back to key the result |
| `queries[].text` | array | yes | Complete clinical query strings; each array item is processed as one query and commas are preserved |
| `queries[].record_types` | array | no | Structured hint; merged ahead of derived values, deduped by name |
| `queries[].temporal` | object | no | Structured hint; **overrides** the derived window |
| `queries[].tags` | array | no | Structured hint; merged ahead of derived values |
| `model_name` | string | no | Model override (same semantics as other endpoints) |
| `generation_config` | object | no | Same semantics as other endpoints |

## Response (200)

Standard service envelope (`status` / `output` / `details` / `service`). `output.queries[]` carries one block per input query, keyed by `id`:

```json
{
  "status": 1,
  "output": {
    "total_queries": 1,
    "queries": [
      {
        "id": "q1",
        "text": "chest x-ray reports from the last year",
        "record_types": [
          { "name": "Radiology Report", "coding": [{ "code": "C0034571" }] },
          { "name": "Clinical Note",    "coding": [{ "code": "C1961136" }] }
        ],
        "temporal": {
          "name": "Last One Year",
          "formula": ["REF_POINT", "REF_POINT - 1Y"],
          "coding": [{ "system": "UMLS", "code": "C4722677" }]
        },
        "tags": [
          { "name": "Chest X-ray", "coding": [], "topics": ["imaging", "radiology", "chest"] }
        ]
      }
    ]
  },
  "details": { "timing": { "...": "..." }, "usage_metadata": { "...": "..." }, "version": "1.1.0", "timestamp": "..." },
  "service": "intent-nature-breakdown-api-1.1.0"
}
```

## Error semantics

Identical to `/v1/nature-breakdown` and `/v2/nature-breakdown`:

| Case | Status | Body |
|---|---|---|
| Empty `queries` array, missing `id`/`text` field | 422 | FastAPI/Pydantic validation body |
| Whitespace-only `id`/`text`, duplicate `id` | 422 | `{"detail": {"error": "validation_error", "message": "...", "query_index": N}}` (same shape as `validate_input_texts`) |
| LLM rate limit | 429 | `{"detail": {"error": "rate_limit_exceeded", "message": "...", "details": {}}}` |
| LLM invalid request | 400 | `{"detail": {"error": "invalid_llm_request", ...}}` |
| LLM error | 502 | `{"detail": {"error": "llm_error", ...}}` |
| LLM service unavailable | 503 | `{"detail": {"error": "service_unavailable", ...}}` |
| LLM timeout | 504 | `{"detail": {"error": "llm_timeout", ...}}` |

An LLM failure aborts the whole request via the shared `handle_llm_exception`, exactly like the existing endpoints — the caller retries the batch.

## Example

```bash
curl -sS -X POST "$HOST/v1/retrieval-signals" \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [
      { "id": "q1", "text": "chest x-ray reports from the last year" },
      { "id": "q2", "text": "HbA1c results in past 3 months",
        "temporal": { "name": "Last 3 Months", "formula": ["REF_POINT", "REF_POINT - 3M"] } }
    ]
  }'
```

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `RECORD_TYPE_GCS_BUCKET` | `""` | GCS bucket for the record-type vocabulary |
| `RECORD_TYPE_GCS_PATH` | `Nature_breakdown/record_type_name_to_cui.json` | Blob path |
| `RECORD_TYPE_LOCAL_PATH` | `app/record_type_name_to_cui.json` | Local fallback file |

Vocabulary file shape (identical to the temporal vocab): `{ "Radiology Report": [{"cui": "C0034571"}], ... }`. Lookup is case- and underscore-insensitive (`radiology_report` matches `Radiology Report`). Keys starting with `_` are ignored.

## Operational notes

- **Cost/latency:** each query costs 4 LLM calls (expansion, intents, rep-terms, contextual environment); intents ∥ rep-terms run in parallel. N queries are processed sequentially — budget roughly N × single-query latency. Batch sizes above ~10 should be discouraged client-side until concurrency/limits are added.
- **Security:** endpoint is unauthenticated at the app layer, consistent with the existing endpoints (auth handled at the platform/gateway). No PII expected in queries but query text is logged as a 100-char snippet — same policy as other endpoints.
- **Known gaps / follow-ups:** see the Open Items section of the change summary (tag coding vocabulary, seed record-type CUIs verification, single-temporal schema limitation, per-request query-count cap).

---

# /v2/retrieval-signals

## What changes vs /v1

| Area | /v1 | /v2 |
|---|---|---|
| Request body | `queries[]`, `model_name`, `generation_config`, `location` | Same **plus** optional `actions[]` and `enable_action_finder` (default `true`) |
| Gate | none | When `actions` is non-empty and `enable_action_finder` is true, each query is classified by the Action Finder **before** the pipeline runs |
| `record_types[].coding` | `record_type_name_to_cui.json` vocabulary, matched by the contextual-environment prompt (`record_type_matches`) | Document cluster: record-type name (+ tag names as context) → cluster-selection service → member documents → BigQuery `MEMBER → MAP → CLUSTER` → document-class CUIs. The prompt runs **without** the vocabulary block. |
| `temporal`, `tags` | as /v1 | identical |
| Per-query block | signals block | signals block **or** skip block (gated-out query) |
| `details.usage_metadata` | 4 pipeline steps | same 4 steps **plus** `action_finder` |
| New failure | — | `503 cluster_service_unavailable` when the cluster dependency is not configured or fails |

`/v1` is untouched; both endpoints share the same pipeline, vocabularies, validation and error handling.

## Action Finder gate

```
POST /v2/retrieval-signals
{ "actions": ["information_retrieval"], "queries": [ { "text": ["...", "..."] } ] }
```

- Taxonomy (`GET /v2/retrieval-signals/action-types`): `admin_response`, `conversational_response`, `information_retrieval`, `knowledge_fact_search`, `out_of_scope`. Names are matched case-, space- and hyphen-insensitively; an unknown name is a `422` (`detail.invalid_actions`), never a silent never-match.
- Decision per query: signals are computed only when the identified actions **overlap** `actions` **and** contain `information_retrieval`. Anything else returns a skip block (HTTP 200 — "not serviceable by retrieval" is a result, not a client error).
- The classification is an in-process LLM call through the pipeline's own model client (same model allow-list, location routing, credentials and error mapping); it is never an HTTP hop to a sibling endpoint. `ACTION_FINDER_MODEL` can pin a different model; by default the request's `model_name` (else `MODEL_VERSION`) is used.
- Failure policy: **fail open**. If the Action Finder errors (LLM failure, unparseable answer), the query is processed normally and the decision is logged with `action_finder_error` / `failed_open`.
- Omitting `actions`, sending `[]`, or `enable_action_finder: false` → no gating, no extra LLM call.

Skip block (contract):

```json
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
```

`skip_reason` is the machine-readable list of actions the Action Finder identified; `skip_reason_detail` is the human-readable explanation. The signal fields keep the same shape as a computed block so clients parse every entry the same way.

## Document-cluster record types

1. The pipeline's record-type names are collected exactly as for `/v1` (union of intent-level `retrieval_signals.record_types`, relevance order, de-duplicated, `snake_case` normalised to display form).
2. For each name, `POST CLUSTER_SELECTION_URL` with `{cluster_set, text_list: [name], context_list: [tag names joined by ", " — or the name itself when there are no tags], top_k, top_p, combined}`; every `member_name` in the answer is a probable document. Calls for one query's record types run concurrently.
3. One BigQuery query maps all probable documents to `(cluster_id AS document_class_cui, cluster_label AS document_class)`; each record type's `coding` is the ordered, de-duplicated list of CUIs of *its* documents. A name the cluster cannot place keeps `coding: []` and is logged in the per-query `Vocabulary coverage` entry (`record_type_source: document_cluster`).
4. Hints are merged afterwards with the same precedence as `/v1`.

Auth: the identity token for the cluster-selection audience comes from the workload's service account (Cloud Run metadata server). It is cached, refreshed before expiry and refreshed once more on `401/403`; with no token available the service is not called and record types come back uncoded. `ALLOW_LOCAL_GCLOUD_AUTH=true` lets developer machines fall back to `gcloud auth print-identity-token`.

## Configuration (additional)

| Env var | Default | Purpose |
|---|---|---|
| `CLUSTER_SELECTION_URL` | `""` (**required**) | Cluster-selection endpoint / token audience |
| `CLUSTER_BQ_DATASET` | `""` (**required**) | BigQuery dataset with the document-cluster tables |
| `CLUSTER_BQ_PROJECT` | `PROJECT_ID` | BigQuery project |
| `CLUSTER_BQ_LOCATION` | `US` | BigQuery location |
| `CLUSTER_MEMBER_TABLE` / `CLUSTER_MAP_TABLE` / `CLUSTER_CLUSTER_TABLE` | `SCENARIO_6_MEMBER` / `SCENARIO_6_MAP` / `SCENARIO_6_CLUSTER` | Table names |
| `CLUSTER_SET` | `loinc_document_v001` | Cluster set |
| `CLUSTER_TOP_K` / `CLUSTER_TOP_P` / `CLUSTER_COMBINED` | `10` / `20` / `true` | Cluster-selection parameters |
| `CLUSTER_REQUEST_TIMEOUT_SECONDS` | `120` | HTTP timeout per cluster-selection call |
| `ALLOW_LOCAL_GCLOUD_AUTH` | `false` | Developer-only gcloud token fallback |
| `ACTION_FINDER_MODEL` | `""` | Model override for the Action Finder (must be an allowed model) |

Startup logs `Document-cluster record-type resolution configured` or a `WARNING` naming the missing settings; a `/v2` request then answers `503` before spending any LLM call.

## Error semantics (additional)

| Case | Status | Body |
|---|---|---|
| Unknown action in `actions` | 422 | `{"detail": {"error": "validation_error", "message": "Unknown action type(s): [...]", "invalid_actions": [...]}}` |
| Cluster dependency not configured | 503 | `{"detail": {"error": "cluster_service_unavailable", "message": "Document-cluster service is not configured", "details": {"missing_settings": [...]}}}` |
| Cluster-selection / BigQuery failure | 503 | `{"detail": {"error": "cluster_service_unavailable", "message": "...", "details": {"error_type": "...", ...}}}` |

## Operational notes (additional)

- **Cost/latency:** with `actions` supplied each query costs **one extra LLM call** (the gate) ahead of the four pipeline calls, plus one cluster-selection HTTP call per record type and one BigQuery query per query. Gated-out queries cost only the gate call. Queries are still processed sequentially.
- **Migration:** clients can switch from `/v1` to `/v2` without changing their request or response parsing; the only new per-query key they may see is the skip block (`skipped: true`) when they opt into `actions`.

---

# /v3/retrieval-signals

## Why

The `/v1`/`/v2` contextual-environment prompt pastes the whole temporal vocabulary (≈1,500 `[id, name]` rows) into every call so the model can pick an id. That block is ≈70K of the ≈83K characters of the prompt (≈20.7K tokens), it is the dominant latency and cost of the endpoint, and the model still has to find the right row among 1,500 near-duplicates. `/v3` removes the list and makes the model do the one thing it is good at — saying *what span* a candidate needs — while code does the lookup.

## What changes vs /v2

| Area | /v2 | /v3 |
|---|---|---|
| Request body, gate, record types, hints, envelope | — | identical |
| Temporal vocabulary in the prompt | full `[id, name]` list (≈70K chars) | **none**. The prompt carries a codeable-window menu (values per unit, from the index), a shortlist of vocabulary names close to the query wording (only when the query states a span) and, when a cadence memory is loaded, the windows the service inferred before for the query's concepts. All derived at request time; nothing authored. |
| What the model emits per window | `selected_id` from the list | `window {relation: last\|within\|range, value, unit[, to_value, to_unit]}` + `basis` (`explicit` \| `inferred`) + `rationale` |
| Resolution | id → name → CUI | window → `REF_POINT` formula → vocabulary entries with that formula (up to `TEMPORAL_MAX_CODES_PER_WINDOW` CUIs, ranked by closeness to the model's wording). An uncodeable window is **kept** with its formula and `coding: []`. |
| Queries without a time expression | one default window (`Recent`) for everything | one window **per candidate**, inferred from its clinical nature (`basis: inferred`, `rationale`); the vocabulary's declared default only when nothing can be inferred (`basis: default`) |
| `temporal` | model's pick | the query's explicit window, else the **widest** inferred window (same object shape) |
| New per-query field | — | `temporal_by_candidate[]`: `candidate`, `intent_title`, `name`, `formula`, `coding`, `basis`, `rationale`, `window` |
| `details` | — | `temporal_mode: canonical`, `temporal_shadow` |
| New failure | — | `503 temporal_index_unavailable` when the vocabulary could not be indexed at startup (`/v1`, `/v2` unaffected) |
| Contextual-environment prompt size (bundled vocabulary) | ≈82.8K chars / ≈20.7K tokens | ≈17.5K chars / ≈4.4K tokens (**−79%**); all four pipeline prompts −69% |

## How canonical mode works internally

1. **Startup — index, not a list.** The loaded temporal vocabulary (same `TEMPORAL_*` sources as `/v1`) is parsed: every entry's formula becomes a canonical window key (`last 6 months`, `between 6 and 12 months`, …), unit codes (`Y`, `M`, `W`, `D`, `H`, `MIN`, `S`) are learned by matching formula codes against the unit words in entry names, and entry names get a character n-gram similarity index (or Vertex embeddings when `TEMPORAL_INDEX_SIMILARITY_PROVIDER=vertex`). Coverage is logged (`Temporal index built for /v3/retrieval-signals`: entries, keyed entries, distinct windows, learned unit codes, vocabulary hash).
2. **Request — small prompt.** `/v3` renders its own template (`app/prompts/v3/contextual_environment.py`; `/v1`/`/v2` keep the v2 template unchanged). Its CANONICAL WINDOW section tells the model to write each window as a structured span with a basis and a rationale, to transcribe stated spans verbatim (36 months stays 36 months), to give qualifier/ordering words ("current", "history of", "most recent") the concrete span they imply *for that candidate*, to prefer a wider span when unsure, and lists which spans the vocabulary can code. The structured-output schema enforces the window shape.
3. **Resolution — code, not the model.** Each window's formula is computed and looked up in the index. Ties between vocabulary names for the same formula are ranked by similarity to the model's wording. No entry → `resolution: formula_only`, formula kept, `coding: []`. No window at all → vocabulary default (`basis: default`).
4. **Audit — one log event per window.** `Temporal inference` (query, intent, candidate, basis, rationale, window, formula, CUI, index version, memory version). These events are the only input to the cadence memory and to the evaluation scripts.
5. **Consistency — cadence memory.** `scripts/build_cadence_memory.py` aggregates the inferred windows per concept from exported logs into a versioned JSON (with a drift report against the previous file; exit `3` when a concept's distribution moved more than `--max-shift`). The service loads it at startup (GCS first, local fallback; absent → `/v3` runs without it) and shows the query's concepts their most common past windows as a *prior*, so the same concept gets the same window across calls and model versions. Learned from traffic, never edited by hand.

## Rollout

1. **Shadow.** Deploy with `TEMPORAL_SHADOW_MODE=true`. Every `/v3` call also runs the legacy list path and logs `Temporal shadow comparison` (per candidate: legacy CUI/formula vs canonical CUI/formula, agreement flag). Costs a second contextual-environment call per query; for rollout only.
2. **Replay + evaluate.** `scripts/replay_retrieval_signals.py --versions v2 v3` against the same queries, then `scripts/evaluate_temporal.py --baseline v2 --candidate v3 --min-agreement 0.995`: CUI/formula agreement on queries where `/v2` coded a window, unresolved rates, mean prompt tokens and latency per version, and — with `--judge-model` — an LLM-judge accept rate over inferred windows with only the rejections listed for a human to sample. Exit `2` below the threshold, for CI.
3. **Cadence memory.** Export the `Temporal inference` events, build the memory file, upload it, restart. Re-run the job on a schedule; a run that trips the drift gate is reviewed before upload.
4. **Switch.** Point clients at `/v3`, set `TEMPORAL_SHADOW_MODE=false`. `/v2` stays available unchanged.

## Configuration (additional)

| Env var | Default | Purpose |
|---|---|---|
| `TEMPORAL_INDEX_SIMILARITY_PROVIDER` | `lexical` | `lexical` (in-process, no network) or `vertex` (embeddings, disk-cached, lexical fallback) |
| `TEMPORAL_EMBEDDING_MODEL` / `TEMPORAL_INDEX_CACHE_DIR` | `""` | Vertex provider model and cache directory |
| `TEMPORAL_MAX_CODES_PER_WINDOW` / `TEMPORAL_MIN_SIMILARITY` | `2` / `0` | CUIs kept per resolved window; similarity floor for the extra ones |
| `TEMPORAL_MENU_MAX_VALUES` / `TEMPORAL_MENU_UNITS` | `12` / `day,week,month,year` | Codeable-window menu shown to the model |
| `TEMPORAL_SHORTLIST_TOP_N` / `TEMPORAL_SHORTLIST_MIN_SCORE` | `8` / `0.3` | Query-time shortlist of vocabulary names |
| `TEMPORAL_BY_CANDIDATE_MAX` | `8` | Cap on `temporal_by_candidate` entries per query |
| `TEMPORAL_SHADOW_MODE` | `false` | Also run the legacy path and log agreement |
| `CADENCE_MEMORY_GCS_BUCKET` / `CADENCE_MEMORY_GCS_PATH` / `CADENCE_MEMORY_LOCAL_PATH` | `""` / `Nature_breakdown/cadence_memory.json` / `app/cadence_memory.json` | Cadence-memory file (optional) |
| `CADENCE_MEMORY_EXAMPLES_PER_CONCEPT` / `CADENCE_MEMORY_MAX_CONCEPTS` / `CADENCE_MEMORY_MIN_SIMILARITY` | `3` / `10` / `0.6` | How much of the memory reaches the prompt |

## Error semantics (additional)

| Case | Status | Body |
|---|---|---|
| Temporal vocabulary could not be indexed at startup | 503 | `{"detail": {"error": "temporal_index_unavailable", "message": "Temporal index is not available; /v3 cannot run canonical temporal mode", "details": {"temporal_mode": "canonical"}}}` |

## Operational notes (additional)

- **Cost/latency:** the contextual-environment call — the largest of the four — sends ≈79% fewer prompt characters; the other three calls are unchanged. Shadow mode adds one legacy contextual-environment call per query while it is on.
- **Accuracy guard-rails:** explicit spans are transcribed, never converted or narrowed; inferred windows prefer the wider span; nothing is dropped for lack of a vocabulary entry; every window carries `basis` and `rationale` for audit; agreement with `/v2` is measured, not assumed.
- **Migration:** a `/v2` client can call `/v3` with the same request and parse the same response; the only additions are `temporal_by_candidate` per query and `temporal_mode` / `temporal_shadow` in `details`.
