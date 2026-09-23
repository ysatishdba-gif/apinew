# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog],
and this project adheres to [Semantic Versioning].

## [1.5.0] - 2026-09-23

### Added
- `POST /v3/retrieval-signals` — the `/v2` contract (Action Finder gate, document-cluster
  record types, hints, envelope) with the temporal step in **canonical mode**:
  - The temporal vocabulary is no longer pasted into the contextual-environment prompt.
    At startup it is indexed in-process (`app/utils/temporal_index.py`: formula-keyed
    window index + name similarity, unit codes learned from the file itself). The model
    emits each window as a structured span (`window {relation, value, unit[, to_value,
    to_unit]}`, `basis`, `rationale`); code computes the formula and attaches CUIs
    (`app/utils/temporal_canonical.py`). Windows the vocabulary cannot code keep their
    formula (`resolution: formula_only`) instead of being dropped.
  - The prompt carries only data derived at request time: the codeable-window menu
    (values per unit, from the index), a query-time shortlist of vocabulary names close
    to the query wording, and — when a cadence-memory file is loaded — the windows the
    service itself inferred before for the query's concepts. No clinical cadence, no
    vocabulary content and no examples are authored in code.
  - Queries without a time expression get a per-candidate window inferred from the
    candidate's clinical nature (`basis: inferred`, with `rationale`); the vocabulary's
    declared default remains the fallback (`basis: default`).
  - Response additions: `temporal` becomes the query's explicit span, else the widest
    inferred span; `temporal_by_candidate[]` (candidate, intent_title, name, formula,
    coding, basis, rationale, window); `details.temporal_mode`, `details.temporal_shadow`.
  - Shadow mode (`TEMPORAL_SHADOW_MODE=true`): `/v3` also runs the legacy vocabulary-list
    path and logs per-candidate agreement (`Temporal shadow comparison`); usage is
    reported under `usage_metadata.shadow_contextual_environment`.
  - Cadence memory (`app/utils/cadence_memory.py`): `scripts/build_cadence_memory.py`
    aggregates the `Temporal inference` log events into a versioned JSON (GCS first,
    local fallback, optional) and reports drift against the previous file.
  - Evaluation: `scripts/replay_retrieval_signals.py` (replay queries against two
    endpoint versions) and `scripts/evaluate_temporal.py` (agreement / unresolved rate /
    token and latency deltas, optional LLM judge, CI exit codes).
  - `TemporalIndexUnavailableError` (503 `temporal_index_unavailable`) when the vocabulary
    could not be indexed at startup.
- Config: `TEMPORAL_INDEX_SIMILARITY_PROVIDER`, `TEMPORAL_EMBEDDING_MODEL`,
  `TEMPORAL_INDEX_CACHE_DIR`, `TEMPORAL_MAX_CODES_PER_WINDOW`, `TEMPORAL_MIN_SIMILARITY`,
  `TEMPORAL_MENU_MAX_VALUES`, `TEMPORAL_MENU_UNITS`, `TEMPORAL_SHORTLIST_TOP_N`,
  `TEMPORAL_SHORTLIST_MIN_SCORE`, `TEMPORAL_BY_CANDIDATE_MAX`, `TEMPORAL_SHADOW_MODE`,
  `CADENCE_MEMORY_GCS_BUCKET`, `CADENCE_MEMORY_GCS_PATH`, `CADENCE_MEMORY_LOCAL_PATH`,
  `CADENCE_MEMORY_EXAMPLES_PER_CONCEPT`, `CADENCE_MEMORY_MAX_CONCEPTS`,
  `CADENCE_MEMORY_MIN_SIMILARITY`.

### Changed
- `ContextualIntentPipeline` accepts `temporal_index`, `cadence_memory`,
  `temporal_settings`; `build_context[_async]` / `run_v2[_async]` accept `temporal_mode`
  (`vocab_list`, the default, or `canonical`) and `run_v2[_async]` `temporal_shadow`.
  Existing callers are unchanged.
- Measured on the bundled vocabulary: the contextual-environment prompt drops from
  ≈82.8K to ≈17.4K characters (≈20.7K → ≈4.3K tokens, −79%); the four pipeline prompts
  together from ≈94.8K to ≈29.4K characters (−69%).

### Unchanged
- `/v1/retrieval-signals`, `/v2/retrieval-signals`, `/v1|v2/nature-breakdown`,
  `/v1/extract-intents` and their request/response schemas. `/v2` still uses the
  vocabulary-list temporal path byte-for-byte.

## [1.4.0] - 2026-09-23

### Added
- `POST /v2/retrieval-signals` — same contract as `/v1/retrieval-signals` with:
  - Action Finder gating: optional `actions` (+ `enable_action_finder`) classify each
    query in-process before the pipeline runs; non-matching queries return a skip
    block (`skipped`, `skip_reason`, `skip_reason_detail`). Finder failures fail open.
  - Record-type CUIs resolved per request through the document cluster
    (cluster-selection service + BigQuery) instead of `record_type_name_to_cui.json`;
    the contextual-environment prompt runs without the vocabulary block
    (`include_record_type_matching=False`).
  - `usage_metadata.action_finder` in the response envelope.
- `GET /v2/retrieval-signals/action-types` — selectable Action Finder actions.
- `app/action_finder.py`, `app/prompts/v2/action_finder.py`, `app/utils/action_gate.py`,
  `app/utils/context_lonic_document_cluster.py`; `ClusterServiceError` (503
  `cluster_service_unavailable`).
- Config: `CLUSTER_SELECTION_URL`, `CLUSTER_BQ_DATASET` (required for /v2),
  `CLUSTER_BQ_PROJECT`, `CLUSTER_BQ_LOCATION`, `CLUSTER_*_TABLE`, `CLUSTER_SET`,
  `CLUSTER_TOP_K`, `CLUSTER_TOP_P`, `CLUSTER_COMBINED`,
  `CLUSTER_REQUEST_TIMEOUT_SECONDS`, `ALLOW_LOCAL_GCLOUD_AUTH`, `ACTION_FINDER_MODEL`.
- `google-cloud-bigquery` dependency.

### Changed
- `ContextualIntentPipeline.build_context[_async]` / `run_v2[_async]` accept
  `include_record_type_matching` (default `True`; existing callers unchanged).

### Unchanged
- `/v1/retrieval-signals`, `/v1|v2/nature-breakdown`, `/v1/extract-intents` and their
  request/response schemas.

## [1.0.0] - 2023-07-15

- initial changelog added, changes before this would require looking at git logs

<!-- Links -->
[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html
