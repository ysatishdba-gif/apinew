# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog],
and this project adheres to [Semantic Versioning].

## [1.5.9] - 2026-09-24

Record types resolved through the transaction-selection pipeline (the standalone
link-code flow), as the service's first coding stage.

### Added
- `app/utils/document_transaction_pipeline.py`: cluster selection → activities of the
  probable documents (`CLUSTER_ACTIVITY_TABLE`) → Stage 1 activity selection
  (transaction-selection service) → documents of the selected activities → Stage 2
  document selection → document classes of the selected documents
  (`CLUSTER_ALL_RESULTS_TABLE`: `document_type` / `document_class` / `node_id`, column
  names configurable) → Stage 3 class selection → node ids as CUIs. Payloads reproduced
  from the pipeline; batched, concurrent, retried (item-count mismatch, 429/5xx) as there;
  BigQuery name comparisons case/whitespace-insensitive.
- Runs for every record type when `TRANSACTION_SELECTION_URL` is set
  (`RECORD_TYPE_RESOLUTION_MODE=auto`; `transactions` / `cluster` force one path); the
  label is tried first, then the vocabulary labels the model matched it to. A record type
  it leaves uncoded continues through the catalog, vocabulary and model stages, so it
  never adds blanks. Stage counts logged per record type; a pipeline failure is logged and
  never fails the request.
- Per-audience identity tokens (`get_auth_headers_for`): the transaction-selection
  service gets its own token; `TRANSACTION_SELECTION_MODEL`, `TRANSACTION_BATCH_SIZE`,
  `TRANSACTION_MAX_WORKERS`, `TRANSACTION_MAX_RETRIES`, `TRANSACTION_RETRY_DELAY_SECONDS`,
  `TRANSACTION_REQUEST_TIMEOUT_SECONDS` settings. Startup log names the active path.

## [1.5.8] - 2026-09-24

Production-readiness pass (execution risks that unit tests with mocked clients cannot see).

### Fixed
- Config: a blank or malformed numeric setting (an `X=` placeholder left in an env file,
  a typo) no longer aborts startup — every numeric setting falls back to its default
  (`_env_int` / `_env_float`), booleans already did.
- BigQuery: both cluster queries order by their output columns
  (`ORDER BY document_class_cui, possible_document_name`), valid with `SELECT DISTINCT`
  on every BigQuery dialect setting; ordering by the source columns can be rejected.
- Record-type model fallback: any exception (not only LLM errors) is caught and logged;
  the fallback can never fail a request.
- Removed a stale `temporal_by_candidate_total` field from the `Vocabulary coverage` log.

### Changed
- Document catalog: name tokens are precomputed once per load, so the abbreviation
  stage is a cheap scan even for large member tables.
- Contract test converts **every** structured-output model the service sends to Gemini
  (query expansion, intent extraction, representative terms, contextual environment ×3,
  Action Finder, record-type fallback) with the installed google-genai.

## [1.5.7] - 2026-09-24

Temporal no longer collapses to the vocabulary default in `/v3` when it should not.

### Fixed
- **Per-candidate windows were keyed by signal text**: the first candidate's window
  for a qualifier ("current", "recent") was reused for every other candidate with the
  same qualifier, and a signal that resolved for no candidate left all of them on the
  default. Canonical mode now resolves per (candidate, signal); the intent level is the
  union across candidates in candidate order (what `temporal` in the response reads).
- A relation word outside `last | within | range` ("past", "previous", "since") no longer
  discards the window: it is read as a single span (or a range when a far edge is given).
- `/v3` matches the model's echo of intent title / candidate name case- and
  punctuation-insensitively and recovers a reworded echo (candidate looked up across
  intents; a single returned intent taken as the one meant) instead of leaving every
  candidate on the default. `/v1` and `/v2` keep their exact match, byte for byte.

### Added
- `Temporal default applied` log per query (WARNING when every candidate defaulted) with
  the defaulted candidates and the index coverage, so a default is diagnosable at once.

## [1.5.6] - 2026-09-24

### Changed
- Configuration: the values that differ per environment (cluster-selection URL, BigQuery
  project / dataset / location / tables, cluster set) are called out in one marked
  `>>> UPDATE PER ENVIRONMENT <<<` block in `app/config.py` and at the top of every
  `config/.env.*` file, with a deployment checklist in the README. No behaviour change.

## [1.5.5] - 2026-09-24

No blank record-type codes while any stage can code them.

### Added
- Document-class stage: a record-type label or alias that names one of the cluster's
  document classes (exact, abbreviation-aware or similar) is coded from the class label.
- Model fallback (`RECORD_TYPE_LLM_FALLBACK`, default on): record types still uncoded
  after the cluster, the catalog and the `/v1` vocabulary are matched by the model against
  the cluster's document classes — one call per request for all of them, labels resolved
  to CUIs through the catalog (`app/utils/record_type_fallback.py`). Usage is logged
  (`Record-type model fallback`); the envelope is unchanged. A remaining blank is logged
  as a WARNING (`Record types uncoded after every stage`).

## [1.5.4] - 2026-09-24

Record-type matching handles abbreviations, acronyms and aliases the way the `/v1` LLM
match did ("prescription record" / "PRES. record"), by code.

### Changed
- Each record type is sent to the cluster service under its own label **and** under the
  vocabulary labels the model matched it to in the contextual-environment call
  (`CLUSTER_ALIAS_TEXTS_MAX`, default 2) — the `/v1` semantic match reused as search text.
- The record-type label and its aliases are also looked up in the document catalog
  directly, so a record type codes even when the service returns nothing usable.
- Document-catalog matching gains an abbreviation-aware token stage
  (`CLUSTER_MEMBER_MATCH_MIN_TOKEN_SCORE`, default 0.6): prefix and contraction
  abbreviations (`pres.` / `prescription`, `outpt` / `outpatient`), plurals, acronyms
  (`H&P` / `History and Physical`), reordered qualifiers; a lone generic token never
  matches. Nothing about what an abbreviation stands for is authored in code.

## [1.5.3] - 2026-09-24

Review pass over accuracy, record-type matching, the temporal prompt, latency and the
output schema.

### Fixed
- **`/v3` structured-output schema was rejected by google-genai** (`CanonicalWindow`
  used `gt=0`, which becomes `exclusiveMinimum`; the Gemini schema conversion refuses
  it). Positivity is now enforced by a validator; a contract test converts every
  response schema with the installed google-genai.
- `/v3` prompt: the "type default" shape example wrongly showed the vocabulary's default
  name as the `signal`; it now shows the Section B qualifier, as `/v1`. New window rule
  for absolute dates/years ("since 2019"): keep the wording, infer a span wide enough to
  cover it.

### Changed
- Record-type coding no longer depends on exact member-name equality: the
  `MEMBER → MAP → CLUSTER` join is loaded once into an in-memory document catalog
  (`CLUSTER_DOCUMENT_CATALOG_TTL_SECONDS`, default 3600) and member names from the
  cluster service are matched by normalised form, punctuation-folded form, then
  character n-gram similarity (`CLUSTER_MEMBER_MATCH_MIN_SIMILARITY`, default 0.75).
  Similarity matches and misses are logged. No BigQuery call per request; the
  per-request query is the fallback when the catalog cannot load. Coding order is
  unchanged (cluster id, member name).
- `CLUSTER_SELECTION_MAX_WORKERS` (default 8) for the per-query cluster-selection fan-out.

## [1.5.2] - 2026-09-23

Contract alignment: the retrieval-signals response is `/v1`'s for every version, and
record-type / temporal **names** are generated exactly as `/v1`; only the matching
behind them differs per version.

### Changed
- `/v3`: the `temporal_by_candidate` field, `details.temporal_mode`,
  `details.temporal_shadow`, `usage_metadata.shadow_contextual_environment` and
  `timing.record_type_resolution_seconds` (1.5.1) are **removed from the response**.
  They are logged instead (`Temporal inference`, `Temporal shadow comparison`,
  `Temporal shadow usage`, `Retrieval signals request timing`). The per-query block
  and `details` are byte-for-byte `/v1`'s shape.
- `/v3` `temporal` uses the `/v1` projection (first non-default window in pipeline
  order, CUIs expanded by vocabulary name). The canonical resolver now always returns a
  **vocabulary entry** (name, CUI, formula): the exact window when the vocabulary has
  it, else the narrowest vocabulary window that contains it, else the widest — the
  "nearest broader, never narrower" rule the list prompt gave the model, applied by
  code. No synthesized window names, no `formula_only` output.
- `/v2` and `/v3`: the contextual-environment prompt is `/v1`'s **with** the
  `KNOWN RECORD TYPES` block (`include_record_type_matching=True`), so record-type
  names are produced by the same instructions as `/v1` (the block carries the
  canonical-name rules). Coding: document cluster first; a name the cluster cannot
  code keeps the `/v1` vocabulary CUIs (`RECORD_TYPE_JSON_FALLBACK`, default `true`) —
  the deterministic fallback.
- Vocabulary entry ranking follows the list prompt's rules by code: entries whose name
  states the span outrank entries that only encode it behind another concept (an age
  band, a questionnaire item); closeness to the model's wording counts only above
  `TEMPORAL_MIN_SIMILARITY` (default now `0.3`), so a shared trigram in a long name no
  longer wins; the nearest-broader search prefers windows some entry name states.
- Removed: `TEMPORAL_BY_CANDIDATE_MAX`, `project_temporal_by_candidate`,
  `project_primary_temporal`. `scripts/replay_retrieval_signals.py` /
  `scripts/evaluate_temporal.py` read only the `/v1`-shaped fields.

## [1.5.1] - 2026-09-23

Hardening from the first live runs of `/v3` (temporal windows missing, a vocabulary
that indexed with zero windows, record types uncoded).

### Changed
- Canonical mode asks the model for a **strict** structured-output schema
  (`ContextualEnvironmentOutputCanonicalSchema`: `window`, `basis`, `rationale` required
  on every temporal entry) and parses with the tolerant models; one malformed window
  costs that entry its window, never the whole contextual environment.
- `TemporalIndex`: formula parsing accepts any reference token (`NOW - 3Y`,
  `ref_date-90d`), single-string formulas (`A | B`, `A .. B`), `{start, end}` objects,
  unit words (`2 years`); unit codes that never co-occur with a unit word in a name are
  derived from the code text (`MIN` → minute, `M` → month); entries whose formula does
  not parse are keyed from the span their **name** states (formula kept verbatim).
  `coverage()` reports `keyed_by_formula` / `keyed_by_name` / `unkeyed` and
  `unkeyed_samples` so a poorly indexing vocabulary is diagnosable from the startup log.
- Document cluster: BigQuery member-name match is case- and whitespace-insensitive on
  both sides; `CLUSTER_MEMBER_NAME_KEYS` (default `member_name`) names the response
  field(s) holding member documents; a service answer with no member names, and names
  without a BigQuery row, are logged with the response shape / unmatched sample.
- `details.timing.record_type_resolution_seconds` on `/v2` and `/v3` (cluster HTTP +
  BigQuery time, separate from `total_llm_seconds`).

### Added
- Optional tag vocabulary for `/v2` and `/v3`: `TAG_GCS_BUCKET`, `TAG_GCS_PATH`,
  `TAG_LOCAL_PATH` (same file shape as the record-type vocabulary). Not configured →
  tags keep names/topics with empty coding, as `/v1`.

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
  - `/v3` renders its own frozen prompt, `app/prompts/v3/contextual_environment.py`
    (`CONTEXTUAL_ENVIRONMENT_PROMPT_V3`, `build_contextual_environment_prompt_v3`): Task 1
    and the Section A / B temporal decision as in v2, the vocabulary-list slot replaced by
    the CANONICAL WINDOW section. The v2 template is untouched and still serves `/v1`/`/v2`.
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
  ≈82.8K to ≈17.5K characters (≈20.7K → ≈4.4K tokens, −79%); the four pipeline prompts
  together from ≈94.8K to ≈29.6K characters (−69%).

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
