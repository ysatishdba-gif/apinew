"""Record-type -> document-class CUIs through the transaction-selection pipeline.

This is the standalone "link code" pipeline (process_search_pair) made
callable per record type inside the service. For one search document name
(the record-type label) and its context (the query's tag names):

  1. CLUSTER SELECTION      cluster-selection service -> probable member documents
  2. ACTIVITIES             BigQuery ACTIVITY table -> candidate activities of those documents
  3. STAGE 1 selection      transaction-selection service: is each activity related? (Yes/No)
  4. RELATED DOCUMENTS      BigQuery ACTIVITY table -> documents of the selected activities
  5. STAGE 2 selection      transaction-selection service: is each document related? (Yes/No)
  6. DOCUMENT CLASSES       BigQuery results table -> classes (+ node id) of the selected documents
  7. STAGE 3 selection      transaction-selection service: is each class related? (Yes/No)

The selected classes' node ids are the CUIs returned for the record type.
Payloads sent to the transaction-selection service are the pipeline's own
(instructions, elements, guidelines, output spec), reproduced as they are;
batch sizes, worker counts, retries and table/column names are settings.
Every BigQuery name comparison is case- and whitespace-insensitive.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests

from app import config
from app.exceptions import ClusterServiceError
from app.utils import context_lonic_document_cluster as cluster


# ---------------------------------------------------------------------------
# Settings (all read at call time so tests / rollouts can change them)
# ---------------------------------------------------------------------------
def _setting(name: str, default: Any) -> Any:
    value = getattr(config, name, default)
    return default if value is None else value


def is_enabled() -> bool:
    """The pipeline runs when a transaction-selection URL is configured and the
    resolution mode is ``transactions`` or ``auto`` (the default)."""
    mode = str(_setting("RECORD_TYPE_RESOLUTION_MODE", "auto")).strip().lower()
    url = str(_setting("TRANSACTION_SELECTION_URL", "")).strip()
    if mode == "cluster":
        return False
    return bool(url)


def _table(name: str) -> str:
    return f"{config.CLUSTER_BQ_PROJECT}.{config.CLUSTER_BQ_DATASET}.{name}"


def _norm_list(names: Iterable[str]) -> list[str]:
    out: list[str] = []
    for n in names:
        k = cluster._norm(n)
        if k and k not in out:
            out.append(k)
    return out


# ---------------------------------------------------------------------------
# Transaction-selection service
# ---------------------------------------------------------------------------
def send_transaction_selection(
    payload: dict[str, Any],
    max_retries: int | None = None,
    retry_delay: float | None = None,
) -> list[dict[str, Any]]:
    """POST one batch to the transaction-selection service and return its
    ``output`` list. Retries an item-count mismatch (400 "incorrect number of
    items"), rate limits and transient server errors, as the pipeline did."""
    url = str(_setting("TRANSACTION_SELECTION_URL", "")).strip()
    if not url:
        raise ClusterServiceError(
            "Transaction-selection service is not configured",
            {"missing_settings": ["TRANSACTION_SELECTION_URL"]},
        )
    retries = int(max_retries or _setting("TRANSACTION_MAX_RETRIES", 5))
    delay = float(retry_delay or _setting("TRANSACTION_RETRY_DELAY_SECONDS", 3))
    timeout = int(_setting("TRANSACTION_REQUEST_TIMEOUT_SECONDS", 300))
    last_error: str = ""
    for attempt in range(1, retries + 1):
        try:
            response = requests.post(
                url,
                json=payload,
                headers=cluster.get_auth_headers_for(url),
                timeout=timeout,
            )
        except requests.RequestException as e:
            last_error = f"{type(e).__name__}: {e}"
            if attempt == retries:
                break
            time.sleep(delay * attempt)
            continue
        if response.status_code in (401, 403):
            cluster.get_auth_headers_for(url, force_refresh=True)
            last_error = f"HTTP {response.status_code}"
            time.sleep(delay)
            continue
        if response.status_code == 200:
            try:
                body = response.json()
            except ValueError:
                last_error = "non-JSON response"
                time.sleep(delay)
                continue
            if not isinstance(body, dict) or "output" not in body:
                last_error = "response has no 'output'"
                time.sleep(delay)
                continue
            output = body["output"]
            return output if isinstance(output, list) else []
        if response.status_code == 400 and "incorrect number of items" in response.text:
            last_error = "incorrect number of items"
            time.sleep(delay)
            continue
        if response.status_code in (429, 500, 502, 503, 504):
            last_error = f"HTTP {response.status_code}"
            time.sleep(delay * attempt)
            continue
        raise ClusterServiceError(
            "Transaction-selection service failed",
            {"status_code": response.status_code, "response": response.text[:500]},
        )
    raise ClusterServiceError(
        f"Transaction-selection service failed after {retries} attempts",
        {"error": last_error},
    )


def _run_batches(
    items: list[Any],
    build_payload: Callable[[list[Any]], dict[str, Any]],
    batch_size: int,
    workers: int,
    stage: str,
) -> list[tuple[Any, dict[str, Any]]]:
    """Send ``items`` in batches (concurrently) and pair each item with its
    answer object. A batch whose answer count does not match, or that fails
    after its retries, contributes nothing (logged), as in the pipeline."""
    if not items:
        return []
    batches = [
        items[i : i + batch_size] for i in range(0, len(items), max(1, batch_size))
    ]

    def one(batch: list[Any]) -> list[tuple[Any, dict[str, Any]]]:
        try:
            results = send_transaction_selection(build_payload(batch))
        except ClusterServiceError as e:
            cluster._log(
                "Transaction-selection batch failed",
                {"stage": stage, "items": len(batch), "error": e.message, **e.details},
            )
            return []
        if len(results) != len(batch):
            cluster._log(
                "Transaction-selection batch answer count mismatch",
                {"stage": stage, "items": len(batch), "answers": len(results)},
            )
            return []
        return [
            (item, r if isinstance(r, dict) else {})
            for item, r in zip(batch, results, strict=True)
        ]

    out: list[tuple[Any, dict[str, Any]]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(batches)))) as pool:
        for chunk in pool.map(one, batches):
            out.extend(chunk)
    return out


def _yes(answer: dict[str, Any]) -> bool:
    return str(answer.get("answer", "")).strip().lower() == "yes"


# ---------------------------------------------------------------------------
# Payloads (the pipeline's own texts, reproduced)
# ---------------------------------------------------------------------------
def _model_name() -> str:
    return str(_setting("TRANSACTION_SELECTION_MODEL", "gemini-2.5-flash"))


def generate_activity_transaction_payload(
    activities: list[dict[str, Any]],
    search_document_name: str,
    search_context: str | None = None,
) -> dict[str, Any]:
    has_context = bool(search_context and search_context.strip())
    if has_context:
        ctx = search_context.strip()
        transactions = [
            [search_document_name, ctx, a["activity_name"], a["activity_definition"]]
            for a in activities
        ]
        instructions = [
            "Each transaction contains exactly 4 elements.",
            "The first element is the search document name.",
            "The second element is the search context.",
            "The third element is the candidate activity name.",
            "The fourth element is the candidate activity definition.",
        ]
        elements = {
            "search_document_name": (
                "The original document name being searched and the target "
                "against which the candidate activity is evaluated."
            ),
            "search_context": (
                "The specific user or clinical context provided to qualify "
                "and guide the relevance evaluation."
            ),
            "activity_name": "The candidate activity name.",
            "activity_definition": "The definition of the candidate activity.",
        }
        guidelines = [
            "Determine whether the candidate activity is related to the search document name in the given search context.",
            "Consider synonyms and synonymous terminology.",
            "Consider procedural relationships between the document and activity.",
            "Consider the clinical context provided by both the document and the explicit search context.",
            "Consider parental and child classifications.",
            "Do not require an exact lexical match.",
            "A semantic equivalent or closely related activity under the provided context should be considered related.",
            "Use Activity Name first and Activity Definition to resolve ambiguity.",
        ]
    else:
        transactions = [
            [search_document_name, a["activity_name"], a["activity_definition"]]
            for a in activities
        ]
        instructions = [
            "Each transaction contains exactly 3 elements.",
            "The first element is the search document name.",
            "The second element is the candidate activity name.",
            "The third element is the candidate activity definition.",
        ]
        elements = {
            "search_document_name": (
                "The original document name being searched and the target "
                "against which the candidate activity is evaluated."
            ),
            "activity_name": "The candidate activity name.",
            "activity_definition": "The definition of the candidate activity.",
        }
        guidelines = [
            "Determine whether the candidate activity is related to the search document name.",
            "Consider synonyms and synonymous terminology.",
            "Consider procedural relationships between the document and activity.",
            "Consider the clinical context of the document and activity.",
            "Consider parental and child classifications.",
            "Do not require an exact lexical match.",
            "A semantic equivalent or closely related activity should be considered related.",
            "Use Activity Name first and Activity Definition to resolve ambiguity.",
        ]
    return {
        "model_name": _model_name(),
        "transactions": transactions,
        "objective": (
            "For each transaction, determine whether the candidate activity "
            "is related to the search document name"
            + (" within the specified search context." if has_context else ".")
            + " The purpose is to retain activities that are relevant to the searched document."
        ),
        "definitions": {"instructions": instructions, "elements": elements},
        "analysis_guidelines": guidelines,
        "output_spec": {
            "instructions": [
                "Return exactly one result for every transaction.",
                "Return results in exactly the same order as the input transactions.",
                "Return Yes when the candidate activity is related to the search document.",
                "Return No when the candidate activity is not related to the search document.",
                "Provide a short reasoning explaining the decision.",
            ],
            "response_fields": {
                "answer": {
                    "Yes": "The candidate activity is related to the search document.",
                    "No": "The candidate activity is not related to the search document.",
                },
                "reasoning": "Brief explanation of the decision.",
            },
        },
    }


def generate_document_transaction_payload(
    document_names: list[str],
    search_document_name: str,
    search_context: str | None = None,
) -> dict[str, Any]:
    has_context = bool(search_context and search_context.strip())
    if has_context:
        ctx = search_context.strip()
        transactions = [[search_document_name, ctx, d] for d in document_names]
        instructions = [
            "Each transaction contains exactly 3 elements.",
            "The first element is the search document name.",
            "The second element is the search context.",
            "The third element is the candidate document name.",
        ]
        elements = {
            "search_document_name": "The original target document name.",
            "search_context": "The specific context guiding the evaluation.",
            "candidate_document_name": "A candidate related document name.",
        }
        guidelines = [
            "Determine whether the candidate document is related to the search document within the provided context.",
            "Consider synonyms and synonymous terminology.",
            "Consider parent-child and hierarchical relationships.",
            "Consider clinically meaningful associations between concepts under the given context.",
            "Consider whether the candidate represents a procedure, finding, condition, measurement, drug, or other concept that is meaningfully related to the target and context.",
            "Do not require an exact lexical match.",
            "Do not infer a relationship merely because the terms share a word.",
            "Return Yes only when there is a reasonable semantic relationship.",
        ]
    else:
        transactions = [[search_document_name, d] for d in document_names]
        instructions = [
            "Each transaction contains exactly 2 elements.",
            "The first element is the search document name.",
            "The second element is the candidate document name.",
        ]
        elements = {
            "search_document_name": "The original target document name.",
            "candidate_document_name": "A candidate related document name.",
        }
        guidelines = [
            "Determine whether the candidate document is related to the search document.",
            "Consider synonyms and synonymous terminology.",
            "Consider parent-child and hierarchical relationships.",
            "Consider clinically meaningful associations between concepts.",
            "Consider whether the candidate represents a procedure, finding, condition, measurement, drug, or other concept that is meaningfully related to the target.",
            "Do not require an exact lexical match.",
            "Do not infer a relationship merely because the terms share a word.",
            "Return Yes only when there is a reasonable semantic relationship.",
        ]
    return {
        "model_name": _model_name(),
        "transactions": transactions,
        "objective": (
            "Determine whether each candidate document name is related to "
            "the search document name"
            + (" within the specified search context." if has_context else ".")
            + " The candidates are possible related documents discovered from activities. Do not require exact wording."
        ),
        "definitions": {"instructions": instructions, "elements": elements},
        "analysis_guidelines": guidelines,
        "output_spec": {
            "instructions": [
                "Return exactly one result for every transaction.",
                "Return results in exactly the same order as the input transactions.",
                "Return Yes if the candidate document is related to the search document.",
                "Return No if it is not related.",
                "Provide a short reasoning.",
            ],
            "response_fields": {
                "answer": {
                    "Yes": "The candidate document is related to the search document.",
                    "No": "The candidate document is not related to the search document.",
                },
                "reasoning": "Brief explanation of the relationship decision.",
            },
        },
    }


def generate_class_transaction_payload(
    classes: list[dict[str, Any]],
    search_document_name: str,
    search_context: str | None = None,
) -> dict[str, Any]:
    has_context = bool(search_context and search_context.strip())
    if has_context:
        ctx = search_context.strip()
        transactions = [
            [search_document_name, ctx, c["document_name"], c["document_class"]]
            for c in classes
        ]
        instructions = [
            "Each transaction contains exactly 4 elements.",
            "The first element is the original search document name.",
            "The second element is the search context.",
            "The third element is the candidate related document name.",
            "The fourth element is the candidate document class.",
        ]
        elements = {
            "search_document_name": "The original target document.",
            "search_context": "The specific context provided to guide the evaluation.",
            "candidate_document_name": "A candidate document discovered through previous stages.",
            "document_class": "The candidate document class to evaluate.",
        }
        guidelines = [
            "Determine whether the candidate document class is related to the search document under the given context.",
            "Use the candidate document name as supporting context for the document class.",
            "Evaluate the meaning of the document class within the provided context.",
            "Consider synonyms and synonymous terminology.",
            "Consider parent-child and hierarchical relationships.",
            "Consider clinically meaningful associations in the given context.",
            "Do not require exact lexical overlap.",
            "Do not reject a relationship solely because terminology differs.",
            "Do not mark Yes merely because the document class belongs to the same broad general domain.",
            "Return Yes only when the document class has a reasonable relationship to the search document within the context.",
        ]
    else:
        transactions = [
            [search_document_name, c["document_name"], c["document_class"]]
            for c in classes
        ]
        instructions = [
            "Each transaction contains exactly 3 elements.",
            "The first element is the original search document name.",
            "The second element is the candidate related document name.",
            "The third element is the candidate document class.",
        ]
        elements = {
            "search_document_name": "The original target document.",
            "candidate_document_name": "A candidate document discovered through previous stages.",
            "document_class": "The candidate document class to evaluate.",
        }
        guidelines = [
            "Determine whether the candidate document class is related to the search document.",
            "Use the candidate document name as supporting context for the document class.",
            "Evaluate the meaning of the document class.",
            "Consider synonyms and synonymous terminology.",
            "Consider parent-child and hierarchical relationships.",
            "Consider clinically meaningful associations.",
            "Do not require exact lexical overlap.",
            "Do not reject a relationship solely because terminology differs.",
            "Do not mark Yes merely because the document class is broadly clinical or belongs to the same very general domain.",
            "Return Yes only when the document class has a reasonable relationship to the search document.",
        ]
    return {
        "model_name": _model_name(),
        "transactions": transactions,
        "objective": (
            "Determine whether the candidate document class is related to "
            "the search document name"
            + (" within the specified search context." if has_context else ".")
            + " The search document remains the original target for every transaction."
        ),
        "definitions": {"instructions": instructions, "elements": elements},
        "analysis_guidelines": guidelines,
        "output_spec": {
            "instructions": [
                "Return exactly one result for every transaction.",
                "Return results in exactly the same order as the input transactions.",
                "Return Yes when the candidate document class is related to the search document.",
                "Return No when the candidate document class is not related.",
                "Provide a short reasoning explaining the decision.",
            ],
            "response_fields": {
                "answer": {
                    "Yes": "The candidate document class is related to the search document.",
                    "No": "The candidate document class is not related to the search document.",
                },
                "reasoning": "Brief explanation of the decision.",
            },
        },
    }


# ---------------------------------------------------------------------------
# BigQuery lookups (name comparisons are case- and whitespace-insensitive)
# ---------------------------------------------------------------------------
def _query(sql: str, name: str, values: list[str]) -> list[dict[str, Any]]:
    from google.cloud import bigquery

    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter(name, "STRING", values)]
    )
    try:
        rows = cluster.get_bq_client().query(sql, job_config=job_config).result()
        return [dict(row.items()) for row in rows]
    except ClusterServiceError:
        raise
    except Exception as e:  # any BigQuery failure is a 503
        raise ClusterServiceError(
            "Transaction pipeline BigQuery lookup failed",
            {"error_type": type(e).__name__, "error": str(e)},
        ) from e


def get_activities_for_documents(document_names: Iterable[str]) -> list[dict[str, Any]]:
    """Candidate activities of the probable documents: one row per
    (document_name, activity_id, activity_name, activity_definition)."""
    names = _norm_list(document_names)
    if not names:
        return []
    table = _table(
        str(
            _setting("CLUSTER_ACTIVITY_TABLE", "SCENARIO_6_ACTIVITY_CODE_DETAILS_ARRAY")
        )
    )
    sql = f"""
        SELECT DISTINCT
            document_name,
            activity_details.activity_id AS activity_id,
            activity_details.activity_name AS activity_name,
            activity_details.activity_definition AS activity_definition
        FROM `{table}`
        CROSS JOIN UNNEST(activity_details) AS activity_details
        WHERE LOWER(TRIM(document_name)) IN UNNEST(@document_names)
        ORDER BY document_name, activity_id
    """
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in _query(sql, "document_names", names):
        activity_name = str(row.get("activity_name") or "").strip()
        activity_id = str(row.get("activity_id") or "").strip()
        if not activity_name or not activity_id:
            continue
        key = (str(row.get("document_name") or "").strip(), activity_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "document_name": key[0],
                "activity_id": activity_id,
                "activity_name": activity_name,
                "activity_definition": str(
                    row.get("activity_definition") or ""
                ).strip(),
            }
        )
    return out


def get_related_document_names(activity_ids: Iterable[str]) -> list[str]:
    """Documents that carry any of the selected activities."""
    ids = [str(a).strip() for a in activity_ids if str(a).strip()]
    ids = list(dict.fromkeys(ids))
    if not ids:
        return []
    table = _table(
        str(
            _setting("CLUSTER_ACTIVITY_TABLE", "SCENARIO_6_ACTIVITY_CODE_DETAILS_ARRAY")
        )
    )
    sql = f"""
        SELECT DISTINCT document_name
        FROM `{table}`
        CROSS JOIN UNNEST(activity_details) AS activity_details
        WHERE CAST(activity_details.activity_id AS STRING) IN UNNEST(@activity_ids)
        ORDER BY document_name
    """
    out: list[str] = []
    for row in _query(sql, "activity_ids", ids):
        name = str(row.get("document_name") or "").strip()
        if name and name not in out:
            out.append(name)
    return out


def get_document_classes_from_results(
    document_names: Iterable[str],
) -> list[dict[str, Any]]:
    """Document classes (with their node id) of the selected documents, from
    the results table: rows of {document_name, document_class, node_id}."""
    names = _norm_list(document_names)
    if not names:
        return []
    table = _table(str(_setting("CLUSTER_ALL_RESULTS_TABLE", "all_results")))
    doc_col = str(_setting("CLUSTER_ALL_RESULTS_DOC_COLUMN", "document_type"))
    class_col = str(_setting("CLUSTER_ALL_RESULTS_CLASS_COLUMN", "document_class"))
    code_col = str(_setting("CLUSTER_ALL_RESULTS_CODE_COLUMN", "node_id"))
    sql = f"""
        SELECT DISTINCT
            {doc_col} AS document_name,
            {class_col} AS document_class,
            CAST({code_col} AS STRING) AS node_id
        FROM `{table}`
        WHERE LOWER(TRIM({doc_col})) IN UNNEST(@document_names)
            AND {class_col} IS NOT NULL
            AND TRIM({class_col}) != ''
        ORDER BY document_name, document_class, node_id
    """
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in _query(sql, "document_names", names):
        item = (
            str(row.get("document_name") or "").strip(),
            str(row.get("document_class") or "").strip(),
            str(row.get("node_id") or "").strip(),
        )
        if not item[1] or item in seen:
            continue
        seen.add(item)
        out.append(
            {"document_name": item[0], "document_class": item[1], "node_id": item[2]}
        )
    return out


# ---------------------------------------------------------------------------
# The pipeline for one (search document name, context)
# ---------------------------------------------------------------------------
def resolve_via_transactions(
    search_document_name: str, search_context: str | None = None
) -> list[dict[str, Any]]:
    """Run the seven steps for one record-type label. Returns
    ``[{"name": document_class, "coding": [{"code": node_id}]}, ...]`` — the
    same hit shape the direct cluster resolver returns — or [] when any stage
    selects nothing (the caller's later stages then apply)."""
    batch = int(_setting("TRANSACTION_BATCH_SIZE", 25))
    workers = int(_setting("TRANSACTION_MAX_WORKERS", 8))
    stages: dict[str, Any] = {"search": search_document_name}

    # 1. cluster selection
    probable = cluster.get_probable_documents(search_document_name, search_context)
    stages["probable_documents"] = len(probable)
    if not probable:
        _log_stages(stages, "no probable documents")
        return []

    # 2-3. activities and their selection
    activities = get_activities_for_documents(probable)
    stages["activities"] = len(activities)
    if not activities:
        _log_stages(stages, "no activities")
        return []
    selected = _run_batches(
        activities,
        lambda b: generate_activity_transaction_payload(
            b, search_document_name, search_context
        ),
        batch,
        workers,
        "activity",
    )
    activity_ids = [a["activity_id"] for a, ans in selected if _yes(ans)]
    stages["activities_selected"] = len(activity_ids)
    if not activity_ids:
        _log_stages(stages, "no activity selected")
        return []

    # 4-5. related documents and their selection
    related = get_related_document_names(activity_ids)
    stages["related_documents"] = len(related)
    if not related:
        _log_stages(stages, "no related documents")
        return []
    selected_docs = _run_batches(
        related,
        lambda b: generate_document_transaction_payload(
            b, search_document_name, search_context
        ),
        batch,
        workers,
        "document",
    )
    yes_docs = [d for d, ans in selected_docs if _yes(ans)]
    stages["documents_selected"] = len(yes_docs)
    if not yes_docs:
        _log_stages(stages, "no document selected")
        return []

    # 6-7. document classes and their selection
    classes = get_document_classes_from_results(yes_docs)
    stages["classes"] = len(classes)
    if not classes:
        _log_stages(stages, "no document classes")
        return []
    selected_classes = _run_batches(
        classes,
        lambda b: generate_class_transaction_payload(
            b, search_document_name, search_context
        ),
        batch,
        workers,
        "class",
    )
    hits: list[dict[str, Any]] = []
    seen: set[str] = set()
    for c, ans in selected_classes:
        if not _yes(ans):
            continue
        code = c["node_id"]
        if not code or code in seen:
            continue
        seen.add(code)
        hits.append({"name": c["document_class"], "coding": [{"code": code}]})
    stages["classes_selected"] = len(hits)
    _log_stages(stages, "ok" if hits else "no class selected")
    return hits


def _log_stages(stages: dict[str, Any], outcome: str) -> None:
    cluster._log(
        "Record-type transaction pipeline",
        {**stages, "outcome": outcome},
        severity="INFO" if outcome == "ok" else "WARNING",
    )
