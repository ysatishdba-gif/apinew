"""Document-cluster record-type resolution for /v2/retrieval-signals.

Resolves a record-type label (e.g. "Radiology Report") to LOINC document-class
CUIs in two hops, replacing the manual record_type_name_to_cui.json lookup
that /v1 uses:

  1. Cluster selection — POST the record-type name (with the query's tag names
     as context) to the cluster-selection service, which returns the probable
     member document names for the configured cluster set.
  2. BigQuery lookup — map those member names through MEMBER -> MAP -> CLUSTER
     to the document class (cluster_label) and its CUI (cluster_id).

Public API (pure functions; no import-time side effects, no network until
called):

    resolve_document_type_cuis(name, context)      -> [{"name", "coding": [{"code"}]}]
    resolve_record_type_cuis(record_types, tags)   -> [{"name", "coding": [{"code"}]}]

Both raise ClusterServiceError when the dependency is not configured or a
call fails; an *authorised but empty* answer is a legitimate "no codes".
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests
from google.auth.exceptions import DefaultCredentialsError
from google.auth.transport.requests import Request
from google.oauth2 import id_token

from app import config
from app.exceptions import ClusterServiceError

# Identity tokens are valid for one hour; refresh comfortably before that.
_TOKEN_TTL_SECONDS = 50 * 60
# Cluster-selection calls are independent per record type; bound the fan-out.
_CLUSTER_SELECTION_MAX_WORKERS = 4

_auth_lock = threading.Lock()
_auth_state: dict[str, Any] = {"token": None, "fetched_at": 0.0}
_bq_client = None
_bq_lock = threading.Lock()

Resolver = Callable[[str, str | None], list[dict[str, Any]]]


# ---------------------------------------------------------------------------
# Configuration guards
# ---------------------------------------------------------------------------
def is_configured() -> bool:
    """True when both the cluster-selection URL and the BigQuery dataset are set."""
    return bool(config.CLUSTER_SELECTION_URL) and bool(config.CLUSTER_BQ_DATASET)


def missing_configuration() -> list[str]:
    missing = []
    if not config.CLUSTER_SELECTION_URL:
        missing.append("CLUSTER_SELECTION_URL")
    if not config.CLUSTER_BQ_DATASET:
        missing.append("CLUSTER_BQ_DATASET")
    return missing


def _require_configured() -> None:
    missing = missing_configuration()
    if missing:
        raise ClusterServiceError(
            "Document-cluster service is not configured",
            {"missing_settings": missing},
        )


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------
def _fetch_identity_token() -> str | None:
    """Identity token for the cluster-selection audience, or None when no
    credential source is available (the caller then sends no Authorization)."""
    try:
        # Cloud Run and other GCP runtimes provide identity credentials through
        # the metadata server / attached service account.
        return id_token.fetch_id_token(Request(), config.CLUSTER_SELECTION_URL) or None
    except DefaultCredentialsError:
        if not config.ALLOW_LOCAL_GCLOUD_AUTH:
            return None

    # Developer setups may use an authenticated gcloud CLI instead.
    gcloud = shutil.which("gcloud")
    if not gcloud:
        return None
    result = subprocess.run(
        [
            gcloud,
            "auth",
            "print-identity-token",
            f"--audiences={config.CLUSTER_SELECTION_URL}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def get_auth_headers(force_refresh: bool = False) -> dict[str, str]:
    """Request headers for the cluster-selection service.

    The identity token is cached and refreshed before it expires (or when the
    caller reports it was rejected). Without a token only Content-Type is sent.
    """
    with _auth_lock:
        now = time.monotonic()
        stale = (now - _auth_state["fetched_at"]) > _TOKEN_TTL_SECONDS
        if force_refresh or stale or _auth_state["token"] is None:
            _auth_state["token"] = _fetch_identity_token()
            _auth_state["fetched_at"] = now
        token = _auth_state["token"]

    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def reset_auth_cache() -> None:
    """Drop the cached identity token (tests / credential rotation)."""
    with _auth_lock:
        _auth_state["token"] = None
        _auth_state["fetched_at"] = 0.0


# ---------------------------------------------------------------------------
# Cluster selection
# ---------------------------------------------------------------------------
def get_context_text(
    search_document_name: str, search_context: str | None = None
) -> str:
    """Tag context when present, otherwise the record-type name itself."""
    if search_context and search_context.strip():
        return search_context.strip()
    return search_document_name


def call_cluster_selection(
    search_document_name: str, search_context: str | None = None
) -> dict[str, Any]:
    """POST one record-type name to the cluster-selection service.

    Returns the raw JSON response, or {} when no identity token is available
    or the service rejects the token (a not-authorised caller gets no codes,
    not an error). Any other non-200 answer raises ClusterServiceError.
    """
    _require_configured()
    headers = get_auth_headers()
    if not headers.get("Authorization"):
        return {}

    payload = {
        "cluster_set": config.CLUSTER_SET,
        "text_list": [search_document_name],
        "context_list": [get_context_text(search_document_name, search_context)],
        "top_k": config.CLUSTER_TOP_K,
        "top_p": config.CLUSTER_TOP_P,
        "combined": config.CLUSTER_COMBINED,
    }

    def _post(hdrs: dict[str, str]) -> requests.Response:
        try:
            return requests.post(
                config.CLUSTER_SELECTION_URL,
                json=payload,
                headers=hdrs,
                timeout=config.CLUSTER_REQUEST_TIMEOUT_SECONDS,
            )
        except requests.RequestException as e:
            raise ClusterServiceError(
                "Cluster-selection request failed",
                {"error_type": type(e).__name__, "error": str(e)},
            ) from e

    response = _post(headers)
    if response.status_code in (401, 403):
        # Token may have expired between refreshes — retry once with a new one.
        headers = get_auth_headers(force_refresh=True)
        if not headers.get("Authorization"):
            return {}
        response = _post(headers)
        if response.status_code in (401, 403):
            return {}
    if response.status_code != 200:
        raise ClusterServiceError(
            "Cluster-selection service returned an error",
            {"status_code": response.status_code, "response": response.text[:500]},
        )
    try:
        return response.json() or {}
    except ValueError as e:
        raise ClusterServiceError(
            "Cluster-selection service returned invalid JSON",
            {"error": str(e)},
        ) from e


def extract_member_names(obj: Any) -> list[str]:
    """Every non-empty `member_name` value anywhere in the response, sorted."""
    member_names: set[str] = set()

    def recursive_extract(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "member_name":
                    if isinstance(child, str):
                        name = child.strip()
                        if name:
                            member_names.add(name)
                    elif isinstance(child, list):
                        for item in child:
                            if isinstance(item, str) and item.strip():
                                member_names.add(item.strip())
                recursive_extract(child)
        elif isinstance(value, list):
            for item in value:
                recursive_extract(item)

    recursive_extract(obj)
    return sorted(member_names)


def get_probable_documents(
    search_document_name: str, search_context: str | None = None
) -> list[str]:
    """Probable member document names for one record-type name."""
    return extract_member_names(
        call_cluster_selection(search_document_name, search_context)
    )


# ---------------------------------------------------------------------------
# BigQuery lookup
# ---------------------------------------------------------------------------
def _table(name: str) -> str:
    return f"{config.CLUSTER_BQ_PROJECT}.{config.CLUSTER_BQ_DATASET}.{name}"


def get_bq_client():
    """Lazily built, process-wide BigQuery client."""
    global _bq_client
    with _bq_lock:
        if _bq_client is None:
            try:
                from google.cloud import bigquery
            except ImportError as e:  # pragma: no cover - dependency guard
                raise ClusterServiceError(
                    "google-cloud-bigquery is not installed", {"error": str(e)}
                ) from e
            _bq_client = bigquery.Client(
                project=config.CLUSTER_BQ_PROJECT, location=config.CLUSTER_BQ_LOCATION
            )
        return _bq_client


def get_document_classes(document_names: Iterable[str]) -> list[dict[str, str]]:
    """member document name -> (document_class_cui, document_class) rows.

    One query for all names; rows are ordered by cluster_id, member name so
    the per-record-type coding order is deterministic.
    """
    names: list[str] = []
    for n in document_names or []:
        n = str(n).strip()
        if n and n not in names:
            names.append(n)
    if not names:
        return []

    _require_configured()
    from google.cloud import bigquery

    query = f"""
        SELECT DISTINCT
            c.cluster_id AS document_class_cui,
            c.cluster_label AS document_class,
            m.MEMBER_NAME AS possible_document_name
        FROM `{_table(config.CLUSTER_MEMBER_TABLE)}` m
        JOIN `{_table(config.CLUSTER_MAP_TABLE)}` b
            ON m.MEMBER_ID = b.MEMBER_ID
        JOIN `{_table(config.CLUSTER_CLUSTER_TABLE)}` c
            ON b.cluster_id = c.cluster_id
        WHERE m.MEMBER_NAME IN UNNEST(@document_names)
        ORDER BY c.cluster_id, m.MEMBER_NAME
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("document_names", "STRING", names)
        ]
    )
    try:
        rows = get_bq_client().query(query, job_config=job_config).result()
        return [
            {
                "document_class_cui": str(row["document_class_cui"] or "").strip(),
                "document_class": str(row["document_class"] or "").strip(),
                "possible_document_name": str(
                    row["possible_document_name"] or ""
                ).strip(),
            }
            for row in rows
        ]
    except ClusterServiceError:
        raise
    except Exception as e:  # any BigQuery failure is a 503
        raise ClusterServiceError(
            "Document-cluster BigQuery lookup failed",
            {"error_type": type(e).__name__, "error": str(e)},
        ) from e


def _classes_to_coding(rows: Iterable[dict[str, str]]) -> list[dict[str, Any]]:
    """Rows -> [{"name": document_class, "coding": [{"code": cui}]}], deduped by CUI."""
    resolved: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        cui = row.get("document_class_cui", "")
        name = row.get("document_class", "")
        if not cui or not name or cui in seen:
            continue
        seen.add(cui)
        resolved.append({"name": name, "coding": [{"code": cui}]})
    return resolved


# ---------------------------------------------------------------------------
# Public resolvers
# ---------------------------------------------------------------------------
def resolve_document_type_cuis(
    search_document_name: str, search_context: str | None = None
) -> list[dict[str, Any]]:
    """Resolve one record-type name through cluster selection and BigQuery."""
    probable_documents = get_probable_documents(search_document_name, search_context)
    if not probable_documents:
        return []
    return _classes_to_coding(get_document_classes(probable_documents))


def _normalize_record_types(record_types: Iterable[Any] | None) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for entry in record_types or []:
        if isinstance(entry, str):
            name = entry.strip()
        elif isinstance(entry, dict):
            name = str(entry.get("name") or entry.get("record_type") or "").strip()
        else:
            continue
        key = name.lower()
        if name and key not in seen:
            seen.add(key)
            names.append(name)
    return names


def _context_from_tags(tags: Iterable[Any] | None) -> str:
    context_names: list[str] = []
    for tag in tags or []:
        if isinstance(tag, str):
            text = tag.strip()
        elif isinstance(tag, dict):
            text = str(tag.get("name") or tag.get("text") or "").strip()
        else:
            text = ""
        if text and text not in context_names:
            context_names.append(text)
    return ", ".join(context_names)


def _coding_from_hits(hits: Iterable[dict[str, Any]] | None) -> list[dict[str, str]]:
    coding: list[dict[str, str]] = []
    seen: set[str] = set()
    for hit in hits or []:
        for code in hit.get("coding") or []:
            code_value = str(code.get("code", "")).strip()
            if code_value and code_value not in seen:
                seen.add(code_value)
                coding.append({"code": code_value})
    return coding


def resolve_record_type_cuis(
    record_types: Iterable[Any] | None,
    tags: Iterable[Any] | None = None,
    resolver: Resolver | None = None,
) -> list[dict[str, Any]]:
    """Resolve record-type objects to CUIs using the tag names as context.

    Input:  [{"name": "Radiology Report", "coding": []}, ...] (or bare strings)
    Tags:   [{"name": "chest X-ray"}, {"name": "imaging"}] (or bare strings)
    Output: [{"name": "Radiology Report", "coding": [{"code": "C..."}]}, ...]

    Each record type keeps its own name; its coding is the ordered, de-duplicated
    set of document-class CUIs the resolver returned for (name, context).
    `resolver(name, context)` defaults to the cluster+BigQuery lookup and can be
    injected (tests, alternative backends). With the default resolver the
    cluster-selection calls run concurrently and one BigQuery query serves all
    record types; the per-record-type result is identical to resolving each
    name on its own.
    """
    names = _normalize_record_types(record_types)
    if not names:
        return []
    context = _context_from_tags(tags)

    if resolver is not None:
        return [
            {"name": name, "coding": _coding_from_hits(resolver(name, context))}
            for name in names
        ]

    # Default path: fan out cluster selection, then a single BigQuery lookup.
    workers = min(_CLUSTER_SELECTION_MAX_WORKERS, len(names))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        probable = list(
            executor.map(lambda n: get_probable_documents(n, context), names)
        )

    all_docs: list[str] = []
    for docs in probable:
        for d in docs:
            if d not in all_docs:
                all_docs.append(d)
    rows = get_document_classes(all_docs) if all_docs else []

    resolved: list[dict[str, Any]] = []
    for name, docs in zip(names, probable, strict=True):
        doc_set = set(docs)
        hits = _classes_to_coding(
            r for r in rows if r.get("possible_document_name") in doc_set
        )
        resolved.append({"name": name, "coding": _coding_from_hits(hits)})
    return resolved
