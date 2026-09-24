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

import json
import logging
import re
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
# Cluster-selection calls are independent per record type; bound the fan-out
# (CLUSTER_SELECTION_MAX_WORKERS).
_CLUSTER_SELECTION_MAX_WORKERS = 8

# Document catalog: the MEMBER -> MAP -> CLUSTER join loaded once per process
# (CLUSTER_DOCUMENT_CATALOG_TTL_SECONDS) so a request costs no BigQuery round
# trip and member names can be matched by normalised form and similarity
# rather than by exact string equality.
_catalog_lock = threading.Lock()
_catalog_state: dict[str, Any] = {"catalog": None, "loaded_at": 0.0}

_auth_lock = threading.Lock()
_auth_state: dict[str, Any] = {"token": None, "fetched_at": 0.0}
_bq_client = None
_bq_lock = threading.Lock()

Resolver = Callable[[str, str | None], list[dict[str, Any]]]

# Diagnostics: the service logger is injected at startup (set_logger); until
# then, or outside the app, a stdlib logger receives the same events.
_std_logger = logging.getLogger(__name__)
_diag: dict[str, Any] = {"lg": None, "base_log": {}}


def set_logger(lg: Any, base_log: dict[str, Any] | None = None) -> None:
    _diag["lg"] = lg
    _diag["base_log"] = dict(base_log or {})


def _log(message: str, data: dict[str, Any], severity: str = "WARNING") -> None:
    lg = _diag["lg"]
    if lg is not None:
        try:
            lg.log_struct(
                message=message,
                structured_data={**_diag["base_log"], **data},
                severity=severity,
            )
            return
        except Exception as log_err:  # noqa: BLE001 — diagnostics never break resolution
            _std_logger.debug("structured logger failed: %s", log_err)
    _std_logger.log(
        logging.getLevelName(severity) if isinstance(severity, str) else severity,
        "%s %s",
        message,
        json.dumps(data, ensure_ascii=False, default=str)[:2000],
    )


def _norm(name: Any) -> str:
    """Case- and whitespace-insensitive key for document names."""
    return re.sub(r"\s+", " ", str(name or "").strip().lower())


def _fold(name: Any) -> str:
    """Alphanumeric-only key: punctuation, separators and spacing differences
    between the service's member names and the table's do not matter."""
    return re.sub(r"[^a-z0-9]+", " ", str(name or "").lower()).strip()


# Function words that carry no document identity ("history AND physical",
# "summary OF care"); generic English, not vocabulary content.
_STOP_TOKENS = frozenset(
    {"a", "an", "and", "the", "of", "for", "in", "on", "to", "with", "or", "at", "by"}
)


def _name_tokens(name: Any) -> list[str]:
    return [t for t in _fold(name).split() if t not in _STOP_TOKENS]


def _singular(tok: str) -> str:
    if len(tok) > 3 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def _is_subsequence(short: str, long_: str) -> bool:
    it = iter(long_)
    return all(ch in it for ch in short)


def _abbreviates(short: str, long_: str) -> bool:
    """``short`` is an abbreviation of ``long_``: a prefix of at least two
    letters ("pres" / "prescription", "rad" / "radiology"), or a contraction
    that keeps the first two letters and drops letters in order ("outpt" /
    "outpatient", "dischg" / "discharge"). Conventions that share no letters
    ("hx" / "history") are not guessed; the data supplies those aliases."""
    if len(short) < 2 or len(long_) <= len(short):
        return False
    if long_.startswith(short):
        return True
    return len(short) >= 3 and long_[:2] == short[:2] and _is_subsequence(short, long_)


def _token_equivalent(a: str, b: str) -> bool:
    """Two name tokens name the same word when they are equal, differ only in
    number, or one abbreviates the other."""
    if a == b:
        return True
    sa, sb = _singular(a), _singular(b)
    if sa == sb:
        return True
    short, long_ = (sa, sb) if len(sa) <= len(sb) else (sb, sa)
    return _abbreviates(short, long_)


def _acronym(tokens: list[str]) -> str:
    return "".join(t[0] for t in tokens if t)


def token_match_score(a: Any, b: Any) -> float:
    """How well two document names agree token by token, abbreviation-aware.

    Every token of the shorter name must be matched by a token of the longer
    one (equal, singular/plural, or prefix abbreviation), or the shorter name
    must be the acronym of the longer ("H&P" / "History and Physical"). The
    score is matched tokens / tokens of the longer name, so a name that is a
    subset of a longer qualified name scores below 1 but above 0 ("consultation
    note" vs "consultation note outpatient" -> 0.67); a single generic token
    ("note") never carries a match on its own."""
    return _token_score(_name_tokens(a), _name_tokens(b))


def _token_score(ta: list[str], tb: list[str]) -> float:
    if not ta or not tb:
        return 0.0
    short, long_ = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    if len(long_) > 1 and (len(short) == 1 or all(len(t) == 1 for t in short)):
        # "hp" / "h p" (from "H&P") / "ds": only the initials of the longer
        # name; one ordinary token ("note") never carries a longer name.
        return 1.0 if "".join(short) == _acronym(long_) else 0.0
    remaining = list(long_)
    matched = 0
    for tok in short:
        for i, cand in enumerate(remaining):
            if _token_equivalent(tok, cand):
                matched += 1
                del remaining[i]
                break
        else:
            return 0.0  # a token of the shorter name has no counterpart
    return matched / len(long_)


def _member_name_keys() -> tuple[str, ...]:
    raw = getattr(config, "CLUSTER_MEMBER_NAME_KEYS", "member_name") or "member_name"
    return tuple(k.strip() for k in str(raw).split(",") if k.strip())


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
def _fetch_identity_token(audience: str | None = None) -> str | None:
    """Identity token for ``audience`` (default: the cluster-selection URL),
    or None when no credential source is available (the caller then sends no
    Authorization). Cloud Run tokens are audience-specific, so each service
    URL gets its own."""
    audience = audience or config.CLUSTER_SELECTION_URL
    try:
        # Cloud Run and other GCP runtimes provide identity credentials through
        # the metadata server / attached service account.
        return id_token.fetch_id_token(Request(), audience) or None
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
            f"--audiences={audience}",
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


_audience_tokens: dict[str, dict[str, Any]] = {}


def get_auth_headers_for(audience: str, force_refresh: bool = False) -> dict[str, str]:
    """Request headers for another service URL (the transaction-selection
    service): one cached identity token per audience, refreshed before expiry
    or when the caller reports a rejection."""
    if not audience or audience == config.CLUSTER_SELECTION_URL:
        return get_auth_headers(force_refresh=force_refresh)
    with _auth_lock:
        state = _audience_tokens.setdefault(
            audience, {"token": None, "fetched_at": 0.0}
        )
        now = time.monotonic()
        stale = (now - state["fetched_at"]) > _TOKEN_TTL_SECONDS
        if force_refresh or stale or state["token"] is None:
            state["token"] = _fetch_identity_token(audience)
            state["fetched_at"] = now
        token = state["token"]
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def reset_auth_cache() -> None:
    """Drop the cached identity tokens (tests / credential rotation)."""
    with _auth_lock:
        _auth_state["token"] = None
        _auth_state["fetched_at"] = 0.0
        _audience_tokens.clear()


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


def extract_member_names(obj: Any, keys: tuple[str, ...] | None = None) -> list[str]:
    """Every non-empty member-name value anywhere in the response, sorted.
    The key(s) holding the name default to ``member_name`` and can be set
    with ``CLUSTER_MEMBER_NAME_KEYS`` (comma-separated) when the service
    answers with a different field name."""
    member_names: set[str] = set()
    wanted = keys or _member_name_keys()

    def recursive_extract(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in wanted:
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
    response = call_cluster_selection(search_document_name, search_context)
    names = extract_member_names(response)
    if not names and response:
        # The service answered but nothing under the configured member-name
        # key: log the shape so a field-name mismatch is visible at once.
        _log(
            "Cluster selection returned no member names",
            {
                "record_type": search_document_name,
                "member_name_keys": list(_member_name_keys()),
                "response_top_level_keys": (
                    sorted(response.keys()) if isinstance(response, dict) else None
                ),
                "response_preview": json.dumps(response, ensure_ascii=False)[:400],
            },
        )
    return names


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


class DocumentCatalog:
    """In-memory copy of the MEMBER -> MAP -> CLUSTER join: every member
    document name with its document class(es). Matching a member name the
    cluster service returned is done here, in three steps that never require
    the two systems to agree on spelling: normalised equality (case,
    whitespace), folded equality (punctuation, separators), then character
    n-gram similarity above CLUSTER_MEMBER_MATCH_MIN_SIMILARITY (aliases,
    abbreviations, reordered qualifiers). Every non-exact match is logged."""

    def __init__(self, rows: Iterable[dict[str, str]]):
        self.rows: list[dict[str, str]] = []
        self._by_norm: dict[str, list[int]] = {}
        self._by_fold: dict[str, list[int]] = {}
        names: list[str] = []
        for row in rows:
            name = str(row.get("possible_document_name") or "").strip()
            cui = str(row.get("document_class_cui") or "").strip()
            if not name or not cui:
                continue
            i = len(self.rows)
            self.rows.append(
                {
                    "document_class_cui": cui,
                    "document_class": str(row.get("document_class") or "").strip(),
                    "possible_document_name": name,
                }
            )
            self._by_norm.setdefault(_norm(name), []).append(i)
            self._by_fold.setdefault(_fold(name), []).append(i)
            names.append(name)
        # One similarity index over the distinct folded names; tokens are
        # precomputed once so the abbreviation stage is a cheap per-name scan.
        self._sim_names = sorted(self._by_fold)
        self._name_tokens = {n: _name_tokens(n) for n in self._sim_names}
        self._similarity = None
        if self._sim_names:
            from app.utils.temporal_index import LexicalSimilarity

            self._similarity = LexicalSimilarity(self._sim_names)
        # Document CLASSES (cluster labels -> CUIs): the last deterministic
        # stage matches a record-type label against these, and the model
        # fallback chooses among them.
        self._class_cuis: dict[str, list[str]] = {}
        self._class_by_fold: dict[str, list[str]] = {}
        for row in self.rows:
            label = row["document_class"]
            if not label:
                continue
            cuis = self._class_cuis.setdefault(label, [])
            if row["document_class_cui"] not in cuis:
                cuis.append(row["document_class_cui"])
            labels = self._class_by_fold.setdefault(_fold(label), [])
            if label not in labels:
                labels.append(label)
        self._class_labels = list(self._class_cuis)
        self._class_similarity = None
        if self._class_by_fold:
            self._class_fold_keys = sorted(self._class_by_fold)
            self._class_similarity = LexicalSimilarity(self._class_fold_keys)
        self.loaded_at = time.time()

    # ---- document classes ---------------------------------------------------
    def class_labels(self) -> list[str]:
        return list(self._class_labels)

    def cuis_for_class_label(self, label: str) -> list[str]:
        """CUIs of the class whose label equals ``label`` (case, whitespace and
        punctuation-insensitive); [] for a label the catalog does not have."""
        exact = self._class_cuis.get(str(label).strip())
        if exact:
            return list(exact)
        out: list[str] = []
        for lab in self._class_by_fold.get(_fold(label), []):
            for cui in self._class_cuis.get(lab, []):
                if cui not in out:
                    out.append(cui)
        return out

    def match_class(
        self, name: str, min_similarity: float, min_token_score: float | None = None
    ) -> tuple[list[str], str, float]:
        """A record-type label matched against the document-class labels with
        the same stages as member names. Returns (CUIs, stage, score)."""
        if not self._class_by_fold:
            return [], "none", 0.0
        cuis = self.cuis_for_class_label(name)
        if cuis:
            return cuis, "class", 1.0
        floor = (
            float(getattr(config, "CLUSTER_MEMBER_MATCH_MIN_TOKEN_SCORE", 0.6) or 0)
            if min_token_score is None
            else min_token_score
        )
        if floor > 0:
            best_key, best_score = None, 0.0
            for key in self._class_fold_keys:
                sc = token_match_score(name, key)
                if sc > best_score:
                    best_key, best_score = key, sc
            if best_key is not None and best_score >= floor:
                cuis = []
                for lab in self._class_by_fold[best_key]:
                    cuis.extend(c for c in self._class_cuis[lab] if c not in cuis)
                return cuis, "class_tokens", float(best_score)
        if self._class_similarity is None or min_similarity <= 0:
            return [], "none", 0.0
        scores = self._class_similarity.scores(_fold(name))
        best = max(range(len(scores)), key=lambda i: scores[i])
        if scores[best] >= min_similarity:
            cuis = []
            for lab in self._class_by_fold[self._class_fold_keys[best]]:
                cuis.extend(c for c in self._class_cuis[lab] if c not in cuis)
            return cuis, "class_similar", float(scores[best])
        return [], "none", float(scores[best])

    def __len__(self) -> int:
        return len(self.rows)

    def match(
        self, name: str, min_similarity: float, min_token_score: float | None = None
    ) -> tuple[list[int], str, float]:
        """Row indexes for one document name and how they were found:
        ``exact`` | ``folded`` | ``tokens`` | ``similar`` | ``none``.

        ``tokens`` is the abbreviation-aware stage ("PRES. record" ->
        "Prescription Record", "H&P" -> "History and Physical", "rad report"
        -> "Radiology Report"); ``similar`` the character n-gram stage for
        spelling variants. Both are data-independent: nothing about what an
        abbreviation stands for is written here."""
        idxs = self._by_norm.get(_norm(name))
        if idxs:
            return idxs, "exact", 1.0
        idxs = self._by_fold.get(_fold(name))
        if idxs:
            return idxs, "folded", 1.0
        floor = (
            float(getattr(config, "CLUSTER_MEMBER_MATCH_MIN_TOKEN_SCORE", 0.6) or 0)
            if min_token_score is None
            else min_token_score
        )
        if floor > 0 and self._sim_names:
            best_name, best_score = None, 0.0
            query_tokens = _name_tokens(name)
            for cand in self._sim_names:
                sc = _token_score(query_tokens, self._name_tokens[cand])
                if sc > best_score or (
                    sc == best_score and best_name and len(cand) < len(best_name)
                ):
                    best_name, best_score = cand, sc
            if best_name is not None and best_score >= floor:
                return self._by_fold[best_name], "tokens", float(best_score)
        if self._similarity is None or min_similarity <= 0:
            return [], "none", 0.0
        scores = self._similarity.scores(_fold(name))
        best = max(range(len(scores)), key=lambda i: scores[i])
        if scores[best] >= min_similarity:
            return self._by_fold[self._sim_names[best]], "similar", float(scores[best])
        return [], "none", float(scores[best])

    def lookup(
        self, document_names: Iterable[str], min_similarity: float
    ) -> list[dict[str, str]]:
        """Rows for the given member names (each row's
        ``possible_document_name`` is the NAME AS REQUESTED, so callers can
        group by the names they sent). Deterministic order."""
        out: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        misses: list[str] = []
        fuzzy: list[dict[str, Any]] = []
        for name in document_names:
            idxs, how, score = self.match(name, min_similarity)
            if not idxs:
                misses.append(name)
                continue
            if how in ("tokens", "similar"):
                fuzzy.append(
                    {
                        "requested": name,
                        "matched": self.rows[idxs[0]]["possible_document_name"],
                        "how": how,
                        "score": round(score, 3),
                    }
                )
            for i in sorted(
                idxs, key=lambda i: (self.rows[i]["document_class_cui"], i)
            ):
                key = (name, self.rows[i]["document_class_cui"])
                if key in seen:
                    continue
                seen.add(key)
                out.append({**self.rows[i], "possible_document_name": name})
        if fuzzy:
            _log(
                "Document-cluster member names matched by similarity",
                {"matches": fuzzy[:20], "min_similarity": min_similarity},
                severity="INFO",
            )
        if misses:
            _log(
                "Document-cluster member names not found in the document catalog",
                {"unmatched_sample": misses[:10], "catalog_size": len(self.rows)},
                severity="WARNING" if not out else "INFO",
            )
        # Same order as the SQL path (cluster id, then member name) so the
        # per-record-type coding order is identical either way.
        return sorted(
            out, key=lambda r: (r["document_class_cui"], r["possible_document_name"])
        )


def _load_catalog_rows() -> list[dict[str, str]]:
    _require_configured()
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
        ORDER BY document_class_cui, possible_document_name
    """
    rows = get_bq_client().query(query).result()
    return [
        {
            "document_class_cui": str(row["document_class_cui"] or "").strip(),
            "document_class": str(row["document_class"] or "").strip(),
            "possible_document_name": str(row["possible_document_name"] or "").strip(),
        }
        for row in rows
    ]


def get_document_catalog(force_reload: bool = False) -> DocumentCatalog | None:
    """The cached catalog, (re)loaded when older than
    CLUSTER_DOCUMENT_CATALOG_TTL_SECONDS. None when the catalog is disabled
    (TTL 0) or could not be loaded — callers then fall back to the
    per-request query."""
    ttl = float(getattr(config, "CLUSTER_DOCUMENT_CATALOG_TTL_SECONDS", 3600) or 0)
    if ttl <= 0:
        return None
    with _catalog_lock:
        cat = _catalog_state["catalog"]
        fresh = cat is not None and (time.time() - _catalog_state["loaded_at"]) < ttl
        if fresh and not force_reload:
            return cat
        try:
            cat = DocumentCatalog(_load_catalog_rows())
        except Exception as e:  # noqa: BLE001 — the per-request query is the fallback
            _log(
                "Document catalog load failed; falling back to per-request BigQuery lookups",
                {"error_type": type(e).__name__, "error": str(e)},
            )
            return _catalog_state["catalog"]  # stale copy if any, else None
        _catalog_state["catalog"] = cat
        _catalog_state["loaded_at"] = time.time()
        _log(
            "Document catalog loaded",
            {"members": len(cat), "ttl_seconds": ttl},
            severity="INFO",
        )
        return cat


def current_catalog() -> DocumentCatalog | None:
    """The catalog already in memory (loaded by an earlier lookup), without
    triggering a load: used by stages that only add value when it exists."""
    with _catalog_lock:
        return _catalog_state["catalog"]


def reset_catalog_cache() -> None:
    with _catalog_lock:
        _catalog_state["catalog"] = None
        _catalog_state["loaded_at"] = 0.0


def get_document_classes(document_names: Iterable[str]) -> list[dict[str, str]]:
    """member document name -> (document_class_cui, document_class) rows.

    With the document catalog (default) this is an in-memory match — exact,
    folded, then similarity — and costs no BigQuery call. Without it, one
    query for all names with a case- and whitespace-insensitive comparison.
    Rows are ordered deterministically.
    """
    names: list[str] = []
    for n in document_names or []:
        n = str(n).strip()
        if n and n not in names:
            names.append(n)
    if not names:
        return []

    catalog = get_document_catalog()
    if catalog is not None:
        return catalog.lookup(
            names,
            float(getattr(config, "CLUSTER_MEMBER_MATCH_MIN_SIMILARITY", 0.75) or 0),
        )

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
        WHERE LOWER(TRIM(m.MEMBER_NAME)) IN UNNEST(@document_names)
        ORDER BY document_class_cui, possible_document_name
    """
    # Case- and whitespace-insensitive on both sides: the service's member
    # names and the MEMBER table are maintained separately.
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter(
                "document_names", "STRING", sorted({_norm(n) for n in names})
            )
        ]
    )
    try:
        rows = get_bq_client().query(query, job_config=job_config).result()
        out = [
            {
                "document_class_cui": str(row["document_class_cui"] or "").strip(),
                "document_class": str(row["document_class"] or "").strip(),
                "possible_document_name": str(
                    row["possible_document_name"] or ""
                ).strip(),
            }
            for row in rows
        ]
        matched = {_norm(r["possible_document_name"]) for r in out}
        unmatched = [n for n in names if _norm(n) not in matched]
        if unmatched:
            _log(
                "Document-cluster BigQuery lookup: member names without a row",
                {
                    "sent": len(names),
                    "matched": len(names) - len(unmatched),
                    "unmatched_sample": unmatched[:10],
                    "member_table": config.CLUSTER_MEMBER_TABLE,
                },
                severity="WARNING" if not out else "INFO",
            )
        return out
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
    aliases: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    """Resolve record-type objects to CUIs using the tag names as context.

    Input:  [{"name": "Radiology Report", "coding": []}, ...] (or bare strings)
    Tags:   [{"name": "chest X-ray"}, {"name": "imaging"}] (or bare strings)
    Output: [{"name": "Radiology Report", "coding": [{"code": "C..."}]}, ...]

    Each record type keeps its own name; its coding is the ordered, de-duplicated
    set of document-class CUIs found for it:
      1. the cluster-selection service is asked for the name AND for its
         ``aliases`` — the vocabulary labels the model itself matched the name
         to in the contextual-environment call (the same semantic match the
         /v1 path relied on), capped by CLUSTER_ALIAS_TEXTS_MAX — and every
         member document it returns is looked up in the document catalog;
      2. the name and its aliases are also looked up in the catalog directly
         (abbreviations, acronyms, spelling variants), so "PRES. record" still
         codes when the service answered with nothing usable.
    `resolver(name, context)` can be injected (tests, alternative backends).
    Cluster-selection calls run concurrently; the catalog is in memory.
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

    alias_cap = max(0, int(getattr(config, "CLUSTER_ALIAS_TEXTS_MAX", 2) or 0))
    texts_by_name: dict[str, list[str]] = {}
    pipeline_hits: dict[str, list[dict[str, Any]]] = {}
    for name in names:
        texts = [name]
        for alt in (aliases or {}).get(name.lower(), []):
            if len(texts) - 1 >= alias_cap:
                break
            alt = str(alt).strip()
            if alt and _fold(alt) not in {_fold(t) for t in texts}:
                texts.append(alt)
        texts_by_name[name] = texts
    jobs = [(name, text) for name in names for text in texts_by_name[name]]

    # Stage 0 — the transaction-selection pipeline (cluster selection ->
    # activities -> documents -> document classes, each filtered by the
    # transaction-selection service), when it is configured. A record type it
    # codes is final; the stages below only run for what it left uncoded.
    from app.utils import document_transaction_pipeline as tx

    if tx.is_enabled():
        fanout = max(
            1,
            min(
                int(
                    getattr(
                        config,
                        "CLUSTER_SELECTION_MAX_WORKERS",
                        _CLUSTER_SELECTION_MAX_WORKERS,
                    )
                    or 1
                ),
                len(names),
            ),
        )
        with ThreadPoolExecutor(max_workers=fanout) as executor:
            for name, hits in zip(
                names,
                executor.map(
                    lambda n: _pipeline_for(tx, n, texts_by_name[n], context), names
                ),
                strict=True,
            ):
                if hits:
                    pipeline_hits[name] = hits
        jobs = [j for j in jobs if j[0] not in pipeline_hits]
        if not jobs:
            return [
                {"name": name, "coding": _coding_from_hits(pipeline_hits.get(name))}
                for name in names
            ]

    # Fan out cluster selection over every (name, text), then one catalog lookup.
    workers = min(
        int(
            getattr(
                config, "CLUSTER_SELECTION_MAX_WORKERS", _CLUSTER_SELECTION_MAX_WORKERS
            )
            or 1
        ),
        len(jobs),
    )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(
            executor.map(lambda j: get_probable_documents(j[1], context), jobs)
        )
    probable: dict[str, list[str]] = {name: [] for name in names}
    for (name, _text), docs in zip(jobs, results, strict=True):
        for d in docs:
            if d not in probable[name]:
                probable[name].append(d)

    # Everything worth looking up: member documents the service named, plus
    # the record-type name and its aliases themselves.
    all_docs: list[str] = []
    for name in names:
        for d in probable[name] + texts_by_name[name]:
            if d not in all_docs:
                all_docs.append(d)
    rows = get_document_classes(all_docs) if all_docs else []

    # get_document_classes above loaded (or refreshed) the catalog when it is
    # enabled; the class stage reuses it and never triggers a load itself.
    catalog = current_catalog()
    min_sim = float(getattr(config, "CLUSTER_MEMBER_MATCH_MIN_SIMILARITY", 0.75) or 0)
    resolved: list[dict[str, Any]] = []
    for name in names:
        if name in pipeline_hits:
            resolved.append(
                {"name": name, "coding": _coding_from_hits(pipeline_hits[name])}
            )
            continue
        doc_set = {_norm(d) for d in probable[name]}
        hits = _classes_to_coding(
            r for r in rows if _norm(r.get("possible_document_name")) in doc_set
        )
        if not hits:
            own = {_norm(t) for t in texts_by_name[name]}
            hits = _classes_to_coding(
                r for r in rows if _norm(r.get("possible_document_name")) in own
            )
        coding = _coding_from_hits(hits)
        if not coding and catalog is not None:
            # Last deterministic stage: the label (or an alias) names a
            # document CLASS of the cluster directly.
            for text in texts_by_name[name]:
                cuis, stage, score = catalog.match_class(text, min_sim)
                if cuis:
                    coding = [{"code": c} for c in cuis]
                    _log(
                        "Record type coded by document-class label match",
                        {
                            "record_type": name,
                            "text": text,
                            "stage": stage,
                            "score": round(score, 3),
                        },
                        severity="INFO",
                    )
                    break
        resolved.append({"name": name, "coding": coding})
    return resolved


def _pipeline_for(
    tx: Any, name: str, texts: list[str], context: str | None
) -> list[dict[str, Any]]:
    """Run the transaction pipeline for the record-type label, then for its
    aliases until one of them selects document classes."""
    for text in texts:
        try:
            hits = tx.resolve_via_transactions(text, context)
        except Exception as e:  # noqa: BLE001 — the pipeline never fails the request
            details = getattr(e, "details", None) or {}
            _log(
                "Record-type transaction pipeline failed",
                {
                    "record_type": name,
                    "text": text,
                    "error_type": type(e).__name__,
                    "error": getattr(e, "message", str(e)),
                    **details,
                },
            )
            continue
        if hits:
            return hits
    return []
