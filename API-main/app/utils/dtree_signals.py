from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from app.utils.concept_vocab import ConceptVocab, normalize_facet_name
from app.utils.temporal_vocab import TemporalVocab

# Generic nature/category tokens that make useless topics.
_TOPIC_STOPWORDS = {
    "clinical",
    "finding",
    "findings",
    "history",
    "condition",
    "conditions",
    "category",
    "type",
    "types",
    "classification",
    "general",
    "other",
    "and",
    "or",
    "of",
    "the",
    "with",
    "patient",
}


def _coding_from_codes(codes) -> list[dict[str, str]]:
    return [{"code": c.cui} for c in codes]


# Fields a later duplicate may backfill when the earlier entry left them empty.
_BACKFILL_FIELDS = ("coding", "topics")


def _dedupe_by_name(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in items:
        key = str(item.get("name", "")).strip().lower()
        if not key:
            continue
        prev = out.get(key)
        if prev is None:
            out[key] = dict(item)
            continue
        for field in _BACKFILL_FIELDS:
            if not prev.get(field) and item.get(field):
                prev[field] = item[field]
    return list(out.values())


def _tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9\-]+", text.lower()) if t]


def _fold_label(text: Any) -> str:
    """Case/underscore-insensitive key for matching an emitted label."""
    return re.sub(r"[^a-z0-9]+", " ", str(text).strip().lower()).strip()


def _record_type_match_map(v2_result: dict[str, Any]) -> dict[str, list[str]]:
    """Emitted record type -> vocabulary entry names the model matched to it.

    Carried alongside the facets (see RecordTypeMatch); absent when the
    pipeline ran without a record-type vocabulary.
    """
    out: dict[str, list[str]] = {}
    for m in v2_result.get("record_type_matches") or []:
        if not isinstance(m, dict):
            continue
        key = _fold_label(m.get("record_type", ""))
        if not key:
            continue
        names = [
            str(n).strip()
            for n in (m.get("selected_names") or [])
            if isinstance(n, str) and str(n).strip()
        ]
        if names:
            out.setdefault(key, []).extend(names)
    return out


def _display_record_type_name(raw_name: str) -> str:
    """Preserve existing label style; only normalize obvious snake_case forms."""
    s = str(raw_name).strip()
    if not s:
        return ""
    return normalize_facet_name(s) if "_" in s else s


def project_record_types(
    v2_result: dict[str, Any],
    record_type_vocab: ConceptVocab | None,
) -> list[dict[str, Any]]:
    """Union of intent-level retrieval_signals.record_types -> named + coded list.

    Order is preserved from the pipeline (SignalsAssembler already dedupes and
    the contextual-environment prompt returns record types by descending
    relevance), so the first entry stays the most definitive source.
    """
    raw: list[str] = []
    seen: set = set()
    for intent in v2_result.get("intents", []) or []:
        signals = intent.get("retrieval_signals") or {}
        for rt in signals.get("record_types") or []:
            k = str(rt).strip().lower()
            if k and k not in seen:
                seen.add(k)
                raw.append(str(rt).strip())

    match_map = _record_type_match_map(v2_result)

    out: list[dict[str, Any]] = []
    seen_identity: set = set()
    for rt in raw:
        display_name = _display_record_type_name(rt)
        codes: list[Any] = []

        if record_type_vocab is not None:
            seen_cuis: set = set()
            for name in match_map.get(_fold_label(rt), []):
                canonical = record_type_vocab.name_from_text(name)
                for code in record_type_vocab.codes_for_name(canonical):
                    if code.cui not in seen_cuis:
                        seen_cuis.add(code.cui)
                        codes.append(code)

        identity = display_name.strip().lower()
        if not identity or identity in seen_identity:
            continue
        seen_identity.add(identity)

        out.append(
            {
                "name": display_name,
                "coding": _coding_from_codes(codes),
            }
        )
    return out


def collect_record_type_names(v2_result: dict[str, Any]) -> list[str]:
    """Union of intent-level retrieval_signals.record_types as display names.

    Same source, order and de-duplication as project_record_types, without any
    vocabulary lookup — the /v2 cluster path codes these names externally.
    """
    names: list[str] = []
    seen: set = set()
    for intent in v2_result.get("intents", []) or []:
        signals = intent.get("retrieval_signals") or {}
        for rt in signals.get("record_types") or []:
            display_name = _display_record_type_name(str(rt))
            identity = display_name.lower()
            if not identity or identity in seen:
                continue
            seen.add(identity)
            names.append(display_name)
    return names


def project_record_types_cluster(
    v2_result: dict[str, Any],
    tags: list[dict[str, Any]],
    resolver: Callable[[str, str | None], list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """/v2 record types: pipeline names coded through the document-cluster lookup.

    Each record type is resolved by its own name with the query's tag names as
    context (see context_lonic_document_cluster.resolve_record_type_cuis). The
    output shape is identical to project_record_types:
    [{"name": ..., "coding": [{"code": cui}, ...]}, ...].
    """
    from app.utils.context_lonic_document_cluster import resolve_record_type_cuis

    record_types = [
        {"name": name, "coding": []} for name in collect_record_type_names(v2_result)
    ]
    if not record_types:
        return []
    # The vocabulary labels the model matched each record type to (the /v1
    # semantic match) double as alias search texts for the cluster path.
    return resolve_record_type_cuis(
        record_types, tags, resolver, aliases=record_type_aliases(v2_result)
    )


def record_type_aliases(v2_result: dict[str, Any]) -> dict[str, list[str]]:
    """Record-type display name (lower-cased) -> the vocabulary labels the
    model matched it to in the contextual-environment call."""
    match_map = _record_type_match_map(v2_result)
    aliases: dict[str, list[str]] = {}
    for name in collect_record_type_names(v2_result):
        found = match_map.get(_fold_label(name), [])
        if found:
            aliases[name.lower()] = list(dict.fromkeys(found))
    return aliases


def project_temporal(
    v2_result: dict[str, Any],
    temporal_vocab: TemporalVocab | None = None,
) -> dict[str, Any] | None:
    """First resolved temporal entry -> single temporal object with all available codes.

    The response shape remains a single temporal object; when multiple CUIs are
    available for that window, emit all of them in coding.
    """

    default_name = temporal_vocab.default_name() if temporal_vocab is not None else None
    fallback: dict[str, Any] | None = None

    for intent in v2_result.get("intents", []) or []:
        signals = intent.get("retrieval_signals") or {}
        for t in signals.get("temporal") or []:
            name = t.get("time_window")

            # Accept either legacy scalar `codes` or list-valued `codes` if present.
            raw_codes = t.get("codes")
            code_list: list[str] = []
            if isinstance(raw_codes, str) and raw_codes.strip():
                code_list.append(raw_codes.strip())
            elif isinstance(raw_codes, list):
                for code in raw_codes:
                    if isinstance(code, str) and code.strip():
                        code_list.append(code.strip())

            # If vocabulary is available, expand by canonical window name.
            if temporal_vocab is not None and isinstance(name, str) and name.strip():
                canonical = temporal_vocab.name_from_text(name)
                for entry in temporal_vocab.codes_for_name(canonical):
                    if isinstance(entry.cui, str) and entry.cui.strip():
                        code_list.append(entry.cui.strip())

            deduped_codes = list(dict.fromkeys(code_list))
            if not name and not deduped_codes:
                continue

            coding = [{"system": "UMLS", "code": code} for code in deduped_codes]

            formula = t.get("formula")
            if isinstance(formula, str):
                formula = [formula]
            elif not isinstance(formula, list):
                formula = []
            entry = {
                "name": name,
                "formula": formula,
                "coding": coding,
            }
            is_default = bool(default_name) and _fold_label(name) == _fold_label(
                default_name
            )
            if not is_default:
                return entry
            if fallback is None:
                fallback = entry
    if fallback is not None:
        return fallback

    # Always return a temporal window when the model emits no temporal signal.
    # Use the configured vocabulary default when available; otherwise retain the
    # service-level Recent fallback used by the v2 pipeline.
    fallback_name = temporal_vocab.default_name() if temporal_vocab else "Recent"
    fallback_codes = (
        [entry.cui for entry in temporal_vocab.codes_for_name(fallback_name)]
        if temporal_vocab
        else ["C0332185"]
    )

    return {
        "name": fallback_name,
        "formula": ["REF_POINT", "REF_POINT"],
        "coding": [{"system": "UMLS", "code": code} for code in fallback_codes],
    }


def _topics_for_tag(
    tag_name: str, v2_result: dict[str, Any], limit: int = 5
) -> list[str]:
    tag_tokens = set(_tokens(tag_name))
    intents = v2_result.get("intents", []) or []

    def related(intent: dict[str, Any]) -> bool:
        hay = " ".join(
            [str(intent.get("intent_title", ""))]
            + [
                str(c)
                for sn in intent.get("sub_natures") or []
                for c in sn.get("atomic_concepts") or []
            ]
        )
        return bool(tag_tokens & set(_tokens(hay)))

    pool = [i for i in intents if related(i)] or intents

    topics: list[str] = []
    seen: set = set()
    for intent in pool:
        sources = [str(intent.get("nature", ""))] + [
            str(sn.get("category_path", "")) for sn in intent.get("sub_natures") or []
        ]
        for src in sources:
            for part in re.split(r"[/>\[\]]+", src):
                for tok in _tokens(part):
                    if tok in _TOPIC_STOPWORDS or tok in tag_tokens or len(tok) < 3:
                        continue
                    if tok not in seen:
                        seen.add(tok)
                        topics.append(tok)
                    if len(topics) >= limit:
                        return topics
    return topics


def project_tags(
    v2_result: dict[str, Any],
    tag_vocab: ConceptVocab | None,
) -> list[dict[str, Any]]:
    """representative_terms -> tags with optional coding and heuristic topics."""
    out: list[dict[str, Any]] = []
    for term in v2_result.get("representative_terms", []) or []:
        term = str(term).strip()
        if not term:
            continue
        if tag_vocab is not None:
            canonical, codes = tag_vocab.lookup(term)
        else:
            canonical, codes = None, []
        out.append(
            {
                "name": canonical or term,
                "coding": _coding_from_codes(codes),
                "topics": _topics_for_tag(term, v2_result),
            }
        )
    return out


def merge_hints(
    derived: dict[str, Any],
    hint_record_types: list[dict[str, Any]] | None,
    hint_temporal: dict[str, Any] | None,
    hint_tags: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Merge validated client hints into the derived block.

    - record_types / tags: hints first, then derived, deduped by name.
    - temporal: an explicit client hint wins over the derived window.
    """
    if hint_record_types:
        derived["record_types"] = _dedupe_by_name(
            list(hint_record_types) + derived.get("record_types", [])
        )
    if hint_tags:
        derived["tags"] = _dedupe_by_name(list(hint_tags) + derived.get("tags", []))
    if hint_temporal:
        derived["temporal"] = hint_temporal
    return derived


def project_query_signals(
    query_id: str,
    text: str,
    v2_result: dict[str, Any],
    record_type_vocab: ConceptVocab | None = None,
    temporal_vocab: TemporalVocab | None = None,
    tag_vocab: ConceptVocab | None = None,
    hint_record_types: list[dict[str, Any]] | None = None,
    hint_temporal: dict[str, Any] | None = None,
    hint_tags: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Full projection for one query: pipeline result -> dtree schema block."""
    derived = {
        "id": query_id,
        "text": text,
        "record_types": project_record_types(v2_result, record_type_vocab),
        "temporal": project_temporal(v2_result, temporal_vocab=temporal_vocab),
        "tags": project_tags(v2_result, tag_vocab),
    }
    return merge_hints(derived, hint_record_types, hint_temporal, hint_tags)
