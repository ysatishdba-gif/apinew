"""Last-resort record-type coding: match the labels that neither the document
cluster nor the /v1 vocabulary could code against the cluster's own document
classes, with the model — the same kind of semantic match the /v1 prompt did
against KNOWN RECORD TYPES, now against the classes the cluster tables define.

Runs only for record types that are still uncoded after every deterministic
stage, once per request (all of them in one call), and only when a document
catalog is loaded. The candidate list is the catalog's document-class labels;
when there are more than RECORD_TYPE_LLM_FALLBACK_MAX_LABELS, the labels
closest to the unresolved names (token and n-gram similarity) are shown, so
nothing about the classes is authored here.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Iterable
from typing import Any

from pydantic import BaseModel, Field

from app.utils.context_lonic_document_cluster import (
    DocumentCatalog,
    _fold,
    token_match_score,
)
from app.utils.temporal_index import LexicalSimilarity

RECORD_TYPE_FALLBACK_MAX_TOKENS = 4096

RECORD_TYPE_FALLBACK_PROMPT = """You are mapping clinical document types to the document classes of a retrieval system.

For EACH record type below, pick the document class label(s) from DOCUMENT CLASSES that name
the same clinical document. Match by MEANING, not spelling: expand abbreviations and
conventions ("PRES. record", "Rx record" -> a prescription / medication order class;
"H&P" -> history and physical; "d/c summary" -> discharge summary), accept
setting-qualified variants of the same document, and ignore labels that name a different
document even when they share a word ("progress note" is not "consultation note").
When no class names the exact document, choose the nearest BROADER class that would
contain it (a specific imaging report -> the imaging report class). Copy labels from the
list VERBATIM. Leave `selected_labels` empty ONLY when no class in the list could plausibly
hold that document.

Query the record types came from: {query}

RECORD TYPES (each with the other names it was referred to by, if any):
{record_types}

DOCUMENT CLASSES:
{classes}

Return JSON: {{"matches": [{{"record_type": "<as given>", "selected_labels": ["<label>", ...],
"reasoning": "<one short sentence>"}}, ...]}} with one object per record type, in the order given.
"""


class RecordTypeFallbackMatch(BaseModel):
    record_type: str
    selected_labels: list[str] = Field(default_factory=list)
    reasoning: str | None = None


class RecordTypeFallbackOutput(BaseModel):
    matches: list[RecordTypeFallbackMatch] = Field(default_factory=list)


def shortlist_labels(
    names: Iterable[str], labels: list[str], max_labels: int
) -> list[str]:
    """All labels when they fit; otherwise the ones closest to the unresolved
    names by token match and character n-gram similarity, plus enough of
    the rest (in catalog order) to fill ``max_labels``."""
    labels = list(labels)
    if max_labels <= 0 or len(labels) <= max_labels:
        return labels
    sim = LexicalSimilarity(labels)
    score: dict[str, float] = {lab: 0.0 for lab in labels}
    for name in names:
        ngram = sim.scores(_fold(name))
        for i, lab in enumerate(labels):
            score[lab] = max(score[lab], token_match_score(name, lab), float(ngram[i]))
    ranked = sorted(labels, key=lambda lab: (-score[lab], labels.index(lab)))
    return ranked[:max_labels]


def _parse(raw: str) -> RecordTypeFallbackOutput:
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return RecordTypeFallbackOutput()
    try:
        return RecordTypeFallbackOutput.model_validate(data)
    except ValueError:
        return RecordTypeFallbackOutput()


async def resolve_with_model(
    unresolved: list[dict[str, Any]],
    catalog: DocumentCatalog,
    query: str,
    call_model_async: Callable[..., Awaitable[tuple[str, dict[str, int]]]],
    *,
    model_name: str | None,
    location: str | None,
    max_labels: int,
) -> tuple[dict[str, list[str]], dict[str, int]]:
    """``unresolved``: [{"name": ..., "aliases": [...]}]. Returns
    (name -> CUIs, usage metadata of the one call). Labels the model returns
    are resolved to CUIs through the catalog's class index, never trusted
    as codes themselves."""
    classes = catalog.class_labels()
    if not unresolved or not classes:
        return {}, {}
    names = [u["name"] for u in unresolved]
    labels = shortlist_labels(names, list(classes), max_labels)
    lines = []
    for u in unresolved:
        aliases = [a for a in u.get("aliases") or [] if a]
        extra = f" (also: {', '.join(aliases)})" if aliases else ""
        lines.append(f"- {u['name']}{extra}")
    prompt = RECORD_TYPE_FALLBACK_PROMPT.format(
        query=(query or "").strip()[:500],
        record_types="\n".join(lines),
        classes=json.dumps(labels, ensure_ascii=False),
    )
    raw, usage = await call_model_async(
        prompt,
        model_name=model_name,
        generation_config=None,
        location=location,
        temperature=0.0,
        max_tokens=RECORD_TYPE_FALLBACK_MAX_TOKENS,
        step_name="record_type_fallback",
        response_schema=RecordTypeFallbackOutput,
    )
    out: dict[str, list[str]] = {}
    for m in _parse(raw).matches:
        key = m.record_type.strip().lower()
        cuis: list[str] = []
        for label in m.selected_labels:
            for cui in catalog.cuis_for_class_label(label):
                if cui not in cuis:
                    cuis.append(cui)
        if cuis:
            out[key] = cuis
    return out, usage or {}
