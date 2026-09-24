from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.utils.temporal_canonical import (
    BASES,
    CanonicalTemporalMapping,
    CanonicalTemporalResolver,
    CanonicalWindow,
)
from app.utils.temporal_vocab import (
    DEFAULT_TEMPORAL_FALLBACK,
    NullTemporalResolver,
    TemporalMapping,
    TemporalResolver,
    TemporalVocab,
)

# ---------------------------------------------------------------------------
# Step 4 LLM response models
# ---------------------------------------------------------------------------


class ConceptContext(BaseModel):
    atomic_concept: str
    intent_title: str = ""
    record_types: list[str] = Field(default_factory=list)
    author_roles: list[str] = Field(default_factory=list)
    longitudinal_scope: list[str] = Field(default_factory=list)
    content_signals: list[str] = Field(default_factory=list)
    clinical_settings: list[str] = Field(default_factory=list)


class TemporalMatch(BaseModel):
    signal: str
    # Which rule produced `signal`: explicit_window | explicit_qualifier |
    # explicit_ordering | type_default. Forces the model to label its own
    # answer, so a skipped explicit-signal scan is visible instead of looking
    # identical to a correctly-read one. Also the prod metric for how often the
    # type default fires.
    signal_basis: str | None = None
    selected_id: str | None = None
    selected_name: str | None = None
    selected_ids: list[str] = Field(default_factory=list)
    selected_names: list[str] = Field(default_factory=list)
    selected_reasoning: str | None = None


class CandidateTemporal(BaseModel):
    candidate: str
    temporal_signal: list[TemporalMatch] = Field(default_factory=list)

    @field_validator("temporal_signal", mode="before")
    @classmethod
    def _coerce(cls, v: Any) -> Any:
        if not isinstance(v, list):
            return v
        prefix_re = re.compile(
            r"^\s*(?:state|status|relative|absolute|rank)\s*[:\-–]\s*",
            re.IGNORECASE,
        )
        out: list[dict[str, Any]] = []
        for item in v:
            if isinstance(item, str):
                s = prefix_re.sub("", item).strip()
                if s:
                    out.append({"signal": s})
            elif isinstance(item, dict):
                sig = prefix_re.sub("", str(item.get("signal", "")).strip()).strip()
                if sig:
                    selected_ids: list[str] = []
                    raw_ids = item.get("selected_ids")
                    if isinstance(raw_ids, list):
                        selected_ids = [
                            str(x).strip()
                            for x in raw_ids
                            if isinstance(x, str) and str(x).strip()
                        ]

                    selected_names: list[str] = []
                    raw_names = item.get("selected_names")
                    if isinstance(raw_names, list):
                        selected_names = [
                            str(x).strip()
                            for x in raw_names
                            if isinstance(x, str) and str(x).strip()
                        ]

                    out.append(
                        {
                            "signal": sig,
                            "signal_basis": item.get("signal_basis"),
                            "selected_id": item.get("selected_id"),
                            "selected_name": item.get("selected_name"),
                            "selected_ids": selected_ids,
                            "selected_names": selected_names,
                            "selected_reasoning": item.get("selected_reasoning"),
                        }
                    )
        return out


class RecordTypeMatch(BaseModel):
    """One emitted record type and the vocabulary entries it matched.

    Parallel to TemporalMatch.selected_names: the model may match a single
    record type to SEVERAL vocabulary entries (e.g. a consultation report to
    the base concept plus its setting-qualified variants). Codes from all of
    them are consolidated under the model's own label — the label never comes
    from the vocabulary, only the CUIs do.
    """

    record_type: str
    selected_names: list[str] = Field(default_factory=list)
    selected_reasoning: str | None = None


class IntentTemporal(BaseModel):
    intent_title: str
    candidates: list[CandidateTemporal] = Field(default_factory=list)


class TemporalExtractionOutput(BaseModel):
    intents: list[IntentTemporal] = Field(default_factory=list)


class ContextualEnvironmentOutput(BaseModel):
    concepts_with_context: list[ConceptContext] = Field(default_factory=list)
    record_type_matches: list[RecordTypeMatch] = Field(default_factory=list)
    temporal_by_intent: list[IntentTemporal] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Canonical temporal mode (/v3): the model emits a structured window per
# temporal_signal entry instead of picking a vocabulary id. Separate models so
# the legacy response schema sent to the model is byte-for-byte unchanged.
# ---------------------------------------------------------------------------


class TemporalMatchCanonical(TemporalMatch):
    window: CanonicalWindow | None = None
    basis: str | None = Field(None, json_schema_extra={"enum": list(BASES)})
    rationale: str | None = None


class CandidateTemporalCanonical(CandidateTemporal):
    temporal_signal: list[TemporalMatchCanonical] = Field(default_factory=list)

    @field_validator("temporal_signal", mode="before")
    @classmethod
    def _coerce(cls, v: Any) -> Any:
        if not isinstance(v, list):
            return v
        base = CandidateTemporal._coerce(v)
        # Re-attach the canonical fields the base coercion drops, by position
        # over the entries that survived (it drops blank signals only).
        kept = [item for item in v if _has_signal(item)]
        for item, out in zip(kept, base, strict=False):
            if isinstance(item, dict):
                out["window"] = _valid_window(item.get("window"))
                basis = item.get("basis")
                out["basis"] = (
                    basis.strip().lower()
                    if isinstance(basis, str) and basis.strip().lower() in BASES
                    else None
                )
                out["rationale"] = item.get("rationale")
        return base


def _valid_window(raw: Any) -> dict[str, Any] | None:
    """A window the resolver can use, or None. Parsing is tolerant on purpose:
    the response schema asks the model for a window on every entry, but one
    malformed window (a zero value, an unknown unit) must cost that entry its
    window, not the whole contextual environment."""
    if not isinstance(raw, dict):
        return None
    try:
        return CanonicalWindow.model_validate(raw).model_dump()
    except ValueError:
        return None


def _exact_key(text: Any) -> str:
    return str(text or "").strip().lower()


def _has_signal(item: Any) -> bool:
    if isinstance(item, str):
        return bool(item.strip())
    if isinstance(item, dict):
        return bool(str(item.get("signal", "")).strip())
    return False


class IntentTemporalCanonical(IntentTemporal):
    candidates: list[CandidateTemporalCanonical] = Field(default_factory=list)


class ContextualEnvironmentOutputCanonical(ContextualEnvironmentOutput):
    temporal_by_intent: list[IntentTemporalCanonical] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Structured-output SCHEMA for canonical mode. This is what the model is asked
# to produce (window / basis / rationale REQUIRED on every entry, so a model
# cannot leave the window out the way it can with an optional field); the
# tolerant *Canonical models above are what the response is parsed with.
# ---------------------------------------------------------------------------
SIGNAL_BASES: tuple[str, ...] = (
    "explicit_window",
    "explicit_qualifier",
    "explicit_ordering",
    "type_default",
)


class TemporalMatchCanonicalSchema(BaseModel):
    signal: str
    signal_basis: str = Field(..., json_schema_extra={"enum": list(SIGNAL_BASES)})
    window: CanonicalWindow
    basis: str = Field(..., json_schema_extra={"enum": list(BASES)})
    rationale: str


class CandidateTemporalCanonicalSchema(BaseModel):
    candidate: str
    temporal_signal: list[TemporalMatchCanonicalSchema] = Field(..., min_length=1)


class IntentTemporalCanonicalSchema(BaseModel):
    intent_title: str
    candidates: list[CandidateTemporalCanonicalSchema]


class ContextualEnvironmentOutputCanonicalSchema(BaseModel):
    concepts_with_context: list[ConceptContext]
    record_type_matches: list[RecordTypeMatch] = Field(default_factory=list)
    temporal_by_intent: list[IntentTemporalCanonicalSchema]


# ---------------------------------------------------------------------------
# Intermediate assembly models (pre _structured_signals)
# ---------------------------------------------------------------------------


class _RetrievalContext(BaseModel):
    record_types: list[str] = Field(default_factory=list)
    author_roles: list[str] = Field(default_factory=list)
    longitudinal_scope: list[str] = Field(default_factory=list)
    content_signals: list[str] = Field(default_factory=list)
    clinical_setting: list[str] = Field(default_factory=list)


class _FlatRetrievalSignals(BaseModel):
    record_types: list[str] = Field(default_factory=list)
    author_roles: list[str] = Field(default_factory=list)
    longitudinal_scope: list[str] = Field(default_factory=list)
    temporal_signal: list[str] = Field(default_factory=list)
    content_signals: list[str] = Field(default_factory=list)
    clinical_setting: list[str] = Field(default_factory=list)

    @classmethod
    def from_retrieval_context(cls, ctx: _RetrievalContext) -> _FlatRetrievalSignals:
        def _as_list(lst: list[str]) -> list[str]:
            return list(
                dict.fromkeys(v.strip() for v in (lst or []) if v and v.strip())
            )

        return cls(
            record_types=_as_list(ctx.record_types),
            author_roles=_as_list(ctx.author_roles),
            longitudinal_scope=_as_list(ctx.longitudinal_scope),
            temporal_signal=[],
            content_signals=_as_list(ctx.content_signals),
            clinical_setting=_as_list(ctx.clinical_setting),
        )


class _FinalCandidateItem(BaseModel):
    candidate_id: str
    intent_title: str
    nature: str
    sub_nature: str
    candidate: str
    retrieval_signals: _FlatRetrievalSignals = Field(
        default_factory=_FlatRetrievalSignals
    )


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


class SignalsAssembler:
    """Build final_candidates and resolve temporal CUIs from pipeline outputs."""

    def __init__(
        self,
        context: ContextualEnvironmentOutput | None,
        temporal: TemporalExtractionOutput | None,
        vocab: TemporalVocab | None = None,
        resolver: TemporalResolver | None = None,
        canonical_resolver: CanonicalTemporalResolver | None = None,
    ):
        self.context = context
        self.temporal = temporal
        self._vocab = vocab
        self._resolver = resolver or NullTemporalResolver()
        # Canonical mode: windows are resolved by the index, uncoded windows keep
        # their formula, and the default comes from the vocabulary, not a constant.
        self._canonical = canonical_resolver
        self._temporal_map: dict[str, list[TemporalMapping]] | None = None
        self._temporal_map_by_candidate: dict[
            tuple[str, str], list[TemporalMapping]
        ] = {}
        self._concept_index: dict[str, ConceptContext] | None = None
        if context and context.concepts_with_context:
            self._concept_index = {
                cc.atomic_concept.strip().lower(): cc
                for cc in context.concepts_with_context
            }

    @staticmethod
    def _aggregate_facets(
        concepts: list[ConceptContext],
    ) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
        records, authors, scope, signals, settings = [], [], [], [], []
        for cc in concepts or []:
            records.extend(cc.record_types or [])
            authors.extend(cc.author_roles or [])
            scope.extend(cc.longitudinal_scope or [])
            signals.extend(cc.content_signals or [])
            settings.extend(cc.clinical_settings or [])

        def dedupe(xs: list[str]) -> list[str]:
            return list(dict.fromkeys(x for x in xs if x))

        return (
            dedupe(records),
            dedupe(authors),
            dedupe(scope),
            dedupe(signals),
            dedupe(settings),
        )

    def _get_concept_context(self, atomic_concept: str) -> ConceptContext | None:
        if self._concept_index is None:
            return None
        return self._concept_index.get(atomic_concept.strip().lower())

    @staticmethod
    def _fold_key(text: Any) -> str:
        return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()

    def _temporal_for_candidate(self, intent_title: str, candidate: str) -> list[str]:
        """The model's temporal signals for this candidate. Matching the
        model's echo of intent title / candidate name is punctuation- and
        case-insensitive; when the title does not match any returned intent
        (a reworded echo) the candidate is looked up across all intents, and a
        single returned intent is taken as the one meant. Without this a
        harmless rewording would leave every candidate on the default."""
        if not self.temporal or not self.temporal.intents:
            return []
        # Legacy (/v1, /v2) keeps its exact case-insensitive match, byte for
        # byte; canonical mode (/v3) folds punctuation and recovers a reworded
        # echo.
        key = self._fold_key if self._canonical is not None else _exact_key
        ikey = key(intent_title)
        ckey = key(candidate)
        intents = list(self.temporal.intents)
        matched = [it for it in intents if key(it.intent_title) == ikey]
        if not matched and self._canonical is not None:
            with_candidate = [
                it
                for it in intents
                if any(key(co.candidate) == ckey for co in it.candidates or [])
            ]
            matched = with_candidate or (intents if len(intents) == 1 else [])
        for it in matched:
            own: list[str] = []
            intent_union: list[str] = []
            seen: set = set()
            for co in it.candidates or []:
                sig = [p.signal for p in (co.temporal_signal or [])]
                if key(co.candidate) == ckey:
                    own = sig
                for s in sig:
                    k = s.strip().lower()
                    if k and k not in seen:
                        intent_union.append(s)
                        seen.add(k)
            if own or intent_union:
                return own if own else intent_union
        return []

    def _all_temporal_phrases(self) -> list[str]:
        phrases: set = set()
        if self.temporal and self.temporal.intents:
            for it in self.temporal.intents:
                for c in it.candidates or []:
                    for p in c.temporal_signal or []:
                        s = p.signal
                        if isinstance(s, str) and s:
                            phrases.add(s)
        return sorted(phrases)

    def _get_temporal_map(self) -> dict[str, list[TemporalMapping]]:
        if self._temporal_map is not None:
            return self._temporal_map

        m: dict[str, list[TemporalMapping]] = {}
        if self._canonical is not None:
            # Canonical mode resolves PER CANDIDATE: the same qualifier
            # ("current", "recent") carries a different window for each
            # candidate, so the map is keyed by (candidate, signal); the
            # signal-only entry is the union across candidates, used for the
            # intent level and for candidates that inherit the intent's
            # signals without windows of their own.
            per_candidate: dict[tuple[str, str], list[TemporalMapping]] = {}
            for it in (self.temporal.intents if self.temporal else []) or []:
                for c in it.candidates or []:
                    ckey = self._fold_key(c.candidate)
                    for p in c.temporal_signal or []:
                        sig = p.signal
                        if not isinstance(sig, str) or not sig:
                            continue
                        basis = getattr(p, "basis", None)
                        rationale = getattr(p, "rationale", None)
                        mappings = self._canonical.resolve_window(
                            getattr(p, "window", None),
                            wording=sig,
                            basis=basis,
                            rationale=rationale,
                        ) or self._canonical.resolve_text(sig, basis=basis)
                        if not mappings:
                            continue
                        own = per_candidate.setdefault((ckey, sig), [])
                        union = m.setdefault(sig, [])
                        for tm in mappings:
                            if tm.codes not in {x.codes for x in own}:
                                own.append(tm)
                            if tm.codes not in {x.codes for x in union}:
                                union.append(tm)
            self._temporal_map_by_candidate = per_candidate
        elif self._vocab is not None:
            if self.temporal and self.temporal.intents:
                for it in self.temporal.intents:
                    for c in it.candidates or []:
                        for p in c.temporal_signal or []:
                            sig = p.signal
                            if not isinstance(sig, str) or not sig:
                                continue
                            names: list[str] = []

                            if isinstance(p.selected_id, str) and p.selected_id.strip():
                                by_id = self._vocab.name_for_id(p.selected_id)
                                if by_id:
                                    names.append(by_id)

                            if (
                                isinstance(p.selected_name, str)
                                and p.selected_name.strip()
                            ):
                                by_name = self._vocab.resolve_name(p.selected_name)
                                if by_name:
                                    names.append(by_name)

                            for sid in p.selected_ids or []:
                                by_id = self._vocab.name_for_id(sid)
                                if by_id:
                                    names.append(by_id)

                            for sname in p.selected_names or []:
                                by_name = self._vocab.resolve_name(sname)
                                if by_name:
                                    names.append(by_name)

                            names = list(dict.fromkeys(names))
                            if not names:
                                continue

                            matches = m.setdefault(sig, [])
                            seen_codes = {tm.codes for tm in matches if tm.codes}
                            for name in names:
                                for entry in self._vocab.codes_for_name(name):
                                    if entry.cui in seen_codes:
                                        continue
                                    matches.append(
                                        TemporalMapping(
                                            time_window=name,
                                            codes=entry.cui,
                                            formula=entry.formula,
                                        )
                                    )
                                    seen_codes.add(entry.cui)
        else:
            phrases = self._all_temporal_phrases()
            mappings = self._resolver.resolve(phrases) if phrases else []
            for phrase, mm in zip(phrases, mappings):
                tm = (
                    mm
                    if isinstance(mm, TemporalMapping)
                    else TemporalMapping.model_validate(mm)
                )
                m[phrase] = [tm]

        self._temporal_map = m
        return self._temporal_map

    def _vocab_mappings(self, signal: str) -> list[TemporalMapping]:
        """Resolve one signal directly against the vocabulary via the full ladder."""
        if self._vocab is None:
            return []
        name = self._vocab.resolve_name(signal)
        return [
            TemporalMapping(
                time_window=name,
                codes=entry.cui,
                formula=entry.formula,
            )
            for entry in self._vocab.codes_for_name(name)
        ]

    def _default_temporal(self) -> list[TemporalMapping]:
        """Stand-in window when nothing resolved: the vocabulary's default,
        else the hardcoded fallback so temporal is never empty or uncoded."""
        if self._vocab is not None:
            name = self._vocab.default_name()
            mappings = [
                TemporalMapping(
                    time_window=name,
                    codes=entry.cui,
                    formula=entry.formula,
                )
                for entry in self._vocab.codes_for_name(name)
            ]
            if mappings:
                return mappings
        return [TemporalMapping(**DEFAULT_TEMPORAL_FALLBACK)]

    def structured_signals(
        self, raw: dict[str, Any], candidate: str | None = None
    ) -> dict[str, Any]:
        """Project raw signals into API shape with LLM-driven temporal output.

        Every vocabulary route is used: the model's inline selections first
        (ids, then echoed names), then a direct resolve of the signal text.
        Temporal is never empty and never uncoded — when nothing resolves the
        vocabulary's declared default window stands in, so downstream always
        receives a real CUI rather than a fabricated REF_POINT span.

        ``candidate`` (canonical mode): the candidate whose signals ``raw``
        holds, so each candidate gets the window resolved for IT; without it
        (intent level) the union across candidates is used.
        """
        tmap = self._get_temporal_map()
        by_candidate = getattr(self, "_temporal_map_by_candidate", None) or {}
        ckey = self._fold_key(candidate) if candidate else ""
        temporal_in = list(raw.get("temporal_signal") or [])

        temporal: list[TemporalMapping] = []
        seen_codes: set = set()
        seen_formulas: set = set()
        for s in temporal_in:
            matches = (
                (by_candidate.get((ckey, s)) if ckey else None)
                or tmap.get(s)
                or self._vocab_mappings(s)
            )
            for t in matches:
                if not t:
                    continue
                if t.codes:
                    if t.codes not in seen_codes:
                        temporal.append(t)
                        seen_codes.add(t.codes)
                elif self._canonical is not None and t.formula:
                    # Canonical mode: a window the vocabulary cannot code still
                    # carries its computed formula (formula-first).
                    fkey = tuple(t.formula)
                    if fkey not in seen_formulas:
                        temporal.append(t)
                        seen_formulas.add(fkey)

        if not temporal:
            if self._canonical is not None:
                temporal = [
                    CanonicalTemporalMapping(
                        **t.model_dump(), basis="default", resolution="default"
                    )
                    for t in self._default_temporal()
                ]
            else:
                # temporal = self._default_temporal()
                temporal = [TemporalMapping(**DEFAULT_TEMPORAL_FALLBACK)]
        return {
            "record_types": list(dict.fromkeys(raw.get("record_types") or [])),
            "temporal": [t.model_dump() for t in temporal],
            "authors": list(raw.get("author_roles") or []),
            "longitudinal_scope": list(raw.get("longitudinal_scope") or []),
            "content_signals": list(raw.get("content_signals") or []),
            "clinical_setting": list(raw.get("clinical_setting") or []),
        }

    def _build_ctx(self, concept_name: str) -> _RetrievalContext:
        cc = self._get_concept_context(concept_name)
        if cc is None:
            return _RetrievalContext()
        records, authors, scope, signals, settings = self._aggregate_facets([cc])
        return _RetrievalContext(
            record_types=records,
            author_roles=authors,
            longitudinal_scope=scope,
            content_signals=signals,
            clinical_setting=settings,
        )

    @staticmethod
    def _merge_signals(
        candidates: list[_FinalCandidateItem],
    ) -> _FlatRetrievalSignals:
        def _union(getter: Callable[[_FlatRetrievalSignals], list[Any]]) -> list[Any]:
            seen_vals: dict = {}
            for fc in candidates:
                for val in getter(fc.retrieval_signals):
                    if isinstance(val, str):
                        val = val.strip()
                        if not val:
                            continue
                        key = val.lower()
                    else:
                        if val is None:
                            continue
                        key = val
                    if key not in seen_vals:
                        seen_vals[key] = val
            return list(seen_vals.values())

        return _FlatRetrievalSignals(
            record_types=_union(lambda s: s.record_types),
            author_roles=_union(lambda s: s.author_roles),
            longitudinal_scope=_union(lambda s: s.longitudinal_scope),
            temporal_signal=_union(lambda s: s.temporal_signal),
            content_signals=_union(lambda s: s.content_signals),
            clinical_setting=_union(lambda s: s.clinical_setting),
        )

    def build_final_candidates(
        self, intents: list[dict[str, Any]]
    ) -> list[list[_FinalCandidateItem]]:
        """Build fc_001… from atomic_concepts; global dedupe across intents."""
        seen: set = set()
        idx = 0
        per_intent: list[list[_FinalCandidateItem]] = []

        for intent in intents:
            intent_title = str(intent.get("intent_title", "")).strip()
            nature = str(intent.get("nature", "")).strip()
            intent_candidates: list[_FinalCandidateItem] = []

            for sn in intent.get("sub_natures") or []:
                sub_nature = str(sn.get("category_path", "")).strip()
                for concept in sn.get("atomic_concepts") or []:
                    candidate = str(concept).strip()
                    key = candidate.lower()
                    if not key or key in seen:
                        continue
                    seen.add(key)
                    idx += 1

                    rich_ctx = self._build_ctx(candidate)
                    signals = _FlatRetrievalSignals.from_retrieval_context(rich_ctx)
                    signals.temporal_signal = self._temporal_for_candidate(
                        intent_title, candidate
                    )
                    intent_candidates.append(
                        _FinalCandidateItem(
                            candidate_id=f"fc_{idx:03d}",
                            intent_title=intent_title,
                            nature=nature,
                            sub_nature=sub_nature,
                            candidate=candidate,
                            retrieval_signals=signals,
                        )
                    )

            per_intent.append(intent_candidates)

        return per_intent


def _parse_context(
    context: ContextualEnvironmentOutput | dict[str, Any] | None,
) -> ContextualEnvironmentOutput | None:
    if context is None:
        return None
    if isinstance(context, ContextualEnvironmentOutput):
        return context
    return ContextualEnvironmentOutput.model_validate(context)


def _parse_temporal(
    temporal: TemporalExtractionOutput | dict[str, Any] | None,
    context: ContextualEnvironmentOutput | None,
) -> TemporalExtractionOutput | None:
    if temporal is not None:
        if isinstance(temporal, TemporalExtractionOutput):
            return temporal
        return TemporalExtractionOutput.model_validate(temporal)
    if context is not None:
        return TemporalExtractionOutput(intents=list(context.temporal_by_intent or []))
    return None


def assemble_v2_intents(
    intents: list[dict[str, Any]],
    context: ContextualEnvironmentOutput | dict[str, Any] | None = None,
    temporal: TemporalExtractionOutput | dict[str, Any] | None = None,
    vocab: TemporalVocab | None = None,
    resolver: TemporalResolver | None = None,
    canonical_resolver: CanonicalTemporalResolver | None = None,
) -> list[dict[str, Any]]:
    """Attach final_candidates and structured retrieval_signals to each intent."""
    parsed_context = _parse_context(context)
    parsed_temporal = _parse_temporal(temporal, parsed_context)
    assembler = SignalsAssembler(
        context=parsed_context,
        temporal=parsed_temporal,
        vocab=vocab,
        resolver=resolver,
        canonical_resolver=canonical_resolver,
    )

    per_intent_candidates = assembler.build_final_candidates(intents)
    enriched: list[dict[str, Any]] = []

    for intent, candidates in zip(intents, per_intent_candidates):
        intent_signals = assembler._merge_signals(candidates)

        # Candidates first — the intent's temporal is derived from theirs.
        final_candidates = []
        for fc in candidates:
            fc_dict = fc.model_dump()
            fc_dict["retrieval_signals"] = assembler.structured_signals(
                fc.retrieval_signals.model_dump(), candidate=fc.candidate
            )
            final_candidates.append(fc_dict)

        out = dict(intent)
        out["final_candidates"] = final_candidates
        out["retrieval_signals"] = assembler.structured_signals(
            intent_signals.model_dump()
        )

        # Intent temporal = deduped union (by cui) of its candidates' already resolved temporal.
        if final_candidates:
            merged: list[dict[str, Any]] = []
            seen: set = set()
            for fc in final_candidates:
                for t in fc["retrieval_signals"].get("temporal") or []:
                    key = t.get("codes")
                    if key not in seen:
                        seen.add(key)
                        merged.append(t)
            out["retrieval_signals"]["temporal"] = merged

        enriched.append(out)

    return enriched
