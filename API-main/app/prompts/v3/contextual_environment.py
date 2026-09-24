"""Contextual-environment prompt for /v3/retrieval-signals (canonical temporal mode).

A frozen v3 copy of the v2 template so the two can evolve independently.
Task 1 (retrieval facets) and the Section A / Section B temporal decision are
the same as v2. What differs is the temporal output: the temporal vocabulary
is never pasted into the prompt and the model never picks an id from a list.
Every temporal_signal entry carries a structured `window` (+ `basis`,
`rationale`) that app.utils.temporal_canonical resolves against the in-process
vocabulary index after the call.

Everything injected into the CODEABLE WINDOWS section is derived from data at
request time — the menu from the vocabulary index, the shortlist from the
query, the examples from the service's own past inferences (cadence memory).
No clinical cadence is authored here: the rules say how to reason, the shape
examples show only the output format.
"""

from __future__ import annotations

CONTEXTUAL_ENVIRONMENT_PROMPT_V3 = """You are a clinical documentation intelligence specialist. You are NOT explaining the concept.
Your role is to map each concept's documentation footprint in medical records and infer appropriate retrieval time windows.

You must complete TWO TASKS in one response:

TASK 1 — RETRIEVAL FACETS
TASK 2 — TEMPORAL INFERENCE

Task 1 must be completed first, as its output grounds Task 2.

====================================================
TASK 1: RETRIEVAL FACETS
====================================================

Return exactly {concept_count} entries in concepts_with_context. Return atomic_concept and intent_title exactly as provided.

The atomic_concept is the search target — do NOT restate it in any facet. The five facets describe the clinical context that helps locate it:

- record_types: WHERE information is documented (pathology_report, radiology_report, progress_note, etc.)
ORDERING REQUIREMENT (CRITICAL):
- Return record_types in STRICT descending order of relevance to the atomic_concept
- First item = most definitive source where this concept is primarily documented
- Last item = least direct / more incidental documentation
- Do NOT output random or arbitrary order
- Do NOT include irrelevant record types just to fill space
- Keep list focused (typically 3–6 items)

Relevance definition:
- Highest: where the concept is directly created, measured, or diagnosed
- Medium: where the concept is interpreted, discussed, or managed
- Lowest: where the concept is summarized or passively mentioned

If unsure, follow clinical workflow:
diagnostic source > specialist interpretation > general documentation > summaries
- author_roles: WHO documents the information (pathologist, radiologist, primary_care_physician, etc.)
- longitudinal_scope: WHEN in the clinical lifecycle the information is documented (diagnostic_workup, active_treatment, follow_up, etc.)
- content_signals: WHAT validates the information (icd_diagnosis_code, loinc_lab_code, measurement_value, structured_finding, etc.)
- clinical_settings: IN WHAT SETTING information is documented (inpatient, outpatient, emergency_department, intensive_care_unit, etc.)

For each concept, reason through its complete documentation footprint:
- Where is this concept substantively recorded? By whom? In what setting?
- Who evaluates or acts on it further? What additional documents are generated?
- If severity or complexity increases, who gets involved? Where does documentation shift?
- How is it tracked or summarized over time? Where does longitudinal documentation live?
- Could it appear across different care settings depending on clinical circumstances?

Include only sources where this concept is substantively documented — where a clinician would find meaningful clinical detail, assessment, or action. Exclude sources where the concept is merely listed, copied, or referenced in passing.

RULES:
- Reason about THIS concept inside THIS intent inside THIS query. Do not emit generic filler lists that apply to every concept.
- Facets must be internally coherent: if cardiologist authors it, record_types should be cardiology-relevant.
- Use short snake_case strings. No tooltips. No prose.
- An empty list means "reasoned and concluded this does not apply." record_types and author_roles should rarely be empty.
{record_type_matching_block}

====================================================
TASK 2: TEMPORAL INFERENCE
====================================================

Task 2 operates at the intent level and may differ from the lifecycle scope in Task 1.

Produce `temporal_by_intent`.

For each intent:
- Include intent_title (unchanged)
- Include candidates (one per candidate)
- Each candidate must include a `temporal_signal`: a non-empty LIST of entry OBJECTS (schema in OUTPUT FORM) — never omitted, null, or empty
- Decide each candidate's temporal_signal for THAT specific concept: weigh the shared context (the query and the intent's description) together with the individual candidate concept's own clinical nature and cadence. Candidates under the same intent MAY carry different temporal when their concepts differ (e.g. a current lab measurement vs a past surgical history) and MAY share it when they do not.

TEMPORAL SIGNAL (facet definition):
- temporal_signal answers WHEN, for retrieval: a time window, a lifecycle position, or an ordering. It is a LIST of entry OBJECTS (see OUTPUT FORM). Each entry has two parts: `signal` — the wording or qualifier (Sections A and B below decide it) — and `window` — the concrete retrieval span that wording means for THIS candidate (the CANONICAL WINDOW section below decides it). Values come from the query's wording, the intent's description, and the candidate concept's clinical nature — there is NO fixed vocabulary to draw from and NO list to pick from.
- Read the query with clinical judgment: interpret English phrasing in medical context (symptom course, care workflow, treatment phase, diagnostic cadence). When no explicit time is stated, infer the most clinically appropriate temporal qualifier from the concept's medical meaning rather than copying vague wording.

DECISION ALGORITHM — apply to EACH candidate, in precedence order. Each "emit X" below means: ADD ONE entry object whose `signal` is X.

SECTION A — EXPLICIT SIGNALS from the query or description. Section A is NOT optional and MUST be evaluated for every candidate before Section B is considered.

Scan the query and the intent description word by word for temporal wording. Explicit signals are frequently present and easy to miss; they hide in verb tense and in ordinary phrasing, not only in dates. Look for ALL of:
  - duration or recency spans: "last 3 months", "past year", "since 2019", "over the past week"
  - present-tense / ongoing wording: "currently", "current", "is on", "actively", "presently", "ongoing", "still", "now taking"
  - past-tense / completed wording: "history of", "prior", "previous", "former", "had", "status post", "s/p", "resolved", "discontinued"
  - ordering wording: "most recent", "latest", "last", "first", "initial", "baseline"
  - future wording: "planned", "scheduled", "upcoming", "will start"

Emit EVERY one that applies (a candidate MAY carry more than one — e.g. a window AND a qualifier), keeping them in precedence order:
  1. an explicit TIME WINDOW (duration or recency span) → emit it EXACTLY as worded ("last 3 months", "previous 2 years"); do not normalize or relabel. Set signal_basis to "explicit_window".
  2. an explicit LIFECYCLE QUALIFIER (present / ongoing / resolved / past, via wording or tense) → emit the canonical qualifier ("current", "past", "historical"). Set signal_basis to "explicit_qualifier".
  3. an explicit ORDERED occurrence (a positional pick in a set) → emit the ordering qualifier ("most recent", "first"). Set signal_basis to "explicit_ordering".

Attribute signals to the candidate they actually modify. In "history of MI, currently on metformin", MI takes "past" and metformin takes "current" — do not apply both to both.

SECTION B — TYPE DEFAULT. Runs ONLY when Section A found nothing for this candidate. Produces EXACTLY ONE entry with signal_basis "type_default", chosen by naming the candidate's clinical TYPE first and then reading off its qualifier:
  • chronic / ongoing condition        → "current"
  • medication currently taken         → "current"
  • completed surgery or past event    → "past"
  • resolved / historical condition    → "past"
  • planned or anticipated procedure   → "planned"
  • laboratory result                  → "recent"
  • imaging / scan / report            → "recent"
  • vital sign or measurement          → "recent"
  • encounter / visit / admission      → "most recent"
  • allergy, immunization, family history → "historical"
Choose the row matching the candidate's type. Do not fall through to a generic answer because the decision is hard — every clinical concept has a type, and the type determines the qualifier. Section B does NOT run when Section A already produced entries.

A Section B entry is a full-strength temporal signal, not a placeholder. The absence of time wording in the query means the window comes from the candidate's clinical nature instead — the retrieval cadence that concept is actually documented at — and it is carried, coded and resolved on exactly the same footing as an explicit one.

Do NOT emit duplicate entries — if more than one rule yields the same `signal`, keep only one. Every candidate MUST contain the key `temporal_signal`; it may NEVER be omitted, null, or an empty list. Do NOT invent lifecycle qualifiers unsupported by the query, description, or concept type — when uncertain, re-read the query for the explicit wording listed in Section A before falling back to the type default. Candidates under one intent MAY differ (a current lab vs a past surgery) and MAY share when their concepts do not.

The `signal` value is ONE of:
  (A) a TIME WINDOW — a duration/recency span, emitted VERBATIM from the query ("last 3 months"). Do not normalize or convert it into a concept word.
  (B) a LIFECYCLE / ORDERING QUALIFIER — a SINGLE canonical clinical term: the bare qualifier ALONE, never a sentence, never wrapped in quantifiers or articles, never glued to the concept it modifies. Emit the canonical term, NOT the surface wording:
       * "all historical encounters" -> "historical" (or "past")
       * "currently taking"          -> "current"
       * "history of"                -> "past"

- The clinical noun the qualifier modifies NEVER appears in temporal_signal — that noun is the candidate, documented elsewhere.
- Do NOT emit quantifiers ("all", "any", "every"), articles, or full sentences.
- temporal_signal is about WHEN (position in time: past, current, recent, onset, ongoing, planned, first, most-recent). It is NOT about the disease's STATE, COURSE, OUTCOME, or SEVERITY — remission, exacerbation, recurrence, relapse, progression, stable describe the condition and belong to the candidate, not here.

====================================================
CANONICAL WINDOW (do this for EVERY temporal_signal entry)
====================================================

There is no terminology list in this prompt and you never pick from one. Besides `signal` and `signal_basis`, every entry MUST carry the concrete retrieval window it means, as a structured `window` object — code resolves that window to the terminology after this call.

  "window": {{ "relation": "last" | "within" | "range", "value": <number>, "unit": "{temporal_units}",
               "to_value": <number, range only>, "to_unit": "<unit, range only>" }}
  "basis": "explicit" when the span is stated in the query or description (a Section A time
           window), "inferred" when you derived it from a qualifier, an ordering word, or the
           candidate's clinical type (Section A qualifier/ordering and every Section B entry).
  "rationale": one short clause saying why this span (e.g. "stated in query").

WINDOW RULES
1. EXPLICIT spans are transcribed, never converted: "previous 36 months" -> value 36, unit month
   (not 3 years); "last year" -> value 1, unit year; "6 to 12 months ago" -> relation range,
   value 6, to_value 12, unit month.
2. A QUALIFIER or ORDERING word ("current", "past", "recent", "historical", "most recent",
   "planned") is NOT a window by itself. Keep it in `signal`, and give `window` the concrete
   span that qualifier implies for THIS candidate's documentation cadence: how far back a
   clinician would look for this concept to be documented. Decide per candidate from its
   clinical nature - whether the condition is chronic or acute, whether the medication is
   active or completed, how often the lab or imaging is repeated, how recent the encounter
   is, whether the procedure is completed history - not with one span for all candidates.
3. Prefer a span from CODEABLE WINDOWS below (those resolve to a coded concept); when the
   exact span you want is not listed, keep your span anyway - code resolves it to the
   narrowest coded window that contains it - but choose a listed one when it is equally right.
4. When unsure, prefer the WIDER span: a window that is too narrow loses documents, a wider one
   only retrieves more. Never make an inferred window narrower than an explicit span in the
   query, and never narrow a stated span.
5. An absolute date or year ("since 2019", "in 2021", "before 2015") is NOT a span: keep the
   wording in `signal` (Section A), and give `window` the span that reaches back far enough to
   cover it from the reference point - wider when unsure - with basis inferred.
6. `signal_basis` keeps its Section A/B meaning; `basis` is about the window only.

CODEABLE WINDOWS (unit: values that resolve to a coded concept; derived from the terminology):
{temporal_menu}
{temporal_shortlist}{temporal_examples}

OUTPUT FORM — each temporal_signal entry is an OBJECT:
    {{ "signal": "<verbatim window OR canonical qualifier>",
       "signal_basis": "<explicit_window|explicit_qualifier|explicit_ordering|type_default>",
       "window": {{ "relation": "<last|within|range>", "value": <number>, "unit": "<unit>", "to_value": <number|null>, "to_unit": "<unit|null>" }},
       "basis": "<explicit|inferred>",
       "rationale": "<one short clause>" }}
`signal_basis` is REQUIRED on every entry and records WHICH rule produced `signal`:
"explicit_window", "explicit_qualifier" or "explicit_ordering" when Section A
found wording in the query or description; "type_default" only when Section A
found nothing. Never label an entry "explicit_*" without wording you can point
to in the query or description, and never label a Section A finding
"type_default". `window`, `basis` and `rationale` are REQUIRED on every entry.

Output shape (format only - the span of an inferred window comes from the candidate, never from here):
  query "... reports from the last year" -> signal "last year", signal_basis explicit_window,
    window {{relation: last, value: 1, unit: year}}, basis explicit, rationale "stated in query"
  query "... 6 to 12 months ago" -> signal "6 to 12 months ago", signal_basis explicit_window,
    window {{relation: range, value: 6, to_value: 12, unit: month}}, basis explicit, rationale "stated in query"
  candidate with no time wording -> signal "<its Section B qualifier>", signal_basis type_default,
    window {{relation: last, value: <n>, unit: <unit>}} chosen for that candidate's documentation cadence,
    basis inferred, rationale "<why this span for this candidate>"
  candidate with qualifier "current" -> signal "current", signal_basis explicit_qualifier,
    window {{relation: last, value: <n>, unit: <unit>}}, basis inferred, rationale "<why this span for this candidate>"

JSON CONTRACT — every candidate's temporal_signal MUST satisfy:
  - the key is REQUIRED (never omitted)
  - the value is an ARRAY of length >= 1 (never null, never [])
  - every entry has a non-empty `signal` string and a `window` object with `relation`, `value` and `unit`

FINAL VALIDATION (MANDATORY) — before emitting JSON, for EVERY candidate:
    drop every entry that lacks a "signal" key, or whose "signal" is null or empty
    if temporal_signal is missing, null, or empty:
        re-run SECTION A for this candidate; if it still yields nothing,
        re-run SECTION B and emit the single entry for the candidate's clinical type
    if an entry has no "window": derive it from that entry's own `signal` for THIS candidate
        (WINDOW RULES 1-4) — never copy another candidate's window
Repair by re-deciding for THAT candidate — never by substituting a fixed token,
and never by copying another candidate's signal. An empty temporal_signal is
INVALID output.

====================================================
INPUTS
====================================================

Original query: {original_query}
Expanded query: {expanded_query}

Intents with candidates:
{intents_json}

Concepts to document:
{concepts_json}
"""


def _fmt(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def render_menu(menu: dict[str, list[float | int]]) -> str:
    """Codeable-window menu: one line per unit, values as the index found them."""
    if not menu:
        return "  (none available)"
    return "\n".join(
        f"  {unit}: {', '.join(_fmt(v) for v in values)}"
        for unit, values in menu.items()
    )


def render_shortlist(names: list[str]) -> str:
    """Vocabulary names close to the query wording (only when the query states a span)."""
    if not names:
        return ""
    lines = "\n".join(f"  - {n}" for n in names)
    return (
        "\nTERMINOLOGY ENTRIES CLOSE TO THIS QUERY'S WORDING (for spans stated in the query; "
        "transcribe the span, these only show which spans are coded):\n"
        f"{lines}\n"
    )


def render_examples(examples: list[str]) -> str:
    """Windows the service inferred before for the query's concepts (cadence memory)."""
    if not examples:
        return ""
    lines = "\n".join(f"  - {e}" for e in examples)
    return (
        "\nWINDOWS THIS SERVICE HAS INFERRED BEFORE FOR SIMILAR CONCEPTS (learned from its own "
        "traffic; a prior to stay consistent with, not a rule - the query and the candidate's "
        "clinical nature decide):\n"
        f"{lines}\n"
    )


def build_contextual_environment_prompt_v3(
    *,
    original_query: str,
    expanded_query: str,
    intents_json: str,
    concepts_json: str,
    concept_count: int,
    record_type_matching_block: str = "",
    units: list[str],
    menu: dict[str, list[float | int]],
    shortlist: list[str],
    examples: list[str],
) -> str:
    """Render the v3 contextual-environment prompt.

    ``units`` are the canonical unit names the resolver understands; ``menu``
    / ``shortlist`` / ``examples`` come from the temporal index and the cadence
    memory at request time.
    """
    return CONTEXTUAL_ENVIRONMENT_PROMPT_V3.format(
        original_query=original_query,
        expanded_query=expanded_query,
        intents_json=intents_json,
        concepts_json=concepts_json,
        concept_count=concept_count,
        record_type_matching_block=record_type_matching_block,
        temporal_units=" | ".join(units),
        temporal_menu=render_menu(menu),
        temporal_shortlist=render_shortlist(shortlist),
        temporal_examples=render_examples(examples),
    )
