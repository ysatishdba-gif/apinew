"""Canonical temporal block for the contextual-environment prompt (/v3).

Replaces the TEMPORAL CONCEPT MATCHING block that pasted the whole temporal
vocabulary into every call. The model never sees the vocabulary: it writes each
window in a small canonical form and code resolves it (see
app.utils.temporal_canonical). Everything injected below is derived from data
at request time — the codeable-window menu from the vocabulary index, the
shortlist from the query, the examples from the service's own past inferences
(cadence memory). No clinical cadence is authored here: the rules say how to
reason, the shape examples show only the output format.
"""

from __future__ import annotations

TEMPORAL_CANONICAL_BLOCK = """
CANONICAL WINDOW (do this for EVERY temporal_signal entry, in addition to `signal` and `signal_basis`):
Besides the qualifier or wording in `signal`, every entry MUST carry the concrete retrieval
window it means, as a structured `window` object — code resolves it to the terminology; you
never pick from a list. Leave selected_id / selected_name / selected_ids / selected_names null.

  "window": {{ "relation": "last" | "within" | "range", "value": <number>, "unit": "{units}",
               "to_value": <number, range only>, "to_unit": "<unit, range only>" }}
  "basis": "explicit" when the span is stated in the query or description (Section A time
           window), "inferred" when you derived it from a qualifier, an ordering word, or the
           candidate's clinical type (Section A qualifier/ordering and every Section B entry).
  "rationale": one short clause saying why this span (e.g. "stated in query",
               "monitoring cadence of a chronic condition", "active medication documentation window").

RULES
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
   exact span you want is not listed, keep your span anyway - it is still used - but choose a
   listed one when it is equally right.
4. When unsure, prefer the WIDER span: a window that is too narrow loses documents, a wider one
   only retrieves more. Never make an inferred window narrower than an explicit span in the
   query, and never narrow a stated span.
5. `signal_basis` keeps its Section A/B meaning; `basis` is about the window only.

CODEABLE WINDOWS (unit: values that resolve to a coded concept; derived from the terminology):
{menu}
{shortlist}{examples}
Output shape (format only - the span of an inferred window comes from the candidate, never from here):
  query "... reports from the last year" -> signal "last year", signal_basis explicit_window,
    window {{relation: last, value: 1, unit: year}}, basis explicit, rationale "stated in query"
  query "... 6 to 12 months ago" -> signal "6 to 12 months ago", signal_basis explicit_window,
    window {{relation: range, value: 6, to_value: 12, unit: month}}, basis explicit, rationale "stated in query"
  candidate with no time wording, type default "{default_name}" -> signal "{default_name}", signal_basis type_default,
    window {{relation: last, value: <n>, unit: <unit>}} chosen for that candidate's documentation cadence,
    basis inferred, rationale "<why this span for this candidate>"
  candidate with qualifier "current" -> signal "current", signal_basis explicit_qualifier,
    window {{relation: last, value: <n>, unit: <unit>}}, basis inferred, rationale "<why this span for this candidate>"
"""


def _fmt(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def render_menu(menu: dict[str, list[float | int]]) -> str:
    if not menu:
        return "  (none available)"
    return "\n".join(
        f"  {unit}: {', '.join(_fmt(v) for v in values)}"
        for unit, values in menu.items()
    )


def render_shortlist(names: list[str]) -> str:
    if not names:
        return ""
    lines = "\n".join(f"  - {n}" for n in names)
    return (
        "\nTERMINOLOGY ENTRIES CLOSE TO THIS QUERY'S WORDING (for spans stated in the query; "
        "transcribe the span, these only show which spans are coded):\n"
        f"{lines}\n"
    )


def render_examples(examples: list[str]) -> str:
    if not examples:
        return ""
    lines = "\n".join(f"  - {e}" for e in examples)
    return (
        "\nWINDOWS THIS SERVICE HAS INFERRED BEFORE FOR SIMILAR CONCEPTS (learned from its own "
        "traffic; a prior to stay consistent with, not a rule - the query and the candidate's "
        "clinical nature decide):\n"
        f"{lines}\n"
    )


def build_temporal_canonical_block(
    units: list[str],
    menu: dict[str, list[float | int]],
    shortlist: list[str],
    examples: list[str],
    default_name: str | None = None,
) -> str:
    """``default_name`` is the vocabulary's own default temporal name
    (``TemporalVocab.default_name()``), so even the format example quotes the
    terminology rather than a literal."""
    return TEMPORAL_CANONICAL_BLOCK.format(
        units=" | ".join(units),
        menu=render_menu(menu),
        shortlist=render_shortlist(shortlist),
        examples=render_examples(examples),
        default_name=default_name or "<type default>",
    )
