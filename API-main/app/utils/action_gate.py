"""
Gates /v2/retrieval-signals on the Action Finder's classification.

When a caller supplies `actions` (the activity types it expects), each query is
classified by the Action Finder IN-PROCESS — a direct function call, never an
HTTP hop to a sibling endpoint. A query whose identified actions do not overlap
the expected list (or do not include information_retrieval) is skipped and a
skip block is returned in its place.

The Action Finder is injected as a callable so this module has no import-time
dependency on it and tests need no LLM:

    action_finder_fn(query_text: str) -> {"actions": [str, ...], "is_processable": bool}

  * NO-MATCH RESPONSE CONTRACT  -> build_skip_block()
      HTTP 200 with a per-query skip block whose skip_reason contains the
      Action Finder's identified actions. The request is valid; "not
      serviceable by retrieval" is a RESULT, not a client error.

  * ACTION FINDER FAILURE      -> GATE_FAILS_OPEN
      True: an Action Finder error lets the query through (v1 behaviour) and
      is flagged on the decision so it is visible in logs. False: the query
      is skipped with skip_reason "action_finder_error".

  * MATCH SEMANTICS            -> normalize_action() / decide()
      Case-, space- and hyphen-insensitive equality on action names. Retrieval
      runs only when an expected action overlaps the identified list and that
      list contains information_retrieval.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

# The Action Finder taxonomy. Caller-supplied actions are validated against it
# so a typo ("informational_retrieval") is a 422, not a silent never-match.
ACTION_TYPES = frozenset(
    {
        "conversational_response",
        "information_retrieval",
        "knowledge_fact_search",
        "admin_response",
        "out_of_scope",
    }
)

# The only action that makes a query serviceable by retrieval.
PROCESSABLE_ACTION = "information_retrieval"

# Team decision: see module docstring.
GATE_FAILS_OPEN = True

SKIP_REASON_MISMATCH = "action_mismatch"
SKIP_REASON_FINDER_ERROR = "action_finder_error"

ActionFinderFn = Callable[[str], dict[str, Any]]
AsyncActionFinderFn = Callable[[str], Awaitable[dict[str, Any]]]


def normalize_action(value: Any) -> str:
    """'Information Retrieval' / 'information-retrieval' -> 'information_retrieval'."""
    if not isinstance(value, str):
        return ""
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def normalize_actions(values: Iterable[Any] | None) -> list[str]:
    """Normalized, de-duplicated, order-preserving, blanks dropped."""
    out: list[str] = []
    for v in values or []:
        n = normalize_action(v)
        if n and n not in out:
            out.append(n)
    return out


def invalid_actions(values: Iterable[Any] | None) -> list[str]:
    """Caller-supplied action names that are not in the taxonomy (as sent)."""
    bad: list[str] = []
    for v in values or []:
        if normalize_action(v) not in ACTION_TYPES:
            bad.append(str(v))
    return bad


@dataclass
class GateDecision:
    matched: bool
    expected_actions: list[str]
    identified_actions: list[str] = field(default_factory=list)
    is_processable: bool | None = None
    error: str | None = None  # set when the Action Finder failed
    failed_open: bool = False  # True when matched=True only because of GATE_FAILS_OPEN
    usage_metadata: dict[str, int] = field(default_factory=dict)

    @property
    def skip_reason(self) -> str | None:
        if self.matched:
            return None
        return SKIP_REASON_FINDER_ERROR if self.error else SKIP_REASON_MISMATCH


def decide(expected: Iterable[Any], identified: Iterable[Any]) -> bool:
    """True when any expected action appears among the identified ones."""
    exp = set(normalize_actions(expected))
    ident = set(normalize_actions(identified))
    return bool(exp & ident)


def _decision_from_result(
    expected: list[str], result: Any, error: str | None = None
) -> GateDecision:
    if error is not None:
        return GateDecision(
            matched=GATE_FAILS_OPEN,
            expected_actions=expected,
            error=error,
            failed_open=GATE_FAILS_OPEN,
        )

    result = result or {}
    is_dict = isinstance(result, dict)
    identified = normalize_actions(result.get("actions") if is_dict else None)
    is_processable = result.get("is_processable") if is_dict else None
    usage = result.get("usage_metadata") if is_dict else None
    return GateDecision(
        matched=(decide(expected, identified) and PROCESSABLE_ACTION in identified),
        expected_actions=expected,
        identified_actions=identified,
        is_processable=is_processable if isinstance(is_processable, bool) else None,
        usage_metadata=dict(usage) if isinstance(usage, dict) else {},
    )


def evaluate_gate(
    query_text: str,
    expected_actions: Iterable[Any],
    action_finder_fn: ActionFinderFn,
) -> GateDecision:
    """Classify one query in-process and decide whether to compute signals.

    Never raises on Action Finder failure; the failure is recorded on the
    decision and resolved per GATE_FAILS_OPEN.
    """
    expected = normalize_actions(expected_actions)
    try:
        result = action_finder_fn(query_text)
    except Exception as e:  # noqa: BLE001 — any finder failure is handled the same way
        return _decision_from_result(expected, None, error=f"{type(e).__name__}: {e}")
    return _decision_from_result(expected, result)


async def evaluate_gate_async(
    query_text: str,
    expected_actions: Iterable[Any],
    action_finder_fn: AsyncActionFinderFn,
) -> GateDecision:
    """Async twin of evaluate_gate for the async retrieval-signals handler."""
    expected = normalize_actions(expected_actions)
    try:
        result = await action_finder_fn(query_text)
    except Exception as e:  # noqa: BLE001 — any finder failure is handled the same way
        return _decision_from_result(expected, None, error=f"{type(e).__name__}: {e}")
    return _decision_from_result(expected, result)


def build_skip_block(
    query_id: str, text: str, decision: GateDecision
) -> dict[str, Any]:
    """The per-query response entry for a query the gate declined.

    NO-MATCH RESPONSE CONTRACT lives here and only here. Returns the identified
    actions so the caller knows what to do with the query INSTEAD of retrieval.
    The signal fields keep the same shape as a computed block (record_types /
    tags empty, temporal null) so clients can parse every entry the same way.
    """
    identified = ", ".join(decision.identified_actions) or "no recognized action"
    if decision.error:
        detail = f"Action Finder failed ({decision.error}); query skipped"
    else:
        detail = f"Action Finder classified this query as {identified}"
    return {
        "id": query_id,
        "text": text,
        "skipped": True,
        # Machine-readable: the Action Finder's identified actions (contract).
        "skip_reason": list(decision.identified_actions),
        "skip_reason_detail": detail,
        "record_types": [],
        "temporal": None,
        "tags": [],
    }
